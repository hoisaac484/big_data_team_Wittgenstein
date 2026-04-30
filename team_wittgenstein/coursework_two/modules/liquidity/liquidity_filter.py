"""Two-stage liquidity filter for the 130/30 multi-factor strategy.

Stage A removes stocks whose ADTV is too low for meaningful position sizing.
Stage B removes the most illiquid survivors using the Amihud ILLIQ measure.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from modules.db.db_connection import PostgresConnection

logger = logging.getLogger(__name__)
SCHEMA = "team_wittgenstein"


@dataclass(frozen=True)
class LiquidityConfig:
    """Parameters for the two-stage liquidity filter."""

    adtv_lookback_days: int = 20
    illiq_lookback_days: int = 21
    illiq_removal_pct: float = 0.10
    adtv_min_dollar: float = 1_000_000


def _fetch_price_window(
    db: PostgresConnection,
    rebalance_date: date,
    lookback_days: int,
) -> pd.DataFrame:
    """Fetch price and volume data for all symbols in the lookback window.

    Uses strict less-than on rebalance_date to prevent look-ahead bias.
    Fetches extra calendar days to account for weekends and holidays.
    """
    buffer_days = lookback_days * 2
    start_date = rebalance_date - timedelta(days=buffer_days)

    query = """
        SELECT symbol, trade_date, adjusted_close, volume
        FROM team_wittgenstein.price_data
        WHERE trade_date >= :start_date
          AND trade_date < :rebalance_date
          AND adjusted_close IS NOT NULL
          AND volume IS NOT NULL
          AND volume > 0
        ORDER BY symbol, trade_date
    """
    return db.read_query(
        query, {"start_date": start_date, "rebalance_date": rebalance_date}
    )


def compute_adtv(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compute average daily traded value over the trailing window.

    Returns one row per symbol with column 'adtv'. Symbols with fewer
    than `lookback` trading days are excluded.
    """
    df = prices.sort_values(["symbol", "trade_date"]).copy()
    df["dollar_volume"] = df["adjusted_close"] * df["volume"]

    # Rolling mean per symbol; min_periods drops symbols with short history
    df["adtv"] = df.groupby("symbol")["dollar_volume"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).mean()
    )

    # Keep last observation per symbol
    result = df.groupby("symbol").tail(1)[["symbol", "adtv"]].copy()
    return result.dropna(subset=["adtv"]).reset_index(drop=True)


def apply_adtv_floor(
    adtv_df: pd.DataFrame,
    adtv_min_dollar: float,
) -> pd.DataFrame:
    """Remove stocks with ADTV below the minimum dollar threshold."""
    if adtv_df.empty:
        return adtv_df

    survivors = adtv_df[adtv_df["adtv"] >= adtv_min_dollar].copy()
    return survivors.reset_index(drop=True)


def compute_amihud_illiq(
    prices: pd.DataFrame,
    adtv_survivors: list[str],
    lookback: int,
) -> pd.DataFrame:
    """Compute Amihud ILLIQ for ADTV survivors over the trailing window.

    ILLIQ is the mean of ``abs(log_return) / dollar_volume`` over the trailing
    ``lookback`` days. Returns one row per symbol with column
    ``amihud_illiq``.
    """
    df = prices[prices["symbol"].isin(adtv_survivors)].copy()
    df = df.sort_values(["symbol", "trade_date"])

    df["log_return"] = df.groupby("symbol")["adjusted_close"].transform(
        lambda p: np.log(p / p.shift(1))
    )
    df["dollar_volume"] = df["adjusted_close"] * df["volume"]
    df["illiq_ratio"] = df["log_return"].abs() / df["dollar_volume"]
    df["illiq_ratio"] = df["illiq_ratio"].replace([np.inf, -np.inf], np.nan)

    df["amihud_illiq"] = df.groupby("symbol")["illiq_ratio"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).mean()
    )

    result = df.groupby("symbol").tail(1)[["symbol", "amihud_illiq"]].copy()
    return result.dropna(subset=["amihud_illiq"]).reset_index(drop=True)


def apply_illiq_filter(
    illiq_df: pd.DataFrame,
    removal_pct: float,
) -> pd.DataFrame:
    """Remove the most illiquid stocks by cross-sectional ILLIQ rank.

    Stocks in the top `removal_pct` percentile of ILLIQ are removed.
    Ranking is across all stocks, not within sectors.
    """
    if illiq_df.empty:
        return illiq_df

    illiq_df = illiq_df.copy()
    illiq_df["illiq_rank_pct"] = illiq_df["amihud_illiq"].rank(pct=True)
    survivors = illiq_df[illiq_df["illiq_rank_pct"] <= (1.0 - removal_pct)].copy()
    return survivors.reset_index(drop=True)


def _persist_metrics(
    db: PostgresConnection,
    metrics_df: pd.DataFrame,
    rebalance_date: date,
) -> None:
    """Write liquidity metrics for all stocks to the liquidity_metrics table."""
    output = pd.DataFrame(
        {
            "symbol": metrics_df["symbol"],
            "calc_date": rebalance_date,
            "adv_20d": metrics_df["adtv"],
            "amihud_illiq": metrics_df["amihud_illiq"],
            "illiq_rank_pct": metrics_df["illiq_rank_pct"],
            "passes_adv": metrics_df["passes_adv"],
            "passes_illiq": metrics_df["passes_illiq"],
            "passes_filter": metrics_df["passes_filter"],
        }
    )
    db.write_dataframe_on_conflict_do_nothing(
        output, "liquidity_metrics", SCHEMA, conflict_columns=["symbol", "calc_date"]
    )
    logger.info(
        "Persisted liquidity metrics for %d stocks (%d pass, date=%s)",
        len(output),
        output["passes_filter"].sum(),
        rebalance_date,
    )


def run_liquidity_filter(
    db: PostgresConnection,
    rebalance_date: date,
    config: LiquidityConfig,
) -> list[str]:
    """Execute two-stage liquidity filter and persist metrics for all stocks.

    Stores metrics and pass/fail flags for every stock in the universe.
    Returns the list of surviving symbols for downstream factor scoring.
    """
    max_lookback = max(config.adtv_lookback_days, config.illiq_lookback_days)
    prices = _fetch_price_window(db, rebalance_date, lookback_days=max_lookback)

    if prices.empty:
        logger.warning("No price data before %s", rebalance_date)
        return []

    # Stage A: compute ADTV for all stocks and flag
    adtv_df = compute_adtv(prices, config.adtv_lookback_days)
    adtv_df["passes_adv"] = adtv_df["adtv"] >= config.adtv_min_dollar

    adtv_survivors = adtv_df[adtv_df["passes_adv"]]["symbol"].tolist()
    logger.info(
        "ADTV floor: %d total, %d pass (date=%s)",
        len(adtv_df),
        len(adtv_survivors),
        rebalance_date,
    )

    # Stage B: compute ILLIQ for ADTV survivors, rank, and flag
    if adtv_survivors:
        illiq_df = compute_amihud_illiq(
            prices, adtv_survivors, config.illiq_lookback_days
        )
        illiq_df["illiq_rank_pct"] = illiq_df["amihud_illiq"].rank(pct=True)
        illiq_df["passes_illiq"] = illiq_df["illiq_rank_pct"] <= (
            1.0 - config.illiq_removal_pct
        )
    else:
        illiq_df = pd.DataFrame(
            columns=["symbol", "amihud_illiq", "illiq_rank_pct", "passes_illiq"]
        )

    # Merge ADTV and ILLIQ into a single DataFrame for all stocks
    all_metrics = adtv_df.merge(
        illiq_df[["symbol", "amihud_illiq", "illiq_rank_pct", "passes_illiq"]],
        on="symbol",
        how="left",
    )
    # Stocks that failed ADTV don't have ILLIQ — mark as failed
    all_metrics["passes_illiq"] = all_metrics["passes_illiq"].fillna(False).astype(bool)
    all_metrics["passes_filter"] = (
        all_metrics["passes_adv"] & all_metrics["passes_illiq"]
    )

    survivors = all_metrics[all_metrics["passes_filter"]]["symbol"].tolist()
    logger.info(
        "ILLIQ filter: %d pass both stages (date=%s)",
        len(survivors),
        rebalance_date,
    )

    # Persist metrics for ALL stocks
    _persist_metrics(db, all_metrics, rebalance_date)

    return survivors
