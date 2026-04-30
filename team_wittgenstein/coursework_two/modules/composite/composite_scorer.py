"""IC-weighted composite score for the 130/30 multi-factor strategy.

Combines four factor z-scores (value, quality, momentum, low_vol) into a
single ranking signal using dynamic weights derived from trailing 36-month
Spearman Information Coefficients.
"""

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr

from modules.db.db_connection import PostgresConnection

logger = logging.getLogger(__name__)
SCHEMA = "team_wittgenstein"
FACTOR_NAMES = ("value", "quality", "momentum", "low_vol")
FACTOR_COLS = ("z_value", "z_quality", "z_momentum", "z_low_vol")


@dataclass(frozen=True)
class CompositeConfig:
    """Parameters for IC-weighted composite score construction."""

    ic_lookback_months: int = 36
    min_ic_months: int = 12


def _fetch_monthly_prices(
    db: PostgresConnection,
    rebalance_date: date,
    lookback_months: int,
) -> pd.DataFrame:
    """Fetch month-end adjusted close prices for all symbols.

    Returns one row per (symbol, month_end) with the last trading day's
    adjusted close for each calendar month.
    """
    start_date = rebalance_date - relativedelta(months=lookback_months + 1)

    query = """
        SELECT symbol, trade_date, adjusted_close
        FROM team_wittgenstein.price_data
        WHERE trade_date < :rebalance_date
          AND trade_date >= :start_date
          AND adjusted_close IS NOT NULL
        ORDER BY symbol, trade_date
    """
    return db.read_query(
        query, {"rebalance_date": rebalance_date, "start_date": start_date}
    )


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly returns from daily price data.

    For each symbol, takes the last trading day of each calendar month
    and computes: return = (price_end / price_start) - 1.
    Returns columns: symbol, month_end, monthly_return.
    """
    df = prices.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["year_month"] = df["trade_date"].dt.to_period("M")

    # Last trading day per symbol per month
    month_end = (
        df.sort_values("trade_date")
        .groupby(["symbol", "year_month"])
        .last()
        .reset_index()
    )
    month_end = month_end.sort_values(["symbol", "year_month"])

    # Monthly return = (P_end / P_prev_end) - 1
    month_end["monthly_return"] = month_end.groupby("symbol")[
        "adjusted_close"
    ].pct_change()
    month_end["month_end"] = month_end["year_month"].dt.to_timestamp("M")

    return month_end[["symbol", "month_end", "monthly_return"]].dropna(
        subset=["monthly_return"]
    )


def _fetch_factor_scores(
    db: PostgresConnection,
    rebalance_date: date,
    lookback_months: int,
) -> pd.DataFrame:
    """Fetch historical factor z-scores from the factor_scores table."""
    start_date = rebalance_date - relativedelta(months=lookback_months + 1)
    query = """
        SELECT symbol, score_date, z_value, z_quality, z_momentum, z_low_vol
        FROM team_wittgenstein.factor_scores
        WHERE score_date < :rebalance_date
          AND score_date >= :start_date
        ORDER BY score_date, symbol
    """
    return db.read_query(
        query, {"rebalance_date": rebalance_date, "start_date": start_date}
    )


def compute_monthly_ic(
    factor_scores: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Spearman IC for each factor for each month.

    For each month, correlates factor z-scores at month start with
    realised returns during that month. Returns columns:
    month_end, factor_name, ic_value.
    """
    fs = factor_scores.copy()
    fs["score_date"] = pd.to_datetime(fs["score_date"])
    fs["year_month"] = fs["score_date"].dt.to_period("M")

    mr = monthly_returns.copy()
    mr["month_end"] = pd.to_datetime(mr["month_end"])
    mr["year_month"] = mr["month_end"].dt.to_period("M")

    # Join factor scores with next month's returns
    # Factor scores at month t predict returns during month t+1
    fs["forward_month"] = fs["year_month"] + 1

    merged = fs.merge(
        mr[["symbol", "year_month", "monthly_return"]],
        left_on=["symbol", "forward_month"],
        right_on=["symbol", "year_month"],
        suffixes=("_score", "_return"),
    )

    if merged.empty:
        return pd.DataFrame(columns=["month_end", "factor_name", "ic_value"])

    results = []
    for period, group in merged.groupby("year_month_score"):
        if len(group) < 10:
            continue
        month_end = period.to_timestamp("M")
        for factor_name, col in zip(FACTOR_NAMES, FACTOR_COLS):
            valid = group[[col, "monthly_return"]].dropna()
            if len(valid) < 10:
                continue
            corr, _ = spearmanr(valid[col], valid["monthly_return"])
            results.append(
                {
                    "month_end": month_end,
                    "factor_name": factor_name,
                    "ic_value": corr,
                }
            )

    return pd.DataFrame(results)


def compute_ic_weights(
    monthly_ics: pd.DataFrame,
    excluded_factor: str | None = None,
) -> pd.DataFrame:
    """Compute IC-derived factor weights with zero-flooring.

    Returns columns: factor_name, ic_mean_36m, ic_weight.
    Weights sum to 1.0. Negative mean ICs are floored to 0.
    If all factors have negative IC, falls back to equal weights.

    Args:
        monthly_ics:     Per-month IC values from compute_monthly_ic.
        excluded_factor: If provided (one of FACTOR_NAMES), force that factor's
                         weight to 0 and renormalise the others. Used by Step 10
                         factor exclusion analysis.
    """
    if excluded_factor is not None and excluded_factor not in FACTOR_NAMES:
        raise ValueError(
            f"excluded_factor must be one of {FACTOR_NAMES}, got '{excluded_factor}'"
        )

    if monthly_ics.empty:
        result = pd.DataFrame(
            {
                "factor_name": list(FACTOR_NAMES),
                "ic_mean_36m": [0.0] * 4,
                "ic_weight": [0.25] * 4,
            }
        )
        if excluded_factor is not None:
            mask = result["factor_name"] == excluded_factor
            result.loc[mask, "ic_weight"] = 0.0
            remaining = result.loc[~mask, "ic_weight"]
            result.loc[~mask, "ic_weight"] = remaining / remaining.sum()
        return result

    mean_ics = (
        monthly_ics.groupby("factor_name")["ic_value"]
        .mean()
        .reindex(FACTOR_NAMES, fill_value=0.0)
        .reset_index()
    )
    mean_ics.columns = ["factor_name", "ic_mean_36m"]

    # Zero-flooring: negative ICs get weight 0
    mean_ics["ic_floored"] = mean_ics["ic_mean_36m"].clip(lower=0.0)

    # Step 10 factor exclusion: zero out the excluded factor before renormalising
    if excluded_factor is not None:
        mean_ics.loc[mean_ics["factor_name"] == excluded_factor, "ic_floored"] = 0.0

    total = mean_ics["ic_floored"].sum()
    if total <= 0:
        # All factors negative — fall back to equal weights (excluding the
        # excluded factor if one is set)
        if excluded_factor is not None:
            mean_ics["ic_weight"] = mean_ics["factor_name"].apply(
                lambda f: 0.0 if f == excluded_factor else 1.0 / 3.0
            )
        else:
            mean_ics["ic_weight"] = 0.25
    else:
        mean_ics["ic_weight"] = mean_ics["ic_floored"] / total

    return mean_ics[["factor_name", "ic_mean_36m", "ic_weight"]]


def compute_composite_score(
    factor_scores: pd.DataFrame,
    ic_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Apply IC-derived weights to factor z-scores to produce composite score.

    Returns columns: symbol, composite_score.
    """
    weights = dict(zip(ic_weights["factor_name"], ic_weights["ic_weight"]))

    result = factor_scores[["symbol"]].copy()
    result["composite_score"] = (
        weights.get("value", 0.25) * factor_scores["z_value"]
        + weights.get("quality", 0.25) * factor_scores["z_quality"]
        + weights.get("momentum", 0.25) * factor_scores["z_momentum"]
        + weights.get("low_vol", 0.25) * factor_scores["z_low_vol"]
    )
    return result


def _update_composite_scores(
    db: PostgresConnection,
    composite: pd.DataFrame,
    score_date: date,
) -> None:
    """Bulk update composite_score in factor_scores table.

    Writes composite scores to a temp table, then updates factor_scores
    via a single JOIN-based UPDATE statement.
    """
    if composite.empty:
        return

    temp_table = "_tmp_composite"

    # Write composite scores to a temporary staging table
    output = pd.DataFrame(
        {"symbol": composite["symbol"], "composite_score": composite["composite_score"]}
    )
    db.write_dataframe(output, temp_table, SCHEMA, if_exists="replace")

    # Bulk UPDATE factor_scores from the temp table.
    # SCHEMA and temp_table are internal constants (not user input).
    # The user-supplied score_date is bound as a parameter.
    update_query = f"""
        UPDATE {SCHEMA}.factor_scores fs
        SET composite_score = t.composite_score
        FROM {SCHEMA}.{temp_table} t
        WHERE fs.symbol = t.symbol
          AND fs.score_date = :score_date
    """  # nosec B608
    db.execute(update_query, {"score_date": score_date})

    # Clean up temp table
    db.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{temp_table}")

    logger.info(
        "Updated composite_score for %d stocks (score_date=%s)",
        len(composite),
        score_date,
    )


def _persist_ic_weights(
    db: PostgresConnection,
    ic_weights: pd.DataFrame,
    rebalance_date: date,
) -> None:
    """Write IC weights to the ic_weights table."""
    output = ic_weights.copy()
    output["rebalance_date"] = rebalance_date
    output = output[["rebalance_date", "factor_name", "ic_mean_36m", "ic_weight"]]
    db.write_dataframe_on_conflict_do_nothing(
        output, "ic_weights", SCHEMA, conflict_columns=["rebalance_date", "factor_name"]
    )
    logger.info("Persisted IC weights for %s", rebalance_date)


def run_composite_scorer(
    db: PostgresConnection,
    rebalance_date: date,
    config: CompositeConfig,
    excluded_factor: str | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    """Compute IC-weighted composite scores and (optionally) persist them.

    Args:
        db:              PostgresConnection used for reading factor z-scores
                         and prices. Writes are gated by `persist`.
        rebalance_date:  Month-end date the composite score applies to.
        config:          CompositeConfig with IC lookback and minimum months.
        excluded_factor: If provided (one of FACTOR_NAMES), the named factor's
                         IC weight is zeroed and remaining factors are
                         renormalised. Used by Step 10 factor exclusion.
        persist:         When True (default), writes ic_weights and updates
                         factor_scores.composite_score in the DB. When False,
                         skips DB writes - used by variant scenarios that
                         must not pollute baseline data.

    Returns:
        DataFrame with symbol and composite_score for downstream use.
    """
    # Step 1: fetch data
    prices = _fetch_monthly_prices(db, rebalance_date, config.ic_lookback_months)
    if prices.empty:
        logger.warning("No price data for IC computation (date=%s)", rebalance_date)
        return pd.DataFrame(columns=["symbol", "composite_score"])

    factor_scores = _fetch_factor_scores(db, rebalance_date, config.ic_lookback_months)
    if factor_scores.empty:
        logger.warning("No factor scores available (date=%s)", rebalance_date)
        return pd.DataFrame(columns=["symbol", "composite_score"])

    # Step 2: compute monthly returns
    monthly_returns = compute_monthly_returns(prices)

    # Step 3: compute monthly IC per factor
    monthly_ics = compute_monthly_ic(factor_scores, monthly_returns)
    n_months = monthly_ics["month_end"].nunique() if not monthly_ics.empty else 0
    logger.info("Computed ICs over %d months (date=%s)", n_months, rebalance_date)

    # Step 4: average ICs + zero-flooring + normalise (with optional exclusion)
    if n_months < config.min_ic_months:
        logger.warning(
            "Only %d months of IC data (min=%d), using equal weights (date=%s)",
            n_months,
            config.min_ic_months,
            rebalance_date,
        )
        ic_weights = compute_ic_weights(pd.DataFrame(), excluded_factor=excluded_factor)
    else:
        ic_weights = compute_ic_weights(monthly_ics, excluded_factor=excluded_factor)
    for _, row in ic_weights.iterrows():
        logger.info(
            "  %s: mean_IC=%.4f, weight=%.3f",
            row["factor_name"],
            row["ic_mean_36m"],
            row["ic_weight"],
        )

    # Step 5: compute composite scores using current month's factor scores
    current_scores = factor_scores[
        factor_scores["score_date"] == factor_scores["score_date"].max()
    ]
    if current_scores.empty:
        logger.warning("No current factor scores (date=%s)", rebalance_date)
        return pd.DataFrame(columns=["symbol", "composite_score"])

    composite = compute_composite_score(current_scores, ic_weights)

    # Step 6: persist (only when explicitly requested)
    if persist:
        _persist_ic_weights(db, ic_weights, rebalance_date)
        _update_composite_scores(db, composite, current_scores["score_date"].max())

    logger.info(
        "Composite scores: %d stocks, range [%.3f, %.3f] (date=%s, excluded=%s)",
        len(composite),
        composite["composite_score"].min(),
        composite["composite_score"].max(),
        rebalance_date,
        excluded_factor or "none",
    )

    return composite
