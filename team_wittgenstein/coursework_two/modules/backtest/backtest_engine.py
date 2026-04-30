"""Steps 3-6: Backtest return computation for the 130/30 strategy.

Step 3: Gross monthly portfolio return
Step 4: Turnover and transaction costs
Step 5: Net return
Step 6: Excess return over benchmark
"""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from modules.backtest.benchmark import load_benchmark_from_db
from modules.db.db_connection import PostgresConnection

logger = logging.getLogger(__name__)
SCHEMA = "team_wittgenstein"


@dataclass(frozen=True)
class BacktestConfig:
    """Transaction cost parameters for baseline run."""

    cost_bps: float = 25.0  # one-way turnover cost in basis points
    borrow_rate: float = 0.0075  # annual short-borrow rate (0.75%)
    scenario_id: str = "baseline"


def _fetch_positions(db: PostgresConnection) -> pd.DataFrame:
    """Load all portfolio positions ordered by date."""
    return db.read_query("""
        SELECT rebalance_date, symbol, direction, final_weight
        FROM team_wittgenstein.portfolio_positions
        ORDER BY rebalance_date, symbol
        """)


def _fetch_prices_at_dates(
    db: PostgresConnection, rebalance_dates: list[date]
) -> pd.DataFrame:
    """Fetch adjusted close prices on or just before each rebalance date.

    Returns a wide DataFrame: index = rebalance_date, columns = symbol.
    Uses the last available price <= each rebalance date to handle
    thin-trading edge cases.
    """
    dates_sql = ", ".join(f"'{d}'" for d in rebalance_dates)
    # dates_sql is built from internal rebalance_date values (not user input).
    # Schema and column names are constants.
    query = f"""
        SELECT DISTINCT ON (symbol, ref_date)
               symbol,
               ref_date,
               adjusted_close
        FROM (
            SELECT p.symbol,
                   d.ref_date,
                   p.trade_date,
                   p.adjusted_close,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.symbol, d.ref_date
                       ORDER BY p.trade_date DESC
                   ) AS rn
            FROM team_wittgenstein.price_data p
            JOIN (SELECT unnest(ARRAY[{dates_sql}]::date[]) AS ref_date) d
              ON p.trade_date <= d.ref_date
        ) sub
        WHERE rn = 1
        ORDER BY symbol, ref_date
    """  # nosec B608
    raw = db.read_query(query)
    if raw is None or raw.empty:
        return pd.DataFrame()

    pivot = raw.pivot(index="ref_date", columns="symbol", values="adjusted_close")
    pivot.index = pd.to_datetime(pivot.index).date
    return pivot


def _compute_stock_returns(
    positions: pd.DataFrame,
    price_t: pd.Series,
    price_t1: pd.Series,
) -> pd.Series:
    """Compute per-symbol stock returns for the holding period.

    Returns a Series indexed by symbol with r_i = price_t1/price_t - 1.
    Symbols missing prices at either date are excluded (NaN dropped).
    """
    df = positions[["symbol"]].copy()
    df["price_t"] = df["symbol"].map(price_t)
    df["price_t1"] = df["symbol"].map(price_t1)
    df = df.dropna(subset=["price_t", "price_t1"])
    df = df[df["price_t"] > 0]
    df["stock_return"] = df["price_t1"] / df["price_t"] - 1
    return df.set_index("symbol")["stock_return"]


def _compute_gross_return(
    positions: pd.DataFrame,
    stock_returns: pd.Series,
) -> tuple[float, float, float]:
    """Compute gross portfolio return for one period.

    Spec (Step 3):
        Portfolio return = Σ(final_weight × stock_return) across all positions.
        Long return  = Σ(final_weight × stock_return) where direction = long.
        Short return = Σ(final_weight × stock_return) where direction = short.

    gross_return = long_return − short_return
    (longs profit when stocks rise; shorts profit when stocks fall)

    Args:
        positions:    DataFrame with columns symbol, direction, final_weight.
        stock_returns: Series indexed by symbol with holding-period returns.

    Returns:
        (gross_return, long_return, short_return)
    """
    df = positions.copy()
    df["stock_return"] = df["symbol"].map(stock_returns)
    df = df.dropna(subset=["stock_return"])

    long_pos = df[df["direction"] == "long"]
    short_pos = df[df["direction"] == "short"]

    long_ret = float((long_pos["final_weight"] * long_pos["stock_return"]).sum())
    short_ret = float((short_pos["final_weight"] * short_pos["stock_return"]).sum())
    gross = long_ret - short_ret

    return gross, long_ret, short_ret


def _compute_drift_adjusted_weights(
    previous: pd.DataFrame,
    stock_returns: pd.Series,
) -> pd.Series:
    """Compute drift-adjusted weights after price moves, before rebalancing.

    Spec (Step 4):
        w'_i,t-1 = (final_weight_i,t-1 × (1 + r_i,t))
                   / Σ_j [final_weight_j,t-1 × (1 + r_j,t)]

    Args:
        previous:     Previous period positions (symbol, final_weight).
        stock_returns: Returns during the holding period, indexed by symbol.

    Returns:
        Series of drift-adjusted weights indexed by symbol.
    """
    df = previous[["symbol", "final_weight"]].copy()
    df["r"] = df["symbol"].map(stock_returns).fillna(0.0)
    # Unnormalized: preserves the 130/30 weight scale so drift and new weights
    # are comparable. The spec denominator assumes weights sum to 1 (long-only);
    # for a 130/30 portfolio (gross notional ≈ 1.60) normalising would create
    # a systematic scale mismatch with the new final_weights.
    # total = df["drifted"].sum()
    # if total <= 0:
    #     return pd.Series(dtype=float)
    # df["w_drift"] = df["drifted"] / total
    df["w_drift"] = df["final_weight"] * (1 + df["r"])
    return df.set_index("symbol")["w_drift"]


def _compute_turnover(
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
    stock_returns: pd.Series | None,
) -> float:
    """Compute one-way portfolio turnover using drift-adjusted prior weights.

    Spec (Step 4):
        Turnover_t = Σ_i |final_weight_i,t − w'_i,t-1|

    where w'_i,t-1 is the drift-adjusted weight from the previous period.
    New entries (no prior weight) use w'=0; exits (no new weight) use w_t=0.

    Args:
        current:      Current period positions (symbol, final_weight).
        previous:     Previous period positions (symbol, final_weight, direction).
        stock_returns: Stock returns during the period for drift adjustment.

    Returns:
        Scalar turnover (one-way, unsigned).
    """
    w_curr = current.set_index("symbol")["final_weight"]

    if previous is None or previous.empty or stock_returns is None:
        # First period: treat as pre-invested — no entry cost
        return 0.0

    w_drift = _compute_drift_adjusted_weights(previous, stock_returns)

    all_symbols = w_curr.index.union(w_drift.index)
    w_curr = w_curr.reindex(all_symbols, fill_value=0.0)
    w_drift = w_drift.reindex(all_symbols, fill_value=0.0)

    # Direction lookup for flip detection
    dir_curr = current.set_index("symbol")["direction"].reindex(all_symbols)
    dir_prev = previous.set_index("symbol")["direction"].reindex(all_symbols)
    flipped = (dir_curr != dir_prev) & dir_curr.notna() & dir_prev.notna()

    # Flipped positions: close old (w_drift) + open new (w_curr) = sum not diff
    turnover = (w_curr - w_drift).abs()
    turnover[flipped] = w_drift[flipped] + w_curr[flipped]

    return float(turnover.sum())


def run_backtest(
    db: PostgresConnection,
    config: BacktestConfig,
    positions_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Execute Steps 1-6 and return monthly backtest results.

    Steps:
        1. Load MSCI USA benchmark monthly returns (EUSA proxy).
        3. Compute gross portfolio return per month.
        4. Compute turnover and transaction cost.
        5. Net return = gross − cost.
        6. Excess return = net − benchmark.

    Args:
        db:                 Active PostgresConnection (used for prices +
                            benchmark, plus positions when override absent).
        config:             BacktestConfig with cost_bps, borrow_rate,
                            scenario_id.
        positions_override: If provided, use this DataFrame instead of
                            reading portfolio_positions from the DB. Required
                            for variant scenarios (factor exclusion, parameter
                            sensitivity) where positions exist only in memory.
                            Must include columns: rebalance_date, symbol,
                            direction, final_weight.

    Returns:
        DataFrame written to backtest_returns (one row per month).
    """
    # Load positions (either from caller or from DB)
    positions_df = (
        positions_override.copy()
        if positions_override is not None
        else _fetch_positions(db)
    )
    if positions_df.empty:
        raise RuntimeError("portfolio_positions is empty — run backfill first.")

    positions_df["rebalance_date"] = pd.to_datetime(
        positions_df["rebalance_date"]
    ).dt.date
    rebalance_dates = sorted(positions_df["rebalance_date"].unique().tolist())

    # Step 1: benchmark keyed by (year, month) to handle business vs calendar month-end
    bench_start = date(rebalance_dates[0].year, rebalance_dates[0].month, 1)
    bench_end = rebalance_dates[-1]
    logger.info(
        "Step 1: Loading MSCI USA benchmark (EUSA) from DB cache %s → %s...",
        bench_start,
        bench_end,
    )
    _benchmark_raw = load_benchmark_from_db(db, bench_start, bench_end)
    if _benchmark_raw.empty:
        raise RuntimeError(
            "benchmark_returns is empty — run backfill_benchmark_returns first."
        )
    benchmark = {(d.year, d.month): v for d, v in _benchmark_raw.items()}
    logger.info("Loaded positions for %d rebalance dates.", len(rebalance_dates))

    # Fetch prices at all rebalance dates in one query
    logger.info("Fetching prices at %d rebalance dates...", len(rebalance_dates))
    price_grid = _fetch_prices_at_dates(db, rebalance_dates)

    results = []
    prev_positions = None

    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]  # portfolio constructed at t
        t1 = rebalance_dates[i + 1]  # held until t1

        pos_t = positions_df[positions_df["rebalance_date"] == t]

        # Step 3: Gross return (t → t1)
        if t not in price_grid.index:
            logger.warning("%s | no prices found — skipping period %s → %s", t, t, t1)
            prev_positions = pos_t
            continue
        if t1 not in price_grid.index:
            logger.warning("%s | no prices found — skipping period %s → %s", t1, t, t1)
            prev_positions = pos_t
            continue

        price_t = price_grid.loc[t]
        price_t1 = price_grid.loc[t1]

        stock_returns = _compute_stock_returns(pos_t, price_t, price_t1)
        gross, long_ret, short_ret = _compute_gross_return(pos_t, stock_returns)

        # Step 4: Turnover — drift-adjust prior weights, then compare to new weights
        turnover = _compute_turnover(pos_t, prev_positions, stock_returns)
        short_notional = pos_t[pos_t["direction"] == "short"]["final_weight"].sum()
        cost = (
            turnover * config.cost_bps / 10_000
            + float(short_notional) * config.borrow_rate / 12
        )

        # Step 5: Net return
        net = gross - cost

        # Step 6: Excess return — match by year-month (business vs calendar end)
        bench = float(benchmark.get((t1.year, t1.month), np.nan))
        excess = net - bench if not np.isnan(bench) else np.nan

        results.append(
            {
                "scenario_id": config.scenario_id,
                "rebalance_date": t1,
                "gross_return": gross,
                "net_return": net,
                "long_return": long_ret,
                "short_return": short_ret,
                "benchmark_return": bench,
                "excess_return": excess,
                "turnover": turnover,
                "transaction_cost": cost,
            }
        )

        logger.info(
            "%s | gross=%.4f  net=%.4f  bench=%.4f  excess=%.4f  turnover=%.4f",
            t1,
            gross,
            net,
            bench,
            excess if not np.isnan(excess) else float("nan"),
            turnover,
        )

        prev_positions = pos_t

    df = pd.DataFrame(results)

    if df.empty:
        logger.warning("No backtest periods produced results")
        return df

    # Cumulative net return
    df = df.sort_values("rebalance_date").reset_index(drop=True)
    df["cumulative_return"] = (1 + df["net_return"]).cumprod() - 1

    logger.info(
        "Backtest complete: %d monthly returns | scenario=%s",
        len(df),
        config.scenario_id,
    )
    return df
