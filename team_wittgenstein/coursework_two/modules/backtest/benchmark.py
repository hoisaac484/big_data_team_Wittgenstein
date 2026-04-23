"""Step 1: Download and cache MSCI USA Index monthly returns.

Uses the iShares MSCI USA ETF (EUSA) as a proxy for the MSCI USA Index,
which the ETF tracks closely with negligible tracking error.

The monthly returns are cached in the `benchmark_returns` DB table so that
subsequent backtest runs and scenario variants read from the DB instead of
hitting yfinance on every run.
"""

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from modules.db.db_connection import PostgresConnection

logger = logging.getLogger(__name__)
SCHEMA = "team_wittgenstein"

MSCI_USA_TICKER = "EUSA"  # iShares MSCI USA ETF — tracks MSCI USA Index


def fetch_benchmark_monthly_returns(
    start_date: date,
    end_date: date,
) -> pd.Series:
    """Download MSCI USA monthly returns for the backtest window.

    Args:
        start_date: First date to include (one month before first return needed,
                    so pct_change has a base price).
        end_date:   Last date to include.

    Returns:
        Series indexed by month-end date (datetime.date) with monthly returns.
        Index represents the end of the return period.
    """
    # Download daily data and resample to avoid gaps in yfinance monthly feed
    raw = yf.download(
        MSCI_USA_TICKER,
        start=str(start_date),
        end=str(end_date),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"No benchmark data downloaded for {MSCI_USA_TICKER} "
            f"({start_date} → {end_date})"
        )

    close = raw["Close"].squeeze().sort_index()

    # Last trading day of each calendar month
    month_end_prices = close.resample("ME").last().dropna()
    monthly_returns = month_end_prices.pct_change().dropna()

    # Normalise index to calendar month-end dates
    monthly_returns.index = (
        pd.DatetimeIndex(monthly_returns.index).to_period("M").to_timestamp("M").date
    )

    logger.info(
        "Benchmark (%s): %d monthly returns | %s → %s",
        MSCI_USA_TICKER,
        len(monthly_returns),
        monthly_returns.index[0],
        monthly_returns.index[-1],
    )
    return monthly_returns


def backfill_benchmark_returns(
    db: PostgresConnection,
    start_date: date,
    end_date: date,
    benchmark: str = MSCI_USA_TICKER,
) -> int:
    """Fetch benchmark returns from yfinance and cache them in the DB.

    Idempotent: uses ON CONFLICT DO NOTHING on (benchmark, month_end) so
    re-running only inserts new rows. Safe to call before every backtest run.

    Args:
        db:         Active PostgresConnection.
        start_date: First date to fetch (fetches slightly earlier to get base price).
        end_date:   Last date to fetch.
        benchmark:  Ticker label stored in the table. Defaults to EUSA (MSCI USA).

    Returns:
        Number of monthly returns in the fetched series.
    """
    returns = fetch_benchmark_monthly_returns(start_date, end_date)

    df = pd.DataFrame(
        {
            "benchmark": benchmark,
            "month_end": list(returns.index),
            "monthly_return": returns.values,
        }
    )

    db.write_dataframe_on_conflict_do_nothing(
        df,
        "benchmark_returns",
        SCHEMA,
        conflict_columns=["benchmark", "month_end"],
    )
    logger.info(
        "Cached %d monthly benchmark returns (%s) to benchmark_returns",
        len(df),
        benchmark,
    )
    return len(df)


def load_benchmark_from_db(
    db: PostgresConnection,
    start_date: date,
    end_date: date,
    benchmark: str = MSCI_USA_TICKER,
) -> pd.Series:
    """Read cached benchmark returns from the DB.

    Args:
        db:         Active PostgresConnection.
        start_date: First month-end to include (inclusive).
        end_date:   Last month-end to include (inclusive).
        benchmark:  Ticker label. Defaults to EUSA (MSCI USA).

    Returns:
        Series indexed by month-end date, values are monthly returns.
        Empty Series if no rows match.
    """
    query = """
        SELECT month_end, monthly_return
        FROM team_wittgenstein.benchmark_returns
        WHERE benchmark = :benchmark
          AND month_end >= :start_date
          AND month_end <= :end_date
        ORDER BY month_end
    """
    df = db.read_query(
        query,
        {
            "benchmark": benchmark,
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)

    series = pd.Series(
        df["monthly_return"].astype(float).values,
        index=pd.to_datetime(df["month_end"]).dt.date,
    )
    return series
