"""All SQL queries used by the dashboard, centralised and cached.

Each function returns a DataFrame from a single query. Centralising means:
  - Schema changes touch one file
  - Caching is consistent (10 min TTL by default)
  - Pages stay free of SQL
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .db import query

CACHE_TTL = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Universe / metadata
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_database_stats() -> dict:
    """Top-line counts shown on the Home page.

    `stocks_used` is the **pre-liquidity-filter** investable universe -
    every symbol that has both price and fundamental data in the DB.
    The strategy then applies liquidity filters monthly to narrow this
    down to the stocks it actually scores and trades.
    """
    df = query("""
        SELECT
            (SELECT COUNT(*) FROM team_wittgenstein.backtest_summary) AS scenarios,
            (SELECT COUNT(DISTINCT p.symbol)
               FROM team_wittgenstein.price_data p
               INNER JOIN team_wittgenstein.financial_data f
                 ON p.symbol = f.symbol) AS stocks_used,
            (SELECT COUNT(DISTINCT rebalance_date)
               FROM team_wittgenstein.portfolio_positions) AS months,
            (SELECT MIN(rebalance_date)
               FROM team_wittgenstein.portfolio_positions) AS start_date,
            (SELECT MAX(rebalance_date)
               FROM team_wittgenstein.portfolio_positions) AS end_date
        """)
    return df.iloc[0].to_dict() if not df.empty else {}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_scenario_list() -> list[str]:
    """All scenario_ids in backtest_summary, sorted by group then name."""
    df = query("""
        SELECT scenario_id FROM team_wittgenstein.backtest_summary
        ORDER BY
          CASE
            WHEN scenario_id = 'baseline' THEN 1
            WHEN scenario_id LIKE 'cost_%' THEN 2
            WHEN scenario_id LIKE 'excl_%' THEN 3
            WHEN scenario_id LIKE 'sens_%' THEN 4
            ELSE 5
          END,
          scenario_id
        """)
    return df["scenario_id"].tolist()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_symbols() -> pd.DataFrame:
    """Universe of symbols with sector/industry mapping for Stock Deep-Dive.

    company_static.symbol is CHAR(12) which pads with trailing spaces;
    portfolio_positions.symbol is VARCHAR(12) which does not. We TRIM
    everywhere to keep symbol comparisons consistent across pages.
    """
    return query("""
        SELECT TRIM(symbol) AS symbol, security, gics_sector, gics_industry, country
        FROM systematic_equity.company_static
        WHERE TRIM(symbol) IN (
            SELECT DISTINCT TRIM(symbol) FROM team_wittgenstein.portfolio_positions
        )
        ORDER BY TRIM(symbol)
        """)


# ---------------------------------------------------------------------------
# Backtest results
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_summary(scenario_id: str = "baseline") -> pd.Series:
    """One-row summary for a scenario as a Series for easy attribute access."""
    df = query(
        "SELECT * FROM team_wittgenstein.backtest_summary WHERE scenario_id = :sid",
        {"sid": scenario_id},
    )
    return df.iloc[0] if not df.empty else pd.Series(dtype=object)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_all_summaries() -> pd.DataFrame:
    """All scenarios in one DataFrame - used for robustness comparisons."""
    return query(
        "SELECT * FROM team_wittgenstein.backtest_summary ORDER BY scenario_id"
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_returns(scenario_id: str = "baseline") -> pd.DataFrame:
    """Monthly returns for a single scenario."""
    df = query(
        """
        SELECT rebalance_date, gross_return, net_return, long_return,
               short_return, benchmark_return, excess_return,
               cumulative_return, turnover, transaction_cost
        FROM team_wittgenstein.backtest_returns
        WHERE scenario_id = :sid
        ORDER BY rebalance_date
        """,
        {"sid": scenario_id},
    )
    if not df.empty:
        df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    return df


# ---------------------------------------------------------------------------
# Portfolio composition (baseline only)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_rebalance_dates() -> list[pd.Timestamp]:
    """All rebalance dates from baseline portfolio_positions."""
    df = query("""
        SELECT DISTINCT rebalance_date
        FROM team_wittgenstein.portfolio_positions
        ORDER BY rebalance_date
        """)
    return pd.to_datetime(df["rebalance_date"]).tolist()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_holdings(rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """Long and short holdings for a specific rebalance date."""
    df = query(
        """
        SELECT pp.symbol, pp.sector, pp.direction, pp.final_weight,
               pp.target_weight, pp.risk_adj_score, pp.ewma_vol,
               pp.liquidity_capped, pp.trade_action,
               cs.security
        FROM team_wittgenstein.portfolio_positions pp
        LEFT JOIN systematic_equity.company_static cs ON pp.symbol = cs.symbol
        WHERE pp.rebalance_date = :rd
        ORDER BY pp.direction, pp.final_weight DESC
        """,
        {
            "rd": (
                rebalance_date.date()
                if hasattr(rebalance_date, "date")
                else rebalance_date
            )
        },
    )
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_selection_status(rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """Selection status for all stocks on a rebalance date."""
    df = query(
        """
        SELECT symbol, sector, composite_score, percentile_rank, status,
               buffer_months_count, entry_date, exit_reason
        FROM team_wittgenstein.selection_status
        WHERE rebalance_date = :rd
        """,
        {
            "rd": (
                rebalance_date.date()
                if hasattr(rebalance_date, "date")
                else rebalance_date
            )
        },
    )
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_selection_status_history() -> pd.DataFrame:
    """All selection statuses across time for the over-time charts."""
    return query("""
        SELECT rebalance_date, symbol, sector, status, percentile_rank,
               buffer_months_count, exit_reason
        FROM team_wittgenstein.selection_status
        ORDER BY rebalance_date, symbol
        """)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_all_positions() -> pd.DataFrame:
    """All baseline positions across all rebalance dates - for over-time charts."""
    df = query("""
        SELECT rebalance_date, symbol, sector, direction, final_weight,
               liquidity_capped, trade_action
        FROM team_wittgenstein.portfolio_positions
        ORDER BY rebalance_date, direction, final_weight DESC
        """)
    if not df.empty:
        df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_position_history(symbol: str) -> pd.DataFrame:
    """All positions held for a single stock over time."""
    df = query(
        """
        SELECT rebalance_date, sector, direction, final_weight,
               target_weight, risk_adj_score, ewma_vol, liquidity_capped,
               trade_action
        FROM team_wittgenstein.portfolio_positions
        WHERE symbol = :sym
        ORDER BY rebalance_date
        """,
        {"sym": symbol},
    )
    if not df.empty:
        df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    return df


# ---------------------------------------------------------------------------
# Factor analysis
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ic_weights() -> pd.DataFrame:
    """All IC weights across time and factor."""
    df = query("""
        SELECT rebalance_date, factor_name, ic_mean_36m, ic_weight
        FROM team_wittgenstein.ic_weights
        ORDER BY rebalance_date, factor_name
        """)
    if not df.empty:
        df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_factor_scores(symbol: str) -> pd.DataFrame:
    """Factor z-scores over time for a specific stock."""
    df = query(
        """
        SELECT score_date, z_value, z_quality, z_momentum, z_low_vol,
               composite_score
        FROM team_wittgenstein.factor_scores
        WHERE symbol = :sym
        ORDER BY score_date
        """,
        {"sym": symbol},
    )
    if not df.empty:
        df["score_date"] = pd.to_datetime(df["score_date"])
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_composite_distribution(rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """Composite scores for all stocks on a single rebalance date."""
    df = query(
        """
        SELECT fs.symbol, fs.composite_score, ss.sector
        FROM team_wittgenstein.factor_scores fs
        LEFT JOIN team_wittgenstein.selection_status ss
          ON ss.symbol = fs.symbol AND ss.rebalance_date = fs.score_date
        WHERE fs.score_date = :rd
          AND fs.composite_score IS NOT NULL
        """,
        {
            "rd": (
                rebalance_date.date()
                if hasattr(rebalance_date, "date")
                else rebalance_date
            )
        },
    )
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_active_factor_count() -> int:
    """Count factors with non-zero average IC weight - i.e. ones currently active."""
    df = query("""
        SELECT factor_name, AVG(ic_weight) AS avg_w
        FROM team_wittgenstein.ic_weights
        GROUP BY factor_name
        HAVING AVG(ic_weight) > 0.001
        """)
    return len(df)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_universe_size_latest() -> int:
    """Number of unique stocks in the most recent rebalance's holdings."""
    df = query("""
        SELECT COUNT(DISTINCT symbol) AS n
        FROM team_wittgenstein.portfolio_positions
        WHERE rebalance_date = (
            SELECT MAX(rebalance_date) FROM team_wittgenstein.portfolio_positions
        )
        """)
    return int(df.iloc[0]["n"]) if not df.empty else 0


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_latest_net_exposure() -> dict:
    """Long sum, short sum, net exposure on the most recent rebalance date.

    Short weights are stored as positive numbers in the DB (the sum of
    absolute weights); net exposure = long - short (since shorts are
    negative real-world exposure).
    """
    df = query("""
        SELECT direction, SUM(final_weight) AS total
        FROM team_wittgenstein.portfolio_positions
        WHERE rebalance_date = (
            SELECT MAX(rebalance_date) FROM team_wittgenstein.portfolio_positions
        )
        GROUP BY direction
        """)
    out = {"long": 0.0, "short": 0.0}
    for _, row in df.iterrows():
        out[row["direction"]] = float(row["total"])
    out["net"] = out["long"] - abs(out["short"])
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_factor_correlations() -> pd.DataFrame:
    """Pull all 4 factor z-scores across all dates+stocks for correlation analysis."""
    df = query("""
        SELECT z_value, z_quality, z_momentum, z_low_vol
        FROM team_wittgenstein.factor_scores
        WHERE z_value IS NOT NULL
          AND z_quality IS NOT NULL
          AND z_momentum IS NOT NULL
          AND z_low_vol IS NOT NULL
        """)
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_zscore_by_sector(score_date: pd.Timestamp, factor_col: str) -> pd.DataFrame:
    """All stocks' z-score for one factor on one date, with sector. For boxplots."""
    valid_cols = {"z_value", "z_quality", "z_momentum", "z_low_vol"}
    if factor_col not in valid_cols:
        raise ValueError(f"Unknown factor column: {factor_col}")
    # factor_col is whitelist-validated above so f-string interpolation is safe.
    sql = f"""
        SELECT fs.symbol, fs.{factor_col} AS z, ss.sector
        FROM team_wittgenstein.factor_scores fs
        LEFT JOIN team_wittgenstein.selection_status ss
          ON ss.symbol = fs.symbol AND ss.rebalance_date = fs.score_date
        WHERE fs.score_date = :rd
          AND fs.{factor_col} IS NOT NULL
          AND ss.sector IS NOT NULL
        """  # nosec B608 - factor_col validated against whitelist
    df = query(
        sql,
        {"rd": score_date.date() if hasattr(score_date, "date") else score_date},
    )
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_factor_metrics(symbol: str) -> pd.DataFrame:
    """Raw factor metrics over time for a specific stock."""
    df = query(
        """
        SELECT calc_date, pb_ratio, asset_growth, roe, leverage,
               earnings_stability, momentum_6m, momentum_12m,
               volatility_3m, volatility_12m
        FROM team_wittgenstein.factor_metrics
        WHERE symbol = :sym
        ORDER BY calc_date
        """,
        {"sym": symbol},
    )
    if not df.empty:
        df["calc_date"] = pd.to_datetime(df["calc_date"])
    return df


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_prices(symbol: str) -> pd.DataFrame:
    """Daily price history for a single stock."""
    df = query(
        """
        SELECT trade_date, adjusted_close, volume
        FROM team_wittgenstein.price_data
        WHERE symbol = :sym
        ORDER BY trade_date
        """,
        {"sym": symbol},
    )
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df
