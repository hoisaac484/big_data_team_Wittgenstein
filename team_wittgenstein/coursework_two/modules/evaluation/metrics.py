"""Step 7: Summary metrics for a backtest scenario.

Reads monthly returns from ``backtest_returns`` for a given ``scenario_id`` and
computes return, risk, risk-adjusted, and trading metrics. Results are written
to ``backtest_summary`` keyed by ``scenario_id``.

Formula summary:
- Sharpe = ``(R_ann - R_f) / sigma_ann``
- Sortino = ``(R_ann - R_f) / downside_deviation``
- Calmar = ``R_ann / abs(max_drawdown)``
- Information ratio = ``alpha / tracking_error``
"""

import logging

import numpy as np
import pandas as pd

from modules.db.db_connection import PostgresConnection
from modules.output.data_writer import DataWriter

logger = logging.getLogger(__name__)
SCHEMA = "team_wittgenstein"


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------


def fetch_scenario_returns(db: PostgresConnection, scenario_id: str) -> pd.DataFrame:
    """Load all monthly return rows for a scenario, sorted by date."""
    query = """
        SELECT rebalance_date, gross_return, net_return, long_return,
               short_return, benchmark_return, excess_return,
               cumulative_return, turnover, transaction_cost
        FROM team_wittgenstein.backtest_returns
        WHERE scenario_id = :scenario_id
        ORDER BY rebalance_date
    """
    return db.read_query(query, {"scenario_id": scenario_id})


def fetch_risk_free_rate(db: PostgresConnection, country: str = "USA") -> float:
    """Return the average annualised risk-free rate for the country.

    Uses the full history in risk_free_rates - for the backtest period
    this is a reasonable approximation of the period-average rate.
    Returns 0.0 if no rates are available.
    """
    query = """
        SELECT AVG(rate) AS avg_rate
        FROM team_wittgenstein.risk_free_rates
        WHERE country = :country
    """
    result = db.read_query(query, {"country": country})
    if result is None or result.empty or pd.isna(result.iloc[0]["avg_rate"]):
        return 0.0
    return float(result.iloc[0]["avg_rate"])


# ---------------------------------------------------------------------------
# Individual metric functions (pure, testable)
# ---------------------------------------------------------------------------


def annualised_return(monthly_returns: pd.Series) -> float:
    """Geometric annualised return: (∏(1 + r))^(12/N) - 1."""
    n = len(monthly_returns)
    if n == 0:
        return 0.0
    total = (1 + monthly_returns).prod() - 1
    return float((1 + total) ** (12 / n) - 1)


def cumulative_return(monthly_returns: pd.Series) -> float:
    """Compounded total return over the full period."""
    if len(monthly_returns) == 0:
        return 0.0
    return float((1 + monthly_returns).prod() - 1)


def annualised_volatility(monthly_returns: pd.Series) -> float:
    """Standard deviation of monthly returns scaled by sqrt(12)."""
    if len(monthly_returns) < 2:
        return 0.0
    return float(monthly_returns.std(ddof=1) * np.sqrt(12))


def max_drawdown(monthly_returns: pd.Series) -> float:
    """Largest peak-to-trough decline in cumulative return (returned as negative)."""
    if len(monthly_returns) == 0:
        return 0.0
    cumulative = (1 + monthly_returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min())


def downside_deviation(monthly_returns: pd.Series, monthly_rf: float = 0.0) -> float:
    """Annualised std of below-MAR returns, where MAR = monthly risk-free rate.

    Per Sortino (1994), only returns below the minimum acceptable return
    contribute. Positive excess returns are treated as zero.
    """
    if len(monthly_returns) == 0:
        return 0.0
    downside = np.minimum(monthly_returns - monthly_rf, 0.0)
    return float(np.sqrt((downside**2).mean()) * np.sqrt(12))


def tracking_error(net_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualised std of (portfolio - benchmark) returns."""
    diff = net_returns - benchmark_returns
    diff = diff.dropna()
    if len(diff) < 2:
        return 0.0
    return float(diff.std(ddof=1) * np.sqrt(12))


def sharpe_ratio(
    ann_return: float, risk_free_rate: float, ann_volatility: float
) -> float:
    if ann_volatility == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_volatility


def sortino_ratio(
    ann_return: float, risk_free_rate: float, downside_dev: float
) -> float:
    if downside_dev == 0:
        return 0.0
    return (ann_return - risk_free_rate) / downside_dev


def calmar_ratio(ann_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return ann_return / abs(max_dd)


def information_ratio(alpha: float, te: float) -> float:
    if te == 0:
        return 0.0
    return alpha / te


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_summary_metrics(
    db: PostgresConnection,
    scenario_id: str,
    risk_free_rate: float | None = None,
) -> dict:
    """Compute and persist summary metrics for a scenario.

    Args:
        db:             Active PostgresConnection.
        scenario_id:    Scenario key, e.g. 'baseline', 'cost_high'.
        risk_free_rate: Annualised US risk-free rate. If None, averages
                        the risk_free_rates table for USA.

    Returns:
        Dict of the computed metrics that were written to backtest_summary.

    Raises:
        ValueError: if no backtest_returns exist for the scenario_id.
    """
    returns = fetch_scenario_returns(db, scenario_id)
    if returns is None or returns.empty:
        raise ValueError(f"No backtest_returns rows for scenario_id='{scenario_id}'")

    if risk_free_rate is None:
        risk_free_rate = fetch_risk_free_rate(db, "USA")
    monthly_rf = risk_free_rate / 12

    net = returns["net_return"].dropna()
    bench = returns["benchmark_return"].dropna()
    long_ret = returns["long_return"].dropna()
    short_ret = returns["short_return"].dropna()
    turnover = returns["turnover"].dropna()

    # Return metrics
    ann_ret = annualised_return(net)
    cum_ret = cumulative_return(net)
    bench_ann = annualised_return(bench)
    bench_cum = cumulative_return(bench)
    alpha = ann_ret - bench_ann
    long_contrib = float(long_ret.sum()) if not long_ret.empty else 0.0
    short_contrib = float(short_ret.sum()) if not short_ret.empty else 0.0

    # Risk metrics
    ann_vol = annualised_volatility(net)
    max_dd = max_drawdown(net)
    down_dev = downside_deviation(net, monthly_rf)
    te = tracking_error(net, bench)

    # Risk-adjusted
    sharpe = sharpe_ratio(ann_ret, risk_free_rate, ann_vol)
    sortino = sortino_ratio(ann_ret, risk_free_rate, down_dev)
    calmar = calmar_ratio(ann_ret, max_dd)
    ir = information_ratio(alpha, te)

    # Benchmark risk + risk-adjusted (same formulas applied to bench series)
    bench_vol = annualised_volatility(bench)
    bench_max_dd = max_drawdown(bench)
    bench_down_dev = downside_deviation(bench, monthly_rf)
    bench_sharpe = sharpe_ratio(bench_ann, risk_free_rate, bench_vol)
    bench_sortino = sortino_ratio(bench_ann, risk_free_rate, bench_down_dev)
    bench_calmar = calmar_ratio(bench_ann, bench_max_dd)

    # Trading
    avg_turnover = float(turnover.mean()) if not turnover.empty else 0.0

    summary = {
        "scenario_id": scenario_id,
        "backtest_start": returns["rebalance_date"].min(),
        "backtest_end": returns["rebalance_date"].max(),
        "annualised_return": ann_ret,
        "cumulative_return": cum_ret,
        "annualised_volatility": ann_vol,
        "max_drawdown": max_dd,
        "downside_deviation": down_dev,
        "tracking_error": te,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "information_ratio": ir,
        "alpha": alpha,
        "benchmark_return_ann": bench_ann,
        "benchmark_return_cum": bench_cum,
        "benchmark_volatility": bench_vol,
        "benchmark_max_drawdown": bench_max_dd,
        "benchmark_sharpe": bench_sharpe,
        "benchmark_sortino": bench_sortino,
        "benchmark_calmar": bench_calmar,
        "avg_monthly_turnover": avg_turnover,
        "long_contribution": long_contrib,
        "short_contribution": short_contrib,
    }

    writer = DataWriter(db)
    writer.write_backtest_summary(summary)

    logger.info(
        "%s | ann_ret=%.4f vol=%.4f sharpe=%.4f sortino=%.4f max_dd=%.4f",
        scenario_id,
        ann_ret,
        ann_vol,
        sharpe,
        sortino,
        max_dd,
    )
    return summary
