"""Tests for Step 7 summary metrics.

Each individual metric function is tested with hand-calculated inputs.
The orchestrator is tested with a mocked DB so no live DB connection
is required.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modules.evaluation.metrics import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    compute_summary_metrics,
    cumulative_return,
    downside_deviation,
    fetch_risk_free_rate,
    fetch_scenario_returns,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)

# ---------------------------------------------------------------------------
# Individual return metric functions
# ---------------------------------------------------------------------------


class TestAnnualisedReturn:

    def test_constant_1pct_monthly(self):
        """12 months of 1% returns compound to ~12.68% annualised."""
        r = pd.Series([0.01] * 12)
        # (1.01)^12 - 1 = 0.126825...
        assert abs(annualised_return(r) - (1.01**12 - 1)) < 1e-10

    def test_short_window_annualises_up(self):
        """6 months of 1% compound to (1.01)^12 - 1 annualised."""
        r = pd.Series([0.01] * 6)
        # Cumulative (1.01)^6 - 1, annualised to (1.01)^12 - 1
        expected = (1.01**6) ** (12 / 6) - 1
        assert abs(annualised_return(r) - expected) < 1e-10

    def test_empty_series_returns_zero(self):
        assert annualised_return(pd.Series([], dtype=float)) == 0.0


class TestCumulativeReturn:

    def test_two_months(self):
        """+10% then -5% gives (1.10)(0.95) - 1 = 0.045."""
        r = pd.Series([0.10, -0.05])
        assert abs(cumulative_return(r) - 0.045) < 1e-10

    def test_empty_series_returns_zero(self):
        assert cumulative_return(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


class TestAnnualisedVolatility:

    def test_known_std(self):
        """Std of [0.01, -0.01, 0.02, -0.02] * sqrt(12)."""
        r = pd.Series([0.01, -0.01, 0.02, -0.02])
        expected = r.std(ddof=1) * np.sqrt(12)
        assert abs(annualised_volatility(r) - expected) < 1e-10

    def test_single_value_returns_zero(self):
        """ddof=1 needs at least 2 observations."""
        assert annualised_volatility(pd.Series([0.01])) == 0.0


class TestMaxDrawdown:

    def test_simple_drawdown(self):
        """+10%, -20%, +5% -> cumulative 1.10, 0.88, 0.924. MDD = (0.88 - 1.10)/1.10."""
        r = pd.Series([0.10, -0.20, 0.05])
        expected = (0.88 - 1.10) / 1.10
        assert abs(max_drawdown(r) - expected) < 1e-10

    def test_monotonic_up_has_zero_drawdown(self):
        """If returns are never negative, drawdown is 0."""
        r = pd.Series([0.01, 0.02, 0.03])
        assert max_drawdown(r) == 0.0

    def test_empty_returns_zero(self):
        assert max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestDownsideDeviation:

    def test_only_negative_excess_contributes(self):
        """Excess returns above RF contribute 0; only below-RF are squared.

        Returns [-0.02, 0.03, -0.01, 0.04], monthly_rf = 0.
        Only -0.02 and -0.01 contribute.
        Downside variance = mean([0.0004, 0, 0.0001, 0]) = 0.000125
        Annualised = sqrt(0.000125) * sqrt(12)
        """
        r = pd.Series([-0.02, 0.03, -0.01, 0.04])
        downside = np.minimum(r - 0, 0)
        expected = float(np.sqrt((downside**2).mean()) * np.sqrt(12))
        assert abs(downside_deviation(r, 0) - expected) < 1e-10

    def test_all_above_rf_returns_zero(self):
        """If every return exceeds RF, downside deviation is 0."""
        r = pd.Series([0.02, 0.03, 0.04])
        assert downside_deviation(r, 0.01) == 0.0

    def test_empty_series_returns_zero(self):
        assert downside_deviation(pd.Series([], dtype=float)) == 0.0


class TestTrackingError:

    def test_known_difference(self):
        """Std of (portfolio - benchmark) * sqrt(12)."""
        net = pd.Series([0.02, 0.01, 0.03, -0.01])
        bench = pd.Series([0.01, 0.01, 0.02, 0.00])
        diff = net - bench
        expected = diff.std(ddof=1) * np.sqrt(12)
        assert abs(tracking_error(net, bench) - expected) < 1e-10

    def test_identical_returns_zero_te(self):
        """If portfolio matches benchmark, tracking error is 0."""
        net = pd.Series([0.02, 0.01, 0.03])
        bench = pd.Series([0.02, 0.01, 0.03])
        assert tracking_error(net, bench) == 0.0

    def test_fewer_than_two_valid_diffs_returns_zero(self):
        """With only one non-NaN diff, std is undefined → return 0."""
        net = pd.Series([0.02, float("nan")])
        bench = pd.Series([0.01, float("nan")])
        assert tracking_error(net, bench) == 0.0


# ---------------------------------------------------------------------------
# Risk-adjusted metrics
# ---------------------------------------------------------------------------


class TestSharpeRatio:

    def test_basic_formula(self):
        """(0.12 - 0.02) / 0.15 = 0.6667."""
        assert abs(sharpe_ratio(0.12, 0.02, 0.15) - (0.10 / 0.15)) < 1e-10

    def test_zero_volatility_returns_zero(self):
        """Avoid divide-by-zero when volatility is 0."""
        assert sharpe_ratio(0.10, 0.02, 0.0) == 0.0


class TestSortinoRatio:

    def test_basic_formula(self):
        """(0.12 - 0.02) / 0.08 = 1.25."""
        assert abs(sortino_ratio(0.12, 0.02, 0.08) - 1.25) < 1e-10

    def test_zero_downside_returns_zero(self):
        assert sortino_ratio(0.10, 0.02, 0.0) == 0.0


class TestCalmarRatio:

    def test_basic_formula(self):
        """0.15 / |-0.20| = 0.75."""
        assert abs(calmar_ratio(0.15, -0.20) - 0.75) < 1e-10

    def test_zero_drawdown_returns_zero(self):
        assert calmar_ratio(0.15, 0.0) == 0.0


class TestInformationRatio:

    def test_basic_formula(self):
        """alpha=0.03, tracking_error=0.05 -> 0.6."""
        assert abs(information_ratio(0.03, 0.05) - 0.6) < 1e-10

    def test_zero_te_returns_zero(self):
        assert information_ratio(0.03, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Compute summary orchestrator
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_returns_df():
    """Minimal backtest_returns DataFrame with known values."""
    return pd.DataFrame(
        {
            "rebalance_date": [
                date(2024, 1, 31),
                date(2024, 2, 29),
                date(2024, 3, 31),
                date(2024, 4, 30),
            ],
            "gross_return": [0.02, 0.01, -0.01, 0.03],
            "net_return": [0.018, 0.008, -0.012, 0.028],
            "long_return": [0.025, 0.015, 0.005, 0.035],
            "short_return": [0.005, 0.005, 0.015, 0.005],
            "benchmark_return": [0.015, 0.010, -0.005, 0.020],
            "excess_return": [0.003, -0.002, -0.007, 0.008],
            "cumulative_return": [0.018, 0.026, 0.014, 0.043],
            "turnover": [0.25, 0.30, 0.28, 0.22],
            "transaction_cost": [0.002, 0.002, 0.002, 0.002],
        }
    )


class TestFetchScenarioReturns:

    def test_query_filtered_by_scenario(self, mock_returns_df):
        """The SQL query is parameterised by scenario_id."""
        db = MagicMock()
        db.read_query.return_value = mock_returns_df
        result = fetch_scenario_returns(db, "baseline")

        assert not result.empty
        args, kwargs = db.read_query.call_args
        # Either 'baseline' is in query as literal (wrong) or in params (right)
        params = kwargs.get("params") or args[1]
        assert params["scenario_id"] == "baseline"


class TestFetchRiskFreeRate:

    def test_returns_average_rate(self):
        """DB returns an average; function unwraps to float."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame({"avg_rate": [0.035]})
        assert fetch_risk_free_rate(db) == 0.035

    def test_empty_db_returns_zero(self):
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()
        assert fetch_risk_free_rate(db) == 0.0

    def test_null_rate_returns_zero(self):
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame({"avg_rate": [float("nan")]})
        assert fetch_risk_free_rate(db) == 0.0


class TestComputeSummaryMetrics:

    def test_empty_returns_raises(self):
        """No backtest_returns for scenario -> ValueError."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No backtest_returns"):
            compute_summary_metrics(db, "missing", risk_free_rate=0.02)

    def test_full_flow_computes_all_metrics(self, mock_returns_df):
        """End-to-end: returns DataFrame in -> all expected keys out."""
        db = MagicMock()
        db.read_query.return_value = mock_returns_df

        # Patch DataWriter so we don't hit the DB for writing
        with patch("modules.evaluation.metrics.DataWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer_cls.return_value = mock_writer

            result = compute_summary_metrics(db, "baseline", risk_free_rate=0.02)

        expected_keys = {
            "scenario_id",
            "backtest_start",
            "backtest_end",
            "annualised_return",
            "cumulative_return",
            "annualised_volatility",
            "max_drawdown",
            "downside_deviation",
            "tracking_error",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "information_ratio",
            "alpha",
            "benchmark_return_ann",
            "benchmark_return_cum",
            "benchmark_volatility",
            "benchmark_max_drawdown",
            "benchmark_sharpe",
            "benchmark_sortino",
            "benchmark_calmar",
            "avg_monthly_turnover",
            "long_contribution",
            "short_contribution",
        }
        assert set(result.keys()) == expected_keys
        assert result["scenario_id"] == "baseline"
        assert result["backtest_start"] == date(2024, 1, 31)
        assert result["backtest_end"] == date(2024, 4, 30)

    def test_writer_called_with_summary(self, mock_returns_df):
        """Result dict is passed to DataWriter.write_backtest_summary."""
        db = MagicMock()
        db.read_query.return_value = mock_returns_df

        with patch("modules.evaluation.metrics.DataWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer_cls.return_value = mock_writer

            compute_summary_metrics(db, "baseline", risk_free_rate=0.02)

        mock_writer.write_backtest_summary.assert_called_once()
        args, _ = mock_writer.write_backtest_summary.call_args
        assert args[0]["scenario_id"] == "baseline"

    def test_cumulative_return_matches_hand_calc(self, mock_returns_df):
        """Cumulative return from net returns matches manual cumprod."""
        db = MagicMock()
        db.read_query.return_value = mock_returns_df

        with patch("modules.evaluation.metrics.DataWriter"):
            result = compute_summary_metrics(db, "baseline", risk_free_rate=0.02)

        net = mock_returns_df["net_return"]
        expected = (1 + net).prod() - 1
        assert abs(result["cumulative_return"] - expected) < 1e-10

    def test_uses_db_rf_rate_when_none_provided(self, mock_returns_df):
        """If risk_free_rate is None, it's fetched from the DB."""
        db = MagicMock()
        # First call: returns; second call: rf rate
        db.read_query.side_effect = [
            mock_returns_df,
            pd.DataFrame({"avg_rate": [0.04]}),
        ]

        with patch("modules.evaluation.metrics.DataWriter"):
            result = compute_summary_metrics(db, "baseline", risk_free_rate=None)

        # Sharpe should use 0.04 as RF
        ann_ret = result["annualised_return"]
        ann_vol = result["annualised_volatility"]
        expected_sharpe = (ann_ret - 0.04) / ann_vol
        assert abs(result["sharpe_ratio"] - expected_sharpe) < 1e-10

    def test_benchmark_metrics_match_pure_functions(self, mock_returns_df):
        """Benchmark metrics use the same pure functions as portfolio metrics."""
        db = MagicMock()
        db.read_query.return_value = mock_returns_df

        with patch("modules.evaluation.metrics.DataWriter"):
            result = compute_summary_metrics(db, "baseline", risk_free_rate=0.02)

        bench = mock_returns_df["benchmark_return"]
        expected_vol = bench.std(ddof=1) * np.sqrt(12)
        assert abs(result["benchmark_volatility"] - expected_vol) < 1e-10

        bench_cum = (1 + bench).cumprod()
        expected_dd = ((bench_cum - bench_cum.cummax()) / bench_cum.cummax()).min()
        assert abs(result["benchmark_max_drawdown"] - expected_dd) < 1e-10

        # Sharpe = (bench_ann - rf) / bench_vol
        bench_ann = result["benchmark_return_ann"]
        expected_bench_sharpe = (bench_ann - 0.02) / expected_vol
        assert abs(result["benchmark_sharpe"] - expected_bench_sharpe) < 1e-10

        # Calmar = bench_ann / |bench_max_dd|
        expected_bench_calmar = bench_ann / abs(expected_dd)
        assert abs(result["benchmark_calmar"] - expected_bench_calmar) < 1e-10
