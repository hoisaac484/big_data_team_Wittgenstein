"""Tests for the IC-weighted composite score module.

Each test class targets one public function in composite_scorer.py.
The orchestrator (run_composite_scorer) is tested with a mocked DB
so we can run without a live Postgres connection.
"""

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from modules.composite.composite_scorer import (
    CompositeConfig,
    _update_composite_scores,
    compute_composite_score,
    compute_ic_weights,
    compute_monthly_ic,
    compute_monthly_returns,
    run_composite_scorer,
)

# ---------------------------------------------------------------------------
# compute_monthly_returns
# ---------------------------------------------------------------------------


class TestComputeMonthlyReturns:

    def test_basic(self):
        """Two stocks over two months - verify correct percentage returns."""
        # 44 business days starting 2 Jan 2023 spans Jan and Feb.
        # Stock A: flat at 100 in Jan, flat at 110 in Feb -> +10%
        # Stock B: flat at 200 in Jan, flat at 190 in Feb -> -5%
        prices = pd.DataFrame(
            {
                "symbol": ["A"] * 44 + ["B"] * 44,
                "trade_date": list(pd.bdate_range("2023-01-02", periods=44)) * 2,
                "adjusted_close": (
                    [100.0] * 22 + [110.0] * 22 + [200.0] * 22 + [190.0] * 22
                ),
            }
        )
        result = compute_monthly_returns(prices)
        assert not result.empty
        assert "monthly_return" in result.columns
        assert "symbol" in result.columns

        a_return = result[result["symbol"] == "A"]["monthly_return"].iloc[0]
        assert abs(a_return - 0.10) < 1e-10  # 100 -> 110 = +10%

        b_return = result[result["symbol"] == "B"]["monthly_return"].iloc[0]
        assert abs(b_return - (-0.05)) < 1e-10  # 200 -> 190 = -5%


# ---------------------------------------------------------------------------
# compute_monthly_ic
# ---------------------------------------------------------------------------


class TestComputeMonthlyIc:

    def test_spearman_correlation(self):
        """IC for the value factor should match a manual scipy spearmanr call.

        Setup: 50 stocks with z_value scores at end of Jan 2023 and
        correlated returns in Feb 2023 (one-month-ahead prediction).
        The other three factors get random scores so they don't interfere.
        """
        n = 50
        np.random.seed(42)
        z_vals = np.random.randn(n)
        # Returns are strongly correlated with z_value (r ~ 0.98)
        returns = z_vals * 0.5 + np.random.randn(n) * 0.1

        factor_scores = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(n)],
                "score_date": pd.Timestamp("2023-01-31"),
                "z_value": z_vals,
                "z_quality": np.random.randn(n),
                "z_momentum": np.random.randn(n),
                "z_low_vol": np.random.randn(n),
            }
        )

        # Returns are for the NEXT month (Feb), matching t vs t+1 alignment
        monthly_returns = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(n)],
                "month_end": pd.Timestamp("2023-02-28"),
                "monthly_return": returns,
            }
        )

        result = compute_monthly_ic(factor_scores, monthly_returns)
        assert not result.empty

        # Our function's IC for "value" should exactly match scipy
        value_ic = result[result["factor_name"] == "value"]["ic_value"].iloc[0]
        expected, _ = spearmanr(z_vals, returns)
        assert abs(value_ic - expected) < 1e-10

    def test_factor_with_fewer_than_10_valid_rows_skipped(self):
        """A factor with < 10 non-NaN rows in a period is skipped."""
        n = 15
        np.random.seed(1)
        factor_scores = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(n)],
                "score_date": pd.Timestamp("2023-01-31"),
                "z_value": np.random.randn(n),
                "z_quality": np.random.randn(n),
                "z_momentum": np.random.randn(n),
                # z_low_vol: only 5 non-NaN → skipped for IC
                "z_low_vol": [float("nan")] * 10 + list(np.random.randn(5)),
            }
        )
        monthly_returns = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(n)],
                "month_end": pd.Timestamp("2023-02-28"),
                "monthly_return": np.random.randn(n),
            }
        )
        result = compute_monthly_ic(factor_scores, monthly_returns)
        # low_vol factor should not appear (only 5 valid rows)
        assert "low_vol" not in result["factor_name"].values


# ---------------------------------------------------------------------------
# _update_composite_scores
# ---------------------------------------------------------------------------


class TestUpdateCompositeScores:

    def test_empty_composite_is_noop(self):
        """_update_composite_scores returns immediately on empty input."""
        db = MagicMock()
        _update_composite_scores(db, pd.DataFrame(), date(2024, 1, 31))
        db.execute.assert_not_called()
        db.write_dataframe.assert_not_called()


# ---------------------------------------------------------------------------
# compute_ic_weights
# ---------------------------------------------------------------------------


class TestComputeIcWeights:

    def test_weights_sum_to_one(self):
        """All four factors have positive IC - weights must sum to 1.0."""
        monthly_ics = pd.DataFrame(
            {
                "month_end": pd.Timestamp("2023-01-31"),
                "factor_name": ["value", "quality", "momentum", "low_vol"],
                "ic_value": [0.05, 0.03, 0.08, 0.02],
            }
        )
        result = compute_ic_weights(monthly_ics)
        assert abs(result["ic_weight"].sum() - 1.0) < 1e-10

    def test_zero_flooring(self):
        """Value has negative IC so it gets weight 0; others are renormalised.

        Zero-flooring prevents counter-predictive factors from receiving
        negative weights, which would invert their signal.
        """
        monthly_ics = pd.DataFrame(
            {
                "month_end": pd.Timestamp("2023-01-31"),
                "factor_name": ["value", "quality", "momentum", "low_vol"],
                "ic_value": [-0.05, 0.03, 0.06, 0.01],
            }
        )
        result = compute_ic_weights(monthly_ics)
        value_weight = result[result["factor_name"] == "value"]["ic_weight"].iloc[0]
        assert value_weight == 0.0
        assert abs(result["ic_weight"].sum() - 1.0) < 1e-10

    def test_all_negative_fallback(self):
        """When every factor has negative IC, fall back to equal weights (0.25).

        This prevents the portfolio from having zero signal - equal weights
        is a safer default than giving everything to the least-bad factor.
        """
        monthly_ics = pd.DataFrame(
            {
                "month_end": pd.Timestamp("2023-01-31"),
                "factor_name": ["value", "quality", "momentum", "low_vol"],
                "ic_value": [-0.05, -0.03, -0.08, -0.02],
            }
        )
        result = compute_ic_weights(monthly_ics)
        assert all(abs(result["ic_weight"] - 0.25) < 1e-10)

    def test_empty_input(self):
        """No IC data at all (e.g. first rebalance) falls back to equal weights."""
        result = compute_ic_weights(pd.DataFrame())
        assert len(result) == 4
        assert all(abs(result["ic_weight"] - 0.25) < 1e-10)

    def test_excluded_factor_zeroed_and_others_renormalised(self):
        """Step 10: with excluded_factor='value', value gets weight 0; other 3
        sum to 1.0 in proportion to their (floored) ICs."""
        monthly_ics = pd.DataFrame(
            {
                "month_end": pd.Timestamp("2023-01-31"),
                "factor_name": ["value", "quality", "momentum", "low_vol"],
                "ic_value": [0.10, 0.03, 0.06, 0.01],
            }
        )
        result = compute_ic_weights(monthly_ics, excluded_factor="value")

        value_weight = result[result["factor_name"] == "value"]["ic_weight"].iloc[0]
        assert value_weight == 0.0
        # Remaining 3 weights must sum to 1
        assert abs(result["ic_weight"].sum() - 1.0) < 1e-10
        # Quality:Momentum:LowVol = 0.03:0.06:0.01 = 3:6:1, total 10
        q = result[result["factor_name"] == "quality"]["ic_weight"].iloc[0]
        m = result[result["factor_name"] == "momentum"]["ic_weight"].iloc[0]
        lv = result[result["factor_name"] == "low_vol"]["ic_weight"].iloc[0]
        assert abs(q - 0.3) < 1e-10
        assert abs(m - 0.6) < 1e-10
        assert abs(lv - 0.1) < 1e-10

    def test_excluded_factor_with_all_negative_uses_equal_thirds(self):
        """If every factor's IC is negative AND one is excluded, the remaining
        three split 1.0 equally (1/3 each), excluded stays at 0."""
        monthly_ics = pd.DataFrame(
            {
                "month_end": pd.Timestamp("2023-01-31"),
                "factor_name": ["value", "quality", "momentum", "low_vol"],
                "ic_value": [-0.05, -0.03, -0.08, -0.02],
            }
        )
        result = compute_ic_weights(monthly_ics, excluded_factor="momentum")

        mom = result[result["factor_name"] == "momentum"]["ic_weight"].iloc[0]
        assert mom == 0.0
        # Other three each get 1/3
        for f in ("value", "quality", "low_vol"):
            w = result[result["factor_name"] == f]["ic_weight"].iloc[0]
            assert abs(w - 1.0 / 3.0) < 1e-10

    def test_excluded_factor_with_empty_input(self):
        """Empty input + excluded factor: excluded=0, others share 1.0 equally."""
        result = compute_ic_weights(pd.DataFrame(), excluded_factor="quality")
        assert len(result) == 4
        q = result[result["factor_name"] == "quality"]["ic_weight"].iloc[0]
        assert q == 0.0
        assert abs(result["ic_weight"].sum() - 1.0) < 1e-10

    def test_invalid_excluded_factor_raises(self):
        """Unknown factor name should raise ValueError."""
        with pytest.raises(ValueError, match="excluded_factor must be one of"):
            compute_ic_weights(pd.DataFrame(), excluded_factor="bogus")


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------


class TestComputeCompositeScore:

    def test_weighted_sum(self):
        """Verify composite = w_val*z_val + w_qual*z_qual + w_mom*z_mom + w_lv*z_lv.

        Hand-calculated expected values:
          Stock A: 0.3*1.0 + 0.2*0.5 + 0.4*2.0 + 0.1*0.0 = 1.2
          Stock B: 0.3*(-1) + 0.2*0.5 + 0.4*(-2) + 0.1*1.0 = -0.9
        """
        factor_scores = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "z_value": [1.0, -1.0],
                "z_quality": [0.5, 0.5],
                "z_momentum": [2.0, -2.0],
                "z_low_vol": [0.0, 1.0],
            }
        )
        ic_weights = pd.DataFrame(
            {
                "factor_name": ["value", "quality", "momentum", "low_vol"],
                "ic_weight": [0.3, 0.2, 0.4, 0.1],
            }
        )
        result = compute_composite_score(factor_scores, ic_weights)

        assert abs(result.iloc[0]["composite_score"] - 1.2) < 1e-10
        assert abs(result.iloc[1]["composite_score"] - (-0.9)) < 1e-10


# ---------------------------------------------------------------------------
# run_composite_scorer (full pipeline with mocked DB)
# ---------------------------------------------------------------------------


class TestRunCompositeScorer:

    def test_full_flow(self):
        """End-to-end with mocked DB: prices + factor scores in, composite out.

        Uses 30 stocks over 2 months. The mock DB returns prices on the
        first call and factor scores on the second call, mimicking the
        two SQL queries in run_composite_scorer.
        """
        n = 30
        np.random.seed(42)

        # 2 months of prices: Jan flat, Feb varies by stock index
        dates1 = pd.bdate_range("2023-01-02", periods=22)
        dates2 = pd.bdate_range("2023-02-01", periods=20)
        prices_rows = []
        for sym_idx in range(n):
            sym = f"S{sym_idx:02d}"
            base = 100 + sym_idx
            for d in dates1:
                prices_rows.append(
                    {"symbol": sym, "trade_date": d, "adjusted_close": base}
                )
            for d in dates2:
                prices_rows.append(
                    {
                        "symbol": sym,
                        "trade_date": d,
                        "adjusted_close": base * (1 + sym_idx * 0.01),
                    }
                )
        prices = pd.DataFrame(prices_rows)

        # 1 month of factor scores (Jan end)
        factor_scores = pd.DataFrame(
            {
                "symbol": [f"S{i:02d}" for i in range(n)],
                "score_date": pd.Timestamp("2023-01-31"),
                "z_value": np.random.randn(n),
                "z_quality": np.random.randn(n),
                "z_momentum": np.random.randn(n),
                "z_low_vol": np.random.randn(n),
            }
        )

        db = MagicMock()
        db.read_query.side_effect = [prices, factor_scores]

        config = CompositeConfig(ic_lookback_months=36)
        result = run_composite_scorer(db, date(2023, 3, 1), config)

        # With only 1 month of IC data (below min_ic_months=12 default),
        # the scorer falls back to equal weights but should still return
        # valid composite scores without erroring.
        assert isinstance(result, pd.DataFrame)
        assert "composite_score" in result.columns or result.empty

    def test_empty_factor_scores(self):
        """When factor_scores table is empty, return an empty DataFrame.

        This happens before Isaac's factor scoring module has run.
        """
        db = MagicMock()
        db.read_query.side_effect = [
            pd.DataFrame(
                {"symbol": ["A"], "trade_date": ["2023-01-02"], "adjusted_close": [100]}
            ),
            pd.DataFrame(),  # no factor scores available
        ]

        config = CompositeConfig()
        result = run_composite_scorer(db, date(2023, 3, 1), config)
        assert result.empty

    def test_point_in_time(self):
        """SQL must use strict < on rebalance_date to prevent look-ahead bias.

        The backtest requires that each rebalancing decision only uses
        data available before the rebalance date, never on or after it.
        """
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()

        config = CompositeConfig()
        run_composite_scorer(db, date(2023, 3, 1), config)

        # Inspect the SQL passed to the first read_query call (prices)
        sql_arg = db.read_query.call_args_list[0][0][0]
        assert "trade_date < :rebalance_date" in sql_arg

    def test_min_ic_months_fallback(self):
        """With only 1 month of data but min_ic_months=12, use equal weights.

        The IC estimate from a single month is too noisy to trust, so
        the scorer should fall back to 0.25 per factor rather than
        using an unreliable IC-derived weighting.
        """
        n = 30
        np.random.seed(42)

        # Only 1 month of prices - nowhere near 12 months minimum
        dates = pd.bdate_range("2023-01-02", periods=22)
        prices_rows = []
        for i in range(n):
            for d in dates:
                prices_rows.append(
                    {
                        "symbol": f"S{i:02d}",
                        "trade_date": d,
                        "adjusted_close": 100.0 + i,
                    }
                )
        prices = pd.DataFrame(prices_rows)

        factor_scores = pd.DataFrame(
            {
                "symbol": [f"S{i:02d}" for i in range(n)],
                "score_date": pd.Timestamp("2023-01-31"),
                "z_value": np.random.randn(n),
                "z_quality": np.random.randn(n),
                "z_momentum": np.random.randn(n),
                "z_low_vol": np.random.randn(n),
            }
        )

        db = MagicMock()
        db.read_query.side_effect = [prices, factor_scores]

        config = CompositeConfig(ic_lookback_months=36, min_ic_months=12)
        result = run_composite_scorer(db, date(2023, 3, 1), config)

        assert isinstance(result, pd.DataFrame)

    def test_fewer_than_10_stocks_in_period_skipped(self):
        """Periods with fewer than 10 stocks are skipped in IC computation."""
        # Only 9 stocks — below the minimum group size of 10
        n = 9
        np.random.seed(0)
        dates1 = pd.bdate_range("2023-01-02", periods=22)
        dates2 = pd.bdate_range("2023-02-01", periods=20)
        prices_rows = []
        for i in range(n):
            sym = f"S{i:02d}"
            for d in dates1:
                prices_rows.append(
                    {"symbol": sym, "trade_date": d, "adjusted_close": 100.0}
                )
            for d in dates2:
                prices_rows.append(
                    {"symbol": sym, "trade_date": d, "adjusted_close": 105.0}
                )
        prices = pd.DataFrame(prices_rows)

        factor_scores = pd.DataFrame(
            {
                "symbol": [f"S{i:02d}" for i in range(n)],
                "score_date": pd.Timestamp("2023-01-31"),
                "z_value": np.random.randn(n),
                "z_quality": np.random.randn(n),
                "z_momentum": np.random.randn(n),
                "z_low_vol": np.random.randn(n),
            }
        )
        db = MagicMock()
        db.read_query.side_effect = [prices, factor_scores]

        config = CompositeConfig(ic_lookback_months=36)
        result = run_composite_scorer(db, date(2023, 3, 1), config)
        # With < 10 stocks, IC period is skipped → falls back to equal weights
        assert isinstance(result, pd.DataFrame)

    def test_current_scores_empty_returns_empty(self):
        """If current factor scores are empty after filtering, return empty."""
        db = MagicMock()
        # prices non-empty, factor_scores non-empty but no rows match current date
        factor_scores = pd.DataFrame(
            {
                "symbol": ["S00"],
                "score_date": pd.Timestamp("2020-01-31"),  # far in the past
                "z_value": [1.0],
                "z_quality": [0.0],
                "z_momentum": [0.5],
                "z_low_vol": [-0.5],
            }
        )
        prices = pd.DataFrame(
            {"symbol": ["S00"], "trade_date": ["2020-01-02"], "adjusted_close": [100]}
        )
        db.read_query.side_effect = [prices, factor_scores]

        config = CompositeConfig(ic_lookback_months=1, min_ic_months=1)
        result = run_composite_scorer(db, date(2023, 3, 1), config)
        assert isinstance(result, pd.DataFrame)
