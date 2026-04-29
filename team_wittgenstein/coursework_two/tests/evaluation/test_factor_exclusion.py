"""Tests for Step 10 factor exclusion robustness analysis."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.backtest.backtest_engine import BacktestConfig
from modules.composite.composite_scorer import CompositeConfig
from modules.evaluation.factor_exclusion import (
    EXCLUDED_FACTORS,
    FactorExclusionConfig,
    _build_one_rebalance,
    run_factor_exclusion,
)
from modules.portfolio.ewma_volatility import EWMAConfig
from modules.portfolio.position_builder import PositionConfig
from modules.portfolio.stock_selector import SelectionConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def factor_config():
    """A complete config bundle for factor exclusion runs."""
    return FactorExclusionConfig(
        composite=CompositeConfig(),
        selection=SelectionConfig(),
        ewma=EWMAConfig(),
        position=PositionConfig(),
        backtest=BacktestConfig(cost_bps=25.0, borrow_rate=0.0075),
    )


@pytest.fixture
def composite_df():
    """Mock output of run_composite_scorer."""
    return pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(20)],
            "composite_score": [float(i) for i in range(20)],
        }
    )


@pytest.fixture
def selected_df():
    """Mock output of run_stock_selection."""
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "sector": ["Tech", "Tech", "Energy"],
            "direction": ["long", "long", "short"],
            "composite_score": [2.0, 1.8, -1.5],
            "percentile_rank": [0.95, 0.92, 0.05],
            "status": ["long_core", "long_core", "short_core"],
            "buffer_months_count": [0, 0, 0],
        }
    )


@pytest.fixture
def positions_df():
    """Mock output of build_portfolio_positions."""
    return pd.DataFrame(
        {
            "rebalance_date": [date(2024, 1, 31)] * 3,
            "symbol": ["A", "B", "C"],
            "sector": ["Tech", "Tech", "Energy"],
            "direction": ["long", "long", "short"],
            "ewma_vol": [0.20, 0.25, 0.18],
            "risk_adj_score": [10.0, 7.2, 8.3],
            "target_weight": [0.7, 0.6, 0.3],
            "final_weight": [0.7, 0.6, 0.3],
            "liquidity_capped": [False, False, False],
            "trade_action": ["trade"] * 3,
        }
    )


# ---------------------------------------------------------------------------
# EXCLUDED_FACTORS module constant
# ---------------------------------------------------------------------------


class TestExcludedFactorsConstant:

    def test_four_factors(self):
        assert set(EXCLUDED_FACTORS) == {
            "value",
            "quality",
            "momentum",
            "low_vol",
        }


# ---------------------------------------------------------------------------
# _build_one_rebalance
# ---------------------------------------------------------------------------


class TestBuildOneRebalance:

    def test_passes_excluded_factor_to_composite_scorer(
        self, factor_config, composite_df, selected_df, positions_df
    ):
        """The excluded_factor argument is forwarded to run_composite_scorer."""
        db = MagicMock()
        with (
            patch(
                "modules.evaluation.factor_exclusion.run_composite_scorer",
                return_value=composite_df,
            ) as mock_comp,
            patch(
                "modules.evaluation.factor_exclusion.run_stock_selection",
                return_value=selected_df,
            ),
            patch(
                "modules.evaluation.factor_exclusion.run_ewma_volatility",
                return_value=pd.DataFrame(
                    {"symbol": ["A", "B", "C"], "ewma_vol": [0.2, 0.25, 0.18]}
                ),
            ),
            patch(
                "modules.evaluation.factor_exclusion.compute_risk_adjusted_scores",
                return_value=selected_df,
            ),
            patch(
                "modules.evaluation.factor_exclusion.build_portfolio_positions",
                return_value=positions_df,
            ),
        ):
            _build_one_rebalance(
                db,
                date(2024, 1, 31),
                {"A": "Tech", "B": "Tech", "C": "Energy"},
                "value",
                factor_config,
                pd.DataFrame(),
                pd.DataFrame(),
            )

        # Verify excluded_factor and persist=False were passed
        _, kwargs = mock_comp.call_args
        assert kwargs["excluded_factor"] == "value"
        assert kwargs["persist"] is False

    def test_passes_composite_and_prior_to_stock_selection(
        self, factor_config, composite_df, selected_df, positions_df
    ):
        """Stock selector receives composite_scores override and prior_selection."""
        db = MagicMock()
        prior_sel = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": ["long_core"],
                "buffer_months_count": [0],
                "entry_date": [date(2023, 12, 31)],
            }
        )
        with (
            patch(
                "modules.evaluation.factor_exclusion.run_composite_scorer",
                return_value=composite_df,
            ),
            patch(
                "modules.evaluation.factor_exclusion.run_stock_selection",
                return_value=selected_df,
            ) as mock_sel,
            patch(
                "modules.evaluation.factor_exclusion.run_ewma_volatility",
                return_value=pd.DataFrame(
                    {"symbol": ["A", "B", "C"], "ewma_vol": [0.2, 0.25, 0.18]}
                ),
            ),
            patch(
                "modules.evaluation.factor_exclusion.compute_risk_adjusted_scores",
                return_value=selected_df,
            ),
            patch(
                "modules.evaluation.factor_exclusion.build_portfolio_positions",
                return_value=positions_df,
            ),
        ):
            _build_one_rebalance(
                db,
                date(2024, 1, 31),
                {"A": "Tech", "B": "Tech", "C": "Energy"},
                "quality",
                factor_config,
                prior_sel,
                pd.DataFrame(),
            )

        _, kwargs = mock_sel.call_args
        # composite_scores should be the composite df we mocked
        pd.testing.assert_frame_equal(kwargs["composite_scores"], composite_df)
        pd.testing.assert_frame_equal(kwargs["prior_selection"], prior_sel)
        assert kwargs["persist"] is False

    def test_empty_composite_short_circuits(self, factor_config):
        """If composite is empty, returns empty DataFrames immediately."""
        db = MagicMock()
        with patch(
            "modules.evaluation.factor_exclusion.run_composite_scorer",
            return_value=pd.DataFrame(),
        ):
            positions, sel = _build_one_rebalance(
                db,
                date(2024, 1, 31),
                {},
                "value",
                factor_config,
                pd.DataFrame(),
                pd.DataFrame(),
            )
        assert positions.empty
        assert sel.empty

    def test_empty_selected_short_circuits(self, factor_config, composite_df):
        """If selection is empty, no further pipeline steps run."""
        db = MagicMock()
        with (
            patch(
                "modules.evaluation.factor_exclusion.run_composite_scorer",
                return_value=composite_df,
            ),
            patch(
                "modules.evaluation.factor_exclusion.run_stock_selection",
                return_value=pd.DataFrame(),
            ),
        ):
            positions, sel = _build_one_rebalance(
                db,
                date(2024, 1, 31),
                {},
                "value",
                factor_config,
                pd.DataFrame(),
                pd.DataFrame(),
            )
        assert positions.empty
        assert sel.empty

    def test_empty_scored_short_circuits(self, factor_config, composite_df):
        """If risk-adjusted scoring returns empty, return empty DataFrames."""
        db = MagicMock()
        selected = pd.DataFrame(
            {
                "symbol": ["A"],
                "sector": ["IT"],
                "direction": ["long"],
                "composite_score": [1.0],
                "status": ["long_core"],
            }
        )
        with (
            patch(
                "modules.evaluation.factor_exclusion.run_composite_scorer",
                return_value=composite_df,
            ),
            patch(
                "modules.evaluation.factor_exclusion.run_stock_selection",
                return_value=selected,
            ),
            patch(
                "modules.evaluation.factor_exclusion.run_ewma_volatility",
                return_value=pd.DataFrame(),
            ),
            patch(
                "modules.evaluation.factor_exclusion.compute_risk_adjusted_scores",
                return_value=pd.DataFrame(),
            ),
        ):
            positions, sel = _build_one_rebalance(
                db,
                date(2024, 1, 31),
                {},
                "value",
                factor_config,
                pd.DataFrame(),
                pd.DataFrame(),
            )
        assert positions.empty
        assert sel.empty


# ---------------------------------------------------------------------------
# run_factor_exclusion (orchestrator)
# ---------------------------------------------------------------------------


class TestRunFactorExclusion:

    def test_empty_portfolio_positions_raises(self, factor_config):
        """If baseline portfolio_positions is empty, raises informative error."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="portfolio_positions is empty"):
            run_factor_exclusion(db, {}, factor_config)

    def test_creates_four_scenarios(self, factor_config, positions_df):
        """End-to-end: 4 factors -> 4 scenario_ids in the result list."""
        db = MagicMock()
        # First read_query: rebalance dates from portfolio_positions
        # Subsequent reads inside run_backtest + compute_summary_metrics.
        rebalance_dates_df = pd.DataFrame(
            {
                "rebalance_date": [
                    date(2024, 1, 31),
                    date(2024, 2, 29),
                    date(2024, 3, 31),
                ]
            }
        )
        db.read_query.return_value = rebalance_dates_df

        with (
            patch(
                "modules.evaluation.factor_exclusion._build_one_rebalance",
                return_value=(positions_df, pd.DataFrame()),
            ),
            patch(
                "modules.evaluation.factor_exclusion.run_backtest",
                return_value=pd.DataFrame(
                    {
                        "scenario_id": ["x"],
                        "rebalance_date": [date(2024, 2, 29)],
                        "gross_return": [0.01],
                        "net_return": [0.008],
                        "long_return": [0.012],
                        "short_return": [0.004],
                        "benchmark_return": [0.005],
                        "excess_return": [0.003],
                        "cumulative_return": [0.008],
                        "turnover": [0.3],
                        "transaction_cost": [0.002],
                    }
                ),
            ),
            patch(
                "modules.evaluation.factor_exclusion.compute_summary_metrics",
                return_value={
                    "scenario_id": "x",
                    "annualised_return": 0.10,
                    "sharpe_ratio": 0.5,
                    "max_drawdown": -0.15,
                },
            ),
            patch("modules.evaluation.factor_exclusion.DataWriter") as mock_writer_cls,
        ):
            mock_writer_cls.return_value = MagicMock()
            result = run_factor_exclusion(db, {}, factor_config)

        # 4 factors -> 4 scenario_ids returned
        assert len(result) == 4
        assert set(result) == {
            "excl_value",
            "excl_quality",
            "excl_momentum",
            "excl_low_vol",
        }

    def test_skips_scenario_when_no_positions(self, factor_config):
        """If every rebalance date returns empty positions, scenario is skipped."""
        db = MagicMock()
        rebalance_dates_df = pd.DataFrame({"rebalance_date": [date(2024, 1, 31)]})
        db.read_query.return_value = rebalance_dates_df

        with (
            patch(
                "modules.evaluation.factor_exclusion._build_one_rebalance",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("modules.evaluation.factor_exclusion.run_backtest") as mock_backtest,
            patch("modules.evaluation.factor_exclusion.DataWriter"),
        ):
            result = run_factor_exclusion(db, {}, factor_config)

        # No scenarios produced - all 4 factors yielded empty positions
        assert result == []
        # run_backtest should never have been called
        mock_backtest.assert_not_called()
