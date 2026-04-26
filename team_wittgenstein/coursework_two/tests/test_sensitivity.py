"""Tests for Step 8 parameter sensitivity analysis."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.backtest.backtest_engine import BacktestConfig
from modules.composite.composite_scorer import CompositeConfig
from modules.evaluation.factor_exclusion import FactorExclusionConfig
from modules.evaluation.sensitivity import (
    SENSITIVITY_VARIANTS,
    _override_config,
    run_parameter_sensitivity,
)
from modules.portfolio.ewma_volatility import EWMAConfig
from modules.portfolio.position_builder import PositionConfig
from modules.portfolio.stock_selector import SelectionConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """Baseline scenario config (the default values for each parameter)."""
    return FactorExclusionConfig(
        composite=CompositeConfig(ic_lookback_months=36, min_ic_months=12),
        selection=SelectionConfig(
            selection_threshold=0.10,
            buffer_exit_threshold=0.20,
            buffer_max_months=3,
        ),
        ewma=EWMAConfig(ewma_lambda=0.94),
        position=PositionConfig(no_trade_threshold=0.01),
        backtest=BacktestConfig(cost_bps=25.0, borrow_rate=0.0075),
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
# SENSITIVITY_VARIANTS module constant
# ---------------------------------------------------------------------------


class TestSensitivityVariants:

    def test_fifteen_total_scenarios(self):
        """5 parameters x 3 non-baseline values each = 15 variants."""
        assert len(SENSITIVITY_VARIANTS) == 15

    def test_unique_scenario_ids(self):
        """Each variant must have a unique scenario_id."""
        ids = [v[0] for v in SENSITIVITY_VARIANTS]
        assert len(ids) == len(set(ids))

    def test_three_variants_per_parameter(self):
        """Each of the 5 parameter groups has exactly 3 entries."""
        groups = {}
        for scenario_id, _, field, _ in SENSITIVITY_VARIANTS:
            groups.setdefault(field, []).append(scenario_id)
        assert len(groups) == 5
        for field, ids in groups.items():
            assert len(ids) == 3, f"Expected 3 variants for {field}, got {len(ids)}"


# ---------------------------------------------------------------------------
# _override_config
# ---------------------------------------------------------------------------


class TestOverrideConfig:

    def test_overrides_one_field(self, base_config):
        """Overriding selection.selection_threshold returns a new config with that
        single field changed."""
        new_config = _override_config(
            base_config, "selection", "selection_threshold", 0.05
        )
        assert new_config.selection.selection_threshold == 0.05
        # Other selection fields unchanged
        assert (
            new_config.selection.buffer_exit_threshold
            == base_config.selection.buffer_exit_threshold
        )
        # Other config sections unchanged
        assert new_config.composite is base_config.composite
        assert new_config.ewma is base_config.ewma

    def test_returns_new_instance_immutable(self, base_config):
        """Original config is not mutated."""
        new_config = _override_config(base_config, "ewma", "ewma_lambda", 0.92)
        assert base_config.ewma.ewma_lambda == 0.94  # unchanged
        assert new_config.ewma.ewma_lambda == 0.92


# ---------------------------------------------------------------------------
# run_parameter_sensitivity (orchestrator)
# ---------------------------------------------------------------------------


class TestRunParameterSensitivity:

    def test_empty_portfolio_positions_raises(self, base_config):
        """If baseline portfolio_positions is empty, raise informative error."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="portfolio_positions is empty"):
            run_parameter_sensitivity(db, {}, base_config)

    def test_creates_fifteen_scenarios(self, base_config, positions_df):
        """End-to-end: 15 variants -> 15 scenario_ids returned."""
        db = MagicMock()
        rebalance_dates_df = pd.DataFrame(
            {
                "rebalance_date": [
                    date(2024, 1, 31),
                    date(2024, 2, 29),
                ]
            }
        )
        db.read_query.return_value = rebalance_dates_df

        with (
            patch(
                "modules.evaluation.sensitivity._build_one_rebalance",
                return_value=(positions_df, pd.DataFrame()),
            ),
            patch(
                "modules.evaluation.sensitivity.run_backtest",
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
                "modules.evaluation.sensitivity.compute_summary_metrics",
                return_value={
                    "scenario_id": "x",
                    "annualised_return": 0.10,
                    "sharpe_ratio": 0.5,
                    "max_drawdown": -0.15,
                },
            ),
            patch("modules.evaluation.sensitivity.DataWriter") as mock_writer_cls,
        ):
            mock_writer_cls.return_value = MagicMock()
            result = run_parameter_sensitivity(db, {}, base_config, skip_existing=False)

        assert len(result) == 15
        assert set(result) == set(v[0] for v in SENSITIVITY_VARIANTS)

    def test_passes_no_excluded_factor_to_pipeline(self, base_config, positions_df):
        """Sensitivity uses excluded_factor=None (different from Step 10)."""
        db = MagicMock()
        rebalance_dates_df = pd.DataFrame({"rebalance_date": [date(2024, 1, 31)]})
        db.read_query.return_value = rebalance_dates_df

        with (
            patch(
                "modules.evaluation.sensitivity._build_one_rebalance",
                return_value=(positions_df, pd.DataFrame()),
            ) as mock_build,
            patch(
                "modules.evaluation.sensitivity.run_backtest",
                return_value=pd.DataFrame(),
            ),
            patch(
                "modules.evaluation.sensitivity.compute_summary_metrics",
                return_value={
                    "annualised_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                },
            ),
            patch("modules.evaluation.sensitivity.DataWriter"),
        ):
            run_parameter_sensitivity(db, {}, base_config, skip_existing=False)

        # Every call should pass excluded_factor=None (4th positional arg)
        for call in mock_build.call_args_list:
            assert call.args[3] is None

    def test_skips_scenarios_with_no_positions(self, base_config):
        """If a variant builds no positions, it's skipped."""
        db = MagicMock()
        rebalance_dates_df = pd.DataFrame({"rebalance_date": [date(2024, 1, 31)]})
        db.read_query.return_value = rebalance_dates_df

        with (
            patch(
                "modules.evaluation.sensitivity._build_one_rebalance",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("modules.evaluation.sensitivity.run_backtest") as mock_backtest,
            patch("modules.evaluation.sensitivity.DataWriter"),
        ):
            result = run_parameter_sensitivity(db, {}, base_config, skip_existing=False)

        assert result == []
        mock_backtest.assert_not_called()

    def test_overridden_config_passed_to_pipeline(self, base_config, positions_df):
        """The first variant's config should reflect its parameter override."""
        db = MagicMock()
        rebalance_dates_df = pd.DataFrame({"rebalance_date": [date(2024, 1, 31)]})
        db.read_query.return_value = rebalance_dates_df

        with (
            patch(
                "modules.evaluation.sensitivity._build_one_rebalance",
                return_value=(positions_df, pd.DataFrame()),
            ) as mock_build,
            patch(
                "modules.evaluation.sensitivity.run_backtest",
                return_value=pd.DataFrame(),
            ),
            patch(
                "modules.evaluation.sensitivity.compute_summary_metrics",
                return_value={
                    "annualised_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                },
            ),
            patch("modules.evaluation.sensitivity.DataWriter"),
        ):
            run_parameter_sensitivity(db, {}, base_config, skip_existing=False)

        # First variant is "sens_sel_0.05" - override selection_threshold to 0.05
        first_call = mock_build.call_args_list[0]
        config_passed = first_call.args[4]  # 5th positional arg
        assert config_passed.selection.selection_threshold == 0.05
