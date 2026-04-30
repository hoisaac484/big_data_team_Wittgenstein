"""Step 8: Parameter sensitivity analysis.

For each of five strategy parameters, runs the full pipeline three times
(excluding the baseline value) while holding all other parameters fixed.
Produces 15 new scenarios in ``backtest_returns`` and ``backtest_summary``.

Implementation reuses the in-memory pipeline helper from
``factor_exclusion._build_one_rebalance`` with ``excluded_factor=None``. This
keeps the baseline database tables intact while letting each variant override
its own no-trade threshold, buffer threshold, IC lookback, EWMA decay, or
selection threshold.

Parameter grid summary, excluding the existing ``baseline`` scenario:
- ``selection_threshold``: 0.05, 0.15, 0.20
- ``ic_lookback_months``: 24, 48, 60
- ``ewma_lambda``: 0.90, 0.92, 0.97
- ``no_trade_threshold``: 0.005, 0.015, 0.020
- ``buffer_exit_threshold``: 0.10, 0.15, 0.25
"""

import logging
from dataclasses import replace

import pandas as pd

from modules.backtest.backtest_engine import run_backtest
from modules.db.db_connection import PostgresConnection
from modules.evaluation.factor_exclusion import (
    FactorExclusionConfig,
    _build_one_rebalance,
)
from modules.evaluation.metrics import compute_summary_metrics
from modules.output.data_writer import DataWriter

logger = logging.getLogger(__name__)
SCHEMA = "team_wittgenstein"


# 15 variants: (scenario_id, config_attr, field_name, value)
# The baseline value of each parameter is excluded - the 'baseline' scenario
# already covers it. "no buffer" is encoded as buffer_exit_threshold equal to
# selection_threshold, which makes the buffer zone empty.
SENSITIVITY_VARIANTS = (
    # selection_threshold
    ("sens_sel_0.05", "selection", "selection_threshold", 0.05),
    ("sens_sel_0.15", "selection", "selection_threshold", 0.15),
    ("sens_sel_0.20", "selection", "selection_threshold", 0.20),
    # ic_lookback_months
    ("sens_ic_24", "composite", "ic_lookback_months", 24),
    ("sens_ic_48", "composite", "ic_lookback_months", 48),
    ("sens_ic_60", "composite", "ic_lookback_months", 60),
    # ewma_lambda
    ("sens_ewma_0.90", "ewma", "ewma_lambda", 0.90),
    ("sens_ewma_0.92", "ewma", "ewma_lambda", 0.92),
    ("sens_ewma_0.97", "ewma", "ewma_lambda", 0.97),
    # no_trade_threshold
    ("sens_notrade_0.005", "position", "no_trade_threshold", 0.005),
    ("sens_notrade_0.015", "position", "no_trade_threshold", 0.015),
    ("sens_notrade_0.020", "position", "no_trade_threshold", 0.020),
    # buffer_exit_threshold ("no_buffer" = same value as selection_threshold)
    ("sens_buffer_none", "selection", "buffer_exit_threshold", 0.10),
    ("sens_buffer_0.15", "selection", "buffer_exit_threshold", 0.15),
    ("sens_buffer_0.25", "selection", "buffer_exit_threshold", 0.25),
)


def _override_config(
    base: FactorExclusionConfig, attr: str, field: str, value
) -> FactorExclusionConfig:
    """Return a new config with one nested field overridden.

    Uses dataclasses.replace which works on frozen dataclasses.
    """
    sub_config = getattr(base, attr)
    new_sub = replace(sub_config, **{field: value})
    return replace(base, **{attr: new_sub})


def run_parameter_sensitivity(
    db: PostgresConnection,
    sector_map: dict,
    base_config: FactorExclusionConfig,
    skip_existing: bool = True,
) -> list[str]:
    """Run all 15 parameter sensitivity scenarios in-memory.

    For each (parameter, value) pair:
      1. Override the corresponding config field
      2. Re-run the full in-memory pipeline across all rebalance dates
      3. Run the backtest with positions_override
      4. Persist backtest_returns + backtest_summary under scenario_id
      5. compute_summary_metrics

    Args:
        db:            Active PostgresConnection.
        sector_map:    symbol -> GICS sector mapping.
        base_config:   Baseline configs to override per variant.
        skip_existing: When True (default), skip scenario_ids that already
                       have rows in backtest_summary. Useful for resuming
                       after a crash without redoing completed scenarios.

    Returns:
        List of scenario_ids created (or already present when skipping).
    """
    # Discover rebalance dates from baseline portfolio_positions
    dates_df = db.read_query("""
        SELECT DISTINCT rebalance_date
        FROM team_wittgenstein.portfolio_positions
        ORDER BY rebalance_date
        """)
    if dates_df.empty:
        raise RuntimeError("portfolio_positions is empty - run baseline backfill first")
    rebalance_dates = pd.to_datetime(dates_df["rebalance_date"]).dt.date.tolist()

    # Find scenarios already completed so we can skip them on resume
    existing = set()
    if skip_existing:
        existing_df = db.read_query(
            "SELECT scenario_id FROM team_wittgenstein.backtest_summary "
            "WHERE scenario_id LIKE 'sens_%'"
        )
        existing = (
            set(existing_df["scenario_id"].tolist()) if not existing_df.empty else set()
        )
        if existing:
            logger.info(
                "Skipping %d already-completed scenarios: %s",
                len(existing),
                sorted(existing),
            )

    writer = DataWriter(db)
    created = []

    for scenario_id, config_attr, field_name, value in SENSITIVITY_VARIANTS:
        if scenario_id in existing:
            logger.info("Skipping %s (already in backtest_summary)", scenario_id)
            continue
        logger.info(
            "===== Parameter sensitivity: %s (%s.%s = %s) =====",
            scenario_id,
            config_attr,
            field_name,
            value,
        )

        # Override the chosen config field
        scenario_config = _override_config(base_config, config_attr, field_name, value)

        # Run in-memory pipeline across all rebalance dates
        all_positions = []
        prior_selection = pd.DataFrame()
        prior_positions = pd.DataFrame()

        for i, rd in enumerate(rebalance_dates, 1):
            logger.info(
                "[%s] [%d/%d] %s",
                scenario_id,
                i,
                len(rebalance_dates),
                rd,
            )
            positions, full_selection = _build_one_rebalance(
                db,
                rd,
                sector_map,
                None,  # no factor exclusion - this is parameter sensitivity
                scenario_config,
                prior_selection=prior_selection,
                prior_positions=prior_positions,
            )
            if not positions.empty:
                all_positions.append(positions)
                prior_positions = positions[
                    ["symbol", "direction", "final_weight"]
                ].copy()
                prior_selection = full_selection

        if not all_positions:
            logger.warning("No positions built for %s - skipping backtest", scenario_id)
            continue

        all_positions_df = pd.concat(all_positions, ignore_index=True)

        # Run backtest with our positions override
        scenario_backtest_config = replace(
            scenario_config.backtest, scenario_id=scenario_id
        )
        backtest_df = run_backtest(
            db, scenario_backtest_config, positions_override=all_positions_df
        )

        # Wipe any prior rows for this scenario before writing
        db.execute(
            "DELETE FROM team_wittgenstein.backtest_returns "
            "WHERE scenario_id = :scenario_id",
            {"scenario_id": scenario_id},
        )
        writer.write_backtest_returns(backtest_df, scenario_id)

        summary = compute_summary_metrics(db, scenario_id)
        created.append(scenario_id)

        logger.info(
            "%s | ann_ret=%.4f sharpe=%.4f max_dd=%.4f",
            scenario_id,
            summary["annualised_return"],
            summary["sharpe_ratio"],
            summary["max_drawdown"],
        )

    logger.info("Parameter sensitivity complete: %d scenarios created", len(created))
    return created
