"""Tests for stock selection with buffer zone logic."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from modules.portfolio.stock_selector import (
    LONG_BUFFER,
    LONG_CORE,
    NOT_SELECTED,
    SHORT_BUFFER,
    SHORT_CORE,
    SelectionConfig,
    apply_selection_rules,
    compute_percentile_ranks,
    run_stock_selection,
)

# ---------------------------------------------------------------------------
# compute_percentile_ranks
# ---------------------------------------------------------------------------


class TestComputePercentileRanks:

    def test_ranks_within_sector(self):
        """Stocks are ranked within their sector, not across all stocks."""
        scores = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "composite_score": [1.0, 2.0, 3.0, 4.0],
            }
        )
        # All in the same sector
        sector_map = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Tech"}
        result = compute_percentile_ranks(scores, sector_map)

        # D has the highest score, should have the highest rank
        d_rank = result[result["symbol"] == "D"]["percentile_rank"].iloc[0]
        a_rank = result[result["symbol"] == "A"]["percentile_rank"].iloc[0]
        assert d_rank > a_rank

    def test_missing_sector_dropped(self):
        """Stocks without a sector mapping are excluded."""
        scores = pd.DataFrame(
            {
                "symbol": ["A", "B", "UNKNOWN"],
                "composite_score": [1.0, 2.0, 3.0],
            }
        )
        sector_map = {"A": "Tech", "B": "Tech"}
        result = compute_percentile_ranks(scores, sector_map)
        assert "UNKNOWN" not in result["symbol"].values


# ---------------------------------------------------------------------------
# apply_selection_rules
# ---------------------------------------------------------------------------


class TestApplySelectionRules:

    def _make_ranked(self, data):
        """Helper to build a ranked DataFrame."""
        return pd.DataFrame(data)

    def test_top_10_percent_enters_long(self):
        """Stock in top 10% gets long_core status."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.95],
                "sector": ["Tech"],
                "composite_score": [2.0],
            }
        )
        previous = pd.DataFrame(
            columns=["symbol", "status", "buffer_months_count", "entry_date"]
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == LONG_CORE
        assert result.iloc[0]["buffer_months_count"] == 0

    def test_bottom_10_percent_enters_short(self):
        """Stock in bottom 10% gets short_core status."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.05],
                "sector": ["Tech"],
                "composite_score": [-2.0],
            }
        )
        previous = pd.DataFrame(
            columns=["symbol", "status", "buffer_months_count", "entry_date"]
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == SHORT_CORE

    def test_buffer_zone_holds_existing_long(self):
        """Stock that was long_core and drifts to 85th pctile gets long_buffer."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.85],
                "sector": ["Tech"],
                "composite_score": [1.5],
            }
        )
        previous = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": [LONG_CORE],
                "buffer_months_count": [0],
                "entry_date": [date(2023, 12, 31)],
            }
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == LONG_BUFFER
        assert result.iloc[0]["buffer_months_count"] == 1

    def test_buffer_timer_exit_after_3_months(self):
        """Stock in buffer for 3 months gets timer_exit."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.85],
                "sector": ["Tech"],
                "composite_score": [1.2],
            }
        )
        # Already been in buffer for 2 months, this would be the 3rd
        previous = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": [LONG_BUFFER],
                "buffer_months_count": [2],
                "entry_date": [date(2023, 10, 31)],
            }
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == NOT_SELECTED
        assert result.iloc[0]["exit_reason"] == "timer_exit"

    def test_recovery_resets_buffer(self):
        """Stock that was long_buffer and recovers to top 10% resets to long_core."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.95],
                "sector": ["Tech"],
                "composite_score": [2.5],
            }
        )
        previous = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": [LONG_BUFFER],
                "buffer_months_count": [2],
                "entry_date": [date(2023, 10, 31)],
            }
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == LONG_CORE
        assert result.iloc[0]["buffer_months_count"] == 0
        # Entry date should be preserved from original entry
        assert result.iloc[0]["entry_date"] == date(2023, 10, 31)

    def test_hard_stop_below_buffer_zone(self):
        """Long stock that drops below 80th pctile gets hard_stop."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.70],
                "sector": ["Tech"],
                "composite_score": [0.5],
            }
        )
        previous = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": [LONG_CORE],
                "buffer_months_count": [0],
                "entry_date": [date(2023, 12, 31)],
            }
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == NOT_SELECTED
        assert result.iloc[0]["exit_reason"] == "hard_stop"

    def test_new_stock_in_buffer_zone_not_selected(self):
        """A stock in 80-90th pctile that was NOT previously held is not selected."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.85],
                "sector": ["Tech"],
                "composite_score": [1.5],
            }
        )
        previous = pd.DataFrame(
            columns=["symbol", "status", "buffer_months_count", "entry_date"]
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == NOT_SELECTED

    def test_short_hard_stop_sets_exit_reason(self):
        """Short that moves outside buffer zone gets hard_stop exit reason."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.50],  # mid-range, outside short buffer
                "sector": ["Tech"],
                "composite_score": [0.0],
            }
        )
        previous = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": [SHORT_CORE],
                "buffer_months_count": [0],
                "entry_date": [date(2023, 12, 31)],
            }
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        row = result.iloc[0]
        assert row["status"] == NOT_SELECTED
        assert row["exit_reason"] == "hard_stop"

    def test_short_buffer_timer_exit(self):
        """Short in buffer for buffer_max_months gets timer_exit."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.15],
                "sector": ["Tech"],
                "composite_score": [-1.0],
            }
        )
        previous = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": [SHORT_BUFFER],
                "buffer_months_count": [2],
                "entry_date": [date(2023, 10, 31)],
            }
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == NOT_SELECTED
        assert result.iloc[0]["exit_reason"] == "timer_exit"

    def test_short_buffer_holds_existing(self):
        """Stock that was short_core and drifts to 15th pctile gets short_buffer."""
        ranked = self._make_ranked(
            {
                "symbol": ["A"],
                "percentile_rank": [0.15],
                "sector": ["Tech"],
                "composite_score": [-1.5],
            }
        )
        previous = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": [SHORT_CORE],
                "buffer_months_count": [0],
                "entry_date": [date(2023, 12, 31)],
            }
        )
        config = SelectionConfig()
        result = apply_selection_rules(ranked, previous, date(2024, 1, 31), config)
        assert result.iloc[0]["status"] == SHORT_BUFFER
        assert result.iloc[0]["buffer_months_count"] == 1


# ---------------------------------------------------------------------------
# run_stock_selection (orchestrator with mocked DB)
# ---------------------------------------------------------------------------


class TestRunStockSelection:

    def test_returns_selected_with_direction(self):
        """Mocked DB: verify selected stocks have direction column."""
        # 10 stocks in one sector, scores 1-10
        scores = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(10)],
                "score_date": date(2024, 1, 31),
                "composite_score": list(range(1, 11)),
            }
        )
        prev_selection = pd.DataFrame(
            columns=["symbol", "status", "buffer_months_count", "entry_date"]
        )

        db = MagicMock()
        db.read_query.side_effect = [scores, prev_selection]

        sector_map = {f"S{i}": "Tech" for i in range(10)}
        config = SelectionConfig()

        result = run_stock_selection(db, date(2024, 2, 28), sector_map, config)

        assert "direction" in result.columns
        assert set(result["direction"].unique()).issubset({"long", "short"})

    def test_empty_scores_returns_empty(self):
        """No composite scores returns empty DataFrame."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()

        result = run_stock_selection(db, date(2024, 2, 28), {}, SelectionConfig())
        assert result.empty

    def test_previous_selection_empty_db_returns_empty_frame(self):
        """fetch_previous_selection returns empty DataFrame when DB has no rows."""
        from modules.portfolio.stock_selector import fetch_previous_selection

        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()
        result = fetch_previous_selection(db, date(2024, 1, 31))
        assert result.empty
        assert list(result.columns) == [
            "symbol",
            "status",
            "buffer_months_count",
            "entry_date",
        ]

    def test_previous_selection_non_empty_returned_as_is(self):
        """fetch_previous_selection returns the DB result unchanged when non-empty."""
        from modules.portfolio.stock_selector import fetch_previous_selection

        db = MagicMock()
        db.read_query.return_value = pd.DataFrame(
            {
                "symbol": ["A"],
                "status": [LONG_CORE],
                "buffer_months_count": [0],
                "entry_date": [date(2023, 12, 31)],
            }
        )
        result = fetch_previous_selection(db, date(2024, 1, 31))
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "A"

    def test_persist_selection_status_empty_is_noop(self):
        """persist_selection_status does nothing when selection is empty."""
        from modules.portfolio.stock_selector import persist_selection_status

        db = MagicMock()
        persist_selection_status(db, pd.DataFrame())
        db.write_dataframe_on_conflict_do_nothing.assert_not_called()
