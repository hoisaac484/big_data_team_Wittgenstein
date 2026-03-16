"""Tests for shared utility functions in data_collector/utils.py."""

import pandas as pd

from modules.input.data_collector import DataFetcher

# ===================================================================
# _ensure_fundamentals_schema
# ===================================================================


class TestEnsureFundamentalsSchema:

    def test_none_returns_empty_with_schema(self):
        result = DataFetcher._ensure_fundamentals_schema(None)
        assert result.empty
        assert "symbol" in result.columns
        assert "source" in result.columns

    def test_empty_returns_empty_with_schema(self):
        result = DataFetcher._ensure_fundamentals_schema(pd.DataFrame())
        assert result.empty
        assert "total_assets" in result.columns

    def test_renames_book_value_equity(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "book_value_equity": [1e11],
            }
        )
        result = DataFetcher._ensure_fundamentals_schema(df)
        assert "book_equity" in result.columns
        assert result.iloc[0]["book_equity"] == 1e11

    def test_renames_total_equity(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "total_equity": [9e10],
            }
        )
        result = DataFetcher._ensure_fundamentals_schema(df)
        assert "book_equity" in result.columns
        assert result.iloc[0]["book_equity"] == 9e10


# ===================================================================
# _apply_fundamentals_period
# ===================================================================


class TestApplyFundamentalsPeriod:

    def test_none_returns_none(self):
        assert DataFetcher._apply_fundamentals_period(None, "5y") is None

    def test_empty_returns_empty(self):
        result = DataFetcher._apply_fundamentals_period(pd.DataFrame(), "5y")
        assert result.empty

    def test_max_returns_all(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 40,
                "fiscal_year": [y for y in range(2014, 2024) for _ in range(4)],
                "fiscal_quarter": list(range(1, 5)) * 10,
            }
        )
        result = DataFetcher._apply_fundamentals_period(df, "max")
        assert len(result) == 40

    def test_limits_to_n_years(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 40,
                "fiscal_year": [y for y in range(2014, 2024) for _ in range(4)],
                "fiscal_quarter": list(range(1, 5)) * 10,
            }
        )
        result = DataFetcher._apply_fundamentals_period(df, "2y")
        assert len(result) == 8  # 2 years x 4 quarters

    def test_invalid_period_returns_all(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 4,
                "fiscal_year": [2024] * 4,
                "fiscal_quarter": [1, 2, 3, 4],
            }
        )
        result = DataFetcher._apply_fundamentals_period(df, "abc")
        assert len(result) == 4


# ===================================================================
# _period_years
# ===================================================================


class TestPeriodYears:

    def test_max_returns_none(self):
        assert DataFetcher._period_years("max") is None

    def test_valid_years(self):
        assert DataFetcher._period_years("5y") == 5
        assert DataFetcher._period_years("1y") == 1

    def test_invalid_returns_default(self):
        assert DataFetcher._period_years("abc") == 5
        assert DataFetcher._period_years("0y") == 5

    def test_none_returns_default(self):
        assert DataFetcher._period_years(None) == 5

    def test_non_numeric_year_returns_default(self):
        """'xy' where x is not a number returns default (line 104-105)."""
        assert DataFetcher._period_years("xy") == 5

    def test_zero_year_returns_default(self):
        """'0y' returns default because 0 is not > 0 (line 89-90)."""
        assert DataFetcher._period_years("0y") == 5

    def test_negative_year_returns_default(self):
        assert DataFetcher._period_years("-1y") == 5


class TestApplyFundamentalsPeriodEdgeCases:

    def test_non_numeric_year_in_period(self):
        """'xy' with non-numeric prefix returns all (line 89-90)."""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 4,
                "fiscal_year": [2024] * 4,
                "fiscal_quarter": [1, 2, 3, 4],
            }
        )
        result = DataFetcher._apply_fundamentals_period(df, "xy")
        assert len(result) == 4
