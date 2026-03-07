"""Tests for modules.processing.data_validator."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from modules.processing.data_validator import DataValidator, ValidationResult


# ===================================================================
# ValidationResult
# ===================================================================

class TestValidationResult:

    def test_initial_state(self):
        r = ValidationResult()
        assert r.passed is True
        assert r.warnings == []
        assert r.errors == []
        assert r.stats == {}

    def test_add_warning_does_not_fail(self):
        r = ValidationResult()
        r.add_warning("minor issue")
        assert len(r.warnings) == 1
        assert r.passed is True

    def test_add_error_sets_passed_false(self):
        r = ValidationResult()
        r.add_error("critical issue")
        assert len(r.errors) == 1
        assert r.passed is False

    def test_multiple_warnings_and_errors(self):
        r = ValidationResult()
        r.add_warning("w1")
        r.add_warning("w2")
        r.add_error("e1")
        assert len(r.warnings) == 2
        assert len(r.errors) == 1
        assert r.passed is False

    def test_summary_passed(self):
        r = ValidationResult()
        r.add_warning("minor")
        s = r.summary()
        assert "PASSED" in s
        assert "Warnings: 1" in s

    def test_summary_failed_with_stats(self):
        r = ValidationResult()
        r.add_error("bad")
        r.stats["total_rows"] = 100
        s = r.summary()
        assert "FAILED" in s
        assert "Errors:   1" in s
        assert "total_rows: 100" in s


# ===================================================================
# validate_prices
# ===================================================================

class TestValidatePrices:

    def test_happy_path(self, validator, sample_prices_df, expected_symbols):
        result = validator.validate_prices(sample_prices_df, expected_symbols)
        assert result.passed is True
        assert result.stats["unique_symbols"] == 2

    def test_empty_df_error(self, validator):
        result = validator.validate_prices(None)
        assert result.passed is False
        assert any("empty" in e.lower() for e in result.errors)

        result2 = validator.validate_prices(pd.DataFrame())
        assert result2.passed is False

    def test_negative_open_high_low_warning(self, validator):
        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "trade_date": pd.bdate_range("2024-01-01", periods=10),
            "open_price": [-1.0] + [100.0] * 9,
            "high_price": [100.0] * 10,
            "low_price": [100.0] * 10,
            "close_price": [100.0] * 10,
        })
        result = validator.validate_prices(df)
        assert result.passed is True
        assert any("open_price" in w for w in result.warnings)

    def test_close_price_error_above_1pct(self, validator):
        n = 100
        df = pd.DataFrame({
            "symbol": ["A"] * n,
            "trade_date": pd.bdate_range("2024-01-01", periods=n),
            "close_price": [0.0] * 5 + [100.0] * 95,
        })
        result = validator.validate_prices(df)
        assert result.passed is False
        assert any("close_price" in e for e in result.errors)

    def test_close_price_warning_below_1pct(self, validator):
        n = 200
        df = pd.DataFrame({
            "symbol": ["A"] * n,
            "trade_date": pd.bdate_range("2024-01-01", periods=n),
            "close_price": [0.0] + [100.0] * 199,
        })
        result = validator.validate_prices(df)
        assert result.passed is True
        assert any("close_price" in w for w in result.warnings)

    def test_short_date_span_warning(self):
        v = DataValidator(min_price_rows=5, min_years=10, max_null_pct=0.5)
        df = pd.DataFrame({
            "symbol": ["A"] * 20,
            "trade_date": pd.bdate_range("2024-01-01", periods=20),
            "close_price": [100.0] * 20,
        })
        result = v.validate_prices(df)
        assert any("Date span" in w for w in result.warnings)

    def test_thin_symbols_warning(self):
        v = DataValidator(min_price_rows=100, min_years=1, max_null_pct=0.5)
        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "trade_date": pd.bdate_range("2024-01-01", periods=10),
            "close_price": [100.0] * 10,
        })
        result = v.validate_prices(df)
        assert any("fewer than" in w for w in result.warnings)

    def test_duplicate_rows_error(self, validator):
        df = pd.DataFrame({
            "symbol": ["A", "A"],
            "trade_date": ["2024-01-02", "2024-01-02"],
            "close_price": [100.0, 100.0],
        })
        result = validator.validate_prices(df)
        assert result.passed is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_high_close_null_rate_error(self):
        v = DataValidator(min_price_rows=1, min_years=0, max_null_pct=0.1)
        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "trade_date": pd.bdate_range("2024-01-01", periods=10),
            "close_price": [np.nan] * 5 + [100.0] * 5,
        })
        result = v.validate_prices(df)
        assert result.passed is False
        assert any("null rate" in e.lower() for e in result.errors)

    def test_low_symbol_coverage_error(self, validator):
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 20,
            "trade_date": pd.bdate_range("2024-01-01", periods=20),
            "close_price": [100.0] * 20,
        })
        expected = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        result = validator.validate_prices(df, expected_symbols=expected)
        assert result.passed is False
        assert any("coverage" in e.lower() for e in result.errors)


# ===================================================================
# validate_financials
# ===================================================================

class TestValidateFinancials:

    def test_happy_path(self, validator, sample_financials_df, expected_symbols):
        result = validator.validate_financials(sample_financials_df, expected_symbols)
        assert result.passed is True

    def test_empty_df(self, validator):
        result = validator.validate_financials(None)
        assert result.passed is False

    def test_negative_total_assets_warning(self, validator):
        df = pd.DataFrame({
            "symbol": ["A"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
            "total_assets": [-1e9],
        })
        result = validator.validate_financials(df)
        assert any("total_assets" in w for w in result.warnings)

    def test_duplicate_rows_error(self, validator):
        df = pd.DataFrame({
            "symbol": ["A", "A"],
            "fiscal_year": [2024, 2024],
            "fiscal_quarter": [1, 1],
            "total_assets": [1e9, 1e9],
        })
        result = validator.validate_financials(df)
        assert result.passed is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_high_null_rate_warning(self):
        v = DataValidator(min_price_rows=5, min_years=1, max_null_pct=0.1)
        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "fiscal_year": list(range(2015, 2025)),
            "fiscal_quarter": [1] * 10,
            "total_assets": [np.nan] * 5 + [1e9] * 5,
            "total_equity": [1e9] * 10,
            "net_income": [1e8] * 10,
        })
        result = v.validate_financials(df)
        assert any("total_assets" in w and "null" in w.lower() for w in result.warnings)


# ===================================================================
# validate_risk_free_rates
# ===================================================================

class TestValidateRiskFreeRates:

    def test_happy_path(self, validator, sample_rates_df, expected_countries):
        result = validator.validate_risk_free_rates(sample_rates_df, expected_countries)
        assert result.passed is True

    def test_empty_df(self, validator):
        result = validator.validate_risk_free_rates(None)
        assert result.passed is False

    def test_out_of_bounds_warning(self, validator):
        df = pd.DataFrame({
            "country": ["US", "US"],
            "rate_date": [date.today(), date.today() - timedelta(days=1)],
            "rate": [2.0, 0.04],
        })
        result = validator.validate_risk_free_rates(df)
        assert any("range" in w.lower() for w in result.warnings)

    def test_old_dates_warning(self, validator):
        old = date.today() - timedelta(days=500)
        df = pd.DataFrame({
            "country": ["US"],
            "rate_date": [old],
            "rate": [0.04],
        })
        result = validator.validate_risk_free_rates(df)
        assert any("old" in w.lower() or "year" in w.lower() for w in result.warnings)


# ===================================================================
# clean_prices
# ===================================================================

class TestCleanPrices:

    def test_removes_bad_rows(self, validator):
        df = pd.DataFrame({
            "close_price": [100.0, 0.0, -5.0, 200.0],
        })
        cleaned = validator.clean_prices(df)
        assert len(cleaned) == 2
        assert (cleaned["close_price"] > 0).all()

    def test_handles_empty_and_none(self, validator):
        assert validator.clean_prices(None) is None
        empty = pd.DataFrame()
        result = validator.clean_prices(empty)
        assert result.empty


# ===================================================================
# validate_all
# ===================================================================

class TestValidateAll:

    def test_returns_all_three(self, validator, sample_prices_df,
                               sample_financials_df, sample_rates_df,
                               expected_symbols, expected_countries):
        results = validator.validate_all(
            sample_prices_df, sample_financials_df, sample_rates_df,
            expected_symbols=expected_symbols,
            expected_countries=expected_countries,
        )
        assert set(results.keys()) == {"prices", "financials", "risk_free_rates"}
        assert all(r.passed for r in results.values())
