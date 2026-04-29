"""Data validation module for quality assurance before database loading.

Validates fetched data against a set of rules before it enters PostgreSQL.
Each validation rule returns a report of issues found, allowing the pipeline
to decide whether to proceed, warn, or halt.

Validation rules:
    - Completeness: minimum row counts, no fully-null critical columns
    - Range checks: prices > 0, dates within expected window
    - Consistency: no duplicate (symbol, date) pairs
    - Coverage: percentage of symbols with usable data
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

from modules.input.data_collector.constants import COUNTRY_CODE_TO_NAME

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation outcomes.

    Attributes:
        passed: Whether all checks passed.
        warnings: List of non-critical issues (data is usable but imperfect).
        errors: List of critical issues (data should not be loaded).
        stats: Dictionary of summary statistics from the validation.
    """

    def __init__(self):
        self.passed = True
        self.warnings = []
        self.errors = []
        self.stats = {}

    def add_warning(self, message):
        """Record a non-critical issue.

        Args:
            message: Description of the warning.
        """
        self.warnings.append(message)
        logger.warning("VALIDATION WARNING: %s", message)

    def add_error(self, message):
        """Record a critical issue and mark validation as failed.

        Args:
            message: Description of the error.
        """
        self.passed = False
        self.errors.append(message)
        logger.error("VALIDATION ERROR: %s", message)

    def summary(self):
        """Return a human-readable summary string.

        Returns:
            str: Summary of validation results.
        """
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Validation {status}",
            f"  Warnings: {len(self.warnings)}",
            f"  Errors:   {len(self.errors)}",
        ]
        for key, val in self.stats.items():
            lines.append(f"  {key}: {val}")
        return "\n".join(lines)


class DataValidator:
    """Validates fetched financial data before loading into PostgreSQL.

    Runs a configurable set of checks on price data, financial
    statements, and risk-free rates. Returns a ValidationResult
    for each dataset so the pipeline can decide how to proceed.

    Args:
        min_price_rows: Minimum expected rows per symbol for prices.
        min_years: Minimum years of price history expected.
        max_null_pct: Maximum allowed percentage of nulls in critical columns.
    """

    def __init__(self, min_price_rows=200, min_years=4, max_null_pct=0.5):
        self.min_price_rows = min_price_rows
        self.min_years = min_years
        self.max_null_pct = max_null_pct

    def validate_prices(self, df, expected_symbols=None):
        """Validate daily price data.

        Checks:
            - DataFrame is not empty
            - No negative or zero prices
            - Date range spans at least min_years
            - Each symbol has at least min_price_rows rows
            - No duplicate (symbol, date) pairs
            - Close price column has acceptable null rate

        Args:
            df: Price DataFrame with columns: symbol, price_date,
                open_price, high_price, low_price, close_price, volume.
            expected_symbols: Optional list of symbols we expected to get.

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        if df is None or df.empty:
            result.add_error("Price DataFrame is empty.")
            return result

        total_rows = len(df)
        symbols = df["symbol"].nunique()
        result.stats["total_rows"] = total_rows
        result.stats["unique_symbols"] = symbols

        # Check for negative or zero prices
        for col in ["open_price", "high_price", "low_price"]:
            if col in df.columns:
                bad = (df[col] <= 0).sum()
                if bad > 0:
                    result.add_warning(f"{bad} rows have {col} <= 0")

        # close_price: error if >1% of rows are zero/negative
        if "close_price" in df.columns:
            bad_close = (df["close_price"] <= 0).sum()
            if bad_close > 0.01 * total_rows:
                result.add_error(
                    f"{bad_close} rows have close_price <= 0 "
                    f"({bad_close / total_rows:.1%} of data)"
                )
            elif bad_close > 0:
                result.add_warning(f"{bad_close} rows have close_price <= 0")
        # Check date range
        if "trade_date" in df.columns:
            df_dates = pd.to_datetime(df["trade_date"])
            date_min = df_dates.min()
            date_max = df_dates.max()
            date_span_years = (date_max - date_min).days / 365.25
            result.stats["date_range"] = f"{date_min.date()} to {date_max.date()}"
            result.stats["date_span_years"] = round(date_span_years, 1)

            if date_span_years < self.min_years:
                result.add_warning(
                    f"Date span is {date_span_years:.1f} years, "
                    f"expected at least {self.min_years}"
                )

        # Check rows per symbol
        rows_per_symbol = df.groupby("symbol").size()
        thin_symbols = rows_per_symbol[rows_per_symbol < self.min_price_rows]
        if len(thin_symbols) > 0:
            result.add_warning(
                f"{len(thin_symbols)} symbols have fewer than "
                f"{self.min_price_rows} rows"
            )
        result.stats["min_rows_per_symbol"] = int(rows_per_symbol.min())
        result.stats["median_rows_per_symbol"] = int(rows_per_symbol.median())

        # Check for duplicates
        dupes = df.duplicated(subset=["symbol", "trade_date"]).sum()
        if dupes > 0:
            result.add_error(f"{dupes} duplicate (symbol, trade_date) rows")

        # Check null rate on close_price
        if "close_price" in df.columns:
            null_pct = df["close_price"].isna().mean()
            result.stats["close_price_null_pct"] = f"{null_pct:.2%}"
            if null_pct > self.max_null_pct:
                result.add_error(
                    f"close_price null rate is {null_pct:.2%}, "
                    f"exceeds {self.max_null_pct:.0%} threshold"
                )

        # Check symbol coverage
        if expected_symbols is not None:
            expected = set(s.strip() for s in expected_symbols)
            actual = set(df["symbol"].unique())
            missing = expected - actual
            coverage = len(actual) / len(expected) if expected else 0
            result.stats["symbol_coverage"] = f"{coverage:.1%}"
            if missing:
                result.add_warning(
                    f"{len(missing)} expected symbols missing "
                    f"({coverage:.1%} coverage)"
                )
            if coverage < 0.95:
                result.add_error(
                    f"Symbol coverage is {coverage:.1%}, below 95% threshold"
                )

        return result

    def validate_financials(self, df, expected_symbols=None):
        """Validate quarterly financial statement data.

        Checks:
            - DataFrame is not empty
            - Total assets are positive where present
            - No duplicate (symbol, fiscal_date) pairs
            - Critical columns have acceptable null rates
            - Reasonable value ranges (no astronomically wrong numbers)

        Args:
            df: Financials DataFrame with columns: symbol, fiscal_date,
                total_assets, total_equity, total_debt, net_income, etc.
            expected_symbols: Optional list of symbols we expected to get.

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        if df is None or df.empty:
            result.add_error("Financials DataFrame is empty.")
            return result

        total_rows = len(df)
        symbols = df["symbol"].nunique()
        result.stats["total_rows"] = total_rows
        result.stats["unique_symbols"] = symbols

        # Check total_assets is positive where present
        if "total_assets" in df.columns:
            non_null = df["total_assets"].dropna()
            if len(non_null) > 0:
                negative = (non_null <= 0).sum()
                if negative > 0:
                    result.add_warning(f"{negative} rows have total_assets <= 0")

        # Check for duplicates
        dupes = df.duplicated(subset=["symbol", "fiscal_year", "fiscal_quarter"]).sum()
        if dupes > 0:
            result.add_error(
                f"{dupes} duplicate (symbol, fiscal_year, fiscal_quarter) rows"
            )

        # Check null rates on critical columns
        critical = ["total_assets", "book_equity", "net_income"]
        for col in critical:
            if col in df.columns:
                null_pct = df[col].isna().mean()
                result.stats[f"{col}_null_pct"] = f"{null_pct:.2%}"
                if null_pct > self.max_null_pct:
                    result.add_warning(
                        f"{col} null rate is {null_pct:.2%}, "
                        f"exceeds {self.max_null_pct:.0%}"
                    )

        # Check symbol coverage
        if expected_symbols is not None:
            expected = set(s.strip() for s in expected_symbols)
            actual = set(df["symbol"].unique())
            missing = expected - actual
            coverage = len(actual) / len(expected) if expected else 0
            result.stats["symbol_coverage"] = f"{coverage:.1%}"
            if len(missing) > 0:
                result.add_warning(
                    f"{len(missing)} expected symbols missing "
                    f"({coverage:.1%} coverage)"
                )

        return result

    def validate_risk_free_rates(self, df, expected_countries=None):
        """Validate risk-free rate data.

        Checks:
            - DataFrame is not empty
            - Rates are within reasonable bounds (0-100%)
            - Date range covers recent period
            - No duplicate (country, rate_date) pairs

        Args:
            df: Risk-free rates DataFrame with columns:
                country, rate_date, rate.
            expected_countries: Optional list of country codes expected.

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        if df is None or df.empty:
            result.add_error("Risk-free rates DataFrame is empty.")
            return result

        result.stats["total_rows"] = len(df)
        result.stats["unique_countries"] = df["country"].nunique()

        # Check rate bounds (should be between -0.1 and 1.0 i.e. -10% to 100%)
        if "rate" in df.columns:
            non_null = df["rate"].dropna()
            out_of_range = ((non_null < -0.1) | (non_null > 1.0)).sum()
            if out_of_range > 0:
                result.add_warning(f"{out_of_range} rates outside [-10%, 100%] range")

        # Check date range covers at least last 1 year
        if "rate_date" in df.columns:
            df_dates = pd.to_datetime(df["rate_date"])
            date_max = df_dates.max()
            one_year_ago = datetime.now() - timedelta(days=365)
            result.stats["latest_rate_date"] = str(date_max.date())
            if date_max < pd.Timestamp(one_year_ago):
                result.add_warning(
                    f"Latest rate date is {date_max.date()}, " f"more than 1 year old"
                )

        # Check for duplicates
        dupes = df.duplicated(subset=["country", "rate_date"]).sum()
        if dupes > 0:
            result.add_error(f"{dupes} duplicate (country, rate_date) rows")

        # Check country coverage — translate codes to full names to match stored values
        if expected_countries is not None:
            expected = set(
                COUNTRY_CODE_TO_NAME.get(c.strip(), c.strip())
                for c in expected_countries
            )
            actual = set(df["country"].unique())
            missing = expected - actual
            if missing:
                result.add_warning(f"Missing countries: {missing}")

        return result

    def clean_prices(self, df):
        """Remove rows with invalid close prices before validation.

        Drops rows where close_price is zero or negative. These are data
        errors (impossible for a real stock) and should not enter the
        database. Logs a warning so there is an audit trail of what was
        dropped.

        Args:
            df: Price DataFrame with a close_price column.

        Returns:
            pd.DataFrame: Cleaned DataFrame with bad rows removed.
        """
        if df is None or df.empty:
            return df

        if "close_price" not in df.columns:
            return df

        before = len(df)
        df = df[df["close_price"] > 0].copy()
        dropped = before - len(df)

        if dropped > 0:
            logger.warning(
                "clean_prices: dropped %d rows with close_price <= 0 "
                "(%.1f%% of data)",
                dropped,
                dropped / before * 100,
            )

        return df

    def validate_all(
        self,
        prices_df,
        financials_df,
        rates_df,
        expected_symbols=None,
        expected_countries=None,
    ):
        """Run all validations and return a combined report.

        Args:
            prices_df: Price data DataFrame.
            financials_df: Financial statements DataFrame.
            rates_df: Risk-free rates DataFrame.
            expected_symbols: Optional list of expected ticker symbols.
            expected_countries: Optional list of expected country codes.

        Returns:
            dict: Mapping of dataset name to ValidationResult.
        """
        results = {
            "prices": self.validate_prices(prices_df, expected_symbols),
            "financials": self.validate_financials(financials_df, expected_symbols),
            "risk_free_rates": self.validate_risk_free_rates(
                rates_df, expected_countries
            ),
        }

        all_passed = all(r.passed for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())

        logger.info(
            "Validation complete: %s | %d warnings | %d errors",
            "ALL PASSED" if all_passed else "FAILURES DETECTED",
            total_warnings,
            total_errors,
        )

        return results
