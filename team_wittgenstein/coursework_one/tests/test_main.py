"""Tests for main.py pipeline orchestration."""

from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from main import (
    _load_universe,
    load_config,
    print_validation_report,
    run_fundamentals,
    run_prices_and_rates,
    setup_logging,
    main,
)
from modules.processing.data_validator import ValidationResult


# ===================================================================
# Helpers
# ===================================================================

def _make_cfg(**overrides):
    cfg = {
        "postgres": {
            "host": "h", "port": 5432, "database": "d",
            "user": "u", "password": "p",
        },
        "mongo": {"host": "h", "port": 27017},
        "minio": {"host": "h", "access_key": "a", "secret_key": "s", "secure": False},
        "logging": {"level": "INFO"},
        "data": {
            "price_period": "5y",
            "fundamentals_period": "5y",
            "fundamentals_source": "waterfall",
        },
        "country_filter": "US",
        "validation": {
            "min_price_rows": 5, "min_years": 1,
            "max_null_pct": 0.5, "strict": True,
        },
        "dev": {"enabled": True, "max_symbols": 2},
        "scheduler": {
            "prices_and_rates": {"day": 1, "hour": 2, "minute": 0},
            "fundamentals": {"month": "1,4,7,10", "day": 1, "hour": 4, "minute": 0},
        },
    }
    cfg.update(overrides)
    return cfg


def _make_universe():
    return pd.DataFrame({"symbol": ["AAPL", "MSFT"], "country": ["US", "US"]})


def _make_prices():
    return pd.DataFrame({
        "symbol": ["AAPL"] * 5,
        "trade_date": pd.bdate_range("2024-01-01", periods=5),
        "close_price": [150.0] * 5,
    })


def _make_financials():
    return pd.DataFrame({
        "symbol": ["AAPL"],
        "fiscal_year": [2024],
        "fiscal_quarter": [1],
        "total_assets": [3e11],
    })


def _make_rates():
    return pd.DataFrame({
        "country": ["US"], "rate_date": ["2024-01-01"], "rate": [0.04],
    })


def _passed():
    return ValidationResult()


def _failed(msg="data is bad"):
    r = ValidationResult()
    r.add_error(msg)
    return r


# ===================================================================
# load_config
# ===================================================================


class TestLoadConfig:

    @patch("main.Path")
    def test_success(self, mock_path_cls):
        mock_path = MagicMock()
        mock_path_cls.return_value.resolve.return_value.parent.__truediv__ = MagicMock(
            return_value=mock_path
        )
        mock_path.exists.return_value = True

        yaml_content = "postgres:\n  host: localhost\n"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            cfg = load_config()
        assert cfg["postgres"]["host"] == "localhost"

    def test_file_not_found(self, tmp_path):
        fake_dir = tmp_path / "nonexistent" / "config"
        with patch("main.Path") as mock_path_cls:
            mock_resolved = MagicMock()
            mock_resolved.parent.__truediv__ = MagicMock(
                return_value=fake_dir / "conf.yaml"
            )
            mock_path_cls.return_value.resolve.return_value = mock_resolved
            with pytest.raises(FileNotFoundError):
                load_config()


# ===================================================================
# setup_logging
# ===================================================================


class TestSetupLogging:

    @patch("main.logging")
    def test_sets_level(self, mock_logging):
        mock_logging.INFO = 20
        mock_logging.DEBUG = 10
        setup_logging("DEBUG")
        mock_logging.basicConfig.assert_called_once()


# ===================================================================
# print_validation_report
# ===================================================================


class TestPrintValidationReport:

    def test_output(self, capsys):
        r = ValidationResult()
        r.add_warning("test warning")
        r.stats["total_rows"] = 100
        print_validation_report({"prices": r})
        captured = capsys.readouterr()
        assert "VALIDATION REPORT" in captured.out
        assert "PRICES" in captured.out
        assert "test warning" in captured.out


# ===================================================================
# _load_universe
# ===================================================================


class TestLoadUniverse:

    def _make_pg(self, df=None):
        pg = MagicMock()
        pg.get_company_list.return_value = df if df is not None else _make_universe()
        pg.delete_symbols_missing_from_company_list.return_value = []
        return pg

    def test_returns_symbols_and_countries(self):
        cfg = _make_cfg()
        pg = self._make_pg()
        fetcher = MagicMock()
        symbols, countries = _load_universe(pg, fetcher, cfg)
        assert "AAPL" in symbols
        assert "US" in countries

    def test_applies_country_filter(self):
        df = pd.DataFrame({"symbol": ["AAPL", "BP"], "country": ["US", "GB"]})
        cfg = _make_cfg(country_filter="US")
        pg = self._make_pg(df)
        fetcher = MagicMock()
        symbols, _ = _load_universe(pg, fetcher, cfg)
        assert "AAPL" in symbols
        assert "BP" not in symbols

    def test_applies_exclusions(self):
        cfg = _make_cfg(exclude_symbols=["MSFT"])
        pg = self._make_pg()
        fetcher = MagicMock()
        symbols, _ = _load_universe(pg, fetcher, cfg)
        assert "MSFT" not in symbols
        assert "AAPL" in symbols

    def test_normalises_dot_tickers(self):
        df = pd.DataFrame({"symbol": ["BRK.B"], "country": ["US"]})
        cfg = _make_cfg()
        pg = self._make_pg(df)
        fetcher = MagicMock()
        symbols, _ = _load_universe(pg, fetcher, cfg)
        assert "BRK-B" in symbols
        assert "BRK.B" not in symbols

    def test_dev_mode_limits_symbols(self):
        df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOG"],
            "country": ["US", "US", "US"],
        })
        cfg = _make_cfg(dev={"enabled": True, "max_symbols": 2})
        pg = self._make_pg(df)
        fetcher = MagicMock()
        symbols, _ = _load_universe(pg, fetcher, cfg)
        assert len(symbols) == 2

    def test_calls_cleanup(self):
        pg = self._make_pg()
        pg.delete_symbols_missing_from_company_list.return_value = ["STALE"]
        fetcher = MagicMock()
        _load_universe(pg, fetcher, _make_cfg())
        fetcher.delete_symbol_cache.assert_called_once_with("STALE")

    def test_cleanup_normalises_dot_tickers(self):
        """Dot-tickers in company_static must be normalised before cleanup check.

        BF.B is stored in company_static but data is written as BF-B.
        Without normalisation, BF-B is wrongly treated as stale and deleted.
        """
        df = pd.DataFrame({"symbol": ["BF.B", "BRK.B"], "country": ["US", "US"]})
        pg = self._make_pg(df)
        pg.delete_symbols_missing_from_company_list.return_value = []
        fetcher = MagicMock()
        _load_universe(pg, fetcher, _make_cfg(country_filter=None))
        passed_symbols = pg.delete_symbols_missing_from_company_list.call_args[0][0]
        assert "BF-B" in passed_symbols
        assert "BRK-B" in passed_symbols
        assert "BF.B" not in passed_symbols
        assert "BRK.B" not in passed_symbols

    def test_raises_on_empty_universe(self):
        pg = self._make_pg(pd.DataFrame())
        fetcher = MagicMock()
        with pytest.raises(RuntimeError, match="company_static"):
            _load_universe(pg, fetcher, _make_cfg())


# ===================================================================
# run_prices_and_rates
# ===================================================================


class TestRunPricesAndRates:

    def _make_ctx(self, validator_overrides=None):
        prices = _make_prices()
        ctx = MagicMock()
        ctx.cfg = _make_cfg()
        ctx.strict = True
        ctx.fetcher.price_failures = {}
        ctx.fetcher.fetch_prices.return_value = prices
        ctx.fetcher.fetch_risk_free_rates.return_value = _make_rates()
        ctx.validator.clean_prices.return_value = prices
        ctx.validator.validate_prices.return_value = _passed()
        ctx.validator.validate_risk_free_rates.return_value = _passed()
        ctx.writer.write_prices.return_value = 5
        ctx.writer.write_risk_free_rates.return_value = 1
        return ctx

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_happy_path(self, mock_lu):
        ctx = self._make_ctx()
        run_prices_and_rates(ctx)
        ctx.writer.write_prices.assert_called_once()
        ctx.writer.write_risk_free_rates.assert_called_once()

    @patch("main._load_universe", return_value=(["AAPL"], ["US"]))
    def test_strict_halts_on_price_failure(self, mock_lu):
        ctx = self._make_ctx()
        ctx.validator.validate_prices.return_value = _failed()
        run_prices_and_rates(ctx)
        ctx.writer.write_prices.assert_not_called()

    @patch("main._load_universe", return_value=(["AAPL"], ["US"]))
    def test_logs_price_failures_to_mongo(self, mock_lu):
        ctx = self._make_ctx()
        ctx.fetcher.price_failures = {"delisted": ["OLD"], "fetch_error": []}
        run_prices_and_rates(ctx)
        ctx.writer.log_fetch_to_mongo.assert_called_once()

    @patch("main._load_universe", side_effect=RuntimeError("db down"))
    def test_exception_is_caught(self, mock_lu):
        ctx = self._make_ctx()
        # Should not raise — exception is caught and logged
        run_prices_and_rates(ctx)
        ctx.writer.write_prices.assert_not_called()


# ===================================================================
# run_fundamentals
# ===================================================================


class TestRunFundamentals:

    def _make_ctx(self):
        ctx = MagicMock()
        ctx.cfg = _make_cfg()
        ctx.strict = True
        ctx.fetcher.fundamentals_failures = {}
        ctx.fetcher.fetch_fundamentals.return_value = _make_financials()
        ctx.validator.validate_financials.return_value = _passed()
        ctx.writer.write_financials.return_value = 1
        return ctx

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_happy_path(self, mock_lu):
        ctx = self._make_ctx()
        run_fundamentals(ctx)
        ctx.writer.write_financials.assert_called_once()

    @patch("main._load_universe", return_value=(["AAPL"], ["US"]))
    def test_strict_halts_on_validation_failure(self, mock_lu):
        ctx = self._make_ctx()
        ctx.validator.validate_financials.return_value = _failed()
        run_fundamentals(ctx)
        ctx.writer.write_financials.assert_not_called()

    @patch("main._load_universe", return_value=(["AAPL"], ["US"]))
    def test_logs_fundamentals_failures_to_mongo(self, mock_lu):
        ctx = self._make_ctx()
        ctx.fetcher.fundamentals_failures = {"delisted": ["OLD"], "fetch_error": []}
        run_fundamentals(ctx)
        ctx.writer.log_fetch_to_mongo.assert_called_once()

    @patch("main._load_universe", side_effect=RuntimeError("db down"))
    def test_exception_is_caught(self, mock_lu):
        ctx = self._make_ctx()
        run_fundamentals(ctx)
        ctx.writer.write_financials.assert_not_called()


# ===================================================================
# main
# ===================================================================


class TestMain:

    def _base_patches(self):
        """Return a dict of patches needed to run main() without side effects."""
        return {
            "main.load_config": _make_cfg(),
            "pg_test": True,
            "mongo_test": True,
            "minio_test": True,
        }

    @patch("main.BlockingScheduler")
    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_happy_path(
        self,
        mock_load_cfg,
        mock_pg_cls,
        mock_mongo_cls,
        mock_minio_cls,
        mock_fetcher_cls,
        mock_validator_cls,
        mock_writer_cls,
        mock_scheduler_cls,
    ):
        mock_load_cfg.return_value = _make_cfg()

        mock_pg = MagicMock()
        mock_pg.test_connection.return_value = True
        mock_pg.get_company_list.return_value = _make_universe()
        mock_pg.delete_symbols_missing_from_company_list.return_value = ["STALE"]
        mock_pg_cls.return_value = mock_pg

        mock_mongo = MagicMock()
        mock_mongo.test_connection.return_value = True
        mock_mongo_cls.return_value = mock_mongo

        mock_minio = MagicMock()
        mock_minio.test_connection.return_value = True
        mock_minio_cls.return_value = mock_minio

        prices = _make_prices()
        mock_fetcher = MagicMock()
        mock_fetcher.price_failures = {}
        mock_fetcher.fundamentals_failures = {}
        mock_fetcher.fetch_prices.return_value = prices
        mock_fetcher.fetch_fundamentals.return_value = _make_financials()
        mock_fetcher.fetch_risk_free_rates.return_value = _make_rates()
        mock_fetcher_cls.return_value = mock_fetcher

        mock_validator = MagicMock()
        mock_validator.clean_prices.return_value = prices
        mock_validator.validate_prices.return_value = _passed()
        mock_validator.validate_risk_free_rates.return_value = _passed()
        mock_validator.validate_financials.return_value = _passed()
        mock_validator_cls.return_value = mock_validator

        mock_writer = MagicMock()
        mock_writer.write_prices.return_value = 5
        mock_writer.write_financials.return_value = 1
        mock_writer.write_risk_free_rates.return_value = 1
        mock_writer.get_table_counts.return_value = {"price_data": 5}
        mock_writer_cls.return_value = mock_writer

        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        main([])

        mock_fetcher.fetch_prices.assert_called()
        mock_fetcher.fetch_fundamentals.assert_called()
        mock_fetcher.fetch_risk_free_rates.assert_called()
        mock_fetcher.delete_symbol_cache.assert_called_with("STALE")
        mock_validator.validate_prices.assert_called()
        mock_validator.validate_financials.assert_called()
        # Scheduler should have two jobs registered
        assert mock_scheduler.add_job.call_count == 2
        mock_scheduler.start.assert_called_once()

    @patch("main.BlockingScheduler")
    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_strict_mode_halts_write(
        self,
        mock_load_cfg,
        mock_pg_cls,
        mock_mongo_cls,
        mock_minio_cls,
        mock_fetcher_cls,
        mock_validator_cls,
        mock_writer_cls,
        mock_scheduler_cls,
    ):
        mock_load_cfg.return_value = _make_cfg()

        mock_pg = MagicMock()
        mock_pg.test_connection.return_value = True
        mock_pg.get_company_list.return_value = _make_universe()
        mock_pg.delete_symbols_missing_from_company_list.return_value = []
        mock_pg_cls.return_value = mock_pg

        mock_mongo = MagicMock()
        mock_mongo.test_connection.return_value = True
        mock_mongo_cls.return_value = mock_mongo

        mock_minio = MagicMock()
        mock_minio.test_connection.return_value = True
        mock_minio_cls.return_value = mock_minio

        mock_fetcher = MagicMock()
        mock_fetcher.price_failures = {}
        mock_fetcher.fundamentals_failures = {}
        mock_fetcher.fetch_prices.return_value = pd.DataFrame()
        mock_fetcher.fetch_fundamentals.return_value = pd.DataFrame()
        mock_fetcher.fetch_risk_free_rates.return_value = pd.DataFrame()
        mock_fetcher_cls.return_value = mock_fetcher

        mock_validator = MagicMock()
        mock_validator.clean_prices.return_value = pd.DataFrame()
        mock_validator.validate_prices.return_value = _failed()
        mock_validator.validate_risk_free_rates.return_value = _passed()
        mock_validator.validate_financials.return_value = _passed()
        mock_validator_cls.return_value = mock_validator

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        mock_scheduler_cls.return_value = MagicMock()

        main([])

        mock_writer.write_prices.assert_not_called()

    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_connection_failure(self, mock_load_cfg, mock_pg_cls):
        mock_load_cfg.return_value = _make_cfg()
        mock_pg = MagicMock()
        mock_pg.test_connection.return_value = False
        mock_pg_cls.return_value = mock_pg

        with pytest.raises(RuntimeError, match="PostgreSQL"):
            main([])

    @patch("main.BlockingScheduler")
    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_scheduler_uses_config_times(
        self,
        mock_load_cfg,
        mock_pg_cls,
        mock_mongo_cls,
        mock_minio_cls,
        mock_fetcher_cls,
        mock_validator_cls,
        mock_writer_cls,
        mock_scheduler_cls,
    ):
        """Scheduler add_job is called with CronTrigger built from config values."""
        cfg = _make_cfg()
        cfg["scheduler"] = {
            "prices_and_rates": {"day": 15, "hour": 6, "minute": 30},
            "fundamentals": {"month": "2,5,8,11", "day": 10, "hour": 8, "minute": 0},
        }
        mock_load_cfg.return_value = cfg

        mock_pg = MagicMock()
        mock_pg.test_connection.return_value = True
        mock_pg.get_company_list.return_value = _make_universe()
        mock_pg.delete_symbols_missing_from_company_list.return_value = []
        mock_pg_cls.return_value = mock_pg

        for mock_cls, test_val in [
            (mock_mongo_cls, True), (mock_minio_cls, True)
        ]:
            mock_cls.return_value.test_connection.return_value = test_val

        prices = _make_prices()
        mock_fetcher = MagicMock()
        mock_fetcher.price_failures = {}
        mock_fetcher.fundamentals_failures = {}
        mock_fetcher.fetch_prices.return_value = prices
        mock_fetcher.fetch_fundamentals.return_value = _make_financials()
        mock_fetcher.fetch_risk_free_rates.return_value = _make_rates()
        mock_fetcher_cls.return_value = mock_fetcher

        mock_validator = MagicMock()
        mock_validator.clean_prices.return_value = prices
        mock_validator.validate_prices.return_value = _passed()
        mock_validator.validate_risk_free_rates.return_value = _passed()
        mock_validator.validate_financials.return_value = _passed()
        mock_validator_cls.return_value = mock_validator

        mock_writer = MagicMock()
        mock_writer.get_table_counts.return_value = {}
        mock_writer_cls.return_value = mock_writer

        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        with patch("main.CronTrigger") as mock_cron:
            main([])

        # First call: prices_and_rates trigger with day=15, hour=6, minute=30
        pr_call = mock_cron.call_args_list[0]
        assert pr_call.kwargs.get("day") == 15
        assert pr_call.kwargs.get("hour") == 6
        assert pr_call.kwargs.get("minute") == 30

        # Second call: fundamentals trigger with month="2,5,8,11", day=10
        fund_call = mock_cron.call_args_list[1]
        assert fund_call.kwargs.get("month") == "2,5,8,11"
        assert fund_call.kwargs.get("day") == 10
