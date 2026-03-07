"""Tests for main.py pipeline orchestration."""

from unittest.mock import MagicMock, patch, mock_open

import pandas as pd
import pytest

from main import load_config, setup_logging, print_validation_report, main
from modules.processing.data_validator import ValidationResult


# ===================================================================
# load_config
# ===================================================================

class TestLoadConfig:

    @patch("main.Path")
    def test_success(self, mock_path_cls):
        mock_path = MagicMock()
        mock_path_cls.return_value.resolve.return_value.parent.__truediv__ = (
            MagicMock(return_value=mock_path)
        )
        mock_path.exists.return_value = True

        yaml_content = "postgres:\n  host: localhost\n"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            cfg = load_config()
        assert cfg["postgres"]["host"] == "localhost"

    def test_file_not_found(self, tmp_path):
        # Point load_config at a non-existent directory
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
# main
# ===================================================================

class TestMain:

    def _build_mocks(self):
        """Set up the standard mocks for main()."""
        cfg = {
            "postgres": {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p"},
            "mongo": {"host": "h", "port": 27017},
            "minio": {"host": "h", "access_key": "a", "secret_key": "s", "secure": False},
            "logging": {"level": "INFO"},
            "data": {"price_period": "5y", "fundamentals_workers": 1},
            "validation": {"min_price_rows": 5, "min_years": 1, "max_null_pct": 0.5, "strict": True},
            "dev": {"enabled": True, "max_symbols": 2},
        }
        return cfg

    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_happy_path(self, mock_load_cfg, mock_pg_cls, mock_mongo_cls,
                        mock_minio_cls, mock_fetcher_cls, mock_validator_cls,
                        mock_writer_cls):
        cfg = self._build_mocks()
        mock_load_cfg.return_value = cfg

        mock_pg = MagicMock()
        mock_pg.test_connection.return_value = True
        mock_pg.get_company_list.return_value = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "country": ["US", "US"],
        })
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
        prices = pd.DataFrame({
            "symbol": ["AAPL"] * 5,
            "trade_date": pd.bdate_range("2024-01-01", periods=5),
            "close_price": [150.0] * 5,
        })
        mock_fetcher.fetch_prices.return_value = prices
        mock_fetcher.fetch_fundamentals.return_value = pd.DataFrame({
            "symbol": ["AAPL"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
            "total_assets": [3e11],
        })
        mock_fetcher.fetch_risk_free_rates.return_value = pd.DataFrame({
            "country": ["US"],
            "rate_date": ["2024-01-01"],
            "rate": [0.04],
        })
        mock_fetcher_cls.return_value = mock_fetcher

        mock_validator = MagicMock()
        passed_result = ValidationResult()
        mock_validator.validate_all.return_value = {
            "prices": passed_result,
            "financials": passed_result,
            "risk_free_rates": passed_result,
        }
        mock_validator.clean_prices.return_value = prices
        mock_validator_cls.return_value = mock_validator

        mock_writer = MagicMock()
        mock_writer.write_prices.return_value = 5
        mock_writer.write_financials.return_value = 1
        mock_writer.write_risk_free_rates.return_value = 1
        mock_writer.get_table_counts.return_value = {"price_data": 5}
        mock_writer_cls.return_value = mock_writer

        main()

        mock_fetcher.fetch_prices.assert_called_once()
        mock_validator.clean_prices.assert_called_once()
        mock_validator.validate_all.assert_called_once()
        mock_writer.write_prices.assert_called_once()

    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_strict_mode_halts(self, mock_load_cfg, mock_pg_cls, mock_mongo_cls,
                               mock_minio_cls, mock_fetcher_cls, mock_validator_cls,
                               mock_writer_cls):
        cfg = self._build_mocks()
        mock_load_cfg.return_value = cfg

        mock_pg = MagicMock()
        mock_pg.test_connection.return_value = True
        mock_pg.get_company_list.return_value = pd.DataFrame({
            "symbol": ["AAPL"], "country": ["US"],
        })
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

        failed_result = ValidationResult()
        failed_result.add_error("data is bad")

        mock_validator = MagicMock()
        mock_validator.validate_all.return_value = {
            "prices": failed_result,
            "financials": ValidationResult(),
            "risk_free_rates": ValidationResult(),
        }
        mock_validator.clean_prices.return_value = pd.DataFrame()
        mock_validator_cls.return_value = mock_validator

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        main()

        mock_writer.write_prices.assert_not_called()

    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_connection_failure(self, mock_load_cfg, mock_pg_cls):
        cfg = self._build_mocks()
        mock_load_cfg.return_value = cfg

        mock_pg = MagicMock()
        mock_pg.test_connection.return_value = False
        mock_pg_cls.return_value = mock_pg

        with pytest.raises(RuntimeError, match="PostgreSQL"):
            main()
