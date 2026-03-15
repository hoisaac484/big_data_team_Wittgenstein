"""End-to-end and CLI argument tests.

E2E tests run the real DataValidator and real DataWriter logic —
only external I/O boundaries (DB connections, API calls) are mocked.
This ensures that validation rules and write deduplication logic
are exercised as they would be in production.

CLI tests verify that parse_args() and main() honour --task,
--no-schedule, and --run-date arguments correctly.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from main import PipelineContext, parse_args, run_fundamentals, run_prices_and_rates, main
from modules.output.data_writer import DataWriter
from modules.processing.data_validator import DataValidator


# ===================================================================
# Synthetic data helpers
# ===================================================================

def _prices_df(symbols=("AAPL", "MSFT"), years=2):
    """Realistic price DataFrame that passes the real DataValidator."""
    dates = pd.bdate_range(
        start=pd.Timestamp.now() - pd.DateOffset(years=years),
        end=pd.Timestamp.now(),
        freq="B",
    )
    frames = []
    for sym in symbols:
        frames.append(pd.DataFrame({
            "symbol": sym,
            "trade_date": dates,
            "open_price": 150.0,
            "high_price": 155.0,
            "low_price": 148.0,
            "close_price": 152.0,
            "adjusted_close": 152.0,
            "volume": 1_000_000,
            "currency": "USD",
        }))
    return pd.concat(frames, ignore_index=True)


def _financials_df(symbols=("AAPL", "MSFT")):
    rows = []
    for sym in symbols:
        for yr, qtr in [(2023, 1), (2023, 2), (2024, 1), (2024, 2)]:
            rows.append({
                "symbol": sym,
                "fiscal_year": yr,
                "fiscal_quarter": qtr,
                "report_date": pd.Timestamp(f"{yr}-0{qtr * 3}-01"),
                "total_assets": 3e11,
                "book_equity": 5e10,
                "net_income": 2e10,
                "total_debt": 1e11,
                "shares_outstanding": 15e9,
                "eps": 1.5,
                "currency": "USD",
                "source": "edgar",
            })
    return pd.DataFrame(rows)


def _rates_df():
    return pd.DataFrame({
        "country": ["US"] * 12,
        "rate_date": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "rate": [5.25] * 12,
    })


def _make_ctx(prices=None, financials=None, rates=None, strict=True):
    """Build a PipelineContext using the REAL DataValidator and DataWriter.

    Only I/O boundaries (pg, mongo, minio, fetcher API calls) are mocked.
    """
    prices = prices if prices is not None else _prices_df()
    financials = financials if financials is not None else _financials_df()
    rates = rates if rates is not None else _rates_df()

    cfg = {
        "data": {"price_period": "5y", "fundamentals_period": "5y", "fundamentals_source": "waterfall"},
        "validation": {"min_price_rows": 5, "min_years": 1, "max_null_pct": 0.5, "strict": strict},
        "dev": {"enabled": False},
        "scheduler": {
            "prices_and_rates": {"day": 1, "hour": 2, "minute": 0},
            "fundamentals": {"month": "1,4,7,10", "day": 1, "hour": 4, "minute": 0},
        },
    }

    pg = MagicMock()
    pg.get_company_list.return_value = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"], "country": ["US", "US"]
    })
    pg.delete_symbols_missing_from_company_list.return_value = []
    # Return empty DataFrames so writer treats all rows as new
    pg.read_query.return_value = pd.DataFrame()

    mongo = MagicMock()
    minio = MagicMock()

    fetcher = MagicMock()
    fetcher.price_failures = {}
    fetcher.fundamentals_failures = {}
    fetcher.fetch_prices.return_value = prices
    fetcher.fetch_fundamentals.return_value = financials
    fetcher.fetch_risk_free_rates.return_value = rates

    # REAL validator and writer
    vcfg = cfg["validation"]
    validator = DataValidator(
        min_price_rows=vcfg["min_price_rows"],
        min_years=vcfg["min_years"],
        max_null_pct=vcfg["max_null_pct"],
    )
    writer = DataWriter(pg_conn=pg, mongo_conn=mongo, fetcher=fetcher)

    return PipelineContext(
        cfg=cfg,
        pg=pg,
        mongo=mongo,
        minio=minio,
        fetcher=fetcher,
        writer=writer,
        validator=validator,
        symbols=["AAPL", "MSFT"],
        countries=["US"],
        strict=strict,
    )


# ===================================================================
# E2E: real validator + real writer logic
# ===================================================================


class TestPipelineE2E:

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_valid_prices_pass_validator_and_reach_writer(self, _):
        """Real validator approves good data → writer.write_prices called."""
        ctx = _make_ctx()
        run_prices_and_rates(ctx)
        ctx.writer.pg.write_dataframe.assert_called()

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_empty_prices_fail_validator_and_skip_writer(self, _):
        """Real validator rejects empty DataFrame → writer never called."""
        ctx = _make_ctx(prices=pd.DataFrame())
        run_prices_and_rates(ctx)
        ctx.writer.pg.write_dataframe.assert_not_called()

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_negative_close_price_fails_validator(self, _):
        """Real validator errors on >1% zero/negative close prices."""
        bad_prices = _prices_df()
        bad_prices["close_price"] = -1.0
        ctx = _make_ctx(prices=bad_prices)
        run_prices_and_rates(ctx)
        ctx.writer.pg.write_dataframe.assert_not_called()

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_valid_financials_pass_validator_and_reach_writer(self, _):
        """Real validator approves good financials → writer.write_financials called."""
        ctx = _make_ctx()
        run_fundamentals(ctx)
        ctx.writer.pg.write_dataframe_on_conflict_do_nothing.assert_called()

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_empty_financials_fail_validator_and_skip_writer(self, _):
        """Real validator rejects empty financials → writer never called."""
        ctx = _make_ctx(financials=pd.DataFrame())
        run_fundamentals(ctx)
        ctx.writer.pg.write_dataframe_on_conflict_do_nothing.assert_not_called()

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_duplicate_prices_trigger_validation_error(self, _):
        """Real validator catches duplicate (symbol, trade_date) rows."""
        prices = _prices_df(symbols=("AAPL",))
        prices = pd.concat([prices, prices], ignore_index=True)  # deliberate dupes
        ctx = _make_ctx(prices=prices)
        run_prices_and_rates(ctx)
        ctx.writer.pg.write_dataframe.assert_not_called()

    @patch("main._load_universe", return_value=(["AAPL", "MSFT"], ["US"]))
    def test_non_strict_mode_writes_despite_warnings(self, _):
        """Non-strict mode: warnings don't halt writes."""
        # Short history (6 months) — generates a date-span warning but not an error
        dates = pd.bdate_range(
            start=pd.Timestamp.now() - pd.DateOffset(months=6),
            end=pd.Timestamp.now(),
        )
        short_prices = pd.concat([
            pd.DataFrame({"symbol": s, "trade_date": dates, "open_price": 150.0,
                          "high_price": 155.0, "low_price": 148.0, "close_price": 152.0,
                          "adjusted_close": 152.0, "volume": 1_000_000, "currency": "USD"})
            for s in ("AAPL", "MSFT")
        ], ignore_index=True)
        ctx = _make_ctx(prices=short_prices, strict=False)
        run_prices_and_rates(ctx)
        ctx.writer.pg.write_dataframe.assert_called()


# ===================================================================
# CLI: parse_args
# ===================================================================


class TestParseArgs:

    def test_defaults(self):
        args = parse_args([])
        assert args.task == "all"
        assert args.no_schedule is False
        assert args.run_date is None

    def test_task_prices(self):
        args = parse_args(["--task", "prices"])
        assert args.task == "prices"

    def test_task_fundamentals(self):
        args = parse_args(["--task", "fundamentals"])
        assert args.task == "fundamentals"

    def test_no_schedule_flag(self):
        args = parse_args(["--no-schedule"])
        assert args.no_schedule is True

    def test_run_date(self):
        args = parse_args(["--run-date", "2026-01-01"])
        assert args.run_date == "2026-01-01"

    def test_combined_args(self):
        args = parse_args(["--task", "prices", "--no-schedule", "--run-date", "2026-06-01"])
        assert args.task == "prices"
        assert args.no_schedule is True
        assert args.run_date == "2026-06-01"

    def test_invalid_task_raises(self):
        with pytest.raises(SystemExit):
            parse_args(["--task", "invalid"])


# ===================================================================
# CLI: main() honours --task and --no-schedule
# ===================================================================


class TestMainCLI:

    def _base_setup(self, mock_load_cfg, mock_pg_cls, mock_mongo_cls,
                    mock_minio_cls, mock_fetcher_cls, mock_validator_cls,
                    mock_writer_cls):
        """Wire up standard mocks used by multiple CLI tests."""
        from tests.test_main import _make_cfg, _make_universe, _make_prices, _make_financials, _make_rates, _passed

        mock_load_cfg.return_value = _make_cfg()

        mock_pg = MagicMock()
        mock_pg.test_connection.return_value = True
        mock_pg.get_company_list.return_value = _make_universe()
        mock_pg.delete_symbols_missing_from_company_list.return_value = []
        mock_pg_cls.return_value = mock_pg

        mock_mongo_cls.return_value.test_connection.return_value = True
        mock_minio_cls.return_value.test_connection.return_value = True

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

        return mock_fetcher, mock_validator, mock_writer

    @patch("main.BlockingScheduler")
    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_task_prices_skips_fundamentals_on_startup(
        self, mock_load_cfg, mock_pg_cls, mock_mongo_cls, mock_minio_cls,
        mock_fetcher_cls, mock_validator_cls, mock_writer_cls, mock_scheduler_cls,
    ):
        mock_fetcher, _, _ = self._base_setup(
            mock_load_cfg, mock_pg_cls, mock_mongo_cls,
            mock_minio_cls, mock_fetcher_cls, mock_validator_cls, mock_writer_cls,
        )
        mock_scheduler_cls.return_value = MagicMock()

        main(["--task", "prices"])

        mock_fetcher.fetch_prices.assert_called()
        mock_fetcher.fetch_fundamentals.assert_not_called()

    @patch("main.BlockingScheduler")
    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_task_fundamentals_skips_prices_on_startup(
        self, mock_load_cfg, mock_pg_cls, mock_mongo_cls, mock_minio_cls,
        mock_fetcher_cls, mock_validator_cls, mock_writer_cls, mock_scheduler_cls,
    ):
        mock_fetcher, _, _ = self._base_setup(
            mock_load_cfg, mock_pg_cls, mock_mongo_cls,
            mock_minio_cls, mock_fetcher_cls, mock_validator_cls, mock_writer_cls,
        )
        mock_scheduler_cls.return_value = MagicMock()

        main(["--task", "fundamentals"])

        mock_fetcher.fetch_fundamentals.assert_called()
        mock_fetcher.fetch_prices.assert_not_called()

    @patch("main.BlockingScheduler")
    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_no_schedule_does_not_start_scheduler(
        self, mock_load_cfg, mock_pg_cls, mock_mongo_cls, mock_minio_cls,
        mock_fetcher_cls, mock_validator_cls, mock_writer_cls, mock_scheduler_cls,
    ):
        self._base_setup(
            mock_load_cfg, mock_pg_cls, mock_mongo_cls,
            mock_minio_cls, mock_fetcher_cls, mock_validator_cls, mock_writer_cls,
        )
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        main(["--no-schedule"])

        mock_scheduler.start.assert_not_called()

    @patch("main.BlockingScheduler")
    @patch("main.DataWriter")
    @patch("main.DataValidator")
    @patch("main.DataFetcher")
    @patch("main.MinioConnection")
    @patch("main.MongoConnection")
    @patch("main.PostgresConnection")
    @patch("main.load_config")
    def test_default_task_runs_full_pipeline_and_starts_scheduler(
        self, mock_load_cfg, mock_pg_cls, mock_mongo_cls, mock_minio_cls,
        mock_fetcher_cls, mock_validator_cls, mock_writer_cls, mock_scheduler_cls,
    ):
        mock_fetcher, _, _ = self._base_setup(
            mock_load_cfg, mock_pg_cls, mock_mongo_cls,
            mock_minio_cls, mock_fetcher_cls, mock_validator_cls, mock_writer_cls,
        )
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        main([])

        mock_fetcher.fetch_prices.assert_called()
        mock_fetcher.fetch_fundamentals.assert_called()
        mock_scheduler.start.assert_called_once()
