"""Tests for benchmark.py — mocked yfinance and DB, no network calls."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.backtest.benchmark import (
    MSCI_USA_TICKER,
    backfill_benchmark_returns,
    fetch_benchmark_monthly_returns,
    load_benchmark_from_db,
)


def _make_raw(prices: dict) -> pd.DataFrame:
    """Build a minimal yf.download-style DataFrame with a Close column."""
    idx = pd.to_datetime(list(prices.keys()))
    close = pd.Series(list(prices.values()), index=idx, name="EUSA")
    return pd.DataFrame({"Close": close})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

START = date(2023, 1, 1)
END = date(2023, 3, 31)

# Two month-end prices → one monthly return
_TWO_MONTHS = _make_raw(
    {
        "2023-01-31": 100.0,
        "2023-02-28": 110.0,
    }
)

# Three month-end prices → two monthly returns
_THREE_MONTHS = _make_raw(
    {
        "2023-01-31": 100.0,
        "2023-02-28": 110.0,
        "2023-03-31": 99.0,
    }
)


# ---------------------------------------------------------------------------
# TestFetchBenchmarkMonthlyReturns
# ---------------------------------------------------------------------------


class TestFetchBenchmarkMonthlyReturns:

    def test_returns_correct_monthly_return(self):
        """100 → 110 gives a +10% monthly return."""
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            result = fetch_benchmark_monthly_returns(START, END)
        assert len(result) == 1
        assert pytest.approx(result.iloc[0], rel=1e-9) == 0.10

    def test_index_is_calendar_month_end_dates(self):
        """Index entries are datetime.date objects at calendar month-end."""
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            result = fetch_benchmark_monthly_returns(START, END)
        assert isinstance(result.index[0], date)
        assert result.index[0] == date(2023, 2, 28)

    def test_multiple_months_returns_correct_count(self):
        """Three month-end prices produce two monthly returns."""
        with patch(
            "modules.backtest.benchmark.yf.download", return_value=_THREE_MONTHS
        ):
            result = fetch_benchmark_monthly_returns(START, END)
        assert len(result) == 2

    def test_negative_return(self):
        """110 → 99 gives a negative monthly return."""
        with patch(
            "modules.backtest.benchmark.yf.download", return_value=_THREE_MONTHS
        ):
            result = fetch_benchmark_monthly_returns(START, END)
        assert result.iloc[-1] < 0
        assert pytest.approx(result.iloc[-1], rel=1e-9) == 99.0 / 110.0 - 1

    def test_empty_download_raises(self):
        """Empty yfinance response raises ValueError."""
        empty = pd.DataFrame()
        with patch("modules.backtest.benchmark.yf.download", return_value=empty):
            with pytest.raises(ValueError, match="No benchmark data"):
                fetch_benchmark_monthly_returns(START, END)

    def test_passes_correct_dates_to_yfinance(self):
        """start_date and end_date are forwarded as strings to yf.download."""
        with patch(
            "modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS
        ) as mock_dl:
            fetch_benchmark_monthly_returns(START, END)
        call_kwargs = mock_dl.call_args
        assert call_kwargs.kwargs["start"] == "2023-01-01"
        assert call_kwargs.kwargs["end"] == "2023-03-31"

    def test_returns_series(self):
        """Return type is a pandas Series."""
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            result = fetch_benchmark_monthly_returns(START, END)
        assert isinstance(result, pd.Series)

    def test_uses_eusa_ticker(self):
        """The MSCI USA ETF proxy ticker (EUSA) is passed to yfinance."""
        with patch(
            "modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS
        ) as mock_dl:
            fetch_benchmark_monthly_returns(START, END)
        args, _ = mock_dl.call_args
        assert args[0] == MSCI_USA_TICKER
        assert MSCI_USA_TICKER == "EUSA"

    def test_resamples_daily_to_month_end_last(self):
        """Daily prices are resampled to month-end using the LAST price of the month.

        If Jan prices ramp from 100 to 105 and Feb prices ramp from 105 to 110,
        the Feb return should be based on end-of-month values: 110 / 105 - 1.
        """
        dates = pd.bdate_range("2023-01-02", "2023-02-28")
        jan_dates = dates[dates.month == 1]
        feb_dates = dates[dates.month == 2]
        # Jan: linear ramp 100 -> 105, Feb: linear ramp 105 -> 110
        jan_prices = [100 + i * 5 / (len(jan_dates) - 1) for i in range(len(jan_dates))]
        feb_prices = [105 + i * 5 / (len(feb_dates) - 1) for i in range(len(feb_dates))]
        prices = pd.Series(jan_prices + feb_prices, index=jan_dates.append(feb_dates))
        fake_raw = pd.DataFrame({"Close": prices})

        with patch("modules.backtest.benchmark.yf.download", return_value=fake_raw):
            result = fetch_benchmark_monthly_returns(
                date(2023, 1, 1), date(2023, 2, 28)
            )

        # Single monthly return: Feb based on LAST prices (110 vs 105)
        assert len(result) == 1
        expected = 110.0 / 105.0 - 1
        assert pytest.approx(result.iloc[0], rel=1e-9) == expected


# ---------------------------------------------------------------------------
# TestBackfillBenchmarkReturns
# ---------------------------------------------------------------------------


class TestBackfillBenchmarkReturns:

    def test_writes_to_db_with_correct_columns(self):
        """Fetched returns are written to benchmark_returns with correct schema."""
        db = MagicMock()
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            count = backfill_benchmark_returns(db, START, END)

        assert count == 1
        db.write_dataframe_on_conflict_do_nothing.assert_called_once()
        args, kwargs = db.write_dataframe_on_conflict_do_nothing.call_args

        written_df = args[0]
        assert list(written_df.columns) == ["benchmark", "month_end", "monthly_return"]
        assert (written_df["benchmark"] == "EUSA").all()
        assert args[1] == "benchmark_returns"
        assert args[2] == "team_wittgenstein"
        assert kwargs["conflict_columns"] == ["benchmark", "month_end"]

    def test_idempotent_via_on_conflict(self):
        """Writes use ON CONFLICT DO NOTHING so re-runs don't duplicate rows."""
        db = MagicMock()
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            backfill_benchmark_returns(db, START, END)

        # Verify it's the on-conflict variant, not plain write
        db.write_dataframe.assert_not_called()
        db.write_dataframe_on_conflict_do_nothing.assert_called_once()

    def test_respects_custom_benchmark_label(self):
        """Custom benchmark label is used as the ticker column value."""
        db = MagicMock()
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            backfill_benchmark_returns(db, START, END, benchmark="SPY")

        args, _ = db.write_dataframe_on_conflict_do_nothing.call_args
        written_df = args[0]
        assert (written_df["benchmark"] == "SPY").all()


# ---------------------------------------------------------------------------
# TestLoadBenchmarkFromDb
# ---------------------------------------------------------------------------


class TestLoadBenchmarkFromDb:

    def test_returns_series_keyed_by_date(self):
        """DB rows are returned as a Series indexed by date."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame(
            {
                "month_end": [date(2023, 1, 31), date(2023, 2, 28)],
                "monthly_return": [0.02, -0.01],
            }
        )
        result = load_benchmark_from_db(db, START, END)

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert result.iloc[0] == 0.02
        assert result.index[0] == date(2023, 1, 31)

    def test_empty_db_returns_empty_series(self):
        """No matching rows returns an empty Series."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()
        result = load_benchmark_from_db(db, START, END)

        assert isinstance(result, pd.Series)
        assert result.empty

    def test_passes_correct_query_params(self):
        """Query parameters match the requested date range and benchmark."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()
        load_benchmark_from_db(db, START, END, benchmark="EUSA")

        _, kwargs = db.read_query.call_args
        params = kwargs.get("params") or db.read_query.call_args.args[1]
        assert params["benchmark"] == "EUSA"
        assert params["start_date"] == START
        assert params["end_date"] == END
