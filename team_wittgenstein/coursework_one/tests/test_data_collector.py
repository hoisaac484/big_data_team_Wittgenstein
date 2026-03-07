"""Tests for modules.input.data_collector."""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from modules.input.data_collector import DataFetcher


@pytest.fixture
def fetcher(mock_minio_conn):
    """DataFetcher with mocked MinIO."""
    return DataFetcher(mock_minio_conn)


# ===================================================================
# _reshape_price_df
# ===================================================================

class TestReshapePriceDf:

    def test_valid_transform(self):
        raw = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-01", periods=5),
            "Open": [100.0] * 5,
            "High": [105.0] * 5,
            "Low": [98.0] * 5,
            "Close": [102.0] * 5,
            "Adj Close": [102.0] * 5,
            "Volume": [1_000_000] * 5,
        }).set_index("Date")

        result = DataFetcher._reshape_price_df(raw, "AAPL")
        assert result is not None
        assert "symbol" in result.columns
        assert "trade_date" in result.columns
        assert "close_price" in result.columns
        assert "source" in result.columns
        assert (result["symbol"] == "AAPL").all()
        assert len(result) == 5

    def test_multiindex_columns(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        arrays = [
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            ["AAPL"] * 6,
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        data = [[100, 105, 98, 102, 102, 1e6]] * 3
        raw = pd.DataFrame(data, index=dates, columns=index)
        raw.index.name = "Date"

        result = DataFetcher._reshape_price_df(raw, "AAPL")
        assert result is not None
        assert "close_price" in result.columns

    def test_none_input(self):
        assert DataFetcher._reshape_price_df(None, "AAPL") is None

    def test_empty_input(self):
        assert DataFetcher._reshape_price_df(pd.DataFrame(), "AAPL") is None


# ===================================================================
# _safe_get
# ===================================================================

class TestSafeGet:

    def test_existing_value(self):
        df = pd.DataFrame(
            {"Q1": [1e9, 2e9]},
            index=["Total Assets", "Total Equity"],
        )
        val = DataFetcher._safe_get(df, "Total Assets", "Q1")
        assert val == 1e9

    def test_missing_key(self):
        df = pd.DataFrame({"Q1": [1e9]}, index=["Total Assets"])
        assert DataFetcher._safe_get(df, "Missing Row", "Q1") is None

    def test_nan_value(self):
        df = pd.DataFrame({"Q1": [np.nan]}, index=["Total Assets"])
        assert DataFetcher._safe_get(df, "Total Assets", "Q1") is None


# ===================================================================
# _classify_missing
# ===================================================================

class TestClassifyMissing:

    @patch("modules.input.data_collector.yf")
    def test_delisted(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": None}
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._classify_missing(["ABMD"])
        assert "ABMD" in result["delisted"]
        assert result["fetch_error"] == []

    @patch("modules.input.data_collector.yf")
    def test_fetch_error(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 150.0}
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._classify_missing(["AAPL"])
        assert "AAPL" in result["fetch_error"]
        assert result["delisted"] == []

    @patch("modules.input.data_collector.yf")
    def test_exception(self, mock_yf, fetcher):
        mock_yf.Ticker.side_effect = Exception("API error")

        result = fetcher._classify_missing(["BAD"])
        assert "BAD" in result["fetch_error"]


# ===================================================================
# _is_cached
# ===================================================================

class TestIsCached:

    def test_cached_true(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = True
        assert fetcher._is_cached("prices", "AAPL") is True

    def test_cached_false_missing_ctl(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.side_effect = [True, False]
        assert fetcher._is_cached("prices", "AAPL") is False


# ===================================================================
# fetch_prices
# ===================================================================

class TestFetchPrices:

    def test_all_cached(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = True
        cached_df = pd.DataFrame({
            "symbol": ["AAPL"] * 5,
            "trade_date": pd.bdate_range("2024-01-01", periods=5),
            "close_price": [150.0] * 5,
        })
        mock_minio_conn.download_dataframe.return_value = cached_df

        with patch("modules.input.data_collector.yf") as mock_yf:
            result = fetcher.fetch_prices(["AAPL"])
            mock_yf.download.assert_not_called()

        assert len(result) == 5

    @patch("modules.input.data_collector.yf")
    def test_downloads_uncached(self, mock_yf, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = False

        raw = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-01", periods=5),
            "Open": [100.0] * 5,
            "High": [105.0] * 5,
            "Low": [98.0] * 5,
            "Close": [102.0] * 5,
            "Adj Close": [102.0] * 5,
            "Volume": [1e6] * 5,
        }).set_index("Date")
        mock_yf.download.return_value = raw

        result = fetcher.fetch_prices(["AAPL"])
        mock_yf.download.assert_called_once()
        assert len(result) > 0
        assert "symbol" in result.columns


# ===================================================================
# fetch_fundamentals
# ===================================================================

class TestFetchFundamentals:

    def test_parallel_fetch(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = False

        fake_df = pd.DataFrame({
            "symbol": ["AAPL"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
            "total_assets": [3e11],
        })

        with patch.object(fetcher, "_fetch_single_fundamental", return_value=fake_df):
            result = fetcher.fetch_fundamentals(["AAPL"], max_workers=1)

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"


# ===================================================================
# _fetch_single_fundamental
# ===================================================================

class TestCacheHelpers:

    def test_ctl_path(self, fetcher):
        assert fetcher._ctl_path("prices", "AAPL") == "prices/AAPL.ctl"

    def test_parquet_path(self, fetcher):
        assert fetcher._parquet_path("prices", "AAPL") == "prices/AAPL.parquet"

    def test_write_ctl(self, fetcher, mock_minio_conn):
        fetcher._write_ctl("prices", "AAPL", 100, "yfinance")
        mock_minio_conn.upload_json.assert_called_once()
        call_args = mock_minio_conn.upload_json.call_args[0]
        assert call_args[1] == "prices/AAPL.ctl"
        assert call_args[2]["rows"] == 100

    def test_cache_dataframe(self, fetcher, mock_minio_conn):
        df = pd.DataFrame({"col": [1, 2]})
        fetcher._cache_dataframe("prices", "AAPL", df, "yfinance")
        mock_minio_conn.upload_dataframe.assert_called_once()
        mock_minio_conn.upload_json.assert_called_once()

    def test_load_cached(self, fetcher, mock_minio_conn):
        mock_minio_conn.download_dataframe.return_value = pd.DataFrame({"col": [1]})
        result = fetcher._load_cached("prices", "AAPL")
        assert len(result) == 1

    def test_mark_loaded(self, fetcher, mock_minio_conn):
        mock_minio_conn.download_json.return_value = {
            "name": "AAPL", "loaded_to_postgres": False
        }
        fetcher.mark_loaded("prices", "AAPL")
        mock_minio_conn.upload_json.assert_called_once()
        uploaded = mock_minio_conn.upload_json.call_args[0][2]
        assert uploaded["loaded_to_postgres"] is True

    def test_mark_loaded_no_ctl(self, fetcher, mock_minio_conn):
        mock_minio_conn.download_json.return_value = None
        fetcher.mark_loaded("prices", "AAPL")
        mock_minio_conn.upload_json.assert_not_called()


class TestFetchSingleFundamental:

    @patch("modules.input.data_collector.yf")
    def test_extracts_data(self, mock_yf, fetcher):
        date_col = pd.Timestamp("2024-03-31")
        bs = pd.DataFrame(
            {date_col: [3e11, 1e11, 5e10, 1e11, 15e9]},
            index=[
                "Total Assets",
                "Stockholders Equity",
                "Total Debt",
                "Ordinary Shares Number",
                "Share Issued",
            ],
        )
        inc = pd.DataFrame(
            {date_col: [2e10, 1.30]},
            index=["Net Income", "Basic EPS"],
        )
        mock_ticker = MagicMock()
        type(mock_ticker).quarterly_balance_sheet = PropertyMock(return_value=bs)
        type(mock_ticker).quarterly_financials = PropertyMock(return_value=inc)
        mock_ticker.info = {"currency": "USD", "sharesOutstanding": 15e9}
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._fetch_single_fundamental("AAPL")
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["total_assets"] == 3e11
        assert result.iloc[0]["net_income"] == 2e10
