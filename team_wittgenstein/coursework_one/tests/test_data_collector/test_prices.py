"""Tests for price fetching, reshaping, currency extraction, and classification."""

from unittest.mock import MagicMock, patch

import pandas as pd

from modules.input.data_collector import DataFetcher

# ===================================================================
# _reshape_price_df
# ===================================================================


class TestReshapePriceDf:

    def test_valid_transform(self):
        raw = pd.DataFrame(
            {
                "Date": pd.bdate_range("2024-01-01", periods=5),
                "Open": [100.0] * 5,
                "High": [105.0] * 5,
                "Low": [98.0] * 5,
                "Close": [102.0] * 5,
                "Adj Close": [102.0] * 5,
                "Volume": [1_000_000] * 5,
            }
        ).set_index("Date")

        result = DataFetcher._reshape_price_df(raw, "AAPL")
        assert result is not None
        assert "symbol" in result.columns
        assert "trade_date" in result.columns
        assert "close_price" in result.columns
        assert "currency" in result.columns
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
# _classify_missing
# ===================================================================


class TestClassifyMissing:

    @patch("modules.input.data_collector.utils.yf")
    def test_delisted(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": None}
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._classify_missing(["ABMD"])
        assert "ABMD" in result["delisted"]
        assert result["fetch_error"] == []

    @patch("modules.input.data_collector.utils.yf")
    def test_fetch_error(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 150.0}
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._classify_missing(["AAPL"])
        assert "AAPL" in result["fetch_error"]
        assert result["delisted"] == []

    @patch("modules.input.data_collector.utils.yf")
    def test_exception(self, mock_yf, fetcher):
        mock_yf.Ticker.side_effect = Exception("API error")

        result = fetcher._classify_missing(["BAD"])
        assert "BAD" in result["fetch_error"]


# ===================================================================
# fetch_prices
# ===================================================================


class TestFetchPrices:

    def test_all_cached(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = True
        cached_df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 5,
                "trade_date": pd.bdate_range("2024-01-01", periods=5),
                "close_price": [150.0] * 5,
            }
        )
        mock_minio_conn.download_dataframe.return_value = cached_df

        with patch("modules.input.data_collector.prices.yf") as mock_yf:
            result = fetcher.fetch_prices(["AAPL"])
            mock_yf.download.assert_not_called()

        assert len(result) == 5

    @patch("modules.input.data_collector.prices.yf")
    def test_downloads_uncached(self, mock_yf, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = False

        raw = pd.DataFrame(
            {
                "Date": pd.bdate_range("2024-01-01", periods=5),
                "Open": [100.0] * 5,
                "High": [105.0] * 5,
                "Low": [98.0] * 5,
                "Close": [102.0] * 5,
                "Adj Close": [102.0] * 5,
                "Volume": [1e6] * 5,
            }
        ).set_index("Date")
        mock_yf.download.return_value = raw

        result = fetcher.fetch_prices(["AAPL"])
        mock_yf.download.assert_called_once()
        assert len(result) > 0
        assert "symbol" in result.columns


# ===================================================================
# _get_price_currency
# ===================================================================


class TestGetPriceCurrency:

    @patch("modules.input.data_collector.prices.yf")
    def test_from_fast_info_dict(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.fast_info = {"currency": "USD"}
        mock_yf.Ticker.return_value = mock_ticker
        assert fetcher._get_price_currency("AAPL") == "USD"

    @patch("modules.input.data_collector.prices.yf")
    def test_from_fast_info_attr(self, mock_yf, fetcher):
        fi = MagicMock()
        fi.currency = "GBP"
        mock_ticker = MagicMock()
        mock_ticker.fast_info = fi
        mock_yf.Ticker.return_value = mock_ticker
        assert fetcher._get_price_currency("VOD.L") == "GBP"

    @patch("modules.input.data_collector.prices.yf")
    def test_fallback_to_info(self, mock_yf, fetcher):
        fi = MagicMock(spec=[])  # no currency attr
        mock_ticker = MagicMock()
        mock_ticker.fast_info = fi
        mock_ticker.info = {"currency": "EUR"}
        mock_yf.Ticker.return_value = mock_ticker
        assert fetcher._get_price_currency("SAP") == "EUR"

    @patch("modules.input.data_collector.prices.yf")
    def test_exception_returns_none(self, mock_yf, fetcher):
        mock_yf.Ticker.side_effect = Exception("fail")
        assert fetcher._get_price_currency("BAD") is None


# ===================================================================
# _batch_download_prices (multi-symbol branch)
# ===================================================================


class TestBatchDownloadPricesMulti:

    @patch("modules.input.data_collector.prices.yf")
    def test_multi_symbol(self, mock_yf, fetcher, mock_minio_conn):
        dates = pd.bdate_range("2024-01-01", periods=5)
        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        symbols = ["AAPL", "MSFT"]
        # yfinance multi-symbol: top level = price field, second = ticker
        mi = pd.MultiIndex.from_product([cols, symbols])
        data = [[100, 200, 105, 210, 98, 195, 102, 205, 102, 205, 1e6, 2e6]] * 5
        raw = pd.DataFrame(data, index=dates, columns=mi)
        raw.index.name = "Date"
        # Swap levels so raw[symbol] works (ticker on top)
        raw = raw.swaplevel(axis=1).sort_index(axis=1)
        mock_yf.download.return_value = raw
        mock_yf.Ticker.return_value = MagicMock(fast_info={"currency": "USD"})

        result = fetcher._batch_download_prices(["AAPL", "MSFT"], "5y")
        assert len(result) == 2

    @patch("modules.input.data_collector.prices.yf")
    def test_single_symbol(self, mock_yf, fetcher, mock_minio_conn):
        dates = pd.bdate_range("2024-01-01", periods=5)
        raw = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [105.0] * 5,
                "Low": [98.0] * 5,
                "Close": [102.0] * 5,
                "Adj Close": [102.0] * 5,
                "Volume": [1e6] * 5,
            },
            index=dates,
        )
        raw.index.name = "Date"
        mock_yf.download.return_value = raw
        mock_yf.Ticker.return_value = MagicMock(fast_info={"currency": "USD"})

        result = fetcher._batch_download_prices(["AAPL"], "5y")
        assert len(result) == 1

    @patch("modules.input.data_collector.prices.yf")
    def test_keyerror_skips_symbol(self, mock_yf, fetcher, mock_minio_conn):
        raw = MagicMock()
        raw.__getitem__ = MagicMock(side_effect=KeyError("BADTICKER"))
        mock_yf.download.return_value = raw
        result = fetcher._batch_download_prices(["AAPL", "BADTICKER"], "5y")
        assert len(result) == 0


# ===================================================================
# fetch_prices edge cases
# ===================================================================


class TestFetchPricesEdgeCases:

    def test_empty_result(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = False
        with patch("modules.input.data_collector.prices.yf") as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            result = fetcher.fetch_prices(["AAPL"])
        assert result.empty

    def test_classify_missing_called(self, fetcher, mock_minio_conn):
        """Missing symbols trigger _classify_missing."""
        mock_minio_conn.object_exists.return_value = False
        # Return a df with only AAPL data via _batch_download_prices
        aapl_df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 5,
                "trade_date": pd.bdate_range("2024-01-01", periods=5),
                "close_price": [102.0] * 5,
            }
        )
        with patch.object(
            fetcher, "_batch_download_prices", return_value=[aapl_df]
        ), patch.object(fetcher, "_classify_missing") as mock_classify:
            mock_classify.return_value = {
                "delisted": [],
                "fetch_error": ["MSFT"],
            }
            fetcher.fetch_prices(["AAPL", "MSFT"])
            mock_classify.assert_called_once()


# ===================================================================
# _batch_download_prices exception handling
# ===================================================================


class TestBatchDownloadPricesExceptions:

    @patch("modules.input.data_collector.prices.yf")
    def test_generic_exception_during_processing(
        self, mock_yf, fetcher, mock_minio_conn
    ):
        """Exception during symbol processing is caught (line 106-107)."""
        raw = MagicMock()
        raw.__getitem__ = MagicMock(side_effect=RuntimeError("bad data"))
        raw.columns = pd.MultiIndex.from_tuples(
            [("Close", "AAPL")], names=["Price", "Ticker"]
        )
        mock_yf.download.return_value = raw
        result = fetcher._batch_download_prices(["AAPL"], "5y")
        assert result == []
