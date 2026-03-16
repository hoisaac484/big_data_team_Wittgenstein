"""Tests for fundamentals fetching, waterfall merge, forward-fill."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.input.data_collector import DataFetcher
from modules.input.data_collector.constants import SimFinServerError

# ===================================================================
# fetch_fundamentals
# ===================================================================


class TestFetchFundamentals:

    def test_fetch(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = False

        fake_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "total_assets": [3e11],
            }
        )

        with patch.object(fetcher, "_fetch_single_fundamental", return_value=fake_df):
            result = fetcher.fetch_fundamentals(["AAPL"], period="1y", source="simfin")

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"


# ===================================================================
# _fetch_single_fundamental
# ===================================================================


class TestFetchSingleFundamental:

    def test_simfin_source(self, fetcher):
        fake_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "total_debt": [1e11],
                "net_income": [2e10],
                "book_equity": [1e11],
                "shares_outstanding": [15_000_000_000],
                "eps": [1.30],
                "currency": ["USD"],
                "source": ["simfin"],
            }
        )

        with patch.object(fetcher, "_fetch_simfin_fundamentals", return_value=fake_df):
            result = fetcher._fetch_single_fundamental(
                "AAPL", period="5y", source="simfin"
            )

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["total_assets"] == 3e11
        assert result.iloc[0]["net_income"] == 2e10

    def test_waterfall_source_delegates(self, fetcher):
        fake_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "source": ["edgar"],
            }
        )

        with patch.object(
            fetcher, "_fetch_waterfall_fundamentals", return_value=fake_df
        ) as mock_wf:
            result = fetcher._fetch_single_fundamental(
                "AAPL", period="5y", source="waterfall"
            )
            mock_wf.assert_called_once_with("AAPL", "5y")

        assert result is not None
        assert len(result) == 1


# ===================================================================
# _normalize_fundamentals_source
# ===================================================================


class TestNormalizeFundamentalsSource:

    def test_waterfall(self):
        assert DataFetcher._normalize_fundamentals_source("waterfall") == "waterfall"

    def test_simfin(self):
        assert DataFetcher._normalize_fundamentals_source("simfin") == "simfin"

    def test_default_is_waterfall(self):
        assert DataFetcher._normalize_fundamentals_source(None) == "waterfall"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            DataFetcher._normalize_fundamentals_source("bloomberg")


# ===================================================================
# _fetch_yfinance_fundamentals
# ===================================================================


class TestFetchYfinanceFundamentals:

    @patch("modules.input.data_collector.yfinance_fundamentals.yf")
    def test_happy_path(self, mock_yf, fetcher):
        dates = [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")]

        bs = pd.DataFrame(
            {"2024-03-31": [3e11, 1e11, 1e11], "2024-06-30": [3.2e11, 1.1e11, 1.2e11]},
            index=["Total Assets", "Total Debt", "Stockholders Equity"],
        )
        bs.columns = dates

        inc = pd.DataFrame(
            {"2024-03-31": [2e10, 1.3], "2024-06-30": [2.2e10, 1.4]},
            index=["Net Income", "Diluted EPS"],
        )
        inc.columns = dates

        mock_ticker = MagicMock()
        mock_ticker.quarterly_balance_sheet = bs
        mock_ticker.quarterly_income_stmt = inc
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._fetch_yfinance_fundamentals("AAPL")
        assert not result.empty
        assert len(result) == 2
        assert result.iloc[0]["source"] == "yfinance"
        assert "total_assets" in result.columns
        assert "net_income" in result.columns

    @patch("modules.input.data_collector.yfinance_fundamentals.yf")
    def test_empty_returns_empty(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.quarterly_income_stmt = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._fetch_yfinance_fundamentals("AAPL")
        assert result.empty

    @patch("modules.input.data_collector.yfinance_fundamentals.yf")
    def test_exception_returns_empty(self, mock_yf, fetcher):
        mock_yf.Ticker.side_effect = Exception("API error")
        result = fetcher._fetch_yfinance_fundamentals("AAPL")
        assert result.empty

    @patch("modules.input.data_collector.yfinance_fundamentals.yf")
    def test_income_stmt_creates_new_records(self, mock_yf, fetcher):
        """Income stmt with dates not in bs creates new records (line 76)."""
        inc = pd.DataFrame(
            {"2024-09-30": [2.5e10, 1.6]},
            index=["Net Income", "Diluted EPS"],
        )
        inc.columns = [pd.Timestamp("2024-09-30")]
        mock_ticker = MagicMock()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.quarterly_income_stmt = inc
        mock_yf.Ticker.return_value = mock_ticker
        result = fetcher._fetch_yfinance_fundamentals("AAPL")
        assert len(result) == 1
        assert result.iloc[0]["net_income"] == 2.5e10

    @patch("modules.input.data_collector.yfinance_fundamentals.yf")
    def test_bs_no_matching_items_empty_records(self, mock_yf, fetcher):
        """BS with no matching index items produces empty records (line 91)."""
        bs = pd.DataFrame(
            {"2024-03-31": [999]},
            index=["SomeRandomField"],
        )
        bs.columns = [pd.Timestamp("2024-03-31")]
        mock_ticker = MagicMock()
        mock_ticker.quarterly_balance_sheet = bs
        mock_ticker.quarterly_income_stmt = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker
        result = fetcher._fetch_yfinance_fundamentals("AAPL")
        # Record is created but has no matching fields,
        # so it's still returned (with symbol, year, quarter)
        assert len(result) == 1


# ===================================================================
# _merge_waterfall
# ===================================================================


class TestMergeWaterfall:

    def test_fills_nulls_from_lower_priority(self, fetcher):
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "net_income": [None],
                "book_equity": [1e11],
                "currency": ["USD"],
                "source": ["edgar"],
            }
        )
        simfin_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [2.9e11],  # lower priority, should not override
                "net_income": [2e10],  # fills null from edgar
                "book_equity": [9e10],  # lower priority, should not override
                "currency": ["USD"],
                "source": ["simfin"],
            }
        )

        result = fetcher._merge_waterfall([("edgar", edgar_df), ("simfin", simfin_df)])
        assert len(result) == 1
        row = result.iloc[0]
        assert row["total_assets"] == 3e11  # kept from edgar
        assert row["net_income"] == 2e10  # filled from simfin
        assert row["book_equity"] == 1e11  # kept from edgar
        assert row["source"] == "edgar"

    def test_adds_new_quarters_from_lower_priority(self, fetcher):
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "source": ["edgar"],
            }
        )
        simfin_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [2],  # new quarter
                "report_date": [pd.Timestamp("2024-06-30")],
                "total_assets": [3.1e11],
                "source": ["simfin"],
            }
        )

        result = fetcher._merge_waterfall([("edgar", edgar_df), ("simfin", simfin_df)])
        assert len(result) == 2

    def test_single_source(self, fetcher):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "total_assets": [3e11],
                "source": ["edgar"],
            }
        )
        result = fetcher._merge_waterfall([("edgar", df)])
        assert len(result) == 1


# ===================================================================
# _forward_fill_fundamentals
# ===================================================================


class TestForwardFillFundamentals:

    def test_fills_from_previous_quarter(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "fiscal_year": [2024, 2024],
                "fiscal_quarter": [1, 2],
                "total_assets": [3e11, None],
                "net_income": [2e10, 2.1e10],
            }
        )
        result = DataFetcher._forward_fill_fundamentals(df)
        assert result.iloc[1]["total_assets"] == 3e11  # forward filled
        assert result.iloc[1]["net_income"] == 2.1e10  # not overwritten

    def test_no_fill_across_symbols(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "fiscal_year": [2024, 2024],
                "fiscal_quarter": [1, 1],
                "total_assets": [3e11, None],
            }
        )
        result = DataFetcher._forward_fill_fundamentals(df)
        assert pd.isna(result[result["symbol"] == "MSFT"].iloc[0]["total_assets"])

    def test_handles_empty(self):
        assert DataFetcher._forward_fill_fundamentals(None) is None
        result = DataFetcher._forward_fill_fundamentals(pd.DataFrame())
        assert result.empty

    def test_caps_at_2_quarters(self):
        """Forward fill should stop after 2 consecutive nulls."""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 4,
                "fiscal_year": [2024, 2024, 2024, 2024],
                "fiscal_quarter": [1, 2, 3, 4],
                "total_assets": [3e11, None, None, None],
            }
        )
        result = DataFetcher._forward_fill_fundamentals(df)
        assert result.iloc[1]["total_assets"] == 3e11  # filled (1st)
        assert result.iloc[2]["total_assets"] == 3e11  # filled (2nd)
        assert pd.isna(result.iloc[3]["total_assets"])  # NOT filled (3rd, beyond limit)


# ===================================================================
# _fetch_waterfall_fundamentals (integration)
# ===================================================================


class TestFetchWaterfallFundamentals:

    def test_uses_all_sources(self, fetcher):
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "net_income": [None],
                "book_equity": [None],
                "currency": ["USD"],
                "source": ["edgar"],
            }
        )
        simfin_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "net_income": [2e10],
                "book_equity": [None],
                "currency": ["USD"],
                "source": ["simfin"],
            }
        )
        yf_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "book_equity": [1e11],
                "currency": ["USD"],
                "source": ["yfinance"],
            }
        )

        with patch.object(
            fetcher, "_fetch_edgar_fundamentals", return_value=edgar_df
        ), patch.object(
            fetcher, "_fetch_simfin_fundamentals", return_value=simfin_df
        ), patch.object(
            fetcher, "_fetch_yfinance_fundamentals", return_value=yf_df
        ):
            result = fetcher._fetch_waterfall_fundamentals("AAPL", "5y")

        assert result is not None
        row = result.iloc[0]
        assert row["total_assets"] == 3e11  # from edgar
        assert row["net_income"] == 2e10  # from simfin
        assert row["book_equity"] == 1e11  # from yfinance

    def test_all_sources_empty(self, fetcher):
        with patch.object(
            fetcher, "_fetch_edgar_fundamentals", return_value=pd.DataFrame()
        ), patch.object(
            fetcher, "_fetch_simfin_fundamentals", return_value=pd.DataFrame()
        ), patch.object(
            fetcher, "_fetch_yfinance_fundamentals", return_value=pd.DataFrame()
        ):
            result = fetcher._fetch_waterfall_fundamentals("AAPL", "5y")

        assert result is None

    def test_edgar_only(self, fetcher):
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "net_income": [2e10],
                "book_equity": [1e11],
                "currency": ["USD"],
                "source": ["edgar"],
            }
        )

        with patch.object(
            fetcher, "_fetch_edgar_fundamentals", return_value=edgar_df
        ), patch.object(
            fetcher, "_fetch_simfin_fundamentals", return_value=pd.DataFrame()
        ), patch.object(
            fetcher, "_fetch_yfinance_fundamentals", return_value=pd.DataFrame()
        ):
            result = fetcher._fetch_waterfall_fundamentals("AAPL", "5y")

        assert result is not None
        assert len(result) == 1

    def test_simfin_server_error_skipped(self, fetcher):
        """SimFin 500 during waterfall should be skipped gracefully."""
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "source": ["edgar"],
            }
        )

        with patch.object(
            fetcher, "_fetch_edgar_fundamentals", return_value=edgar_df
        ), patch.object(
            fetcher,
            "_fetch_simfin_fundamentals",
            side_effect=SimFinServerError("500"),
        ), patch.object(
            fetcher,
            "_fetch_yfinance_fundamentals",
            return_value=pd.DataFrame(),
        ):
            result = fetcher._fetch_waterfall_fundamentals("AAPL", "5y")

        assert result is not None
        assert len(result) == 1


# ===================================================================
# fetch_fundamentals (orchestration)
# ===================================================================


class TestFetchFundamentalsOrchestration:

    def test_cached_fundamentals(self, fetcher, mock_minio_conn):
        """Cached fundamentals are loaded without re-fetching."""
        cached_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "source": ["edgar"],
            }
        )
        with patch.object(fetcher, "_is_cached", return_value=True), patch.object(
            fetcher, "_load_cached", return_value=cached_df
        ):
            result = fetcher.fetch_fundamentals(
                ["AAPL"], period="5y", source="waterfall"
            )
        assert len(result) == 1

    def test_no_data_returns_empty(self, fetcher, mock_minio_conn):
        """Returns empty DataFrame when all symbols fail."""
        with patch.object(fetcher, "_is_cached", return_value=False), patch.object(
            fetcher,
            "_fetch_single_fundamental",
            return_value=pd.DataFrame(),
        ), patch.object(
            fetcher,
            "_classify_missing",
            return_value={"delisted": ["ZZZZ"], "fetch_error": []},
        ):
            result = fetcher.fetch_fundamentals(
                ["ZZZZ"], period="5y", source="waterfall"
            )
        assert result.empty

    def test_sequential_fetch_exception(self, fetcher, mock_minio_conn):
        """Exception during single-symbol fetch is caught."""
        with patch.object(fetcher, "_is_cached", return_value=False), patch.object(
            fetcher,
            "_fetch_single_fundamental",
            side_effect=Exception("network fail"),
        ), patch.object(
            fetcher,
            "_classify_missing",
            return_value={"delisted": [], "fetch_error": ["AAPL"]},
        ):
            result = fetcher.fetch_fundamentals(
                ["AAPL"], period="5y", source="waterfall"
            )
        assert result.empty


# ===================================================================
# _fetch_single_fundamental edge cases
# ===================================================================


class TestFetchSingleFundamentalEdgeCases:

    def test_simfin_server_error(self, fetcher):
        """SimFinServerError is caught when source=simfin."""
        with patch.object(
            fetcher,
            "_fetch_simfin_fundamentals",
            side_effect=SimFinServerError("500"),
        ):
            result = fetcher._fetch_single_fundamental(
                "AAPL", period="5y", source="simfin"
            )
        assert result is None

    def test_simfin_empty_returns_none(self, fetcher):
        """Empty simfin result returns None."""
        with patch.object(
            fetcher,
            "_fetch_simfin_fundamentals",
            return_value=pd.DataFrame(),
        ):
            result = fetcher._fetch_single_fundamental(
                "AAPL", period="5y", source="simfin"
            )
        assert result is None


# ===================================================================
# _merge_waterfall — field not in primary source
# ===================================================================


class TestMergeWaterfallFieldMissing:

    def test_field_only_in_secondary_source(self, fetcher):
        """Field absent from primary but present in secondary (line 302)."""
        primary = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "total_assets": [3e11],
            }
        )
        secondary = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "total_assets": [2.9e11],
                "eps": [1.5],
            }
        )
        result = fetcher._merge_waterfall([("edgar", primary), ("simfin", secondary)])
        assert "eps" in result.columns
        assert result.iloc[0]["eps"] == 1.5


# ===================================================================
# _fetch_waterfall_fundamentals — empty merged result
# ===================================================================


class TestFetchWaterfallEmpty:

    def test_all_sources_empty_returns_none(self, fetcher):
        """All sources return empty → waterfall returns None (line 251)."""
        with patch.object(
            fetcher, "_fetch_edgar_fundamentals", return_value=pd.DataFrame()
        ), patch.object(
            fetcher,
            "_fetch_simfin_fundamentals",
            return_value=pd.DataFrame(),
        ), patch.object(
            fetcher,
            "_fetch_yfinance_fundamentals",
            return_value=pd.DataFrame(),
        ):
            result = fetcher._fetch_waterfall_fundamentals("AAPL", "5y")
        assert result is None
