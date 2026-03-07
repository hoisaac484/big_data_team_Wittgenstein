"""Tests for modules.output.data_writer."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from modules.output.data_writer import DataWriter


@pytest.fixture
def writer(mock_pg_conn, mock_mongo_conn):
    """DataWriter with mocked connections and no fetcher."""
    return DataWriter(pg_conn=mock_pg_conn, mongo_conn=mock_mongo_conn)


@pytest.fixture
def writer_with_fetcher(mock_pg_conn, mock_mongo_conn):
    """DataWriter with a mocked fetcher for mark_loaded tests."""
    fetcher = MagicMock()
    return DataWriter(pg_conn=mock_pg_conn, mongo_conn=mock_mongo_conn, fetcher=fetcher)


# ===================================================================
# write_prices
# ===================================================================

class TestWritePrices:

    def test_happy_path(self, writer, mock_pg_conn):
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 3,
            "trade_date": pd.bdate_range("2024-01-01", periods=3),
            "close_price": [150.0] * 3,
        })
        count = writer.write_prices(df)
        assert count == 3
        mock_pg_conn.write_dataframe.assert_called_once()

    def test_skips_duplicates(self, writer, mock_pg_conn):
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close_price": [150.0, 151.0],
        })
        existing = pd.DataFrame({
            "symbol": ["AAPL"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
        })
        mock_pg_conn.read_query.return_value = existing
        count = writer.write_prices(df)
        assert count == 1

    def test_empty_df(self, writer, mock_pg_conn):
        assert writer.write_prices(None) == 0
        assert writer.write_prices(pd.DataFrame()) == 0
        mock_pg_conn.write_dataframe.assert_not_called()

    def test_marks_loaded(self, writer_with_fetcher):
        df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "close_price": [150.0, 300.0],
        })
        writer_with_fetcher.write_prices(df)
        assert writer_with_fetcher.fetcher.mark_loaded.call_count == 2


# ===================================================================
# write_financials
# ===================================================================

class TestWriteFinancials:

    def test_happy_path(self, writer, mock_pg_conn):
        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
            "total_assets": [3e11],
        })
        count = writer.write_financials(df)
        assert count == 1
        mock_pg_conn.write_dataframe.assert_called_once()

    def test_skips_duplicates(self, writer, mock_pg_conn):
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "fiscal_year": [2024, 2024],
            "fiscal_quarter": [1, 2],
            "total_assets": [3e11, 3.1e11],
        })
        existing = pd.DataFrame({
            "symbol": ["AAPL"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
        })
        mock_pg_conn.read_query.return_value = existing
        count = writer.write_financials(df)
        assert count == 1

    def test_empty_df(self, writer, mock_pg_conn):
        assert writer.write_financials(None) == 0
        mock_pg_conn.write_dataframe.assert_not_called()


# ===================================================================
# write_risk_free_rates
# ===================================================================

class TestWriteRiskFreeRates:

    def test_happy_path(self, writer, mock_pg_conn):
        df = pd.DataFrame({
            "country": ["US"],
            "rate_date": [date.today()],
            "rate": [0.04],
        })
        count = writer.write_risk_free_rates(df)
        assert count == 1

    def test_skips_duplicates(self, writer, mock_pg_conn):
        df = pd.DataFrame({
            "country": ["US", "US"],
            "rate_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "rate": [0.04, 0.05],
        })
        existing = pd.DataFrame({
            "country": ["US"],
            "rate_date": pd.to_datetime(["2024-01-01"]),
        })
        mock_pg_conn.read_query.return_value = existing
        count = writer.write_risk_free_rates(df)
        assert count == 1

    def test_empty_df(self, writer, mock_pg_conn):
        assert writer.write_risk_free_rates(None) == 0


# ===================================================================
# write_factor_metrics / write_factor_scores
# ===================================================================

class TestWriteFactorMetricsAndScores:

    def test_write_factor_metrics(self, writer, mock_pg_conn):
        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "calc_date": [date.today()],
            "pb_ratio": [3.5],
        })
        count = writer.write_factor_metrics(df)
        assert count == 1

    def test_write_factor_scores(self, writer, mock_pg_conn):
        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "score_date": [date.today()],
            "z_value": [0.5],
            "composite_score": [1.2],
        })
        count = writer.write_factor_scores(df)
        assert count == 1


# ===================================================================
# MongoDB writers
# ===================================================================

class TestMongoWriters:

    def test_log_fetch_to_mongo_with_df(self, writer, mock_mongo_conn):
        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "close_price": [150.0],
        })
        writer.log_fetch_to_mongo("prices", "AAPL", df)
        mock_mongo_conn.insert_one.assert_called_once()
        doc = mock_mongo_conn.insert_one.call_args[0][2]
        assert doc["data_type"] == "prices"
        assert doc["symbol"] == "AAPL"
        assert isinstance(doc["data"], list)

    def test_log_fetch_to_mongo_none(self, writer, mock_mongo_conn):
        writer.log_fetch_to_mongo("prices", "AAPL", None)
        mock_mongo_conn.insert_one.assert_not_called()

    def test_log_batch_to_mongo(self, writer, mock_mongo_conn):
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "close_price": [150.0, 151.0, 300.0],
        })
        writer.log_batch_to_mongo("prices", df)
        assert mock_mongo_conn.insert_one.call_count == 2


# ===================================================================
# get_table_counts
# ===================================================================

class TestGetTableCounts:

    def test_returns_counts(self, writer, mock_pg_conn):
        mock_pg_conn.read_query.return_value = pd.DataFrame({"cnt": [100]})
        counts = writer.get_table_counts()
        assert isinstance(counts, dict)
        assert "price_data" in counts
        assert counts["price_data"] == 100

    def test_handles_exception(self, writer, mock_pg_conn):
        mock_pg_conn.read_query.side_effect = Exception("table not found")
        counts = writer.get_table_counts()
        assert all(v == 0 for v in counts.values())
