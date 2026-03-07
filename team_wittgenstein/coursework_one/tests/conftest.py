"""Shared pytest fixtures for the data pipeline test suite."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from modules.db.db_connection import MinioConnection, MongoConnection, PostgresConnection
from modules.processing.data_validator import DataValidator


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices_df():
    """Valid price DataFrame: 2 symbols, ~5 years of daily data."""
    dates = pd.bdate_range(end=date.today(), periods=1300)
    rows = []
    for sym in ["AAPL", "MSFT"]:
        for d in dates:
            rows.append({
                "symbol": sym,
                "trade_date": d,
                "open_price": 150.0,
                "high_price": 155.0,
                "low_price": 148.0,
                "close_price": 152.0,
                "adjusted_close": 152.0,
                "volume": 1_000_000,
                "source": "yfinance",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_financials_df():
    """Valid financials DataFrame: 2 symbols, 4 quarters each."""
    rows = []
    for sym in ["AAPL", "MSFT"]:
        for yr in [2024, 2025]:
            for q in [1, 2]:
                rows.append({
                    "symbol": sym,
                    "fiscal_year": yr,
                    "fiscal_quarter": q,
                    "report_date": pd.Timestamp(f"{yr}-{q * 3:02d}-28"),
                    "currency": "USD",
                    "total_assets": 3e11,
                    "total_equity": 1e11,
                    "total_debt": 1e11,
                    "book_value_equity": 1e11,
                    "shares_outstanding": 15_000_000_000,
                    "net_income": 2e10,
                    "eps": 1.30,
                    "source": "yfinance",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_rates_df():
    """Valid risk-free rates DataFrame: 2 countries, recent dates."""
    rows = []
    for country in ["US", "GB"]:
        for i in range(30):
            rows.append({
                "country": country,
                "rate_date": date.today() - timedelta(days=i * 7),
                "rate": 0.04,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Expected lists
# ---------------------------------------------------------------------------

@pytest.fixture
def expected_symbols():
    return ["AAPL", "MSFT"]


@pytest.fixture
def expected_countries():
    return ["US", "GB"]


# ---------------------------------------------------------------------------
# Mock connections
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pg_conn():
    mock = MagicMock(spec=PostgresConnection)
    mock.read_query.return_value = pd.DataFrame()
    return mock


@pytest.fixture
def mock_mongo_conn():
    mock = MagicMock(spec=MongoConnection)
    mock.insert_one.return_value = "fake_id"
    return mock


@pytest.fixture
def mock_minio_conn():
    mock = MagicMock(spec=MinioConnection)
    mock.object_exists.return_value = False
    return mock


# ---------------------------------------------------------------------------
# Validator with relaxed thresholds for small test data
# ---------------------------------------------------------------------------

@pytest.fixture
def validator():
    return DataValidator(min_price_rows=5, min_years=1, max_null_pct=0.5)
