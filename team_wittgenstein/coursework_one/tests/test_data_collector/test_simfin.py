"""Tests for SimFin HTTP, statement frames, shares, and fundamentals."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests as req_lib

from modules.input.data_collector import DataFetcher
from modules.input.data_collector.constants import SimFinServerError

# ===================================================================
# _normalize_quarter_value
# ===================================================================


class TestNormalizeQuarterValue:

    def test_q_strings(self):
        assert DataFetcher._normalize_quarter_value("Q1") == 1
        assert DataFetcher._normalize_quarter_value("Q4") == 4

    def test_lowercase(self):
        assert DataFetcher._normalize_quarter_value("q2") == 2

    def test_numeric_strings(self):
        assert DataFetcher._normalize_quarter_value("3") == 3
        assert DataFetcher._normalize_quarter_value("1") == 1

    def test_integer(self):
        assert DataFetcher._normalize_quarter_value(2) == 2

    def test_invalid(self):
        assert DataFetcher._normalize_quarter_value("Q5") is None
        assert DataFetcher._normalize_quarter_value("abc") is None

    def test_none(self):
        assert DataFetcher._normalize_quarter_value(None) is None


# ===================================================================
# _simfin_statement_frame
# ===================================================================


class TestSimfinStatementFrame:

    def test_valid_frame(self, fetcher):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "Fiscal Year": ["2024"],
                "Fiscal Period": ["Q1"],
                "Report Date": ["2024-03-31"],
                "Net Income": ["20000000"],
            }
        )
        result = fetcher._simfin_statement_frame(
            df,
            {
                "Fiscal Year": "fiscal_year",
                "Fiscal Period": "fiscal_quarter",
                "Report Date": "report_date",
                "Net Income": "net_income",
            },
            extra_cols=["net_income"],
        )
        assert len(result) == 1
        assert result.iloc[0]["fiscal_year"] == 2024
        assert result.iloc[0]["net_income"] == 20000000

    def test_none_input(self, fetcher):
        result = fetcher._simfin_statement_frame(
            None,
            {"Fiscal Year": "fiscal_year"},
            extra_cols=[],
        )
        assert result.empty

    def test_empty_input(self, fetcher):
        result = fetcher._simfin_statement_frame(
            pd.DataFrame(),
            {"Fiscal Year": "fiscal_year"},
            extra_cols=[],
        )
        assert result.empty

    def test_missing_columns(self, fetcher):
        df = pd.DataFrame({"symbol": ["AAPL"], "other": [1]})
        result = fetcher._simfin_statement_frame(
            df,
            {
                "Fiscal Year": "fiscal_year",
                "Fiscal Period": "fiscal_quarter",
                "Report Date": "report_date",
                "Net Income": "net_income",
            },
            extra_cols=["net_income"],
        )
        assert result.empty


# ===================================================================
# _simfin_weighted_shares_frame
# ===================================================================


class TestSimfinWeightedSharesFrame:

    def test_valid_payload(self, fetcher):
        payload = [
            {
                "ticker": "AAPL",
                "fyear": 2024,
                "period": "Q1",
                "diluted": 15000000000,
                "endDate": "2024-03-31",
            }
        ]
        result = fetcher._simfin_weighted_shares_frame(payload)
        assert len(result) == 1
        assert result.iloc[0]["shares_outstanding"] == 15000000000

    def test_none_payload(self, fetcher):
        result = fetcher._simfin_weighted_shares_frame(None)
        assert result.empty
        assert "shares_outstanding" in result.columns

    def test_empty_list_payload(self, fetcher):
        result = fetcher._simfin_weighted_shares_frame([])
        assert result.empty

    def test_default_symbol_fillna(self, fetcher):
        payload = [
            {
                "fyear": 2024,
                "period": "Q1",
                "diluted": 1000,
                "endDate": "2024-03-31",
            }
        ]
        result = fetcher._simfin_weighted_shares_frame(payload, default_symbol="MSFT")
        assert result.iloc[0]["symbol"] == "MSFT"


# ===================================================================
# _simfin_get
# ===================================================================


class TestSimfinGet:

    @patch("modules.input.data_collector.simfin.requests.get")
    def test_success_200(self, mock_get, fetcher):
        fetcher.simfin_api_key = "test-key"
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [{"data": "ok"}]
        mock_get.return_value = resp
        result = fetcher._simfin_get("http://example.com", params={"ticker": "AAPL"})
        assert result == [{"data": "ok"}]

    @patch("modules.input.data_collector.simfin.requests.get")
    def test_http_500_raises(self, mock_get, fetcher):
        fetcher.simfin_api_key = "test-key"
        resp = MagicMock()
        resp.status_code = 500
        mock_get.return_value = resp
        with pytest.raises(SimFinServerError):
            fetcher._simfin_get("http://example.com", params={}, max_retries=1)

    @patch("modules.input.data_collector.simfin.sleep")
    @patch("modules.input.data_collector.simfin.requests.get")
    def test_429_retries(self, mock_get, mock_sleep, fetcher):
        fetcher.simfin_api_key = "test-key"
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "0"}
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}
        mock_get.side_effect = [rate_resp, ok_resp]
        result = fetcher._simfin_get("http://example.com", params={}, max_retries=3)
        assert result == {"ok": True}

    @patch("modules.input.data_collector.simfin.sleep")
    @patch("modules.input.data_collector.simfin.requests.get")
    def test_request_exception_retries(self, mock_get, mock_sleep, fetcher):
        fetcher.simfin_api_key = "test-key"
        mock_get.side_effect = req_lib.RequestException("fail")
        result = fetcher._simfin_get("http://example.com", params={}, max_retries=2)
        assert result is None
        assert mock_get.call_count == 2

    @patch("modules.input.data_collector.simfin.sleep")
    @patch("modules.input.data_collector.simfin.requests.get")
    def test_other_status_retries(self, mock_get, mock_sleep, fetcher):
        fetcher.simfin_api_key = "test-key"
        resp = MagicMock()
        resp.status_code = 503
        mock_get.return_value = resp
        result = fetcher._simfin_get("http://example.com", params={}, max_retries=2)
        assert result is None


# ===================================================================
# _fetch_simfin_fundamentals
# ===================================================================


class TestFetchSimfinFundamentals:

    def test_no_api_key(self, fetcher):
        fetcher.simfin_api_key = None
        result = fetcher._fetch_simfin_fundamentals("AAPL")
        assert result.empty

    def test_empty_statements_payload(self, fetcher):
        fetcher.simfin_api_key = "test-key"
        with patch.object(fetcher, "_simfin_get", return_value=None):
            result = fetcher._fetch_simfin_fundamentals("AAPL")
        assert result.empty

    def test_happy_path(self, fetcher):
        fetcher.simfin_api_key = "test-key"
        statements_payload = [
            {
                "ticker": "AAPL",
                "statements": [
                    {
                        "statement": "PL",
                        "columns": [
                            "symbol",
                            "Fiscal Year",
                            "Fiscal Period",
                            "Report Date",
                            "Net Income",
                        ],
                        "data": [["AAPL", "2024", "Q1", "2024-03-31", "20000000"]],
                    },
                    {
                        "statement": "BS",
                        "columns": [
                            "symbol",
                            "Fiscal Year",
                            "Fiscal Period",
                            "Report Date",
                            "Total Assets",
                            "Total Equity",
                        ],
                        "data": [
                            [
                                "AAPL",
                                "2024",
                                "Q1",
                                "2024-03-31",
                                "300000000000",
                                "100000000000",
                            ]
                        ],
                    },
                    {
                        "statement": "DERIVED",
                        "columns": [
                            "symbol",
                            "Fiscal Year",
                            "Fiscal Period",
                            "Report Date",
                            "Total Debt",
                            "Earnings Per Share, Diluted",
                        ],
                        "data": [
                            [
                                "AAPL",
                                "2024",
                                "Q1",
                                "2024-03-31",
                                "100000000000",
                                "1.3",
                            ]
                        ],
                    },
                ],
            }
        ]
        shares_payload = [
            {
                "ticker": "AAPL",
                "fyear": 2024,
                "period": "Q1",
                "diluted": 15000000000,
                "endDate": "2024-03-31",
            }
        ]

        with patch.object(
            fetcher,
            "_simfin_get",
            side_effect=[statements_payload, shares_payload],
        ):
            result = fetcher._fetch_simfin_fundamentals("AAPL")

        assert not result.empty
        assert result.iloc[0]["source"] == "simfin"
        assert result.iloc[0]["fiscal_quarter"] == 1
        assert result.iloc[0]["total_assets"] == 300000000000

    def test_no_shares_data(self, fetcher):
        fetcher.simfin_api_key = "test-key"
        statements_payload = [
            {
                "ticker": "AAPL",
                "statements": [
                    {
                        "statement": "PL",
                        "columns": [
                            "symbol",
                            "Fiscal Year",
                            "Fiscal Period",
                            "Report Date",
                            "Net Income",
                        ],
                        "data": [["AAPL", "2024", "Q1", "2024-03-31", "20000000"]],
                    },
                    {
                        "statement": "BS",
                        "columns": [
                            "symbol",
                            "Fiscal Year",
                            "Fiscal Period",
                            "Report Date",
                            "Total Assets",
                            "Total Equity",
                        ],
                        "data": [
                            [
                                "AAPL",
                                "2024",
                                "Q1",
                                "2024-03-31",
                                "3e11",
                                "1e11",
                            ]
                        ],
                    },
                    {
                        "statement": "DERIVED",
                        "columns": [
                            "symbol",
                            "Fiscal Year",
                            "Fiscal Period",
                            "Report Date",
                            "Total Debt",
                            "Earnings Per Share, Diluted",
                        ],
                        "data": [
                            [
                                "AAPL",
                                "2024",
                                "Q1",
                                "2024-03-31",
                                "1e11",
                                "1.3",
                            ]
                        ],
                    },
                ],
            }
        ]

        with patch.object(
            fetcher,
            "_simfin_get",
            side_effect=[statements_payload, None],
        ):
            result = fetcher._fetch_simfin_fundamentals("AAPL")

        assert not result.empty
        assert pd.isna(result.iloc[0]["shares_outstanding"])

    def test_empty_statement_data_skipped(self, fetcher):
        """Statement with empty data/columns is skipped (line 53)."""
        fetcher.simfin_api_key = "test-key"
        statements_payload = [
            {
                "ticker": "AAPL",
                "statements": [
                    {
                        "statement": "PL",
                        "columns": [],
                        "data": [],
                    },
                ],
            }
        ]
        with patch.object(
            fetcher,
            "_simfin_get",
            side_effect=[statements_payload, None],
        ):
            result = fetcher._fetch_simfin_fundamentals("AAPL")
        assert result.empty

    def test_no_statement_rows_returns_empty(self, fetcher):
        """No valid statement rows produces empty result (line 105)."""
        fetcher.simfin_api_key = "test-key"
        statements_payload = [
            {
                "ticker": "AAPL",
                "statements": [],
            }
        ]
        with patch.object(
            fetcher,
            "_simfin_get",
            side_effect=[statements_payload, None],
        ):
            result = fetcher._fetch_simfin_fundamentals("AAPL")
        assert result.empty

    @patch("modules.input.data_collector.simfin.sleep")
    @patch("modules.input.data_collector.simfin.requests.get")
    def test_429_retry_after_non_numeric(self, mock_get, mock_sleep, fetcher):
        """429 with non-numeric Retry-After falls back to 2s (line 185-186)."""
        fetcher.simfin_api_key = "test-key"
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "not-a-number"}
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}
        mock_get.side_effect = [rate_resp, ok_resp]
        result = fetcher._simfin_get("http://example.com", params={}, max_retries=3)
        assert result == {"ok": True}
        mock_sleep.assert_any_call(2.0)

    def test_empty_shares_payload_list(self, fetcher):
        """Empty shares payload returns empty frame (line 254)."""
        result = fetcher._simfin_weighted_shares_frame([], default_symbol="AAPL")
        assert result.empty
        assert "shares_outstanding" in result.columns
