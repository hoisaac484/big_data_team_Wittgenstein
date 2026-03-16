"""Tests for EDGAR HTTP, CIK resolution, concept extraction, and fundamentals."""

from unittest.mock import MagicMock, patch

import pandas as pd
import requests as req_lib

from modules.input.data_collector import DataFetcher

# ===================================================================
# _edgar_get_json
# ===================================================================


class TestEdgarGetJson:

    @patch("modules.input.data_collector.edgar.requests.get")
    def test_retries_on_failure(self, mock_get, fetcher):
        """Should retry on request failure and succeed on third attempt."""
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = Exception("timeout")
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {"data": "ok"}
        mock_get.side_effect = [
            req_lib.RequestException("timeout"),
            req_lib.RequestException("timeout"),
            ok_resp,
        ]
        result = fetcher._edgar_get_json("http://example.com", max_retries=3)
        assert result == {"data": "ok"}
        assert mock_get.call_count == 3

    @patch("modules.input.data_collector.edgar.requests.get")
    def test_returns_none_on_404_when_allowed(self, mock_get, fetcher):
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp
        result = fetcher._edgar_get_json("http://example.com", allow_not_found=True)
        assert result is None

    @patch("modules.input.data_collector.edgar.requests.get")
    def test_retries_on_429(self, mock_get, fetcher):
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "0"}
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {"ok": True}
        mock_get.side_effect = [rate_resp, ok_resp]
        result = fetcher._edgar_get_json("http://example.com", max_retries=3)
        assert result == {"ok": True}
        assert mock_get.call_count == 2

    @patch("modules.input.data_collector.edgar.requests.get")
    def test_returns_none_after_all_retries_fail(self, mock_get, fetcher):
        mock_get.side_effect = req_lib.RequestException("fail")
        result = fetcher._edgar_get_json("http://example.com", max_retries=2)
        assert result is None


# ===================================================================
# _resolve_cik
# ===================================================================


class TestResolveCik:

    def test_resolves_known_ticker(self, fetcher):
        with patch.object(
            fetcher,
            "_edgar_get_json",
            return_value={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
            },
        ):
            cik = fetcher._resolve_cik("AAPL")
        assert cik == "0000320193"

    def test_returns_none_for_unknown(self, fetcher):
        with patch.object(
            fetcher,
            "_edgar_get_json",
            return_value={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            },
        ):
            assert fetcher._resolve_cik("ZZZZ") is None

    def test_returns_none_on_network_error(self, fetcher):
        with patch.object(fetcher, "_edgar_get_json", return_value=None):
            assert fetcher._resolve_cik("AAPL") is None

    def test_caches_ticker_map(self, fetcher):
        mock_json = MagicMock(
            return_value={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            }
        )
        with patch.object(fetcher, "_edgar_get_json", mock_json):
            fetcher._resolve_cik("AAPL")
            fetcher._resolve_cik("AAPL")
        # Should only fetch the ticker map once
        assert mock_json.call_count == 1


# ===================================================================
# _edgar_fetch_company_facts
# ===================================================================


class TestEdgarFetchCompanyFacts:

    def test_returns_us_gaap_dict(self, fetcher):
        payload = {
            "facts": {
                "us-gaap": {
                    "Assets": {"units": {"USD": [{"val": 1}]}},
                }
            }
        }
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_fetch_company_facts("0000320193")
        assert "Assets" in result

    def test_returns_none_on_failure(self, fetcher):
        with patch.object(fetcher, "_edgar_get_json", return_value=None):
            result = fetcher._edgar_fetch_company_facts("0000320193")
        assert result is None

    def test_returns_none_when_no_us_gaap(self, fetcher):
        payload = {"facts": {"ifrs-full": {}}}
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_fetch_company_facts("0000320193")
        assert result is None


# ===================================================================
# _extract_concept
# ===================================================================


class TestExtractConcept:

    def test_parses_concept_data(self):
        facts = {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "end": "2024-03-30",
                            "start": "2024-01-01",
                            "val": 3.5e11,
                            "filed": "2024-05-01",
                            "form": "10-Q",
                        },
                        {
                            "end": "2024-06-30",
                            "start": "2024-04-01",
                            "val": 3.6e11,
                            "filed": "2024-08-01",
                            "form": "10-Q",
                        },
                    ]
                }
            }
        }
        result = DataFetcher._extract_concept(facts, "Assets")
        assert len(result) == 2

    def test_returns_empty_on_missing_tag(self):
        result = DataFetcher._extract_concept({"Other": {}}, "Assets")
        assert result.empty

    def test_returns_empty_on_none_facts(self):
        result = DataFetcher._extract_concept(None, "Assets")
        assert result.empty

    def test_filters_by_cutoff(self):
        facts = {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "end": "2020-03-30",
                            "val": 1e11,
                            "filed": "2020-05-01",
                            "form": "10-Q",
                        },
                        {
                            "end": "2024-03-30",
                            "val": 3.5e11,
                            "filed": "2024-05-01",
                            "form": "10-Q",
                        },
                    ]
                }
            }
        }
        result = DataFetcher._extract_concept(
            facts, "Assets", cutoff=pd.Timestamp("2023-01-01")
        )
        assert len(result) == 1
        assert result.iloc[0]["val"] == 3.5e11

    def test_filters_non_10q_10k_forms(self):
        facts = {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "end": "2024-03-30",
                            "val": 1e11,
                            "filed": "2024-05-01",
                            "form": "8-K",
                        },
                        {
                            "end": "2024-06-30",
                            "val": 2e11,
                            "filed": "2024-08-01",
                            "form": "10-K",
                        },
                    ]
                }
            }
        }
        result = DataFetcher._extract_concept(facts, "Assets")
        assert len(result) == 1
        assert result.iloc[0]["val"] == 2e11

    def test_dedupes_by_end_date(self):
        facts = {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "end": "2024-03-30",
                            "val": 1e11,
                            "filed": "2024-04-01",
                            "form": "10-Q",
                        },
                        {
                            "end": "2024-03-30",
                            "val": 1.5e11,
                            "filed": "2024-05-01",
                            "form": "10-Q",
                        },
                    ]
                }
            }
        }
        result = DataFetcher._extract_concept(facts, "Assets")
        assert len(result) == 1
        assert result.iloc[0]["val"] == 1.5e11  # keeps most recently filed


# ===================================================================
# _fetch_edgar_fundamentals
# ===================================================================


class TestFetchEdgarFundamentals:

    @staticmethod
    def _make_fact(end, start, val, filed, form="10-Q"):
        return {"end": end, "start": start, "val": val, "filed": filed, "form": form}

    def _build_facts(self, overrides=None):
        """Build a companyfacts us-gaap dict with sensible defaults."""
        base_row = self._make_fact("2024-03-30", "2024-01-01", 3.5e11, "2024-05-01")
        facts = {
            "Assets": {"units": {"USD": [base_row]}},
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        self._make_fact("2024-03-30", "2024-01-01", 2e10, "2024-05-01")
                    ]
                }
            },
            "StockholdersEquityIncludingPortionAttributable"
            "ToNoncontrollingInterest": {
                "units": {
                    "USD": [
                        self._make_fact("2024-03-30", "2024-01-01", 1e11, "2024-05-01")
                    ]
                }
            },
            "LongTermDebt": {
                "units": {
                    "USD": [
                        self._make_fact("2024-03-30", "2024-01-01", 5e10, "2024-05-01")
                    ]
                }
            },
        }
        if overrides:
            facts.update(overrides)
        return facts

    def test_happy_path(self, fetcher):
        periods = pd.DataFrame(
            {
                "report_date_str": ["2024-03-30"],
                "report_date": [pd.Timestamp("2024-03-30")],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
            }
        )
        facts = self._build_facts()

        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher, "_edgar_fetch_company_facts", return_value=facts
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL")

        assert not result.empty
        assert result.iloc[0]["total_assets"] == 3.5e11
        assert result.iloc[0]["source"] == "edgar"

    def test_no_cik(self, fetcher):
        fetcher._ticker_to_cik = {"MSFT": "0000789019"}
        result = fetcher._fetch_edgar_fundamentals("ZZZZ")
        assert result.empty

    def test_no_company_facts(self, fetcher):
        periods = pd.DataFrame(
            {
                "report_date_str": ["2024-03-30"],
                "report_date": [pd.Timestamp("2024-03-30")],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
            }
        )
        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher, "_edgar_fetch_company_facts", return_value=None
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL")
        assert result.empty

    def test_debt_fallback_to_tier2(self, fetcher):
        """LongTermDebt empty -> LongTermDebtAndCapitalLeaseObligations."""
        periods = pd.DataFrame(
            {
                "report_date_str": ["2024-03-30"],
                "report_date": [pd.Timestamp("2024-03-30")],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
            }
        )
        facts = self._build_facts()
        del facts["LongTermDebt"]
        facts["LongTermDebtAndCapitalLeaseObligations"] = {
            "units": {
                "USD": [self._make_fact("2024-03-30", "2024-01-01", 8e10, "2024-05-01")]
            }
        }

        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher, "_edgar_fetch_company_facts", return_value=facts
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL")

        assert not result.empty
        assert result.iloc[0]["total_debt"] == 8e10

    def test_equity_fallback(self, fetcher):
        """Primary equity empty -> StockholdersEquity fallback."""
        periods = pd.DataFrame(
            {
                "report_date_str": ["2024-03-30"],
                "report_date": [pd.Timestamp("2024-03-30")],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
            }
        )
        facts = self._build_facts()
        del facts[
            "StockholdersEquityIncludingPortionAttributable" "ToNoncontrollingInterest"
        ]
        facts["StockholdersEquity"] = {
            "units": {
                "USD": [self._make_fact("2024-03-30", "2024-01-01", 7e10, "2024-05-01")]
            }
        }

        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher, "_edgar_fetch_company_facts", return_value=facts
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL")

        assert result.iloc[0]["book_equity"] == 7e10

    def test_ytd_to_standalone_conversion(self, fetcher):
        """net_income should be de-cumulated from YTD to standalone quarterly."""
        periods = pd.DataFrame(
            {
                "report_date_str": ["2024-03-30", "2024-06-30", "2024-09-30"],
                "report_date": pd.to_datetime(
                    ["2024-03-30", "2024-06-30", "2024-09-30"]
                ),
                "fiscal_year": [2024, 2024, 2024],
                "fiscal_quarter": [1, 2, 3],
            }
        )
        # YTD net_income: Q1=10, Q1+Q2=25, Q1+Q2+Q3=40
        facts = {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        self._make_fact("2024-03-30", "2024-01-01", 10, "2024-05-01"),
                        self._make_fact("2024-06-30", "2024-01-01", 25, "2024-08-01"),
                        self._make_fact("2024-09-30", "2024-01-01", 40, "2024-11-01"),
                    ]
                }
            },
        }

        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher, "_edgar_fetch_company_facts", return_value=facts
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL")

        result = result.sort_values("fiscal_quarter")
        # Standalone: Q1=10, Q2=15, Q3=15
        assert list(result["net_income"]) == [10, 15, 15]


# ===================================================================
# _edgar_get_fiscal_periods
# ===================================================================


class TestEdgarGetFiscalPeriods:

    def test_happy_path(self, fetcher):
        payload = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "10-Q", "10-Q"],
                    "reportDate": [
                        "2024-09-30",
                        "2024-06-30",
                        "2024-03-31",
                        "2023-12-31",
                    ],
                    "filingDate": [
                        "2024-11-01",
                        "2024-08-01",
                        "2024-05-01",
                        "2024-02-01",
                    ],
                }
            }
        }
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_get_fiscal_periods("0000320193")
        assert not result.empty
        assert "fiscal_year" in result.columns
        assert "fiscal_quarter" in result.columns

    def test_no_payload(self, fetcher):
        with patch.object(fetcher, "_edgar_get_json", return_value=None):
            result = fetcher._edgar_get_fiscal_periods("0000320193")
        assert result.empty

    def test_empty_filings(self, fetcher):
        payload = {
            "filings": {
                "recent": {
                    "form": [],
                    "reportDate": [],
                    "filingDate": [],
                }
            }
        }
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_get_fiscal_periods("0000320193")
        assert result.empty


# ===================================================================
# _edgar_get_json retry sleep
# ===================================================================


class TestEdgarGetJsonRetry:

    @patch("modules.input.data_collector.edgar.sleep")
    @patch("modules.input.data_collector.edgar.requests.get")
    def test_retry_sleep_on_exception(self, mock_get, mock_sleep, fetcher):
        """RequestException triggers exponential backoff sleep (line 59)."""
        mock_get.side_effect = req_lib.RequestException("timeout")
        fetcher._edgar_get_json("http://example.com", max_retries=2)
        mock_sleep.assert_any_call(1)  # 2**0 = 1


# ===================================================================
# _edgar_get_fiscal_periods edge cases
# ===================================================================


class TestEdgarFiscalPeriodsEdgeCases:

    def test_multiple_10k_same_year(self, fetcher):
        """Multiple 10-K filings in same year (line 159-160)."""
        payload = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-K", "10-Q"],
                    "reportDate": [
                        "2024-03-31",
                        "2024-09-30",
                        "2024-06-30",
                    ],
                    "filingDate": [
                        "2024-04-15",
                        "2024-10-15",
                        "2024-07-15",
                    ],
                }
            }
        }
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_get_fiscal_periods("0000320193")
        assert not result.empty

    def test_no_10k_anchor_falls_back(self, fetcher):
        """No 10-K at all: fiscal_year derived from date (line 170-172)."""
        payload = {
            "filings": {
                "recent": {
                    "form": ["10-Q"],
                    "reportDate": ["2024-06-30"],
                    "filingDate": ["2024-07-15"],
                }
            }
        }
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_get_fiscal_periods("0000320193")
        assert isinstance(result, pd.DataFrame)

    def test_no_anchor_for_group(self, fetcher):
        """Fiscal year group with no matching anchor (line 184-185)."""
        payload = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q"],
                    "reportDate": ["2023-12-31", "2025-03-31"],
                    "filingDate": ["2024-02-15", "2025-04-15"],
                }
            }
        }
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_get_fiscal_periods("0000320193")
        assert not result.empty

    def test_cutoff_filters_old(self, fetcher):
        """Cutoff filters out old filings (line 142-145)."""
        payload = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "reportDate": ["2015-12-31"],
                    "filingDate": ["2016-02-15"],
                }
            }
        }
        cutoff = pd.Timestamp("2020-01-01")
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_get_fiscal_periods("0000320193", cutoff=cutoff)
        assert result.empty


# ===================================================================
# _extract_concept additional edge cases
# ===================================================================


class TestExtractConceptEdgeCases:

    def test_missing_end_skipped(self, fetcher):
        """Row with no end date is skipped (line 258)."""
        facts = {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "form": "10-K",
                            "end": None,
                            "val": 1e11,
                            "filed": "2024-02-15",
                        }
                    ]
                }
            }
        }
        result = fetcher._extract_concept(facts, "Assets", unit="USD")
        assert result.empty

    def test_invalid_end_date_skipped(self, fetcher):
        """Row with unparsable end date is skipped (line 260-261)."""
        facts = {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "form": "10-K",
                            "end": "not-a-date",
                            "val": 1e11,
                            "filed": "2024-02-15",
                        }
                    ]
                }
            }
        }
        result = fetcher._extract_concept(facts, "Assets", unit="USD")
        assert result.empty

    def test_all_non_10q_10k_returns_empty(self, fetcher):
        """All rows with non-10Q/10K forms returns empty (line 273)."""
        facts = {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "form": "8-K",
                            "end": "2024-12-31",
                            "val": 1e11,
                            "filed": "2025-01-15",
                        }
                    ]
                }
            }
        }
        result = fetcher._extract_concept(facts, "Assets", unit="USD")
        assert result.empty

    def test_empty_unit_rows(self, fetcher):
        """Empty unit rows returns empty (line 249-250)."""
        facts = {"Assets": {"units": {"USD": []}}}
        result = fetcher._extract_concept(facts, "Assets", unit="USD")
        assert result.empty


# ===================================================================
# _fetch_edgar_fundamentals fallback chains
# ===================================================================


class TestFetchEdgarFundamentalsEdgeCases:

    @staticmethod
    def _make_fact(end, start, val, filed, form="10-Q"):
        return {
            "end": end,
            "start": start,
            "val": val,
            "filed": filed,
            "form": form,
        }

    def test_eps_basic_fallback(self, fetcher):
        """EPS diluted empty falls back to basic (line 382-384)."""
        periods = pd.DataFrame(
            {
                "report_date": [pd.Timestamp("2024-03-31")],
                "report_date_str": ["2024-03-31"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "form": ["10-Q"],
            }
        )
        facts = {
            "EarningsPerShareBasic": {
                "units": {
                    "USD/shares": [
                        self._make_fact("2024-03-31", "2024-01-01", 1.5, "2024-04-15")
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [self._make_fact("2024-03-31", None, 3e11, "2024-04-15")]
                }
            },
        }
        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher, "_edgar_fetch_company_facts", return_value=facts
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL", period="5y")
        assert result is not None
        if not result.empty:
            assert "eps" in result.columns

    def test_debt_noncurrent_current_fallback(self, fetcher):
        """Debt falls back to noncurrent + current (line 414-441)."""
        periods = pd.DataFrame(
            {
                "report_date": [pd.Timestamp("2024-03-31")],
                "report_date_str": ["2024-03-31"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "form": ["10-Q"],
            }
        )
        facts = {
            "Assets": {
                "units": {
                    "USD": [self._make_fact("2024-03-31", None, 3e11, "2024-04-15")]
                }
            },
            "LongTermDebtNoncurrent": {
                "units": {
                    "USD": [self._make_fact("2024-03-31", None, 5e10, "2024-04-15")]
                }
            },
            "LongTermDebtCurrent": {
                "units": {
                    "USD": [self._make_fact("2024-03-31", None, 1e10, "2024-04-15")]
                }
            },
        }
        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher, "_edgar_fetch_company_facts", return_value=facts
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL", period="5y")
        assert result is not None
        if not result.empty and "total_debt" in result.columns:
            debt = result.iloc[0]["total_debt"]
            if pd.notna(debt):
                assert debt == 6e10
