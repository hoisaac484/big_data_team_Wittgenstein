"""Tests for risk-free rates: OECD, yfinance fallback, and deduplication."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests as req_lib

# ===================================================================
# fetch_risk_free_rates
# ===================================================================


class TestFetchRiskFreeRates:

    def test_returns_cached(self, fetcher, mock_minio_conn):
        cached_df = pd.DataFrame(
            {
                "country": ["US"],
                "rate_date": [pd.Timestamp("2024-01-31")],
                "rate": [0.04],
            }
        )
        with patch.object(fetcher, "_is_cached", return_value=True), patch.object(
            fetcher, "_load_cached", return_value=cached_df
        ):
            result = fetcher.fetch_risk_free_rates(["US"])
        assert len(result) == 1

    def test_oecd_success(self, fetcher):
        oecd_df = pd.DataFrame(
            {
                "country": ["US"],
                "rate_date": [pd.Timestamp("2024-01-31")],
                "rate": [0.04],
            }
        )
        with patch.object(fetcher, "_is_cached", return_value=False), patch.object(
            fetcher, "_fetch_rates_oecd", return_value=oecd_df
        ):
            result = fetcher.fetch_risk_free_rates(["US"])
        assert len(result) == 1

    def test_oecd_fails_yfinance_fallback(self, fetcher):
        yf_df = pd.DataFrame(
            {
                "country": ["US"],
                "rate_date": [pd.Timestamp("2024-01-31")],
                "rate": [0.045],
            }
        )
        with patch.object(fetcher, "_is_cached", return_value=False), patch.object(
            fetcher, "_fetch_rates_oecd", return_value=None
        ), patch.object(fetcher, "_fetch_rates_yfinance", return_value=yf_df):
            result = fetcher.fetch_risk_free_rates(["US"])
        assert len(result) == 1
        assert result.iloc[0]["rate"] == 0.045

    def test_both_fail(self, fetcher):
        with patch.object(fetcher, "_is_cached", return_value=False), patch.object(
            fetcher, "_fetch_rates_oecd", return_value=None
        ), patch.object(fetcher, "_fetch_rates_yfinance", return_value=pd.DataFrame()):
            result = fetcher.fetch_risk_free_rates(["US"])
        assert result.empty


# ===================================================================
# _fetch_rates_oecd
# ===================================================================


class TestFetchRatesOecd:

    @patch("modules.input.data_collector.rates.requests.get")
    def test_valid_response(self, mock_get, fetcher):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "dataSets": [
                {
                    "series": {
                        "0:0:0": {
                            "observations": {
                                "0": [4.5],
                                "1": [4.2],
                            }
                        }
                    }
                }
            ],
            "structure": {
                "dimensions": {
                    "observation": [
                        {
                            "values": [
                                {"id": "2024-01"},
                                {"id": "2024-02"},
                            ]
                        }
                    ]
                }
            },
        }
        mock_get.return_value = mock_resp
        result = fetcher._fetch_rates_oecd(["US"])
        assert len(result) == 2
        assert result.iloc[0]["rate"] == 0.045
        assert "source" in result.columns
        assert (result["source"] == "oecd").all()

    @patch("modules.input.data_collector.rates.requests.get")
    def test_network_error(self, mock_get, fetcher):
        mock_get.side_effect = req_lib.RequestException("timeout")
        result = fetcher._fetch_rates_oecd(["US"])
        assert result is None

    def test_unknown_country(self, fetcher):
        result = fetcher._fetch_rates_oecd(["ZZ"])
        assert result is None


# ===================================================================
# _fetch_rates_yfinance
# ===================================================================


class TestFetchRatesYfinance:

    @patch("modules.input.data_collector.rates.yf")
    def test_valid_download(self, mock_yf, fetcher):
        dates = pd.bdate_range("2024-01-01", periods=60)
        irx = pd.DataFrame(
            {"Close": [4.5] * 60},
            index=dates,
        )
        irx.index.name = "Date"
        mock_yf.download.return_value = irx
        result = fetcher._fetch_rates_yfinance(["US"])
        assert not result.empty
        assert "country" in result.columns
        assert (result["country"] == "United States").all()
        assert result.iloc[0]["rate"] == pytest.approx(0.045)
        assert "source" in result.columns
        assert (result["source"] == "yfinance").all()

    @patch("modules.input.data_collector.rates.yf")
    def test_empty_download(self, mock_yf, fetcher):
        mock_yf.download.return_value = pd.DataFrame()
        result = fetcher._fetch_rates_yfinance(["US"])
        assert result.empty

    @patch("modules.input.data_collector.rates.yf")
    def test_exception(self, mock_yf, fetcher):
        mock_yf.download.side_effect = Exception("API error")
        result = fetcher._fetch_rates_yfinance(["US"])
        assert result.empty

    @patch("modules.input.data_collector.rates.yf")
    def test_multiple_countries(self, mock_yf, fetcher):
        dates = pd.bdate_range("2024-01-01", periods=60)
        irx = pd.DataFrame({"Close": [4.5] * 60}, index=dates)
        irx.index.name = "Date"
        mock_yf.download.return_value = irx
        result = fetcher._fetch_rates_yfinance(["US", "GB"])
        assert set(result["country"].unique()) == {"United States", "United Kingdom"}


# ===================================================================
# _dedupe_dataframe edge cases
# ===================================================================


class TestDedupeEdgeCases:

    def test_drops_duplicates(self, fetcher):
        df = pd.DataFrame(
            {
                "country": ["US", "US"],
                "rate_date": ["2024-01-31", "2024-01-31"],
                "rate": [0.04, 0.045],
            }
        )
        result = fetcher._dedupe_dataframe("risk_free_rates", df, name="all")
        assert len(result) == 1

    def test_unknown_data_type(self, fetcher):
        df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
        result = fetcher._dedupe_dataframe("unknown_type", df)
        assert len(result) == 2  # no subset matched, no dedup


class TestFetchRatesYfinanceEdgeCases:

    @patch("modules.input.data_collector.rates.yf")
    def test_multiindex_columns_flattened(self, mock_yf, fetcher):
        """MultiIndex columns from yfinance are flattened (line 135)."""
        dates = pd.bdate_range("2024-01-01", periods=5)
        mi = pd.MultiIndex.from_tuples([("Close", "^IRX")])
        data = [[4.5]] * 5
        irx = pd.DataFrame(data, index=dates, columns=mi)
        irx.index.name = "Date"
        mock_yf.download.return_value = irx
        result = fetcher._fetch_rates_yfinance(["US"])
        assert not result.empty
        assert "rate" in result.columns
