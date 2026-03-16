"""Tests for cache existence, helpers, TTL expiry, deletion, and mark loaded."""

import pandas as pd

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
# Cache helpers
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
            "name": "AAPL",
            "loaded_to_postgres": False,
        }
        fetcher.mark_loaded("prices", "AAPL")
        mock_minio_conn.upload_json.assert_called_once()
        uploaded = mock_minio_conn.upload_json.call_args[0][2]
        assert uploaded["loaded_to_postgres"] is True

    def test_mark_loaded_no_ctl(self, fetcher, mock_minio_conn):
        mock_minio_conn.download_json.return_value = None
        fetcher.mark_loaded("prices", "AAPL")
        mock_minio_conn.upload_json.assert_not_called()


# ===================================================================
# Cache TTL
# ===================================================================


class TestCacheExpiry:

    def test_expired_cache_returns_false(self, fetcher):
        fetcher.cache_ttl_days = 7
        fetcher.minio.object_exists.return_value = True
        fetcher.minio.download_json.return_value = {
            "fetched_at": (pd.Timestamp.now() - pd.DateOffset(days=10)).isoformat()
        }
        assert fetcher._is_cached("prices", "AAPL") is False

    def test_fresh_cache_returns_true(self, fetcher):
        fetcher.cache_ttl_days = 7
        fetcher.minio.object_exists.return_value = True
        fetcher.minio.download_json.return_value = {
            "fetched_at": pd.Timestamp.now().isoformat()
        }
        assert fetcher._is_cached("prices", "AAPL") is True

    def test_no_ttl_always_cached(self, fetcher):
        fetcher.cache_ttl_days = None
        fetcher.minio.object_exists.return_value = True
        assert fetcher._is_cached("prices", "AAPL") is True


# ===================================================================
# delete_symbol_cache
# ===================================================================


class TestDeleteSymbolCache:

    def test_deletes_objects(self, fetcher, mock_minio_conn):
        mock_minio_conn.delete_object.return_value = True
        mock_minio_conn.list_objects.return_value = []
        removed = fetcher.delete_symbol_cache("AAPL")
        assert removed == 4  # 2 data types x (parquet + ctl)

    def test_empty_symbol(self, fetcher):
        assert fetcher.delete_symbol_cache("") == 0
        assert fetcher.delete_symbol_cache("  ") == 0

    def test_with_source_scoped_files(self, fetcher, mock_minio_conn):
        mock_minio_conn.delete_object.return_value = True
        mock_minio_conn.list_objects.return_value = [
            "fundamentals/AAPL.simfin.parquet",
            "fundamentals/AAPL.simfin.ctl",
        ]
        removed = fetcher.delete_symbol_cache("AAPL")
        assert removed == 6  # 4 base + 2 source-scoped


# ===================================================================
# mark_loaded (fundamentals branch)
# ===================================================================


class TestMarkLoadedFundamentals:

    def test_fundamentals_with_source_scoped(self, fetcher, mock_minio_conn):
        mock_minio_conn.list_objects.return_value = ["fundamentals/AAPL.edgar.ctl"]
        mock_minio_conn.download_json.return_value = {
            "name": "AAPL",
            "loaded_to_postgres": False,
        }
        fetcher.mark_loaded("fundamentals", "AAPL")
        assert mock_minio_conn.upload_json.call_count >= 1

    def test_fundamentals_dotted_name(self, fetcher, mock_minio_conn):
        mock_minio_conn.download_json.return_value = {
            "name": "AAPL.simfin",
            "loaded_to_postgres": False,
        }
        fetcher.mark_loaded("fundamentals", "AAPL.simfin")
        uploaded = mock_minio_conn.upload_json.call_args[0][2]
        assert uploaded["loaded_to_postgres"] is True


# ===================================================================
# _is_cached edge cases
# ===================================================================


class TestDedupeDataframe:

    def test_none_returns_none(self, fetcher):
        assert fetcher._dedupe_dataframe("prices", None) is None

    def test_empty_returns_empty(self, fetcher):
        result = fetcher._dedupe_dataframe("prices", pd.DataFrame())
        assert result.empty


class TestMarkLoadedCtlNotFound:

    def test_fundamentals_ctl_not_found_skips(self, fetcher, mock_minio_conn):
        """When CTL file is not found for a fundamentals name, skip it."""
        mock_minio_conn.list_objects.return_value = []
        mock_minio_conn.download_json.return_value = None
        fetcher.mark_loaded("fundamentals", "AAPL")
        mock_minio_conn.upload_json.assert_not_called()


class TestIsCachedEdgeCases:

    def test_ctl_missing_fetched_at(self, fetcher, mock_minio_conn):
        fetcher.cache_ttl_days = 7
        mock_minio_conn.object_exists.return_value = True
        mock_minio_conn.download_json.return_value = {"name": "AAPL"}
        assert fetcher._is_cached("prices", "AAPL") is True

    def test_ctl_malformed_fetched_at(self, fetcher, mock_minio_conn):
        fetcher.cache_ttl_days = 7
        mock_minio_conn.object_exists.return_value = True
        mock_minio_conn.download_json.return_value = {"fetched_at": "not-a-date"}
        # Should handle exception and return True (pass through)
        assert fetcher._is_cached("prices", "AAPL") is True
