"""Tests for modules.db.db_connection."""

import io
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from modules.db.db_connection import (
    MinioConnection,
    MongoConnection,
    PostgresConnection,
)

# ===================================================================
# PostgresConnection
# ===================================================================


class TestPostgresConnection:

    @patch("modules.db.db_connection.create_engine")
    def test_creates_engine(self, mock_create_engine):
        PostgresConnection("localhost", 5432, "fift", "user", "pass")
        mock_create_engine.assert_called_once()
        assert "postgresql://user:pass@localhost:5432/fift" in str(
            mock_create_engine.call_args
        )

    @patch("modules.db.db_connection.create_engine")
    def test_read_query(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")

        with patch("modules.db.db_connection.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({"col": [1]})
            result = pg.read_query("SELECT 1")
            mock_read_sql.assert_called_once()
            assert len(result) == 1

    @patch("modules.db.db_connection.create_engine")
    def test_write_dataframe(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        df = pd.DataFrame({"col": [1, 2, 3]})

        with patch.object(df, "to_sql") as mock_to_sql:
            pg.write_dataframe(df, "test_table", "test_schema")
            mock_to_sql.assert_called_once_with(
                "test_table",
                mock_engine,
                schema="test_schema",
                if_exists="append",
                index=False,
            )

    @patch("modules.db.db_connection.create_engine")
    def test_test_connection(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        assert pg.test_connection() is True

        mock_engine.connect.side_effect = Exception("conn refused")
        assert pg.test_connection() is False

    @patch("modules.db.db_connection.create_engine")
    def test_write_dataframe_on_conflict_do_nothing(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        df = pd.DataFrame(
            {"symbol": ["AAPL"], "fiscal_year": [2024], "fiscal_quarter": [1]}
        )

        with patch("modules.db.db_connection.Table"), patch(
            "modules.db.db_connection.pg_insert"
        ) as mock_pg_insert:
            mock_stmt = MagicMock()
            mock_pg_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_nothing.return_value = mock_stmt

            mock_conn = MagicMock()
            mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

            pg.write_dataframe_on_conflict_do_nothing(
                df,
                "financial_data",
                "team_wittgenstein",
                ["symbol", "fiscal_year", "fiscal_quarter"],
            )
            mock_stmt.on_conflict_do_nothing.assert_called_once()
            mock_conn.execute.assert_called_once()

    @patch("modules.db.db_connection.create_engine")
    def test_write_dataframe_on_conflict_empty(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        pg.write_dataframe_on_conflict_do_nothing(
            None,
            "financial_data",
            "team_wittgenstein",
            ["symbol"],
        )
        pg.write_dataframe_on_conflict_do_nothing(
            pd.DataFrame(),
            "financial_data",
            "team_wittgenstein",
            ["symbol"],
        )
        mock_engine.begin.assert_not_called()

    @patch("modules.db.db_connection.create_engine")
    def test_execute(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        pg.execute("CREATE TABLE test (id int)")
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("modules.db.db_connection.create_engine")
    def test_get_company_list(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(
            pg, "read_query", return_value=pd.DataFrame({"symbol": ["AAPL"]})
        ) as mock_rq:
            result = pg.get_company_list()
            mock_rq.assert_called_once()
            assert "symbol" in result.columns

    @patch("modules.db.db_connection.create_engine")
    def test_get_managed_symbol_tables(self, mock_create_engine):
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(
            pg,
            "read_query",
            return_value=pd.DataFrame({"table_name": ["price_data", "financial_data"]}),
        ):
            assert pg.get_managed_symbol_tables() == ["price_data", "financial_data"]

    @patch("modules.db.db_connection.create_engine")
    def test_get_managed_symbol_tables_empty(self, mock_create_engine):
        """Empty result returns empty list (line 138-139)."""
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(pg, "read_query", return_value=pd.DataFrame()):
            assert pg.get_managed_symbol_tables() == []

    @patch("modules.db.db_connection.create_engine")
    def test_get_managed_symbol_tables_none(self, mock_create_engine):
        """None result returns empty list (line 138-139)."""
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(pg, "read_query", return_value=None):
            assert pg.get_managed_symbol_tables() == []

    @patch("modules.db.db_connection.create_engine")
    def test_get_tracked_symbols(self, mock_create_engine):
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(
            pg,
            "get_managed_symbol_tables",
            return_value=["price_data", "financial_data"],
        ), patch.object(
            pg,
            "read_query",
            side_effect=[
                pd.DataFrame({"symbol": ["AAPL", "MSFT"]}),
                pd.DataFrame({"symbol": ["MSFT", "GOOG"]}),
            ],
        ):
            assert pg.get_tracked_symbols() == ["AAPL", "GOOG", "MSFT"]

    @patch("modules.db.db_connection.create_engine")
    def test_delete_symbols_missing_from_company_list(self, mock_create_engine):
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(
            pg,
            "get_tracked_symbols",
            return_value=["AAPL", "MSFT", "SAP"],
        ), patch.object(pg, "delete_symbol_data") as mock_delete:
            removed = pg.delete_symbols_missing_from_company_list(["AAPL", "SAP"])
            assert removed == ["MSFT"]
            mock_delete.assert_called_once_with(["MSFT"])

    @patch("modules.db.db_connection.create_engine")
    def test_delete_symbols_missing_none_removed(self, mock_create_engine):
        """No removed symbols returns empty list (line 193-194)."""
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(pg, "get_tracked_symbols", return_value=["AAPL"]):
            result = pg.delete_symbols_missing_from_company_list(["AAPL"])
        assert result == []

    @patch("modules.db.db_connection.create_engine")
    def test_get_tracked_symbols_empty_table(self, mock_create_engine):
        """Empty query result during symbol scan is skipped (line 147-148)."""
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(
            pg, "get_managed_symbol_tables", return_value=["price_data"]
        ), patch.object(pg, "read_query", return_value=pd.DataFrame()):
            result = pg.get_tracked_symbols()
        assert result == []

    @patch("modules.db.db_connection.create_engine")
    def test_delete_symbol_data_empty_symbols(self, mock_create_engine):
        """Empty symbols list returns 0 (line 158-159)."""
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        assert pg.delete_symbol_data([]) == 0

    @patch("modules.db.db_connection.create_engine")
    def test_delete_symbol_data_no_tables(self, mock_create_engine):
        """No managed tables returns 0 (line 162-163)."""
        mock_create_engine.return_value = MagicMock()
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(pg, "get_managed_symbol_tables", return_value=[]):
            assert pg.delete_symbol_data(["AAPL"]) == 0

    @patch("modules.db.db_connection.create_engine")
    def test_delete_symbol_data_executes(self, mock_create_engine):
        """Executes DELETE on managed tables (line 165-182)."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        with patch.object(pg, "get_managed_symbol_tables", return_value=["price_data"]):
            result = pg.delete_symbol_data(["AAPL", "MSFT"])
        assert result == 2
        mock_conn.execute.assert_called_once()

    @patch("modules.db.db_connection.create_engine")
    def test_execute_sql_file(self, mock_create_engine, tmp_path):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        sql_file = tmp_path / "test.sql"
        sql_file.write_text("CREATE TABLE a (id int);\nCREATE TABLE b (id int);")

        pg = PostgresConnection("localhost", 5432, "fift", "user", "pass")
        pg.execute_sql_file(str(sql_file))
        assert mock_conn.execute.call_count == 2
        mock_conn.commit.assert_called_once()


# ===================================================================
# MongoConnection
# ===================================================================


class TestMongoConnection:

    @patch("modules.db.db_connection.MongoClient")
    def test_insert_one(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.insert_one.return_value.inserted_id = "abc123"
        mock_client.__getitem__ = MagicMock(
            return_value=MagicMock(__getitem__=MagicMock(return_value=mock_collection))
        )

        mongo = MongoConnection("localhost", 27017)
        result = mongo.insert_one("testdb", "testcol", {"key": "value"})
        assert result == "abc123"

    @patch("modules.db.db_connection.MongoClient")
    def test_insert_many(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.insert_many.return_value.inserted_ids = ["id1", "id2"]
        mock_client.__getitem__ = MagicMock(
            return_value=MagicMock(__getitem__=MagicMock(return_value=mock_collection))
        )

        mongo = MongoConnection("localhost", 27017)
        result = mongo.insert_many("testdb", "testcol", [{"a": 1}, {"a": 2}])
        assert result == ["id1", "id2"]

    @patch("modules.db.db_connection.MongoClient")
    def test_find(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.find.return_value = iter([{"a": 1}, {"a": 2}])
        mock_client.__getitem__ = MagicMock(
            return_value=MagicMock(__getitem__=MagicMock(return_value=mock_collection))
        )

        mongo = MongoConnection("localhost", 27017)
        results = mongo.find("testdb", "testcol")
        assert len(results) == 2

    @patch("modules.db.db_connection.MongoClient")
    def test_test_connection(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mongo = MongoConnection("localhost", 27017)
        assert mongo.test_connection() is True

        mock_client.admin.command.side_effect = Exception("refused")
        assert mongo.test_connection() is False


# ===================================================================
# MinioConnection
# ===================================================================


class TestMinioConnection:

    @patch("modules.db.db_connection.Minio")
    def test_object_exists(self, mock_minio_cls):
        from minio.error import S3Error

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        minio = MinioConnection("localhost:9000", "key", "secret")

        mock_client.stat_object.return_value = True
        assert minio.object_exists("bucket", "file.parquet") is True

        mock_client.stat_object.side_effect = S3Error(
            MagicMock(), "NoSuchKey", "not found", "resource", "req", "host"
        )
        assert minio.object_exists("bucket", "file.parquet") is False

    @patch("modules.db.db_connection.Minio")
    def test_ensure_bucket(self, mock_minio_cls):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        minio = MinioConnection("localhost:9000", "key", "secret")

        mock_client.bucket_exists.return_value = False
        minio._ensure_bucket("new-bucket")
        mock_client.make_bucket.assert_called_once_with("new-bucket")

        mock_client.reset_mock()
        mock_client.bucket_exists.return_value = True
        minio._ensure_bucket("existing-bucket")
        mock_client.make_bucket.assert_not_called()

    @patch("modules.db.db_connection.Minio")
    def test_upload_json(self, mock_minio_cls):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = True

        minio = MinioConnection("localhost:9000", "key", "secret")
        minio.upload_json("bucket", "test.json", {"key": "val"})
        mock_client.put_object.assert_called_once()
        call_args = mock_client.put_object.call_args
        assert call_args[0][0] == "bucket"
        assert call_args[0][1] == "test.json"

    @patch("modules.db.db_connection.Minio")
    def test_download_json(self, mock_minio_cls):
        from minio.error import S3Error

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"key": "val"}).encode()
        mock_client.get_object.return_value = mock_response

        minio = MinioConnection("localhost:9000", "key", "secret")
        result = minio.download_json("bucket", "test.json")
        assert result == {"key": "val"}
        mock_response.close.assert_called_once()

        # Test NoSuchKey returns None
        mock_resp = MagicMock()
        err = S3Error(mock_resp, "NoSuchKey", "not found", "resource", "req", "host")
        mock_client.get_object.side_effect = err
        assert minio.download_json("bucket", "missing.json") is None

    @patch("modules.db.db_connection.Minio")
    def test_upload_dataframe(self, mock_minio_cls):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = True

        minio = MinioConnection("localhost:9000", "key", "secret")
        df = pd.DataFrame({"col": [1, 2, 3]})
        minio.upload_dataframe("bucket", "data.parquet", df)
        mock_client.put_object.assert_called_once()

    @patch("modules.db.db_connection.Minio")
    def test_download_dataframe(self, mock_minio_cls):
        from minio.error import S3Error

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        # Create actual parquet bytes
        df = pd.DataFrame({"col": [1, 2, 3]})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        parquet_bytes = buf.getvalue()

        mock_response = MagicMock()
        mock_response.read.return_value = parquet_bytes
        mock_client.get_object.return_value = mock_response

        minio = MinioConnection("localhost:9000", "key", "secret")
        result = minio.download_dataframe("bucket", "data.parquet")
        assert len(result) == 3

        # Test NoSuchKey returns None
        mock_resp = MagicMock()
        err = S3Error(mock_resp, "NoSuchKey", "not found", "resource", "req", "host")
        mock_client.get_object.side_effect = err
        assert minio.download_dataframe("bucket", "missing.parquet") is None

    @patch("modules.db.db_connection.Minio")
    def test_list_objects(self, mock_minio_cls):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        obj1 = MagicMock()
        obj1.object_name = "prices/AAPL.parquet"
        obj2 = MagicMock()
        obj2.object_name = "prices/MSFT.parquet"
        mock_client.list_objects.return_value = [obj1, obj2]

        minio = MinioConnection("localhost:9000", "key", "secret")
        names = minio.list_objects("bucket", prefix="prices/")
        assert len(names) == 2
        assert "prices/AAPL.parquet" in names

    @patch("modules.db.db_connection.Minio")
    def test_test_connection(self, mock_minio_cls):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        minio = MinioConnection("localhost:9000", "key", "secret")
        assert minio.test_connection() is True

        mock_client.list_buckets.side_effect = Exception("unreachable")
        assert minio.test_connection() is False

    @patch("modules.db.db_connection.Minio")
    def test_download_json_other_s3error_raises(self, mock_minio_cls):
        """Non-NoSuchKey S3Error is re-raised (line 376)."""
        from minio.error import S3Error

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        err = S3Error(
            MagicMock(), "AccessDenied", "forbidden", "resource", "req", "host"
        )
        mock_client.get_object.side_effect = err

        minio = MinioConnection("localhost:9000", "key", "secret")
        import pytest

        with pytest.raises(S3Error):
            minio.download_json("bucket", "secret.json")

    @patch("modules.db.db_connection.Minio")
    def test_download_dataframe_other_s3error_raises(self, mock_minio_cls):
        """Non-NoSuchKey S3Error is re-raised (line 418)."""
        from minio.error import S3Error

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        err = S3Error(
            MagicMock(), "AccessDenied", "forbidden", "resource", "req", "host"
        )
        mock_client.get_object.side_effect = err

        minio = MinioConnection("localhost:9000", "key", "secret")
        import pytest

        with pytest.raises(S3Error):
            minio.download_dataframe("bucket", "secret.parquet")

    @patch("modules.db.db_connection.Minio")
    def test_delete_object_success(self, mock_minio_cls):
        """Successful delete returns True (line 452-454)."""
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        minio = MinioConnection("localhost:9000", "key", "secret")
        assert minio.delete_object("bucket", "file.json") is True

    @patch("modules.db.db_connection.Minio")
    def test_delete_object_nosuchkey(self, mock_minio_cls):
        """NoSuchKey returns False (line 456-457)."""
        from minio.error import S3Error

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        err = S3Error(MagicMock(), "NoSuchKey", "not found", "resource", "req", "host")
        mock_client.remove_object.side_effect = err
        minio = MinioConnection("localhost:9000", "key", "secret")
        assert minio.delete_object("bucket", "missing.json") is False

    @patch("modules.db.db_connection.Minio")
    def test_delete_object_other_error_raises(self, mock_minio_cls):
        """Non-NoSuchKey S3Error is re-raised (line 458)."""
        from minio.error import S3Error

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        err = S3Error(
            MagicMock(), "AccessDenied", "forbidden", "resource", "req", "host"
        )
        mock_client.remove_object.side_effect = err
        minio = MinioConnection("localhost:9000", "key", "secret")
        import pytest

        with pytest.raises(S3Error):
            minio.delete_object("bucket", "secret.json")
