"""Tests for db_connection.py — PostgresConnection (no real DB required)."""

from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.db.db_connection import PostgresConnection

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def pg():
    """PostgresConnection with mocked SQLAlchemy engine."""
    with patch("modules.db.db_connection.create_engine") as mock_engine_factory:
        mock_engine = MagicMock()
        mock_engine_factory.return_value = mock_engine
        conn = PostgresConnection(
            host="localhost",
            port=5432,
            database="testdb",
            user="user",
            password="pass",
        )
        conn._mock_engine = mock_engine
        yield conn


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestPostgresConnectionInit:

    def test_connection_string_format(self):
        with patch("modules.db.db_connection.create_engine") as mock_factory:
            mock_factory.return_value = MagicMock()
            PostgresConnection("myhost", 5439, "mydb", "myuser", "mypass")
            call_arg = mock_factory.call_args[0][0]
            assert "myhost" in call_arg
            assert "5439" in call_arg
            assert "mydb" in call_arg
            assert "myuser" in call_arg

    def test_engine_created(self):
        with patch("modules.db.db_connection.create_engine") as mock_factory:
            mock_engine = MagicMock()
            mock_factory.return_value = mock_engine
            conn = PostgresConnection("h", 1, "db", "u", "p")
            assert conn.engine is mock_engine


# ── read_query ────────────────────────────────────────────────────────────────


class TestReadQuery:

    def test_calls_read_sql(self, pg):
        expected = pd.DataFrame({"col": [1, 2]})
        with patch(
            "modules.db.db_connection.pd.read_sql", return_value=expected
        ) as mock_read:
            result = pg.read_query("SELECT 1")
            assert mock_read.called
            pd.testing.assert_frame_equal(result, expected)

    def test_passes_params(self, pg):
        expected = pd.DataFrame({"x": [1]})
        with patch("modules.db.db_connection.pd.read_sql", return_value=expected):
            pg.read_query("SELECT :val", params={"val": 42})


# ── execute ───────────────────────────────────────────────────────────────────


class TestExecute:

    def test_calls_conn_execute(self, pg):
        mock_conn = MagicMock()
        pg._mock_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        pg._mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        pg.execute("CREATE TABLE foo (id INT)")
        assert mock_conn.execute.called


# ── write_dataframe ───────────────────────────────────────────────────────────


class TestWriteDataframe:

    def test_calls_to_sql(self, pg):
        df = pd.DataFrame({"a": [1, 2]})
        with patch.object(df, "to_sql") as mock_to_sql:
            pg.write_dataframe(df, "mytable", "myschema")
            mock_to_sql.assert_called_once_with(
                "mytable",
                pg.engine,
                schema="myschema",
                if_exists="append",
                index=False,
            )

    def test_if_exists_passed_through(self, pg):
        df = pd.DataFrame({"a": [1]})
        with patch.object(df, "to_sql") as mock_to_sql:
            pg.write_dataframe(df, "t", "s", if_exists="replace")
            _, kwargs = mock_to_sql.call_args
            assert kwargs["if_exists"] == "replace"


# ── write_dataframe_on_conflict_do_nothing ────────────────────────────────────


class TestWriteOnConflictDoNothing:

    def test_empty_df_does_nothing(self, pg):
        pg.write_dataframe_on_conflict_do_nothing(pd.DataFrame(), "t", "s", ["id"])
        pg._mock_engine.begin.assert_not_called()

    def test_none_does_nothing(self, pg):
        pg.write_dataframe_on_conflict_do_nothing(None, "t", "s", ["id"])
        pg._mock_engine.begin.assert_not_called()

    def test_calls_engine_begin_with_data(self, pg):
        df = pd.DataFrame({"symbol": ["AAPL"], "calc_date": ["2024-01-31"]})
        with patch("modules.db.db_connection.Table"), patch(
            "modules.db.db_connection.pg_insert"
        ) as mock_insert:
            mock_stmt = MagicMock()
            on_conflict = mock_insert.return_value.values.return_value
            on_conflict.on_conflict_do_nothing.return_value = mock_stmt
            mock_conn = MagicMock()
            pg._mock_engine.begin.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            pg._mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
            pg.write_dataframe_on_conflict_do_nothing(
                df, "factor_scores", "team_wittgenstein", ["symbol", "calc_date"]
            )
            assert pg._mock_engine.begin.called


# ── test_connection ───────────────────────────────────────────────────────────


class TestTestConnection:

    def test_returns_true_on_success(self, pg):
        mock_conn = MagicMock()
        pg._mock_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        pg._mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        assert pg.test_connection() is True

    def test_returns_false_on_exception(self, pg):
        pg._mock_engine.connect.side_effect = Exception("connection refused")
        assert pg.test_connection() is False


# ── get_company_list ──────────────────────────────────────────────────────────


class TestGetCompanyList:

    def test_calls_read_query(self, pg):
        expected = pd.DataFrame({"symbol": ["AAPL"], "gics_sector": ["IT"]})
        with patch.object(pg, "read_query", return_value=expected) as mock_rq:
            result = pg.get_company_list()
            assert mock_rq.called
            pd.testing.assert_frame_equal(result, expected)


# ── execute_sql_file ──────────────────────────────────────────────────────────


class TestExecuteSqlFile:

    def test_executes_each_statement(self, pg):
        sql = "CREATE TABLE a (id INT);\nCREATE TABLE b (id INT);\n"
        with NamedTemporaryFile(suffix=".sql", mode="w", delete=False) as f:
            f.write(sql)
            tmp_path = f.name

        mock_conn = MagicMock()
        pg._mock_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        pg._mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        pg.execute_sql_file(tmp_path)
        assert mock_conn.execute.call_count == 2

    def test_skips_comment_only_lines(self, pg):
        sql = "-- this is a comment\nSELECT 1;\n"
        with NamedTemporaryFile(suffix=".sql", mode="w", delete=False) as f:
            f.write(sql)
            tmp_path = f.name

        mock_conn = MagicMock()
        pg._mock_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        pg._mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        pg.execute_sql_file(tmp_path)
        assert mock_conn.execute.call_count == 1
