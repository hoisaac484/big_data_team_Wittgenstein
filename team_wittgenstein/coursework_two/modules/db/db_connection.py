"""Database connection module for PostgreSQL, MongoDB, and MinIO.

Provides connection classes for all three storage systems used in the
coursework data pipeline.
"""

import logging
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = logging.getLogger(__name__)
TEAM_SCHEMA = "team_wittgenstein"


class PostgresConnection:
    """Manages connections and queries to the PostgreSQL database.

    Args:
        host: Database host address.
        port: Database port number.
        database: Name of the database.
        user: Database username.
        password: Database password.
    """

    def __init__(self, host, port, database, user, password):
        self.connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )
        self.engine = create_engine(self.connection_string)
        logger.info("PostgreSQL engine created for %s:%s/%s", host, port, database)

    def read_query(self, query, params=None):
        """Execute a SELECT query and return results as a DataFrame.

        Args:
            query: SQL query string.
            params: Optional dictionary of query parameters.

        Returns:
            pd.DataFrame: Query results.
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)

    def execute(self, query, params=None):
        """Execute a non-returning SQL statement (CREATE, INSERT, etc.).

        Args:
            query: SQL statement string.
            params: Optional dictionary of query parameters.
        """
        with self.engine.connect() as conn:
            conn.execute(text(query), params)
            conn.commit()
        logger.info("Executed SQL statement successfully.")

    def write_dataframe(self, df, table_name, schema, if_exists="append"):
        """Write a DataFrame to a PostgreSQL table.

        Args:
            df: DataFrame to write.
            table_name: Target table name.
            schema: Target schema name.
            if_exists: What to do if table exists ('append', 'replace', 'fail').
        """
        df.to_sql(
            table_name,
            self.engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
        )
        logger.info("Wrote %d rows to %s.%s", len(df), schema, table_name)

    def write_dataframe_on_conflict_do_nothing(
        self, df, table_name, schema, conflict_columns
    ):
        """Write rows to PostgreSQL using ON CONFLICT DO NOTHING.

        Args:
            df: DataFrame to write.
            table_name: Target table name.
            schema: Target schema name.
            conflict_columns: Columns defining the conflict target.
        """
        if df is None or df.empty:
            return

        table = Table(
            table_name,
            MetaData(),
            schema=schema,
            autoload_with=self.engine,
        )
        records = df.to_dict(orient="records")
        stmt = pg_insert(table).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)

        with self.engine.begin() as conn:
            conn.execute(stmt)

        logger.debug(
            "Attempted to write %d rows to %s.%s (ON CONFLICT DO NOTHING)",
            len(df),
            schema,
            table_name,
        )

    def get_company_list(self):
        """Retrieve the full company universe from company_static.

        Returns:
            pd.DataFrame: All companies with symbol, security, sector, etc.
        """
        query = "SELECT * FROM systematic_equity.company_static"
        return self.read_query(query)

    def test_connection(self):
        """Test the database connection.

        Returns:
            bool: True if connection is successful.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("PostgreSQL connection test passed.")
            return True
        except Exception as e:
            logger.error("PostgreSQL connection test failed: %s", e)
            return False

    def execute_sql_file(self, sql_path: str):
        """Execute all statements in a SQL file.

        Args:
            sql_path: Path to the SQL file.
        """
        path = Path(sql_path)
        sql = path.read_text(encoding="utf-8")

        sql = re.sub(r"(?m)^\s*--.*$", "", sql).strip()
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        with self.engine.connect() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
            conn.commit()

        logger.info("Executed SQL file: %s (%d statements)", sql_path, len(statements))
