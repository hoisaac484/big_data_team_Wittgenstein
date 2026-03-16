"""Data writer module for loading validated data into databases.

Writes processed data to PostgreSQL (structured tables) and raw API
responses to MongoDB (audit trail). Updates CTL files in MinIO after
successful loads to maintain pipeline idempotency.
"""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA = "team_wittgenstein"
MONGO_DB = "wittgenstein"


class DataWriter:
    """Loads validated data into PostgreSQL and MongoDB.

    Handles duplicate prevention via upsert-style logic: checks what
    already exists before inserting, so the pipeline can be re-run
    safely without creating duplicate rows.

    Args:
        pg_conn: PostgresConnection instance.
        mongo_conn: MongoConnection instance.
        fetcher: DataFetcher instance (to update CTL files after loading).
    """

    def __init__(self, pg_conn, mongo_conn, fetcher=None):
        self.pg = pg_conn
        self.mongo = mongo_conn
        self.fetcher = fetcher

    # PostgreSQL writers

    def write_prices(self, df):
        """Write daily price data to PostgreSQL.

        Filters out rows that already exist (by symbol + trade_date)
        to prevent duplicate key violations on re-runs.

        Args:
            df: Price DataFrame with columns: symbol, trade_date,
                open_price, high_price, low_price, close_price,
                adjusted_close, currency, volume.

        Returns:
            int: Number of new rows written.
        """
        if df is None or df.empty:
            logger.warning("No price data to write.")
            return 0

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # Get existing (symbol, date) pairs to avoid duplicates
        existing = self._get_existing_keys("price_data", "symbol", "trade_date")

        before = len(df)
        if existing is not None and not existing.empty:
            existing["trade_date"] = pd.to_datetime(existing["trade_date"])
            merged = df.merge(
                existing,
                on=["symbol", "trade_date"],
                how="left",
                indicator=True,
            )
            df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        new_rows = len(df)
        skipped = before - new_rows
        if skipped > 0:
            logger.info("Prices: skipping %d existing rows", skipped)

        if new_rows == 0:
            logger.info("Prices: no new rows to write.")
            return 0

        # Strip columns not in the price_data schema before writing
        schema_cols = [
            "symbol",
            "trade_date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "adjusted_close",
            "currency",
            "volume",
        ]
        df = df[[c for c in schema_cols if c in df.columns]]

        self.pg.write_dataframe(df, "price_data", SCHEMA)
        logger.info("Prices: wrote %d new rows to PostgreSQL", new_rows)

        # Update CTL files
        if self.fetcher:
            for symbol in df["symbol"].unique():
                self.fetcher.mark_loaded("prices", symbol)

        return new_rows

    def write_financials(self, df):
        """Write quarterly financial data to PostgreSQL.

        Filters out rows that already exist (by symbol + fiscal_year +
        fiscal_quarter) to prevent duplicate key violations on re-runs.

        Args:
            df: Financials DataFrame with columns: symbol, fiscal_year,
                fiscal_quarter, report_date, currency, total_assets,
                total_debt, net_income, book_equity,
                shares_outstanding, eps.

        Returns:
            int: Number of new rows written.
        """
        if df is None or df.empty:
            logger.warning("No financial data to write.")
            return 0

        df = df.copy()
        before_incoming_dedupe = len(df)
        df = df.drop_duplicates(
            subset=["symbol", "fiscal_year", "fiscal_quarter"],
            keep="last",
        )
        incoming_dupes = before_incoming_dedupe - len(df)
        if incoming_dupes > 0:
            logger.info(
                "Financials: dropped %d duplicate incoming rows by key",
                incoming_dupes,
            )

        # 3-column unique key: symbol + fiscal_year + fiscal_quarter
        try:
            query = (
                f"SELECT symbol, fiscal_year, fiscal_quarter "  # nosec B608
                f"FROM {SCHEMA}.financial_data"
            )
            existing = self.pg.read_query(query)
        except Exception as e:
            logger.warning("Could not read existing financials keys: %s", e)
            existing = pd.DataFrame()

        before = len(df)
        if existing is not None and not existing.empty:
            merged = df.merge(
                existing,
                on=["symbol", "fiscal_year", "fiscal_quarter"],
                how="left",
                indicator=True,
            )
            df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        new_rows = len(df)
        skipped = before - new_rows
        if skipped > 0:
            logger.info("Financials: skipping %d existing rows", skipped)

        if new_rows == 0:
            logger.info("Financials: no new rows to write.")
            return 0

        # Strip columns not in the financial_data schema before writing
        schema_cols = [
            "symbol",
            "report_date",
            "currency",
            "total_assets",
            "total_debt",
            "net_income",
            "book_equity",
            "shares_outstanding",
            "eps",
            "fiscal_year",
            "fiscal_quarter",
            "source",
        ]
        df_write = df[[c for c in schema_cols if c in df.columns]]

        self.pg.write_dataframe_on_conflict_do_nothing(
            df_write,
            "financial_data",
            SCHEMA,
            ["symbol", "fiscal_year", "fiscal_quarter"],
        )
        logger.info("Financials: wrote %d new rows to PostgreSQL", new_rows)

        if self.fetcher:
            for symbol in df["symbol"].unique():
                self.fetcher.mark_loaded("fundamentals", symbol)

        return new_rows

    def write_risk_free_rates(self, df):
        """Write risk-free rate data to PostgreSQL.

        Filters out rows that already exist (by country + rate_date)
        to prevent duplicate key violations on re-runs.

        Args:
            df: Rates DataFrame with columns: country, rate_date, rate.

        Returns:
            int: Number of new rows written.
        """
        if df is None or df.empty:
            logger.warning("No risk-free rate data to write.")
            return 0

        df = df.copy()
        df["rate_date"] = pd.to_datetime(df["rate_date"])

        existing = self._get_existing_keys("risk_free_rates", "country", "rate_date")

        before = len(df)
        if existing is not None and not existing.empty:
            existing["rate_date"] = pd.to_datetime(existing["rate_date"])
            merged = df.merge(
                existing,
                on=["country", "rate_date"],
                how="left",
                indicator=True,
            )
            df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        new_rows = len(df)
        skipped = before - new_rows
        if skipped > 0:
            logger.info("Rates: skipping %d existing rows", skipped)

        if new_rows == 0:
            logger.info("Rates: no new rows to write.")
            return 0

        self.pg.write_dataframe(df, "risk_free_rates", SCHEMA)
        logger.info("Rates: wrote %d new rows to PostgreSQL", new_rows)

        if self.fetcher:
            self.fetcher.mark_loaded("risk_free_rates", "all")

        return new_rows

    def write_factor_metrics(self, df):
        """Write raw factor metrics to PostgreSQL.

        Args:
            df: Metrics DataFrame with columns: symbol, calc_date,
                pb_ratio, asset_growth, roe, leverage,
                earnings_stability, momentum_6m, momentum_12m,
                volatility_3m, volatility_12m.

        Returns:
            int: Number of new rows written.
        """
        if df is None or df.empty:
            logger.warning("No factor metrics to write.")
            return 0

        df = df.copy()
        df["calc_date"] = pd.to_datetime(df["calc_date"])

        existing = self._get_existing_keys("factor_metrics", "symbol", "calc_date")

        if existing is not None and not existing.empty:
            existing["calc_date"] = pd.to_datetime(existing["calc_date"])
            merged = df.merge(
                existing,
                on=["symbol", "calc_date"],
                how="left",
                indicator=True,
            )
            df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        new_rows = len(df)
        if new_rows == 0:
            logger.info("Factor metrics: no new rows to write.")
            return 0

        self.pg.write_dataframe(df, "factor_metrics", SCHEMA)
        logger.info("Factor metrics: wrote %d new rows", new_rows)
        return new_rows

    def write_factor_scores(self, df):
        """Write sector-normalised factor z-scores to PostgreSQL.

        Args:
            df: Scores DataFrame with columns: symbol, score_date,
                z_value, z_quality, z_momentum, z_low_vol,
                composite_score.

        Returns:
            int: Number of new rows written.
        """
        if df is None or df.empty:
            logger.warning("No factor scores to write.")
            return 0

        df = df.copy()
        df["score_date"] = pd.to_datetime(df["score_date"])

        existing = self._get_existing_keys("factor_scores", "symbol", "score_date")

        if existing is not None and not existing.empty:
            existing["score_date"] = pd.to_datetime(existing["score_date"])
            merged = df.merge(
                existing,
                on=["symbol", "score_date"],
                how="left",
                indicator=True,
            )
            df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        new_rows = len(df)
        if new_rows == 0:
            logger.info("Factor scores: no new rows to write.")
            return 0

        self.pg.write_dataframe(df, "factor_scores", SCHEMA)
        logger.info("Factor scores: wrote %d new rows", new_rows)
        return new_rows

    def _get_existing_keys(self, table, key_col1, key_col2):
        """Query existing primary key pairs from a table.

        Args:
            table: Table name.
            key_col1: First key column name.
            key_col2: Second key column name.

        Returns:
            pd.DataFrame with the two key columns, or empty DataFrame.
        """
        try:
            query = (
                f"SELECT {key_col1}, {key_col2} "  # nosec B608
                f"FROM {SCHEMA}.{table}"
            )
            return self.pg.read_query(query)
        except Exception as e:
            logger.warning("Could not read existing keys from %s: %s", table, e)
            return pd.DataFrame()

    # MongoDB writers (audit trail)

    def log_fetch_to_mongo(self, data_type, symbol, raw_data):
        """Store a raw API response in MongoDB for auditing.

        Each document records what was fetched, when, and the raw
        content, providing a complete audit trail.

        Args:
            data_type: Type of data ('prices', 'fundamentals', 'rates').
            symbol: Ticker symbol or identifier.
            raw_data: The raw data to store (dict, list, or DataFrame).
        """
        if raw_data is None:
            return

        # Convert DataFrame to dict for MongoDB storage
        if isinstance(raw_data, pd.DataFrame):
            raw_data = raw_data.to_dict(orient="records")

        document = {
            "data_type": data_type,
            "symbol": symbol,
            "fetched_at": datetime.utcnow().isoformat(),
            "row_count": len(raw_data) if isinstance(raw_data, list) else 1,
            "data": raw_data,
        }

        self.mongo.insert_one(MONGO_DB, f"raw_{data_type}", document)
        logger.info("MongoDB: logged %s data for %s", data_type, symbol)

    def log_batch_to_mongo(self, data_type, df):
        """Store a full batch of fetched data in MongoDB.

        Groups by symbol and stores one document per symbol,
        keeping the audit trail organised.

        Args:
            data_type: Type of data ('prices', 'fundamentals').
            df: Full DataFrame to log.
        """
        if df is None or df.empty:
            return

        if "symbol" not in df.columns:
            self.log_fetch_to_mongo(data_type, "all", df)
            return

        for symbol, group in df.groupby("symbol"):
            self.log_fetch_to_mongo(data_type, symbol, group)

        logger.info(
            "MongoDB: logged %s batch (%d symbols)",
            data_type,
            df["symbol"].nunique(),
        )

    # Pipeline summary

    def get_table_counts(self):
        """Get current row counts for all team tables.

        Returns:
            dict: Mapping of table name to row count.
        """
        tables = [
            "price_data",
            "financial_data",
            "risk_free_rates",
            "factor_metrics",
            "factor_scores",
            "portfolio_positions",
        ]
        counts = {}
        for table in tables:
            try:
                df = self.pg.read_query(
                    f"SELECT COUNT(*) as cnt FROM {SCHEMA}.{table}"  # nosec B608
                )
                counts[table] = int(df.iloc[0]["cnt"])
            except Exception:
                counts[table] = 0
        return counts
