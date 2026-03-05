"""Database connection module for PostgreSQL, MongoDB, and MinIO.

Provides connection classes for all three storage systems used in the
coursework data pipeline.
"""

import io
import json
import logging

import pandas as pd
from minio import Minio
from minio.error import S3Error
from pymongo import MongoClient
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
import re
from pathlib import Path

logger = logging.getLogger(__name__)


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
        logger.info(
            "Wrote %d rows to %s.%s", len(df), schema, table_name
        )

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

        logger.info(
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
        path = Path(sql_path)
        sql = path.read_text(encoding="utf-8")

        # Remove comments and split into statements
        sql = re.sub(r"(?m)^\s*--.*$", "", sql).strip()
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        with self.engine.connect() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
            conn.commit()

        logger.info("Executed SQL file: %s (%d statements)", sql_path, len(statements))


class MongoConnection:
    """Manages connections and operations on the MongoDB database.

    Args:
        host: MongoDB host address.
        port: MongoDB port number.
    """

    def __init__(self, host, port):
        self.client = MongoClient(host, port)
        logger.info("MongoDB client created for %s:%s", host, port)

    def insert_one(self, db_name, collection, document):
        """Insert a single document into a collection.

        Args:
            db_name: Database name.
            collection: Collection name.
            document: Dictionary to insert.

        Returns:
            The inserted document's ID.
        """
        db = self.client[db_name]
        result = db[collection].insert_one(document)
        logger.info("Inserted document into %s.%s", db_name, collection)
        return result.inserted_id

    def insert_many(self, db_name, collection, documents):
        """Insert multiple documents into a collection.

        Args:
            db_name: Database name.
            collection: Collection name.
            documents: List of dictionaries to insert.

        Returns:
            List of inserted document IDs.
        """
        db = self.client[db_name]
        result = db[collection].insert_many(documents)
        logger.info(
            "Inserted %d documents into %s.%s",
            len(documents), db_name, collection,
        )
        return result.inserted_ids

    def find(self, db_name, collection, query=None):
        """Query documents from a collection.

        Args:
            db_name: Database name.
            collection: Collection name.
            query: Optional filter dictionary. Returns all if None.

        Returns:
            list: Matching documents.
        """
        db = self.client[db_name]
        cursor = db[collection].find(query or {})
        return list(cursor)

    def test_connection(self):
        """Test the MongoDB connection.

        Returns:
            bool: True if connection is successful.
        """
        try:
            self.client.admin.command("ping")
            logger.info("MongoDB connection test passed.")
            return True
        except Exception as e:
            logger.error("MongoDB connection test failed: %s", e)
            return False


class MinioConnection:
    """Manages connections and file operations on MinIO object storage.

    Args:
        host: MinIO host address (host:port).
        access_key: MinIO access key.
        secret_key: MinIO secret key.
        secure: Whether to use HTTPS. Defaults to False for local dev.
    """

    def __init__(self, host, access_key, secret_key, secure=False):
        self.client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        logger.info("MinIO client created for %s", host)

    def _ensure_bucket(self, bucket):
        """Create bucket if it doesn't exist.

        Args:
            bucket: Bucket name.
        """
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
            logger.info("Created MinIO bucket: %s", bucket)

    def upload_json(self, bucket, object_name, data):
        """Upload a Python object as JSON to MinIO.

        Args:
            bucket: Target bucket name.
            object_name: Object key/path in the bucket.
            data: Python object serialisable to JSON.
        """
        self._ensure_bucket(bucket)
        json_bytes = json.dumps(data, default=str).encode("utf-8")
        stream = io.BytesIO(json_bytes)
        self.client.put_object(
            bucket, object_name, stream, len(json_bytes),
            content_type="application/json",
        )
        logger.info("Uploaded %s to bucket %s", object_name, bucket)

    def download_json(self, bucket, object_name):
        """Download a JSON object from MinIO and parse it.

        Args:
            bucket: Source bucket name.
            object_name: Object key/path in the bucket.

        Returns:
            Parsed Python object, or None if not found.
        """
        try:
            response = self.client.get_object(bucket, object_name)
            data = json.loads(response.read().decode("utf-8"))
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise

    def upload_dataframe(self, bucket, object_name, df):
        """Upload a DataFrame as Parquet to MinIO.

        Args:
            bucket: Target bucket name.
            object_name: Object key/path in the bucket.
            df: DataFrame to upload.
        """
        self._ensure_bucket(bucket)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        self.client.put_object(
            bucket, object_name, buffer, len(buffer.getvalue()),
            content_type="application/octet-stream",
        )
        logger.info("Uploaded DataFrame to %s/%s", bucket, object_name)

    def download_dataframe(self, bucket, object_name):
        """Download a Parquet file from MinIO as a DataFrame.

        Args:
            bucket: Source bucket name.
            object_name: Object key/path in the bucket.

        Returns:
            pd.DataFrame, or None if not found.
        """
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return pd.read_parquet(io.BytesIO(data))
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise

    def list_objects(self, bucket, prefix=None):
        """List objects in a bucket with an optional prefix filter.

        Args:
            bucket: Bucket name.
            prefix: Optional prefix to filter objects.

        Returns:
            list: Object names matching the prefix.
        """
        objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def object_exists(self, bucket, object_name):
        """Check if an object exists in a bucket.

        Args:
            bucket: Bucket name.
            object_name: Object key/path.

        Returns:
            bool: True if the object exists.
        """
        try:
            self.client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False

    def test_connection(self):
        """Test the MinIO connection.

        Returns:
            bool: True if connection is successful.
        """
        try:
            self.client.list_buckets()
            logger.info("MinIO connection test passed.")
            return True
        except Exception as e:
            logger.error("MinIO connection test failed: %s", e)
            return False
