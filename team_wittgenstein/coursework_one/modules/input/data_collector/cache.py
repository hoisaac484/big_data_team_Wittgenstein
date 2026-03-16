"""Cache management using MinIO with CTL control files."""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class CacheMixin:
    """MinIO caching methods for DataFetcher."""

    def _ctl_path(self, data_type, name):
        """Build the CTL file path in MinIO.

        Args:
            data_type: Category ('prices', 'fundamentals', 'risk_free_rates').
            name: Identifier (usually ticker symbol or 'all').

        Returns:
            str: Object path like 'prices/AAPL.ctl'.
        """
        return f"{data_type}/{name}.ctl"

    def _parquet_path(self, data_type, name):
        """Build the Parquet file path in MinIO.

        Args:
            data_type: Category of data.
            name: Identifier.

        Returns:
            str: Object path like 'prices/AAPL.parquet'.
        """
        return f"{data_type}/{name}.parquet"

    def _write_ctl(self, data_type, name, rows, source):
        """Write a CTL control file to MinIO.

        Args:
            data_type: Category of data.
            name: Identifier.
            rows: Number of rows in the companion data file.
            source: Data source name (e.g. 'yfinance', 'oecd').
        """
        ctl = {
            "name": name,
            "data_type": data_type,
            "fetched_at": datetime.utcnow().isoformat(),
            "rows": rows,
            "source": source,
            "loaded_to_postgres": False,
        }
        self.minio.upload_json(self.bucket, self._ctl_path(data_type, name), ctl)

    def _read_ctl(self, data_type, name):
        """Read a CTL control file from MinIO.

        Args:
            data_type: Category of data.
            name: Identifier.

        Returns:
            dict or None: CTL metadata, or None if not cached.
        """
        return self.minio.download_json(self.bucket, self._ctl_path(data_type, name))

    def _is_cached(self, data_type, name):
        """Check if data is already cached in MinIO.

        Returns False if files don't exist or if cache has expired
        (based on cache_ttl_days setting).

        Args:
            data_type: Category of data.
            name: Identifier.

        Returns:
            bool: True if both parquet and CTL files exist and are fresh.
        """
        if not (
            self.minio.object_exists(self.bucket, self._parquet_path(data_type, name))
            and self.minio.object_exists(self.bucket, self._ctl_path(data_type, name))
        ):
            return False

        # Check TTL if configured
        if self.cache_ttl_days is not None:
            ctl = self._read_ctl(data_type, name)
            if ctl and "fetched_at" in ctl:
                try:
                    fetched = pd.to_datetime(ctl["fetched_at"])
                    age_days = (pd.Timestamp.now() - fetched).days
                    if age_days > self.cache_ttl_days:
                        logger.info(
                            "Cache expired for %s/%s (%d days old)",
                            data_type,
                            name,
                            age_days,
                        )
                        return False
                except Exception:
                    logger.debug(
                        "Could not parse fetched_at for %s/%s; treating cache as valid",
                        data_type,
                        name,
                    )

        return True

    def _dedupe_dataframe(self, data_type, df, name=None):
        """Drop duplicate business keys for a dataframe type.

        Args:
            data_type: Category of data.
            df: DataFrame to deduplicate.
            name: Optional identifier used in logs.

        Returns:
            pd.DataFrame: Deduplicated dataframe copy.
        """
        if df is None or df.empty:
            return df

        df = df.copy()
        if data_type == "prices":
            subset = ["symbol", "trade_date"]
        elif data_type == "fundamentals":
            subset = ["symbol", "fiscal_year", "fiscal_quarter"]
        elif data_type == "risk_free_rates":
            subset = ["country", "rate_date"]
        else:
            subset = None

        if subset and all(col in df.columns for col in subset):
            before = len(df)
            df = df.drop_duplicates(subset=subset, keep="last")
            dropped = before - len(df)
            if dropped > 0:
                target = f"{data_type}/{name}" if name else data_type
                logger.info(
                    "Dedupe %s: dropped %d duplicate rows by key %s",
                    target,
                    dropped,
                    subset,
                )
        return df

    def _cache_dataframe(self, data_type, name, df, source):
        """Save a DataFrame as parquet with a companion CTL file.

        Args:
            data_type: Category of data.
            name: Identifier.
            df: DataFrame to cache.
            source: Data source name.
        """
        # Ensure cache file never stores duplicate business keys.
        df = self._dedupe_dataframe(data_type, df, name=name)

        self.minio.upload_dataframe(
            self.bucket, self._parquet_path(data_type, name), df
        )
        self._write_ctl(data_type, name, len(df), source)

    def _load_cached(self, data_type, name):
        """Load a cached DataFrame from MinIO.

        Args:
            data_type: Category of data.
            name: Identifier.

        Returns:
            pd.DataFrame or None.
        """
        return self.minio.download_dataframe(
            self.bucket, self._parquet_path(data_type, name)
        )

    def _fundamentals_cache_name(self, symbol):
        """Build fundamentals cache key name (symbol only)."""
        return f"{symbol}"

    def mark_loaded(self, data_type, name):
        """Update a CTL file to mark data as loaded into PostgreSQL.

        Args:
            data_type: Category of data.
            name: Identifier.
        """
        if data_type != "fundamentals":
            ctl = self._read_ctl(data_type, name)
            if ctl:
                ctl["loaded_to_postgres"] = True
                ctl["loaded_at"] = datetime.utcnow().isoformat()
                self.minio.upload_json(
                    self.bucket, self._ctl_path(data_type, name), ctl
                )
            return

        # Backward-compatible marking for fundamentals:
        # - exact source-scoped cache key (e.g. AAPL.simfin)
        # - legacy symbol-only key (e.g. AAPL)
        names_to_mark = set()
        if "." in str(name):
            names_to_mark.add(name)
        else:
            names_to_mark.add(name)
            for object_name in self.minio.list_objects(
                self.bucket, prefix=f"fundamentals/{name}."
            ):
                if object_name.endswith(".ctl"):
                    names_to_mark.add(object_name.split("/", 1)[1].rsplit(".ctl", 1)[0])

        for cache_name in names_to_mark:
            ctl = self._read_ctl(data_type, cache_name)
            if not ctl:
                continue
            ctl["loaded_to_postgres"] = True
            ctl["loaded_at"] = datetime.utcnow().isoformat()
            self.minio.upload_json(
                self.bucket, self._ctl_path(data_type, cache_name), ctl
            )

    def delete_symbol_cache(self, symbol):
        """Delete cached price and fundamentals objects for a symbol."""
        removed = 0
        symbol = str(symbol).strip()
        if not symbol:
            return removed

        for data_type in ("prices", "fundamentals"):
            for object_name in (
                self._parquet_path(data_type, symbol),
                self._ctl_path(data_type, symbol),
            ):
                if self.minio.delete_object(self.bucket, object_name):
                    removed += 1

        for object_name in self.minio.list_objects(
            self.bucket, prefix=f"fundamentals/{symbol}."
        ):
            if self.minio.delete_object(self.bucket, object_name):
                removed += 1

        if removed > 0:
            logger.info("Deleted %d cached objects for %s", removed, symbol)
        return removed
