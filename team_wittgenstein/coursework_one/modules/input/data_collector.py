"""Data fetcher module for retrieving financial data from external APIs.

Fetches price data, financial statements, and risk-free rates,
caching all data as Parquet files in MinIO with CTL control files.

CTL Pattern:
    Every data file (e.g. prices/AAPL.parquet) has a companion control
    file (prices/AAPL.ctl) that tracks when it was fetched, how many
    rows it contains, and whether it has been loaded into PostgreSQL.
    This makes the pipeline idempotent and resumable.
"""

import logging
import os
from datetime import datetime
from threading import Lock
from time import monotonic, sleep

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

BUCKET = "wittgenstein-cache"
SIMFIN_STATEMENTS_URL = (
    "https://backend.simfin.com/api/v3/companies/statements/compact"
)
SIMFIN_WEIGHTED_SHARES_URL = (
    "https://backend.simfin.com/api/v3/companies/weighted-shares-outstanding"
)

# SEC EDGAR (free, no API key, US-listed companies only)
SEC_EDGAR_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_EDGAR_USER_AGENT = "TeamWittgenstein/1.0 (bigdata-coursework@university.ac.uk)"


class SimFinServerError(Exception):
    """Raised when SimFin returns HTTP 500 so caller can trigger fallback."""


# Map country codes (from company_static) to OECD 3-letter codes
COUNTRY_TO_OECD = {
    "US": "USA",
    "GB": "GBR",
    "CA": "CAN",
    "FR": "FRA",
    "DE": "DEU",
    "CH": "CHE",
    "IT": "ITA",
    "SP": "ESP",
}

# Map country codes to yfinance Treasury/bond yield tickers (fallback)
# These are short-term government bond yield indices
COUNTRY_TO_YIELD_TICKER = {
    "US": "^IRX",    # 13-Week US Treasury Bill
    "GB": "^IRX",    # Use US T-bill as proxy
    "CA": "^IRX",    # Use US T-bill as proxy
    "FR": "^IRX",    # Use US T-bill as proxy
    "DE": "^IRX",    # Use US T-bill as proxy
    "CH": "^IRX",    # Use US T-bill as proxy
    "IT": "^IRX",    # Use US T-bill as proxy
    "SP": "^IRX",    # Use US T-bill as proxy
}


class DataFetcher:
    """Fetches financial data from external APIs with MinIO caching.

    Uses the parquet + CTL (control file) pattern:
    - Data is stored as .parquet files in MinIO
    - Each data file has a companion .ctl JSON file tracking metadata
    - Before fetching, checks if cached data exists via CTL files

    Args:
        minio_conn: MinioConnection instance for caching.
    """

    def __init__(self, minio_conn):
        self.minio = minio_conn
        self.bucket = BUCKET
        self.simfin_api_key = os.getenv("SIMFIN_API_KEY")
        self._simfin_min_interval_seconds = float(
            os.getenv("SIMFIN_MIN_INTERVAL_SECONDS", "0.55")
        )
        self._simfin_last_request_ts = 0.0
        self._simfin_rate_limit_lock = Lock()
        self.minio._ensure_bucket(self.bucket)
        # SEC EDGAR ticker → CIK mapping (lazy-loaded on first use)
        self._ticker_to_cik = {}
        self._edgar_min_interval_seconds = float(
            os.getenv("EDGAR_MIN_INTERVAL_SECONDS", "0.5")
        )
        self._edgar_last_request_ts = 0.0
        self._edgar_rate_limit_lock = Lock()
        # Cache TTL (None = never expire)
        self.cache_ttl_days = None
        # Populated after each fetch — keyed by 'delisted' and 'fetch_error'
        self.price_failures = {}
        self.fundamentals_failures = {}

    # ================================================================
    # CTL file helpers
    # ================================================================

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
        self.minio.upload_json(
            self.bucket, self._ctl_path(data_type, name), ctl
        )

    def _read_ctl(self, data_type, name):
        """Read a CTL control file from MinIO.

        Args:
            data_type: Category of data.
            name: Identifier.

        Returns:
            dict or None: CTL metadata, or None if not cached.
        """
        return self.minio.download_json(
            self.bucket, self._ctl_path(data_type, name)
        )

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
            self.minio.object_exists(
                self.bucket, self._parquet_path(data_type, name)
            )
            and self.minio.object_exists(
                self.bucket, self._ctl_path(data_type, name)
            )
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
                            data_type, name, age_days,
                        )
                        return False
                except Exception:
                    pass

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
                    names_to_mark.add(
                        object_name.split("/", 1)[1].rsplit(".ctl", 1)[0]
                    )

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

    # ================================================================
    # Failure classification
    # ================================================================

    def _classify_missing(self, symbols):
        """Classify why symbols returned no data.

        For each symbol that produced no rows, checks ticker.info to
        determine whether the company is genuinely delisted (expected,
        acceptable) or whether the fetch failed for another reason
        (network error, rate limit — should be investigated).

        A symbol is classified as delisted if ticker.info is empty or
        has no regularMarketPrice (no active trading). Otherwise it is
        classified as a fetch_error (the company exists but data
        could not be retrieved).

        Args:
            symbols: List of symbol strings that returned no data.

        Returns:
            dict with keys:
                'delisted'    — list of symbols confirmed delisted/invalid
                'fetch_error' — list of symbols that should have data
        """
        delisted = []
        fetch_error = []

        for symbol in symbols:
            try:
                info = yf.Ticker(symbol).info
                if not info or info.get("regularMarketPrice") is None:
                    delisted.append(symbol)
                    logger.info("classify_missing: %s appears delisted", symbol)
                else:
                    fetch_error.append(symbol)
                    logger.warning(
                        "classify_missing: %s is active but returned no data "
                        "(possible fetch error)", symbol
                    )
            except Exception as e:
                fetch_error.append(symbol)
                logger.warning(
                    "classify_missing: could not check %s: %s", symbol, e
                )

        logger.info(
            "Missing symbol classification: %d delisted, %d fetch errors",
            len(delisted), len(fetch_error),
        )
        return {"delisted": delisted, "fetch_error": fetch_error}

    # Price data (source: Yahoo Finance)

    def fetch_prices(self, symbols, period="5y"):
        """Fetch daily price data for all symbols.

        Uses yfinance batch download for efficiency. Caches per-symbol
        parquet files in MinIO so individual symbols can be re-fetched
        without re-downloading everything.

        Args:
            symbols: List of stock ticker symbols.
            period: Historical period to fetch (default '5y').

        Returns:
            pd.DataFrame: Combined price data with columns: symbol,
                price_date, open_price, high_price, low_price,
                close_price, adj_close, volume.
        """
        uncached = []
        cached_dfs = []

        for symbol in symbols:
            sym = symbol.strip()
            if self._is_cached("prices", sym):
                df = self._load_cached("prices", sym)
                if df is not None:
                    df = self._dedupe_dataframe("prices", df, name=sym)
                    cached_dfs.append(df)
                    continue
            uncached.append(sym)

        logger.info(
            "Prices: %d cached, %d to fetch", len(cached_dfs), len(uncached)
        )

        if uncached:
            fetched = self._batch_download_prices(uncached, period)
            cached_dfs.extend(fetched)

        if not cached_dfs:
            return pd.DataFrame()

        result = pd.concat(cached_dfs, ignore_index=True)

        # Classify symbols that returned no data at all
        all_symbols = set(s.strip() for s in symbols)
        returned = set(result["symbol"].unique())
        missing = all_symbols - returned
        if missing:
            self.price_failures = self._classify_missing(list(missing))

        return result

    def _batch_download_prices(self, symbols, period):
        """Batch download prices via yfinance and cache per symbol.

        Args:
            symbols: List of symbols to download.
            period: Historical period string.

        Returns:
            list[pd.DataFrame]: Per-symbol DataFrames.
        """
        logger.info(
            "Downloading prices for %d symbols from yfinance...", len(symbols)
        )

        raw = yf.download(
            symbols,
            period=period,
            group_by="ticker",
            threads=True,
            progress=True,
            auto_adjust=False,
        )

        result_dfs = []

        if len(symbols) == 1:
            symbol = symbols[0]
            currency = self._get_price_currency(symbol)
            df = self._reshape_price_df(raw, symbol, currency=currency)
            if df is not None and not df.empty:
                df = self._dedupe_dataframe("prices", df, name=symbol)
                self._cache_dataframe("prices", symbol, df, "yfinance")
                result_dfs.append(df)
        else:
            for symbol in symbols:
                try:
                    symbol_data = raw[symbol].dropna(how="all")
                    currency = self._get_price_currency(symbol)
                    df = self._reshape_price_df(
                        symbol_data, symbol, currency=currency
                    )
                    if df is not None and not df.empty:
                        df = self._dedupe_dataframe("prices", df, name=symbol)
                        self._cache_dataframe(
                            "prices", symbol, df, "yfinance"
                        )
                        result_dfs.append(df)
                except KeyError:
                    logger.warning("No price data returned for %s", symbol)
                except Exception as e:
                    logger.error(
                        "Error processing prices for %s: %s", symbol, e
                    )

        logger.info("Cached prices for %d symbols", len(result_dfs))
        return result_dfs

    def _get_price_currency(self, symbol):
        """Resolve the trading currency for a symbol from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None)
            currency = None
            if fast_info is not None:
                if isinstance(fast_info, dict):
                    currency = fast_info.get("currency")
                else:
                    currency = getattr(fast_info, "currency", None)
            if not currency:
                info = getattr(ticker, "info", {}) or {}
                currency = info.get("currency")
            if pd.notna(currency):
                currency = str(currency).strip().upper()
                return currency[:3] if currency else None
        except Exception as e:
            logger.debug("Could not resolve currency for %s: %s", symbol, e)
        return None

    @staticmethod
    def _reshape_price_df(raw_df, symbol, currency=None):
        """Transform raw yfinance output into our schema format.

        Args:
            raw_df: Raw DataFrame from yfinance.
            symbol: Stock ticker symbol.

        Returns:
            pd.DataFrame with standardised columns, or None.
        """
        if raw_df is None or raw_df.empty:
            return None

        # Flatten MultiIndex columns (yfinance returns these for single-symbol downloads)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df = raw_df.copy()
            raw_df.columns = raw_df.columns.get_level_values(0)

        df = raw_df.reset_index()

        column_map = {
            "Date": "trade_date",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
        df = df.rename(columns=column_map)
        df["symbol"] = symbol
        df["currency"] = currency

        keep = [
            "symbol", "trade_date", "open_price", "high_price",
            "low_price", "close_price", "adjusted_close", "currency", "volume",
        ]
        return df[[c for c in keep if c in df.columns]]

    # Fundamentals

    def fetch_fundamentals(
        self,
        symbols,
        period="5y",
        source="waterfall",
    ):
        """Fetch quarterly financial statements for all symbols.

        Supports source routing:
        - waterfall: EDGAR → SimFin → yfinance → forward fill.
        - simfin: SimFin only.

        Per-symbol parquet files are cached in MinIO.

        Args:
            symbols: List of stock ticker symbols.
            period: Historical window to keep (e.g. '1y', '5y', 'max').
            source: Fundamentals data source strategy.

        Returns:
            pd.DataFrame: Combined financial data with columns: symbol,
                fiscal_year, fiscal_quarter, report_date, total_assets,
                total_debt, net_income, eps, book_equity,
                shares_outstanding.
        """
        source = self._normalize_fundamentals_source(source)
        to_refresh = []
        cached_dfs = []

        for symbol in symbols:
            sym = symbol.strip()
            cache_name = self._fundamentals_cache_name(sym)
            if self._is_cached("fundamentals", cache_name):
                df = self._load_cached("fundamentals", cache_name)
                if df is not None:
                    df = self._ensure_fundamentals_schema(df)
                    df = self._dedupe_dataframe("fundamentals", df, name=sym)
                    cached_dfs.append(
                        self._apply_fundamentals_period(df, period)
                    )
                    continue
            to_refresh.append(sym)

        logger.info(
            "Fundamentals: %d cache-ready, %d to fetch/refresh",
            len(cached_dfs), len(to_refresh),
        )

        if to_refresh:
            fetched, failed = self._parallel_fetch_fundamentals(
                to_refresh,
                period,
                source,
            )
            cached_dfs.extend(fetched)
            if failed:
                self.fundamentals_failures = self._classify_missing(failed)
            else:
                self.fundamentals_failures = {}
        else:
            self.fundamentals_failures = {}

        if not cached_dfs:
            return pd.DataFrame()

        out = pd.concat(cached_dfs, ignore_index=True)
        out = self._ensure_fundamentals_schema(out)
        out = self._dedupe_dataframe("fundamentals", out)
        return out

    def _parallel_fetch_fundamentals(
        self,
        symbols,
        period,
        source,
    ):
        """Fetch fundamentals sequentially (rate-limit safe).

        Args:
            symbols: Symbols to fetch.
            period: Historical window to keep.
            source: Fundamentals data source strategy.

        Returns:
            list[pd.DataFrame]
        """
        result_dfs = []
        failed = []

        logger.info(
            "Fetching fundamentals for %d symbols sequentially "
            "(source=%s).",
            len(symbols),
            source,
        )
        for symbol in symbols:
            try:
                df = self._fetch_single_fundamental(
                    symbol,
                    period=period,
                    source=source,
                )
                if df is not None and not df.empty:
                    df = self._ensure_fundamentals_schema(df)
                    df = self._dedupe_dataframe("fundamentals", df, name=symbol)
                    cache_source = ",".join(
                        sorted(df["source"].dropna().astype(str).unique())
                    ) or "unknown"
                    cache_name = self._fundamentals_cache_name(symbol)
                    self._cache_dataframe(
                        "fundamentals", cache_name, df, cache_source
                    )
                    result_dfs.append(df)
                else:
                    failed.append(symbol)
            except Exception as e:
                logger.error("Failed fundamentals for %s: %s", symbol, e)
                failed.append(symbol)

        logger.info(
            "Fundamentals: %d success, %d failed",
            len(result_dfs), len(failed),
        )
        if failed:
            logger.warning("Failed symbols (first 20): %s", failed[:20])

        return result_dfs, failed

    @staticmethod
    def _normalize_fundamentals_source(source):
        """Validate and normalise fundamentals source parameter."""
        source = (source or "waterfall").strip().lower()
        allowed = {"simfin", "waterfall"}
        if source not in allowed:
            raise ValueError(
                "Invalid fundamentals source "
                f"'{source}'. Expected one of {sorted(allowed)}."
            )
        return source

    def _fetch_single_fundamental(
        self, symbol, period="5y", source="waterfall"
    ):
        """Fetch one symbol using the requested source strategy.

        Args:
            symbol: Stock ticker symbol.
            period: Historical window to keep.
            source: Fundamentals data source strategy.
                'waterfall' — EDGAR → SimFin → yfinance → forward fill.
                'simfin'    — SimFin only.

        Returns:
            pd.DataFrame or None.
        """
        source = self._normalize_fundamentals_source(source)

        if source == "waterfall":
            return self._fetch_waterfall_fundamentals(symbol, period)

        try:
            df = self._fetch_simfin_fundamentals(symbol)
        except SimFinServerError:
            logger.warning("SimFin HTTP 500 for %s; skipping.", symbol)
            df = None
        df = self._ensure_fundamentals_schema(df)

        if df is None or df.empty:
            return None

        df = self._ensure_fundamentals_schema(df)
        df = self._apply_fundamentals_period(df, period)
        return df.reset_index(drop=True)

    # ================================================================
    # Waterfall: EDGAR → SimFin → yfinance → forward fill
    # ================================================================

    def _fetch_waterfall_fundamentals(self, symbol, period="5y"):
        """Fetch fundamentals using the waterfall pattern.

        Tries sources in priority order: EDGAR → SimFin → yfinance.
        For each (symbol, fiscal_year, fiscal_quarter), null fields are
        filled from the next available source. Remaining nulls are
        forward-filled from the most recent non-null quarter.

        Args:
            symbol: Stock ticker symbol.
            period: Historical window to keep.

        Returns:
            pd.DataFrame or None.
        """
        sources = []

        # 1. SEC EDGAR (free, no API key, US-listed only)
        edgar_df = self._fetch_edgar_fundamentals(symbol)
        if edgar_df is not None and not edgar_df.empty:
            sources.append(("edgar", edgar_df))
            logger.info("Waterfall %s: EDGAR returned %d rows", symbol, len(edgar_df))

        # 2. SimFin
        try:
            simfin_df = self._fetch_simfin_fundamentals(symbol)
            if simfin_df is not None and not simfin_df.empty:
                sources.append(("simfin", simfin_df))
                logger.info("Waterfall %s: SimFin returned %d rows", symbol, len(simfin_df))
        except SimFinServerError:
            logger.warning("Waterfall %s: SimFin HTTP 500, skipping.", symbol)

        # 3. yfinance (lowest priority fallback)
        yf_df = self._fetch_yfinance_fundamentals(symbol)
        if yf_df is not None and not yf_df.empty:
            sources.append(("yfinance", yf_df))
            logger.info("Waterfall %s: yfinance returned %d rows", symbol, len(yf_df))

        if not sources:
            logger.warning("Waterfall %s: all sources returned no data.", symbol)
            return None

        # 4. Merge per-field in priority order
        merged = self._merge_waterfall(sources)

        # 5. Forward fill remaining nulls from previous quarters
        merged = self._forward_fill_fundamentals(merged)

        merged = self._ensure_fundamentals_schema(merged)
        if merged is None or merged.empty:
            return None

        merged = self._apply_fundamentals_period(merged, period)
        return merged.reset_index(drop=True)

    # ================================================================
    # SEC EDGAR infrastructure (rate-limited, retried, per-concept)
    # ================================================================

    def _edgar_throttle_wait(self):
        """Rate limit EDGAR requests (~2 req/sec)."""
        with self._edgar_rate_limit_lock:
            elapsed = monotonic() - self._edgar_last_request_ts
            if elapsed < self._edgar_min_interval_seconds:
                sleep(self._edgar_min_interval_seconds - elapsed)
            self._edgar_last_request_ts = monotonic()

    def _edgar_get_json(self, url, timeout=30, max_retries=3, allow_not_found=False):
        """Call SEC EDGAR API with throttling and retry.

        Args:
            url: EDGAR API endpoint.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            allow_not_found: If True, return None on 404 instead of retrying.

        Returns:
            dict or None: Parsed JSON response, or None on failure.
        """
        headers = {"User-Agent": SEC_EDGAR_USER_AGENT}
        for attempt in range(max_retries):
            try:
                self._edgar_throttle_wait()
                resp = requests.get(url, headers=headers, timeout=timeout)
                if resp.status_code == 404 and allow_not_found:
                    return None
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 2 * (attempt + 1)))
                    logger.warning("EDGAR 429 rate limited, waiting %ds", wait)
                    sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logger.warning(
                        "EDGAR request failed after %d retries: %s", max_retries, e
                    )
                    return None
                sleep(2 ** attempt)
        return None

    def _resolve_cik(self, symbol):
        """Resolve a ticker symbol to its 10-digit SEC CIK string.

        Lazy-loads the SEC ticker-to-CIK mapping on first call.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            str or None: Zero-padded 10-digit CIK, or None if not found.
        """
        if not self._ticker_to_cik:
            data = self._edgar_get_json(SEC_EDGAR_COMPANY_TICKERS_URL)
            if not isinstance(data, dict):
                logger.warning("Failed to load SEC ticker map")
                return None
            self._ticker_to_cik = {
                str(entry.get("ticker", "")).strip().upper(): str(entry.get("cik_str", "")).zfill(10)
                for entry in data.values()
                if entry.get("ticker") and entry.get("cik_str") is not None
            }
            logger.info("Loaded SEC ticker map: %d tickers", len(self._ticker_to_cik))

        return self._ticker_to_cik.get(symbol.upper())

    def _edgar_get_fiscal_periods(self, cik, cutoff=None):
        """Build fiscal year/quarter index from EDGAR submissions.

        Uses 10-K filing dates as fiscal year-end anchors and numbers
        quarters relative to those boundaries. This correctly handles
        companies with non-calendar fiscal years.

        Args:
            cik: 10-digit CIK string.
            cutoff: Optional earliest report_date to include.

        Returns:
            pd.DataFrame with columns: report_date_str, report_date,
                fiscal_year, fiscal_quarter.
        """
        empty = pd.DataFrame(
            columns=["report_date_str", "report_date", "fiscal_year", "fiscal_quarter"]
        )
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        payload = self._edgar_get_json(url)
        if not payload:
            return empty

        filings = (payload or {}).get("filings", {}).get("recent", {})
        df = pd.DataFrame({
            "form": filings.get("form", []),
            "report_date": filings.get("reportDate", []),
            "filed": filings.get("filingDate", []),
        })
        if df.empty:
            return empty

        df = df[df["form"].isin(["10-Q", "10-K"])].copy()
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        df = df.dropna(subset=["report_date"]).sort_values("report_date").reset_index(drop=True)

        if cutoff is not None:
            cutoff = pd.Timestamp(cutoff).tz_localize(None)
            df = df[df["report_date"] >= cutoff]
        if df.empty:
            return empty

        # Use 10-K dates as fiscal year-end anchors
        tenk_rows = df[df["form"] == "10-K"].sort_values("report_date")
        year_counts = {}
        for _, row in tenk_rows.iterrows():
            year = int(row["report_date"].year)
            year_counts[year] = year_counts.get(year, 0) + 1

        year_seen = {}
        tenk_anchors = []
        for _, row in tenk_rows.iterrows():
            year = int(row["report_date"].year)
            year_seen[year] = year_seen.get(year, 0) + 1
            if year_counts.get(year, 0) > 1:
                label = year if year_seen[year] == year_counts[year] else year - 1
            else:
                label = year - 1 if int(row["report_date"].month) <= 6 else year
            tenk_anchors.append((row["report_date"], label))
        tenk_anchor_dates = [item[0] for item in tenk_anchors]

        def fiscal_year_for_row(report_date):
            future = [(d, lbl) for d, lbl in tenk_anchors if d >= report_date]
            if future:
                return min(future, key=lambda x: x[0])[1]
            if tenk_anchors:
                return tenk_anchors[-1][1] + 1
            return int(report_date.year)

        df["fiscal_year"] = df["report_date"].apply(fiscal_year_for_row)
        df["fiscal_quarter"] = None

        for fiscal_year, group in df.groupby("fiscal_year"):
            anchors = [(d, lbl) for d, lbl in tenk_anchors if lbl == fiscal_year]
            if anchors:
                fiscal_year_end = max(anchors, key=lambda x: x[0])[0]
                prev = [d for d in tenk_anchor_dates if d < fiscal_year_end]
                fiscal_year_start = max(prev) if prev else pd.Timestamp("1900-01-01")
            else:
                fiscal_year_end = pd.Timestamp("2999-12-31")
                fiscal_year_start = pd.Timestamp("1900-01-01")

            quarters = group[
                (group["form"] == "10-Q")
                & (group["report_date"] > fiscal_year_start)
                & (group["report_date"] <= fiscal_year_end)
            ].sort_values("report_date")
            for index, row_index in enumerate(quarters.index):
                if index < 3:
                    df.loc[row_index, "fiscal_quarter"] = index + 1
            df.loc[group[group["form"] == "10-K"].index, "fiscal_quarter"] = 4

        df = df.dropna(subset=["fiscal_quarter"]).copy()
        if df.empty:
            return empty
        df["fiscal_quarter"] = pd.to_numeric(df["fiscal_quarter"], errors="coerce").astype("Int64")
        df["report_date_str"] = df["report_date"].dt.strftime("%Y-%m-%d")
        return df[
            ["report_date_str", "report_date", "fiscal_year", "fiscal_quarter"]
        ].reset_index(drop=True)

    def _edgar_fetch_concept(self, cik, tag, unit="USD", cutoff=None):
        """Fetch a single XBRL concept from EDGAR companyconcept API.

        Args:
            cik: 10-digit CIK string.
            tag: XBRL concept tag (e.g. 'Assets', 'NetIncomeLoss').
            unit: Unit key (e.g. 'USD', 'shares', 'USD/shares').
            cutoff: Optional earliest end date to include.

        Returns:
            pd.DataFrame with columns: end, start, val, filed.
        """
        empty = pd.DataFrame(columns=["end", "start", "val", "filed"])
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        payload = self._edgar_get_json(url, allow_not_found=True)
        if not payload:
            return empty

        rows = []
        for row in (payload.get("units", {}) or {}).get(unit, []):
            if row.get("form") not in ("10-Q", "10-K"):
                continue
            end = row.get("end")
            if not end:
                continue
            end_ts = pd.to_datetime(end, errors="coerce")
            if pd.isna(end_ts):
                continue
            if cutoff is not None and end_ts < pd.Timestamp(cutoff):
                continue
            rows.append({
                "end": end,
                "start": row.get("start"),
                "val": row.get("val"),
                "filed": row.get("filed"),
            })
        if not rows:
            return empty

        out = pd.DataFrame(rows)
        out["filed"] = pd.to_datetime(out["filed"], errors="coerce")
        out = out.sort_values("filed", ascending=False).drop_duplicates("end", keep="first")
        return out[["end", "start", "val", "filed"]].reset_index(drop=True)

    def _fetch_edgar_fundamentals(self, symbol, period="5y"):
        """Fetch quarterly fundamentals from SEC EDGAR per-concept APIs.

        Uses the submissions API for accurate fiscal period detection and
        the companyconcept API for each financial field with fallback chains.
        Converts cumulative YTD figures to standalone quarterly values.

        Args:
            symbol: Stock ticker symbol.
            period: Historical window (e.g. '5y').

        Returns:
            pd.DataFrame with fundamentals, or empty DataFrame.
        """
        cik = self._resolve_cik(symbol)
        if not cik:
            logger.info("No CIK found for %s; skipping EDGAR.", symbol)
            return pd.DataFrame()

        # Determine cutoff for period filtering
        years_back = self._period_years(period, default_years=5)
        cutoff = None
        if years_back is not None:
            cutoff = (
                pd.Timestamp.now() - pd.DateOffset(years=years_back + 1)
            )

        # Build fiscal period index from submissions
        periods = self._edgar_get_fiscal_periods(cik, cutoff=cutoff)
        if periods.empty:
            return pd.DataFrame()

        result = periods.copy()
        result["symbol"] = symbol

        def merge_field(tag, field, unit="USD"):
            nonlocal result
            concept = self._edgar_fetch_concept(cik, tag, unit=unit, cutoff=cutoff)
            if concept.empty:
                result[field] = None
                return
            concept = concept.rename(columns={"val": field, "end": "report_date_str"})
            result = result.merge(
                concept[["report_date_str", field]],
                on="report_date_str",
                how="left",
            )

        # --- total_assets ---
        merge_field("Assets", "total_assets")

        # --- net_income (with ProfitLoss fallback) ---
        merge_field("NetIncomeLoss", "net_income")
        if "net_income" in result.columns and result["net_income"].isna().any():
            fallback = self._edgar_fetch_concept(cik, "ProfitLoss", unit="USD", cutoff=cutoff)
            if not fallback.empty:
                fallback = fallback.rename(columns={"val": "_ni_fb", "end": "report_date_str"})
                result = result.merge(
                    fallback[["report_date_str", "_ni_fb"]],
                    on="report_date_str", how="left",
                )
                mask = result["net_income"].isna() & result["_ni_fb"].notna()
                result.loc[mask, "net_income"] = result.loc[mask, "_ni_fb"]
                result = result.drop(columns=["_ni_fb"])

        # --- book_equity (with broader equity concept fallback) ---
        eq_primary = self._edgar_fetch_concept(
            cik,
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            unit="USD", cutoff=cutoff,
        )
        eq_fallback = self._edgar_fetch_concept(
            cik, "StockholdersEquity", unit="USD", cutoff=cutoff,
        )
        if not eq_primary.empty:
            eq_df = eq_primary.rename(columns={"val": "book_equity", "end": "report_date_str"})
            result = result.merge(
                eq_df[["report_date_str", "book_equity"]],
                on="report_date_str", how="left",
            )
        else:
            result["book_equity"] = None
        if not eq_fallback.empty and result["book_equity"].isna().any():
            eq_fb = eq_fallback.rename(columns={"val": "_eq_fb", "end": "report_date_str"})
            result = result.merge(
                eq_fb[["report_date_str", "_eq_fb"]],
                on="report_date_str", how="left",
            )
            mask = result["book_equity"].isna() & result["_eq_fb"].notna()
            result.loc[mask, "book_equity"] = result.loc[mask, "_eq_fb"]
            result = result.drop(columns=["_eq_fb"])

        # --- shares_outstanding (two concept fallbacks) ---
        result["shares_outstanding"] = None
        for tag in ["CommonStockSharesOutstanding",
                     "WeightedAverageNumberOfDilutedSharesOutstanding"]:
            if not result["shares_outstanding"].isna().any():
                break
            shares = self._edgar_fetch_concept(cik, tag, unit="shares", cutoff=cutoff)
            if shares.empty:
                continue
            shares = shares.rename(columns={"val": "_shares_tmp", "end": "report_date_str"})
            result = result.merge(
                shares[["report_date_str", "_shares_tmp"]],
                on="report_date_str", how="left",
            )
            mask = result["shares_outstanding"].isna() & result["_shares_tmp"].notna()
            result.loc[mask, "shares_outstanding"] = result.loc[mask, "_shares_tmp"]
            result = result.drop(columns=["_shares_tmp"])

        # --- eps (diluted with basic fallback) ---
        eps_df = self._edgar_fetch_concept(
            cik, "EarningsPerShareDiluted", unit="USD/shares", cutoff=cutoff,
        )
        if eps_df.empty:
            eps_df = self._edgar_fetch_concept(
                cik, "EarningsPerShareBasic", unit="USD/shares", cutoff=cutoff,
            )
        if not eps_df.empty:
            eps_df = eps_df.copy()
            eps_df["period_days"] = (
                pd.to_datetime(eps_df["end"], errors="coerce")
                - pd.to_datetime(eps_df["start"], errors="coerce")
            ).dt.days
            eps_df = eps_df.sort_values("period_days", ascending=False).drop_duplicates(
                "end", keep="first"
            )
            eps_df = eps_df.rename(columns={"val": "eps", "end": "report_date_str"})
            result = result.merge(
                eps_df[["report_date_str", "eps"]],
                on="report_date_str", how="left",
            )
        else:
            result["eps"] = None

        # --- total_debt (3-tier fallback) ---
        merge_field("LongTermDebt", "total_debt")
        if result["total_debt"].isna().any():
            debt_lease = self._edgar_fetch_concept(
                cik, "LongTermDebtAndCapitalLeaseObligations",
                unit="USD", cutoff=cutoff,
            )
            if not debt_lease.empty:
                debt_lease = debt_lease.rename(
                    columns={"val": "_dl", "end": "report_date_str"}
                )
                result = result.merge(
                    debt_lease[["report_date_str", "_dl"]],
                    on="report_date_str", how="left",
                )
                mask = result["total_debt"].isna() & result["_dl"].notna()
                result.loc[mask, "total_debt"] = result.loc[mask, "_dl"]
                result = result.drop(columns=["_dl"])

        if result["total_debt"].isna().any():
            lt_nc = self._edgar_fetch_concept(
                cik, "LongTermDebtNoncurrent", unit="USD", cutoff=cutoff,
            )
            lt_c = self._edgar_fetch_concept(
                cik, "LongTermDebtCurrent", unit="USD", cutoff=cutoff,
            )
            if not lt_nc.empty:
                lt_nc = lt_nc.rename(columns={"val": "_nc", "end": "report_date_str"})
                result = result.merge(
                    lt_nc[["report_date_str", "_nc"]],
                    on="report_date_str", how="left",
                )
            else:
                result["_nc"] = None
            if not lt_c.empty:
                lt_c = lt_c.rename(columns={"val": "_c", "end": "report_date_str"})
                result = result.merge(
                    lt_c[["report_date_str", "_c"]],
                    on="report_date_str", how="left",
                )
            else:
                result["_c"] = None

            nc_num = pd.to_numeric(result["_nc"], errors="coerce")
            c_num = pd.to_numeric(result["_c"], errors="coerce")
            both_nan = nc_num.isna() & c_num.isna()
            fill_mask = result["total_debt"].isna() & ~both_nan
            if fill_mask.any():
                result["total_debt"] = pd.to_numeric(result["total_debt"], errors="coerce")
                result.loc[fill_mask, "total_debt"] = (
                    nc_num.loc[fill_mask].fillna(0.0).to_numpy(dtype=float)
                    + c_num.loc[fill_mask].fillna(0.0).to_numpy(dtype=float)
                )
            result = result.drop(columns=["_nc", "_c"])

        # Clean up: drop helper column, set source/currency
        result = result.drop(columns=["report_date_str"], errors="ignore")
        result["source"] = "edgar"
        result["currency"] = "USD"

        # --- YTD → standalone quarterly conversion ---
        # EDGAR 10-Qs often report cumulative YTD for net_income and eps.
        # De-cumulate to get standalone quarter values.
        result = result.sort_values(["fiscal_year", "fiscal_quarter"])
        for field in ["net_income", "eps"]:
            if field not in result.columns:
                continue
            standalone = []
            for _, group in result.groupby("fiscal_year", sort=False):
                group = group.sort_values("fiscal_quarter")
                prev_cumulative = 0
                for quarter, value in zip(
                    group["fiscal_quarter"].tolist(),
                    group[field].tolist(),
                ):
                    if value is None or pd.isna(value):
                        standalone.append(value)
                        prev_cumulative = 0
                    elif int(quarter) == 4:
                        standalone.append(value - prev_cumulative)
                        prev_cumulative = 0
                    else:
                        standalone.append(value - prev_cumulative)
                        prev_cumulative = value
            result[field] = standalone

        # Ensure numeric types
        for col in ["total_assets", "total_debt", "book_equity",
                     "shares_outstanding", "net_income", "eps"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")
        result = self._ensure_fundamentals_schema(result)
        result = result.dropna(subset=["fiscal_year", "fiscal_quarter", "report_date"])
        result = result.sort_values("report_date", ascending=False)
        result = result.drop_duplicates(
            subset=["symbol", "fiscal_year", "fiscal_quarter"],
            keep="first",
        )
        return result.reset_index(drop=True)

    @staticmethod
    def _period_years(period, default_years=5):
        """Parse period string ('5y', 'max') to integer years or None."""
        p = (period or f"{default_years}y").strip().lower()
        if p == "max":
            return None
        if p.endswith("y"):
            try:
                years = int(p[:-1])
                if years > 0:
                    return years
            except ValueError:
                pass
        return default_years

    def _fetch_yfinance_fundamentals(self, symbol):
        """Fetch quarterly fundamentals from yfinance as lowest-priority fallback.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            pd.DataFrame with fundamentals, or empty DataFrame.
        """
        try:
            ticker = yf.Ticker(symbol)
            bs = ticker.quarterly_balance_sheet
            inc = ticker.quarterly_income_stmt
        except Exception as e:
            logger.warning("yfinance fundamentals failed for %s: %s", symbol, e)
            return pd.DataFrame()

        if (bs is None or bs.empty) and (inc is None or inc.empty):
            return pd.DataFrame()

        records = {}

        bs_field_map = [
            ("Total Assets", "total_assets"),
            ("Total Debt", "total_debt"),
            ("Stockholders Equity", "book_equity"),
            ("Total Stockholders Equity", "book_equity"),
            ("Ordinary Shares Number", "shares_outstanding"),
            ("Share Issued", "shares_outstanding"),
        ]
        inc_field_map = [
            ("Net Income", "net_income"),
            ("Diluted EPS", "eps"),
        ]

        if bs is not None and not bs.empty:
            for col_date in bs.columns:
                ts = pd.Timestamp(col_date)
                quarter = int((ts.month - 1) // 3) + 1
                key = (ts.year, quarter)
                if key not in records:
                    records[key] = {
                        "symbol": symbol,
                        "fiscal_year": ts.year,
                        "fiscal_quarter": quarter,
                        "report_date": ts,
                        "currency": "USD",
                        "source": "yfinance",
                    }
                for item, field in bs_field_map:
                    if item in bs.index:
                        val = bs.at[item, col_date]
                        if pd.notna(val) and records[key].get(field) is None:
                            records[key][field] = val

        if inc is not None and not inc.empty:
            for col_date in inc.columns:
                ts = pd.Timestamp(col_date)
                quarter = int((ts.month - 1) // 3) + 1
                key = (ts.year, quarter)
                if key not in records:
                    records[key] = {
                        "symbol": symbol,
                        "fiscal_year": ts.year,
                        "fiscal_quarter": quarter,
                        "report_date": ts,
                        "currency": "USD",
                        "source": "yfinance",
                    }
                for item, field in inc_field_map:
                    if item in inc.index:
                        val = inc.at[item, col_date]
                        if pd.notna(val) and records[key].get(field) is None:
                            records[key][field] = val

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(list(records.values()))
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        return (
            df.sort_values(
                ["fiscal_year", "fiscal_quarter"], ascending=[False, False]
            )
            .reset_index(drop=True)
        )

    def _merge_waterfall(self, source_dfs):
        """Merge DataFrames from multiple sources, filling nulls per-field.

        For each (symbol, fiscal_year, fiscal_quarter), fields that are
        null in the higher-priority source get filled from the next source.

        Args:
            source_dfs: List of (source_name, DataFrame) tuples in
                priority order (highest first).

        Returns:
            pd.DataFrame: Merged fundamentals.
        """
        keys = ["symbol", "fiscal_year", "fiscal_quarter"]
        fill_fields = [
            "total_assets", "total_debt", "net_income", "book_equity",
            "shares_outstanding", "eps", "report_date", "currency",
        ]

        name, result = source_dfs[0]
        result = self._ensure_fundamentals_schema(result.copy())
        result["source"] = name

        for next_name, next_df in source_dfs[1:]:
            next_df = self._ensure_fundamentals_schema(next_df.copy())
            next_cols = keys + [f for f in fill_fields if f in next_df.columns]
            next_subset = next_df[next_cols].copy()

            result = result.merge(
                next_subset, on=keys, how="outer", suffixes=("", "_next")
            )

            for field in fill_fields:
                next_col = f"{field}_next"
                if next_col in result.columns:
                    if field in result.columns:
                        result[field] = result[field].where(
                            result[field].notna(), result[next_col]
                        )
                    else:
                        result[field] = result[next_col]
                    result = result.drop(columns=[next_col])

            # For rows only in next source (outer join), set source
            result["source"] = result["source"].fillna(next_name)

        return result

    @staticmethod
    def _forward_fill_fundamentals(df):
        """Forward fill remaining null fields from previous quarters.

        After waterfall merge, some fields may still be null. This fills
        them using the most recent non-null value for the same symbol,
        sorted chronologically by (fiscal_year, fiscal_quarter).

        Args:
            df: Merged fundamentals DataFrame.

        Returns:
            pd.DataFrame with forward-filled values.
        """
        if df is None or df.empty:
            return df

        fill_cols = [
            "total_assets", "total_debt", "net_income", "book_equity",
            "shares_outstanding", "eps",
        ]
        df = df.sort_values(["symbol", "fiscal_year", "fiscal_quarter"])

        existing = [c for c in fill_cols if c in df.columns]
        if existing:
            # Track which rows had ALL numeric fields null before ffill
            all_null_before = df[existing].isna().all(axis=1)
            df[existing] = df.groupby("symbol")[existing].ffill(limit=2)
            # Mark rows that were entirely filled by forward fill
            if "source" in df.columns:
                df.loc[all_null_before & df[existing].notna().any(axis=1), "source"] = "ffill"

        return df

    @staticmethod
    def _apply_fundamentals_period(df, period):
        """Filter fundamentals to the most recent N quarters per symbol.

        For a period like "5y", keeps the most recent 20 quarters (5 × 4)
        per symbol rather than using a date cutoff, which can miss quarters
        near the boundary.
        """
        if df is None or df.empty:
            return df

        out = df.copy()
        out = out.sort_values(
            ["symbol", "fiscal_year", "fiscal_quarter"],
            ascending=[True, False, False],
        )

        p = (period or "5y").strip().lower()
        if p == "max":
            return out

        if p.endswith("y"):
            try:
                years = int(p[:-1])
                if years > 0:
                    max_quarters = years * 4
                    out = out.groupby("symbol").head(max_quarters)
                    return out
            except ValueError:
                pass
        return out

    def _fetch_simfin_fundamentals(self, symbol):
        """Fetch quarterly fundamentals from SimFin statements + shares."""
        if not self.simfin_api_key:
            logger.warning(
                "SIMFIN_API_KEY is not set; skipping SimFin fetch for %s",
                symbol,
            )
            return pd.DataFrame()
        statements_payload = self._simfin_get(
            SIMFIN_STATEMENTS_URL,
            params={
                "ticker": symbol,
                "statements": "pl,bs,derived",
                "period": "q1,q2,q3,q4",
            },
        )
        shares_payload = self._simfin_get(
            SIMFIN_WEIGHTED_SHARES_URL,
            params={"ticker": symbol, "period": "q1,q2,q3,q4"},
        )

        if not statements_payload:
            return pd.DataFrame()
        statement_rows = []
        for company in statements_payload:
            ticker = company.get("ticker", symbol)
            statement_dfs = {}
            for stmt in company.get("statements", []):
                stmt_name = str(stmt.get("statement", "")).upper()
                data = stmt.get("data") or []
                columns = stmt.get("columns") or []
                if not data or not columns:
                    continue
                df = pd.DataFrame(data, columns=columns)
                df["symbol"] = ticker
                statement_dfs[stmt_name] = df

            pl = self._simfin_statement_frame(
                statement_dfs.get("PL"),
                {
                    "Fiscal Year": "fiscal_year",
                    "Fiscal Period": "fiscal_quarter",
                    "Report Date": "report_date",
                    "Net Income": "net_income",
                },
                extra_cols=["net_income"],
            )
            bs = self._simfin_statement_frame(
                statement_dfs.get("BS"),
                {
                    "Fiscal Year": "fiscal_year",
                    "Fiscal Period": "fiscal_quarter",
                    "Report Date": "report_date",
                    "Total Assets": "total_assets",
                    "Total Equity": "total_equity",
                },
                extra_cols=["total_assets", "total_equity"],
            )
            derived = self._simfin_statement_frame(
                statement_dfs.get("DERIVED"),
                {
                    "Fiscal Year": "fiscal_year",
                    "Fiscal Period": "fiscal_quarter",
                    "Report Date": "report_date",
                    "Total Debt": "total_debt",
                    "Earnings Per Share, Diluted": "eps",
                },
                extra_cols=["total_debt", "eps"],
            )
            keys = ["symbol", "fiscal_year", "fiscal_quarter", "report_date"]
            merged = pl.merge(bs, on=keys, how="outer").merge(
                derived, on=keys, how="outer"
            )
            merged["book_equity"] = merged.get("total_equity", merged.get("book_equity"))
            statement_rows.append(merged)

        if not statement_rows:
            return pd.DataFrame()

        out = pd.concat(statement_rows, ignore_index=True)

        shares_df = self._simfin_weighted_shares_frame(
            shares_payload, default_symbol=symbol
        )
        if not shares_df.empty:
            out = out.merge(
                shares_df,
                on=["symbol", "fiscal_year", "fiscal_quarter"],
                how="left",
            )
        else:
            out["shares_outstanding"] = None

        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
        out["fiscal_quarter"] = out["fiscal_quarter"].apply(
            self._normalize_quarter_value
        )
        for col in [
            "total_assets",
            "total_equity",
            "total_debt",
            "net_income",
            "eps",
            "book_equity",
            "shares_outstanding",
        ]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out["currency"] = None
        out["source"] = "simfin"
        out = self._ensure_fundamentals_schema(out)
        out = out.dropna(subset=["fiscal_year", "fiscal_quarter"])
        out = out.sort_values("report_date", ascending=False)
        out = out.drop_duplicates(
            subset=["symbol", "fiscal_year", "fiscal_quarter"],
            keep="first",
        )
        return out.reset_index(drop=True)

    def _simfin_get(self, url, params, timeout=20, max_retries=3):
        """Call SimFin API with auth header + free-tier throttle."""
        headers = {
            "Authorization": f"api-key {self.simfin_api_key}",
            "accept": "application/json",
        }
        for attempt in range(max_retries):
            try:
                self._simfin_throttle_wait()
                response = requests.get(
                    url, params=params, headers=headers, timeout=timeout
                )
                status_code = response.status_code

                if status_code == 200:
                    return response.json()

                if status_code == 500:
                    logger.warning(
                        "SimFin returned HTTP 500 (no retry) "
                        "(%s, params=%s)",
                        url,
                        params,
                    )
                    raise SimFinServerError(
                        f"SimFin HTTP 500 for {url} params={params}"
                    )

                if status_code == 429:
                    logger.warning(
                        "SimFin rate limited (429) (%s, params=%s)",
                        url,
                        params,
                    )
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep(float(retry_after))
                        except ValueError:
                            sleep(2.0)
                    else:
                        sleep(2.0)
                    continue

                logger.warning(
                    "SimFin request failed with HTTP %s (%s, params=%s)",
                    status_code,
                    url,
                    params,
                )
                sleep(1.5)
            except requests.RequestException as exc:
                logger.warning(
                    "SimFin request failed (%s, params=%s): %s",
                    url,
                    params,
                    exc,
                )
                sleep(1.5)
        logger.warning("SimFin unavailable after retries (%s)", params)
        return None

    def _simfin_throttle_wait(self):
        """Ensure SimFin request spacing (free tier: 2 req/sec)."""
        with self._simfin_rate_limit_lock:
            elapsed = monotonic() - self._simfin_last_request_ts
            if elapsed < self._simfin_min_interval_seconds:
                sleep(self._simfin_min_interval_seconds - elapsed)
            self._simfin_last_request_ts = monotonic()

    def _simfin_statement_frame(self, df, rename_map, extra_cols):
        """Project and rename one SimFin statement frame safely."""
        base_cols = ["symbol"] + list(rename_map.keys())
        out_cols = ["symbol", "fiscal_year", "fiscal_quarter", "report_date"]
        out_cols.extend(extra_cols)
        if df is None or df.empty or not all(col in df.columns for col in base_cols):
            return pd.DataFrame(columns=out_cols)

        out = df[base_cols].copy().rename(columns=rename_map)
        out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce")
        out["fiscal_quarter"] = out["fiscal_quarter"].apply(
            self._normalize_quarter_value
        )
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
        for col in extra_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out[out_cols]

    def _simfin_weighted_shares_frame(self, payload, default_symbol=None):
        """Build quarterly diluted weighted-shares dataframe from SimFin."""
        if not payload:
            return pd.DataFrame(
                columns=["symbol", "fiscal_year", "fiscal_quarter", "shares_outstanding"]
            )
        shares_df = pd.DataFrame(payload)
        if shares_df.empty:
            return pd.DataFrame(
                columns=["symbol", "fiscal_year", "fiscal_quarter", "shares_outstanding"]
            )

        rename_map = {
            "ticker": "symbol",
            "fyear": "fiscal_year",
            "period": "fiscal_quarter",
            "diluted": "shares_outstanding",
            "endDate": "share_end_date",
        }
        shares_df = shares_df.rename(columns=rename_map)

        for col in ["symbol", "fiscal_year", "fiscal_quarter", "shares_outstanding"]:
            if col not in shares_df.columns:
                shares_df[col] = None
        if default_symbol is not None:
            shares_df["symbol"] = shares_df["symbol"].fillna(default_symbol)
        if "share_end_date" not in shares_df.columns:
            shares_df["share_end_date"] = None

        shares_df["fiscal_year"] = pd.to_numeric(
            shares_df["fiscal_year"], errors="coerce"
        )
        shares_df["fiscal_quarter"] = shares_df["fiscal_quarter"].apply(
            self._normalize_quarter_value
        )
        shares_df["share_end_date"] = pd.to_datetime(
            shares_df["share_end_date"], errors="coerce"
        )
        shares_df["shares_outstanding"] = pd.to_numeric(
            shares_df["shares_outstanding"], errors="coerce"
        )
        shares_df = shares_df.dropna(subset=["fiscal_year", "fiscal_quarter"])
        shares_df = shares_df.sort_values("share_end_date")
        shares_df = shares_df.drop_duplicates(
            subset=["symbol", "fiscal_year", "fiscal_quarter"], keep="last"
        )
        return shares_df[
            ["symbol", "fiscal_year", "fiscal_quarter", "shares_outstanding"]
        ]

    @staticmethod
    def _normalize_quarter_value(value):
        """Normalize quarter representations to integer 1..4."""
        if value is None:
            return None
        text = str(value).strip().upper()
        mapping = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        if text in mapping:
            return mapping[text]
        try:
            q = int(float(text))
            return q if q in (1, 2, 3, 4) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ensure_fundamentals_schema(df):
        """Ensure fundamentals DataFrame conforms to expected output schema."""
        columns = [
            "symbol",
            "fiscal_year",
            "fiscal_quarter",
            "report_date",
            "currency",
            "total_assets",
            "total_debt",
            "net_income",
            "book_equity",
            "shares_outstanding",
            "eps",
            "source",
        ]
        if df is None or df.empty:
            return pd.DataFrame(columns=columns)

        out = df.copy()

        # Handle legacy column names
        if "book_value_equity" in out.columns and "book_equity" not in out.columns:
            out = out.rename(columns={"book_value_equity": "book_equity"})
        if "total_equity" in out.columns and "book_equity" not in out.columns:
            out = out.rename(columns={"total_equity": "book_equity"})

        for col in columns:
            if col not in out.columns:
                out[col] = None

        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
        out["fiscal_year"] = pd.to_numeric(
            out["fiscal_year"], errors="coerce"
        ).astype("Int64")
        out["fiscal_quarter"] = pd.to_numeric(
            out["fiscal_quarter"], errors="coerce"
        ).astype("Int64")
        out["shares_outstanding"] = pd.to_numeric(
            out["shares_outstanding"], errors="coerce"
        ).round().astype("Int64")
        out["source"] = out["source"].fillna("unknown")
        return out[columns]

    # Risk-free rates (source: OECD API with yfinance fallback)

    def fetch_risk_free_rates(self, countries):
        """Fetch monthly short-term interest rates for risk-free rate proxy.

        Attempts the OECD SDMX API first. If that fails (unreliable),
        falls back to yfinance Treasury yield data (^IRX = 13-week T-bill).
        Results are cached as a single parquet + CTL in MinIO.

        Args:
            countries: List of country codes from company_static.

        Returns:
            pd.DataFrame with columns: country, rate_date, rate.
        """
        if self._is_cached("risk_free_rates", "all"):
            logger.info("Risk-free rates loaded from cache.")
            cached = self._load_cached("risk_free_rates", "all")
            return self._dedupe_dataframe("risk_free_rates", cached, name="all")

        unique_countries = list(set(c.strip() for c in countries))

        # Try OECD API first
        result = self._fetch_rates_oecd(unique_countries)

        # Fallback to yfinance Treasury yields
        if result is None or result.empty:
            logger.info("OECD API unavailable, using yfinance fallback.")
            result = self._fetch_rates_yfinance(unique_countries)

        if result is not None and not result.empty:
            result = self._dedupe_dataframe(
                "risk_free_rates", result, name="all"
            )
            self._cache_dataframe("risk_free_rates", "all", result, "oecd")
            logger.info("Fetched risk-free rates: %d rows", len(result))
            return result

        logger.error("No risk-free rate data fetched from any source.")
        return pd.DataFrame()

    def _fetch_rates_oecd(self, countries):
        """Try fetching rates from the OECD SDMX-JSON API.

        Args:
            countries: List of country codes.

        Returns:
            pd.DataFrame or None if API fails.
        """
        all_rates = []
        for country in countries:
            oecd_code = COUNTRY_TO_OECD.get(country)
            if not oecd_code:
                continue
            try:
                url = (
                    f"https://stats.oecd.org/SDMX-JSON/data/"
                    f"MEI_FIN/IRSTCI01.{oecd_code}.M/all"
                    f"?startTime=2020&endTime=2026"
                )
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()

                observations = (
                    data.get("dataSets", [{}])[0]
                    .get("series", {})
                    .get("0:0:0", {})
                    .get("observations", {})
                )
                time_periods = (
                    data.get("structure", {})
                    .get("dimensions", {})
                    .get("observation", [{}])[0]
                    .get("values", [])
                )

                for idx_str, values in observations.items():
                    idx = int(idx_str)
                    if idx < len(time_periods) and values[0] is not None:
                        period = time_periods[idx]
                        all_rates.append({
                            "country": country,
                            "rate_date": pd.Timestamp(period["id"]).date(),
                            "rate": values[0] / 100,
                        })
            except Exception as e:
                logger.warning("OECD failed for %s: %s", country, e)
                return None

        if all_rates:
            return pd.DataFrame(all_rates)
        return None

    def _fetch_rates_yfinance(self, countries):
        """Fetch risk-free rates from yfinance Treasury yield data.

        Downloads the 13-week US Treasury Bill yield (^IRX) and
        converts it to a monthly rate series for each country.

        Args:
            countries: List of country codes.

        Returns:
            pd.DataFrame with columns: country, rate_date, rate.
        """
        logger.info("Fetching Treasury yields from yfinance...")

        try:
            irx = yf.download(
                "^IRX", period="5y", progress=False, auto_adjust=False
            )
        except Exception as e:
            logger.error("Failed to download ^IRX: %s", e)
            return pd.DataFrame()

        if irx is None or irx.empty:
            return pd.DataFrame()

        # Flatten multi-level columns from yfinance
        if isinstance(irx.columns, pd.MultiIndex):
            irx.columns = irx.columns.get_level_values(0)

        irx = irx.reset_index()
        irx = irx[["Date", "Close"]].dropna()
        irx = irx.rename(columns={"Date": "rate_date", "Close": "rate"})

        # ^IRX is quoted as annualised %, convert to decimal
        irx["rate"] = irx["rate"] / 100

        # Resample to month-end to get monthly rates
        irx["rate_date"] = pd.to_datetime(irx["rate_date"])
        irx = irx.set_index("rate_date").resample("ME").last().reset_index()

        all_rates = []
        for country in countries:
            country_df = irx.copy()
            country_df["country"] = country
            all_rates.append(country_df)

        result = pd.concat(all_rates, ignore_index=True)
        result = result[["country", "rate_date", "rate"]]
        return result
