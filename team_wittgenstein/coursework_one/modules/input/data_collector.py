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
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
SIMFIN_STATEMENTS_URL = (
    "https://backend.simfin.com/api/v3/companies/statements/compact"
)
SIMFIN_WEIGHTED_SHARES_URL = (
    "https://backend.simfin.com/api/v3/companies/weighted-shares-outstanding"
)


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
        self.alpha_vantage_api_key = (
            os.getenv("ALPHA_VANTAGE_API_KEY")
            or os.getenv("ALPHAVANTAGE_API_KEY")
        )
        self._av_min_interval_seconds = float(
            os.getenv("ALPHA_VANTAGE_MIN_INTERVAL_SECONDS", "1.1")
        )
        self._av_last_request_ts = 0.0
        self._av_rate_limit_lock = Lock()
        self.simfin_api_key = os.getenv("SIMFIN_API_KEY")
        self._simfin_min_interval_seconds = float(
            os.getenv("SIMFIN_MIN_INTERVAL_SECONDS", "0.55")
        )
        self._simfin_last_request_ts = 0.0
        self._simfin_rate_limit_lock = Lock()
        self.minio._ensure_bucket(self.bucket)
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

        Args:
            data_type: Category of data.
            name: Identifier.

        Returns:
            bool: True if both parquet and CTL files exist.
        """
        return (
            self.minio.object_exists(
                self.bucket, self._parquet_path(data_type, name)
            )
            and self.minio.object_exists(
                self.bucket, self._ctl_path(data_type, name)
            )
        )

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
        # - exact source-scoped cache key (e.g. AAPL.alphavantage)
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
            df = self._reshape_price_df(raw, symbol)
            if df is not None and not df.empty:
                df = self._dedupe_dataframe("prices", df, name=symbol)
                self._cache_dataframe("prices", symbol, df, "yfinance")
                result_dfs.append(df)
        else:
            for symbol in symbols:
                try:
                    symbol_data = raw[symbol].dropna(how="all")
                    df = self._reshape_price_df(symbol_data, symbol)
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

    @staticmethod
    def _reshape_price_df(raw_df, symbol):
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
        df["source"] = "yfinance"

        keep = [
            "symbol", "trade_date", "open_price", "high_price",
            "low_price", "close_price", "adjusted_close", "volume", "source",
        ]
        return df[[c for c in keep if c in df.columns]]

    # Fundamentals (source: SimFin with Alpha Vantage backup)

    def fetch_fundamentals(
        self,
        symbols,
        period="5y",
        source="simfin",
    ):
        """Fetch quarterly financial statements for all symbols.

        Supports source routing:
        - simfin: SimFin primary with Alpha Vantage fallback on SimFin HTTP 500.
        - alphavantage: Alpha Vantage only (no fallback).

        Per-symbol parquet files are cached in MinIO.

        Args:
            symbols: List of stock ticker symbols.
            period: Historical window to keep (e.g. '1y', '5y', 'max').
            source: Fundamentals data source strategy.

        Returns:
            pd.DataFrame: Combined financial data with columns: symbol,
                fiscal_year, fiscal_quarter, report_date, total_assets,
                total_equity, total_debt, net_income, eps, book_value_equity,
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
        source = (source or "simfin").strip().lower()
        allowed = {"simfin", "alphavantage"}
        if source not in allowed:
            raise ValueError(
                "Invalid fundamentals source "
                f"'{source}'. Expected one of {sorted(allowed)}."
            )
        return source

    def _fetch_single_fundamental(
        self, symbol, period="5y", source="simfin"
    ):
        """Fetch one symbol using the requested source strategy.

        Args:
            symbol: Stock ticker symbol.
            period: Historical window to keep.
            source: Fundamentals data source strategy.

        Returns:
            pd.DataFrame or None.
        """
        source = self._normalize_fundamentals_source(source)
        if source == "alphavantage":
            df = self._fetch_alpha_vantage_fundamentals(symbol)
        else:
            try:
                df = self._fetch_simfin_fundamentals(symbol)
            except SimFinServerError:
                logger.warning(
                    "SimFin HTTP 500 for %s; falling back to Alpha Vantage.",
                    symbol,
                )
                df = self._fetch_alpha_vantage_fundamentals(symbol)
        df = self._ensure_fundamentals_schema(df)

        if df is None or df.empty:
            return None

        df = self._ensure_fundamentals_schema(df)
        df = self._apply_fundamentals_period(df, period)
        return df.reset_index(drop=True)

    @staticmethod
    def _apply_fundamentals_period(df, period):
        """Filter fundamentals to a historical lookback window by report_date."""
        if df is None or df.empty:
            return df

        out = df.copy()
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
        out = out.dropna(subset=["report_date"]).sort_values(
            "report_date", ascending=False
        )

        p = (period or "5y").strip().lower()
        if p == "max":
            return out

        if p.endswith("y"):
            try:
                years = int(p[:-1])
                if years > 0:
                    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(
                        years=years
                    )
                    return out[out["report_date"] >= cutoff]
            except ValueError:
                pass
        return out

    def _fetch_alpha_vantage_fundamentals(self, symbol):
        """Fetch quarterly balance sheet + income + earnings data from AV."""
        if not self.alpha_vantage_api_key:
            return pd.DataFrame()

        bs_payload = self._alpha_vantage_get("BALANCE_SHEET", symbol)
        inc_payload = self._alpha_vantage_get("INCOME_STATEMENT", symbol)
        earnings_payload = self._alpha_vantage_get("EARNINGS", symbol)

        if not any([bs_payload, inc_payload, earnings_payload]):
            return pd.DataFrame()

        rows = {}

        bs_reports = (bs_payload or {}).get("quarterlyReports", [])
        for report in bs_reports:
            fiscal_date = report.get("fiscalDateEnding")
            if not fiscal_date:
                continue
            key = pd.Timestamp(fiscal_date)
            short_long_debt = self._to_float(report.get("shortLongTermDebtTotal"))
            long_term_debt = self._to_float(report.get("longTermDebt"))
            derived_debt = None
            if short_long_debt is not None or long_term_debt is not None:
                derived_debt = (short_long_debt or 0.0) + (long_term_debt or 0.0)
            total_debt = self._coalesce_numeric(
                report.get("totalDebt"),
                derived_debt,
            )
            shares = self._to_float(report.get("commonStockSharesOutstanding"))
            total_equity = self._to_float(report.get("totalShareholderEquity"))
            rows[key] = {
                "symbol": symbol,
                "fiscal_year": key.year,
                "fiscal_quarter": int(key.quarter),
                "report_date": key,
                "currency": report.get("reportedCurrency"),
                "total_assets": self._to_float(report.get("totalAssets")),
                "total_equity": total_equity,
                "total_debt": total_debt,
                "book_value_equity": total_equity,
                "shares_outstanding": shares,
                "net_income": None,
                "eps": None,
                "source": "alphavantage",
            }

        inc_reports = (inc_payload or {}).get("quarterlyReports", [])
        for report in inc_reports:
            fiscal_date = report.get("fiscalDateEnding")
            if not fiscal_date:
                continue
            key = pd.Timestamp(fiscal_date)
            if key not in rows:
                rows[key] = {
                    "symbol": symbol,
                    "fiscal_year": key.year,
                    "fiscal_quarter": int(key.quarter),
                    "report_date": key,
                    "currency": report.get("reportedCurrency"),
                    "total_assets": None,
                    "total_equity": None,
                    "total_debt": None,
                    "book_value_equity": None,
                    "shares_outstanding": None,
                    "net_income": None,
                    "eps": None,
                    "source": "alphavantage",
                }
            rows[key]["net_income"] = self._to_float(report.get("netIncome"))
            rows[key]["eps"] = self._coalesce_numeric(
                report.get("reportedEPS"),
                report.get("dilutedEPS"),
            )

        earnings_rows = (earnings_payload or {}).get("quarterlyEarnings", [])
        for report in earnings_rows:
            fiscal_date = report.get("fiscalDateEnding")
            if not fiscal_date:
                continue
            key = pd.Timestamp(fiscal_date)
            if key not in rows:
                rows[key] = {
                    "symbol": symbol,
                    "fiscal_year": key.year,
                    "fiscal_quarter": int(key.quarter),
                    "report_date": key,
                    "currency": None,
                    "total_assets": None,
                    "total_equity": None,
                    "total_debt": None,
                    "book_value_equity": None,
                    "shares_outstanding": None,
                    "net_income": None,
                    "eps": None,
                    "source": "alphavantage",
                }
            if rows[key].get("eps") is None:
                rows[key]["eps"] = self._to_float(report.get("reportedEPS"))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(list(rows.values()))

        # For Alpha Vantage rows, treat book value equity as total equity.
        df["book_value_equity"] = df["book_value_equity"].fillna(df["total_equity"])
        df["report_date"] = pd.to_datetime(df["report_date"])
        return df.sort_values("report_date", ascending=False).reset_index(drop=True)

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
            merged["book_value_equity"] = merged.get("total_equity")
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
            "book_value_equity",
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

    def _alpha_vantage_get(self, function_name, symbol, max_retries=3):
        """Call Alpha Vantage API with basic retry for rate limits."""
        params = {
            "function": function_name,
            "symbol": symbol,
            "apikey": self.alpha_vantage_api_key,
        }

        for attempt in range(max_retries):
            try:
                self._alpha_vantage_throttle_wait()
                response = requests.get(
                    ALPHA_VANTAGE_BASE_URL, params=params, timeout=20
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                logger.warning(
                    "Alpha Vantage request failed for %s (%s): %s",
                    symbol,
                    function_name,
                    exc,
                )
                sleep(2.0)
                continue

            if "Error Message" in payload:
                logger.warning(
                    "Alpha Vantage error for %s (%s): %s",
                    symbol,
                    function_name,
                    payload["Error Message"],
                )
                return None

            if "Note" in payload:
                logger.warning(
                    "Alpha Vantage rate limit hit for %s (%s), retrying...",
                    symbol,
                    function_name,
                )
                sleep(2 * (attempt + 1))
                continue

            return payload

        logger.warning(
            "Alpha Vantage unavailable for %s (%s) after retries",
            symbol,
            function_name,
        )
        return None

    def _alpha_vantage_throttle_wait(self):
        """Ensure minimum spacing between Alpha Vantage API calls."""
        with self._av_rate_limit_lock:
            elapsed = monotonic() - self._av_last_request_ts
            if elapsed < self._av_min_interval_seconds:
                sleep(self._av_min_interval_seconds - elapsed)
            self._av_last_request_ts = monotonic()

    @staticmethod
    def _to_float(value):
        """Parse API numeric values safely."""
        if value in (None, "", "None", "null"):
            return None
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    def _coalesce_numeric(self, *values):
        """Return first successfully parsed numeric value."""
        for value in values:
            parsed = self._to_float(value)
            if parsed is not None:
                return parsed
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
            "total_equity",
            "total_debt",
            "book_value_equity",
            "shares_outstanding",
            "net_income",
            "eps",
            "source",
        ]
        if df is None or df.empty:
            return pd.DataFrame(columns=columns)

        out = df.copy()
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
