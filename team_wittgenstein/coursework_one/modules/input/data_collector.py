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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.minio._ensure_bucket(self.bucket)

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

    def mark_loaded(self, data_type, name):
        """Update a CTL file to mark data as loaded into PostgreSQL.

        Args:
            data_type: Category of data.
            name: Identifier.
        """
        ctl = self._read_ctl(data_type, name)
        if ctl:
            ctl["loaded_to_postgres"] = True
            ctl["loaded_at"] = datetime.utcnow().isoformat()
            self.minio.upload_json(
                self.bucket, self._ctl_path(data_type, name), ctl
            )

   
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

        return pd.concat(cached_dfs, ignore_index=True)

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

    # Fundamentals (source: Alpha Vantage only)

    def fetch_fundamentals(self, symbols, max_workers=10, target_quarters=20):
        """Fetch quarterly financial statements for all symbols.

        Uses Alpha Vantage only. For each symbol, calls BALANCE_SHEET,
        INCOME_STATEMENT, and EARNINGS in sequence.
        Per-symbol parquet files are cached in MinIO.

        Args:
            symbols: List of stock ticker symbols.
            max_workers: Kept for compatibility; fundamentals are fetched
                sequentially to respect API throttling.
            target_quarters: Number of recent quarters to keep per symbol.

        Returns:
            pd.DataFrame: Combined financial data with columns: symbol,
                fiscal_year, fiscal_quarter, report_date, total_assets,
                total_equity, total_debt, net_income, eps, book_value_equity,
                shares_outstanding.
        """
        to_refresh = []
        cached_dfs = []

        for symbol in symbols:
            sym = symbol.strip()
            if self._is_cached("fundamentals", sym):
                df = self._load_cached("fundamentals", sym)
                if df is not None:
                    df = self._ensure_fundamentals_schema(df)
                    df = self._dedupe_dataframe("fundamentals", df, name=sym)
                    source_series = df.get("source", pd.Series(dtype=object))
                    source_series = source_series.dropna().astype(str).str.lower()
                    has_non_alpha_source = (
                        source_series.empty
                        or source_series.ne("alphavantage").any()
                    )
                    needs_refresh = (
                        len(df) < target_quarters
                        or (
                            bool(self.alpha_vantage_api_key)
                            and self._has_fundamental_gaps(df)
                        )
                        or has_non_alpha_source
                    )
                    if needs_refresh:
                        to_refresh.append(sym)
                    else:
                        cached_dfs.append(
                            df.sort_values("report_date", ascending=False).head(
                                target_quarters
                            )
                        )
                    continue
            to_refresh.append(sym)

        logger.info(
            "Fundamentals: %d cache-ready, %d to fetch/refresh",
            len(cached_dfs), len(to_refresh),
        )

        if to_refresh:
            fetched = self._parallel_fetch_fundamentals(
                to_refresh, max_workers, target_quarters
            )
            cached_dfs.extend(fetched)

        if not cached_dfs:
            return pd.DataFrame()

        out = pd.concat(cached_dfs, ignore_index=True)
        out = self._ensure_fundamentals_schema(out)
        out = self._dedupe_dataframe("fundamentals", out)
        return out

    def _parallel_fetch_fundamentals(self, symbols, max_workers, target_quarters):
        """Fetch fundamentals sequentially to respect API throttling.

        Args:
            symbols: Symbols to fetch.
            max_workers: Unused for fundamentals; kept for compatibility.
            target_quarters: Number of recent quarters to retain.

        Returns:
            list[pd.DataFrame]
        """
        logger.info(
            "Fetching fundamentals for %d symbols sequentially "
            "(alpha vantage throttle-safe).",
            len(symbols),
        )

        result_dfs = []
        failed = []

        for symbol in symbols:
            try:
                df = self._fetch_single_fundamental(symbol, target_quarters)
                if df is not None and not df.empty:
                    df = self._ensure_fundamentals_schema(df)
                    df = self._dedupe_dataframe("fundamentals", df, name=symbol)
                    cache_source = ",".join(
                        sorted(df["source"].dropna().astype(str).unique())
                    ) or "unknown"
                    self._cache_dataframe(
                        "fundamentals", symbol, df, cache_source
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

        return result_dfs

    def _fetch_single_fundamental(self, symbol, target_quarters=20):
        """Fetch one symbol from Alpha Vantage only.

        Args:
            symbol: Stock ticker symbol.
            target_quarters: Number of recent quarters to return.

        Returns:
            pd.DataFrame or None.
        """
        av_df = self._fetch_alpha_vantage_fundamentals(symbol)

        if av_df is None or av_df.empty:
            return None

        df = self._ensure_fundamentals_schema(av_df)
        df = df.sort_values("report_date", ascending=False).head(target_quarters)
        return df.reset_index(drop=True)

    def _fetch_yfinance_fundamentals(self, symbol):
        """Fetch quarterly financial statements for one symbol from yfinance."""
        ticker = yf.Ticker(symbol)

        bs = ticker.quarterly_balance_sheet
        inc = ticker.quarterly_financials
        info = ticker.info or {}

        if bs is None or bs.empty:
            logger.warning("No balance sheet for %s", symbol)
            return None

        records = []
        for date_col in bs.columns:
            ts = pd.Timestamp(date_col)
            quarter = (ts.month - 1) // 3 + 1
            record = {
                "symbol": symbol,
                "fiscal_year": ts.year,
                "fiscal_quarter": quarter,
                "report_date": ts.date(),
                "currency": info.get("currency"),
                "total_assets": self._safe_get(bs, "Total Assets", date_col),
                "total_equity": (
                    self._safe_get(bs, "Stockholders Equity", date_col)
                    or self._safe_get(
                        bs, "Total Stockholder Equity", date_col
                    )
                ),
                "total_debt": (
                    self._safe_get(bs, "Total Debt", date_col)
                    or self._safe_get(bs, "Long Term Debt", date_col)
                ),
                "book_value_equity": info.get("bookValue"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "source": "yfinance",
            }

            if inc is not None and date_col in inc.columns:
                record["net_income"] = self._safe_get(
                    inc, "Net Income", date_col
                )
                record["eps"] = (
                    self._safe_get(inc, "Basic EPS", date_col)
                    or self._safe_get(inc, "Diluted EPS", date_col)
                )

            records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.sort_values("report_date", ascending=False).reset_index(drop=True)
        return df

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
            rows[key] = {
                "symbol": symbol,
                "fiscal_year": key.year,
                "fiscal_quarter": int(key.quarter),
                "report_date": key,
                "currency": report.get("reportedCurrency"),
                "total_assets": self._to_float(report.get("totalAssets")),
                "total_equity": self._to_float(
                    report.get("totalShareholderEquity")
                ),
                "total_debt": total_debt,
                "book_value_equity": None,
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

        # If book value/share is missing, compute as equity / shares.
        can_compute = (
            df["book_value_equity"].isna()
            & df["total_equity"].notna()
            & df["shares_outstanding"].notna()
            & (df["shares_outstanding"] > 0)
        )
        df.loc[can_compute, "book_value_equity"] = (
            df.loc[can_compute, "total_equity"]
            / df.loc[can_compute, "shares_outstanding"]
        )
        df["report_date"] = pd.to_datetime(df["report_date"])
        return df.sort_values("report_date", ascending=False).reset_index(drop=True)

    def _merge_fundamentals_sources(
        self, symbol, yf_df, av_df, target_quarters=20
    ):
        """Blend yfinance (primary) with Alpha Vantage backfill/top-up."""
        yf_df = self._ensure_fundamentals_schema(yf_df)
        av_df = self._ensure_fundamentals_schema(av_df)

        if yf_df.empty and av_df.empty:
            return pd.DataFrame()

        if yf_df.empty:
            merged = av_df.copy()
            merged = merged.sort_values("report_date", ascending=False)
            return merged.head(target_quarters).reset_index(drop=True)

        if av_df.empty:
            merged = yf_df.copy()
            merged = merged.sort_values("report_date", ascending=False)
            return merged.head(target_quarters).reset_index(drop=True)

        key_cols = ["fiscal_year", "fiscal_quarter"]
        fill_cols = [
            "report_date",
            "currency",
            "total_assets",
            "total_equity",
            "total_debt",
            "book_value_equity",
            "shares_outstanding",
            "net_income",
            "eps",
        ]

        yf = yf_df.copy().sort_values("report_date", ascending=False)
        av = av_df.copy().sort_values("report_date", ascending=False)

        yf = yf.drop_duplicates(subset=key_cols, keep="first").set_index(key_cols)
        av = av.drop_duplicates(subset=key_cols, keep="first").set_index(key_cols)
        av_aligned = av.reindex(yf.index)

        na_before = yf[fill_cols].isna()

        for col in fill_cols:
            yf[col] = yf[col].where(yf[col].notna(), av_aligned[col])

        filled_mask = (na_before & yf[fill_cols].notna()).any(axis=1)
        yf.loc[filled_mask, "source"] = "yfinance+alphavantage"
        yf.loc[~filled_mask, "source"] = yf.loc[~filled_mask, "source"].fillna(
            "yfinance"
        )
        yf["symbol"] = symbol

        av_only = av.loc[~av.index.isin(yf.index)].copy()
        if not av_only.empty:
            av_only["source"] = "alphavantage"
            av_only["symbol"] = symbol

        combined = pd.concat([yf, av_only], axis=0).reset_index()
        combined = self._ensure_fundamentals_schema(combined)
        combined = combined.sort_values("report_date", ascending=False)
        combined = combined.drop_duplicates(subset=["symbol"] + key_cols, keep="first")
        return combined.head(target_quarters).reset_index(drop=True)

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
                sleep(1.5 * (attempt + 1))
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
                sleep(12 * (attempt + 1))
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
    def _has_fundamental_gaps(df):
        """Check whether critical financial fields contain nulls."""
        critical_cols = [
            "total_assets",
            "total_equity",
            "total_debt",
            "net_income",
            "eps",
        ]
        available = [c for c in critical_cols if c in df.columns]
        if not available:
            return True
        return df[available].isna().any().any()

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

    @staticmethod
    def _safe_get(df, row_label, col_label):
        """Safely extract a value from a DataFrame cell.

        Args:
            df: Source DataFrame.
            row_label: Row index label.
            col_label: Column label.

        Returns:
            float or None.
        """
        try:
            val = df.loc[row_label, col_label]
            if pd.isna(val):
                return None
            return float(val)
        except (KeyError, TypeError, ValueError):
            return None

    
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
