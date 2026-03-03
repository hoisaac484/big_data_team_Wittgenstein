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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

BUCKET = "wittgenstein-cache"

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

    def _cache_dataframe(self, data_type, name, df, source):
        """Save a DataFrame as parquet with a companion CTL file.

        Args:
            data_type: Category of data.
            name: Identifier.
            df: DataFrame to cache.
            source: Data source name.
        """
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
                self._cache_dataframe("prices", symbol, df, "yfinance")
                result_dfs.append(df)
        else:
            for symbol in symbols:
                try:
                    symbol_data = raw[symbol].dropna(how="all")
                    df = self._reshape_price_df(symbol_data, symbol)
                    if df is not None and not df.empty:
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

    # Fundamentals (source: Yahoo Finance)

    def fetch_fundamentals(self, symbols, max_workers=10):
        """Fetch quarterly financial statements for all symbols.

        Uses ThreadPoolExecutor for parallel fetching. Caches per-symbol
        parquet files in MinIO.

        Args:
            symbols: List of stock ticker symbols.
            max_workers: Number of parallel download threads.

        Returns:
            pd.DataFrame: Combined financial data with columns: symbol,
                fiscal_date, total_assets, total_equity, total_debt,
                net_income, eps, book_value, shares_outstanding.
        """
        uncached = []
        cached_dfs = []

        for symbol in symbols:
            sym = symbol.strip()
            if self._is_cached("fundamentals", sym):
                df = self._load_cached("fundamentals", sym)
                if df is not None:
                    cached_dfs.append(df)
                    continue
            uncached.append(sym)

        logger.info(
            "Fundamentals: %d cached, %d to fetch",
            len(cached_dfs), len(uncached),
        )

        if uncached:
            fetched = self._parallel_fetch_fundamentals(uncached, max_workers)
            cached_dfs.extend(fetched)

        if not cached_dfs:
            return pd.DataFrame()

        return pd.concat(cached_dfs, ignore_index=True)

    def _parallel_fetch_fundamentals(self, symbols, max_workers):
        """Fetch fundamentals in parallel using ThreadPoolExecutor.

        Args:
            symbols: Symbols to fetch.
            max_workers: Thread count.

        Returns:
            list[pd.DataFrame]
        """
        logger.info(
            "Fetching fundamentals for %d symbols (%d workers)...",
            len(symbols), max_workers,
        )

        result_dfs = []
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_single_fundamental, sym): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        self._cache_dataframe(
                            "fundamentals", symbol, df, "yfinance"
                        )
                        result_dfs.append(df)
                    else:
                        failed.append(symbol)
                except Exception as e:
                    logger.error(
                        "Failed fundamentals for %s: %s", symbol, e
                    )
                    failed.append(symbol)

        logger.info(
            "Fundamentals: %d success, %d failed",
            len(result_dfs), len(failed),
        )
        if failed:
            logger.warning("Failed symbols (first 20): %s", failed[:20])

        return result_dfs

    def _fetch_single_fundamental(self, symbol):
        """Fetch financial statements for one symbol from yfinance.

        Extracts balance sheet, income statement, and company info
        to build a single DataFrame of quarterly financial records.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            pd.DataFrame or None.
        """
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
            return None

        df = pd.DataFrame(records)
        df["report_date"] = pd.to_datetime(df["report_date"])
        return df

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
            return self._load_cached("risk_free_rates", "all")

        unique_countries = list(set(c.strip() for c in countries))

        # Try OECD API first
        result = self._fetch_rates_oecd(unique_countries)

        # Fallback to yfinance Treasury yields
        if result is None or result.empty:
            logger.info("OECD API unavailable, using yfinance fallback.")
            result = self._fetch_rates_yfinance(unique_countries)

        if result is not None and not result.empty:
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
