"""Price data fetching from Yahoo Finance."""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceMixin:
    """Price fetching methods for DataFetcher."""

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

        logger.info("Prices: %d cached, %d to fetch", len(cached_dfs), len(uncached))

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
        logger.info("Downloading prices for %d symbols from yfinance...", len(symbols))

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
                    df = self._reshape_price_df(symbol_data, symbol, currency=currency)
                    if df is not None and not df.empty:
                        df = self._dedupe_dataframe("prices", df, name=symbol)
                        self._cache_dataframe("prices", symbol, df, "yfinance")
                        result_dfs.append(df)
                except KeyError:
                    logger.warning("No price data returned for %s", symbol)
                except Exception as e:
                    logger.error("Error processing prices for %s: %s", symbol, e)

        # Retry symbols that the batch download silently dropped
        fetched_symbols = set()
        for df in result_dfs:
            if "symbol" in df.columns:
                fetched_symbols.update(df["symbol"].unique())

        missed = [s for s in symbols if s not in fetched_symbols]
        if missed:
            logger.info(
                "Batch download missed %d symbols, retrying individually: %s",
                len(missed),
                missed[:20],
            )
            for symbol in missed:
                try:
                    single_raw = yf.download(
                        symbol,
                        period=period,
                        threads=False,
                        progress=False,
                        auto_adjust=False,
                    )
                    if single_raw is not None and not single_raw.empty:
                        currency = self._get_price_currency(symbol)
                        df = self._reshape_price_df(
                            single_raw, symbol, currency=currency
                        )
                        if df is not None and not df.empty:
                            df = self._dedupe_dataframe("prices", df, name=symbol)
                            self._cache_dataframe("prices", symbol, df, "yfinance")
                            result_dfs.append(df)
                            logger.info("Retry succeeded for %s", symbol)
                        else:
                            logger.warning("Retry returned empty data for %s", symbol)
                    else:
                        logger.warning("Retry returned no data for %s", symbol)
                except Exception as e:
                    logger.error("Retry failed for %s: %s", symbol, e)

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
            currency: Trading currency code.

        Returns:
            pd.DataFrame with standardised columns, or None.
        """
        if raw_df is None or raw_df.empty:
            return None

        # Flatten MultiIndex columns (yfinance returns these for
        # single-symbol downloads)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df = raw_df.copy()
            price_keys = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            level0 = set(raw_df.columns.get_level_values(0))
            if price_keys & level0:
                raw_df.columns = raw_df.columns.get_level_values(0)
            else:
                raw_df.columns = raw_df.columns.get_level_values(1)

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
        return df[[c for c in keep if c in df.columns]]
