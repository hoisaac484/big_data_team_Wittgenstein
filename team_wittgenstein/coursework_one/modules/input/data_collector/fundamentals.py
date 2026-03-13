"""Fundamentals fetching orchestration and waterfall merge."""

import logging

import pandas as pd

from .constants import SimFinServerError

logger = logging.getLogger(__name__)


class FundamentalsMixin:
    """Fundamentals orchestration methods for DataFetcher."""

    def fetch_fundamentals(
        self,
        symbols,
        period="5y",
        source="waterfall",
    ):
        """Fetch quarterly financial statements for all symbols.

        Supports source routing:
        - waterfall: EDGAR -> SimFin -> yfinance -> forward fill.
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
                    cached_dfs.append(self._apply_fundamentals_period(df, period))
                    continue
            to_refresh.append(sym)

        logger.info(
            "Fundamentals: %d cache-ready, %d to fetch/refresh",
            len(cached_dfs),
            len(to_refresh),
        )

        if to_refresh:
            fetched, failed = self._sequential_fetch_fundamentals(
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

    def _sequential_fetch_fundamentals(
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
            "Fetching fundamentals for %d symbols sequentially " "(source=%s).",
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
                    cache_source = (
                        ",".join(sorted(df["source"].dropna().astype(str).unique()))
                        or "unknown"
                    )
                    cache_name = self._fundamentals_cache_name(symbol)
                    self._cache_dataframe("fundamentals", cache_name, df, cache_source)
                    result_dfs.append(df)
                else:
                    failed.append(symbol)
            except Exception as e:
                logger.error("Failed fundamentals for %s: %s", symbol, e)
                failed.append(symbol)

        logger.info(
            "Fundamentals: %d success, %d failed",
            len(result_dfs),
            len(failed),
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

    def _fetch_single_fundamental(self, symbol, period="5y", source="waterfall"):
        """Fetch one symbol using the requested source strategy.

        Args:
            symbol: Stock ticker symbol.
            period: Historical window to keep.
            source: Fundamentals data source strategy.
                'waterfall' - EDGAR -> SimFin -> yfinance -> forward fill.
                'simfin'    - SimFin only.

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
    # Waterfall: EDGAR -> SimFin -> yfinance -> forward fill
    # ================================================================

    def _fetch_waterfall_fundamentals(self, symbol, period="5y"):
        """Fetch fundamentals using the waterfall pattern.

        Tries sources in priority order: EDGAR -> SimFin -> yfinance.
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
            logger.info(
                "Waterfall %s: EDGAR returned %d rows",
                symbol,
                len(edgar_df),
            )

        # 2. SimFin
        try:
            simfin_df = self._fetch_simfin_fundamentals(symbol)
            if simfin_df is not None and not simfin_df.empty:
                sources.append(("simfin", simfin_df))
                logger.info(
                    "Waterfall %s: SimFin returned %d rows",
                    symbol,
                    len(simfin_df),
                )
        except SimFinServerError:
            logger.warning("Waterfall %s: SimFin HTTP 500, skipping.", symbol)

        # 3. yfinance (lowest priority fallback)
        yf_df = self._fetch_yfinance_fundamentals(symbol)
        if yf_df is not None and not yf_df.empty:
            sources.append(("yfinance", yf_df))
            logger.info(
                "Waterfall %s: yfinance returned %d rows",
                symbol,
                len(yf_df),
            )

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
            "total_assets",
            "total_debt",
            "net_income",
            "book_equity",
            "shares_outstanding",
            "eps",
            "report_date",
            "currency",
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
            "total_assets",
            "total_debt",
            "net_income",
            "book_equity",
            "shares_outstanding",
            "eps",
        ]
        df = df.sort_values(["symbol", "fiscal_year", "fiscal_quarter"])

        existing = [c for c in fill_cols if c in df.columns]
        if existing:
            # Track which rows had ALL numeric fields null before ffill
            all_null_before = df[existing].isna().all(axis=1)
            df[existing] = df.groupby("symbol")[existing].ffill(limit=2)
            # Mark rows that were entirely filled by forward fill
            if "source" in df.columns:
                df.loc[
                    all_null_before & df[existing].notna().any(axis=1),
                    "source",
                ] = "ffill"

        return df
