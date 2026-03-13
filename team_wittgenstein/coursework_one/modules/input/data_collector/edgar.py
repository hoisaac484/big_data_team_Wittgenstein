"""SEC EDGAR data fetching infrastructure."""

import logging
from time import monotonic, sleep

import pandas as pd
import requests

from .constants import SEC_EDGAR_COMPANY_TICKERS_URL, SEC_EDGAR_USER_AGENT

logger = logging.getLogger(__name__)


class EdgarMixin:
    """SEC EDGAR fetching methods for DataFetcher."""

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
                        "EDGAR request failed after %d retries: %s",
                        max_retries,
                        e,
                    )
                    return None
                sleep(2**attempt)
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
                str(entry.get("ticker", ""))
                .strip()
                .upper(): str(entry.get("cik_str", ""))
                .zfill(10)
                for entry in data.values()
                if entry.get("ticker") and entry.get("cik_str") is not None
            }
            logger.info(
                "Loaded SEC ticker map: %d tickers",
                len(self._ticker_to_cik),
            )

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
            columns=[
                "report_date_str",
                "report_date",
                "fiscal_year",
                "fiscal_quarter",
            ]
        )
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        payload = self._edgar_get_json(url)
        if not payload:
            return empty

        filings = (payload or {}).get("filings", {}).get("recent", {})
        df = pd.DataFrame(
            {
                "form": filings.get("form", []),
                "report_date": filings.get("reportDate", []),
                "filed": filings.get("filingDate", []),
            }
        )
        if df.empty:
            return empty

        df = df[df["form"].isin(["10-Q", "10-K"])].copy()
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        df = (
            df.dropna(subset=["report_date"])
            .sort_values("report_date")
            .reset_index(drop=True)
        )

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
        df["fiscal_quarter"] = pd.to_numeric(
            df["fiscal_quarter"], errors="coerce"
        ).astype("Int64")
        df["report_date_str"] = df["report_date"].dt.strftime("%Y-%m-%d")
        return df[
            ["report_date_str", "report_date", "fiscal_year", "fiscal_quarter"]
        ].reset_index(drop=True)

    def _edgar_fetch_company_facts(self, cik):
        """Fetch all XBRL facts for a company in a single request.

        Uses the SEC EDGAR companyfacts bulk endpoint which returns every
        concept the company has ever filed, avoiding per-concept HTTP calls.

        Args:
            cik: 10-digit CIK string.

        Returns:
            dict or None: The 'facts' -> 'us-gaap' dict keyed by concept tag,
                or None on failure.
        """
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        payload = self._edgar_get_json(url, allow_not_found=True)
        if not payload:
            return None
        return (payload.get("facts") or {}).get("us-gaap")

    @staticmethod
    def _extract_concept(facts, tag, unit="USD", cutoff=None):
        """Extract a single XBRL concept from in-memory company facts.

        Same filtering logic as the old per-concept HTTP call: filter by
        form (10-Q/10-K), dedupe by end date, apply cutoff.

        Args:
            facts: The us-gaap dict from _edgar_fetch_company_facts().
            tag: XBRL concept tag (e.g. 'Assets', 'NetIncomeLoss').
            unit: Unit key (e.g. 'USD', 'shares', 'USD/shares').
            cutoff: Optional earliest end date to include.

        Returns:
            pd.DataFrame with columns: end, start, val, filed.
        """
        empty = pd.DataFrame(columns=["end", "start", "val", "filed"])
        if not facts or tag not in facts:
            return empty

        concept_data = facts[tag]
        unit_rows = (concept_data.get("units") or {}).get(unit, [])
        if not unit_rows:
            return empty

        rows = []
        for row in unit_rows:
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
            rows.append(
                {
                    "end": end,
                    "start": row.get("start"),
                    "val": row.get("val"),
                    "filed": row.get("filed"),
                }
            )
        if not rows:
            return empty

        out = pd.DataFrame(rows)
        out["filed"] = pd.to_datetime(out["filed"], errors="coerce")
        out = out.sort_values("filed", ascending=False).drop_duplicates(
            "end", keep="first"
        )
        return out[["end", "start", "val", "filed"]].reset_index(drop=True)

    def _fetch_edgar_fundamentals(self, symbol, period="5y"):
        """Fetch quarterly fundamentals from SEC EDGAR using bulk companyfacts.

        Makes 3 HTTP requests per symbol (ticker map, submissions,
        companyfacts) instead of ~10-15 per-concept requests. Uses the
        submissions API for fiscal period detection and the companyfacts
        bulk endpoint for all financial fields, with in-memory fallback
        chains. Converts cumulative YTD figures to standalone quarterly
        values.

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
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years_back + 1)

        # Build fiscal period index from submissions
        periods = self._edgar_get_fiscal_periods(cik, cutoff=cutoff)
        if periods.empty:
            return pd.DataFrame()

        # Fetch ALL concepts in one bulk request
        facts = self._edgar_fetch_company_facts(cik)
        if not facts:
            logger.warning("No companyfacts for %s (CIK %s).", symbol, cik)
            return pd.DataFrame()

        result = periods.copy()
        result["symbol"] = symbol

        def merge_field(tag, field, unit="USD"):
            nonlocal result
            concept = self._extract_concept(facts, tag, unit=unit, cutoff=cutoff)
            if concept.empty:
                result[field] = None
                return
            concept = concept.rename(columns={"val": field, "end": "report_date_str"})
            result = result.merge(
                concept[["report_date_str", field]],
                on="report_date_str",
                how="left",
            )

        def fill_nulls(field, tag, unit="USD"):
            """Fill null values in field from a fallback concept tag."""
            nonlocal result
            if field not in result.columns or not result[field].isna().any():
                return
            fb = self._extract_concept(facts, tag, unit=unit, cutoff=cutoff)
            if fb.empty:
                return
            fb = fb.rename(columns={"val": "_fb", "end": "report_date_str"})
            result = result.merge(
                fb[["report_date_str", "_fb"]],
                on="report_date_str",
                how="left",
            )
            mask = result[field].isna() & result["_fb"].notna()
            result.loc[mask, field] = result.loc[mask, "_fb"]
            result = result.drop(columns=["_fb"])

        # --- total_assets ---
        merge_field("Assets", "total_assets")

        # --- net_income (with ProfitLoss fallback) ---
        merge_field("NetIncomeLoss", "net_income")
        fill_nulls("net_income", "ProfitLoss")

        # --- book_equity (with broader equity concept fallback) ---
        merge_field(
            "StockholdersEquityIncludingPortionAttributable" "ToNoncontrollingInterest",
            "book_equity",
        )
        fill_nulls("book_equity", "StockholdersEquity")

        # --- shares_outstanding (two concept fallbacks) ---
        merge_field("CommonStockSharesOutstanding", "shares_outstanding", unit="shares")
        fill_nulls(
            "shares_outstanding",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            unit="shares",
        )

        # --- eps (diluted with basic fallback) ---
        eps_df = self._extract_concept(
            facts, "EarningsPerShareDiluted", unit="USD/shares", cutoff=cutoff
        )
        if eps_df.empty:
            eps_df = self._extract_concept(
                facts, "EarningsPerShareBasic", unit="USD/shares", cutoff=cutoff
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
                on="report_date_str",
                how="left",
            )
        else:
            result["eps"] = None

        # --- total_debt (3-tier fallback) ---
        merge_field("LongTermDebt", "total_debt")
        fill_nulls("total_debt", "LongTermDebtAndCapitalLeaseObligations")

        if "total_debt" in result.columns and result["total_debt"].isna().any():
            lt_nc = self._extract_concept(
                facts, "LongTermDebtNoncurrent", unit="USD", cutoff=cutoff
            )
            lt_c = self._extract_concept(
                facts, "LongTermDebtCurrent", unit="USD", cutoff=cutoff
            )
            if not lt_nc.empty:
                lt_nc = lt_nc.rename(columns={"val": "_nc", "end": "report_date_str"})
                result = result.merge(
                    lt_nc[["report_date_str", "_nc"]],
                    on="report_date_str",
                    how="left",
                )
            else:
                result["_nc"] = None
            if not lt_c.empty:
                lt_c = lt_c.rename(columns={"val": "_c", "end": "report_date_str"})
                result = result.merge(
                    lt_c[["report_date_str", "_c"]],
                    on="report_date_str",
                    how="left",
                )
            else:
                result["_c"] = None

            nc_num = pd.to_numeric(result["_nc"], errors="coerce")
            c_num = pd.to_numeric(result["_c"], errors="coerce")
            both_nan = nc_num.isna() & c_num.isna()
            fill_mask = result["total_debt"].isna() & ~both_nan
            if fill_mask.any():
                result["total_debt"] = pd.to_numeric(
                    result["total_debt"], errors="coerce"
                )
                result.loc[fill_mask, "total_debt"] = nc_num.loc[fill_mask].fillna(
                    0.0
                ).to_numpy(dtype=float) + c_num.loc[fill_mask].fillna(0.0).to_numpy(
                    dtype=float
                )
            result = result.drop(columns=["_nc", "_c"])

        # Clean up: drop helper column, set source/currency
        result = result.drop(columns=["report_date_str"], errors="ignore")
        result["source"] = "edgar"
        result["currency"] = "USD"

        # --- YTD -> standalone quarterly conversion ---
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
        for col in [
            "total_assets",
            "total_debt",
            "book_equity",
            "shares_outstanding",
            "net_income",
            "eps",
        ]:
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
