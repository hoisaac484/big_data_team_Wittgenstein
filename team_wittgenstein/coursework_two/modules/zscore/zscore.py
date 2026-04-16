"""
Stage 3, Step 1: Raw factor ratio calculations.

Computes all nine sub-metrics for each symbol on a given rebalancing date:
  Value     : pb_ratio, asset_growth
  Quality   : roe, leverage, earnings_stability
  Momentum  : momentum_6m, momentum_12m
  Low Vol   : volatility_3m, volatility_12m

When a metric cannot be computed (missing data, invalid inputs), the value is
set to None and a WARNING is emitted. None values are excluded from sector
mean/std in the winsorisation and Z-score stages — they are never imputed.
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BMonthEnd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FILING_LAG_DAYS = 45  # conservative look-ahead buffer for quarterly filings
MIN_VOL_OBS_3M = 50  # min trading days for 3-month vol (window = 63)
MIN_VOL_OBS_12M = 200  # min trading days for 12-month vol (window = 252)
MIN_EARN_HISTORY = 5  # min valid YoY growth observations for earnings_stability
EPS_FLOOR = 0.01  # minimum |EPS| used as denominator in YoY growth
YOY_GROWTH_CAP = 5.0  # cap individual YoY EPS growth at ±500%
STALE_PRICE_DAYS = 5  # warn if latest price is more than N days before rebalancing


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fallback(symbol: str, metric: str, reason: str) -> None:
    """Log at DEBUG and return None to exclude from Z-score calculation."""
    logger.debug("%-6s | %-22s | %s — excluded (None)", symbol, metric, reason)
    return None


def _last_bday_of_month(ref: pd.Timestamp, n_months_back: int) -> pd.Timestamp:
    """Return the last business day of the month n_months_back before ref."""
    return (ref - BMonthEnd(n_months_back)).normalize()


def _nearest_price(prices: pd.Series, target: pd.Timestamp):
    """Return the most recent price on or before target, or None."""
    available = prices[prices.index <= target]
    return float(available.iloc[-1]) if not available.empty else None


# ── Data fetchers ─────────────────────────────────────────────────────────────


def _fetch_price_data(
    pg, symbols: list, start_date: date, end_date: date
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for all symbols between start_date and end_date.
    Returns DataFrame: symbol, trade_date (datetime), adjusted_close.
    """
    query = """
        SELECT symbol, trade_date, adjusted_close
        FROM team_wittgenstein.price_data
        WHERE symbol = ANY(:symbols)
          AND trade_date BETWEEN :start_date AND :end_date
        ORDER BY symbol, trade_date
    """
    df = pg.read_query(
        query,
        params={"symbols": symbols, "start_date": start_date, "end_date": end_date},
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _fetch_latest_financials(pg, symbols: list, rebalance_date: date) -> pd.DataFrame:
    """
    Fetch the most recent quarterly filing per symbol available as of
    rebalance_date minus FILING_LAG_DAYS.

    Returns one row per symbol with: symbol, report_date, total_assets, total_debt,
    net_income, book_equity, shares_outstanding, eps, fiscal_year, fiscal_quarter.
    """
    cutoff = rebalance_date - timedelta(days=FILING_LAG_DAYS)
    query = """
        SELECT DISTINCT ON (symbol)
            symbol, report_date, total_assets, total_debt, net_income,
            book_equity, shares_outstanding, eps, fiscal_year, fiscal_quarter
        FROM team_wittgenstein.financial_data
        WHERE symbol = ANY(:symbols)
          AND report_date <= :cutoff
        ORDER BY symbol, report_date DESC
    """
    df = pg.read_query(query, params={"symbols": symbols, "cutoff": cutoff})
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def _fetch_prior_year_financials(
    pg, symbols: list, rebalance_date: date
) -> pd.DataFrame:
    """
    Fetch the same fiscal quarter from one year prior per symbol, used as the
    q-4 denominator in asset_growth.

    Matches on (fiscal_year - 1, fiscal_quarter) so Q2 2024 is always compared
    against Q2 2023, regardless of non-standard fiscal year calendars.

    Returns one row per symbol: symbol, report_date, total_assets.
    """
    cutoff = rebalance_date - timedelta(days=FILING_LAG_DAYS)
    query = """
        SELECT DISTINCT ON (curr.symbol)
            curr.symbol,
            prior.report_date,
            prior.total_assets
        FROM (
            SELECT DISTINCT ON (symbol)
                symbol, fiscal_year, fiscal_quarter
            FROM team_wittgenstein.financial_data
            WHERE symbol = ANY(:symbols)
              AND report_date <= :cutoff
            ORDER BY symbol, report_date DESC
        ) curr
        JOIN team_wittgenstein.financial_data prior
          ON prior.symbol         = curr.symbol
         AND prior.fiscal_year    = curr.fiscal_year - 1
         AND prior.fiscal_quarter = curr.fiscal_quarter
         AND prior.report_date   <= :cutoff
        ORDER BY curr.symbol
    """
    df = pg.read_query(query, params={"symbols": symbols, "cutoff": cutoff})
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def _fetch_ttm_financials(pg, symbols: list, rebalance_date: date) -> pd.DataFrame:
    """
    Fetch the last six quarters of net_income and book_equity per symbol.
    The caller takes only the four most recent rows — six quarters of buffer
    handles late-filing companies without excluding genuinely available data.

    Returns all qualifying rows sorted by (symbol, report_date DESC).
    """
    cutoff = rebalance_date - timedelta(days=FILING_LAG_DAYS)
    year_start = cutoff - timedelta(days=365 + 180)  # 6 quarters of buffer
    query = """
        SELECT symbol, report_date, net_income, book_equity
        FROM team_wittgenstein.financial_data
        WHERE symbol = ANY(:symbols)
          AND report_date BETWEEN :year_start AND :cutoff
        ORDER BY symbol, report_date DESC
    """
    df = pg.read_query(
        query,
        params={"symbols": symbols, "cutoff": cutoff, "year_start": year_start},
    )
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def _fetch_eps_history(pg, symbols: list, rebalance_date: date) -> pd.DataFrame:
    """
    Fetch the last ~20 quarters of EPS per symbol for earnings_stability.
    Returns all qualifying rows sorted by (symbol, report_date ASC).
    """
    cutoff = rebalance_date - timedelta(days=FILING_LAG_DAYS)
    five_yrs_ago = cutoff - timedelta(days=365 * 5 + 90)
    query = """
        SELECT symbol, report_date, eps, fiscal_year, fiscal_quarter
        FROM team_wittgenstein.financial_data
        WHERE symbol = ANY(:symbols)
          AND report_date BETWEEN :five_yrs_ago AND :cutoff
        ORDER BY symbol, report_date ASC
    """
    df = pg.read_query(
        query,
        params={"symbols": symbols, "cutoff": cutoff, "five_yrs_ago": five_yrs_ago},
    )
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def _fetch_risk_free_rate(pg, country: str, rebalance_date: date) -> float:
    """
    Fetch the most recent annualised risk-free rate for country on or before
    rebalance_date. Returns 0.0 and warns if not found.
    """
    query = """
        SELECT rate
        FROM team_wittgenstein.risk_free_rates
        WHERE country = :country
          AND rate_date <= :rate_date
        ORDER BY rate_date DESC
        LIMIT 1
    """
    df = pg.read_query(query, params={"country": country, "rate_date": rebalance_date})
    if df.empty:
        logger.warning(
            "No risk-free rate found for country=%s as of %s — using 0",
            country,
            rebalance_date,
        )
        return 0.0
    return float(df["rate"].iloc[0])


# ── Per-metric calculations ───────────────────────────────────────────────────


def _calc_pb_ratio(
    symbol: str, price: float, book_equity, shares_outstanding
) -> float | None:
    """P/B = adjusted_close / (book_equity / shares_outstanding)."""
    if price is None or pd.isna(price):
        return _fallback(symbol, "pb_ratio", "price is null")
    if book_equity is None or pd.isna(book_equity):
        return _fallback(symbol, "pb_ratio", "book_equity is null")
    if shares_outstanding is None or pd.isna(shares_outstanding):
        return _fallback(symbol, "pb_ratio", "shares_outstanding is null")
    if float(book_equity) == 0:
        return _fallback(symbol, "pb_ratio", "zero book equity (division by zero)")
    if float(shares_outstanding) <= 0:
        return _fallback(
            symbol,
            "pb_ratio",
            f"non-positive shares_outstanding ({float(shares_outstanding):.0f})",
        )
    bvps = float(book_equity) / float(shares_outstanding)
    return float(price) / bvps


def _calc_asset_growth(symbol: str, assets_now, assets_year_ago) -> float | None:
    """Asset growth = (total_assets_q - total_assets_q4) / |total_assets_q4|."""
    if assets_now is None or pd.isna(assets_now):
        return _fallback(symbol, "asset_growth", "current total_assets is null")
    if assets_year_ago is None or pd.isna(assets_year_ago):
        return _fallback(symbol, "asset_growth", "prior-year total_assets is null")
    if float(assets_year_ago) == 0:
        return _fallback(symbol, "asset_growth", "prior-year total_assets is zero")
    return (float(assets_now) - float(assets_year_ago)) / abs(float(assets_year_ago))


def _calc_roe(symbol: str, ttm_rows: pd.DataFrame) -> float | None:
    """
    ROE = most recent quarter net_income / mean(most recent 2 quarters book_equity).
    """
    if ttm_rows.empty:
        return _fallback(symbol, "roe", "no quarterly financial data available")

    latest = ttm_rows.head(1).iloc[0]

    net_income = latest["net_income"]
    if net_income is None or pd.isna(net_income):
        return _fallback(symbol, "roe", "most recent quarter net_income is null")

    recent_equity = ttm_rows.head(2)["book_equity"].dropna()
    if recent_equity.empty:
        return _fallback(symbol, "roe", "book_equity null in 2 most recent quarters")

    avg_equity = float(recent_equity.mean())
    if avg_equity <= 0:
        return _fallback(
            symbol, "roe", f"non-positive average equity ({avg_equity:.2f})"
        )
    return float(net_income) / avg_equity


def _calc_leverage(symbol: str, total_debt, book_equity) -> float | None:
    """Leverage = total_debt / book_equity."""
    if total_debt is None or pd.isna(total_debt):
        return _fallback(symbol, "leverage", "total_debt is null")
    if book_equity is None or pd.isna(book_equity):
        return _fallback(symbol, "leverage", "book_equity is null")
    if float(book_equity) <= 0:
        return _fallback(
            symbol, "leverage", f"non-positive book equity ({float(book_equity):.2f})"
        )
    return float(total_debt) / float(book_equity)


def _calc_earnings_stability(symbol: str, eps_rows: pd.DataFrame) -> float | None:
    """
    earnings_stability = std(YoY EPS growth) over last 5 fiscal years.
    eps_rows must be sorted by report_date ASC with columns eps, fiscal_year,
    fiscal_quarter.
    """
    if eps_rows.empty:
        return _fallback(symbol, "earnings_stability", "no EPS history available")

    # Build lookup keyed by (fiscal_year, fiscal_quarter) for q-4 comparison
    eps_map: dict[tuple, float] = {
        (int(r.fiscal_year), int(r.fiscal_quarter)): r.eps
        for r in eps_rows.itertuples()
        if not pd.isna(r.eps)
    }

    yoy_growths = []
    for r in eps_rows.itertuples():
        if pd.isna(r.eps):
            continue
        prior_eps = eps_map.get((int(r.fiscal_year) - 1, int(r.fiscal_quarter)))
        if prior_eps is None or abs(float(prior_eps)) < EPS_FLOOR:
            continue  # skip near-zero base to avoid division explosion
        growth = (float(r.eps) - float(prior_eps)) / abs(float(prior_eps))
        # Cap individual YoY growth at ±500% before computing std
        growth = max(-YOY_GROWTH_CAP, min(YOY_GROWTH_CAP, growth))
        yoy_growths.append(growth)

    if len(yoy_growths) < MIN_EARN_HISTORY:
        return _fallback(
            symbol,
            "earnings_stability",
            f"only {len(yoy_growths)} valid YoY observations (need {MIN_EARN_HISTORY})",
        )

    return float(np.std(yoy_growths, ddof=1))


def _calc_momentum(
    symbol: str,
    prices: pd.Series,
    rebalance_ts: pd.Timestamp,
    annual_rf: float,
) -> tuple[float | None, float | None]:
    """
    Skip-1-month momentum:
      mom_6m  = (P[T-1] / P[T-7]  - 1) - rf_1m
      mom_12m = (P[T-1] / P[T-13] - 1) - rf_1m

    prices: Series indexed by trade_date (datetime), adjusted close, sorted ASC.
    annual_rf: annualised rate as decimal (e.g. 0.036 for 3.6%) — converted to the
               1-month equivalent per the methodology.
    Returns: (momentum_6m, momentum_12m)
    """
    monthly_rate = (1 + annual_rf) ** (1 / 12) - 1  # annualised → 1-month RF

    t1 = _last_bday_of_month(rebalance_ts, 1)
    t7 = _last_bday_of_month(rebalance_ts, 7)
    t13 = _last_bday_of_month(rebalance_ts, 13)

    p_t1 = _nearest_price(prices, t1)
    p_t7 = _nearest_price(prices, t7)
    p_t13 = _nearest_price(prices, t13)

    # 6-month momentum
    if p_t1 is None or p_t7 is None:
        mom_6m = _fallback(
            symbol, "momentum_6m", "insufficient price history (need 7 months)"
        )
    elif p_t7 <= 0:
        mom_6m = _fallback(symbol, "momentum_6m", "price at T-7 is non-positive")
    else:
        mom_6m = (p_t1 / p_t7 - 1) - monthly_rate

    # 12-month momentum
    if p_t1 is None or p_t13 is None:
        mom_12m = _fallback(
            symbol, "momentum_12m", "insufficient price history (need 13 months)"
        )
    elif p_t13 <= 0:
        mom_12m = _fallback(symbol, "momentum_12m", "price at T-13 is non-positive")
    else:
        mom_12m = (p_t1 / p_t13 - 1) - monthly_rate

    return mom_6m, mom_12m


def _calc_volatility(
    symbol: str, prices: pd.Series
) -> tuple[float | None, float | None]:
    """
    Annualised volatility from daily log returns:
      vol_3m  = std(log_returns[-63:])  * sqrt(252)
      vol_12m = std(log_returns[-252:]) * sqrt(252)

    prices: Series indexed by trade_date (datetime), adjusted close, sorted ASC.
    Returns: (volatility_3m, volatility_12m)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # 3-month volatility (63 trading days)
    obs_3m = log_returns.iloc[-63:]
    if len(obs_3m) < MIN_VOL_OBS_3M:
        vol_3m = _fallback(
            symbol,
            "volatility_3m",
            f"only {len(obs_3m)} observations (need {MIN_VOL_OBS_3M})",
        )
    else:
        if (obs_3m == 0).mean() > 0.10:
            logger.warning(
                "%-6s | volatility_3m  | >10%% zero returns — possible stale prices",
                symbol,
            )
        vol_3m = float(obs_3m.std(ddof=1) * np.sqrt(252))

    # 12-month volatility (252 trading days)
    obs_12m = log_returns.iloc[-252:]
    if len(obs_12m) < MIN_VOL_OBS_12M:
        vol_12m = _fallback(
            symbol,
            "volatility_12m",
            f"only {len(obs_12m)} observations (need {MIN_VOL_OBS_12M})",
        )
    else:
        if (obs_12m == 0).mean() > 0.10:
            logger.warning(
                "%-6s | volatility_12m | >10%% zero returns — possible stale prices",
                symbol,
            )
        vol_12m = float(obs_12m.std(ddof=1) * np.sqrt(252))

    return vol_3m, vol_12m


# ── Orchestrator ──────────────────────────────────────────────────────────────


def calculate_ratios(
    pg,
    rebalance_date: date,
    symbols: list,
    country: str = "United States",
) -> pd.DataFrame:
    """
    Calculate all nine factor sub-metrics for every symbol on the given
    rebalancing date.

    Data is fetched in bulk (one query per table) and then distributed
    per-symbol, so the number of round-trips to the database is fixed
    regardless of how many symbols are processed.

    Args:
        pg:             PostgresConnection instance.
        rebalance_date: The monthly rebalancing date (last trading day of month).
        symbols:        List of ticker symbols to process.
        country:        Country for risk-free rate lookup (default: "United States").

    Returns:
        pd.DataFrame with one row per symbol and columns:
            symbol, calc_date,
            pb_ratio, asset_growth,
            roe, leverage, earnings_stability,
            momentum_6m, momentum_12m,
            volatility_3m, volatility_12m

        Missing values are stored as None (NULL in the DB). The Z-score stage
        must call dropna() per sector before computing mean and std.
    """
    rebalance_ts = pd.Timestamp(rebalance_date)

    logger.debug(
        "Starting ratio calculations for %d symbols as of %s (rf country: %s)",
        len(symbols),
        rebalance_date,
        country,
    )

    # ── Bulk data fetches ─────────────────────────────────────────────────────
    # Prices: 13 months for momentum T-13 + 252 trading days ≈ 16 calendar months
    # (16 instead of 15 gives a full extra month of buffer for thin-history months)
    price_start = (rebalance_ts - BMonthEnd(16)).date()
    price_df = _fetch_price_data(pg, symbols, price_start, rebalance_date)

    latest_fin = _fetch_latest_financials(pg, symbols, rebalance_date)
    prior_fin = _fetch_prior_year_financials(pg, symbols, rebalance_date)
    ttm_fin = _fetch_ttm_financials(pg, symbols, rebalance_date)
    eps_hist = _fetch_eps_history(pg, symbols, rebalance_date)
    annual_rf = _fetch_risk_free_rate(pg, country, rebalance_date)

    # Convert to dicts keyed by symbol for O(1) per-symbol lookup.
    # Using to_dict("index") raises a clean KeyError on schema changes
    # rather than silently returning None like Series.get() would.
    latest_map: dict[str, dict] = (
        latest_fin.set_index("symbol").to_dict("index") if not latest_fin.empty else {}
    )
    prior_map: dict[str, dict] = (
        prior_fin.set_index("symbol").to_dict("index") if not prior_fin.empty else {}
    )

    records = []

    for symbol in symbols:
        row: dict = {"symbol": symbol, "calc_date": rebalance_date}

        # ── Price series ──────────────────────────────────────────────────
        sym_prices = (
            price_df[price_df["symbol"] == symbol]
            .set_index("trade_date")["adjusted_close"]
            .sort_index()
            .dropna()
        )

        # ── Latest quarterly fundamentals ─────────────────────────────────
        fin = latest_map.get(symbol)  # dict or None
        prior = prior_map.get(symbol)  # dict or None

        # — P/B ratio -------------------------------------------------------
        if sym_prices.empty:
            row["pb_ratio"] = _fallback(symbol, "pb_ratio", "no price data")
        elif fin is None:
            row["pb_ratio"] = _fallback(
                symbol, "pb_ratio", "no quarterly financial data"
            )
        else:
            last_price_date = sym_prices.index[-1]
            days_stale = (rebalance_ts - last_price_date).days
            if days_stale > STALE_PRICE_DAYS:
                logger.warning(
                    "%-6s | %-22s | last price is %d days before rebalancing date",
                    symbol,
                    "pb_ratio",
                    days_stale,
                )
            row["pb_ratio"] = _calc_pb_ratio(
                symbol,
                float(sym_prices.iloc[-1]),
                fin["book_equity"],
                fin["shares_outstanding"],
            )

        # — Asset growth ----------------------------------------------------
        if fin is None:
            row["asset_growth"] = _fallback(
                symbol, "asset_growth", "no current quarterly data"
            )
        elif prior is None:
            row["asset_growth"] = _fallback(
                symbol, "asset_growth", "no prior-year same-quarter data"
            )
        else:
            row["asset_growth"] = _calc_asset_growth(
                symbol, fin["total_assets"], prior["total_assets"]
            )

        # — ROE (TTM) -------------------------------------------------------
        sym_ttm = ttm_fin[ttm_fin["symbol"] == symbol].sort_values(
            "report_date", ascending=False
        )
        row["roe"] = _calc_roe(symbol, sym_ttm)

        # — Leverage --------------------------------------------------------
        if fin is None:
            row["leverage"] = _fallback(
                symbol, "leverage", "no quarterly financial data"
            )
        else:
            row["leverage"] = _calc_leverage(
                symbol, fin["total_debt"], fin["book_equity"]
            )

        # — Earnings stability ----------------------------------------------
        sym_eps = eps_hist[eps_hist["symbol"] == symbol].sort_values("report_date")
        row["earnings_stability"] = _calc_earnings_stability(symbol, sym_eps)

        # — Momentum (6m and 12m) ------------------------------------------
        row["momentum_6m"], row["momentum_12m"] = _calc_momentum(
            symbol, sym_prices, rebalance_ts, annual_rf=annual_rf
        )

        # — Volatility (3m and 12m) ----------------------------------------
        row["volatility_3m"], row["volatility_12m"] = _calc_volatility(
            symbol, sym_prices
        )

        records.append(row)

    result = pd.DataFrame(records)

    metrics = [
        "pb_ratio",
        "asset_growth",
        "roe",
        "leverage",
        "earnings_stability",
        "momentum_6m",
        "momentum_12m",
        "volatility_3m",
        "volatility_12m",
    ]
    null_counts = {
        m: int(result[m].isna().sum()) for m in metrics if m in result.columns
    }
    any_nulls = {m: n for m, n in null_counts.items() if n > 0}
    if any_nulls:
        summary = "  ".join(f"{m}={n}" for m, n in any_nulls.items())
        logger.info("%s | nulls: %s", rebalance_date, summary)

    return result
