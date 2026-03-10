"""Company universe filtering helpers."""

import logging
import os

import requests

logger = logging.getLogger(__name__)

SEC_EXCHANGE_LIST_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def _ticker_candidates(symbol):
    """Generate common ticker format variants for matching."""
    raw = str(symbol or "").strip().upper()
    if not raw:
        return []
    variants = [raw]
    if "." in raw:
        variants.append(raw.replace(".", "-"))
    if "-" in raw:
        variants.append(raw.replace("-", "."))
    # Keep order while deduplicating.
    return list(dict.fromkeys(variants))


def filter_sec_listed_symbols(symbols, user_agent=None, timeout=30):
    """Filter symbols to only those present in SEC exchange listings.

    Args:
        symbols: Iterable of ticker symbols.
        user_agent: Optional SEC-compliant User-Agent.
        timeout: HTTP timeout in seconds.

    Returns:
        tuple[list[str], list[str]]: (passed_symbols, dropped_symbols)
    """
    cleaned = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not cleaned:
        return [], []

    ua = (
        user_agent
        or os.getenv("EDGAR_USER_AGENT")
        or os.getenv("SEC_USER_AGENT")
        or "research@example.com"
    )
    headers = {"User-Agent": ua}

    try:
        response = requests.get(SEC_EXCHANGE_LIST_URL, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning(
            "SEC listing filter unavailable (%s); continuing with original symbols.",
            exc,
        )
        return cleaned, []

    fields = payload.get("fields", [])
    rows = payload.get("data", [])
    if not fields or not rows:
        logger.warning("SEC listing payload is empty; continuing with original symbols.")
        return cleaned, []

    try:
        ticker_idx = fields.index("ticker")
    except ValueError:
        logger.warning(
            "SEC listing payload missing 'ticker' field; continuing with original symbols."
        )
        return cleaned, []

    listed = {
        str(row[ticker_idx]).strip().upper()
        for row in rows
        if isinstance(row, list) and len(row) > ticker_idx and row[ticker_idx]
    }

    passed = []
    dropped = []
    for symbol in cleaned:
        if symbol in listed:
            passed.append(symbol)
            continue

        # If dotted ticker is not listed, retry with hyphenated variant
        # and normalize to that SEC-listed form.
        if "." in symbol:
            hyphen_symbol = symbol.replace(".", "-")
            if hyphen_symbol in listed:
                passed.append(hyphen_symbol)
                logger.info(
                    "SEC listing filter remap: %s -> %s",
                    symbol,
                    hyphen_symbol,
                )
                continue

        # Fallback to candidate match for other representations.
        candidates = _ticker_candidates(symbol)
        if any(candidate in listed for candidate in candidates):
            passed.append(symbol)
        else:
            dropped.append(symbol)

    logger.info(
        "SEC listing filter: %d input, %d passed, %d dropped",
        len(cleaned),
        len(passed),
        len(dropped),
    )
    if dropped:
        logger.warning(
            "Dropped %d symbols not found in SEC listings (first 20): %s",
            len(dropped),
            dropped[:20],
        )

    return passed, dropped
