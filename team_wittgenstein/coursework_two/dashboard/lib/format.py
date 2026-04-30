"""Number and date formatting helpers for consistent display.

Every page calls these instead of hand-formatting. Keeps the dashboard
visually consistent and reduces mistakes.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd


def pct(value: float, places: int = 2) -> str:
    """Format a decimal as a percentage. 0.0316 -> '3.16%'."""
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{places}f}%"


def pct_signed(value: float, places: int = 2) -> str:
    """Same as pct but with explicit + sign for positives."""
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.{places}f}%"


def num(value: float, places: int = 3) -> str:
    """Format a float to a fixed number of decimal places."""
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{places}f}"


def num_signed(value: float, places: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{places}f}"


def big_num(value: float) -> str:
    """Compact format for big numbers: 1500000 -> '1.5M', 432 -> '432'."""
    if value is None or pd.isna(value):
        return "-"
    abs_v = abs(value)
    if abs_v >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_v >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}" if abs_v >= 1 else f"{value:.2f}"


def fmt_date(d, format: str = "%b %Y") -> str:
    """Format a date or string for display."""
    if d is None or pd.isna(d):
        return "-"
    if isinstance(d, str):
        d = pd.to_datetime(d)
    if isinstance(d, datetime):
        return d.strftime(format)
    if isinstance(d, date):
        return d.strftime(format)
    return str(d)


def fmt_date_range(start, end) -> str:
    """Format a date range like 'Mar 2021 - Mar 2026'."""
    return f"{fmt_date(start)} - {fmt_date(end)}"


def safe_get(series: pd.Series, key: str, default: float = math.nan) -> float:
    """Defensive scalar getter for series-like objects."""
    if series is None or len(series) == 0:
        return default
    try:
        v = series.get(key, default)
        return float(v) if v is not None and not pd.isna(v) else default
    except (KeyError, TypeError, ValueError):
        return default


# Mapping for human-readable scenario names in dropdowns
SCENARIO_LABELS = {
    "baseline": "Baseline (default parameters)",
    "cost_frictionless": "Cost: Frictionless (0 bps)",
    "cost_low": "Cost: Low (10 bps)",
    "cost_high": "Cost: High (50 bps)",
    "excl_value": "Exclude Value factor",
    "excl_quality": "Exclude Quality factor",
    "excl_momentum": "Exclude Momentum factor",
    "excl_low_vol": "Exclude Low Volatility factor",
}


def scenario_label(scenario_id: str) -> str:
    """Human-readable label for a scenario, including ``sens_*`` variants."""
    if scenario_id in SCENARIO_LABELS:
        return SCENARIO_LABELS[scenario_id]
    if scenario_id.startswith("sens_"):
        # sens_sel_0.05 -> "Sensitivity: selection threshold = 0.05"
        parts = scenario_id.replace("sens_", "").split("_")
        if len(parts) >= 2:
            param = parts[0]
            value = "_".join(parts[1:])
            label_map = {
                "sel": "selection threshold",
                "ic": "IC lookback (months)",
                "ewma": "EWMA lambda",
                "notrade": "no-trade threshold",
                "buffer": "buffer exit threshold",
            }
            return f"Sensitivity: {label_map.get(param, param)} = {value}"
    return scenario_id
