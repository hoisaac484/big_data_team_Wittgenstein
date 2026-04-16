"""
Stage 3, Step 2: Winsorisation of raw factor ratios.

Within each (GICS sector, calc_date) group, every metric is clipped to the
[5th, 95th] percentile of non-null values in that group. This prevents a
single extreme outlier from distorting sector means and standard deviations
in the Z-score stage.

None values are excluded from percentile calculation and remain None after
winsorisation — they are never imputed.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

METRICS = [
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

MIN_GROUP_SIZE = 5  # minimum non-null observations to compute reliable percentiles


def winsorise_metrics(
    df: pd.DataFrame,
    sector_map: dict,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.DataFrame:
    """Winsorise each metric within each (sector, calc_date) group.

    Args:
        df:         DataFrame from calculate_ratios — rows are (symbol, calc_date).
        sector_map: dict mapping symbol → GICS sector string.
        lower:      Lower percentile clip bound (default 0.05).
        upper:      Upper percentile clip bound (default 0.95).

    Returns:
        DataFrame with the same shape as df, metrics clipped per group.
    """
    df = df.copy()
    df["_sector"] = df["symbol"].map(sector_map)

    missing = df.loc[df["_sector"].isna(), "symbol"].unique()
    if len(missing):
        logger.warning(
            "%d symbols have no sector mapping — winsorisation skipped for them: %s",
            len(missing),
            sorted(missing)[:10],
        )

    def _clip(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < MIN_GROUP_SIZE:
            return series
        p_lo = valid.quantile(lower)
        p_hi = valid.quantile(upper)
        return series.clip(lower=p_lo, upper=p_hi)

    for metric in METRICS:
        if metric not in df.columns:
            continue
        df[metric] = df.groupby(["_sector", "calc_date"])[metric].transform(_clip)

    df = df.drop(columns=["_sector"])
    return df
