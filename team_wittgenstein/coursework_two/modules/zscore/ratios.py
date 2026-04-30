"""Stage 3, Steps 3 and 4: z-score normalisation and sub-factor aggregation.

Step 3 computes sector-relative z-scores for each raw metric within each
``(GICS sector, calc_date)`` group using ``(x - mu_sector) / sigma_sector``.
The sign convention is adjusted so that higher z-scores are always more
attractive. ``pb_ratio``, ``asset_growth``, ``leverage``,
``earnings_stability``, ``volatility_3m``, and ``volatility_12m`` are flipped.
``roe``, ``momentum_6m``, and ``momentum_12m`` retain their natural sign.

Step 4 aggregates the signed z-scores into factor-level scores using
equal-weighted means. ``value_score`` averages price-to-book and asset growth,
``quality_score`` averages ROE, leverage, and earnings stability,
``momentum_score`` averages 6m and 12m momentum, and ``lowvol_raw`` averages
3m and 12m volatility before orthogonalisation.

If one sub-metric is missing for a stock, the factor score is the mean of the
remaining available sub-metrics. If all sub-metrics are missing, the factor
score remains missing.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_GROUP_SIZE = 5  # minimum non-null values to compute reliable sector z-scores

# Metrics where lower raw value = more attractive → flip sign after z-scoring
FLIP_METRICS = {
    "pb_ratio",
    "asset_growth",
    "leverage",
    "earnings_stability",
    "volatility_3m",
    "volatility_12m",
}

# Sub-metric composition of each factor
FACTOR_METRICS = {
    "value_score": ["pb_ratio", "asset_growth"],
    "quality_score": ["roe", "leverage", "earnings_stability"],
    "momentum_score": ["momentum_6m", "momentum_12m"],
    "lowvol_raw": ["volatility_3m", "volatility_12m"],
}


def compute_factor_scores(
    df: pd.DataFrame, sector_map: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Z-score normalisation and aggregate into factor scores.

    Args:
        df:         Winsorised DataFrame from winsorise_metrics — rows are
                    (symbol, calc_date), columns include the 9 raw metrics.
        sector_map: dict mapping symbol → GICS sector string.

    Returns:
        Tuple of two dataframes:
        - ``factor_df`` with factor-level scores by ``symbol`` and ``calc_date``
        - ``zscore_df`` with the per-metric signed z-score audit trail

        Raw metric columns are dropped from ``factor_df``.
    """
    df = df.copy()
    df["_sector"] = df["symbol"].map(sector_map)

    missing = df.loc[df["_sector"].isna(), "symbol"].unique()
    if len(missing):
        logger.warning(
            "%d symbols have no sector mapping — z-scores will be NaN: %s",
            len(missing),
            sorted(missing)[:10],
        )

    all_metrics = [m for factor_cols in FACTOR_METRICS.values() for m in factor_cols]

    # ── Step 3: Z-score per (sector, calc_date) ────────────────────────────────
    def _zscore(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < MIN_GROUP_SIZE:
            return pd.Series(np.nan, index=series.index)
        mu = valid.mean()
        sigma = valid.std()
        if sigma == 0:
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    z_cols = {}
    for metric in all_metrics:
        if metric not in df.columns:
            continue
        z = df.groupby(["_sector", "calc_date"])[metric].transform(_zscore)
        sign = -1 if metric in FLIP_METRICS else 1
        z_cols[f"_z_{metric}"] = sign * z

    df = pd.concat([df, pd.DataFrame(z_cols, index=df.index)], axis=1)

    # ── Step 4: Sub-factor aggregation ────────────────────────────────────────
    for factor, metrics in FACTOR_METRICS.items():
        z_names = [f"_z_{m}" for m in metrics if f"_z_{m}" in df.columns]
        if not z_names:
            df[factor] = np.nan
            continue
        # mean(axis=1, skipna=True): uses available sub-metrics, NaN if all missing
        df[factor] = df[z_names].mean(axis=1, skipna=True)
        # If every sub-metric was NaN, mean returns NaN — set explicitly for clarity
        all_nan = df[z_names].isna().all(axis=1)
        df.loc[all_nan, factor] = np.nan

    # ── Extract z-scores before cleanup ───────────────────────────────────────
    z_rename = {f"_z_{m}": f"z_{m}" for m in all_metrics if f"_z_{m}" in df.columns}
    zscore_df = (
        df[["symbol", "calc_date"] + list(z_rename.keys())]
        .rename(columns=z_rename)
        .reset_index(drop=True)
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    drop_cols = [c for c in df.columns if c.startswith("_z_") or c == "_sector"]
    raw_metric_cols = [m for m in all_metrics if m in df.columns]
    df = df.drop(columns=drop_cols + raw_metric_cols)

    logger.info(
        "Factor scores computed: %d rows, %d dates",
        len(df),
        df["calc_date"].nunique(),
    )
    return df, zscore_df


MIN_OLS_ROWS = (
    50  # minimum paired observations for cross-sectional OLS to be meaningful
)


def orthogonalise_lowvol(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Replace lowvol_raw with the residuals from cross-sectional OLS
    of lowvol_raw on momentum_score at each rebalancing date.

    For each calc_date:
        lowvol_raw = α + β·momentum_score + ε
        lowvol_score = ε  (residual)

    Rows where either lowvol_raw or momentum_score is NaN are excluded from the
    regression but receive NaN in lowvol_score. If fewer than MIN_OLS_ROWS paired
    observations exist for a date, lowvol_score is set to NaN for all rows on
    that date.

    Args:
        df: DataFrame with columns lowvol_raw and momentum_score, plus calc_date.

    Returns:
        DataFrame with lowvol_raw dropped and lowvol_score added.
    """
    df = df.copy()
    lowvol_score = pd.Series(np.nan, index=df.index)

    for date, group in df.groupby("calc_date"):
        mask = group["lowvol_raw"].notna() & group["momentum_score"].notna()
        paired = group[mask]
        if len(paired) < MIN_OLS_ROWS:
            logger.debug(
                "%s | orthogonalise_lowvol: only %d paired rows — skipping OLS",
                date,
                len(paired),
            )
            continue
        slope, intercept, _, _, _ = stats.linregress(
            paired["momentum_score"].values,
            paired["lowvol_raw"].values,
        )
        predicted = intercept + slope * group.loc[mask, "momentum_score"]
        residuals = group.loc[mask, "lowvol_raw"] - predicted
        # Re-standardise so lowvol_score is on the same scale as other factor scores
        std = residuals.std(ddof=1)
        if std > 0:
            residuals = (residuals - residuals.mean()) / std
        lowvol_score.loc[residuals.index] = residuals.values

    df["lowvol_score"] = lowvol_score
    df = df.drop(columns=["lowvol_raw"])

    n_valid = lowvol_score.notna().sum()
    logger.info(
        "Orthogonalisation complete: %d/%d rows have lowvol_score",
        n_valid,
        len(df),
    )
    return df
