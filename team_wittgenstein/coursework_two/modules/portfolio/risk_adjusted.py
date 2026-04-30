"""Step 3: Risk-adjusted score calculation.

Scales composite scores by EWMA volatility so that high-volatility stocks
receive smaller positions. This allocates more weight to stocks with better
return per unit of risk.

Formula summary:
- Longs use ``composite_score / ewma_vol``
- Shorts use ``abs(composite_score) / ewma_vol``
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_risk_adjusted_scores(
    selected: pd.DataFrame,
    ewma_vols: pd.DataFrame,
) -> pd.DataFrame:
    """Merge EWMA volatility with selected stocks and compute risk-adjusted scores.

    Args:
        selected: DataFrame from stock_selector with symbol, sector, direction,
                  composite_score, percentile_rank, status.
        ewma_vols: DataFrame from ewma_volatility with symbol, ewma_vol.

    Returns:
        DataFrame with symbol, sector, direction, composite_score, ewma_vol,
        risk_adj_score. Stocks without EWMA vol are dropped.
    """
    merged = selected.merge(ewma_vols, on="symbol", how="inner")

    if merged.empty:
        logger.warning("No stocks with both selection and EWMA vol data")
        return pd.DataFrame(
            columns=[
                "symbol",
                "sector",
                "direction",
                "composite_score",
                "ewma_vol",
                "risk_adj_score",
            ]
        )

    # Drop stocks with zero or negative vol (shouldn't happen, but be safe)
    valid = merged[merged["ewma_vol"] > 0].copy()
    dropped = len(merged) - len(valid)
    if dropped > 0:
        logger.warning("Dropped %d stocks with non-positive EWMA vol", dropped)

    # Longs: composite / vol; Shorts: |composite| / vol
    valid["risk_adj_score"] = valid.apply(
        lambda row: (
            row["composite_score"] / row["ewma_vol"]
            if row["direction"] == "long"
            else abs(row["composite_score"]) / row["ewma_vol"]
        ),
        axis=1,
    )

    logger.info(
        "Risk-adjusted scores: %d stocks (%d long, %d short)",
        len(valid),
        (valid["direction"] == "long").sum(),
        (valid["direction"] == "short").sum(),
    )

    return valid[
        [
            "symbol",
            "sector",
            "direction",
            "composite_score",
            "ewma_vol",
            "risk_adj_score",
            "status",
            "percentile_rank",
            "buffer_months_count",
        ]
    ]
