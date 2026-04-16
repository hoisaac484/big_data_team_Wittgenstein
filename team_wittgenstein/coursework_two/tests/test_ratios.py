"""Tests for z-score normalisation, factor aggregation, and orthogonalisation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from modules.zscore.ratios import (
    FLIP_METRICS,
    MIN_GROUP_SIZE,
    MIN_OLS_ROWS,
    compute_factor_scores,
    orthogonalise_lowvol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_METRICS = [
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

_FACTOR_COLS = [
    "symbol",
    "calc_date",
    "value_score",
    "quality_score",
    "momentum_score",
    "lowvol_raw",
]

_ZSCORE_COLS = [
    "symbol",
    "calc_date",
    "z_pb_ratio",
    "z_asset_growth",
    "z_roe",
    "z_leverage",
    "z_earnings_stability",
    "z_momentum_6m",
    "z_momentum_12m",
    "z_volatility_3m",
    "z_volatility_12m",
]


def _make_df(
    n: int = 20,
    calc_date: date = date(2024, 1, 31),
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic winsorised DataFrame with N symbols and all metric columns."""
    np.random.seed(seed)
    symbols = [f"S{i:02d}" for i in range(n)]
    data = {"symbol": symbols, "calc_date": [calc_date] * n}
    for metric in _ALL_METRICS:
        data[metric] = np.random.standard_normal(n).tolist()
    return pd.DataFrame(data)


def _make_sector_map(df: pd.DataFrame, n_sectors: int = 2) -> dict:
    """Assign symbols round-robin to sectors, ensuring each sector is large enough."""
    sectors = [f"Sector{j}" for j in range(n_sectors)]
    return {sym: sectors[i % n_sectors] for i, sym in enumerate(df["symbol"])}


def _make_factor_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a factor_df suitable for orthogonalise_lowvol tests."""
    np.random.seed(seed)
    symbols = [f"S{i:03d}" for i in range(n)]
    calc_date = date(2024, 1, 31)
    momentum = np.random.standard_normal(n)
    # lowvol_raw correlated with momentum so orthogonalisation has something to remove
    lowvol_raw = 0.7 * momentum + 0.3 * np.random.standard_normal(n)
    return pd.DataFrame(
        {
            "symbol": symbols,
            "calc_date": [calc_date] * n,
            "value_score": np.random.standard_normal(n),
            "quality_score": np.random.standard_normal(n),
            "momentum_score": momentum,
            "lowvol_raw": lowvol_raw,
        }
    )


# ---------------------------------------------------------------------------
# TestComputeFactorScores — output structure
# ---------------------------------------------------------------------------


class TestComputeFactorScores:

    def test_factor_scores_columns(self):
        """factor_df contains exactly symbol, calc_date, and the four factor columns."""
        df = _make_df(n=20)
        sector_map = _make_sector_map(df, n_sectors=2)

        factor_df, _ = compute_factor_scores(df, sector_map)

        for col in _FACTOR_COLS:
            assert col in factor_df.columns, f"Missing column: {col}"

        # Raw metric columns must not survive into factor_df
        for metric in _ALL_METRICS:
            assert metric not in factor_df.columns, f"Raw metric leaked: {metric}"

    def test_zscore_df_columns(self):
        """zscore_df contains symbol, calc_date, and all nine z_* columns."""
        df = _make_df(n=20)
        sector_map = _make_sector_map(df, n_sectors=2)

        _, zscore_df = compute_factor_scores(df, sector_map)

        for col in _ZSCORE_COLS:
            assert col in zscore_df.columns, f"Missing z-score column: {col}"

    def test_flip_metrics_negative_correlation(self):
        """Higher raw pb_ratio produces a lower (more negative) z_pb_ratio."""
        df = _make_df(n=20)
        # Put all in one sector so within-group z-scoring applies cleanly
        sector_map = {sym: "Tech" for sym in df["symbol"]}

        # Sort symbols by pb_ratio ascending; the highest pb_ratio should get the
        # most negative z_pb_ratio because pb_ratio is a FLIP_METRIC.
        assert "pb_ratio" in FLIP_METRICS

        _, zscore_df = compute_factor_scores(df, sector_map)

        # Merge raw values back for comparison
        merged = pd.merge(
            df[["symbol", "pb_ratio"]],
            zscore_df[["symbol", "z_pb_ratio"]],
            on="symbol",
        )
        # Correlation between raw and z-scored should be negative
        corr = merged["pb_ratio"].corr(merged["z_pb_ratio"])
        assert corr < 0, f"Expected negative correlation for pb_ratio; got {corr:.4f}"

    def test_keep_metrics_positive_correlation(self):
        """Higher raw roe produces a higher (more positive) z_roe."""
        df = _make_df(n=20)
        sector_map = {sym: "Tech" for sym in df["symbol"]}

        assert "roe" not in FLIP_METRICS

        _, zscore_df = compute_factor_scores(df, sector_map)

        merged = pd.merge(
            df[["symbol", "roe"]],
            zscore_df[["symbol", "z_roe"]],
            on="symbol",
        )
        corr = merged["roe"].corr(merged["z_roe"])
        assert corr > 0, f"Expected positive correlation for roe; got {corr:.4f}"

    def test_all_nan_factor_is_nan(self):
        """If all sub-metrics for a factor are NaN, the factor score is NaN."""
        df = _make_df(n=20)
        sector_map = {sym: "Tech" for sym in df["symbol"]}

        # Set all value_score sub-metrics to NaN for the first symbol
        df.loc[df["symbol"] == "S00", "pb_ratio"] = np.nan
        df.loc[df["symbol"] == "S00", "asset_growth"] = np.nan

        factor_df, _ = compute_factor_scores(df, sector_map)

        row = factor_df.loc[factor_df["symbol"] == "S00", "value_score"]
        assert row.notna().sum() == 0 or pd.isna(
            row.iloc[0]
        ), "Expected NaN value_score when all sub-metrics are NaN"

    def test_partial_nan_uses_available(self):
        """If only one sub-metric is NaN, the factor is computed from the other."""
        df = _make_df(n=20)
        sector_map = {sym: "Tech" for sym in df["symbol"]}

        # Null out only pb_ratio for "S00"; asset_growth remains valid
        df.loc[df["symbol"] == "S00", "pb_ratio"] = np.nan

        factor_df, zscore_df = compute_factor_scores(df, sector_map)

        # value_score should still be non-NaN (uses only z_asset_growth)
        row = factor_df.loc[factor_df["symbol"] == "S00", "value_score"]
        assert pd.notna(
            row.iloc[0]
        ), "Expected non-NaN value_score when one sub-metric is available"

    def test_small_group_nan_zscores(self):
        """Groups smaller than MIN_GROUP_SIZE produce NaN z-scores."""
        # Build a 3-symbol group — smaller than MIN_GROUP_SIZE
        small_n = MIN_GROUP_SIZE - 1
        df = _make_df(n=small_n)
        sector_map = {sym: "SmallSector" for sym in df["symbol"]}

        _, zscore_df = compute_factor_scores(df, sector_map)

        for col in ["z_pb_ratio", "z_roe", "z_momentum_6m"]:
            assert (
                zscore_df[col].isna().all()
            ), f"Expected all NaN in {col} for group below MIN_GROUP_SIZE"


# ---------------------------------------------------------------------------
# TestOrthogonaliseLowvol
# ---------------------------------------------------------------------------


class TestOrthogonaliseLowvol:

    def test_orthogonalise_returns_lowvol_score(self):
        """Output has lowvol_score column and lowvol_raw is removed."""
        factor_df = _make_factor_df(n=100)

        result = orthogonalise_lowvol(factor_df)

        assert "lowvol_score" in result.columns
        assert "lowvol_raw" not in result.columns

    def test_orthogonalise_insufficient_rows(self):
        """Dates with fewer than MIN_OLS_ROWS observations yield NaN lowvol_score."""
        # Create two dates: one with enough rows, one with too few
        np.random.seed(0)
        n_large = MIN_OLS_ROWS + 10
        n_small = MIN_OLS_ROWS - 1

        date_large = date(2024, 1, 31)
        date_small = date(2024, 2, 29)

        rows_large = pd.DataFrame(
            {
                "symbol": [f"L{i:03d}" for i in range(n_large)],
                "calc_date": [date_large] * n_large,
                "value_score": np.random.standard_normal(n_large),
                "quality_score": np.random.standard_normal(n_large),
                "momentum_score": np.random.standard_normal(n_large),
                "lowvol_raw": np.random.standard_normal(n_large),
            }
        )
        rows_small = pd.DataFrame(
            {
                "symbol": [f"S{i:03d}" for i in range(n_small)],
                "calc_date": [date_small] * n_small,
                "value_score": np.random.standard_normal(n_small),
                "quality_score": np.random.standard_normal(n_small),
                "momentum_score": np.random.standard_normal(n_small),
                "lowvol_raw": np.random.standard_normal(n_small),
            }
        )
        factor_df = pd.concat([rows_large, rows_small], ignore_index=True)

        result = orthogonalise_lowvol(factor_df)

        small_rows = result[result["calc_date"] == date_small]
        large_rows = result[result["calc_date"] == date_large]

        assert (
            small_rows["lowvol_score"].isna().all()
        ), "Expected all NaN lowvol_score for date with insufficient paired rows"
        assert (
            large_rows["lowvol_score"].notna().all()
        ), "Expected valid lowvol_score for date with sufficient paired rows"

    def test_orthogonalise_reduces_correlation(self):
        """After orthogonalisation, lowvol_score vs momentum_score correlation ≈ 0."""
        factor_df = _make_factor_df(n=200, seed=99)

        # Confirm there IS meaningful correlation before orthogonalisation
        pre_corr = factor_df["lowvol_raw"].corr(factor_df["momentum_score"])
        msg = f"Pre-orth correlation too small ({pre_corr:.4f})"
        assert abs(pre_corr) > 0.3, msg

        result = orthogonalise_lowvol(factor_df)

        valid = result[result["lowvol_score"].notna()]
        post_corr = valid["lowvol_score"].corr(valid["momentum_score"])

        msg = f"Post-orth correlation not near zero: {post_corr:.4f}"
        assert abs(post_corr) == pytest.approx(0.0, abs=0.05), msg
