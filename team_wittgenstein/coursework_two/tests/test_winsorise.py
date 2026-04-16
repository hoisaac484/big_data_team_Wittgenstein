"""Tests for the winsorisation stage of the factor pipeline."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from modules.zscore.winsorise import METRICS, MIN_GROUP_SIZE, winsorise_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n: int = 20,
    calc_date: date = date(2024, 1, 31),
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic DataFrame with N symbols and all metric columns."""
    rng = np.random.default_rng(seed)
    symbols = [f"S{i:02d}" for i in range(n)]
    data = {"symbol": symbols, "calc_date": [calc_date] * n}
    for metric in METRICS:
        data[metric] = rng.standard_normal(n).tolist()
    return pd.DataFrame(data)


def _make_sector_map(df: pd.DataFrame, n_sectors: int = 3) -> dict:
    """Assign symbols round-robin to sectors."""
    sectors = [f"Sector{j}" for j in range(n_sectors)]
    return {sym: sectors[i % n_sectors] for i, sym in enumerate(df["symbol"])}


# ---------------------------------------------------------------------------
# TestWinsoriseMetrics
# ---------------------------------------------------------------------------


class TestWinsoriseMetrics:

    def test_clips_extreme_values(self):
        """Values beyond p95 are clipped to p95, values below p5 to p5."""
        df = _make_df(n=20)
        sector_map = _make_sector_map(df, n_sectors=1)

        # Inject an extreme outlier into pb_ratio for the first symbol
        df.loc[0, "pb_ratio"] = 1_000_000.0

        result = winsorise_metrics(df, sector_map)

        # The outlier should have been clipped down; it cannot remain at 1e6
        assert result.loc[0, "pb_ratio"] < 1_000_000.0

        # The clipped value must equal the 95th percentile of the original series
        original_valid = df["pb_ratio"].dropna()
        p95 = original_valid.quantile(0.95)
        assert result.loc[0, "pb_ratio"] == pytest.approx(p95, rel=1e-6)

    def test_none_values_unchanged(self):
        """NaN values pass through winsorisation without being imputed."""
        df = _make_df(n=20)
        sector_map = _make_sector_map(df, n_sectors=1)

        # Scatter NaNs across several metrics
        nan_positions = [(2, "pb_ratio"), (5, "roe"), (10, "momentum_6m")]
        for row, col in nan_positions:
            df.loc[row, col] = np.nan

        result = winsorise_metrics(df, sector_map)

        for row, col in nan_positions:
            assert pd.isna(
                result.loc[row, col]
            ), f"Expected NaN at row {row}, column {col}"

    def test_small_group_not_clipped(self):
        """Groups with fewer than MIN_GROUP_SIZE non-null observations are unchanged."""
        # Create a dataset where each (sector, calc_date) group has only 3 symbols
        n = 3
        df = _make_df(n=n)
        # Force extreme values so clipping would be obvious if it occurred
        df["pb_ratio"] = [100.0, 200.0, 300.0]
        sector_map = {sym: "TinySector" for sym in df["symbol"]}

        result = winsorise_metrics(df, sector_map)

        # Values must be unchanged because the group is too small
        assert n < MIN_GROUP_SIZE
        pd.testing.assert_series_equal(
            result["pb_ratio"].reset_index(drop=True),
            df["pb_ratio"].reset_index(drop=True),
        )

    def test_missing_sector_warning(self):
        """A symbol absent from sector_map triggers a logger.warning."""
        df = _make_df(n=20)
        # Deliberately omit the first symbol from the map
        sector_map = _make_sector_map(df, n_sectors=2)
        missing_symbol = df.iloc[0]["symbol"]
        del sector_map[missing_symbol]

        with patch("modules.zscore.winsorise.logger") as mock_logger:
            winsorise_metrics(df, sector_map)
            mock_logger.warning.assert_called_once()
            # The warning message should reference the missing symbol
            warning_args = mock_logger.warning.call_args[0]
            # The formatted message string or positional args should mention the symbol
            full_message = " ".join(str(a) for a in warning_args)
            assert missing_symbol in full_message or "no sector mapping" in full_message

    def test_missing_column_skipped(self):
        """A metric column absent from the DataFrame is silently skipped."""
        df = _make_df(n=20)
        sector_map = _make_sector_map(df)

        # Drop one of the standard metric columns
        df = df.drop(columns=["pb_ratio"])

        # Should not raise; the missing column is simply ignored
        result = winsorise_metrics(df, sector_map)
        assert "pb_ratio" not in result.columns

    def test_returns_same_shape(self):
        """Output DataFrame has exactly the same shape and columns as input."""
        df = _make_df(n=20)
        sector_map = _make_sector_map(df)

        result = winsorise_metrics(df, sector_map)

        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)
