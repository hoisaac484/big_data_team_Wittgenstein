"""Tests for DataWriter — DB layer is mocked entirely."""

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from modules.output.data_writer import DataWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factor_scores_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal factor scores DataFrame matching the pipeline column names."""
    np.random.seed(42)
    symbols = [f"S{i:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "calc_date": [date(2024, 1, 31)] * n,
            "value_score": np.random.standard_normal(n),
            "quality_score": np.random.standard_normal(n),
            "momentum_score": np.random.standard_normal(n),
            "lowvol_score": np.random.standard_normal(n),
        }
    )


def _make_factor_zscores_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal factor z-scores DataFrame."""
    np.random.seed(0)
    symbols = [f"S{i:02d}" for i in range(n)]
    data = {"symbol": symbols, "calc_date": [date(2024, 1, 31)] * n}
    for col in [
        "z_pb_ratio",
        "z_asset_growth",
        "z_roe",
        "z_leverage",
        "z_earnings_stability",
        "z_momentum_6m",
        "z_momentum_12m",
        "z_volatility_3m",
        "z_volatility_12m",
    ]:
        data[col] = np.random.standard_normal(n)
    return pd.DataFrame(data)


def _make_writer() -> tuple[DataWriter, MagicMock]:
    """Return a (DataWriter, mock_pg) pair."""
    mock_pg = MagicMock()
    return DataWriter(mock_pg), mock_pg


# ---------------------------------------------------------------------------
# TestWriteFactorScores
# ---------------------------------------------------------------------------


class TestWriteFactorScores:

    def test_write_factor_scores_renames_columns(self):
        """write_factor_scores calls the DB with the schema-mapped column names."""
        writer, mock_pg = _make_writer()
        df = _make_factor_scores_df(n=3)

        writer.write_factor_scores(df)

        mock_pg.write_dataframe_on_conflict_do_nothing.assert_called_once()
        kwargs = mock_pg.write_dataframe_on_conflict_do_nothing.call_args

        # Retrieve the DataFrame passed to the DB call (positional or keyword)
        written_df = kwargs[1].get("df") if kwargs[1] else kwargs[0][0]

        # Pipeline names must be gone; DB schema names must be present
        assert "calc_date" not in written_df.columns
        assert "score_date" in written_df.columns
        assert "value_score" not in written_df.columns
        assert "z_value" in written_df.columns
        assert "quality_score" not in written_df.columns
        assert "z_quality" in written_df.columns
        assert "momentum_score" not in written_df.columns
        assert "z_momentum" in written_df.columns
        assert "lowvol_score" not in written_df.columns
        assert "z_low_vol" in written_df.columns

        # symbol column is passed through unchanged
        assert "symbol" in written_df.columns

    def test_write_factor_scores_empty(self):
        """write_factor_scores returns 0 and does not touch the DB when df is empty."""
        writer, mock_pg = _make_writer()
        empty_df = pd.DataFrame(
            columns=[
                "symbol",
                "calc_date",
                "value_score",
                "quality_score",
                "momentum_score",
                "lowvol_score",
            ]
        )

        result = writer.write_factor_scores(empty_df)

        assert result == 0
        mock_pg.write_dataframe_on_conflict_do_nothing.assert_not_called()

    def test_write_factor_scores_returns_count(self):
        """write_factor_scores returns the number of rows in the DataFrame."""
        writer, mock_pg = _make_writer()
        n = 7
        df = _make_factor_scores_df(n=n)

        result = writer.write_factor_scores(df)

        assert result == n

    def test_write_factor_scores_correct_table_and_conflict(self):
        """write_factor_scores passes correct table_name and conflict columns."""
        writer, mock_pg = _make_writer()
        df = _make_factor_scores_df(n=3)

        writer.write_factor_scores(df)

        kwargs = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1]
        assert kwargs.get("table_name") == "factor_scores"
        assert kwargs.get("conflict_columns") == ["symbol", "score_date"]


# ---------------------------------------------------------------------------
# TestWriteFactorZscores
# ---------------------------------------------------------------------------


class TestWriteFactorZscores:

    def test_write_factor_zscores_correct_table(self):
        """write_factor_zscores calls write_dataframe_on_conflict_do_nothing."""
        writer, mock_pg = _make_writer()
        df = _make_factor_zscores_df(n=4)

        writer.write_factor_zscores(df)

        mock_pg.write_dataframe_on_conflict_do_nothing.assert_called_once()
        kwargs = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1]
        assert kwargs.get("table_name") == "factor_zscores"

    def test_write_factor_zscores_empty(self):
        """write_factor_zscores returns 0 and does not touch the DB when df is empty."""
        writer, mock_pg = _make_writer()
        empty_df = pd.DataFrame(columns=["symbol", "calc_date", "z_pb_ratio", "z_roe"])

        result = writer.write_factor_zscores(empty_df)

        assert result == 0
        mock_pg.write_dataframe_on_conflict_do_nothing.assert_not_called()

    def test_write_factor_zscores_returns_count(self):
        """write_factor_zscores returns the number of rows in the DataFrame."""
        writer, mock_pg = _make_writer()
        n = 9
        df = _make_factor_zscores_df(n=n)

        result = writer.write_factor_zscores(df)

        assert result == n

    def test_write_factor_zscores_conflict_columns(self):
        """write_factor_zscores uses conflict_columns=['symbol', 'calc_date']."""
        writer, mock_pg = _make_writer()
        df = _make_factor_zscores_df(n=3)

        writer.write_factor_zscores(df)

        kwargs = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1]
        assert kwargs.get("conflict_columns") == ["symbol", "calc_date"]

    def test_write_factor_zscores_passes_df_unchanged(self):
        """write_factor_zscores does not rename or mutate the DataFrame."""
        writer, mock_pg = _make_writer()
        df = _make_factor_zscores_df(n=4)
        original_columns = list(df.columns)

        writer.write_factor_zscores(df)

        written_df = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1].get(
            "df"
        )
        assert list(written_df.columns) == original_columns
