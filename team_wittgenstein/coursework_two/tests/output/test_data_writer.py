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
# TestWriteFactorMetrics
# ---------------------------------------------------------------------------


def _make_factor_metrics_df(n: int = 5) -> pd.DataFrame:
    np.random.seed(1)
    return pd.DataFrame(
        {
            "symbol": [f"S{i:02d}" for i in range(n)],
            "calc_date": [date(2024, 1, 31)] * n,
            "pb_ratio": np.random.standard_normal(n),
            "asset_growth": np.random.standard_normal(n),
            "roe": np.random.standard_normal(n),
            "leverage": np.random.standard_normal(n),
            "earnings_stability": np.random.standard_normal(n),
            "momentum_6m": np.random.standard_normal(n),
            "momentum_12m": np.random.standard_normal(n),
            "volatility_3m": np.random.standard_normal(n),
            "volatility_12m": np.random.standard_normal(n),
        }
    )


class TestWriteFactorMetrics:

    def test_write_factor_metrics_empty(self):
        writer, mock_pg = _make_writer()
        result = writer.write_factor_metrics(pd.DataFrame())
        assert result == 0
        mock_pg.write_dataframe_on_conflict_do_nothing.assert_not_called()

    def test_write_factor_metrics_returns_count(self):
        writer, mock_pg = _make_writer()
        n = 6
        result = writer.write_factor_metrics(_make_factor_metrics_df(n))
        assert result == n

    def test_write_factor_metrics_correct_table_and_conflict(self):
        writer, mock_pg = _make_writer()
        writer.write_factor_metrics(_make_factor_metrics_df(n=3))
        kwargs = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1]
        assert kwargs.get("table_name") == "factor_metrics"
        assert kwargs.get("conflict_columns") == ["symbol", "calc_date"]

    def test_write_factor_metrics_passes_df_unchanged(self):
        writer, mock_pg = _make_writer()
        df = _make_factor_metrics_df(n=4)
        original_columns = list(df.columns)
        writer.write_factor_metrics(df)
        written_df = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1].get(
            "df"
        )
        assert list(written_df.columns) == original_columns


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


# ---------------------------------------------------------------------------
# TestWritePortfolioPositions
# ---------------------------------------------------------------------------


def _make_positions_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rebalance_date": [date(2024, 3, 29)] * n,
            "symbol": [f"S{i:02d}" for i in range(n)],
            "sector": ["IT"] * n,
            "direction": ["long"] * n,
            "ewma_vol": [0.2] * n,
            "risk_adj_score": [1.0] * n,
            "target_weight": [0.05] * n,
            "final_weight": [0.05] * n,
            "liquidity_capped": [False] * n,
            "trade_action": ["trade"] * n,
        }
    )


# ---------------------------------------------------------------------------
# TestWriteBacktestReturns
# ---------------------------------------------------------------------------


def _make_backtest_returns_df(n: int = 3) -> pd.DataFrame:
    from datetime import date

    return pd.DataFrame(
        {
            "scenario_id": ["baseline"] * n,
            "rebalance_date": [
                date(2024, i + 1, 29 if i == 1 else 31) for i in range(n)
            ],
            "gross_return": [0.01, -0.02, 0.03][:n],
            "net_return": [0.009, -0.021, 0.028][:n],
            "long_return": [0.015, -0.01, 0.04][:n],
            "short_return": [0.005, 0.01, 0.01][:n],
            "benchmark_return": [0.005, -0.015, 0.02][:n],
            "excess_return": [0.004, -0.006, 0.008][:n],
            "cumulative_return": [0.009, -0.012, 0.016][:n],
            "turnover": [0.5, 0.6, 0.7][:n],
            "transaction_cost": [0.001, 0.0015, 0.002][:n],
        }
    )


class TestWriteBacktestReturns:

    def test_write_backtest_returns_empty(self):
        writer, mock_pg = _make_writer()
        result = writer.write_backtest_returns(pd.DataFrame(), "baseline")
        assert result == 0
        mock_pg.write_dataframe_on_conflict_do_nothing.assert_not_called()

    def test_write_backtest_returns_none(self):
        writer, mock_pg = _make_writer()
        result = writer.write_backtest_returns(None, "baseline")
        assert result == 0
        mock_pg.write_dataframe_on_conflict_do_nothing.assert_not_called()

    def test_write_backtest_returns_count(self):
        writer, mock_pg = _make_writer()
        n = 3
        result = writer.write_backtest_returns(_make_backtest_returns_df(n), "baseline")
        assert result == n

    def test_write_backtest_returns_correct_table_and_conflict(self):
        writer, mock_pg = _make_writer()
        writer.write_backtest_returns(_make_backtest_returns_df(), "baseline")
        kwargs = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1]
        assert kwargs.get("table_name") == "backtest_returns"
        assert kwargs.get("conflict_columns") == ["scenario_id", "rebalance_date"]

    def test_write_backtest_returns_correct_schema(self):
        writer, mock_pg = _make_writer()
        writer.write_backtest_returns(_make_backtest_returns_df(), "baseline")
        kwargs = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1]
        assert kwargs.get("schema") == "team_wittgenstein"


class TestWritePortfolioPositions:

    def test_write_portfolio_positions_empty(self):
        writer, mock_pg = _make_writer()
        result = writer.write_portfolio_positions(pd.DataFrame())
        assert result == 0
        mock_pg.write_dataframe_on_conflict_do_nothing.assert_not_called()

    def test_write_portfolio_positions_none(self):
        writer, mock_pg = _make_writer()
        result = writer.write_portfolio_positions(None)
        assert result == 0
        mock_pg.write_dataframe_on_conflict_do_nothing.assert_not_called()

    def test_write_portfolio_positions_returns_count(self):
        writer, mock_pg = _make_writer()
        n = 6
        result = writer.write_portfolio_positions(_make_positions_df(n))
        assert result == n

    def test_write_portfolio_positions_correct_table_and_conflict(self):
        writer, mock_pg = _make_writer()
        writer.write_portfolio_positions(_make_positions_df(3))
        kwargs = mock_pg.write_dataframe_on_conflict_do_nothing.call_args[1]
        assert kwargs.get("table_name") == "portfolio_positions"
        assert kwargs.get("conflict_columns") == ["rebalance_date", "symbol"]


class TestWriteBacktestSummary:

    def test_deletes_existing_then_writes(self):
        writer, mock_pg = _make_writer()
        summary = {"scenario_id": "baseline", "annualised_return": 0.12}
        writer.write_backtest_summary(summary)
        mock_pg.execute.assert_called_once()
        call_sql = mock_pg.execute.call_args[0][0]
        assert "DELETE" in call_sql
        assert "baseline" in str(mock_pg.execute.call_args)
        mock_pg.write_dataframe.assert_called_once()

    def test_write_dataframe_called_with_correct_table(self):
        writer, mock_pg = _make_writer()
        writer.write_backtest_summary({"scenario_id": "test"})
        args = mock_pg.write_dataframe.call_args[0]
        assert args[1] == "backtest_summary"

    def test_write_dataframe_uses_append(self):
        writer, mock_pg = _make_writer()
        writer.write_backtest_summary({"scenario_id": "test"})
        kwargs = mock_pg.write_dataframe.call_args[1]
        assert kwargs.get("if_exists") == "append"
