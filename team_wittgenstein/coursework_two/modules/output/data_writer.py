"""Data writer for coursework_two — writes processed factor data to PostgreSQL."""

import logging

import pandas as pd

from modules.db.db_connection import PostgresConnection

logger = logging.getLogger(__name__)

SCHEMA = "team_wittgenstein"

# Pipeline column names → factor_scores table column names
_FACTOR_SCORE_COLUMNS = {
    "calc_date": "score_date",
    "value_score": "z_value",
    "quality_score": "z_quality",
    "momentum_score": "z_momentum",
    "lowvol_score": "z_low_vol",
}


class DataWriter:
    """Writes factor pipeline outputs to PostgreSQL.

    Args:
        pg: Active PostgresConnection instance.
    """

    def __init__(self, pg: PostgresConnection):
        self.pg = pg

    def write_factor_scores(self, df: pd.DataFrame) -> int:
        """Write processed factor scores to factor_scores using ON CONFLICT DO NOTHING.

        Renames pipeline columns to match the DB schema:
            calc_date      → score_date
            value_score    → z_value
            quality_score  → z_quality
            momentum_score → z_momentum
            lowvol_score   → z_low_vol

        Safe to re-run — existing (symbol, score_date) pairs are skipped.

        Args:
            df: DataFrame with columns symbol, calc_date, value_score,
                quality_score, momentum_score, lowvol_score.

        Returns:
            Number of rows attempted.
        """
        if df is None or df.empty:
            logger.warning("No factor scores to write.")
            return 0

        out = df.rename(columns=_FACTOR_SCORE_COLUMNS)
        self.pg.write_dataframe_on_conflict_do_nothing(
            df=out,
            table_name="factor_scores",
            schema=SCHEMA,
            conflict_columns=["symbol", "score_date"],
        )
        logger.info("Wrote %d factor score rows to %s.factor_scores", len(out), SCHEMA)
        return len(out)

    def write_factor_metrics(self, df: pd.DataFrame) -> int:
        """Write raw factor ratios to factor_metrics.

        Args:
            df: DataFrame with columns symbol, calc_date, pb_ratio,
                asset_growth, roe, leverage, earnings_stability,
                momentum_6m, momentum_12m, volatility_3m, volatility_12m.

        Returns:
            Number of rows attempted.
        """
        if df is None or df.empty:
            logger.warning("No factor metrics to write.")
            return 0
        self.pg.write_dataframe_on_conflict_do_nothing(
            df=df,
            table_name="factor_metrics",
            schema=SCHEMA,
            conflict_columns=["symbol", "calc_date"],
        )
        logger.info("Wrote %d metric rows to %s.factor_metrics", len(df), SCHEMA)
        return len(df)

    def write_factor_zscores(self, df: pd.DataFrame) -> int:
        """Write individual sub-metric z-scores to factor_zscores.

        Args:
            df: DataFrame with columns symbol, calc_date, z_pb_ratio,
                z_asset_growth, z_roe, z_leverage, z_earnings_stability,
                z_momentum_6m, z_momentum_12m, z_volatility_3m, z_volatility_12m.

        Returns:
            Number of rows attempted.
        """
        if df is None or df.empty:
            logger.warning("No factor z-scores to write.")
            return 0
        self.pg.write_dataframe_on_conflict_do_nothing(
            df=df,
            table_name="factor_zscores",
            schema=SCHEMA,
            conflict_columns=["symbol", "calc_date"],
        )
        logger.info("Wrote %d z-score rows to %s.factor_zscores", len(df), SCHEMA)
        return len(df)

    def write_backtest_returns(self, df: pd.DataFrame, scenario_id: str) -> int:
        """Write monthly backtest returns to backtest_returns table.

        Args:
            df:          DataFrame with columns matching backtest_returns schema.
            scenario_id: Scenario identifier (e.g. 'baseline').

        Returns:
            Number of rows attempted.
        """
        if df is None or df.empty:
            logger.warning("No backtest returns to write.")
            return 0
        self.pg.write_dataframe_on_conflict_do_nothing(
            df=df,
            table_name="backtest_returns",
            schema=SCHEMA,
            conflict_columns=["scenario_id", "rebalance_date"],
        )
        logger.info(
            "Wrote %d backtest return rows to %s.backtest_returns (scenario=%s)",
            len(df),
            SCHEMA,
            scenario_id,
        )
        return len(df)

    def write_backtest_summary(self, summary: dict) -> None:
        """Upsert a summary row into backtest_summary keyed by scenario_id.

        Deletes any existing row for the scenario_id, then inserts the new row.
        Ensures the latest computation always overwrites prior metrics.

        Args:
            summary: Dict with keys matching backtest_summary columns (except
                     summary_id and created_at which are auto-generated).
        """
        scenario_id = summary["scenario_id"]
        self.pg.execute(
            "DELETE FROM team_wittgenstein.backtest_summary "
            "WHERE scenario_id = :scenario_id",
            {"scenario_id": scenario_id},
        )
        df = pd.DataFrame([summary])
        self.pg.write_dataframe(df, "backtest_summary", SCHEMA, if_exists="append")
        logger.info(
            "Wrote summary row to %s.backtest_summary (scenario=%s)",
            SCHEMA,
            scenario_id,
        )

    def write_portfolio_positions(self, df: pd.DataFrame) -> int:
        """Write portfolio positions to portfolio_positions table.

        Args:
            df: DataFrame with rebalance_date, symbol, sector, direction,
                ewma_vol, risk_adj_score, target_weight, final_weight,
                liquidity_capped, trade_action.

        Returns:
            Number of rows attempted.
        """
        if df is None or df.empty:
            logger.warning("No portfolio positions to write.")
            return 0
        self.pg.write_dataframe_on_conflict_do_nothing(
            df=df,
            table_name="portfolio_positions",
            schema=SCHEMA,
            conflict_columns=["rebalance_date", "symbol"],
        )
        logger.info("Wrote %d position rows to %s.portfolio_positions", len(df), SCHEMA)
        return len(df)
