"""Smoke test: run the liquidity filter against the real database.

Creates the liquidity_metrics table if it doesn't exist, runs the
two-stage filter for a single date, and prints the results.
"""

from datetime import date
from pathlib import Path

import yaml

from modules.db.db_connection import PostgresConnection
from modules.liquidity.liquidity_filter import LiquidityConfig, run_liquidity_filter


def main():
    # Load config
    config_path = Path(__file__).parent / "config" / "conf.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Connect to PostgreSQL
    pg_cfg = cfg["postgres"]
    db = PostgresConnection(
        host=pg_cfg["host"],
        port=pg_cfg["port"],
        database=pg_cfg["database"],
        user=pg_cfg["user"],
        password=pg_cfg["password"],
    )

    if not db.test_connection():
        print("Cannot connect to PostgreSQL. Is Docker running?")
        return

    # Create CW2 tables if they don't exist
    sql_path = Path(__file__).parent / "sql" / "create_cw2_tables.sql"
    db.execute_sql_file(str(sql_path))
    print("Tables created (or already exist).")

    # Count total stocks in price_data
    total = db.read_query(
        "SELECT COUNT(DISTINCT symbol) AS n FROM team_wittgenstein.price_data"
    )
    print(f"\nTotal stocks in price_data: {total.iloc[0]['n']}")

    # Run the liquidity filter
    liq_cfg = LiquidityConfig(**cfg["liquidity"])
    rebalance_date = date(2024, 1, 1)
    print(f"\nRunning liquidity filter for {rebalance_date}...")
    print(f"  ADTV min dollar: ${liq_cfg.adtv_min_dollar:,.0f}")
    print(f"  ILLIQ removal: top {liq_cfg.illiq_removal_pct:.0%}")

    survivors = run_liquidity_filter(db, rebalance_date, liq_cfg)
    print(f"\nSurvivors: {len(survivors)} stocks")

    # Show pass/fail breakdown
    breakdown = db.read_query(
        """
        SELECT passes_adv, passes_illiq, passes_filter, COUNT(*) AS n
        FROM team_wittgenstein.liquidity_metrics
        WHERE calc_date = :calc_date
        GROUP BY passes_adv, passes_illiq, passes_filter
        ORDER BY passes_filter DESC, passes_adv DESC
        """,
        {"calc_date": rebalance_date},
    )
    if not breakdown.empty:
        print(f"\n--- Pass/fail breakdown ---")
        print(breakdown.to_string(index=False))

    # Show stocks that failed each stage
    failed_adv = db.read_query(
        """
        SELECT symbol, adv_20d
        FROM team_wittgenstein.liquidity_metrics
        WHERE calc_date = :calc_date AND NOT passes_adv
        ORDER BY adv_20d
        """,
        {"calc_date": rebalance_date},
    )
    if not failed_adv.empty:
        print(f"\n--- Failed ADTV floor ({len(failed_adv)} stocks) ---")
        print(failed_adv.to_string(index=False))
    else:
        print("\n--- No stocks failed the ADTV floor ---")

    failed_illiq = db.read_query(
        """
        SELECT symbol, amihud_illiq, illiq_rank_pct
        FROM team_wittgenstein.liquidity_metrics
        WHERE calc_date = :calc_date AND passes_adv AND NOT passes_illiq
        ORDER BY illiq_rank_pct DESC
        """,
        {"calc_date": rebalance_date},
    )
    if not failed_illiq.empty:
        print(f"\n--- Failed ILLIQ filter ({len(failed_illiq)} stocks) ---")
        print(failed_illiq.to_string(index=False))


if __name__ == "__main__":
    main()
