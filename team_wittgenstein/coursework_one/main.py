import logging
from pathlib import Path

import pandas as pd
import yaml

from modules.db.db_connection import PostgresConnection, MongoConnection, MinioConnection
from modules.input.data_collector import DataFetcher
from modules.processing.data_validator import DataValidator
from modules.output.data_writer import DataWriter

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config():
    config_path = Path(__file__).resolve().parent / "config" / "conf.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_validation_report(results: dict):
    # results: {"prices": ValidationResult, ...}
    print("\n" + "=" * 72)
    print("VALIDATION REPORT")
    print("=" * 72)
    for name, res in results.items():
        print(f"\n[{name.upper()}]")
        print(res.summary())
        if res.warnings:
            print("  Warning examples:")
            for w in res.warnings[:5]:
                print(f"   - {w}")
        if res.errors:
            print("  Error examples:")
            for e in res.errors[:5]:
                print(f"   - {e}")
    print("\n" + "=" * 72 + "\n")


def main():
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    # ---- Connections -------------------------------------------------
    pg_cfg = cfg["postgres"]
    mongo_cfg = cfg["mongo"]
    minio_cfg = cfg["minio"]

    pg = PostgresConnection(
        host=pg_cfg["host"],
        port=pg_cfg["port"],
        database=pg_cfg["database"],
        user=pg_cfg["user"],
        password=pg_cfg["password"],
    )
    mongo = MongoConnection(host=mongo_cfg["host"], port=mongo_cfg["port"])
    minio = MinioConnection(
        host=minio_cfg["host"],
        access_key=minio_cfg["access_key"],
        secret_key=minio_cfg["secret_key"],
        secure=minio_cfg.get("secure", False),
    )

    # Optional quick connectivity checks
    if not pg.test_connection():
        raise RuntimeError("PostgreSQL connection failed.")
    if not mongo.test_connection():
        raise RuntimeError("MongoDB connection failed.")
    if not minio.test_connection():
        raise RuntimeError("MinIO connection failed.")
    
    pg.execute_sql_file("sql/create_schema.sql")

    # ---- Load universe ----------------------------------------------
    # Pull symbols + countries from the existing company universe table
    universe = pg.get_company_list()
    if universe is None or universe.empty:
        raise RuntimeError("company_static is empty or not found.")

    # Heuristic column names: adapt if your company_static differs
    # Common variants: symbol / ticker / country / country_code
    symbol_col = "symbol" if "symbol" in universe.columns else "ticker"
    country_col = (
        "country" if "country" in universe.columns
        else "country_code" if "country_code" in universe.columns
        else None
    )

    symbols = (
        universe[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    countries = (
        universe[country_col].dropna().astype(str).str.strip().unique().tolist()
        if country_col else []
    )

    logger.info("Universe loaded: %d symbols | %d countries", len(symbols), len(countries))

    # ---- Fetch -------------------------------------------------------
    fetcher = DataFetcher(minio)

    prices_df = fetcher.fetch_prices(symbols, period=cfg.get("data", {}).get("price_period", "5y"))
    fin_df = fetcher.fetch_fundamentals(symbols, max_workers=cfg.get("data", {}).get("fundamentals_workers", 10))
    rates_df = fetcher.fetch_risk_free_rates(countries)

    # ---- Optional: audit to Mongo (raw-ish) --------------------------
    # This is not truly "raw API response" because you log the processed DataFrames,
    # but it still gives you an audit trail per symbol.
    writer = DataWriter(pg_conn=pg, mongo_conn=mongo, fetcher=fetcher)
    writer.log_batch_to_mongo("prices", prices_df)
    writer.log_batch_to_mongo("fundamentals", fin_df)
    if rates_df is not None and not rates_df.empty:
        writer.log_fetch_to_mongo("rates", "all", rates_df)

    # ---- Validate ----------------------------------------------------
    vcfg = cfg.get("validation", {})
    validator = DataValidator(
        min_price_rows=vcfg.get("min_price_rows", 200),
        min_years=vcfg.get("min_years", 4),
        max_null_pct=vcfg.get("max_null_pct", 0.5),
    )

    results = validator.validate_all(
        prices_df,
        fin_df,
        rates_df,
        expected_symbols=symbols,
        expected_countries=countries,
    )
    print_validation_report(results)

    # Decide whether to proceed
    # Strict mode: if any dataset failed, halt (recommended)
    strict = cfg.get("validation", {}).get("strict", True)
    all_passed = all(r.passed for r in results.values())

    if strict and not all_passed:
        logger.error("Pipeline halted: validation failures in strict mode.")
        return

    # ---- Load into Postgres -----------------------------------------
    prices_written = writer.write_prices(prices_df)
    fin_written = writer.write_financials(fin_df)
    rates_written = writer.write_risk_free_rates(rates_df)

    # ---- Summary -----------------------------------------------------
    counts = writer.get_table_counts()
    print("\n" + "=" * 72)
    print("PIPELINE OUTCOME")
    print("=" * 72)
    print(f"Prices written:        {prices_written}")
    print(f"Financials written:    {fin_written}")
    print(f"Risk-free rates written:{rates_written}")
    print("\nCurrent table counts:")
    for k, v in counts.items():
        print(f" - {k}: {v}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()