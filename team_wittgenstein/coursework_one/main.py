import argparse
import logging
import signal
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from modules.db.db_connection import PostgresConnection, MongoConnection, MinioConnection
from modules.input.data_collector import DataFetcher
from modules.processing.data_validator import DataValidator
from modules.output.data_writer import DataWriter

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    cfg: dict
    pg: PostgresConnection
    mongo: MongoConnection
    minio: MinioConnection
    fetcher: DataFetcher
    writer: DataWriter
    validator: DataValidator
    symbols: list
    countries: list
    strict: bool


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Wittgenstein data pipeline — fetch prices, fundamentals, and risk-free rates."
    )
    parser.add_argument(
        "--task",
        choices=["all", "prices", "fundamentals"],
        default="all",
        help="Task to run on startup: 'all' (default), 'prices' (prices + risk-free rates), or 'fundamentals'.",
    )
    parser.add_argument(
        "--no-schedule",
        action="store_true",
        help="Run the startup task once then exit without starting the recurring scheduler.",
    )
    parser.add_argument(
        "--run-date",
        metavar="YYYY-MM-DD",
        help="Override today's date for audit and logging purposes (ISO 8601 format).",
    )
    return parser.parse_args(argv)


def load_config():
    config_path = Path(__file__).resolve().parent / "config" / "conf.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _cleanup_removed_symbols(pg, fetcher, universe):
    """Delete managed data/cache for symbols no longer in company_static."""
    symbol_col = "symbol" if "symbol" in universe.columns else "ticker"
    current_symbols = (
        universe[symbol_col].dropna().astype(str).str.strip().unique().tolist()
    )
    # Normalise dot-delimited tickers to dashes to match what is stored in managed
    # tables (e.g. company_static has BF.B but price_data/financial_data store BF-B)
    current_symbols = [s.replace(".", "-") for s in current_symbols]
    removed_symbols = pg.delete_symbols_missing_from_company_list(current_symbols)
    for symbol in removed_symbols:
        fetcher.delete_symbol_cache(symbol)
    if removed_symbols:
        logger.warning(
            "Removed %d stale symbols from managed tables/cache",
            len(removed_symbols),
        )
    return removed_symbols


def print_validation_report(results: dict):
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


def _load_universe(pg, fetcher, cfg) -> tuple:
    """Load and normalise the company universe, run cleanup, return (symbols, countries).

    Called at startup and at the start of every scheduled task so each run
    reflects any company additions or delistings since the last run.
    """
    universe = pg.get_company_list()
    if universe is None or universe.empty:
        raise RuntimeError("company_static is empty or not found.")

    _cleanup_removed_symbols(pg, fetcher, universe)

    symbol_col = "symbol" if "symbol" in universe.columns else "ticker"
    country_col = (
        "country" if "country" in universe.columns
        else "country_code" if "country_code" in universe.columns
        else None
    )

    country_filter = cfg.get("country_filter")
    if country_filter and country_col:
        before = len(universe)
        universe = universe[
            universe[country_col].astype(str).str.strip().str.upper()
            == country_filter.upper()
        ]
        logger.info(
            "Country filter '%s': %d → %d companies",
            country_filter, before, len(universe),
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

    exclude = set(cfg.get("exclude_symbols") or [])
    if exclude:
        before = len(symbols)
        symbols = [s for s in symbols if s not in exclude]
        logger.info(
            "Excluded %d known-bad symbols: %d → %d",
            before - len(symbols), before, len(symbols),
        )

    normalised = []
    for s in symbols:
        if "." in s:
            fixed = s.replace(".", "-")
            logger.info("Ticker normalised: %s → %s", s, fixed)
            normalised.append(fixed)
        else:
            normalised.append(s)
    symbols = normalised

    dev_cfg = cfg.get("dev", {})
    if dev_cfg.get("enabled", False):
        max_sym = dev_cfg.get("max_symbols", 10)
        symbols = symbols[:max_sym]
        logger.warning("DEV MODE: limited to %d symbols", max_sym)

    logger.info("Universe loaded: %d symbols | %d countries", len(symbols), len(countries))
    return symbols, countries


def build_context() -> PipelineContext:
    """Set up all connections and infrastructure. Called once at startup."""
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

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

    if not pg.test_connection():
        raise RuntimeError("PostgreSQL connection failed.")
    if not mongo.test_connection():
        raise RuntimeError("MongoDB connection failed.")
    if not minio.test_connection():
        raise RuntimeError("MinIO connection failed.")

    pg.execute_sql_file("sql/create_schema.sql")

    fetcher = DataFetcher(minio)
    fetcher.cache_ttl_days = cfg.get("data", {}).get("cache_ttl_days")

    vcfg = cfg.get("validation", {})
    validator = DataValidator(
        min_price_rows=vcfg.get("min_price_rows", 200),
        min_years=vcfg.get("min_years", 4),
        max_null_pct=vcfg.get("max_null_pct", 0.5),
    )
    writer = DataWriter(pg_conn=pg, mongo_conn=mongo, fetcher=fetcher)

    symbols, countries = _load_universe(pg, fetcher, cfg)

    return PipelineContext(
        cfg=cfg,
        pg=pg,
        mongo=mongo,
        minio=minio,
        fetcher=fetcher,
        writer=writer,
        validator=validator,
        symbols=symbols,
        countries=countries,
        strict=vcfg.get("strict", True),
    )


def run_prices_and_rates(ctx: PipelineContext):
    """Fetch, validate, and write prices + risk-free rates."""
    logger.info("Task: prices + risk-free rates starting")
    try:
        symbols, countries = _load_universe(ctx.pg, ctx.fetcher, ctx.cfg)

        prices_df = ctx.fetcher.fetch_prices(
            symbols, period=ctx.cfg.get("data", {}).get("price_period", "5y")
        )
        rates_df = ctx.fetcher.fetch_risk_free_rates(countries)

        if ctx.fetcher.price_failures:
            ctx.writer.log_fetch_to_mongo(
                "price_failures", "classification", ctx.fetcher.price_failures
            )
            logger.warning(
                "Prices: %d delisted, %d fetch errors",
                len(ctx.fetcher.price_failures.get("delisted", [])),
                len(ctx.fetcher.price_failures.get("fetch_error", [])),
            )

        prices_df = ctx.validator.clean_prices(prices_df)
        delisted = set(ctx.fetcher.price_failures.get("delisted", []))
        expected = [s for s in symbols if s not in delisted]

        results = {
            "prices": ctx.validator.validate_prices(prices_df, expected),
            "risk_free_rates": ctx.validator.validate_risk_free_rates(rates_df, countries),
        }
        print_validation_report(results)

        if ctx.strict and not all(r.passed for r in results.values()):
            logger.error("prices+rates task halted: validation failures in strict mode.")
            return

        prices_written = ctx.writer.write_prices(prices_df)
        rates_written = ctx.writer.write_risk_free_rates(rates_df)
        logger.info(
            "Task complete: %d price rows, %d rate rows written",
            prices_written, rates_written,
        )
    except Exception:
        logger.exception("prices+rates task failed — scheduler will continue")


def run_fundamentals(ctx: PipelineContext):
    """Fetch, validate, and write fundamentals data."""
    logger.info("Task: fundamentals starting")
    try:
        symbols, _ = _load_universe(ctx.pg, ctx.fetcher, ctx.cfg)

        fin_df = ctx.fetcher.fetch_fundamentals(
            symbols,
            period=ctx.cfg.get("data", {}).get("fundamentals_period", "5y"),
            source=ctx.cfg.get("data", {}).get("fundamentals_source", "simfin"),
        )

        if ctx.fetcher.fundamentals_failures:
            ctx.writer.log_fetch_to_mongo(
                "fundamentals_failures", "classification", ctx.fetcher.fundamentals_failures
            )
            logger.warning(
                "Fundamentals: %d delisted, %d fetch errors",
                len(ctx.fetcher.fundamentals_failures.get("delisted", [])),
                len(ctx.fetcher.fundamentals_failures.get("fetch_error", [])),
            )

        fin_result = ctx.validator.validate_financials(fin_df, symbols)
        results = {"financials": fin_result}
        print_validation_report(results)

        if ctx.strict and not fin_result.passed:
            logger.error("fundamentals task halted: validation failures in strict mode.")
            return

        fin_written = ctx.writer.write_financials(fin_df)
        logger.info("Task complete: %d financials rows written", fin_written)
    except Exception:
        logger.exception("fundamentals task failed — scheduler will continue")


def run_full_pipeline(ctx: PipelineContext):
    """Run prices+rates then fundamentals. Used for the initial startup run."""
    logger.info("Full pipeline starting")
    run_prices_and_rates(ctx)
    run_fundamentals(ctx)
    counts = ctx.writer.get_table_counts()
    logger.info("Full pipeline complete. Table counts: %s", counts)
    print("\n" + "=" * 72)
    print("PIPELINE OUTCOME")
    print("=" * 72)
    print("\nCurrent table counts:")
    for k, v in counts.items():
        print(f" - {k}: {v}")
    print("=" * 72 + "\n")


def main(argv=None):
    args = parse_args(argv)
    ctx = build_context()

    if args.run_date:
        logger.info("Run date override: %s", args.run_date)

    # Run startup task based on --task argument
    if args.task == "prices":
        run_prices_and_rates(ctx)
    elif args.task == "fundamentals":
        run_fundamentals(ctx)
    else:
        run_full_pipeline(ctx)

    if args.no_schedule:
        logger.info("--no-schedule: exiting after startup run.")
        return

    # Schedule recurring tasks
    # max_workers=1 serialises all jobs — if one task is still running when
    # another trigger fires, the new job queues and waits rather than running
    # concurrently. This prevents simultaneous API calls and DB write contention.
    scheduler = BlockingScheduler(
        executors={"default": ThreadPoolExecutor(max_workers=1)},
        timezone="UTC",
    )

    sched_cfg = ctx.cfg.get("scheduler", {})
    pr_cfg = sched_cfg.get("prices_and_rates", {})
    fund_cfg = sched_cfg.get("fundamentals", {})
    pr_trigger = CronTrigger(
        day=pr_cfg.get("day", 1),
        hour=pr_cfg.get("hour", 2),
        minute=pr_cfg.get("minute", 0),
    )
    fund_trigger = CronTrigger(
        month=fund_cfg.get("month", "1,4,7,10"),
        day=fund_cfg.get("day", 1),
        hour=fund_cfg.get("hour", 4),
        minute=fund_cfg.get("minute", 0),
    )

    scheduler.add_job(
        run_prices_and_rates,
        trigger=pr_trigger,
        args=[ctx],
        id="prices_and_rates",
        name="prices + risk-free rates",
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        run_fundamentals,
        trigger=fund_trigger,
        args=[ctx],
        id="fundamentals",
        name="fundamentals",
        misfire_grace_time=3600,
        coalesce=True,
    )

    def _shutdown(signum, frame):
        logger.info("Shutdown signal received, stopping scheduler...")
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(
        "Scheduler started. Jobs: monthly prices+rates (1st @ 02:00 UTC), "
        "quarterly fundamentals (1st Jan/Apr/Jul/Oct @ 04:00 UTC)"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
