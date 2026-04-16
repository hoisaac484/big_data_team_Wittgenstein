
import logging

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from modules.db.db_connection import (
    PostgresConnection
)
from modules.liquidity.liquidity_filter import LiquidityConfig, run_liquidity_filter
from modules.output.data_writer import DataWriter
from modules.zscore.ratios import compute_factor_scores, orthogonalise_lowvol
from modules.zscore.winsorise import winsorise_metrics
from modules.zscore.zscore import calculate_ratios

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    cfg: dict
    pg: PostgresConnection
    writer: DataWriter
    symbols: list
    countries: list
    sector_map: dict
    strict: bool



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


def _load_universe(pg, cfg) -> tuple[list, list]:
    """Load universe as symbols present in both price_data and financial_data.

    Applies in order:
      1. Intersection    — symbols with both price and fundamental data in DB
      2. Ticker normalise — replace dots with dashes (BF.B → BF-B)
      3. Dev mode cap    — limit to cfg['dev']['max_symbols'] when enabled
    """
    result = pg.read_query(
        """
        SELECT DISTINCT p.symbol
        FROM team_wittgenstein.price_data p
        INNER JOIN team_wittgenstein.financial_data f
            ON p.symbol = f.symbol
        ORDER BY p.symbol
        """
    )
    if result is None or result.empty:
        raise RuntimeError("No symbols found with both price and fundamental data.")

    symbols = result["symbol"].dropna().astype(str).str.strip().tolist()
    logger.info("Universe: %d symbols with both price and fundamental data", len(symbols))

    # Countries for risk-free rate selection — still pulled from company_static
    universe = pg.get_company_list()
    country_col = (
        "country" if universe is not None and "country" in universe.columns
        else "country_code" if universe is not None and "country_code" in universe.columns
        else None
    )
    countries = (
        universe[country_col].dropna().astype(str).str.strip().unique().tolist()
        if country_col else []
    )

    # 2. Normalise dot-delimited tickers (BF.B → BF-B)
    symbols = [s.replace(".", "-") for s in symbols]

    # 3. Dev mode cap
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

    pg_cfg    = cfg["postgres"]

    pg = PostgresConnection(
        host=pg_cfg["host"],
        port=pg_cfg["port"],
        database=pg_cfg["database"],
        user=pg_cfg["user"],
        password=pg_cfg["password"],
    )

    if not pg.test_connection():
        raise RuntimeError("PostgreSQL connection failed.")
    
    pg.execute_sql_file("sql/create_cw2_tables.sql")

    symbols, countries = _load_universe(pg, cfg)

    universe = pg.get_company_list()
    sector_map = (
        universe
        .assign(symbol=universe["symbol"].astype(str).str.strip().str.replace(".", "-", regex=False))
        .dropna(subset=["gics_sector"])
        .set_index("symbol")["gics_sector"]
        .to_dict()
    )
    logger.info("Sector map loaded: %d symbols across %d sectors",
                len(sector_map), len(set(sector_map.values())))

    return PipelineContext(
        cfg=cfg,
        pg=pg,
        writer=DataWriter(pg),
        symbols=symbols,
        countries=countries,
        sector_map=sector_map,
        strict=cfg.get("strict", False),
    )


def backfill_factor_metrics(ctx: PipelineContext, years: int = 5) -> None:
    """
    Calculate and persist factor ratios for every month-end rebalancing date
    over the last `years` years, carrying forward the last known quarterly
    fundamentals for each date.

    Rows are written with ON CONFLICT DO NOTHING — re-running is safe and
    will skip any dates already present in factor_metrics.
    """
    end   = pd.Timestamp.today() - pd.offsets.BMonthEnd(1)
    start = end - pd.DateOffset(years=years)
    rebalance_dates = pd.date_range(start=start, end=end, freq=pd.offsets.BMonthEnd())

    logger.info(
        "Backfill: %d rebalancing dates from %s to %s across %d symbols",
        len(rebalance_dates),
        rebalance_dates[0].date(),
        rebalance_dates[-1].date(),
        len(ctx.symbols),
    )

    liq_cfg_raw = ctx.cfg.get("liquidity", {})
    liq_config = LiquidityConfig(
        adtv_lookback_days=liq_cfg_raw.get("adtv_lookback_days", 20),
        illiq_lookback_days=liq_cfg_raw.get("illiq_lookback_days", 21),
        illiq_removal_pct=liq_cfg_raw.get("illiq_removal_pct", 0.10),
        adtv_min_dollar=liq_cfg_raw.get("adtv_min_dollar", 1_000_000),
    )

    all_ratios = []
    for i, ts in enumerate(rebalance_dates, 1):
        rebalance_date = ts.date()
        logger.info("[%d/%d] %s — running liquidity filter", i, len(rebalance_dates), rebalance_date)
        liquid_symbols = run_liquidity_filter(ctx.pg, rebalance_date, liq_config)
        if not liquid_symbols:
            logger.warning("%s | liquidity filter removed all symbols — skipping", rebalance_date)
            continue
        symbols_this_date = [s for s in ctx.symbols if s in set(liquid_symbols)]
        logger.info("%s | %d/%d symbols pass liquidity filter", rebalance_date, len(symbols_this_date), len(ctx.symbols))
        ratios = calculate_ratios(
            pg=ctx.pg,
            rebalance_date=rebalance_date,
            symbols=symbols_this_date,
        )
        all_ratios.append(ratios)

    all_ratios = [r for r in all_ratios if r is not None and not r.empty]
    combined = pd.concat(all_ratios, ignore_index=True)
    # Cast numeric columns explicitly to avoid FutureWarning from all-NA early dates
    numeric_cols = [
        "pb_ratio", "asset_growth", "roe", "leverage", "earnings_stability",
        "momentum_6m", "momentum_12m", "volatility_3m", "volatility_12m",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    logger.info("All dates computed (%d rows). Winsorising...", len(combined))
    combined = winsorise_metrics(combined, ctx.sector_map)
    logger.info("Winsorisation complete. Computing factor scores...")
    combined, zscores = compute_factor_scores(combined, ctx.sector_map)
    logger.info("Factor scores complete. Orthogonalising low vol on momentum...")
    combined = orthogonalise_lowvol(combined)
    logger.info("Orthogonalisation complete. Writing to DB...")
    ctx.writer.write_factor_zscores(zscores)
    ctx.writer.write_factor_scores(combined)
    logger.info("Backfill complete: %d rows across %d dates", len(combined), len(rebalance_dates))


def main(argv=None):
    ctx = build_context()
    backfill_factor_metrics(ctx, years=5)


if __name__ == "__main__":
    main()
