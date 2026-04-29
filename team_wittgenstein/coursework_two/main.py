import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from modules.backtest.backtest_engine import BacktestConfig, run_backtest
from modules.backtest.benchmark import backfill_benchmark_returns
from modules.composite.composite_scorer import CompositeConfig, run_composite_scorer
from modules.db.db_connection import PostgresConnection
from modules.evaluation.cost_sensitivity import run_cost_sensitivity
from modules.evaluation.factor_exclusion import (
    FactorExclusionConfig,
    run_factor_exclusion,
)
from modules.evaluation.metrics import compute_summary_metrics
from modules.evaluation.reporting import run_reporting
from modules.evaluation.sensitivity import run_parameter_sensitivity
from modules.liquidity.liquidity_filter import LiquidityConfig, run_liquidity_filter
from modules.output.data_writer import DataWriter
from modules.portfolio.ewma_volatility import EWMAConfig, run_ewma_volatility
from modules.portfolio.position_builder import PositionConfig, build_portfolio_positions
from modules.portfolio.risk_adjusted import compute_risk_adjusted_scores
from modules.portfolio.stock_selector import SelectionConfig, run_stock_selection
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
    result = pg.read_query("""
        SELECT DISTINCT p.symbol
        FROM team_wittgenstein.price_data p
        INNER JOIN team_wittgenstein.financial_data f
            ON p.symbol = f.symbol
        ORDER BY p.symbol
        """)
    if result is None or result.empty:
        raise RuntimeError("No symbols found with both price and fundamental data.")

    symbols = result["symbol"].dropna().astype(str).str.strip().tolist()
    logger.info(
        "Universe: %d symbols with both price and fundamental data", len(symbols)
    )

    # Countries for risk-free rate selection — still pulled from company_static
    universe = pg.get_company_list()
    country_col = (
        "country"
        if universe is not None and "country" in universe.columns
        else (
            "country_code"
            if universe is not None and "country_code" in universe.columns
            else None
        )
    )
    countries = (
        universe[country_col].dropna().astype(str).str.strip().unique().tolist()
        if country_col
        else []
    )

    # 2. Normalise dot-delimited tickers (BF.B → BF-B)
    symbols = [s.replace(".", "-") for s in symbols]

    # 3. Dev mode cap
    dev_cfg = cfg.get("dev", {})
    if dev_cfg.get("enabled", False):
        max_sym = dev_cfg.get("max_symbols", 10)
        symbols = symbols[:max_sym]
        logger.warning("DEV MODE: limited to %d symbols", max_sym)

    logger.info(
        "Universe loaded: %d symbols | %d countries", len(symbols), len(countries)
    )
    return symbols, countries


def init_schema(pg: PostgresConnection) -> None:
    """Drop and recreate all CW2 tables from sql/create_cw2_tables.sql.

    Destructive: wipes backtest_returns, backtest_summary, factor_scores,
    portfolio_positions, etc. Only call when starting a fresh full pipeline
    run via main(). Do NOT call from build_context() or downstream helpers.
    """
    pg.execute_sql_file("sql/create_cw2_tables.sql")


def build_context() -> PipelineContext:
    """Set up all connections and infrastructure. Called once at startup.

    Does NOT create or drop tables - schema setup is done explicitly via
    init_schema() at the start of main(). This keeps build_context() safe
    to call from any sub-script (e.g. cost sensitivity, reporting) without
    destroying baseline data.
    """
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    pg_cfg = cfg["postgres"]

    import os

    pg = PostgresConnection(
        host=os.getenv("DB_HOST", pg_cfg["host"]),
        port=int(os.getenv("DB_PORT", pg_cfg["port"])),
        database=os.getenv("DB_NAME", pg_cfg["database"]),
        user=os.getenv("DB_USER", pg_cfg["user"]),
        password=os.getenv("DB_PASSWORD", pg_cfg["password"]),
    )

    if not pg.test_connection():
        raise RuntimeError("PostgreSQL connection failed.")

    symbols, countries = _load_universe(pg, cfg)

    universe = pg.get_company_list()
    sector_map = (
        universe.assign(
            symbol=universe["symbol"]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
        )
        .dropna(subset=["gics_sector"])
        .set_index("symbol")["gics_sector"]
        .to_dict()
    )
    logger.info(
        "Sector map loaded: %d symbols across %d sectors",
        len(sector_map),
        len(set(sector_map.values())),
    )

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
    end = pd.Timestamp.today() - pd.offsets.BMonthEnd(1)
    rebalance_dates = pd.date_range(
        end=end, periods=years * 12 + 2, freq=pd.offsets.BMonthEnd()
    )

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
        logger.info(
            "[%d/%d] %s — running liquidity filter",
            i,
            len(rebalance_dates),
            rebalance_date,
        )
        liquid_symbols = run_liquidity_filter(ctx.pg, rebalance_date, liq_config)
        if not liquid_symbols:
            logger.warning(
                "%s | liquidity filter removed all symbols — skipping", rebalance_date
            )
            continue
        symbols_this_date = [s for s in ctx.symbols if s in set(liquid_symbols)]
        logger.info(
            "%s | %d/%d symbols pass liquidity filter",
            rebalance_date,
            len(symbols_this_date),
            len(ctx.symbols),
        )
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
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    logger.info("All dates computed (%d rows). Writing raw metrics...", len(combined))
    ctx.writer.write_factor_metrics(combined)
    logger.info("Winsorising...")
    combined = winsorise_metrics(combined, ctx.sector_map)
    logger.info("Winsorisation complete. Computing factor scores...")
    combined, zscores = compute_factor_scores(combined, ctx.sector_map)
    logger.info("Factor scores complete. Orthogonalising low vol on momentum...")
    combined = orthogonalise_lowvol(combined)
    logger.info("Orthogonalisation complete. Writing to DB...")
    ctx.writer.write_factor_zscores(zscores)
    ctx.writer.write_factor_scores(combined)
    logger.info(
        "Backfill complete: %d rows across %d dates",
        len(combined),
        len(rebalance_dates),
    )


def backfill_composite_scores(ctx: PipelineContext, years: int = 5) -> None:
    """Compute IC-weighted composite scores for every rebalancing date.

    Must run after backfill_factor_metrics so that factor_scores is populated.
    For each date, computes trailing 36-month IC weights and writes composite
    scores back to factor_scores + persists IC weights to ic_weights table.
    """
    end = pd.Timestamp.today() - pd.offsets.BMonthEnd(1)
    rebalance_dates = pd.date_range(
        end=end, periods=years * 12 + 2, freq=pd.offsets.BMonthEnd()
    )

    comp_cfg_raw = ctx.cfg.get("composite", {})
    comp_config = CompositeConfig(
        ic_lookback_months=comp_cfg_raw.get("ic_lookback_months", 36),
    )

    for i, ts in enumerate(rebalance_dates, 1):
        rebalance_date = ts.date()
        logger.info(
            "[%d/%d] %s — computing composite scores",
            i,
            len(rebalance_dates),
            rebalance_date,
        )
        result = run_composite_scorer(ctx.pg, rebalance_date, comp_config)
        logger.info(
            "%s | composite scores: %d stocks",
            rebalance_date,
            len(result),
        )

    logger.info("Composite score backfill complete.")


def backfill_portfolio_positions(ctx: PipelineContext, years: int = 5) -> None:
    """Run Steps 1-7 of portfolio construction for every rebalancing date.

    Must run after backfill_composite_scores so that factor_scores is
    populated with composite_score values.

    For each date:
      1. Stock selection with buffer zone (stock_selector)
      2. EWMA volatility (ewma_volatility)
      3. Risk-adjusted scores (risk_adjusted)
      4-7. Sector weights, liquidity cap, no-trade zone, constraint check
           (position_builder) → written to portfolio_positions
    """
    end = pd.Timestamp.today() - pd.offsets.BMonthEnd(1)
    rebalance_dates = pd.date_range(
        end=end, periods=years * 12 + 2, freq=pd.offsets.BMonthEnd()
    )

    port_cfg_raw = ctx.cfg.get("portfolio", {})

    sel_config = SelectionConfig(
        selection_threshold=port_cfg_raw.get("selection_threshold", 0.10),
        buffer_exit_threshold=port_cfg_raw.get("buffer_exit_threshold", 0.20),
        buffer_max_months=port_cfg_raw.get("buffer_max_months", 3),
    )
    ewma_config = EWMAConfig(
        ewma_lambda=port_cfg_raw.get("ewma_lambda", 0.94),
        lookback_days=port_cfg_raw.get("ewma_lookback_days", 252),
    )
    pos_config = PositionConfig(
        aum=port_cfg_raw.get("aum", 1_000_000_000),
        liquidity_cap_pct=port_cfg_raw.get("liquidity_cap_pct", 0.05),
        no_trade_threshold=port_cfg_raw.get("no_trade_threshold", 0.005),
        adv_lookback_days=port_cfg_raw.get("adv_lookback_days", 20),
        constraint_tolerance=port_cfg_raw.get("constraint_tolerance", 0.02),
    )

    logger.info(
        "Portfolio backfill: %d dates from %s to %s",
        len(rebalance_dates),
        rebalance_dates[0].date(),
        rebalance_dates[-1].date(),
    )

    for i, ts in enumerate(rebalance_dates, 1):
        rebalance_date = ts.date()
        logger.info(
            "[%d/%d] %s — portfolio construction",
            i,
            len(rebalance_dates),
            rebalance_date,
        )

        # Step 1: stock selection with buffer zone
        selected = run_stock_selection(
            ctx.pg, rebalance_date, ctx.sector_map, sel_config
        )
        if selected.empty:
            logger.warning("%s | no stocks selected — skipping", rebalance_date)
            continue

        # Step 2: EWMA volatility
        symbols_selected = selected["symbol"].tolist()
        ewma_vols = run_ewma_volatility(
            ctx.pg, symbols_selected, rebalance_date, ewma_config
        )

        # Step 3: risk-adjusted scores
        scored = compute_risk_adjusted_scores(selected, ewma_vols)
        if scored.empty:
            logger.warning("%s | no risk-adjusted scores — skipping", rebalance_date)
            continue

        # Steps 4-7: weights, liquidity cap, no-trade zone, constraints
        positions = build_portfolio_positions(
            ctx.pg, scored, rebalance_date, pos_config
        )
        if positions.empty:
            continue

        ctx.writer.write_portfolio_positions(positions)
        logger.info(
            "%s | wrote %d positions (%d long, %d short)",
            rebalance_date,
            len(positions),
            (positions["direction"] == "long").sum(),
            (positions["direction"] == "short").sum(),
        )

    logger.info("Portfolio construction backfill complete.")


def run_baseline_backtest(ctx: PipelineContext) -> None:
    """Steps 1-6: compute monthly returns and write to backtest_returns.

    Caches benchmark returns in the DB first so subsequent scenario runs
    don't need to hit yfinance.
    """
    # Derive benchmark date range from portfolio positions
    date_range = ctx.pg.read_query("""
        SELECT MIN(rebalance_date) AS min_date, MAX(rebalance_date) AS max_date
        FROM team_wittgenstein.portfolio_positions
        """)
    if date_range.empty or pd.isna(date_range["min_date"].iloc[0]):
        raise RuntimeError("No portfolio positions found — run backfill first.")

    bench_start = pd.to_datetime(date_range["min_date"].iloc[0]).date()
    bench_end = pd.to_datetime(date_range["max_date"].iloc[0]).date()

    # Cache benchmark returns to DB (idempotent via ON CONFLICT DO NOTHING)
    backfill_benchmark_returns(ctx.pg, bench_start, bench_end)

    config = BacktestConfig(
        cost_bps=ctx.cfg.get("backtest", {}).get("cost_bps", 25.0),
        borrow_rate=ctx.cfg.get("backtest", {}).get("borrow_rate", 0.0075),
        scenario_id="baseline",
    )
    df = run_backtest(ctx.pg, config)
    ctx.writer.write_backtest_returns(df, config.scenario_id)
    logger.info(
        "Backtest written: %d months | cumulative net return: %.4f",
        len(df),
        df["cumulative_return"].iloc[-1] if not df.empty else float("nan"),
    )


def run_baseline_summary(ctx: PipelineContext) -> None:
    """Step 7: compute summary metrics for the baseline scenario."""
    risk_free_rate = ctx.cfg.get("backtest", {}).get("risk_free_rate")
    summary = compute_summary_metrics(
        ctx.pg, scenario_id="baseline", risk_free_rate=risk_free_rate
    )
    logger.info(
        "Summary metrics | sharpe=%.3f sortino=%.3f calmar=%.3f IR=%.3f",
        summary["sharpe_ratio"],
        summary["sortino_ratio"],
        summary["calmar_ratio"],
        summary["information_ratio"],
    )


def run_cost_sensitivity_scenarios(ctx: PipelineContext) -> None:
    """Step 9: re-apply alternative cost assumptions to the baseline positions.

    Creates 3 new scenarios (frictionless, low, high) and populates their
    backtest_returns and backtest_summary rows. Baseline = moderate cost
    is already in the DB.
    """
    created = run_cost_sensitivity(ctx.pg)
    logger.info("Cost sensitivity complete: %d scenarios (%s)", len(created), created)


def run_factor_exclusion_scenarios(ctx: PipelineContext) -> None:
    """Step 10: run the full pipeline with each factor excluded one at a time.

    Uses the in-memory pipeline so baseline DB tables are untouched. Each of
    the 4 scenarios produces backtest_returns + backtest_summary rows with
    scenario_id 'excl_<factor>'.
    """
    port_cfg_raw = ctx.cfg.get("portfolio", {})
    bt_cfg_raw = ctx.cfg.get("backtest", {})
    composite_cfg_raw = ctx.cfg.get("composite", {})

    config = FactorExclusionConfig(
        composite=CompositeConfig(
            ic_lookback_months=composite_cfg_raw.get("ic_lookback_months", 36),
        ),
        selection=SelectionConfig(
            selection_threshold=port_cfg_raw.get("selection_threshold", 0.10),
            buffer_exit_threshold=port_cfg_raw.get("buffer_exit_threshold", 0.20),
            buffer_max_months=port_cfg_raw.get("buffer_max_months", 3),
        ),
        ewma=EWMAConfig(
            ewma_lambda=port_cfg_raw.get("ewma_lambda", 0.94),
            lookback_days=port_cfg_raw.get("ewma_lookback_days", 252),
        ),
        position=PositionConfig(
            aum=port_cfg_raw.get("aum", 1_000_000_000),
            liquidity_cap_pct=port_cfg_raw.get("liquidity_cap_pct", 0.05),
            no_trade_threshold=port_cfg_raw.get("no_trade_threshold", 0.005),
            adv_lookback_days=port_cfg_raw.get("adv_lookback_days", 20),
            constraint_tolerance=port_cfg_raw.get("constraint_tolerance", 0.02),
        ),
        backtest=BacktestConfig(
            cost_bps=bt_cfg_raw.get("cost_bps", 25.0),
            borrow_rate=bt_cfg_raw.get("borrow_rate", 0.0075),
        ),
    )
    created = run_factor_exclusion(ctx.pg, ctx.sector_map, config)
    logger.info("Factor exclusion complete: %d scenarios (%s)", len(created), created)


def run_parameter_sensitivity_scenarios(ctx: PipelineContext) -> None:
    """Step 8: vary one strategy parameter at a time, holding others at baseline.

    Produces 15 scenarios in backtest_returns / backtest_summary. Uses the
    same in-memory pipeline as factor exclusion - baseline DB tables remain
    untouched.
    """
    port_cfg_raw = ctx.cfg.get("portfolio", {})
    bt_cfg_raw = ctx.cfg.get("backtest", {})
    composite_cfg_raw = ctx.cfg.get("composite", {})

    base_config = FactorExclusionConfig(
        composite=CompositeConfig(
            ic_lookback_months=composite_cfg_raw.get("ic_lookback_months", 36),
        ),
        selection=SelectionConfig(
            selection_threshold=port_cfg_raw.get("selection_threshold", 0.10),
            buffer_exit_threshold=port_cfg_raw.get("buffer_exit_threshold", 0.20),
            buffer_max_months=port_cfg_raw.get("buffer_max_months", 3),
        ),
        ewma=EWMAConfig(
            ewma_lambda=port_cfg_raw.get("ewma_lambda", 0.94),
            lookback_days=port_cfg_raw.get("ewma_lookback_days", 252),
        ),
        position=PositionConfig(
            aum=port_cfg_raw.get("aum", 1_000_000_000),
            liquidity_cap_pct=port_cfg_raw.get("liquidity_cap_pct", 0.05),
            no_trade_threshold=port_cfg_raw.get("no_trade_threshold", 0.005),
            adv_lookback_days=port_cfg_raw.get("adv_lookback_days", 20),
            constraint_tolerance=port_cfg_raw.get("constraint_tolerance", 0.02),
        ),
        backtest=BacktestConfig(
            cost_bps=bt_cfg_raw.get("cost_bps", 25.0),
            borrow_rate=bt_cfg_raw.get("borrow_rate", 0.0075),
        ),
    )
    created = run_parameter_sensitivity(ctx.pg, ctx.sector_map, base_config)
    logger.info(
        "Parameter sensitivity complete: %d scenarios (%s)",
        len(created),
        created,
    )


def run_reporting_outputs(ctx: PipelineContext) -> None:
    """Step 11: produce charts (PNG) and tables (CSV) for the report."""
    output_dir = ctx.cfg.get("reporting", {}).get("output_dir", "reports")
    run_reporting(ctx.pg, output_dir=output_dir)


def main(argv=None):
    ctx = build_context()
    # Recreate the schema for a fresh full pipeline run. Sub-scripts that
    # consume existing data (cost sensitivity, reporting, etc.) call
    # build_context() directly without invoking init_schema.
    init_schema(ctx.pg)
    backfill_factor_metrics(ctx, years=9)
    backfill_composite_scores(ctx, years=9)
    backfill_portfolio_positions(ctx, years=5)
    run_baseline_backtest(ctx)
    run_baseline_summary(ctx)
    run_cost_sensitivity_scenarios(ctx)
    run_factor_exclusion_scenarios(ctx)
    run_parameter_sensitivity_scenarios(ctx)
    run_reporting_outputs(ctx)


if __name__ == "__main__":
    main()
