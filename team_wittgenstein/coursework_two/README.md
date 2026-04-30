# Coursework Two — 130/30 Multi-Factor Strategy

Backtesting engine and interactive dashboard for a sector-neutral 130/30 long-short multi-factor equity strategy. Built for Big Data in Quantitative Finance (UCL IFT, 2025-26).

Reads the raw price, financial, and risk-free-rate data populated by [Coursework One](../coursework_one/README.md), runs the strategy through 23 backtest scenarios (1 baseline + 3 cost + 4 factor exclusion + 15 parameter sensitivity), and exposes the results in a Streamlit dashboard.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- Docker and Docker Compose for running PostgreSQL, MongoDB, MinIO, and the dashboard

There are two ways to run this:

- **Quick Start** (~30 seconds) - load the pre-built database dump and launch the dashboard. Best for graders who just want to view results. **Does not require CW1.**
- **Full Pipeline** (~2-3 hours) - run CW1 + CW2 from scratch, regenerate everything. Best if you want to verify the code reproduces the results. **CW1 must run first** to populate price, financial, and risk-free-rate tables. See the [CW1 README](../coursework_one/README.md).

## Quick Start (recommended for first look)

Loads the committed seed file into Postgres and launches the dashboard. No CW1 / CW2 pipeline needed.

```bash
# 1 — start platform Postgres
cd /path/to/repo
docker compose up -d

# 2 — load seed into platform Postgres
gunzip -c team_wittgenstein/coursework_two/docker/seed/seed.sql.gz \
  | docker exec -i postgres_db_cw psql -U postgres -d fift

# 3 — install Python deps and launch the dashboard
cd team_wittgenstein/coursework_two
poetry install
poetry run streamlit run dashboard/Home.py
# Open http://localhost:8501

# 4 — teardown (when done)
cd /path/to/repo
docker compose down
```

The seed file (`docker/seed/seed.sql.gz`) is a snapshot of the `team_wittgenstein` schema after a full pipeline run. It contains all 23 backtest scenarios plus baseline factor scores, IC weights, and portfolio positions.

**Note on data freshness:** the seed is a **frozen snapshot** taken when it was generated. The dashboard's latest rebalance date is whatever was in the database at that moment - it does not advance with calendar time. To see fresher data, run the **Full Pipeline** below, which fetches live data from the APIs.

## Full Pipeline (run from scratch)

Use this if you want to reproduce the data yourself instead of trusting the seed.

```bash
# 1 — start platform Postgres
cd /path/to/repo
docker compose up -d

# 2 — run CW1 (data ingestion, ~30-60 min)
cd team_wittgenstein/coursework_one
poetry install
poetry run python main.py

# 3 — run CW2 (factor scoring + 23 scenarios, ~1.5-2.5 hours)
cd ../coursework_two
poetry install
poetry run python main.py

# 4 — launch the dashboard
poetry run streamlit run dashboard/Home.py
# Open http://localhost:8501

# 5 — teardown (when done)
cd /path/to/repo
docker compose down
```

## Regenerating the seed

If you re-run the full pipeline and want to update the committed seed (e.g. after fixing strategy logic), dump the schema and gzip it:

```bash
docker exec postgres_db_cw pg_dump -U postgres -d fift \
  --schema=team_wittgenstein --no-owner --no-acl \
  | gzip > team_wittgenstein/coursework_two/docker/seed/seed.sql.gz
```

Run this from the **repo root**. The resulting file is ~43 MB.

## Project structure

```
coursework_two/
├── main.py                 # Pipeline orchestrator
├── config/
│   └── conf.yaml           # All strategy and backtest parameters
├── dashboard/              # Streamlit UI (8 pages)
│   ├── Home.py
│   ├── pages/              # Performance, Compare Scenarios, etc.
│   └── lib/                # Reusable charts, queries, theme
├── modules/
│   ├── liquidity/          # ADV + Amihud ILLIQ filter
│   ├── zscore/             # Factor metric calculation, winsorisation, z-scores
│   ├── composite/          # IC-weighted composite scoring
│   ├── portfolio/          # Stock selection, EWMA vol, weight construction
│   ├── backtest/           # Walk-forward backtest + benchmark
│   ├── evaluation/         # Metrics, cost/factor/parameter sensitivity, reporting
│   ├── output/             # PostgreSQL data writer
│   └── db/                 # Database connection wrapper
├── sql/
│   └── create_cw2_tables.sql   # Strategy tables DDL
├── tests/                  # Unit tests, mirroring the modules/ + dashboard/ layout
│   ├── backtest/           # Tests for modules/backtest/
│   ├── composite/          # Tests for modules/composite/
│   ├── db/                 # Tests for modules/db/
│   ├── evaluation/         # Tests for modules/evaluation/ (5 files)
│   ├── liquidity/          # Tests for modules/liquidity/
│   ├── output/             # Tests for modules/output/
│   ├── portfolio/          # Tests for modules/portfolio/
│   ├── zscore/             # Tests for modules/zscore/
│   └── dashboard/          # Tests for dashboard/lib/
└── reports/                # Generated charts and CSVs (gitignored)
```

## Configuration

All strategy parameters live in `config/conf.yaml`.

| Group | Setting | Default | Description |
|-------|---------|---------|-------------|
| `liquidity` | `adtv_min_dollar` | 1,000,000 | Minimum 20-day average daily trading volume in USD |
| `liquidity` | `illiq_removal_pct` | 0.10 | Fraction of stocks dropped by Amihud ILLIQ rank (most illiquid) |
| `composite` | `ic_lookback_months` | 36 | Rolling window for IC weight calculation |
| `portfolio` | `selection_threshold` | 0.10 | Top/bottom percentile per sector for long/short |
| `portfolio` | `buffer_exit_threshold` | 0.20 | Outer band before forcing a stock out of the basket |
| `portfolio` | `buffer_max_months` | 3 | Max months a stock can sit in the buffer zone |
| `portfolio` | `ewma_lambda` | 0.94 | RiskMetrics decay for volatility |
| `portfolio` | `liquidity_cap_pct` | 0.05 | Max position size relative to stock's 20-day ADTV |
| `portfolio` | `no_trade_threshold` | 0.01 | Minimum target weight change to trigger a trade |
| `portfolio` | `aum` | 50,000,000 | AUM used for liquidity-cap dollar sizing |
| `backtest` | `cost_bps` | 25 | One-way transaction cost (bps) for net returns |
| `backtest` | `borrow_rate` | 0.0075 | Annual short-borrow rate |

## Pipeline stages

1. **Liquidity filter** — Computes 20-day average daily trading volume (ADTV) and 21-day Amihud ILLIQ for every stock. Stocks with ADTV below the dollar threshold or in the worst 10% of ILLIQ are dropped from the universe each rebalance.
2. **Factor scoring** — For each stock, computes raw inputs for the four factors (Value: P/B + asset growth; Quality: ROE + leverage + earnings stability; Momentum: 6m + 12m; Low Vol: 3m + 12m). Each metric is winsorised at the 5th/95th percentile within sector, standardised to z-scores, and Low Vol is orthogonalised against Momentum.
3. **Composite score** — Combines the four factor z-scores using rolling 36-month Spearman ICs. Negative ICs are floored to zero so counter-predictive factors get zero weight rather than inverted.
4. **Stock selection** — Within each of 11 GICS sectors, the top 10% (by composite) enter the long basket and the bottom 10% enter the short basket. A buffer zone (10-20%) holds existing positions for up to 3 months before forcing exit.
5. **Risk-adjusted weighting** — Each selected stock gets a weight proportional to `composite / EWMA volatility`. Sector budgets are 130%/11 long and 30%/11 short, ensuring sector neutrality.
6. **Liquidity cap** — Caps each position at 5% of the stock's 20-day ADTV. Excess weight is redistributed pro-rata within the same sector.
7. **No-trade zone** — If the target weight is within ±1% of the current weight, the position is held instead of traded. Reduces transaction costs.
8. **Walk-forward backtest** — Steps month by month using only data available at each rebalance date. Computes gross returns, transaction costs, short-borrow charges, and net returns.
9. **Scenario sensitivity** — Runs 22 variant scenarios:
   - **Cost** — frictionless / low (10 bps) / high (50 bps)
   - **Factor exclusion** — drops one factor at a time (Value / Quality / Momentum / Low Vol)
   - **Parameter sensitivity** — 15 variants across selection threshold, IC lookback, EWMA λ, no-trade threshold, and buffer exit
10. **Reporting** — Generates 10 charts (PNG) and 4 CSV summary tables in `reports/`.

## Backtest scenarios

Re-running `main.py` produces 23 scenarios in `backtest_summary`:

| Group | Count | scenario_id examples |
|-------|-------|----------------------|
| Baseline | 1 | `baseline` |
| Cost sensitivity | 3 | `cost_frictionless`, `cost_low`, `cost_high` |
| Factor exclusion | 4 | `excl_value`, `excl_quality`, `excl_momentum`, `excl_low_vol` |
| Parameter sensitivity | 15 | `sens_sel_0.05`, `sens_ic_24`, `sens_ewma_0.97`, ... |

The strategy itself only writes intermediate data (`portfolio_positions`, `factor_scores`, `selection_status`, etc.) for the **baseline** scenario. Variants compute positions in memory and only persist `backtest_returns` and `backtest_summary` rows. This keeps the database compact and keeps the schema unambiguous.

## Dashboard

After the pipeline finishes, the dashboard reads from PostgreSQL and presents the results across 8 pages.

```bash
poetry run streamlit run dashboard/Home.py
```

| Page | Purpose |
|------|---------|
| Home | Strategy overview, pipeline diagram, system health, parameters table |
| Performance | Deep-dive metrics for any scenario - equity curve, drawdown, returns, rolling Sharpe, turnover |
| Compare Scenarios | Side-by-side comparison of any two scenarios |
| Strategy Tuner | Slider-driven parameter exploration, single-parameter changes |
| Portfolio Composition | Holdings, sector breakdown, constraint health, click-through to deep-dive |
| Stock Deep-Dive | Per-stock factor history, fundamentals, position record |
| Factor Analysis | IC weight evolution, composite distribution, sector z-scores, factor correlations |
| Methodology | Formulas, schema, references, known limitations |

## Testing

Run all tests:

```bash
poetry run pytest
```

With HTML coverage report (opens in browser):

```bash
poetry run pytest --cov --cov-report=html && open htmlcov/index.html
```

Run a single subsystem:

```bash
poetry run pytest tests/portfolio/        # all portfolio tests
poetry run pytest tests/evaluation/       # all evaluation tests (metrics, sensitivity, etc.)
poetry run pytest tests/dashboard/        # all dashboard helper tests
```

The suite has 529 tests covering the strategy pipeline (`modules/`) and the dashboard helpers (`dashboard/lib/`). Total coverage is 98%.

## Documentation

Build the Sphinx documentation from the `coursework_two` directory:

```bash
poetry lock
poetry install
cd docs
make html
open build/index.html
```

The docs source lives under `docs/source/` and covers:

- pipeline and runtime flow
- dashboard structure and helper APIs
- database schema overview
- Python API documentation for `modules/*` and `dashboard/lib/*`

## Linting and formatting

```bash
poetry run black dashboard/ modules/ tests/ main.py        # Code formatting
poetry run isort dashboard/ modules/ tests/ main.py        # Import sorting
poetry run flake8 dashboard/ modules/ tests/ main.py       # Style checks
poetry run bandit -r dashboard/ modules/ main.py           # Security linting
```

## Running individual stages

`main.py` is composed of independent stages that you can run on their own. Useful if you only want to refresh one piece (e.g. just cost sensitivity) without re-running the slow factor scoring.

Every command below assumes:

```bash
cd team_wittgenstein/coursework_two
```

### Schema only (drop and recreate CW2 tables)

```bash
poetry run python -c "
from main import build_context, init_schema
ctx = build_context()
init_schema(ctx.pg)
"
```

### Factor metrics + composite scores + portfolio positions (~30 min)

```bash
poetry run python -c "
from main import build_context, backfill_factor_metrics, backfill_composite_scores, backfill_portfolio_positions
ctx = build_context()
backfill_factor_metrics(ctx, years=5)
backfill_composite_scores(ctx, years=5)
backfill_portfolio_positions(ctx, years=5)
"
```

### Baseline backtest only (~5 min, requires positions populated)

```bash
poetry run python -c "
from main import build_context, run_baseline_backtest, run_baseline_summary
ctx = build_context()
run_baseline_backtest(ctx)
run_baseline_summary(ctx)
"
```

### Cost sensitivity only (3 scenarios, ~2 min, requires baseline populated)

```bash
poetry run python -c "
from main import build_context, run_cost_sensitivity_scenarios
ctx = build_context()
run_cost_sensitivity_scenarios(ctx)
"
```

### Factor exclusion only (4 scenarios, ~10-30 min)

```bash
poetry run python -c "
from main import build_context, run_factor_exclusion_scenarios
ctx = build_context()
run_factor_exclusion_scenarios(ctx)
"
```

### Parameter sensitivity only (15 scenarios, ~30-75 min)

```bash
poetry run python -c "
from main import build_context, run_parameter_sensitivity_scenarios
ctx = build_context()
run_parameter_sensitivity_scenarios(ctx)
"
```

Resume support is built in - if a previous run was interrupted, this command picks up where it left off.

### Reports only (regenerate charts + CSVs, ~5 seconds)

```bash
poetry run python -c "
from main import build_context, run_reporting_outputs
ctx = build_context()
run_reporting_outputs(ctx)
"
```

### Dashboard only

```bash
poetry run streamlit run dashboard/Home.py
```

Browser opens automatically at `http://localhost:8501`.

## Inspecting the database directly

The `fift` PostgreSQL database is exposed on `localhost:5439`. Useful queries:

```bash
# Quick metric summary across all 23 scenarios
PGPASSWORD=postgres psql -h localhost -p 5439 -U postgres -d fift -c "
  SELECT scenario_id,
         ROUND(annualised_return::numeric, 4) AS ann_ret,
         ROUND(sharpe_ratio::numeric, 4) AS sharpe,
         ROUND(max_drawdown::numeric, 4) AS max_dd
  FROM team_wittgenstein.backtest_summary
  ORDER BY scenario_id;
"

# Holdings on the latest rebalance
PGPASSWORD=postgres psql -h localhost -p 5439 -U postgres -d fift -c "
  SELECT direction, COUNT(*) AS stocks, ROUND(SUM(final_weight)::numeric, 4) AS total_weight
  FROM team_wittgenstein.portfolio_positions
  WHERE rebalance_date = (SELECT MAX(rebalance_date) FROM team_wittgenstein.portfolio_positions)
  GROUP BY direction;
"
```

Or via pgAdmin at `http://localhost:5051` (login: `admin@admin.com` / `admin`).

## Troubleshooting

**Database connection refused**
```bash
docker compose up -d postgres_db
# wait 10 seconds, try again
```

**Postgres container keeps crashing under load (parameter sensitivity)**
The 15-scenario sweep makes ~4,000 DB calls. The Postgres container can drop the connection. The `run_parameter_sensitivity` function has resume support: re-running `main.py` skips already-completed scenarios.

**`Command not found: streamlit`**
Make sure to use `poetry run streamlit ...` rather than calling `streamlit` directly. Poetry's virtualenv has it installed; your shell's default Python doesn't.

## Team

Team Wittgenstein — UCL IFT Big Data in Quantitative Finance 2025-26
