# Coursework One — Data Pipeline

Data pipeline for a 130/30 multi-factor equity strategy. Built for Big Data in Quantitative Finance (UCL IFT, 2025-26).

Fetches daily prices, quarterly financials, and risk-free rates from multiple sources, validates data quality, and loads results into PostgreSQL for downstream factor modelling.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- Docker and Docker Compose for running PostgreSQL, MongoDB, and MinIO

## Quickstart

Start the backing services:

```bash
docker compose up -d
```

Install Python dependencies:

```bash
cd team_wittgenstein/coursework_one
poetry install
```

Run the pipeline once:

```bash
poetry run python main.py --no-schedule
```

You can also run individual stages:

```bash
poetry run python main.py --no-schedule --task prices
poetry run python main.py --no-schedule --task fundamentals
```

Without `--no-schedule`, the pipeline starts a recurring scheduler (see [Scheduling](#scheduling) below).

## Project structure

```
coursework_one/
├── main.py                 # Entry point and scheduler
├── config/
│   └── conf.yaml           # All configuration
├── modules/
│   ├── input/              # DataFetcher (prices, fundamentals, rates, caching)
│   ├── processing/         # DataValidator (quality checks)
│   ├── output/             # DataWriter (PostgreSQL + MongoDB loading)
│   └── db/                 # Database connection helpers
├── sql/
│   └── create_schema.sql   # PostgreSQL schema DDL
├── tests/                  # Unit and integration tests
└── docs/                   # Sphinx documentation source
```

## Configuration

All settings are in `config/conf.yaml`.

| Setting | Description |
|---------|-------------|
| `data.fundamentals_source` | `waterfall` (EDGAR → SimFin → yfinance → forward-fill) or `simfin` |
| `data.cache_ttl_days` | Re-fetch cached data older than N days (default 7) |
| `country_filter` | Restrict to one country (default `US`) |
| `exclude_symbols` | Tickers to skip (delisted, renamed, or broken — 72 total) |
| `dev.enabled` | Limit to `max_symbols` tickers for faster iteration |
| `validation.strict` | Halt on validation errors when `true` |

## Pipeline stages

1. **Fetch** — pulls prices, financials, and risk-free rates from external APIs. Prices are downloaded in a single yfinance batch call; any symbols silently dropped by the batch are retried individually. Fundamentals are fetched concurrently (5 threads) using a waterfall strategy: EDGAR first, then SimFin to fill gaps, then yfinance as a last resort, with remaining nulls forward-filled from prior quarters. All fetched data is cached as parquet files in MinIO (see [Caching](#caching)).
2. **Validate** — checks each dataset against configurable thresholds: minimum row count (200), minimum date coverage (4 years), maximum null percentage (50%), and duplicate keys. Sparse data (e.g. newly listed companies) triggers warnings but still passes. Corrupt or empty data raises errors, which halt the pipeline when `validation.strict` is `true`.
3. **Load** — writes validated DataFrames to PostgreSQL. Before each insert, existing rows are checked by primary key to prevent duplicates. Raw API responses are logged to MongoDB as timestamped audit documents for traceability.

## Data sources

| Data | Primary source | Fallback(s) | Notes |
|------|---------------|-------------|-------|
| Daily prices | yfinance batch download | Individual retry per missed symbol | 5-year history by default |
| Quarterly financials | SEC EDGAR (free, no key) | SimFin (free tier), then yfinance | Waterfall fills nulls from each successive source |
| Risk-free rates | OECD API | yfinance (^IRX) | US 3-month Treasury bill rate |

Rate limiting is handled per API: a minimum interval between requests (0.55s for SimFin, 0.5s for EDGAR) plus automatic retry on HTTP 429 responses.

## Caching

Fetched data is cached in MinIO as parquet files. Each file has a companion `.ctl` (control) JSON file that records the source, fetch timestamp, and row count. On subsequent runs, the pipeline checks the `.ctl` timestamp against `cache_ttl_days` (default 7) and only re-fetches stale data. This makes re-runs fast and avoids unnecessary API calls.

The cache lives under the `wittgenstein-cache` bucket, organised by data type:

```
wittgenstein-cache/
├── prices/          # One parquet + ctl per symbol
├── fundamentals/    # One parquet + ctl per symbol
└── risk_free_rates/ # Single parquet + ctl
```

## Scheduling

When run without `--no-schedule`, the pipeline starts an APScheduler `BlockingScheduler` that runs jobs on a cron schedule (all times UTC, configurable in `conf.yaml`):

| Job | Default schedule |
|-----|-----------------|
| Prices + risk-free rates | Monthly — 1st of every month at 02:00 |
| Fundamentals | Quarterly — 15th of Feb, May, Aug, Nov at 04:00 |

Jobs run one at a time to avoid concurrent database writes.

## Testing

Run all tests:

```bash
poetry run pytest
```

Coverage threshold is 80% (configured in `pyproject.toml`). To generate an HTML coverage report:

```bash
poetry run pytest --cov-report=html
```

## Linting and formatting

```bash
poetry run black modules/ main.py          # Code formatting
poetry run isort modules/ main.py          # Import sorting
poetry run flake8 modules/ main.py         # Style checks
poetry run bandit -r modules/ main.py      # Security linting
```

## Documentation

Build the Sphinx docs:

```bash
cd docs
poetry run sphinx-build -b html . _build/html
```

Open `docs/_build/html/index.html` to view.

## Team

Team Wittgenstein — UCL IFT Big Data in Quantitative Finance 2025-26
