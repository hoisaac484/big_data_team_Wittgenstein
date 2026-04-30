# Changelog

All notable changes to the Wittgenstein data pipeline are documented here.

## [Unreleased]

---

## [0.5.0] - 2026-04-30

### Added
- Coursework Two Sphinx documentation:
  - `docs/Makefile` build wrapper
  - installation, usage, pipeline, dashboard, and schema pages
  - API documentation for `modules/*` and `dashboard/lib/*`
- Waterfall early-exit: SimFin and yfinance are skipped when the upstream
  source already covers all fill fields with no nulls, reducing unnecessary
  API calls for well-covered symbols
- Streamlit dashboard Quick Start mode backed by a committed compressed
  `seed.sql.gz` snapshot for graders and reviewers
- Coursework Two dashboard:
  - multi-page Streamlit app
  - reusable chart, query, formatting, component, theme, and DB helper layers
  - expanded dashboard test coverage
- Coursework Two scenario analysis extensions:
  - benchmark caching and benchmark risk metrics
  - summary metrics, cost sensitivity, factor exclusion, parameter sensitivity
  - visualisation/reporting outputs

### Changed
- Coursework Two README expanded with:
  - Quick Start and Full Pipeline run modes
  - seed regeneration workflow
  - direct database inspection and stage-by-stage execution guidance
- Streamlit app launch model simplified by removing Dockerised Streamlit in
  favour of local `poetry run streamlit run ...`
- Teammate dashboard integration updated, including Docker wiring changes and
  Streamlit width API adjustments

### Fixed
- EDGAR submissions archive fetching: `_edgar_get_fiscal_periods` now reads
  `filings.files` archive entries in addition to `filings.recent`, preventing
  silent data loss for companies whose older 10-Q/10-K filings were pushed out
  of the ~1 000-entry `recent` window (e.g. JPM was missing ~65 quarters)
- Sphinx build stability for Coursework Two:
  - malformed docstrings normalised so autodoc no longer emits substitution /
    indentation errors
  - docs navigation corrected so pages like Installation and Usage appear
    consistently in the sidebar
- Minor tooling fixes:
  - missing import-spacing blank line added for `isort`
  - DB env var override and additional tests brought Coursework Two coverage up
    to 99%

---

## [0.4.0] - 2026-03-15

### Added
- APScheduler integration: pipeline runs once on startup then schedules recurring tasks
  — prices + risk-free rates monthly (1st of month, 02:00 UTC),
  — fundamentals quarterly (1st of Jan/Apr/Jul/Oct, 04:00 UTC)
- `PipelineContext` dataclass to bundle all pipeline dependencies
- CLI arguments: `--task {all,prices,fundamentals}`, `--no-schedule`, `--run-date`
- Config-driven scheduler cron times (`scheduler.prices_and_rates`, `scheduler.fundamentals`)
- Bandit (SAST) and pip-audit (dependency vulnerability) scanning in CI security job
- 80% test coverage threshold enforced via pytest-cov (`--cov-fail-under=80`)
- End-to-end tests (`test_e2e.py`) using real `DataValidator` and `DataWriter` with mocked I/O
- Ticker exclusion list in `conf.yaml` for delisted/renamed symbols

### Fixed
- Dot-ticker bug: symbols like `BF.B` now normalised to `BF-B` before comparing against
  managed tables, preventing valid tickers from being incorrectly deleted
- Single-symbol yfinance fetch: `_reshape_price_df` now detects which MultiIndex level
  contains price column names, fixing missing price data when only one symbol is re-fetched
  after a partial cache invalidation
- `black` bumped to `^26.3` to resolve CVE-2026-32274

### Changed
- Scheduler serialises jobs with `ThreadPoolExecutor(max_workers=1)` to prevent concurrent
  API calls and DB write contention
- Universe refresh (`_load_universe`) called at the start of each scheduled task, not only
  at startup, so company additions and delistings are reflected in every run

---

## [0.3.0] - 2026-03-13

### Added
- EDGAR waterfall pipeline: fundamentals fetched via EDGAR → SimFin → yfinance → forward-fill
- Bulk EDGAR `companyfacts` endpoint replaces per-filing requests for ~10× speed improvement
- Support for removing companies deleted from `company_static` — stale data pruned from
  managed tables and MinIO cache automatically
- Currency column added to price data

### Changed
- `data_collector.py` refactored into a package with mixins (`prices`, `fundamentals`,
  `rates`, `cache`, `edgar`) for maintainability
- Alpha Vantage removed as a fundamentals source; EDGAR is now the primary source
- isort applied across entire codebase

### Fixed
- Duplicate row prevention in price and fundamentals writes

---

## [0.2.0] - 2026-03-06

### Added
- Delisted stock detection and classification (`delisted` vs `fetch_error`)
- Composite primary keys in PostgreSQL schema to enforce deduplication
- `risk_free_rates` table added to schema; auto-increment PKs added to all tables
- SimFin and Alpha Vantage as alternative fundamentals sources

### Changed
- Data validator thresholds tightened (stricter null %, minimum row counts)
- Schema updated: removed ROA column, added `risk_free_rates`, updated comments

---

## [0.1.0] - 2026-03-01

### Added
- Initial data pipeline: `DataFetcher`, `DataValidator`, `DataWriter`
- PostgreSQL schema (`create_schema.sql`) with `price_data`, `financial_data` tables
- MinIO cache with CTL control files and configurable TTL
- MongoDB logging for fetch failures and metadata
- yfinance price fetching with per-symbol parquet caching
- Project structure, `.gitignore`, and documentation scaffold

---

## [0.0.1] - 2026-02-27

### Added
- Initial project structure and folder layout
- Data dictionary and factor definitions documentation
- Factor formula documentation (Value, Momentum, Quality, Volatility)
