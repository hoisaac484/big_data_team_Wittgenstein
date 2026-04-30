# Pipeline Overview

Coursework Two consumes the price, financial, and risk-free-rate data prepared
by Coursework One and produces factor scores, portfolios, backtest results, and
dashboard-facing summary tables.

## Main entrypoint

The batch pipeline starts in `main.py`. At a high level it:

1. Loads configuration from `config/conf.yaml`
2. Connects to PostgreSQL
3. Creates or refreshes Coursework Two tables from `sql/create_cw2_tables.sql`
4. Runs the factor, portfolio, backtest, and evaluation modules
5. Persists baseline outputs and scenario summaries

## Module flow

- `modules/liquidity/`: liquidity screening with ADTV and Amihud ILLIQ
- `modules/zscore/`: raw factor metric calculation, winsorisation, and z-scores
- `modules/composite/`: IC-weighted composite factor scoring
- `modules/portfolio/`: selection, volatility adjustment, and position sizing
- `modules/backtest/`: walk-forward return simulation and benchmark handling
- `modules/evaluation/`: metrics, cost sensitivity, factor exclusion, and reporting
- `modules/output/`: writes baseline strategy outputs and scenario summaries
- `modules/db/`: database access helpers

## Scenarios

The baseline strategy writes the detailed intermediate tables. Variant
scenarios are evaluated in memory and primarily persist backtest summaries and
returns, which keeps the database compact while still exposing scenario
comparison in the dashboard.
