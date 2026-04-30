# Database Schema

Coursework Two table definitions are stored in:

```text
sql/create_cw2_tables.sql
```

This schema contains the strategy outputs used by the dashboard and evaluation
pipeline, including baseline positions, factor scores, backtest returns, and
scenario summaries.

## Table groups

### Factor construction

- `factor_metrics`: raw per-symbol factor inputs at each calculation date
- `factor_zscores`: sub-metric z-scores retained as an audit trail
- `factor_scores`: top-level factor z-scores and composite score used downstream
- `ic_weights`: rolling information-coefficient weights by rebalance date

### Liquidity and selection

- `liquidity_metrics`: ADTV, Amihud ILLIQ, and pass/fail flags
- `selection_status`: per-rebalance membership and buffer-zone state

### Portfolio and returns

- `portfolio_positions`: baseline long/short portfolio weights and trade actions
- `benchmark_returns`: cached monthly benchmark series
- `backtest_returns`: monthly scenario returns, turnover, and cost fields
- `backtest_summary`: one-row aggregate metrics per scenario

## Grain and purpose

- `factor_metrics`, `factor_zscores`, `factor_scores`, and `liquidity_metrics`
  are at `symbol x date` grain.
- `portfolio_positions` and `selection_status` are at `symbol x rebalance_date`
  grain.
- `ic_weights` is at `rebalance_date x factor_name` grain.
- `backtest_returns` is at `scenario_id x rebalance_date` grain.
- `backtest_summary` is at `scenario_id` grain.

## Source of truth

Treat `sql/create_cw2_tables.sql` as the source of truth for exact DDL, types,
indexes, and constraints. This page is intended to explain how the tables fit
together rather than duplicate every column definition.
