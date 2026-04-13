-- CW2 tables for the 130/30 multi-factor strategy
-- Run against the 'fift' database

-- Drop old version if it exists (schema changed)
DROP TABLE IF EXISTS team_wittgenstein.liquidity_metrics CASCADE;

CREATE TABLE team_wittgenstein.liquidity_metrics (
    liquidity_id    SERIAL PRIMARY KEY,
    symbol          VARCHAR(12) NOT NULL,
    calc_date       DATE NOT NULL,
    adv_20d         NUMERIC,
    amihud_illiq    NUMERIC,
    illiq_rank_pct  NUMERIC,
    passes_adv      BOOLEAN,
    passes_illiq    BOOLEAN,
    passes_filter   BOOLEAN,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, calc_date)
);

CREATE INDEX IF NOT EXISTS idx_liquidity_metrics_date
    ON team_wittgenstein.liquidity_metrics (calc_date);
