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

-- Individual z-scores for each sub-metric (pre-aggregation audit trail)
DROP TABLE IF EXISTS team_wittgenstein.factor_zscores CASCADE;

CREATE TABLE team_wittgenstein.factor_zscores (
    zscore_id           SERIAL PRIMARY KEY,
    symbol              VARCHAR(12)     NOT NULL,
    calc_date           DATE            NOT NULL,
    z_pb_ratio          NUMERIC,
    z_asset_growth      NUMERIC,
    z_roe               NUMERIC,
    z_leverage          NUMERIC,
    z_earnings_stability NUMERIC,
    z_momentum_6m       NUMERIC,
    z_momentum_12m      NUMERIC,
    z_volatility_3m     NUMERIC,
    z_volatility_12m    NUMERIC,
    created_at          TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (symbol, calc_date)
);

CREATE INDEX IF NOT EXISTS idx_factor_zscores_date
    ON team_wittgenstein.factor_zscores (calc_date);

-- IC-weighted factor weights per rebalancing date
DROP TABLE IF EXISTS team_wittgenstein.ic_weights CASCADE;

CREATE TABLE team_wittgenstein.ic_weights (
    ic_id           SERIAL PRIMARY KEY,
    rebalance_date  DATE            NOT NULL,
    factor_name     VARCHAR(20)     NOT NULL,
    ic_mean_36m     NUMERIC,
    ic_weight       NUMERIC,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (rebalance_date, factor_name)
);

CREATE INDEX IF NOT EXISTS idx_ic_weights_date
    ON team_wittgenstein.ic_weights (rebalance_date);

-- Portfolio positions after 130/30 construction
DROP TABLE IF EXISTS team_wittgenstein.portfolio_positions CASCADE;

CREATE TABLE team_wittgenstein.portfolio_positions (
    position_id     SERIAL PRIMARY KEY,
    rebalance_date  DATE            NOT NULL,
    symbol          VARCHAR(12)     NOT NULL,
    sector          VARCHAR(50),
    direction       VARCHAR(5)      NOT NULL CHECK (direction IN ('long', 'short')),
    ewma_vol        NUMERIC,
    risk_adj_score  NUMERIC,
    target_weight   NUMERIC         NOT NULL,
    final_weight    NUMERIC         NOT NULL,
    liquidity_capped BOOLEAN        DEFAULT FALSE,
    trade_action    VARCHAR(10),
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (rebalance_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_positions_date
    ON team_wittgenstein.portfolio_positions (rebalance_date);

-- Selection status and buffer zone tracking per rebalancing date
DROP TABLE IF EXISTS team_wittgenstein.selection_status CASCADE;

CREATE TABLE team_wittgenstein.selection_status (
    selection_id        SERIAL PRIMARY KEY,
    symbol              VARCHAR(12)     NOT NULL,
    rebalance_date      DATE            NOT NULL,
    sector              VARCHAR(50),
    composite_score     NUMERIC,
    percentile_rank     NUMERIC,
    status              VARCHAR(20)     NOT NULL,
    buffer_months_count INT             DEFAULT 0,
    entry_date          DATE,
    exit_reason         VARCHAR(20),
    created_at          TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (symbol, rebalance_date)
);

CREATE INDEX IF NOT EXISTS idx_selection_status_date
    ON team_wittgenstein.selection_status (rebalance_date);
CREATE INDEX IF NOT EXISTS idx_selection_status_symbol
    ON team_wittgenstein.selection_status (symbol);
