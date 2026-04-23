-- CW2 tables for the 130/30 multi-factor strategy
-- Run against the 'fift' database

-- Drop and recreate shared factor tables so every run starts fresh
DROP TABLE IF EXISTS team_wittgenstein.factor_scores CASCADE;
DROP TABLE IF EXISTS team_wittgenstein.factor_metrics CASCADE;

CREATE TABLE team_wittgenstein.factor_metrics (
    metric_id           INT             GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol              VARCHAR(12)     NOT NULL,
    calc_date           DATE            NOT NULL,
    pb_ratio            NUMERIC,
    asset_growth        NUMERIC,
    roe                 NUMERIC,
    leverage            NUMERIC,
    earnings_stability  NUMERIC,
    momentum_6m         NUMERIC,
    momentum_12m        NUMERIC,
    volatility_3m       NUMERIC,
    volatility_12m      NUMERIC,
    created_at          TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (symbol, calc_date)
);

CREATE INDEX IF NOT EXISTS idx_factor_metrics_symbol
    ON team_wittgenstein.factor_metrics (symbol);
CREATE INDEX IF NOT EXISTS idx_factor_metrics_date
    ON team_wittgenstein.factor_metrics (calc_date);

CREATE TABLE team_wittgenstein.factor_scores (
    score_id        INT             GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol          VARCHAR(12)     NOT NULL,
    score_date      DATE            NOT NULL,
    z_value         NUMERIC,
    z_quality       NUMERIC,
    z_momentum      NUMERIC,
    z_low_vol       NUMERIC,
    composite_score NUMERIC,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (symbol, score_date)
);

CREATE INDEX IF NOT EXISTS idx_factor_scores_symbol
    ON team_wittgenstein.factor_scores (symbol);
CREATE INDEX IF NOT EXISTS idx_factor_scores_date
    ON team_wittgenstein.factor_scores (score_date);

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
    ic_mean_36m     NUMERIC         NOT NULL,
    ic_weight       NUMERIC         NOT NULL,
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

-- Cached monthly benchmark returns (MSCI USA via EUSA ETF)
DROP TABLE IF EXISTS team_wittgenstein.benchmark_returns CASCADE;
CREATE TABLE team_wittgenstein.benchmark_returns (
    bench_id        SERIAL PRIMARY KEY,
    benchmark       VARCHAR(20)     NOT NULL,
    month_end       DATE            NOT NULL,
    monthly_return  NUMERIC         NOT NULL,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (benchmark, month_end)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_returns_date
    ON team_wittgenstein.benchmark_returns (month_end);

-- Monthly backtest returns per scenario
DROP TABLE IF EXISTS team_wittgenstein.backtest_returns CASCADE;
CREATE TABLE team_wittgenstein.backtest_returns (
    return_id           SERIAL PRIMARY KEY,
    scenario_id         VARCHAR(50)     NOT NULL,
    rebalance_date      DATE            NOT NULL,
    gross_return        NUMERIC,
    net_return          NUMERIC,
    long_return         NUMERIC,
    short_return        NUMERIC,
    benchmark_return    NUMERIC,
    excess_return       NUMERIC,
    cumulative_return   NUMERIC,
    turnover            NUMERIC,
    transaction_cost    NUMERIC,
    created_at          TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (scenario_id, rebalance_date)
);

CREATE INDEX IF NOT EXISTS idx_backtest_returns_scenario
    ON team_wittgenstein.backtest_returns (scenario_id);
CREATE INDEX IF NOT EXISTS idx_backtest_returns_date
    ON team_wittgenstein.backtest_returns (rebalance_date);

-- Aggregate backtest performance summary per scenario
DROP TABLE IF EXISTS team_wittgenstein.backtest_summary CASCADE;
CREATE TABLE team_wittgenstein.backtest_summary (
    summary_id              SERIAL PRIMARY KEY,
    scenario_id             VARCHAR(50)     NOT NULL UNIQUE,
    backtest_start          DATE            NOT NULL,
    backtest_end            DATE            NOT NULL,
    annualised_return       NUMERIC,
    annualised_volatility   NUMERIC,
    sharpe_ratio            NUMERIC,
    sortino_ratio           NUMERIC,
    calmar_ratio            NUMERIC,
    information_ratio       NUMERIC,
    max_drawdown            NUMERIC,
    alpha                   NUMERIC,
    avg_monthly_turnover    NUMERIC,
    long_contribution       NUMERIC,
    short_contribution      NUMERIC,
    created_at              TIMESTAMPTZ     DEFAULT NOW()
);
