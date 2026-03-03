-- Team Wittgenstein schema and table definitions
-- Run against the 'fift' database

-- Create team schema
CREATE SCHEMA IF NOT EXISTS team_wittgenstein AUTHORIZATION postgres;


-- Daily price data (source: Yahoo Finance)
-- 5 years of OHLCV + adjusted close for all 678 companies

CREATE TABLE IF NOT EXISTS team_wittgenstein.price_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(12) NOT NULL,
    trade_date DATE NOT NULL,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    adjusted_close NUMERIC,
    volume BIGINT,
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_price_data_symbol
    ON team_wittgenstein.price_data (symbol);

CREATE INDEX IF NOT EXISTS idx_price_data_date
    ON team_wittgenstein.price_data (trade_date);


-- Quarterly financial statements (source: yfinance)
-- Balance sheet + income statement fields needed for factors

CREATE TABLE IF NOT EXISTS team_wittgenstein.financial_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(12) NOT NULL,
    fiscal_year INT NOT NULL,
    fiscal_quarter INT NOT NULL,
    report_date DATE,
    currency VARCHAR(10),
    total_assets NUMERIC,
    total_equity NUMERIC,
    total_debt NUMERIC,
    book_value_equity NUMERIC,
    shares_outstanding BIGINT,
    net_income NUMERIC,
    eps NUMERIC,
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, fiscal_year, fiscal_quarter)
);

CREATE INDEX IF NOT EXISTS idx_financial_data_symbol
    ON team_wittgenstein.financial_data (symbol);


-- Risk-free rates by country (source: OECD API / yfinance fallback)
-- Monthly short-term interest rates used for momentum factor

CREATE TABLE IF NOT EXISTS team_wittgenstein.risk_free_rates (
    id SERIAL PRIMARY KEY,
    country VARCHAR(50) NOT NULL,
    rate_date DATE NOT NULL,
    rate NUMERIC,
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(country, rate_date)
);


-- Raw calculated factor metrics (before normalisation)
-- Derived from prices + financials + risk-free rates

CREATE TABLE IF NOT EXISTS team_wittgenstein.factor_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(12) NOT NULL,
    calc_date DATE NOT NULL,

    -- Value factor
    pb_ratio NUMERIC,
    asset_growth NUMERIC,

    -- Quality factor
    roe NUMERIC,
    roa NUMERIC,
    leverage NUMERIC,
    earnings_stability NUMERIC,

    -- Momentum factor
    momentum_6m NUMERIC,
    momentum_12m NUMERIC,

    -- Low volatility factor
    volatility_3m NUMERIC,
    volatility_12m NUMERIC,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, calc_date)
);

CREATE INDEX IF NOT EXISTS idx_factor_metrics_symbol
    ON team_wittgenstein.factor_metrics (symbol);

CREATE INDEX IF NOT EXISTS idx_factor_metrics_date
    ON team_wittgenstein.factor_metrics (calc_date);


-- Sector-normalised factor z-scores and composite score
-- Derived from factor_metrics after winsorisation and standardisation

CREATE TABLE IF NOT EXISTS team_wittgenstein.factor_scores (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(12) NOT NULL,
    score_date DATE NOT NULL,
    z_value NUMERIC,
    z_quality NUMERIC,
    z_momentum NUMERIC,
    z_low_vol NUMERIC,
    composite_score NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, score_date)
);

CREATE INDEX IF NOT EXISTS idx_factor_scores_symbol
    ON team_wittgenstein.factor_scores (symbol);

CREATE INDEX IF NOT EXISTS idx_factor_scores_date
    ON team_wittgenstein.factor_scores (score_date);


-- Portfolio positions output
-- Monthly rebalancing: 130% long top decile / 30% short bottom decile

CREATE TABLE IF NOT EXISTS team_wittgenstein.portfolio_positions (
    id SERIAL PRIMARY KEY,
    rebalance_date DATE NOT NULL,
    symbol VARCHAR(12) NOT NULL,
    sector VARCHAR(50),
    direction VARCHAR(5) NOT NULL CHECK (direction IN ('long', 'short')),
    weight NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(rebalance_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_positions_date
    ON team_wittgenstein.portfolio_positions (rebalance_date);
