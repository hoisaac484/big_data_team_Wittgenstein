-- Team Wittgenstein schema and table definitions
-- Run against the 'fift' database

-- Create team schema
CREATE SCHEMA IF NOT EXISTS team_wittgenstein AUTHORIZATION postgres;


-- Daily price data (source: Yahoo Finance)
-- 5 years of OHLCV + adjusted close for all 678 companies

CREATE TABLE IF NOT EXISTS team_wittgenstein.daily_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(12) NOT NULL,
    price_date DATE NOT NULL,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    adj_close NUMERIC,
    volume BIGINT,
    UNIQUE(symbol, price_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol
    ON team_wittgenstein.daily_prices (symbol);

CREATE INDEX IF NOT EXISTS idx_daily_prices_date
    ON team_wittgenstein.daily_prices (price_date);


-- Quarterly financial statements (source: yfinance / SEC EDGAR)
-- Balance sheet + income statement fields needed for factors

CREATE TABLE IF NOT EXISTS team_wittgenstein.financials (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(12) NOT NULL,
    fiscal_date DATE NOT NULL,
    total_assets NUMERIC,
    total_equity NUMERIC,
    total_debt NUMERIC,
    net_income NUMERIC,
    eps NUMERIC,
    book_value NUMERIC,
    shares_outstanding BIGINT,
    UNIQUE(symbol, fiscal_date)
);

CREATE INDEX IF NOT EXISTS idx_financials_symbol
    ON team_wittgenstein.financials (symbol);


-- Risk-free rates by country (source: OECD API)
-- Monthly short-term interest rates used for momentum factor
CREATE TABLE IF NOT EXISTS team_wittgenstein.risk_free_rates (
    id SERIAL PRIMARY KEY,
    country VARCHAR(50) NOT NULL,
    rate_date DATE NOT NULL,
    rate NUMERIC,
    UNIQUE(country, rate_date)
);

-- Calculated factor metrics and composite scores
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
    leverage NUMERIC,
    earnings_stability NUMERIC,

    -- Momentum factor
    momentum_6m NUMERIC,
    momentum_12m NUMERIC,

    -- Low volatility factor
    volatility_3m NUMERIC,
    volatility_12m NUMERIC,

    -- Z-scores (sector-neutralised)
    z_value NUMERIC,
    z_quality NUMERIC,
    z_momentum NUMERIC,
    z_low_vol NUMERIC,

    -- Final composite score
    composite_score NUMERIC,

    UNIQUE(symbol, calc_date)
);

CREATE INDEX IF NOT EXISTS idx_factor_metrics_symbol
    ON team_wittgenstein.factor_metrics (symbol);

CREATE INDEX IF NOT EXISTS idx_factor_metrics_date
    ON team_wittgenstein.factor_metrics (calc_date);

-- Portfolio positions output (for coursework two)
-- Monthly rebalancing: 130% long / 30% short

CREATE TABLE IF NOT EXISTS team_wittgenstein.long_short_positions (
    id SERIAL PRIMARY KEY,
    position_date DATE NOT NULL,
    ticker VARCHAR(12) NOT NULL,
    side VARCHAR(5) NOT NULL CHECK (side IN ('long', 'short')),
    weight NUMERIC NOT NULL,
    sector VARCHAR(50),
    market_cap NUMERIC,
    UNIQUE(position_date, ticker)
);
