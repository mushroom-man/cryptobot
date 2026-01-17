-- CryptoBot Platform - Database Initialization
-- This script runs automatically when the TimescaleDB container starts for the first time.

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- OHLCV Data (Core market data)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ohlcv (
    timestamp    TIMESTAMPTZ NOT NULL,
    pair         TEXT NOT NULL,
    open         DOUBLE PRECISION,
    high         DOUBLE PRECISION,
    low          DOUBLE PRECISION,
    close        DOUBLE PRECISION,
    volume       DOUBLE PRECISION,
    source       TEXT,
    PRIMARY KEY (timestamp, pair)
);

SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_pair ON ohlcv (pair, timestamp DESC);

-- ============================================================================
-- Computed Features
-- ============================================================================
CREATE TABLE IF NOT EXISTS features (
    timestamp       TIMESTAMPTZ NOT NULL,
    pair            TEXT NOT NULL,
    -- Regime features
    regime_binseg   INTEGER,
    regime_msm      INTEGER,
    regime_hybrid   INTEGER,
    -- MA features
    ma_score        INTEGER,
    price_vs_sma_6  DOUBLE PRECISION,
    price_vs_sma_24 DOUBLE PRECISION,
    price_vs_sma_72 DOUBLE PRECISION,
    -- Volatility features
    rolling_std_168 DOUBLE PRECISION,
    garch_vol       DOUBLE PRECISION,
    -- Target
    target_24h      INTEGER,
    -- Prediction
    prediction      DOUBLE PRECISION,
    model_version   TEXT,
    PRIMARY KEY (timestamp, pair)
);

SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_features_pair ON features (pair, timestamp DESC);

-- ============================================================================
-- Trading Signals
-- ============================================================================
CREATE TABLE IF NOT EXISTS signals (
    timestamp       TIMESTAMPTZ NOT NULL,
    pair            TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    signal          TEXT,
    target_position DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    regime          INTEGER,
    prediction      DOUBLE PRECISION,
    PRIMARY KEY (timestamp, pair, strategy)
);

SELECT create_hypertable('signals', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- Executed Trades
-- ============================================================================
CREATE TABLE IF NOT EXISTS trades (
    id               SERIAL PRIMARY KEY,
    timestamp        TIMESTAMPTZ NOT NULL,
    pair             TEXT NOT NULL,
    strategy         TEXT NOT NULL,
    direction        TEXT NOT NULL,
    size             DOUBLE PRECISION,
    price            DOUBLE PRECISION,
    slippage_bps     DOUBLE PRECISION,
    transaction_cost DOUBLE PRECISION,
    execution_type   TEXT,
    order_id         TEXT,
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades (pair, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp DESC);

-- ============================================================================
-- Portfolio State (snapshots)
-- ============================================================================
CREATE TABLE IF NOT EXISTS portfolio (
    timestamp      TIMESTAMPTZ NOT NULL,
    pair           TEXT NOT NULL,
    position       DOUBLE PRECISION,
    entry_price    DOUBLE PRECISION,
    current_price  DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    PRIMARY KEY (timestamp, pair)
);

SELECT create_hypertable('portfolio', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- Account Equity
-- ============================================================================
CREATE TABLE IF NOT EXISTS equity (
    timestamp    TIMESTAMPTZ NOT NULL PRIMARY KEY,
    total_equity DOUBLE PRECISION,
    cash         DOUBLE PRECISION,
    invested     DOUBLE PRECISION,
    daily_pnl    DOUBLE PRECISION,
    drawdown     DOUBLE PRECISION,
    peak_equity  DOUBLE PRECISION
);

SELECT create_hypertable('equity', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- Model Registry
-- ============================================================================
CREATE TABLE IF NOT EXISTS models (
    id           SERIAL PRIMARY KEY,
    model_name   TEXT NOT NULL,
    version      TEXT NOT NULL,
    pair         TEXT,
    trained_at   TIMESTAMPTZ,
    train_start  TIMESTAMPTZ,
    train_end    TIMESTAMPTZ,
    val_auc      DOUBLE PRECISION,
    features     JSONB,
    hyperparams  JSONB,
    model_path   TEXT,
    is_active    BOOLEAN DEFAULT FALSE,
    UNIQUE (model_name, version, pair)
);

-- ============================================================================
-- Backtest Results
-- ============================================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id              SERIAL PRIMARY KEY,
    run_timestamp   TIMESTAMPTZ DEFAULT NOW(),
    strategy        TEXT NOT NULL,
    pair            TEXT NOT NULL,
    test_start      TIMESTAMPTZ,
    test_end        TIMESTAMPTZ,
    total_return    DOUBLE PRECISION,
    annual_return   DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    sortino_ratio   DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    num_trades      INTEGER,
    win_rate        DOUBLE PRECISION,
    config          JSONB,
    notes           TEXT
);

-- ============================================================================
-- Continuous Aggregates (auto-computed views)
-- ============================================================================

-- Daily OHLCV from hourly
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    pair,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM ohlcv
GROUP BY day, pair
WITH NO DATA;

-- Refresh policy
SELECT add_continuous_aggregate_policy('ohlcv_daily',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ============================================================================
-- Data Retention (optional - compress old data)
-- ============================================================================

-- Enable compression on ohlcv after 30 days
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'pair'
);

SELECT add_compression_policy('ohlcv', INTERVAL '30 days', if_not_exists => TRUE);

-- ============================================================================
-- Helpful Views
-- ============================================================================

-- Latest price per pair
CREATE OR REPLACE VIEW latest_prices AS
SELECT DISTINCT ON (pair)
    pair,
    timestamp,
    close as price
FROM ohlcv
ORDER BY pair, timestamp DESC;

-- Current positions
CREATE OR REPLACE VIEW current_positions AS
SELECT DISTINCT ON (pair)
    pair,
    timestamp,
    position,
    entry_price,
    current_price,
    unrealized_pnl
FROM portfolio
ORDER BY pair, timestamp DESC;

-- Latest equity
CREATE OR REPLACE VIEW latest_equity AS
SELECT *
FROM equity
ORDER BY timestamp DESC
LIMIT 1;

-- ============================================================================
-- Done
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE 'CryptoBot database initialized successfully!';
END $$;
