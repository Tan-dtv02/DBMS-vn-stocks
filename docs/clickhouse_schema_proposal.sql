-- Proposed ClickHouse schema for HQTCSDL Stocks.
-- This file is a blueprint. Do not run blindly on production before checking
-- current table names used by the pipeline.

CREATE DATABASE IF NOT EXISTS stock_staging;
CREATE DATABASE IF NOT EXISTS stock_dw;
CREATE DATABASE IF NOT EXISTS stock_mart;
CREATE DATABASE IF NOT EXISTS stock_audit;

-- =========================
-- 1. STAGING
-- =========================

CREATE TABLE IF NOT EXISTS stock_staging.raw_daily_prices
(
    time Nullable(DateTime),
    open Nullable(Float64),
    high Nullable(Float64),
    low Nullable(Float64),
    close Nullable(Float64),
    volume Nullable(Float64),
    symbol String,
    target_date Nullable(Date),
    source_file String,
    source_system String DEFAULT 'daily_crawl',
    file_checksum Nullable(String),
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (symbol, target_date, source_file);

CREATE TABLE IF NOT EXISTS stock_staging.raw_dirty_prices
(
    time Nullable(DateTime),
    open Nullable(Float64),
    high Nullable(Float64),
    low Nullable(Float64),
    close Nullable(Float64),
    volume Nullable(Float64),
    symbol String,
    source_file String,
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (symbol, time);

CREATE TABLE IF NOT EXISTS stock_staging.raw_company_info
(
    symbol String,
    company_name Nullable(String),
    sector Nullable(String),
    listed_date Nullable(Date),
    source_file String,
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY symbol;

-- =========================
-- 2. DATA WAREHOUSE
-- =========================

CREATE TABLE IF NOT EXISTS stock_dw.fact_stock_prices
(
    symbol String,
    trading_date Date,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (symbol, trading_date);

CREATE TABLE IF NOT EXISTS stock_dw.dim_stock_symbols
(
    symbol String,
    company_name String,
    sector Nullable(String),
    encode_sector Nullable(Int32),
    listed_date Nullable(Date)
)
ENGINE = MergeTree
ORDER BY symbol;

CREATE TABLE IF NOT EXISTS stock_dw.dim_sector_encoding
(
    sector String,
    encode_sector Int32,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY encode_sector;

CREATE TABLE IF NOT EXISTS stock_dw.features_all
(
    trading_date Date,
    symbol String,
    encode_sector Nullable(Int32),
    open Nullable(Float64),
    high Nullable(Float64),
    low Nullable(Float64),
    close Nullable(Float64),
    volume Nullable(Float64),
    return_1d Nullable(Float64),
    return_3d Nullable(Float64),
    return_5d Nullable(Float64),
    return_10d Nullable(Float64),
    return_20d Nullable(Float64),
    ma_5 Nullable(Float64),
    ma_20 Nullable(Float64),
    ma_50 Nullable(Float64),
    price_vs_ma20 Nullable(Float64),
    ma5_vs_ma20 Nullable(Float64),
    volatility_5d Nullable(Float64),
    volatility_20d Nullable(Float64),
    volatility_change Nullable(Float64),
    rolling_max_20d Nullable(Float64),
    drawdown_20d Nullable(Float64),
    volume_ma_5 Nullable(Float64),
    volume_ma_20 Nullable(Float64),
    volume_ratio_5_20 Nullable(Float64),
    volume_change_1d Nullable(Float64),
    daily_range Nullable(Float64),
    body_ratio Nullable(Float64),
    close_position Nullable(Float64),
    created_at DateTime
)
ENGINE = MergeTree
ORDER BY (symbol, trading_date);

-- =========================
-- 3. SHARED MODEL REPORTS
-- =========================

CREATE TABLE IF NOT EXISTS stock_mart.model_runs
(
    model_run_id UUID,
    model_name String,
    model_version Nullable(String),
    run_type String,
    train_start_date Nullable(Date),
    train_end_date Nullable(Date),
    test_start_date Nullable(Date),
    test_end_date Nullable(Date),
    train_rows Nullable(UInt64),
    test_rows Nullable(UInt64),
    status String,
    artifact_dir Nullable(String),
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (model_name, created_at, model_run_id);

CREATE TABLE IF NOT EXISTS stock_mart.model_metrics
(
    model_run_id UUID,
    model_name String,
    metric_name String,
    metric_value Nullable(Float64),
    metric_text Nullable(String),
    metric_json Nullable(String),
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (model_name, model_run_id, metric_name);

CREATE TABLE IF NOT EXISTS stock_mart.model_feature_importance
(
    model_run_id UUID,
    model_name String,
    feature String,
    importance Float64,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (model_name, model_run_id, importance);

CREATE TABLE IF NOT EXISTS stock_mart.model_backtest_daily
(
    model_run_id UUID,
    model_name String,
    trading_date Date,
    selected_symbols Nullable(String),
    selected_count Nullable(UInt32),
    daily_return Nullable(Float64),
    benchmark_return Nullable(Float64),
    daily_return_net Nullable(Float64),
    cumulative_return Nullable(Float64),
    cumulative_return_net Nullable(Float64),
    benchmark_cumulative_return Nullable(Float64),
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (model_name, trading_date, model_run_id);

CREATE TABLE IF NOT EXISTS stock_mart.model_backtest_sweep
(
    model_run_id UUID,
    model_name String,
    params_json String,
    average_daily_return Nullable(Float64),
    cumulative_return Nullable(Float64),
    hit_rate Nullable(Float64),
    sharpe_ratio Nullable(Float64),
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (model_name, model_run_id);

-- =========================
-- 4. MODEL-SPECIFIC MARTS
-- =========================

CREATE TABLE IF NOT EXISTS stock_mart.mart_model1_price_forecast
(
    model_run_id UUID,
    prediction_date Date,
    target_date Nullable(Date),
    symbol String,
    real_close Nullable(Float64),
    predicted_close Nullable(Float64),
    actual_return Nullable(Float64),
    predicted_return Nullable(Float64),
    direction_correct Nullable(UInt8),
    model_name String DEFAULT 'model1',
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (symbol, prediction_date, model_run_id);

CREATE TABLE IF NOT EXISTS stock_mart.mart_model2_future_return
(
    model_run_id UUID,
    trading_date Date,
    symbol String,
    close Nullable(Float64),
    volume Nullable(Float64),
    encode_sector Nullable(Int32),
    actual_future_return_5d Nullable(Float64),
    predicted_future_return_5d Nullable(Float64),
    prediction_error Nullable(Float64),
    abs_error Nullable(Float64),
    signal String,
    model_name String DEFAULT 'model2',
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (symbol, trading_date, model_run_id);

CREATE TABLE IF NOT EXISTS stock_mart.mart_model3_trade_signal
(
    model_run_id UUID,
    trading_date Date,
    symbol String,
    predicted_signal String,
    adjusted_signal Nullable(String),
    sell_probability Nullable(Float64),
    hold_probability Nullable(Float64),
    buy_probability Nullable(Float64),
    signal_confidence Nullable(Float64),
    buy_sell_margin Nullable(Float64),
    actual_signal Nullable(String),
    is_correct Nullable(UInt8),
    model_name String DEFAULT 'model3',
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (symbol, trading_date, model_run_id);

CREATE TABLE IF NOT EXISTS stock_mart.mart_model4_benchmark_outperformance
(
    model_run_id UUID,
    trading_date Date,
    symbol String,
    close Nullable(Float64),
    label Nullable(Int8),
    predicted_label Nullable(Int8),
    outperform_probability Nullable(Float64),
    prediction_correct Nullable(UInt8),
    model_name String DEFAULT 'model4',
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (symbol, trading_date, model_run_id);

CREATE TABLE IF NOT EXISTS stock_mart.mart_model5_risk_alerts
(
    model_run_id UUID,
    prediction_date Date,
    target_date Date,
    symbol String,
    company_name Nullable(String),
    open Nullable(Float64),
    high Nullable(Float64),
    low Nullable(Float64),
    close Nullable(Float64),
    volume Nullable(Float64),
    return_5d Nullable(Float64),
    drawdown_20d Nullable(Float64),
    volatility_5d Nullable(Float64),
    volume_ratio_5_20 Nullable(Float64),
    daily_range Nullable(Float64),
    close_position Nullable(Float64),
    risk_probability Nullable(Float64),
    risk_label String,
    model_name String DEFAULT 'model5',
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (symbol, prediction_date, model_run_id);

-- =========================
-- 5. AUDIT
-- =========================

CREATE TABLE IF NOT EXISTS stock_audit.pipeline_runs
(
    pipeline_run_id UUID,
    workflow_name String,
    trigger_type Nullable(String),
    git_sha Nullable(String),
    started_at DateTime,
    finished_at Nullable(DateTime),
    status String,
    dirty_rows Nullable(UInt64),
    clean_rows Nullable(UInt64),
    features_rows Nullable(UInt64),
    completed_models Array(String),
    error_message Nullable(String)
)
ENGINE = MergeTree
ORDER BY (workflow_name, started_at, pipeline_run_id);

CREATE TABLE IF NOT EXISTS stock_audit.pipeline_steps
(
    pipeline_run_id UUID,
    step_name String,
    script_path String,
    started_at DateTime,
    finished_at Nullable(DateTime),
    status String,
    rows_in Nullable(UInt64),
    rows_out Nullable(UInt64),
    duration_seconds Nullable(Float64),
    error_message Nullable(String)
)
ENGINE = MergeTree
ORDER BY (pipeline_run_id, started_at, step_name);

CREATE TABLE IF NOT EXISTS stock_audit.file_ingestion_log
(
    pipeline_run_id UUID,
    file_name String,
    file_path String,
    target_date Nullable(Date),
    rows_read Nullable(UInt64),
    rows_merged Nullable(UInt64),
    skipped_reason Nullable(String),
    file_checksum Nullable(String),
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (target_date, file_name);

CREATE TABLE IF NOT EXISTS stock_audit.data_quality_summary
(
    pipeline_run_id UUID,
    check_name String,
    source_file Nullable(String),
    bad_rows UInt64,
    total_rows Nullable(UInt64),
    bad_ratio Nullable(Float64),
    report_file Nullable(String),
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (pipeline_run_id, check_name);

CREATE TABLE IF NOT EXISTS stock_audit.telegram_alert_log
(
    pipeline_run_id Nullable(UUID),
    workflow_name String,
    alert_type String,
    message String,
    sent_at DateTime DEFAULT now(),
    status String,
    error_message Nullable(String)
)
ENGINE = MergeTree
ORDER BY (workflow_name, sent_at);

