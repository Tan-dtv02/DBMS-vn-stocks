# Thiet ke Data Model ClickHouse cho HQTCSDL Stocks

Tai lieu nay tong hop cac file CSV/report hien co trong du an va de xuat cach dua du lieu len ClickHouse theo cac tang: staging, data warehouse, mart/report va audit.

## 1. Hien trang CSV chinh

| Nhom | File/thumuc | So dong xap xi | Vai tro |
| --- | --- | ---: | --- |
| Raw/dirty | `data/dirty/Data_500_stocks_dirty.csv` | 734,464 | Du lieu ban sau khi tao dirty va merge daily crawl |
| Raw daily crawl | `data/dirty/stock_*.csv`, `data/output/*.csv`, `data/stock_*.csv` | nho | File cao theo ngay, co `target_date` |
| Raw symbol crawl | `ingestion/data_crawl_2026/*.csv` | 103 file | Du lieu cao theo tung ma, nen coi la archive/staging |
| Master symbol | `ingestion/company_infor.csv` | 500 | Danh muc ma co phieu, company, sector, listed_date |
| Clean price | `data/clean/Data_500_stocks_clean_ver2.csv` | 717,455 | Gia da lam sach, nguon cho `stock_prices` |
| Feature store | `data/clean/features_all.csv` | 717,455 | Bang feature chung cho tat ca model |
| Sector encoding | `data/clean/symbol_sector_encoding.csv`, `sector_label_encoding.csv` | 500 / nho | Mapping sector va ma hoa sector |
| Data quality | `data/khaosatdata/*.csv`, `data/clean_log/*.csv` | vua/nho | Ket qua khao sat va log lam sach |
| Model1 reports | `models/model1/reports/*.csv,json` | lon/vua | Forecast gia/return, accuracy, backtest, walk-forward |
| Model2 reports | `models/model2/reports/*.csv,json,html` | vua | Du bao future return 5d, backtest, insights |
| Model3 reports | `models/model3/reports/*.csv,json,html` | vua | Phan loai tin hieu SELL/HOLD/BUY |
| Model4 output | `models/model4/output/*.csv,json` | vua/lon | Benchmark outperformance |
| Model5 output | `models/model5/output_model5/*.csv,json` | lon | Risk alert, risk mart, evaluation |

## 2. De xuat database ClickHouse

Nen tach logic thanh 4 database:

| Database | Muc dich |
| --- | --- |
| `stock_staging` | Du lieu raw/dirty/daily crawl, giu dau vet dau vao |
| `stock_dw` | Data Warehouse va Feature Store sach, dung cho model |
| `stock_mart` | Bang ket qua model/dashboard/report cho web |
| `stock_audit` | Audit pipeline, data quality, model run, alert Telegram |

Hien tai code dang dung nhieu bang trong database `stock` va `stock_mart_model5_risk_prediction`. Co the giu tam de pipeline khong vo, sau do migrate dan sang 4 database tren bang cach tao view/alias hoac sua config theo tung buoc.

## 3. Tang staging

Nen upload:

| Bang de xuat | Nguon CSV | Ghi chu |
| --- | --- | --- |
| `stock_staging.raw_daily_prices` | `data/dirty/stock_*.csv`, `data/output/*.csv`, `data/stock_*.csv` | Luu du lieu cao theo ngay, them `source_file`, `ingested_at`, `target_date` |
| `stock_staging.raw_dirty_prices` | `data/dirty/Data_500_stocks_dirty.csv` | Bang dirty tong hop, dung de audit/lap lai cleaning |
| `stock_staging.raw_company_info` | `ingestion/company_infor.csv` | Danh muc cong ty raw |
| `stock_staging.raw_symbol_history_2026` | `ingestion/data_crawl_2026/*.csv` | Co the upload neu can luu archive raw theo tung symbol |

Khong bat buoc upload:

- `data/dirty/Data_500_stocks_01-17.csv`: chi la file batch lich su dung de merge.
- Cac file daily da duoc merge va da co trong `raw_daily_prices`: co the chi luu local/artifact neu da co `file_ingestion_log`.

Cot nen them cho staging:

```sql
source_file String,
source_system String,
target_date Nullable(Date),
ingested_at DateTime DEFAULT now(),
file_checksum Nullable(String)
```

## 4. Tang Data Warehouse / Feature Store

Nen upload va dung lam nguon chinh cho tat ca model:

| Bang de xuat | Bang hien tai | Nguon |
| --- | --- | --- |
| `stock_dw.fact_stock_prices` | `stock.stock_prices` | `data/clean/Data_500_stocks_clean_ver2.csv` |
| `stock_dw.dim_stock_symbols` | `stock.stock_symbols` | `ingestion/company_infor.csv` |
| `stock_dw.dim_sector_encoding` | `stock.symbol_sector_encoding` + `sector_label_encoding.csv` | Sector label/encoding |
| `stock_dw.features_all` | `stock.features_all` | `connect_clickhouse/features_all.py` |

`features_all` nen la Feature Store chung, gom:

- key: `trading_date`, `symbol`;
- dimension: `encode_sector`;
- OHLCV: `open`, `high`, `low`, `close`, `volume`;
- technical features: `return_*`, `ma_*`, `volatility_*`, `drawdown_20d`, `volume_ratio_5_20`, `daily_range`, `body_ratio`, `close_position`;
- metadata: `created_at`.

Khong nen tiep tuc copy `features_all.csv` vao tung model nhu `models/model1/data/raw/features_all.csv`, `models/model3/data/raw/features_all.csv` vi de lech schema. Tat ca model nen select truc tiep tu `stock_dw.features_all` hoac `stock.features_all` trong giai doan chua migrate.

## 5. Tang mart/report cho web dashboard

### Bang chung cho tat ca model

Nen co cac bang dung chung de web lay du lieu de dang:

| Bang | Muc dich |
| --- | --- |
| `stock_mart.model_runs` | Moi lan train/inference cua tung model |
| `stock_mart.model_metrics` | Metrics dang key-value hoac JSON |
| `stock_mart.model_feature_importance` | Feature importance cua moi model |
| `stock_mart.model_backtest_daily` | Ket qua backtest theo ngay |
| `stock_mart.model_backtest_sweep` | Ket qua sweep tham so backtest |
| `stock_mart.model_prediction_accuracy` | Do dung/sai theo tung symbol/date |

Cot chuan nen co:

```sql
model_run_id UUID,
model_name String,
model_version Nullable(String),
created_at DateTime,
artifact_path Nullable(String)
```

### Mart rieng tung model

| Model | File hien co | Bang mart nen co | Nen upload? |
| --- | --- | --- | --- |
| Model1 | `predictions.csv`, `prediction_accuracy.csv`, `backtest.csv`, `walk_forward_*.csv` | `stock_mart.mart_model1_price_forecast` | Co |
| Model2 | `prediction_error.csv`, `backtest.csv`, `feature_importance.csv` | `stock_mart.mart_model2_future_return` | Co |
| Model3 | `predictions.csv`, `prediction_accuracy.csv`, `backtest.csv` | `stock_mart.mart_model3_trade_signal` | Co |
| Model4 | `benchmark_predictions.csv`, `feature_importance.csv`, `benchmark_metrics.json` | `stock_mart.mart_model4_benchmark_outperformance` | Co |
| Model5 | `risk_predictions.csv`, `risk_test_evaluation.csv`, `mart_risk_alerts.csv` | `stock_mart.mart_model5_risk_alerts` | Co |

Training feature CSV lon nhu `models/model4/output/benchmark_features.csv` va `models/model5/output_model5/risk_features.csv` khong nen coi la mart dashboard. Neu can reproducibility thi dua vao `stock_mart.model_training_datasets` hoac `stock_audit.model_training_rows`, con web chi nen dung prediction/mart/evaluation.

## 6. File nao nen/khong nen dua len ClickHouse

Nen dua len:

- `data/clean/Data_500_stocks_clean_ver2.csv` -> DW fact price.
- `data/clean/features_all.csv` hoac output tu `features_all.py` -> Feature Store.
- `ingestion/company_infor.csv`, sector encoding -> dimension.
- Model predictions/evaluations/backtest/feature importance -> mart/report.
- `metrics.json`, `backtest_metrics.json`, `benchmark_metrics.json`, `risk_metrics.json` -> `model_metrics`.
- `data/khaosatdata/missing_summary.csv`, `numeric_stats.csv`, `symbol_counts.csv`, clean summary -> audit/data quality.

Khong nen dua len truc tiep:

- File HTML dashboard cua model2/model3: nen luu local/artifact, ClickHouse chi luu `artifact_path`, `model_name`, `created_at`.
- Root `reports/*.csv`: co ve la output cu/ban sao cua model1, de tranh trung lap nen khong upload nua.
- `models/*/data/raw/features_all.csv`: nen bo qua vi Feature Store da co trong DW.
- Tat ca `data/clean_log/*` chi tiet qua lon: chi upload summary va sample bad rows, tru khi can audit rat sau.

## 7. Output con thieu/can chuan hoa theo model

### Model1

Da co kha day du: predictions, prediction_accuracy, feature_importance, backtest, sweep, walk-forward.

Can them:

- `mart_model1_price_forecast.csv`: ban compact cho web gom `prediction_date`, `target_date`, `symbol`, `real_close`, `predicted_close`, `predicted_return`, `actual_return`, `direction_correct`, `model_name`, `created_at`.
- `model_run_id` trong tat ca output.

### Model2

Da co `prediction_error.csv`, metrics, feature_importance, backtest, dashboard HTML.

Can them:

- `predictions.csv` hoac `mart_model2_future_return.csv` co day du `symbol`, `trading_date`, `close`, `volume`, `encode_sector`, `actual_future_return_5d`, `predicted_future_return_5d`, `signal`, `model_name`, `created_at`.
- Hien `prediction_error.csv` thieu `close`, `volume`, `encode_sector` neu muon lam dashboard chi tiet.
- `model_run_id`.

### Model3

Da co predictions rat day du, accuracy, feature_importance, backtest, dashboard HTML.

Can them:

- `mart_model3_trade_signal.csv`: ban compact gom `symbol`, `trading_date`, `predicted_signal`, `buy_probability`, `hold_probability`, `sell_probability`, `signal_confidence`, `buy_sell_margin`, `actual_signal`, `is_correct`, `model_name`, `created_at`.
- `model_run_id`.

### Model4

Da co benchmark predictions, feature importance, metrics.

Can them:

- `backtest.csv` va `backtest_sweep.csv` cho benchmark strategy neu web can tab performance.
- `mart_model4_benchmark_outperformance.csv` nen upload vao `stock_mart`, khong nen de trong database `stock`.
- Fix config upload de dung helper ClickHouse chung va root `.env`.
- `model_run_id`.

### Model5

Da co output kha tot: risk features, risk predictions, risk test evaluation, mart risk alerts, metrics.

Can them:

- `feature_importance.csv` cho XGBoost/logistic neu muon so sanh feature tren web.
- `backtest_risk_alerts.csv`: ty le dung theo ngay/nganh/nhom rui ro.
- `model_run_id`.
- Nen doi database hien tai `stock_mart_model5_risk_prediction` ve chung `stock_mart`.

## 8. Bang audit nen co

| Bang | Noi dung |
| --- | --- |
| `stock_audit.pipeline_runs` | Run tong: workflow, start/end, status, git_sha, rows dirty/clean/features, models completed |
| `stock_audit.pipeline_steps` | Tung buoc trong `main.py`: script, status, duration, error |
| `stock_audit.file_ingestion_log` | File daily nao da merge/chua merge, so dong, ngay, checksum |
| `stock_audit.data_quality_summary` | So dong loi theo check: missing, invalid symbol, invalid OHLC, duplicate |
| `stock_audit.data_quality_samples` | Sample row loi quan trong, gioi han dung luong |
| `stock_audit.model_run_log` | Thong tin train/inference cua tung model |
| `stock_audit.telegram_alert_log` | Log alert bat dau/ket thuc workflow/gui thanh cong hay that bai |

## 9. Thu tu uu tien thuc hien

1. Giu pipeline hien tai chay duoc voi database `stock`.
2. Tao `stock_audit.pipeline_runs`, `pipeline_steps`, `file_ingestion_log`.
3. Chuan hoa output moi model co `model_run_id`, `model_name`, `created_at`.
4. Tao cac bang chung `stock_mart.model_metrics`, `model_feature_importance`, `model_backtest_daily`, `model_backtest_sweep`.
5. Tao mart rieng cho 5 model.
6. Web dashboard doc truc tiep tu `stock_mart` va `stock_audit`; file HTML hien tai chi dung nhu artifact/prototype.
7. Khi on dinh, migrate `stock.stock_prices`, `stock.stock_symbols`, `stock.features_all` sang `stock_dw`.

