# HQTCSDL Stocks

Dự án này xây dựng một pipeline dữ liệu và mô hình cho bộ dữ liệu cổ phiếu Việt Nam. Pipeline bắt đầu từ dữ liệu giá thô, làm sạch dữ liệu, upload lên ClickHouse, tạo bảng feature dùng chung, train các mô hình trong `models/`, lưu kết quả dự đoán, và gửi cảnh báo vận hành qua Telegram khi workflow tự động bắt đầu hoặc kết thúc.

## Mục Tiêu

- Thu thập dữ liệu giá cổ phiếu theo ngày cho khoảng 500 mã.
- Merge dữ liệu crawl hằng ngày vào tập dirty gốc.
- Làm sạch dữ liệu dirty và tạo file clean chuẩn.
- Upload dữ liệu nền lên ClickHouse.
- Tạo bảng feature dùng chung `stock.features_all`.
- Train lại toàn bộ model trên dữ liệu mới nhất.
- Upload một số output model lên ClickHouse để phục vụ phân tích/dashboard.
- Tự động hóa bằng GitHub Actions.
- Gửi cảnh báo Telegram cho admin khi workflow bắt đầu/kết thúc.

## Cấu Trúc Chính

```text
.
├── main.py
├── main_cao_dl.py
├── requirement.txt
├── .github/workflows/
│   ├── schedule.yml
│   └── merge_daily_csv_and_run_pipeline.yml
├── ingestion/
│   ├── merge_daily_csv_to_dirty.py
│   ├── symbol500.txt
│   └── company_infor.csv
├── etl/xu_li_du_lieu/
│   ├── kiemtradl_ghiralog_extract.py
│   └── clean_db_ghiralog_transform.py
├── connect_clickhouse/
│   ├── load_prices_to_click_house.py
│   ├── load_symbols_to_clickhouse.py
│   └── features_all.py
├── notifications/
│   └── telegram_alert.py
├── data/
│   ├── dirty/
│   ├── clean/
│   ├── clean_log/
│   └── khaosatdata/
└── models/
    ├── model1/
    ├── model2/
    ├── model3/
    ├── model4/
    └── model5/
```

## Luồng Dữ Liệu

### 1. Dữ liệu dirty

File dirty chính:

```text
data/dirty/Data_500_stocks_dirty.csv
```

Schema chính:

```text
time, open, high, low, close, volume, symbol
```

Các file crawl theo ngày được lưu trong:

```text
data/dirty/stock_*.csv
```

Ví dụ:

```text
data/dirty/stock_21_5.csv
```

### 2. Merge dữ liệu crawl hằng ngày

Script:

```text
ingestion/merge_daily_csv_to_dirty.py
```

Chức năng:

- Đọc các file `data/dirty/stock_*.csv`.
- Kiểm tra ngày đã tồn tại trong `Data_500_stocks_dirty.csv` chưa.
- Chỉ append các ngày mới.
- Tránh trùng theo cặp `symbol + time`.
- Bỏ qua file `no_data` nếu ngày đó không có phiên giao dịch.

Chạy thử không ghi file:

```powershell
python ingestion\merge_daily_csv_to_dirty.py --dry-run
```

Chạy merge thật:

```powershell
python ingestion\merge_daily_csv_to_dirty.py
```

### 3. Làm sạch dữ liệu

Script:

```text
etl/xu_li_du_lieu/clean_db_ghiralog_transform.py
```

Input:

```text
data/dirty/Data_500_stocks_dirty.csv
```

Reference:

```text
data/clean/Data_500_stocks_2015-2026.csv
```

Output:

```text
data/clean/Data_500_stocks_clean_ver2.csv
```

Log:

```text
data/clean_log/
```

Logic hiện tại:

- Dữ liệu cũ có trong reference sẽ được dùng reference để khôi phục các ô bị bẩn.
- Dữ liệu mới sau ngày cuối của reference vẫn được giữ lại từ dirty nếu qua các rule clean.
- Loại symbol bẩn như `ZZZZZZ`, `???`, `123`, `NULL`, `A@`.
- Loại ngày lỗi, giá thiếu, giá âm hoặc bằng 0, volume âm, OHLC không hợp lệ, duplicate key.

### 4. Upload ClickHouse và tạo feature

Các bước chính:

```text
connect_clickhouse/load_prices_to_click_house.py
connect_clickhouse/load_symbols_to_clickhouse.py
connect_clickhouse/features_all.py
```

Bảng ClickHouse nền:

```text
stock.stock_prices
stock.stock_symbols
stock.symbol_sector_encoding
stock.features_all
```

`load_symbols_to_clickhouse.py` đọc:

```text
ingestion/company_infor.csv
```

và tạo thêm mã hóa ngành:

```text
encode_sector
```

`features_all.py` tạo bảng feature dùng chung từ `stock.stock_prices`, join thêm `encode_sector` từ `stock.symbol_sector_encoding`, rồi upload lên:

```text
stock.features_all
```

Bảng `features_all` là nguồn dữ liệu chung cho các mô hình. Các model không lấy feature chính từ CSV nữa mà đọc từ ClickHouse.

## Full Pipeline Local

Entrypoint chính:

```powershell
python main.py
```

Thứ tự chạy trong `main.py`:

```text
merge_daily_csv_to_dirty
survey_data
clean_data
load_prices
load_symbols
upload_features_all
train_model1
train_model2
train_model3
train_model4
upload_model4_outputs
train_model5
upload_model5_outputs
```

Ý nghĩa từng bước:

- `merge_daily_csv_to_dirty`: merge CSV crawl hằng ngày vào dirty.
- `survey_data`: khảo sát dữ liệu dirty và ghi log vào `data/khaosatdata/`.
- `clean_data`: làm sạch dirty và sinh `Data_500_stocks_clean_ver2.csv`.
- `load_prices`: upload clean price lên `stock.stock_prices`.
- `load_symbols`: upload thông tin công ty, sector, sector encoding.
- `upload_features_all`: tạo và upload `stock.features_all`.
- `train_model1`: train model 1 và lưu model/output trong `models/model1/`.
- `train_model2`: train model 2 LightGBM cho `future_return_5d`.
- `train_model3`: train model 3 tín hiệu giao dịch.
- `train_model4`: train benchmark outperformance model.
- `upload_model4_outputs`: upload prediction model 4 lên ClickHouse.
- `train_model5`: train risk alert model.
- `upload_model5_outputs`: upload output model 5 lên ClickHouse.

Trong khi chạy, `main.py` ghi tiến độ vào:

```text
data/pipeline_run_summary.json
```

File này được workflow và Telegram alert dùng để biết step nào đã chạy xong, step nào fail.

## Các Mô Hình

### Model 1

Thư mục:

```text
models/model1/
```

Mục tiêu:

- Dự báo giá/return tương lai bằng XGBoost.
- Đọc dữ liệu từ `stock.features_all`.
- Lưu model `.pkl` trong `models/model1/models/`.
- Lưu metric, prediction, backtest vào `models/model1/reports/`.

Chạy riêng:

```powershell
python models\model1\main.py
```

### Model 2

Thư mục:

```text
models/model2/
```

Mục tiêu:

- Hồi quy `future_return_5d`.
- Dùng LightGBM.
- Có walk-forward validation, train final model, evaluate và predict thử một mã.
- Đọc dữ liệu từ `stock.features_all`.
- Có sử dụng `encode_sector`.

Chạy riêng:

```powershell
python models\model2\run_pipeline.py
```

### Model 3

Thư mục:

```text
models/model3/
```

Mục tiêu:

- Phân loại tín hiệu giao dịch.
- Đọc dữ liệu từ `stock.features_all`.
- Lưu model `.pkl` trong `models/model3/models/`.
- Lưu báo cáo trong `models/model3/reports/`.

Chạy riêng:

```powershell
python models\model3\main.py
```

### Model 4

Thư mục:

```text
models/model4/
```

Mục tiêu:

- Dự đoán cổ phiếu có outperform benchmark hay không.
- Đọc dữ liệu từ `stock.features_all`.
- Tạo label benchmark trong quá trình train.
- Train LightGBM classifier.
- Lưu model:

```text
models/model4/models/benchmark_outperformance_lgbm.pkl
```

Output chính:

```text
models/model4/output/benchmark_metrics.json
models/model4/output/benchmark_predictions.csv
models/model4/output/feature_importance.csv
```

Upload lên ClickHouse:

```text
stock.model4_benchmark_predictions
```

Chạy riêng:

```powershell
python models\model4\train_benchmark_model.py
python models\model4\upload_predictions.py
```

### Model 5

Thư mục:

```text
models/model5/
```

Mục tiêu:

- Cảnh báo rủi ro giảm giá.
- Đọc dữ liệu từ `stock.features_all`.
- Tạo `future_return_5d`.
- Tạo nhãn nhị phân `risk_drop_label`.
- Train baseline logistic regression và XGBoost risk alert.
- Chọn model chính để sinh xác suất rủi ro.

Lưu model:

```text
models/model5/models/risk_alert_model.pkl
```

Output chính:

```text
models/model5/output_model5/risk_features.csv
models/model5/output_model5/risk_predictions.csv
models/model5/output_model5/risk_test_evaluation.csv
models/model5/output_model5/risk_metrics.json
models/model5/output_model5/mart_risk_alerts.csv
```

Upload lên ClickHouse database:

```text
stock_mart_model5_risk_prediction
```

Các bảng/view chính:

```text
risk_features
risk_predictions
risk_test_evaluation
mart_risk_alerts
dashboard_risk_alerts
```

Chạy riêng:

```powershell
python models\model5\run_pipeline.py
python models\model5\upload_outputs_to_clickhouse.py
```

## Workflow Tự Động Trên GitHub Actions

Dự án có 2 workflow chính.

### 1. Workflow `cao_dl`

File:

```text
.github/workflows/schedule.yml
```

Trigger:

- Tự động chạy mỗi ngày lúc `08:00 UTC`, tức `15:00 giờ Việt Nam`.
- Có thể chạy thủ công bằng `Run workflow`.

Khi chạy thủ công có input:

```text
target_date
```

Nếu để trống, script sẽ lấy ngày hôm qua theo giờ Việt Nam.

Luồng chạy:

```text
checkout code
setup Python 3.11
notify Telegram: workflow bắt đầu
install dependencies
python main_cao_dl.py
upload artifact data/dirty/stock_*.csv
commit CSV crawl mới nếu có thay đổi
notify Telegram: workflow kết thúc
```

Output:

```text
data/dirty/stock_ngày_tháng.csv
```

Ví dụ:

```text
data/dirty/stock_21_5.csv
```

### 2. Workflow `merge_daily_csv_and_run_pipeline`

File:

```text
.github/workflows/merge_daily_csv_and_run_pipeline.yml
```

Trigger:

- Tự động chạy sau khi workflow `cao_dl` hoàn tất thành công.
- Có thể chạy thủ công bằng `Run workflow`.

Luồng chạy:

```text
checkout latest code
setup Python 3.11
notify Telegram: workflow bắt đầu
install dependencies
python main.py
upload artifact dữ liệu và model output
commit data/pipeline_run_summary.json nếu thay đổi
notify Telegram: workflow kết thúc
```

Workflow này chính là bước nối sau `cao_dl`. Nó lấy CSV đã crawl trong `data/dirty/stock_*.csv`, merge vào dirty, clean lại dữ liệu, upload ClickHouse, train model, upload output model, sau đó gửi thông báo Telegram.

## Telegram Alert

Script:

```text
notifications/telegram_alert.py
```

Script này gửi cảnh báo Telegram khi workflow bắt đầu và kết thúc.

### Secrets cần tạo trên GitHub

Vào:

```text
Settings -> Secrets and variables -> Actions
```

Tạo 2 secret:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Không commit token vào code.

### Nội dung cảnh báo workflow `cao_dl`

Khi bắt đầu:

```text
Workflow: cao_dl
Trang thai: bat dau
Ngay cao: YYYY-MM-DD
File CSV: data/dirty/stock_D_M.csv
```

Khi kết thúc:

```text
Workflow: cao_dl
Trang thai: ket thuc - success/failure
Ngay cao: YYYY-MM-DD
File CSV: data/dirty/stock_D_M.csv
So dong cao: ...
Commit CSV: thanh cong neu co thay doi
Run: ...
```

### Nội dung cảnh báo workflow pipeline

Khi bắt đầu:

```text
Workflow: merge_daily_csv_and_run_pipeline
Trang thai: bat dau
Run: ...
```

Khi kết thúc:

```text
Workflow: merge_daily_csv_and_run_pipeline
Trang thai: ket thuc - success/failure
Model da chay xong: model1, model2, model3, model4, model5
File clean cuoi: data/clean/Data_500_stocks_clean_ver2.csv
Tong dong clean: ...
Tong dong dirty: ...
ClickHouse: upload thanh cong
Run: ...
```

Nếu pipeline fail giữa chừng, alert sẽ đọc `data/pipeline_run_summary.json` để báo step đã chạy xong và step dừng.

### Test Telegram local

PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN="token_cua_bot"
$env:TELEGRAM_CHAT_ID="chat_id"

python notifications\telegram_alert.py --workflow cao_dl --phase start
python notifications\telegram_alert.py --workflow cao_dl --phase end --status success
```

Nếu thiếu token/chat id, script không làm fail workflow. Nó chỉ in log:

```text
Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Skip alert.
```

## Biến Môi Trường Và Secrets

### ClickHouse

Cần có các biến môi trường hoặc GitHub Secrets:

```text
CLICKHOUSE_HOST
CLICKHOUSE_PORT
CLICKHOUSE_USER
CLICKHOUSE_PASSWORD
CLICKHOUSE_DATABASE
CLICKHOUSE_SECURE
```

Local có thể đặt trong `.env` hoặc set trực tiếp trên shell.

GitHub Actions cần tạo trong:

```text
Settings -> Secrets and variables -> Actions
```

### Telegram

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

## Cài Đặt Local

Tạo virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Cài dependency:

```powershell
pip install -r requirement.txt
```

Chạy toàn bộ pipeline:

```powershell
python main.py
```

Chạy crawler một ngày cụ thể:

```powershell
$env:TARGET_DATE="2026-05-21"
python main_cao_dl.py
```

Chạy merge daily ở chế độ kiểm tra:

```powershell
python ingestion\merge_daily_csv_to_dirty.py --dry-run
```

## Output Quan Trọng

Dữ liệu:

```text
data/dirty/Data_500_stocks_dirty.csv
data/dirty/stock_*.csv
data/clean/Data_500_stocks_clean_ver2.csv
data/clean/features_all.csv
data/clean/symbol_sector_encoding.csv
data/clean/sector_label_encoding.csv
```

Log:

```text
data/khaosatdata/
data/clean_log/
data/pipeline_run_summary.json
```

Model artifacts:

```text
models/model1/models/
models/model2/saved_models/
models/model3/models/
models/model4/models/
models/model5/models/
```

Model outputs:

```text
models/model1/reports/
models/model3/reports/
models/model4/output/
models/model5/output_model5/
```

## Ghi Chú Vận Hành

- Workflow `cao_dl` chỉ chịu trách nhiệm crawl CSV ngày mới và commit file `data/dirty/stock_*.csv`.
- Workflow `merge_daily_csv_and_run_pipeline` chạy sau `cao_dl`, thực hiện toàn bộ pipeline qua `python main.py`.
- Nếu một ngày không có phiên giao dịch, crawler vẫn tạo file `no_data`; merge script sẽ bỏ qua file đó.
- `features_all.py` upload ClickHouse theo chunk để tránh lỗi bộ nhớ khi insert bảng lớn.
- `main.py` ghi `data/pipeline_run_summary.json` sau từng step để hỗ trợ debug và Telegram alert.
- Nếu bấm `Run workflow` bị lỗi queue, hãy kiểm tra workflow đã nằm trên nhánh mặc định, Actions có quyền `Read and write permissions`, và YAML đã được push lên GitHub.

## Quy Trình Chuẩn Hằng Ngày

```text
15:00 giờ Việt Nam
    workflow cao_dl chạy
        -> crawl dữ liệu ngày mục tiêu
        -> lưu data/dirty/stock_*.csv
        -> Telegram báo bắt đầu/kết thúc

sau khi cao_dl success
    workflow merge_daily_csv_and_run_pipeline chạy
        -> merge stock_*.csv vào Data_500_stocks_dirty.csv
        -> khảo sát dữ liệu
        -> làm sạch ra Data_500_stocks_clean_ver2.csv
        -> upload stock_prices, stock_symbols, features_all
        -> train model1, model2, model3, model4, model5
        -> upload output model4/model5
        -> ghi pipeline_run_summary.json
        -> Telegram báo kết quả cuối
```
