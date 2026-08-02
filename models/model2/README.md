# Model 2 - Future Return Prediction

## 1. Giới thiệu

Model 2 thực hiện bài toán dự đoán lợi suất cổ phiếu trong 5 ngày giao dịch tiếp theo (Future Return Prediction) bằng thuật toán LightGBM Regressor.

Đầu ra của mô hình là giá trị:

future_return_5d = (close_t+5 / close_t) - 1

Ví dụ:

- 0.05 → dự đoán tăng 5%
- -0.02 → dự đoán giảm 2%

---

## 2. Nguồn dữ liệu

Dữ liệu được thu thập từ thư viện vnstock và lưu trữ trên ClickHouse.

Các trường dữ liệu cơ bản:

- symbol
- trading_date
- open
- high
- low
- close
- volume
- encode_sector

Ngoài ra hệ thống sử dụng nhiều feature kỹ thuật như:

- return_1d
- return_3d
- return_5d
- return_10d
- return_20d
- ma_5
- ma_20
- ma_50
- price_vs_ma20
- ma5_vs_ma20
- volatility_5d
- volatility_20d
- volatility_change
- drawdown_20d
- volume_ratio_5_20
- daily_range
- body_ratio
- close_position

---

## 3. Thuật toán sử dụng

- LightGBM Regressor

Các chỉ số đánh giá:

- MAE
- RMSE
- R² Score
- MAPE

---

## 4. Cấu trúc thư mục

model2/

├── models/

├── reports/

├── src/

├── dashboard_model2.py

├── walk_forward.py

├── main.py

├── README.md

├── BAO_CAO_MO_HINH.txt

└── requirements.txt

---

## 5. Cài đặt

Tạo môi trường:

python -m venv venv

Kích hoạt:

Windows:

venv\Scripts\activate

Linux:

source venv/bin/activate

Cài thư viện:

pip install -r requirements.txt

---

## 6. Cấu hình

Tạo file .env:

CLICKHOUSE_HOST=...

CLICKHOUSE_PORT=8443

CLICKHOUSE_USER=default

CLICKHOUSE_PASSWORD=...

CLICKHOUSE_DATABASE=stock

CLICKHOUSE_TABLE=features_all

CLICKHOUSE_SECURE=true

---

## 7. Huấn luyện và đánh giá

Chạy:

python main.py

Pipeline:

1. Load dữ liệu
2. Feature Engineering
3. Train Model
4. Predict
5. Evaluate
6. Backtest
7. Create Data Mart
8. Generate Dashboard

---

## 8. Kết quả đầu ra

### Model

models/future_return_lgbm.pkl

### Reports

reports/metrics.json

reports/feature_importance.csv

reports/prediction_error.csv

reports/backtest.csv

reports/backtest_metrics.json

reports/backtest_sweep.csv

reports/model2_dashboard.html

### Data Mart

mart_future_return_prediction

---

## 9. Dashboard

Dashboard hiển thị:

- MAE
- RMSE
- R²
- MAPE
- Top Feature Importance
- Backtest Performance
- Cumulative Return

Dashboard được lưu tại:

reports/model2_dashboard.html

---

## 10. Tác giả

Nhóm xây dựng hệ thống phân tích và dự đoán cổ phiếu.
