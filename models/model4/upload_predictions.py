# ==============================================
# MODULE 4 — BENCHMARK OUTPERFORMANCE MODEL
# File: upload_predictions.py
# Mục đích: Upload kết quả dự đoán lên ClickHouse
# ==============================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import clickhouse_connect

load_dotenv()

# ==================
# CONFIG
# ==================
MODEL_DIR = Path(__file__).resolve().parent
PREDICTIONS_CSV = MODEL_DIR / "output" / "benchmark_predictions.csv"
DATABASE        = "stock"
TABLE           = "model4_benchmark_predictions"

# ==================
# KẾT NỐI
# ==================
def get_client():
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8443")),
        username=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        database=os.getenv("CLICKHOUSE_DATABASE", "default"),
        secure=True,
    )

# ==================
# TẠO BẢNG
# ==================
def create_table(client):
    """Tạo bảng trong ClickHouse nếu chưa có"""
    client.command(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    client.command(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{TABLE}
        (
            symbol                  String,
            trading_date            DateTime,
            close                   Float64,
            label                   Int8,
            predicted_label         Int8,
            outperform_probability  Float64,
            prediction_correct      UInt8,
            created_at              DateTime
        )
        ENGINE = MergeTree
        ORDER BY (symbol, trading_date)
    """)
    print(f"[model4] Bảng {DATABASE}.{TABLE} sẵn sàng!")

# ==================
# UPLOAD
# ==================
def upload_predictions(client):
    """Đọc CSV và upload lên ClickHouse"""

    print("[model4] Đọc file predictions...")
    df = pd.read_csv(PREDICTIONS_CSV)

    # Chuẩn hóa dữ liệu
    df["trading_date"] = pd.to_datetime(
        df["trading_date"], errors="coerce"
    )
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
    df["outperform_probability"] = pd.to_numeric(
        df["outperform_probability"], errors="coerce"
    )
    df["label"]             = df["label"].astype(int)
    df["predicted_label"]   = df["predicted_label"].astype(int)
    df["prediction_correct"] = df["prediction_correct"].astype(int)
    df["created_at"]        = pd.Timestamp.now().floor("s")

    # Xóa dòng thiếu key
    df = df.dropna(subset=["symbol", "trading_date", "close"])

    # Chọn đúng cột theo thứ tự bảng
    df = df[[
        "symbol",
        "trading_date",
        "close",
        "label",
        "predicted_label",
        "outperform_probability",
        "prediction_correct",
        "created_at",
    ]]

    print(f"[model4] Chuẩn bị upload {len(df):,} dòng...")

    # Xóa dữ liệu cũ trước khi upload
    client.command(
        f"TRUNCATE TABLE IF EXISTS {DATABASE}.{TABLE}"
    )
    print(f"[model4] Đã xóa dữ liệu cũ!")

    # Upload lên ClickHouse
    client.insert_df(
        table=f"{DATABASE}.{TABLE}",
        df=df
    )
    print(f"[model4] Upload thành công "
          f"{len(df):,} dòng lên {DATABASE}.{TABLE}!")

# ==================
# KIỂM TRA
# ==================
def verify_upload(client):
    """Kiểm tra dữ liệu đã upload"""
    result = client.query_df(f"""
        SELECT
            COUNT(*)           AS total_rows,
            COUNT(DISTINCT symbol) AS total_symbols,
            MIN(trading_date)  AS date_from,
            MAX(trading_date)  AS date_to,
            AVG(outperform_probability) AS avg_probability,
            SUM(prediction_correct) * 100.0 / COUNT(*) AS accuracy_pct
        FROM {DATABASE}.{TABLE}
    """)
    print("\n[model4] Kết quả kiểm tra trên ClickHouse:")
    print(result.to_string(index=False))

# ==================
# MAIN
# ==================
if __name__ == "__main__":
    # Bước 1: Kết nối
    print("[model4] Kết nối ClickHouse...")
    client = get_client()
    print("[model4] Kết nối thành công!")

    # Bước 2: Tạo bảng
    create_table(client)

    # Bước 3: Upload
    upload_predictions(client)

    # Bước 4: Kiểm tra
    verify_upload(client)

    print("\n[model4] HOÀN THÀNH! ✅")
