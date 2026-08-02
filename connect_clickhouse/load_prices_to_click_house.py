from pathlib import Path

import pandas as pd

from clickhouse_client import get_clickhouse_client

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Prefer ver2, fall back to cleaned if present.
PRICE_CSV_CANDIDATES = [
    PROJECT_DIR / "data" / "clean" / "Data_500_stocks_clean_ver2.csv",
    PROJECT_DIR / "data" / "clean" / "Data_500_stocks_cleaned.csv",
]

CLEAN_PRICE_CSV = next((p for p in PRICE_CSV_CANDIDATES if p.exists()), None)
if CLEAN_PRICE_CSV is None:
    candidate_list = "\n".join(str(p) for p in PRICE_CSV_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a clean price CSV. Tried:\n" + candidate_list
    )
client = get_clickhouse_client()

client.command("""
CREATE DATABASE IF NOT EXISTS stock
""")

client.command("""
CREATE TABLE IF NOT EXISTS stock.stock_prices
(
    symbol String,
    date DateTime,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64
)
ENGINE = MergeTree
ORDER BY (symbol, date)
""")

client.command("TRUNCATE TABLE stock.stock_prices")
print("[clickhouse] Truncated table: stock.stock_prices")

df = pd.read_csv(CLEAN_PRICE_CSV)

df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["symbol", "date"])

client.insert_df(
    table="stock.stock_prices",
    df=df
)

print(f"Uploaded {len(df)} rows to stock.stock_prices")
