from pathlib import Path

import pandas as pd

from clickhouse_client import get_clickhouse_client

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "ingestion" / "company_infor.csv"
SYMBOL_ENCODING_CSV_PATH = PROJECT_DIR / "data" / "clean" / "symbol_sector_encoding.csv"
SECTOR_ENCODING_CSV_PATH = PROJECT_DIR / "data" / "clean" / "sector_label_encoding.csv"

SOURCE_COLUMNS = ["symbol", "company_name", "sector", "listed_date"]
TABLE_COLUMNS = ["symbol", "company_name", "sector", "encode_sector", "listed_date"]
SYMBOL_ENCODING_COLUMNS = ["symbol", "encode_sector"]

client = get_clickhouse_client()

client.command("""
CREATE DATABASE IF NOT EXISTS stock
""")

client.command("DROP TABLE IF EXISTS stock.stock_symbols")

client.command("""
CREATE TABLE stock.stock_symbols
(
    symbol String,
    company_name String,
    sector Nullable(String),
    encode_sector Nullable(Int32),
    listed_date Nullable(DateTime)
)
ENGINE = MergeTree
ORDER BY symbol
""")

client.command("DROP TABLE IF EXISTS stock.symbol_sector_encoding")

client.command("""
CREATE TABLE stock.symbol_sector_encoding
(
    symbol String,
    encode_sector Nullable(Int32)
)
ENGINE = MergeTree
ORDER BY symbol
""")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

missing_columns = [column for column in SOURCE_COLUMNS if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in {CSV_PATH}: {missing_columns}")

df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
df["company_name"] = df["company_name"].astype(str).str.strip()
df["sector"] = df["sector"].astype("string").str.strip()
df["sector"] = df["sector"].replace(
    {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA}
)
df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")

sector_classes = df["sector"].dropna().astype(str).drop_duplicates().tolist()
sector_to_code = {sector: code for code, sector in enumerate(sector_classes)}
df["encode_sector"] = df["sector"].map(sector_to_code).astype("Int64")

df = df[TABLE_COLUMNS].dropna(subset=["symbol"])
df = df[df["symbol"] != ""]
df = df.drop_duplicates(subset=["symbol"], keep="last")

symbol_encoding_df = df[SYMBOL_ENCODING_COLUMNS].copy()
sector_encoding_df = pd.DataFrame(
    {
        "sector": list(sector_to_code.keys()),
        "encode_sector": list(sector_to_code.values()),
    }
)

SYMBOL_ENCODING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
symbol_encoding_df.to_csv(SYMBOL_ENCODING_CSV_PATH, index=False)
sector_encoding_df.to_csv(SECTOR_ENCODING_CSV_PATH, index=False)

df["encode_sector"] = df["encode_sector"].astype(object).where(
    df["encode_sector"].notna(),
    None,
)
symbol_encoding_df["encode_sector"] = symbol_encoding_df["encode_sector"].astype(
    object
).where(symbol_encoding_df["encode_sector"].notna(), None)

df = df.where(pd.notna(df), None)
symbol_encoding_df = symbol_encoding_df.where(pd.notna(symbol_encoding_df), None)

client.insert_df(
    table="stock.stock_symbols",
    df=df
)

client.insert_df(
    table="stock.symbol_sector_encoding",
    df=symbol_encoding_df
)

print(f"Uploaded {len(df)} rows to stock.stock_symbols")
print(f"Uploaded {len(symbol_encoding_df)} rows to stock.symbol_sector_encoding")
print(f"Saved symbol encoding CSV: {SYMBOL_ENCODING_CSV_PATH}")
print(f"Saved sector label CSV: {SECTOR_ENCODING_CSV_PATH}")
