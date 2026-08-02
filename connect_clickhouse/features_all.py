import argparse
import gc
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Bây giờ lệnh import này sẽ chạy thành công 100% vì hàm đã tồn tại
from clickhouse_client import get_clickhouse_client

# Ở đoạn code phía dưới của file, chỗ nào cần dùng client thì gọi hàm:
# client = get_clickhouse_client()

OUTPUT_CSV = PROJECT_DIR / "data" / "clean" / "features_all.csv"
SYMBOL_ENCODING_CSV = PROJECT_DIR / "data" / "clean" / "symbol_sector_encoding.csv"
SOURCE_DATABASE = "stock"
SOURCE_TABLE = "stock_prices"
SYMBOL_ENCODING_DATABASE = "stock"
SYMBOL_ENCODING_TABLE = "symbol_sector_encoding"
TARGET_DATABASE = "stock"
TARGET_TABLE = "features_all"
DEFAULT_EXPORT_CHUNKSIZE = 1_000
DEFAULT_INSERT_CHUNKSIZE = 10_000

FEATURES_ALL_COLUMNS = [
    "trading_date",
    "symbol",
    "encode_sector",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma_5",
    "ma_20",
    "ma_50",
    "price_vs_ma20",
    "ma5_vs_ma20",
    "volatility_5d",
    "volatility_20d",
    "volatility_change",
    "rolling_max_20d",
    "drawdown_20d",
    "volume_ma_5",
    "volume_ma_20",
    "volume_ratio_5_20",
    "volume_change_1d",
    "daily_range",
    "body_ratio",
    "close_position",
    "created_at",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma_5",
    "ma_20",
    "ma_50",
    "price_vs_ma20",
    "ma5_vs_ma20",
    "volatility_5d",
    "volatility_20d",
    "volatility_change",
    "rolling_max_20d",
    "drawdown_20d",
    "volume_ma_5",
    "volume_ma_20",
    "volume_ratio_5_20",
    "volume_change_1d",
    "daily_range",
    "body_ratio",
    "close_position",
]
INTEGER_COLUMNS = ["encode_sector"]


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def create_features_all_table(client, database: str = "stock") -> None:
    client.command(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(database)}")
    client.command(
        f"DROP TABLE IF EXISTS {quote_identifier(database)}.{quote_identifier(TARGET_TABLE)}"
    )
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {quote_identifier(database)}.{quote_identifier(TARGET_TABLE)}
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
        ORDER BY (symbol, trading_date)
        """
    )
    print(f"[clickhouse] Recreated table: {database}.{TARGET_TABLE}")


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["trading_date", "symbol", "open", "high", "low", "close", "volume"]
        )

    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    if "trading_date" not in normalized.columns and "date" in normalized.columns:
        normalized = normalized.rename(columns={"date": "trading_date"})

    required_columns = {"trading_date", "symbol", "open", "high", "low", "close", "volume"}
    missing_columns = sorted(required_columns - set(normalized.columns))
    if missing_columns:
        raise ValueError(f"Missing stock price columns: {missing_columns}")

    normalized["symbol"] = normalized["symbol"].astype("string").str.strip().str.upper()
    normalized["trading_date"] = pd.to_datetime(
        normalized["trading_date"], errors="coerce"
    ).dt.normalize()

    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(
        subset=["trading_date", "symbol", "open", "high", "low", "close", "volume"]
    )
    normalized = normalized[normalized["symbol"] != ""]
    normalized = normalized.drop_duplicates(
        subset=["symbol", "trading_date"], keep="last"
    )
    return normalized.sort_values(["symbol", "trading_date"]).reset_index(drop=True)


def load_stock_prices(
    client,
    database: str = SOURCE_DATABASE,
    table: str = SOURCE_TABLE,
) -> pd.DataFrame:
    query = f"""
        SELECT
            date AS trading_date,
            symbol,
            open,
            high,
            low,
            close,
            volume
        FROM {quote_identifier(database)}.{quote_identifier(table)}
        ORDER BY symbol, date
    """
    print(f"[clickhouse] Loading prices from {database}.{table}")
    prices = normalize_prices(client.query_df(query))
    print(
        "[features_all] Loaded prices "
        f"{len(prices):,} rows, "
        f"{prices['symbol'].nunique():,} symbols"
    )
    return prices


def normalize_symbol_sector_encoding(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "encode_sector"])

    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    missing_columns = {"symbol", "encode_sector"} - set(normalized.columns)
    if missing_columns:
        print(
            "[features_all] Missing symbol sector encoding columns: "
            f"{sorted(missing_columns)}"
        )
        return pd.DataFrame(columns=["symbol", "encode_sector"])

    normalized["symbol"] = normalized["symbol"].astype("string").str.strip().str.upper()
    normalized["encode_sector"] = pd.to_numeric(
        normalized["encode_sector"], errors="coerce"
    ).astype("Int64")
    normalized = normalized.dropna(subset=["symbol"])
    normalized = normalized[normalized["symbol"] != ""]
    normalized = normalized.drop_duplicates(subset=["symbol"], keep="last")
    return normalized[["symbol", "encode_sector"]].reset_index(drop=True)


def load_symbol_sector_encoding(
    client,
    database: str = SYMBOL_ENCODING_DATABASE,
    table: str = SYMBOL_ENCODING_TABLE,
) -> pd.DataFrame:
    query = f"""
        SELECT
            symbol,
            encode_sector
        FROM {quote_identifier(database)}.{quote_identifier(table)}
        ORDER BY symbol
    """
    print(f"[features_all] Loading sector encoding from {database}.{table}")
    try:
        encoding = normalize_symbol_sector_encoding(client.query_df(query))
    except Exception as exc:
        print(f"[features_all] Could not load sector encoding table: {exc}")
        return pd.DataFrame(columns=["symbol", "encode_sector"])

    print(f"[features_all] Loaded sector encodings: {len(encoding):,} symbols")
    return encoding


def load_symbol_sector_encoding_csv(
    csv_path: Path = SYMBOL_ENCODING_CSV,
) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[features_all] Sector encoding CSV not found: {csv_path}")
        return pd.DataFrame(columns=["symbol", "encode_sector"])
    return normalize_symbol_sector_encoding(pd.read_csv(csv_path))


def merge_symbol_sector_encoding(
    features_df: pd.DataFrame,
    symbol_encoding_df: pd.DataFrame,
) -> pd.DataFrame:
    features = features_df.drop(columns=["encode_sector"], errors="ignore").copy()
    encoding = normalize_symbol_sector_encoding(symbol_encoding_df)

    if encoding.empty:
        features["encode_sector"] = pd.NA
        print("[features_all] No sector encoding available; encode_sector is NULL.")
        return features

    merged = features.merge(encoding, on="symbol", how="left")
    missing_count = int(merged["encode_sector"].isna().sum())
    if missing_count:
        print(f"[features_all] Rows without encode_sector after join: {missing_count:,}")
    else:
        print("[features_all] Joined encode_sector for all feature rows.")
    return merged


def build_features_all(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_prices(prices_df)
    if df.empty:
        print("[features_all] No stock price rows available.")
        for column in FEATURES_ALL_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        return df[FEATURES_ALL_COLUMNS]

    group = df.groupby("symbol", group_keys=False)

    for window in [1, 3, 5, 10, 20]:
        df[f"return_{window}d"] = group["close"].pct_change(window)

    for window in [5, 20, 50]:
        df[f"ma_{window}"] = group["close"].transform(
            lambda series, w=window: series.rolling(window=w, min_periods=w).mean()
        )

    df["price_vs_ma20"] = _safe_divide(df["close"], df["ma_20"]) - 1
    df["ma5_vs_ma20"] = _safe_divide(df["ma_5"], df["ma_20"]) - 1

    df["volatility_5d"] = group["return_1d"].transform(
        lambda series: series.rolling(window=5, min_periods=5).std()
    )
    df["volatility_20d"] = group["return_1d"].transform(
        lambda series: series.rolling(window=20, min_periods=20).std()
    )
    df["volatility_change"] = _safe_divide(
        df["volatility_5d"], df["volatility_20d"]
    ) - 1

    df["rolling_max_20d"] = group["close"].transform(
        lambda series: series.rolling(window=20, min_periods=20).max()
    )
    df["drawdown_20d"] = _safe_divide(df["close"], df["rolling_max_20d"]) - 1

    df["volume_ma_5"] = group["volume"].transform(
        lambda series: series.rolling(window=5, min_periods=5).mean()
    )
    df["volume_ma_20"] = group["volume"].transform(
        lambda series: series.rolling(window=20, min_periods=20).mean()
    )
    df["volume_ratio_5_20"] = _safe_divide(df["volume_ma_5"], df["volume_ma_20"])
    df["volume_change_1d"] = group["volume"].pct_change(1)

    high_low_range = df["high"] - df["low"]
    df["daily_range"] = _safe_divide(high_low_range, df["close"])
    df["body_ratio"] = np.where(
        high_low_range.ne(0),
        (df["close"] - df["open"]).abs() / high_low_range,
        0.0,
    )
    df["close_position"] = np.where(
        high_low_range.ne(0),
        (df["close"] - df["low"]) / high_low_range,
        0.5,
    )
    df["created_at"] = pd.Timestamp.now().floor("s")
    df["encode_sector"] = pd.NA

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[FEATURES_ALL_COLUMNS].copy()
    print(f"[features_all] Built common features: {len(df):,} rows")
    return df


def prepare_features_all(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    for column in FEATURES_ALL_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA

    prepared = prepared[FEATURES_ALL_COLUMNS].copy()

    prepared["symbol"] = prepared["symbol"].astype("string").str.strip().str.upper()
    prepared["trading_date"] = pd.to_datetime(
        prepared["trading_date"], errors="coerce"
    ).dt.date
    prepared["created_at"] = pd.to_datetime(prepared["created_at"], errors="coerce")
    prepared["created_at"] = prepared["created_at"].fillna(
        pd.Timestamp.now().floor("s")
    )

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(prepared[column], errors="coerce").astype("Int64")
        prepared[column] = values.astype(object).where(values.notna(), None)

    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    return prepared.where(pd.notna(prepared), None)


def order_features_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in FEATURES_ALL_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    return output[FEATURES_ALL_COLUMNS]


def upload_features_all_in_chunks(
    client,
    df: pd.DataFrame,
    table: str,
    chunksize: int = DEFAULT_INSERT_CHUNKSIZE,
) -> None:
    total_rows = len(df)
    if total_rows == 0:
        print(f"[clickhouse] No rows to upload to {table}.")
        return

    chunksize = max(1, int(chunksize))
    for start in range(0, total_rows, chunksize):
        end = min(start + chunksize, total_rows)
        chunk = order_features_all_columns(df.iloc[start:end]).copy()
        client.insert_df(table=table, df=chunk)
        print(f"[clickhouse] Uploaded rows {start + 1:,}-{end:,}/{total_rows:,}")
        del chunk
        gc.collect()


def write_features_all_csv(
    df: pd.DataFrame,
    output_path: Path,
    chunksize: int = DEFAULT_EXPORT_CHUNKSIZE,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    temp_path.write_text(",".join(FEATURES_ALL_COLUMNS) + "\n", encoding="utf-8")
    total_rows = len(df)
    if total_rows == 0:
        temp_path.replace(output_path)
        return output_path

    for start in range(0, total_rows, chunksize):
        chunk = order_features_all_columns(df.iloc[start : start + chunksize])
        chunk.to_csv(
            temp_path,
            mode="a",
            index=False,
            header=False,
        )

    temp_path.replace(output_path)
    return output_path


def export_features_all_to_csv(
    client,
    output_path: Path = OUTPUT_CSV,
    database: str = TARGET_DATABASE,
    table: str = TARGET_TABLE,
    source_df: pd.DataFrame | None = None,
    chunksize: int = DEFAULT_EXPORT_CHUNKSIZE,
) -> Path:
    if source_df is not None:
        write_features_all_csv(source_df, output_path, chunksize=chunksize)
        print(f"[clickhouse] Exported {len(source_df):,} rows to {output_path}")
        return output_path

    columns_sql = ",\n            ".join(FEATURES_ALL_COLUMNS)
    query = f"""
        SELECT
            {columns_sql}
        FROM {quote_identifier(database)}.{quote_identifier(table)}
        ORDER BY symbol, trading_date
    """

    df = client.query_df(query)
    write_features_all_csv(df, output_path, chunksize=chunksize)
    print(f"[clickhouse] Exported {len(df):,} rows to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/upload stock.features_all and export it to CSV."
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export stock.features_all to local CSV; do not rebuild/upload.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Upload stock.features_all but skip local CSV export.",
    )
    parser.add_argument(
        "--export-chunksize",
        type=int,
        default=DEFAULT_EXPORT_CHUNKSIZE,
        help="Rows per CSV export chunk.",
    )
    parser.add_argument(
        "--insert-chunksize",
        type=int,
        default=DEFAULT_INSERT_CHUNKSIZE,
        help="Rows per ClickHouse insert chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = get_clickhouse_client()
    if client is None:
        print("[clickhouse] Could not create ClickHouse client. Upload/export stopped.")
        return

    if args.export_only:
        export_features_all_to_csv(client, chunksize=args.export_chunksize)
        return

    prices_df = load_stock_prices(client)
    symbol_encoding_df = load_symbol_sector_encoding(client)
    features_df = build_features_all(prices_df)
    features_df = merge_symbol_sector_encoding(features_df, symbol_encoding_df)
    df = prepare_features_all(features_df)

    create_features_all_table(client, database=TARGET_DATABASE)
    upload_features_all_in_chunks(
        client=client,
        df=df,
        table=f"{TARGET_DATABASE}.{TARGET_TABLE}",
        chunksize=args.insert_chunksize,
    )
    print(f"Uploaded {len(df):,} rows to {TARGET_DATABASE}.{TARGET_TABLE}")

    if not args.skip_export:
        export_features_all_to_csv(
            client,
            source_df=df,
            chunksize=args.export_chunksize,
        )


if __name__ == "__main__":
    main()
