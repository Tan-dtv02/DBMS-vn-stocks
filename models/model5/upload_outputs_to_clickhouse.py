import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model5.clickhouse_client import get_clickhouse_client


DEFAULT_MODEL_DIR = MODEL_DIR / "output_model5"
DEFAULT_DATABASE = "stock_mart_model5_risk_prediction"


TABLE_CONFIGS = {
    "risk_features": {
        "file_name": "risk_features.csv",
        "order_by": "(symbol, trading_date)",
        "columns": [
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
            "future_return_5d",
            "risk_drop_label",
            "created_at",
        ],
        "schema": """
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
            future_return_5d Nullable(Float64),
            risk_drop_label Nullable(UInt8),
            created_at DateTime
        """,
        "date_columns": ["trading_date"],
        "datetime_columns": ["created_at"],
        "string_columns": ["symbol"],
        "integer_columns": ["encode_sector", "risk_drop_label"],
    },
    "risk_predictions": {
        "file_name": "risk_predictions.csv",
        "order_by": "(symbol, prediction_date, model_name)",
        "columns": [
            "prediction_date",
            "target_date",
            "symbol",
            "model_name",
            "risk_probability",
            "risk_label",
            "created_at",
        ],
        "schema": """
            prediction_date Date,
            target_date Date,
            symbol String,
            model_name String,
            risk_probability Nullable(Float64),
            risk_label String,
            created_at DateTime
        """,
        "date_columns": ["prediction_date", "target_date"],
        "datetime_columns": ["created_at"],
        "string_columns": ["symbol", "model_name", "risk_label"],
        "integer_columns": [],
    },
    "risk_test_evaluation": {
        "file_name": "risk_test_evaluation.csv",
        "order_by": "(symbol, prediction_date, model_name)",
        "columns": [
            "prediction_date",
            "target_date",
            "symbol",
            "model_name",
            "real_close_prediction_date",
            "real_close_target_date",
            "actual_future_return_5d",
            "actual_risk_drop_label",
            "actual_risk_label",
            "risk_probability",
            "predicted_risk_label",
            "prediction_correct",
            "percentage_accuracy",
            "created_at",
        ],
        "schema": """
            prediction_date Date,
            target_date Date,
            symbol String,
            model_name String,
            real_close_prediction_date Nullable(Float64),
            real_close_target_date Nullable(Float64),
            actual_future_return_5d Nullable(Float64),
            actual_risk_drop_label Nullable(UInt8),
            actual_risk_label String,
            risk_probability Nullable(Float64),
            predicted_risk_label String,
            prediction_correct UInt8,
            percentage_accuracy Nullable(Float64),
            created_at DateTime
        """,
        "date_columns": ["prediction_date", "target_date"],
        "datetime_columns": ["created_at"],
        "string_columns": [
            "symbol",
            "model_name",
            "actual_risk_label",
            "predicted_risk_label",
        ],
        "integer_columns": ["actual_risk_drop_label", "prediction_correct"],
    },
    "mart_risk_alerts": {
        "file_name": "mart_risk_alerts.csv",
        "order_by": "(symbol, prediction_date, model_name)",
        "columns": [
            "prediction_date",
            "target_date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "company_name",
            "return_5d",
            "drawdown_20d",
            "volatility_5d",
            "volume_ratio_5_20",
            "daily_range",
            "close_position",
            "risk_probability",
            "risk_label",
            "model_name",
        ],
        "schema": """
            prediction_date Date,
            target_date Date,
            symbol String,
            open Nullable(Float64),
            high Nullable(Float64),
            low Nullable(Float64),
            close Nullable(Float64),
            volume Nullable(Float64),
            company_name Nullable(String),
            return_5d Nullable(Float64),
            drawdown_20d Nullable(Float64),
            volatility_5d Nullable(Float64),
            volume_ratio_5_20 Nullable(Float64),
            daily_range Nullable(Float64),
            close_position Nullable(Float64),
            risk_probability Nullable(Float64),
            risk_label String,
            model_name String
        """,
        "date_columns": ["prediction_date", "target_date"],
        "datetime_columns": [],
        "string_columns": ["symbol", "company_name", "risk_label", "model_name"],
        "integer_columns": [],
    },
}


def quote_identifier(name):
    return "`" + str(name).replace("`", "``") + "`"


def full_table_name(database, table_name):
    return f"{quote_identifier(database)}.{quote_identifier(table_name)}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload model CSV outputs to ClickHouse."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Folder containing model CSV files.",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help="Target ClickHouse database.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Number of CSV rows inserted per batch.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows instead of truncating existing target tables.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads without connecting to ClickHouse.",
    )
    return parser.parse_args()


def create_database(client, database):
    client.command(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(database)}")
    print(f"[clickhouse] Database ready: {database}")


def create_table_with_schema(client, database, table_name, schema, order_by):
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {full_table_name(database, table_name)}
        (
            {schema}
        )
        ENGINE = MergeTree
        ORDER BY {order_by}
        """
    )
    print(f"[clickhouse] Table ready: {database}.{table_name}")


def create_table(client, database, table_name, config):
    create_table_with_schema(
        client,
        database,
        table_name,
        config["schema"],
        config["order_by"],
    )


def parse_schema_columns(schema: str):
    columns = {}
    for raw_line in schema.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name, col_type = parts
        columns[name.strip("`")] = col_type.strip()
    return columns


def ensure_table_columns(client, database, table_name, config):
    desired_columns = parse_schema_columns(config["schema"])
    if not desired_columns:
        return

    existing = client.query_df(
        f"""
        SELECT name
        FROM system.columns
        WHERE database = '{database}' AND table = '{table_name}'
        """
    )
    existing_names = set(existing["name"].tolist()) if not existing.empty else set()

    for name, col_type in desired_columns.items():
        if name in existing_names:
            continue
        client.command(
            f"ALTER TABLE {full_table_name(database, table_name)} "
            f"ADD COLUMN IF NOT EXISTS {quote_identifier(name)} {col_type}"
        )
        print(f"[clickhouse] Added column: {database}.{table_name}.{name}")


def build_schema_from_csv_columns(csv_columns, config):
    type_map = parse_schema_columns(config["schema"])
    schema_lines = []
    for column in csv_columns:
        col_type = type_map.get(column)
        if not col_type:
            col_type = "Nullable(Float64)"
        schema_lines.append(f"{column} {col_type}")
    return ",\n            ".join(schema_lines)


def truncate_table(client, database, table_name):
    client.command(f"TRUNCATE TABLE {full_table_name(database, table_name)}")
    print(f"[clickhouse] Truncated table: {database}.{table_name}")


def normalize_string_column(series, nullable=False):
    normalized = series.astype("string").str.strip()
    if nullable:
        nullable_values = normalized.replace({"": pd.NA})
        return nullable_values.astype(object).where(
            nullable_values.notna(),
            None,
        )
    return normalized.fillna("").astype(str)


def normalize_integer_column(series):
    if series.dtype == bool:
        return series.astype(int)

    normalized = series.astype("string").str.strip().str.lower()
    normalized = normalized.replace(
        {
            "true": "1",
            "false": "0",
            "yes": "1",
            "no": "0",
            "": pd.NA,
            "nan": pd.NA,
            "none": pd.NA,
            "<na>": pd.NA,
        }
    )
    normalized = pd.to_numeric(normalized, errors="coerce").astype("Int64")
    return normalized.astype(object).where(normalized.notna(), None)


def prepare_chunk(chunk, config):
    for column in config["columns"]:
        if column not in chunk.columns:
            print(f"[upload] Missing column '{column}', filling NULL values.")
            chunk[column] = pd.NA

    chunk = chunk[config["columns"]].copy()
    chunk = chunk.replace([np.inf, -np.inf], np.nan)

    for column in config["date_columns"]:
        chunk[column] = pd.to_datetime(chunk[column], errors="coerce").dt.date

    for column in config["datetime_columns"]:
        chunk[column] = pd.to_datetime(chunk[column], errors="coerce")
        chunk[column] = chunk[column].fillna(pd.Timestamp.now().floor("s"))

    for column in config["string_columns"]:
        chunk[column] = normalize_string_column(
            chunk[column],
            nullable=(column == "company_name"),
        )

    for column in config["integer_columns"]:
        chunk[column] = normalize_integer_column(chunk[column])

    numeric_columns = [
        column
        for column in config["columns"]
        if column
        not in set(
            config["date_columns"]
            + config["datetime_columns"]
            + config["string_columns"]
            + config["integer_columns"]
        )
    ]
    for column in numeric_columns:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    return chunk.where(pd.notna(chunk), None)


def upload_csv_table(client, database, table_name, csv_path, config, chunksize):
    total_rows = 0
    print(f"[upload] Reading {csv_path}")

    for chunk_index, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize), 1):
        prepared_chunk = prepare_chunk(chunk, config)
        client.insert_df(
            table=f"{database}.{table_name}",
            df=prepared_chunk,
        )
        total_rows += len(prepared_chunk)
        print(
            f"[upload] {table_name}: inserted chunk {chunk_index} "
            f"({len(prepared_chunk):,} rows, total={total_rows:,})"
        )

    print(f"[upload] Finished {database}.{table_name}: {total_rows:,} rows")


def create_dashboard_view(client, database):
    client.command(
        f"""
        CREATE OR REPLACE VIEW {full_table_name(database, "dashboard_risk_alerts")} AS
        SELECT
            prediction_date,
            target_date,
            symbol,
            company_name,
            close,
            return_5d,
            drawdown_20d,
            volatility_5d,
            volume_ratio_5_20,
            daily_range,
            close_position,
            risk_probability,
            risk_label,
            model_name
        FROM {full_table_name(database, "mart_risk_alerts")}
        """
    )
    print(f"[clickhouse] View ready: {database}.dashboard_risk_alerts")


def print_dry_run(model_dir, database, append):
    print("[dry-run] No ClickHouse changes will be made.")
    print(f"[dry-run] Target database: {database}")
    print(f"[dry-run] Mode: {'append' if append else 'replace/truncate'}")
    for table_name, config in TABLE_CONFIGS.items():
        csv_path = model_dir / config["file_name"]
        if csv_path.exists():
            print(f"[dry-run] {csv_path} -> {database}.{table_name}")
        else:
            print(f"[dry-run] Missing CSV, skipped: {csv_path}")


def main():
    args = parse_args()
    model_dir = args.model_dir

    if args.dry_run:
        print_dry_run(model_dir, args.database, args.append)
        return

    client = get_clickhouse_client()
    if client is None:
        print("[clickhouse] Could not create ClickHouse client. Upload stopped.")
        return

    create_database(client, args.database)

    for table_name, config in TABLE_CONFIGS.items():
        csv_path = model_dir / config["file_name"]
        if not csv_path.exists():
            print(f"[upload] CSV not found, skipped: {csv_path}")
            continue

        if table_name == "risk_features":
            csv_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
            schema = build_schema_from_csv_columns(csv_columns, config)
            if not args.append:
                client.command(
                    f"DROP TABLE IF EXISTS {full_table_name(args.database, table_name)}"
                )
            create_table_with_schema(
                client,
                args.database,
                table_name,
                schema,
                config["order_by"],
            )
        else:
            create_table(client, args.database, table_name, config)
            ensure_table_columns(client, args.database, table_name, config)
        if not args.append:
            truncate_table(client, args.database, table_name)
        upload_csv_table(
            client=client,
            database=args.database,
            table_name=table_name,
            csv_path=csv_path,
            config=config,
            chunksize=args.chunksize,
        )

    create_dashboard_view(client, args.database)
    print("[upload] Done.")


if __name__ == "__main__":
    main()
