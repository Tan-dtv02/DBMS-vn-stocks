from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MART_CSV = MODEL_DIR / "output_model5" / "mart_risk_alerts.csv"

MART_COLUMNS = [
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
]


def _normalize_symbol_frame(symbols_df: pd.DataFrame):
    symbols = symbols_df.copy()
    symbols.columns = [str(col).strip().lower() for col in symbols.columns]
    if "symbol" not in symbols.columns or "company_name" not in symbols.columns:
        return pd.DataFrame(columns=["symbol", "company_name"])
    symbols["symbol"] = symbols["symbol"].astype(str).str.strip().str.upper()
    return symbols[["symbol", "company_name"]].drop_duplicates("symbol")


def create_local_risk_mart(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_path: Path | str = DEFAULT_MART_CSV,
    symbols_df: pd.DataFrame | None = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_columns = [
        "prediction_date",
        "target_date",
        "symbol",
        "risk_probability",
        "risk_label",
        "model_name",
    ]
    feature_columns = [
        "trading_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_5d",
        "drawdown_20d",
        "volatility_5d",
        "volume_ratio_5_20",
        "daily_range",
        "close_position",
    ]

    if "volume" not in features_df.columns and "volumn" in features_df.columns:
        features_df = features_df.rename(columns={"volumn": "volume"})

    missing_prediction_columns = sorted(
        set(prediction_columns) - set(predictions_df.columns)
    )
    missing_feature_columns = sorted(set(feature_columns) - set(features_df.columns))
    if missing_prediction_columns:
        print(
            "[mart] Missing prediction columns, filling nulls: "
            f"{missing_prediction_columns}"
        )
        for column in missing_prediction_columns:
            predictions_df[column] = pd.NA
    if missing_feature_columns:
        print(f"[mart] Missing feature columns, filling nulls: {missing_feature_columns}")
        for column in missing_feature_columns:
            features_df[column] = pd.NA

    predictions = predictions_df[prediction_columns].copy()
    predictions["prediction_date"] = pd.to_datetime(
        predictions["prediction_date"]
    ).dt.normalize()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"]).dt.normalize()
    predictions["symbol"] = predictions["symbol"].astype(str).str.strip().str.upper()

    features = features_df[feature_columns].copy()
    features["prediction_date"] = pd.to_datetime(
        features["trading_date"]
    ).dt.normalize()
    features["symbol"] = features["symbol"].astype(str).str.strip().str.upper()
    features = features.drop(columns=["trading_date"])

    mart = predictions.merge(features, on=["prediction_date", "symbol"], how="left")

    if symbols_df is not None:
        symbols = _normalize_symbol_frame(symbols_df)
        mart = mart.merge(symbols, on="symbol", how="left")
    else:
        mart["company_name"] = pd.NA

    mart = mart[MART_COLUMNS].sort_values(
        ["prediction_date", "risk_probability", "symbol"],
        ascending=[False, False, True],
    )
    mart["prediction_date"] = pd.to_datetime(mart["prediction_date"]).dt.date
    mart["target_date"] = pd.to_datetime(mart["target_date"]).dt.date
    mart.to_csv(output_path, index=False)
    print(f"[mart] Saved local mart CSV: {output_path} ({len(mart):,} rows)")
    return output_path


def _clickhouse_table_exists(client: Any, database: str, table: str):
    query = f"""
        SELECT count() AS cnt
        FROM system.tables
        WHERE database = '{database}' AND name = '{table}'
    """
    result = client.query_df(query)
    return int(result.iloc[0]["cnt"]) > 0


def create_clickhouse_risk_mart(
    client: Any,
    database: str = "stock",
):
    """Future ClickHouse view creator. Not used by the current local CSV pipeline."""
    symbols_exists = _clickhouse_table_exists(client, database, "dw_stock_symbols")

    if symbols_exists:
        company_select = "s.company_name AS company_name"
        symbol_join = (
            f"LEFT JOIN {database}.dw_stock_symbols AS s ON p.symbol = s.symbol"
        )
    else:
        company_select = "CAST(NULL, 'Nullable(String)') AS company_name"
        symbol_join = ""

    client.command(
        f"""
        CREATE OR REPLACE VIEW {database}.mart_risk_alerts AS
        SELECT
            p.prediction_date AS prediction_date,
            p.target_date AS target_date,
            p.symbol AS symbol,
            f.open AS open,
            f.high AS high,
            f.low AS low,
            f.close AS close,
            f.volume AS volume,
            {company_select},
            f.return_5d AS return_5d,
            f.drawdown_20d AS drawdown_20d,
            f.volatility_5d AS volatility_5d,
            f.volume_ratio_5_20 AS volume_ratio_5_20,
            f.daily_range AS daily_range,
            f.close_position AS close_position,
            p.risk_probability AS risk_probability,
            p.risk_label AS risk_label,
            p.model_name AS model_name
        FROM {database}.dw_stock_risk_predictions AS p
        LEFT JOIN {database}.dw_stock_risk_features AS f
            ON p.symbol = f.symbol
            AND p.prediction_date = f.trading_date
        {symbol_join}
        """
    )
    print(f"[mart] Created ClickHouse view: {database}.mart_risk_alerts")
