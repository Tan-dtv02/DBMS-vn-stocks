from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS_CSV = MODEL_DIR / "output_model5" / "risk_predictions.csv"
DEFAULT_TEST_EVALUATION_CSV = MODEL_DIR / "output_model5" / "risk_test_evaluation.csv"

PREDICTION_COLUMNS = [
    "prediction_date",
    "target_date",
    "symbol",
    "model_name",
    "risk_probability",
    "risk_label",
    "created_at",
]

TEST_EVALUATION_COLUMNS = [
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
]


def prepare_prediction_df(prediction_df: pd.DataFrame):
    output = prediction_df.copy()
    if "prediction_date" not in output.columns and "trading_date" in output.columns:
        output["prediction_date"] = output["trading_date"]

    missing_columns = sorted(set(PREDICTION_COLUMNS) - set(output.columns))
    if missing_columns:
        print(f"[predictions] Missing prediction columns, filling nulls: {missing_columns}")
        for column in missing_columns:
            output[column] = pd.NA

    output = output.dropna(subset=["prediction_date", "target_date", "risk_probability"])
    output = output[PREDICTION_COLUMNS].copy()
    output["prediction_date"] = pd.to_datetime(output["prediction_date"]).dt.date
    output["target_date"] = pd.to_datetime(output["target_date"]).dt.date
    output["symbol"] = output["symbol"].astype(str).str.strip().str.upper()
    output["risk_probability"] = pd.to_numeric(
        output["risk_probability"], errors="coerce"
    ).astype(float)
    output = output.dropna(subset=["risk_probability"])
    output["created_at"] = pd.to_datetime(output["created_at"])
    return output.sort_values(
        ["prediction_date", "risk_probability", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def save_predictions_csv(
    prediction_df: pd.DataFrame,
    output_path: Path | str = DEFAULT_PREDICTIONS_CSV,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = prepare_prediction_df(prediction_df)
    output.to_csv(output_path, index=False)
    print(
        "[predictions] Saved prediction CSV: "
        f"{output_path} ({len(output):,} rows)"
    )
    return output_path


def prepare_test_evaluation_df(prediction_df: pd.DataFrame):
    output = prediction_df.copy()
    if "prediction_date" not in output.columns and "trading_date" in output.columns:
        output["prediction_date"] = output["trading_date"]

    source_columns = [
        "prediction_date",
        "target_date",
        "symbol",
        "model_name",
        "close",
        "future_close_5d",
        "future_return_5d",
        "risk_drop_label",
        "risk_probability",
        "risk_label",
        "created_at",
    ]
    missing_columns = sorted(set(source_columns) - set(output.columns))
    if missing_columns:
        print(
            "[predictions] Missing test evaluation columns, filling nulls: "
            f"{missing_columns}"
        )
        for column in missing_columns:
            output[column] = pd.NA

    output = output.dropna(
        subset=[
            "prediction_date",
            "target_date",
            "close",
            "future_close_5d",
            "risk_drop_label",
            "risk_probability",
            "risk_label",
        ]
    )

    evaluation = pd.DataFrame()
    evaluation["prediction_date"] = pd.to_datetime(
        output["prediction_date"]
    ).dt.date
    evaluation["target_date"] = pd.to_datetime(output["target_date"]).dt.date
    evaluation["symbol"] = output["symbol"].astype(str).str.strip().str.upper()
    evaluation["model_name"] = output["model_name"]
    evaluation["real_close_prediction_date"] = pd.to_numeric(
        output["close"], errors="coerce"
    )
    evaluation["real_close_target_date"] = pd.to_numeric(
        output["future_close_5d"], errors="coerce"
    )
    evaluation["actual_future_return_5d"] = pd.to_numeric(
        output["future_return_5d"], errors="coerce"
    )
    evaluation["actual_risk_drop_label"] = pd.to_numeric(
        output["risk_drop_label"], errors="coerce"
    ).astype("Int64")
    evaluation["actual_risk_label"] = evaluation["actual_risk_drop_label"].map(
        {1: "HIGH_RISK", 0: "LOW_RISK"}
    )
    evaluation["risk_probability"] = pd.to_numeric(
        output["risk_probability"], errors="coerce"
    )
    evaluation["predicted_risk_label"] = output["risk_label"]
    evaluation["prediction_correct"] = (
        evaluation["actual_risk_label"] == evaluation["predicted_risk_label"]
    )
    evaluation["percentage_accuracy"] = evaluation["prediction_correct"].map(
        {True: 100.0, False: 0.0}
    )
    evaluation["created_at"] = pd.to_datetime(output["created_at"])

    evaluation = evaluation.dropna(
        subset=[
            "real_close_prediction_date",
            "real_close_target_date",
            "actual_risk_drop_label",
            "risk_probability",
        ]
    )
    return evaluation[TEST_EVALUATION_COLUMNS].sort_values(
        ["prediction_date", "risk_probability", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def save_test_evaluation_csv(
    prediction_df: pd.DataFrame,
    output_path: Path | str = DEFAULT_TEST_EVALUATION_CSV,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = prepare_test_evaluation_df(prediction_df)
    output.to_csv(output_path, index=False)
    print(
        "[predictions] Saved test evaluation CSV: "
        f"{output_path} ({len(output):,} rows)"
    )
    return output_path


def create_clickhouse_prediction_table(
    client: Any,
    database: str = "stock",
    table: str = "dw_stock_risk_predictions",
):
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {database}.{table}
        (
            prediction_date Date,
            target_date Date,
            symbol String,
            model_name String,
            risk_probability Float64,
            risk_label String,
            created_at DateTime
        )
        ENGINE = MergeTree
        ORDER BY (symbol, prediction_date, model_name)
        """
    )


def save_predictions_to_clickhouse(
    client: Any,
    prediction_df: pd.DataFrame,
    database: str = "stock",
    table: str = "dw_stock_risk_predictions",
):
    """Future ClickHouse writer. Not used by the current local CSV pipeline."""
    create_clickhouse_prediction_table(client, database=database, table=table)
    output = prepare_prediction_df(prediction_df)
    client.insert_df(table=f"{database}.{table}", df=output)
    print(f"[predictions] Inserted {len(output):,} rows into {database}.{table}")
