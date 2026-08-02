from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model5.create_risk_mart import create_local_risk_mart
from models.model5.clickhouse_client import get_clickhouse_client
from models.model5.risk_features import (
    DEFAULT_FEATURES_ALL_DATABASE,
    DEFAULT_FEATURES_ALL_TABLE,
    create_risk_features,
    load_features_all,
    save_risk_features_csv,
)
from models.model5.save_risk_predictions import (
    prepare_prediction_df,
    save_predictions_csv,
    save_test_evaluation_csv,
)
from models.model5.train_risk_model import save_metrics_json, save_model, train_models


DEFAULT_OUTPUT_DIR = MODEL_DIR / "output_model5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model 5 risk-alert pipeline from ClickHouse features_all."
    )
    parser.add_argument(
        "--features-database",
        default=DEFAULT_FEATURES_ALL_DATABASE,
        help="ClickHouse database containing the common features_all table.",
    )
    parser.add_argument(
        "--features-table",
        default=DEFAULT_FEATURES_ALL_TABLE,
        help="ClickHouse table containing common model features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV/JSON outputs.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Time-based train ratio. No shuffle is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="HIGH_RISK threshold for class-1 probability.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[pipeline] Starting model 5 risk-alert pipeline")
    print(f"[pipeline] Output dir: {output_dir}")

    print(
        "[pipeline] Source: ClickHouse "
        f"{args.features_database}.{args.features_table}"
    )
    client = get_clickhouse_client()
    if client is None:
        raise SystemExit("[pipeline] Could not create ClickHouse client.")
    features_all_df = load_features_all(
        client=client,
        database=args.features_database,
        table=args.features_table,
    )

    features_df = create_risk_features(features_all_df)

    features_path = save_risk_features_csv(
        features_df,
        output_dir / "risk_features.csv",
    )

    model, prediction_df, metrics = train_models(
        features_df,
        train_ratio=args.train_ratio,
        threshold=args.threshold,
    )
    model_path = save_model(model, metrics)

    predictions_path = save_predictions_csv(
        prediction_df,
        output_dir / "risk_predictions.csv",
    )
    evaluation_path = save_test_evaluation_csv(
        prediction_df,
        output_dir / "risk_test_evaluation.csv",
    )
    metrics_path = save_metrics_json(metrics, output_dir / "risk_metrics.json")

    prepared_predictions = prepare_prediction_df(prediction_df)
    mart_path = create_local_risk_mart(
        predictions_df=prepared_predictions,
        features_df=features_df,
        output_path=output_dir / "mart_risk_alerts.csv",
    )

    high_risk_count = int((prepared_predictions["risk_label"] == "HIGH_RISK").sum())
    print("\n[pipeline] Done")
    print(f"  Features:    {features_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Evaluation:  {evaluation_path}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Model:       {model_path}")
    print(f"  Mart:        {mart_path}")
    print(f"  HIGH_RISK rows: {high_risk_count:,}/{len(prepared_predictions):,}")


if __name__ == "__main__":
    main()
