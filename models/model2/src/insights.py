import json
import pandas as pd

from .config import REPORT_DIR


def generate_insights():

    prediction_file = REPORT_DIR / "prediction_error.csv"
    importance_file = REPORT_DIR / "feature_importance.csv"
    backtest_metrics_file = REPORT_DIR / "backtest_metrics.json"

    prediction_df = pd.read_csv(prediction_file)
    importance_df = pd.read_csv(importance_file)

    with open(backtest_metrics_file, "r", encoding="utf-8") as f:
        backtest_metrics = json.load(f)

    insights = {}

    # =========================
    # Top predicted stocks
    # =========================

    top_stocks = (
        prediction_df
        .sort_values(
            "predicted_future_return_5d",
            ascending=False
        )
        .head(10)
    )

    insights["top_10_predicted_stocks"] = (
        top_stocks[
            [
                "symbol",
                "predicted_future_return_5d"
            ]
        ]
        .to_dict("records")
    )

    # =========================
    # Worst predicted stocks
    # =========================

    worst_stocks = (
        prediction_df
        .sort_values(
            "predicted_future_return_5d",
            ascending=True
        )
        .head(10)
    )

    insights["worst_10_predicted_stocks"] = (
      worst_stocks[
          [
              "symbol",
              "predicted_future_return_5d"
          ]
      ]
      .to_dict("records")
    )

    # =========================
    # Top features
    # =========================

    top_features = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(10)
    )

    insights["top_10_features"] = (
        top_features.to_dict("records")
    )

    # =========================
    # Prediction statistics
    # =========================

    insights["prediction_summary"] = {

        "avg_predicted_return":

            float(
                prediction_df[
                    "predicted_future_return_5d"
                ].mean()
            ),

        "max_predicted_return":

            float(
                prediction_df[
                    "predicted_future_return_5d"
                ].max()
            ),

        "min_predicted_return":

            float(
                prediction_df[
                    "predicted_future_return_5d"
                ].min()
            ),

        "positive_prediction_ratio":

            float(
                (
                    prediction_df[
                        "predicted_future_return_5d"
                    ] > 0
                ).mean()
            )
    }

    # =========================
    # Error statistics
    # =========================

    insights["error_summary"] = {

        "avg_abs_error":

            float(
                prediction_df[
                    "abs_error"
                ].mean()
            ),

        "max_abs_error":

            float(
                prediction_df[
                    "abs_error"
                ].max()
            )
    }

    # =========================
    # Backtest metrics
    # =========================

    insights["backtest"] = backtest_metrics

    output_path = REPORT_DIR / "insights.json"

    with open(
            output_path,
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            insights,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"Insights saved to {output_path}"
    )

    return insights


if __name__ == "__main__":
    generate_insights()
