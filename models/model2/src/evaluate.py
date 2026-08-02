import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import REPORT_DIR
from .utils import save_json


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))
    )

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "mape": float(mape)
    }

    save_json(metrics, REPORT_DIR / "metrics.json")

    return metrics


def export_feature_importance(model, feature_columns):
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return importance_df


def export_prediction_error(prediction_df):
    error_df = prediction_df[
        [
            "symbol",
            "trading_date",
            "actual_future_return_5d",
            "predicted_future_return_5d",
            "prediction_error",
            "abs_error",
            "signal"
        ]
    ].copy()

    error_df.to_csv(
        REPORT_DIR / "prediction_error.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return error_df