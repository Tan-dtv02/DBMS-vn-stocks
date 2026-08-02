import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.signal_decision import apply_confidence_adjusted_signals


SIGNAL_LABELS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}


def evaluate_model(
    model,
    X_test,
    test_df,
    signal_labels=None,
    min_action_probability=0.40,
    min_action_margin=0.03,
):
    signal_labels = signal_labels or SIGNAL_LABELS
    result_df = test_df.copy()

    pred_label = model.predict(X_test).astype(int)
    pred_proba = model.predict_proba(X_test)

    if "target_return" not in result_df.columns:
        result_df["target_return"] = result_df["target_close"] / result_df["close"] - 1

    result_df["predicted_signal_label"] = pred_label
    result_df["predicted_signal"] = result_df["predicted_signal_label"].map(signal_labels)

    for label_id, label_name in signal_labels.items():
        result_df[f"{label_name.lower()}_probability"] = pred_proba[:, label_id]

    result_df["predicted_signal_score"] = (
        result_df["buy_probability"] - result_df["sell_probability"]
    )
    result_df = apply_confidence_adjusted_signals(
        result_df,
        min_action_probability=min_action_probability,
        min_action_margin=min_action_margin,
    )

    y_true = result_df["target_signal_label"].astype(int)
    y_pred = result_df["predicted_signal_label"].astype(int)
    y_adjusted = result_df["adjusted_signal_label"].astype(int)

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(signal_labels.keys()),
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(signal_labels.keys()))

    baseline_label = int(y_true.mode().iloc[0])
    baseline_pred = np.full(len(y_true), baseline_label)
    baseline_accuracy = accuracy_score(y_true, baseline_pred) * 100
    adjusted_accuracy = accuracy_score(y_true, y_adjusted) * 100
    adjusted_precision, adjusted_recall, adjusted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_adjusted,
        labels=list(signal_labels.keys()),
        average="macro",
        zero_division=0,
    )
    adjusted_cm = confusion_matrix(y_true, y_adjusted, labels=list(signal_labels.keys()))
    adjusted_buy_mask = result_df["adjusted_signal"] == "BUY"
    adjusted_sell_mask = result_df["adjusted_signal"] == "SELL"
    adjusted_buy_precision = 0.0
    if adjusted_buy_mask.any():
        adjusted_buy_precision = (
            result_df.loc[adjusted_buy_mask, "target_signal"].eq("BUY").mean() * 100
        )
    adjusted_sell_precision = 0.0
    if adjusted_sell_mask.any():
        adjusted_sell_precision = (
            result_df.loc[adjusted_sell_mask, "target_signal"].eq("SELL").mean() * 100
        )

    metrics = {
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": f1,
        "Adjusted_Accuracy": adjusted_accuracy,
        "Adjusted_Macro_Precision": adjusted_precision,
        "Adjusted_Macro_Recall": adjusted_recall,
        "Adjusted_Macro_F1": adjusted_f1,
        "Adjusted_BUY_Precision": adjusted_buy_precision,
        "Adjusted_SELL_Precision": adjusted_sell_precision,
        "Adjusted_Confusion_Matrix": adjusted_cm.tolist(),
        "Signal_Min_Action_Probability": min_action_probability,
        "Signal_Min_Action_Margin": min_action_margin,
        "Baseline_Accuracy": baseline_accuracy,
        "Baseline_Label": signal_labels[baseline_label],
        "Confusion_Matrix": cm.tolist(),
    }

    return metrics, result_df


def build_prediction_accuracy_table(result_df):
    required_columns = [
        "future_trading_date",
        "symbol",
        "target_signal",
        "predicted_signal",
    ]
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise ValueError(
            "Missing prediction accuracy columns: " + ", ".join(missing_columns)
        )

    accuracy_df = pd.DataFrame(
        {
            "date": pd.to_datetime(result_df["future_trading_date"]),
            "symbol": result_df["symbol"],
            "real_signal": result_df["target_signal"],
            "predict_signal": result_df["predicted_signal"],
        }
    )

    accuracy_df["is_correct"] = (
        accuracy_df["real_signal"] == accuracy_df["predict_signal"]
    )

    return accuracy_df[
        ["date", "symbol", "real_signal", "predict_signal", "is_correct"]
    ]


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def save_feature_importance(model, features, path):
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(path, index=False)
