import joblib
import pandas as pd
import numpy as np

from src.signal_decision import apply_confidence_adjusted_signals


def load_saved_model(model_path):
    saved = joblib.load(model_path)

    model = saved["model"]
    features = saved["features"]
    signal_labels = saved.get("signal_labels", {0: "SELL", 1: "HOLD", 2: "BUY"})

    return model, features, signal_labels


def predict_latest_signal(
    df,
    model,
    features,
    signal_labels=None,
    min_action_probability=0.40,
    min_action_margin=0.03,
):
    signal_labels = signal_labels or {0: "SELL", 1: "HOLD", 2: "BUY"}
    df = df.copy()

    df = df.replace(["NULL", "null", "None", ""], np.nan)

    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.sort_values(["symbol", "trading_date"])

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    latest_df = df.groupby("symbol").tail(1).copy()

    latest_df = latest_df.dropna(subset=features)

    X_latest = latest_df[features]

    predicted_label = model.predict(X_latest).astype(int)
    predicted_proba = model.predict_proba(X_latest)

    latest_df["predicted_signal_label"] = predicted_label
    latest_df["predicted_signal"] = latest_df["predicted_signal_label"].map(
        signal_labels
    )

    for label_id, label_name in signal_labels.items():
        latest_df[f"{label_name.lower()}_probability"] = predicted_proba[:, label_id]

    latest_df["predicted_signal_score"] = (
        latest_df["buy_probability"] - latest_df["sell_probability"]
    )
    latest_df = apply_confidence_adjusted_signals(
        latest_df,
        min_action_probability=min_action_probability,
        min_action_margin=min_action_margin,
    )

    return latest_df


def predict_latest_price(df, model, features):
    return predict_latest_signal(df, model, features)
