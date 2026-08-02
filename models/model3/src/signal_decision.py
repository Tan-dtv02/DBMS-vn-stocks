import numpy as np
import pandas as pd


def apply_confidence_adjusted_signals(
    df,
    min_action_probability=0.40,
    min_action_margin=0.03,
):
    required_columns = ["sell_probability", "hold_probability", "buy_probability"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError("Missing signal probability columns: " + ", ".join(missing_columns))

    if not 0 <= min_action_probability <= 1:
        raise ValueError("min_action_probability must be between 0 and 1")

    if min_action_margin < 0:
        raise ValueError("min_action_margin must be non-negative")

    result_df = df.copy()
    probabilities = result_df[required_columns].apply(pd.to_numeric, errors="coerce")

    buy_edge = probabilities["buy_probability"] - probabilities["sell_probability"]
    sell_edge = probabilities["sell_probability"] - probabilities["buy_probability"]

    adjusted_signal = np.full(len(result_df), "HOLD", dtype=object)
    adjusted_label = np.full(len(result_df), 1, dtype=int)

    buy_mask = (
        (probabilities["buy_probability"] >= min_action_probability)
        & (probabilities["buy_probability"] >= probabilities["hold_probability"])
        & (buy_edge >= min_action_margin)
    )
    sell_mask = (
        (probabilities["sell_probability"] >= min_action_probability)
        & (probabilities["sell_probability"] >= probabilities["hold_probability"])
        & (sell_edge >= min_action_margin)
    )

    adjusted_signal[buy_mask.to_numpy()] = "BUY"
    adjusted_label[buy_mask.to_numpy()] = 2
    adjusted_signal[sell_mask.to_numpy()] = "SELL"
    adjusted_label[sell_mask.to_numpy()] = 0

    result_df["adjusted_signal_label"] = adjusted_label
    result_df["adjusted_signal"] = adjusted_signal
    result_df["signal_confidence"] = probabilities.max(axis=1)
    result_df["buy_sell_margin"] = buy_edge

    return result_df
