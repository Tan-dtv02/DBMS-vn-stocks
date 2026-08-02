# src/preprocessing.py

import numpy as np
import pandas as pd


def preprocess_data(
    df,
    features,
    horizon,
    max_abs_target_return=None,
    sell_return_threshold=-0.01,
    buy_return_threshold=0.01,
):
    df = df.copy()

    df = df.replace(["NULL", "null", "None", ""], np.nan)

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if sell_return_threshold >= buy_return_threshold:
        raise ValueError("sell_return_threshold must be less than buy_return_threshold")

    required_columns = list(
        dict.fromkeys(["trading_date", "symbol", "open", "close"] + features)
    )
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing_columns)
        )

    df["trading_date"] = pd.to_datetime(df["trading_date"])

    df = df.sort_values(["symbol", "trading_date"])

    final_features = list(features)

    numeric_columns = list(dict.fromkeys(final_features + ["open", "close"]))
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    symbol_groups = df.groupby("symbol", sort=False)
    df["future_trading_date"] = symbol_groups["trading_date"].shift(-horizon)
    df["future_open"] = symbol_groups["open"].shift(-horizon)
    df["future_close"] = symbol_groups["close"].shift(-horizon)
    df["target_close"] = df["future_close"]
    df["target_return"] = df["target_close"] / df["close"] - 1
    df["target_signal_label"] = np.select(
        [
            df["target_return"] <= sell_return_threshold,
            df["target_return"] >= buy_return_threshold,
        ],
        [0, 2],
        default=1,
    )
    df["target_signal"] = df["target_signal_label"].map(
        {
            0: "SELL",
            1: "HOLD",
            2: "BUY",
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=final_features
        + [
            "future_trading_date",
            "future_open",
            "future_close",
            "target_close",
            "target_return",
            "target_signal_label",
            "target_signal",
        ]
    )

    if max_abs_target_return is not None:
        if max_abs_target_return <= 0:
            raise ValueError("max_abs_target_return must be positive")

        df = df[df["target_return"].abs() <= max_abs_target_return].copy()

    return df, final_features


def split_train_test_by_time(df, split_ratio=0.8):
    dates = np.sort(df["trading_date"].unique())

    split_index = int(len(dates) * split_ratio)
    split_date = dates[split_index]

    train_df = df[df["trading_date"] < split_date].copy()
    test_df = df[df["trading_date"] >= split_date].copy()

    return train_df, test_df, split_date


def split_train_validation_test_by_time(
    df,
    train_ratio=0.7,
    validation_ratio=0.15,
):
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("train_ratio and validation_ratio must be positive")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    dates = np.sort(df["trading_date"].unique())
    if len(dates) < 3:
        raise ValueError("Need at least 3 unique trading dates to split data")

    validation_index = int(len(dates) * train_ratio)
    test_index = int(len(dates) * (train_ratio + validation_ratio))

    if validation_index == 0 or test_index <= validation_index or test_index >= len(dates):
        raise ValueError("Split ratios create an empty train, validation, or test set")

    validation_start_date = pd.Timestamp(dates[validation_index])
    test_start_date = pd.Timestamp(dates[test_index])

    train_df = df[df["trading_date"] < validation_start_date].copy()
    validation_df = df[
        (df["trading_date"] >= validation_start_date)
        & (df["trading_date"] < test_start_date)
    ].copy()
    test_df = df[df["trading_date"] >= test_start_date].copy()

    return train_df, validation_df, test_df, validation_start_date, test_start_date
