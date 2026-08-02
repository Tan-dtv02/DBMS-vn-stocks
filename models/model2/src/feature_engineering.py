import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COL


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["symbol", "trading_date"])

    group = df.groupby("symbol")

    df[TARGET_COL] = group["close"].shift(-5) / df["close"] - 1

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL])

    return df


def get_train_test_data(df: pd.DataFrame, test_start_date: str):
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    cutoff_date = pd.to_datetime(test_start_date)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL, "trading_date"])

    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df = df[df["trading_date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/Test split produced an empty dataset. "
            f"Check TEST_START_DATE={test_start_date}."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test, train_df, test_df
