import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from src.config import FEATURE_COLUMNS, TARGET_COL


def walk_forward_validation(
        df,
        train_window=500,
        test_window=50
):
    results = []

    start = train_window

    while start + test_window <= len(df):

        train_df = df.iloc[start - train_window:start]
        test_df = df.iloc[start:start + test_window]

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COL]

        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COL]

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            pred
        )

        results.append({
            "start_idx": start,
            "mae": mae
        })

        start += test_window

    return pd.DataFrame(results)


if __name__ == "__main__":
  print("Run from main.py")