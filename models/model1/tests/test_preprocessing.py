import unittest

import pandas as pd

from src.preprocessing import preprocess_data, split_train_validation_test_by_time


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_raises_when_requested_feature_is_missing(self):
        df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-02"],
                "symbol": ["AAA", "AAA"],
                "open": [10.0, 11.0],
                "close": [10.0, 11.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "Missing required feature columns"):
            preprocess_data(df=df, features=["open", "not_a_feature"], horizon=1)

    def test_preprocess_targets_close_after_symbol_horizon_sessions(self):
        df = pd.DataFrame(
            {
                "trading_date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-04",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-15",
                    "2024-01-02",
                    "2024-01-03",
                ],
                "symbol": ["AAA", "AAA", "AAA", "AAA", "AAA", "AAA", "BBB", "BBB"],
                "open": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 50.0, 51.0],
                "close": [10.0, 11.0, 12.0, 13.0, 14.0, 22.0, 50.0, 52.0],
            }
        )

        result_df, _ = preprocess_data(df=df, features=["open"], horizon=5)

        aaa_rows = result_df[result_df["symbol"] == "AAA"]
        self.assertEqual([pd.Timestamp("2024-01-01")], aaa_rows["trading_date"].tolist())
        self.assertEqual(pd.Timestamp("2024-01-15"), aaa_rows.iloc[0]["future_trading_date"])
        self.assertAlmostEqual(20.0, aaa_rows.iloc[0]["future_open"])
        self.assertAlmostEqual(22.0, aaa_rows.iloc[0]["future_close"])
        self.assertAlmostEqual(22.0, aaa_rows.iloc[0]["target_close"])
        self.assertAlmostEqual(1.20, aaa_rows.iloc[0]["target_return"])

    def test_preprocess_filters_extreme_target_returns_when_configured(self):
        df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "symbol": ["AAA", "AAA", "AAA"],
                "open": [10.0, 11.0, 30.0],
                "close": [10.0, 11.0, 30.0],
            }
        )

        result_df, _ = preprocess_data(
            df=df,
            features=["open"],
            horizon=1,
            max_abs_target_return=0.2,
        )

        self.assertEqual([pd.Timestamp("2024-01-01")], result_df["trading_date"].tolist())
        self.assertAlmostEqual(11.0, result_df.iloc[0]["target_close"])
        self.assertAlmostEqual(0.10, result_df.iloc[0]["target_return"])

    def test_split_train_validation_test_by_time_keeps_chronological_order(self):
        df = pd.DataFrame(
            {
                "trading_date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "symbol": ["AAA"] * 10,
                "close": range(10),
            }
        )

        train_df, validation_df, test_df, validation_start, test_start = (
            split_train_validation_test_by_time(
                df=df,
                train_ratio=0.6,
                validation_ratio=0.2,
            )
        )

        self.assertEqual(6, len(train_df))
        self.assertEqual(2, len(validation_df))
        self.assertEqual(2, len(test_df))
        self.assertEqual(pd.Timestamp("2024-01-07"), validation_start)
        self.assertEqual(pd.Timestamp("2024-01-09"), test_start)
        self.assertLess(train_df["trading_date"].max(), validation_start)
        self.assertGreaterEqual(validation_df["trading_date"].min(), validation_start)
        self.assertLess(validation_df["trading_date"].max(), test_start)
        self.assertGreaterEqual(test_df["trading_date"].min(), test_start)


if __name__ == "__main__":
    unittest.main()
