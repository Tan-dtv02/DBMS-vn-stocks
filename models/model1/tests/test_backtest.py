import unittest

import pandas as pd

from src.backtest import compute_top_k_backtest, run_backtest_sweep


class BacktestTests(unittest.TestCase):
    def test_compute_top_k_backtest_selects_highest_predictions_each_day(self):
        result_df = pd.DataFrame(
            {
                "trading_date": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ],
                "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
                "predicted_return": [0.03, 0.02, 0.01, 0.01, 0.04, 0.03],
                "target_return": [0.10, -0.02, 0.01, -0.01, 0.05, 0.03],
            }
        )

        daily_df, metrics = compute_top_k_backtest(result_df, top_k=2)

        self.assertEqual(["AAA,BBB", "BBB,CCC"], daily_df["selected_symbols"].tolist())
        self.assertEqual([2, 2], daily_df["selected_count"].tolist())
        self.assertAlmostEqual(0.04, daily_df.loc[0, "daily_return"])
        self.assertAlmostEqual(0.04, daily_df.loc[1, "daily_return"])
        self.assertAlmostEqual((1.04 * 1.04) - 1, metrics["Cumulative_Return"])
        self.assertAlmostEqual(0.04, metrics["Average_Daily_Return"])
        self.assertEqual(100.0, metrics["Hit_Rate"])

    def test_compute_top_k_backtest_rejects_invalid_top_k(self):
        result_df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01"],
                "symbol": ["AAA"],
                "predicted_return": [0.01],
                "target_return": [0.02],
            }
        )

        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            compute_top_k_backtest(result_df, top_k=0)

    def test_compute_top_k_backtest_filters_untradeable_rows(self):
        result_df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "symbol": ["AAA", "BBB", "CCC"],
                "predicted_return": [0.20, 0.10, 0.05],
                "target_return": [0.10, 0.03, 0.02],
                "volume": [100.0, 200000.0, 200000.0],
                "close": [10.0, 4.0, 6.0],
            }
        )

        daily_df, metrics = compute_top_k_backtest(
            result_df,
            top_k=2,
            min_volume=100000,
            min_close=5.0,
        )

        self.assertEqual(["CCC"], daily_df["selected_symbols"].tolist())
        self.assertEqual(1, metrics["Eligible_Rows"])
        self.assertEqual(1, metrics["Eligible_Symbols"])
        self.assertEqual(100000, metrics["Min_Volume"])
        self.assertEqual(5.0, metrics["Min_Close"])

    def test_compute_top_k_backtest_reports_net_returns_after_costs(self):
        result_df = pd.DataFrame(
            {
                "trading_date": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ],
                "symbol": ["AAA", "BBB", "AAA", "BBB"],
                "predicted_return": [0.03, 0.02, 0.04, 0.01],
                "target_return": [0.05, 0.03, 0.06, 0.02],
            }
        )

        daily_df, metrics = compute_top_k_backtest(
            result_df,
            top_k=2,
            transaction_cost_rate=0.001,
            slippage_rate=0.001,
        )

        self.assertAlmostEqual(0.004, metrics["Round_Trip_Cost_Rate"])
        self.assertAlmostEqual(0.036, daily_df.loc[0, "daily_return_net"])
        self.assertAlmostEqual(0.036, daily_df.loc[1, "daily_return_net"])
        self.assertAlmostEqual((1.036 * 1.036) - 1, metrics["Cumulative_Return_Net"])
        self.assertAlmostEqual(0.036, metrics["Average_Daily_Return_Net"])

    def test_compute_top_k_backtest_can_skip_days_below_prediction_threshold(self):
        result_df = pd.DataFrame(
            {
                "trading_date": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ],
                "symbol": ["AAA", "BBB", "AAA", "BBB"],
                "predicted_return": [0.002, 0.001, 0.050, 0.001],
                "target_return": [0.10, 0.02, 0.05, -0.01],
            }
        )

        daily_df, metrics = compute_top_k_backtest(
            result_df,
            top_k=2,
            min_predicted_return=0.004,
            transaction_cost_rate=0.001,
            slippage_rate=0.001,
        )

        self.assertEqual([0, 1], daily_df["selected_count"].tolist())
        self.assertEqual(["", "AAA"], daily_df["selected_symbols"].tolist())
        self.assertAlmostEqual(0.0, daily_df.loc[0, "daily_return"])
        self.assertAlmostEqual(0.0, daily_df.loc[0, "daily_return_net"])
        self.assertAlmostEqual(0.05, daily_df.loc[1, "daily_return"])
        self.assertAlmostEqual(0.046, daily_df.loc[1, "daily_return_net"])
        self.assertEqual(1, metrics["Trade_Days"])
        self.assertAlmostEqual(0.004, metrics["Min_Predicted_Return"])

    def test_run_backtest_sweep_returns_one_row_per_configuration(self):
        result_df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "symbol": ["AAA", "BBB", "CCC"],
                "predicted_return": [0.03, 0.02, 0.01],
                "target_return": [0.04, 0.01, -0.01],
                "volume": [200000.0, 200000.0, 200000.0],
                "close": [10.0, 10.0, 10.0],
            }
        )

        sweep_df = run_backtest_sweep(
            result_df=result_df,
            top_k_values=[1, 2],
            min_volume_values=[100000],
            min_close_values=[5.0],
            min_predicted_return_values=[0.0],
            transaction_cost_rate=0.001,
            slippage_rate=0.001,
        )

        self.assertEqual(2, len(sweep_df))
        self.assertEqual([1, 2], sweep_df["Top_K"].tolist())
        self.assertIn("Cumulative_Return_Net", sweep_df.columns)
        self.assertIn("Sharpe_Ratio_Net", sweep_df.columns)
        self.assertIn("Min_Predicted_Return", sweep_df.columns)


if __name__ == "__main__":
    unittest.main()
