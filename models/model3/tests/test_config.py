import unittest

from src.config import FEATURES


class ConfigFeatureTests(unittest.TestCase):
    def test_backtest_config_is_available(self):
        from src import config

        self.assertEqual(
            "cvzq3t560s.ap-southeast-1.aws.clickhouse.cloud",
            config.CLICKHOUSE_HOST,
        )
        self.assertEqual("stock", config.CLICKHOUSE_DATABASE)
        self.assertEqual("features_all", config.CLICKHOUSE_FEATURES_TABLE)
        self.assertGreater(len(FEATURES), 0)
        self.assertEqual(5, config.HORIZON)
        self.assertEqual("reports/backtest.csv", config.BACKTEST_PATH)
        self.assertEqual("reports/backtest_metrics.json", config.BACKTEST_METRICS_PATH)
        self.assertEqual("reports/backtest_sweep.csv", config.BACKTEST_SWEEP_PATH)
        self.assertGreater(config.BACKTEST_TOP_K, 0)
        self.assertGreater(config.BACKTEST_MIN_VOLUME, 0)
        self.assertGreater(config.BACKTEST_MIN_CLOSE, 0)
        self.assertGreaterEqual(config.BACKTEST_MIN_BUY_PROBABILITY, 0)
        self.assertLessEqual(config.BACKTEST_MIN_BUY_PROBABILITY, 1)
        self.assertGreaterEqual(config.TRANSACTION_COST_RATE, 0)
        self.assertGreaterEqual(config.SLIPPAGE_RATE, 0)
        self.assertIn(config.BACKTEST_TOP_K, config.BACKTEST_TOP_K_VALUES)
        self.assertIn(config.BACKTEST_MIN_VOLUME, config.BACKTEST_MIN_VOLUME_VALUES)
        self.assertIn(config.BACKTEST_MIN_CLOSE, config.BACKTEST_MIN_CLOSE_VALUES)
        self.assertIn(
            config.BACKTEST_MIN_BUY_PROBABILITY,
            config.BACKTEST_MIN_BUY_PROBABILITY_VALUES,
        )
        self.assertLess(config.SELL_RETURN_THRESHOLD, config.BUY_RETURN_THRESHOLD)
        self.assertEqual({0: "SELL", 1: "HOLD", 2: "BUY"}, config.SIGNAL_LABELS)

    def test_walk_forward_config_is_available(self):
        from src import config

        self.assertEqual(
            "reports/walk_forward_predictions.csv",
            config.WALK_FORWARD_PREDICTION_PATH,
        )
        self.assertEqual(
            "reports/walk_forward_fold_metrics.csv",
            config.WALK_FORWARD_FOLD_METRICS_PATH,
        )
        self.assertEqual(
            "reports/walk_forward_backtest.csv",
            config.WALK_FORWARD_BACKTEST_PATH,
        )
        self.assertEqual(
            "reports/walk_forward_backtest_metrics.json",
            config.WALK_FORWARD_BACKTEST_METRICS_PATH,
        )
        self.assertGreater(config.WALK_FORWARD_INITIAL_TRAIN_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_VALIDATION_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_TEST_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_STEP_RATIO, 0)


if __name__ == "__main__":
    unittest.main()
