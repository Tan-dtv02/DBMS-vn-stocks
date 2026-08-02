import unittest

import numpy as np
import pandas as pd

from src.evaluate import build_prediction_accuracy_table, evaluate_model


class DummyModel:
    def __init__(self, predictions, probabilities):
        self.predictions = np.array(predictions)
        self.probabilities = np.array(probabilities)

    def predict(self, X):
        return self.predictions

    def predict_proba(self, X):
        return self.probabilities


class EvaluateModelTests(unittest.TestCase):
    def test_evaluate_model_reports_signal_classification_metrics(self):
        test_df = pd.DataFrame(
            {
                "close": [100.0, 100.0],
                "future_close": [110.0, 90.0],
                "target_close": [110.0, 90.0],
                "target_return": [0.10, -0.10],
                "target_signal_label": [2, 0],
                "target_signal": ["BUY", "SELL"],
            }
        )
        X_test = pd.DataFrame({"feature": [1.0, 2.0]})
        model = DummyModel(
            predictions=[2, 1],
            probabilities=[
                [0.05, 0.10, 0.85],
                [0.20, 0.60, 0.20],
            ],
        )

        metrics, result_df = evaluate_model(model=model, X_test=X_test, test_df=test_df)

        self.assertAlmostEqual(50.0, metrics["Accuracy"])
        self.assertIn("Macro_F1", metrics)
        self.assertEqual([[0, 1, 0], [0, 0, 0], [0, 0, 1]], metrics["Confusion_Matrix"])
        self.assertEqual("BUY", result_df.loc[0, "predicted_signal"])
        self.assertEqual("HOLD", result_df.loc[1, "predicted_signal"])
        self.assertAlmostEqual(0.85, result_df.loc[0, "buy_probability"])
        self.assertAlmostEqual(0.0, result_df.loc[1, "predicted_signal_score"])

    def test_build_prediction_accuracy_table_reports_signal_match(self):
        result_df = pd.DataFrame(
            {
                "future_trading_date": pd.to_datetime(["2024-01-10", "2024-01-11"]),
                "symbol": ["AAA", "BBB"],
                "target_signal": ["BUY", "SELL"],
                "predicted_signal": ["BUY", "HOLD"],
            }
        )

        accuracy_df = build_prediction_accuracy_table(result_df)

        self.assertEqual(
            ["date", "symbol", "real_signal", "predict_signal", "is_correct"],
            accuracy_df.columns.tolist(),
        )
        self.assertEqual(pd.Timestamp("2024-01-10"), accuracy_df.loc[0, "date"])
        self.assertEqual("AAA", accuracy_df.loc[0, "symbol"])
        self.assertEqual("BUY", accuracy_df.loc[0, "real_signal"])
        self.assertEqual("BUY", accuracy_df.loc[0, "predict_signal"])
        self.assertTrue(accuracy_df.loc[0, "is_correct"])
        self.assertFalse(accuracy_df.loc[1, "is_correct"])


if __name__ == "__main__":
    unittest.main()
