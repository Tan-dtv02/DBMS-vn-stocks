import joblib

from .config import MODEL_PATH
from .utils import add_prediction_signal


def load_model():
    return joblib.load(MODEL_PATH)


def predict_future_return(model, test_df, X_test):
    result_df = test_df.copy()

    result_df["predicted_future_return_5d"] = model.predict(X_test)       #model du doan
    result_df["actual_future_return_5d"] = result_df["future_return_5d"]  #gia tri thuc te 

    # saiso = thuc te - du doan
    result_df["prediction_error"] = (
        result_df["actual_future_return_5d"]
        - result_df["predicted_future_return_5d"]
    )

    # saiso = |sai so|
    result_df["abs_error"] = result_df["prediction_error"].abs()

    #goi y nen: buy/hold/sell
    result_df["signal"] = result_df["predicted_future_return_5d"].apply(
        add_prediction_signal
    )

    return result_df