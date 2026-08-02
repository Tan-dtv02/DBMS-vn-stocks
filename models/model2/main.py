from src.data_loader import load_stock_data

from src.feature_engineering import (
    create_features,
    get_train_test_data
)

from src.train_model import train_lgbm_model

from src.predict import predict_future_return

from src.evaluate import (
    evaluate_regression,
    export_feature_importance,
    export_prediction_error
)

from src.backtest import (
    run_backtest,
    run_backtest_sweep
)

from src.create_mart import (
    create_mart_table,
    insert_mart_data
)

from src.config import (
    TEST_START_DATE,
    FEATURE_COLUMNS,
    REPORT_DIR
)

from src.utils import ensure_dirs

from dashboard_model2 import create_dashboard
from src.insights import generate_insights

def main():

    ensure_dirs(REPORT_DIR)

    print("1. Load data...")
    df = load_stock_data()

    print("2. Feature engineering...")
    df = create_features(df)

    print("3. Train/Test split...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df
    ) = get_train_test_data(
        df,
        TEST_START_DATE
    )

    print("4. Train model...")
    model = train_lgbm_model(
        X_train,
        y_train
    )

    print("5. Predict...")
    prediction_df = predict_future_return(
        model,
        test_df,
        X_test
    )

    print("6. Evaluate...")
    metrics = evaluate_regression(
        y_test,
        prediction_df["predicted_future_return_5d"]
    )

    print(metrics)

    print("7. Export feature importance...")
    export_feature_importance(
        model,
        FEATURE_COLUMNS
    )

    print("8. Export prediction error...")
    export_prediction_error(
        prediction_df
    )

    print("9. Backtest...")
    run_backtest(prediction_df)

    print("10. Backtest sweep...")
    run_backtest_sweep(prediction_df)

    print("11. Generate insights...")
    generate_insights()

    print("12. Create mart...")
    create_mart_table()

    insert_mart_data(
        prediction_df
    )

    print("13. Dashboard...")
    create_dashboard()

    print("Done.")


if __name__ == "__main__":
    main()
