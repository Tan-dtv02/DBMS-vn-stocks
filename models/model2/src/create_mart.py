from .config import MART_DATABASE, MART_TABLE, MODEL_NAME
from .data_loader import get_clickhouse_client, table_expression


def create_mart_table():
    client = get_clickhouse_client()
    target_table = table_expression(MART_DATABASE, MART_TABLE)

    query = f"""
    CREATE TABLE IF NOT EXISTS {target_table}
    (
        symbol String,
        trading_date Date,
        close Float64,
        volume Float64,
        encode_sector Float64,

        actual_future_return_5d Float64,
        predicted_future_return_5d Float64,
        prediction_error Float64,
        abs_error Float64,

        signal String,
        model_name String,
        created_at DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    ORDER BY (trading_date, symbol)
    """

    client.command(query)


def insert_mart_data(prediction_df):
    client = get_clickhouse_client()
    target_table = f"{MART_DATABASE}.{MART_TABLE}"

    mart_df = prediction_df[
        [
            "symbol",
            "trading_date",
            "close",
            "volume",
            "encode_sector",
            "actual_future_return_5d",
            "predicted_future_return_5d",
            "prediction_error",
            "abs_error",
            "signal"
        ]
    ].copy()

    mart_df["model_name"] = MODEL_NAME
    mart_df["trading_date"] = mart_df["trading_date"].dt.date

    client.insert_df(target_table, mart_df)
