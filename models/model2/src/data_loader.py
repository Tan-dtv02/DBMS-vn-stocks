import clickhouse_connect
import pandas as pd

from .config import (
    CLICKHOUSE_HOST,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_SECURE,
    SOURCE_DATABASE,
    SOURCE_TABLE
)


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def table_expression(database: str, table: str) -> str:
    if "." in table:
        database_name, table_name = table.split(".", 1)
        return f"{quote_identifier(database_name)}.{quote_identifier(table_name)}"
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def get_clickhouse_client():
    missing = [
        name
        for name, value in {
            "CLICKHOUSE_HOST": CLICKHOUSE_HOST,
            "CLICKHOUSE_USER": CLICKHOUSE_USER,
            "CLICKHOUSE_PASSWORD": CLICKHOUSE_PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing ClickHouse environment variables: " + ", ".join(missing)
        )

    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE,
    )


def load_stock_data():
    client = get_clickhouse_client()

    query = f"""
        SELECT *
        FROM {table_expression(SOURCE_DATABASE, SOURCE_TABLE)}
        ORDER BY symbol, trading_date
    """

    df = client.query_df(query)

    if df.empty:
        raise ValueError("Không lấy được dữ liệu từ ClickHouse.")

    df["trading_date"] = pd.to_datetime(df["trading_date"])

    return df

