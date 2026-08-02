import clickhouse_connect
import pandas as pd

from src.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_SOURCE_DATABASE,
    CLICKHOUSE_TABLE,
    CLICKHOUSE_USER,
)


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def table_expression(database: str, table: str) -> str:
    if "." in table:
        database_name, table_name = table.split(".", 1)
        return f"{quote_identifier(database_name)}.{quote_identifier(table_name)}"
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def load_data(path=None):
    if path is not None:
        print("[model1] Ignoring CSV path; loading data from ClickHouse.")

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

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE,
    )

    query = f"""
        SELECT *
        FROM {table_expression(CLICKHOUSE_SOURCE_DATABASE, CLICKHOUSE_TABLE)}
        ORDER BY symbol, trading_date
    """
    df = client.query_df(query)
    df.columns = df.columns.str.strip()
    return df
