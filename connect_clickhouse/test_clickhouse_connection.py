from clickhouse_client import get_clickhouse_client

client = get_clickhouse_client()

result = client.query("SELECT version()")

print("Connected to ClickHouse")
print("Version:", result.result_rows[0][0])