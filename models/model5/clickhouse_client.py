from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_bool_env(name: str, default: bool = True):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y"}


def get_clickhouse_client():
    """Create a clickhouse-connect client from .env settings.

    This module is kept ready for the ClickHouse phase. The local CSV pipeline
    does not call it.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

    try:
        import clickhouse_connect
    except ImportError:
        print(
            "clickhouse-connect is not installed. Install it with: "
            "pip install clickhouse-connect"
        )
        return None

    host = os.getenv("CLICKHOUSE_HOST")
    username = os.getenv("CLICKHOUSE_USER")
    password = os.getenv("CLICKHOUSE_PASSWORD")

    if not host or not username or not password:
        print(
            "Missing ClickHouse settings. Required: CLICKHOUSE_HOST, "
            "CLICKHOUSE_USER, CLICKHOUSE_PASSWORD."
        )
        return None

    return clickhouse_connect.get_client(
        host=host,
        port=int(os.getenv("CLICKHOUSE_PORT", "8443")),
        username=username,
        password=password,
        database=os.getenv("CLICKHOUSE_DATABASE", "stock"),
        secure=_get_bool_env("CLICKHOUSE_SECURE", default=True),
    )
