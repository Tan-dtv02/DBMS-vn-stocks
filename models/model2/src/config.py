import os
from pathlib import Path
from dotenv import load_dotenv

MODEL2_ROOT = Path(__file__).resolve().parents[1] #...\HQTCSDL_stocks\models\model2
PROJECT_ROOT = Path(__file__).resolve().parents[3] #...\HQTCSDL_stocks
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT") or 8443)
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "stock")
CLICKHOUSE_SECURE = os.getenv("CLICKHOUSE_SECURE", "true").lower() == "true"

SOURCE_DATABASE = os.getenv("CLICKHOUSE_SOURCE_DATABASE", "stock")
SOURCE_TABLE = os.getenv("CLICKHOUSE_TABLE", "features_all")
MART_DATABASE = os.getenv("CLICKHOUSE_MART_DATABASE", "stock")
MART_TABLE = os.getenv("MODEL2_MART_TABLE", "mart_future_return_prediction")

DATE_COL = "trading_date"
SYMBOL_COL = "symbol"
TARGET_COL = "future_return_5d"

MODEL_NAME = "future_return_lgbm"

MODEL_DIR = MODEL2_ROOT / "models"
REPORT_DIR = MODEL2_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "future_return_lgbm.pkl"

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume", "encode_sector",
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "ma_5", "ma_20", "ma_50",
    "price_vs_ma20", "ma5_vs_ma20",
    "volatility_5d", "volatility_20d", "volatility_change",
    "rolling_max_20d", "drawdown_20d",
    "volume_ma_5", "volume_ma_20", "volume_ratio_5_20",
    "volume_change_1d",
    "daily_range", "body_ratio", "close_position"
]

TEST_START_DATE = "2024-01-01"
RANDOM_STATE = 42
