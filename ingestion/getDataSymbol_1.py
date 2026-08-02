# Crawl dữ liệu 1 mã cổ phiếu: ABC
# In ra 10 dòng đầu tiên
# Không ghi file

import pandas as pd
from vnstock.api.quote import Quote

# =========================
# CONFIG
# =========================

SYMBOL = "ABC"
SOURCE = "KBS"

START_DATE = "2026-01-01"
END_DATE = "2026-04-30"

# =========================
# LẤY DỮ LIỆU
# =========================

try:
    print(f"Đang lấy dữ liệu: {SYMBOL}")

    q = Quote(
        symbol=SYMBOL,
        source=SOURCE
    )

    df_history = q.history(
        start=START_DATE,
        end=END_DATE
    )

    if df_history is None or df_history.empty:
        print("Không có dữ liệu")
    else:
        # thêm cột symbol
        df_history["symbol"] = SYMBOL

        print(df_history.head(10))

        print("\nThông tin dataframe:\n")
        print(df_history.info())

except Exception as e:
    print(f"Lỗi: {e}")