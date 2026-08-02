import ast
import time
from pathlib import Path
import pandas as pd
from vnstock.api.company import Company

# =========================
# CONFIG
# =========================

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL_LIST_FILE = DATA_DIR / "symbol500.txt"

OUTPUT_FILE = DATA_DIR / "company_infor.csv"

SOURCE = "VCI"

MAX_SYMBOLS = 500
REQUEST_DELAY = 1
BATCH_SIZE = 20
SLEEP_TIME = 90

# =========================
# LOAD SYMBOLS
# =========================

all_symbols = []

with open(SYMBOL_LIST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_symbols.extend(ast.literal_eval(line))

all_symbols = list(dict.fromkeys(all_symbols))[:MAX_SYMBOLS]

print(f"Tổng số mã: {len(all_symbols)}")

# =========================
# FETCH
# =========================

result = []

for index, symbol in enumerate(all_symbols, start=1):
    try:
        print(f"[{index}/{len(all_symbols)}] {symbol}")

        company = Company(symbol=symbol, source=SOURCE)
        overview = company.overview()

        if overview is None or overview.empty:
            print(f"Không có data: {symbol}")
            continue

        row0 = overview.iloc[0].to_dict()

        company_name = (
            row0.get("organ_name")
            or row0.get("organ_short_name")
            or symbol
            or ""
        )

        sector = (
            row0.get('sector')
        )

        result.append({
            "symbol": symbol,
            "company_name": company_name,
            'sector': sector,
            "listed_date": row0.get("listing_date", "")
        })

        time.sleep(REQUEST_DELAY)

    except Exception as e:
        print(f"Lỗi {symbol}: {e}")

    # =========================
    # BATCH SAVE
    # =========================
    if index % BATCH_SIZE == 0:
        df_batch = pd.DataFrame(result)

        # ghi append an toàn
        if index == BATCH_SIZE:
            df_batch.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        else:
            df_batch.to_csv(OUTPUT_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")

        print(f"💾 Saved {len(result)} rows")

        # reset buffer sau khi save
        result = []

        if index < len(all_symbols):
            print(f" Sleep {SLEEP_TIME}s...\n")
            time.sleep(SLEEP_TIME)

# =========================
# FINAL SAVE
# =========================

if result:
    df_final = pd.DataFrame(result)

    if OUTPUT_FILE.exists():
        df_final.to_csv(OUTPUT_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nDONE")
print(f"File saved at: {OUTPUT_FILE}")
