import pandas as pd
import numpy as np
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "data_clean" / "Data_500_stocks_2015-2026.csv"
OUTPUT_FILE = BASE_DIR / "data" / "data_dirty" / "Data_500_stocks_dirty.csv"

DIRTY_RATIO = 0.07   # < 7%
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

total_rows = len(df)
dirty_rows = int(total_rows * DIRTY_RATIO)

print(f"Total rows: {total_rows}")
print(f"Dirty rows target: {dirty_rows}")

# =========================
# 1. Missing values
# =========================
missing_idx = np.random.choice(df.index, int(dirty_rows * 0.25), replace=False)

for col in ["open", "high", "low", "close", "volume"]:
    selected = np.random.choice(missing_idx, len(missing_idx)//3, replace=False)
    df.loc[selected, col] = np.nan

# =========================
# 2. Duplicate rows
# =========================
dup_rows = df.sample(int(dirty_rows * 0.15), random_state=RANDOM_SEED)
df = pd.concat([df, dup_rows], ignore_index=True)

# =========================
# 3. Outliers
# =========================
outlier_idx = np.random.choice(df.index, int(dirty_rows * 0.20), replace=False)

df.loc[outlier_idx, "close"] *= np.random.uniform(3, 10, size=len(outlier_idx))
df.loc[outlier_idx, "volume"] *= np.random.randint(20, 100, size=len(outlier_idx))

# =========================
# 4. Wrong low/high logic
# low > high
# =========================
logic_idx = np.random.choice(df.index, int(dirty_rows * 0.15), replace=False)

temp = df.loc[logic_idx, "low"].copy()
df.loc[logic_idx, "low"] = df.loc[logic_idx, "high"]
df.loc[logic_idx, "high"] = temp

# =========================
# 5. Wrong symbol format
# =========================
symbol_idx = np.random.choice(df.index, int(dirty_rows * 0.10), replace=False)

wrong_symbols = ["???", "123", "NULL", "A@", "ZZZZZZ"]
df.loc[symbol_idx, "symbol"] = np.random.choice(wrong_symbols, len(symbol_idx))

# =========================
# 6. Negative values
# =========================
negative_idx = np.random.choice(df.index, int(dirty_rows * 0.15), replace=False)

df.loc[negative_idx, "volume"] *= -1
df.loc[negative_idx, "open"] *= -1

# =========================
# SAVE
# =========================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDirty dataset saved to: {OUTPUT_FILE}")
print(f"Final rows: {len(df)}")
