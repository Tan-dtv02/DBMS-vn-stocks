import random
from pathlib import Path
from vnstock import Listing


SYMBOL_LIST_FILE = Path(__file__).resolve().parent / "symbol500.txt"
# =========================
# SEED (CỐ ĐỊNH KẾT QUẢ RANDOM)
# =========================
SEED = 42
random.seed(SEED)

# =========================
# LẤY TOÀN BỘ MÃ
# =========================
listing = Listing()
df = listing.all_symbols()
codes = df["symbol"].dropna().unique().tolist()

# =========================
# RANDOM 500 MÃ CỐ ĐỊNH
# =========================
random_codes = random.sample(codes, 500)

# =========================
# RANDOM THỨ TỰ (CŨNG CỐ ĐỊNH NHỜ SEED)
# =========================
# reverse_sort = random.choice([True, False])
reverse_sort = False

random_codes.sort(reverse=reverse_sort)

# =========================
# TẠO THƯ MỤC NẾU CHƯA CÓ
# =========================
SYMBOL_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)

# =========================
# GHI FILE
# =========================
with SYMBOL_LIST_FILE.open("w", encoding="utf-8") as f:
    for i in range(0, len(random_codes), 10):
        chunk = random_codes[i:i + 10]
        f.write(str(chunk) + "\n")

# =========================
# OUTPUT
# =========================
order_type = "Z-A" if reverse_sort else "A-Z"

print("================================")
print("Đã tạo file:", SYMBOL_LIST_FILE)
print(f"SEED: {SEED}")
print(f"Tổng số mã: {len(random_codes)}")
print(f"Sắp xếp: {order_type}")
print("Mỗi dòng: 10 mã")
print("================================")
