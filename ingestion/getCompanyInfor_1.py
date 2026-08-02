from vnstock.api.company import Company
import pandas as pd
# Source: VCI

symbol = "VE8"

company = Company(
    symbol=symbol,
    source="VCI"
)

overview = company.overview()

print("\n===== RAW DATA =====")
print(overview)

print("\n===== COLUMNS =====")
print(overview.columns)

print("\n===== FIRST ROW (DICT) =====")
print(overview.iloc[0].to_dict() if overview is not None and not overview.empty else "NO DATA")