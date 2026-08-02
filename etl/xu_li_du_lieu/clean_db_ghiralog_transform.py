from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "dirty" / "Data_500_stocks_dirty.csv"
REFERENCE_PATH = PROJECT_ROOT / "data" / "clean" / "Data_500_stocks_2015-2026.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "clean" / "Data_500_stocks_clean_ver2.csv"
LOG_DIR = PROJECT_ROOT / "data" / "clean_log"

KEY_COLUMNS = ["symbol", "date"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume"]
OUTPUT_COLUMNS = ["symbol", "date", *NUMERIC_COLUMNS]
SOURCE_COLUMNS = ["time", *NUMERIC_COLUMNS, "symbol"]
DIRTY_SYMBOLS = {"???", "123", "NULL", "A@", "ZZZZZZ"}


def save_log_df(df: pd.DataFrame, filename: str) -> None:
    if df.empty:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_DIR / filename, index=False)


def write_summary(stats: Dict[str, int]) -> None:
    lines = [
        "clean_summary",
        f"input_path: {INPUT_PATH}",
        f"reference_path: {REFERENCE_PATH}",
        f"output_path: {OUTPUT_PATH}",
    ]
    for key, value in stats.items():
        lines.append(f"{key}: {value}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "clean_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def read_stock_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df.columns = [str(column).strip().lower() for column in df.columns]

    if "date" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "date"})
        else:
            raise ValueError(f"Missing 'date' or 'time' column: {path}")

    missing_columns = [column for column in OUTPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    return df[OUTPUT_COLUMNS].copy()


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["symbol"] = (
        normalized["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    )
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    return normalized


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    for column in NUMERIC_COLUMNS:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def build_reference() -> pd.DataFrame:
    reference = coerce_numeric(normalize_keys(read_stock_csv(REFERENCE_PATH)))
    reference = reference.dropna(subset=[*KEY_COLUMNS, *NUMERIC_COLUMNS])
    reference = reference.drop_duplicates(subset=KEY_COLUMNS, keep="first")
    return reference.sort_values(KEY_COLUMNS).reset_index(drop=True)


def log_dirty_artifacts(
    dirty: pd.DataFrame,
    reference: pd.DataFrame,
    valid_symbols: set[str],
    stats: Dict[str, int],
) -> pd.DataFrame:
    dirty = coerce_numeric(normalize_keys(dirty))

    invalid_date_mask = dirty["date"].isna()
    stats["invalid_date_rows"] = int(invalid_date_mask.sum())
    save_log_df(dirty.loc[invalid_date_mask], "removed_invalid_date.csv")
    dirty = dirty.loc[~invalid_date_mask].copy()

    invalid_symbol_mask = (
        dirty["symbol"].isna()
        | dirty["symbol"].isin(DIRTY_SYMBOLS)
        | ~dirty["symbol"].isin(valid_symbols)
    )
    stats["invalid_symbol_rows"] = int(invalid_symbol_mask.sum())
    save_log_df(dirty.loc[invalid_symbol_mask], "removed_invalid_symbol.csv")
    dirty = dirty.loc[~invalid_symbol_mask].copy()

    missing_numeric_mask = dirty[NUMERIC_COLUMNS].isna().any(axis=1)
    stats["missing_numeric_rows"] = int(missing_numeric_mask.sum())
    save_log_df(dirty.loc[missing_numeric_mask], "dirty_missing_numeric.csv")

    non_positive_price_mask = (dirty[PRICE_COLUMNS] <= 0).any(axis=1)
    stats["non_positive_price_rows"] = int(non_positive_price_mask.sum())
    save_log_df(dirty.loc[non_positive_price_mask], "dirty_non_positive_prices.csv")

    negative_volume_mask = dirty["volume"].notna() & (dirty["volume"] < 0)
    stats["negative_volume_rows"] = int(negative_volume_mask.sum())
    save_log_df(dirty.loc[negative_volume_mask], "dirty_negative_volume.csv")

    valid_ohlc = dirty[PRICE_COLUMNS].notna().all(axis=1)
    invalid_ohlc_mask = valid_ohlc & (
        (dirty["high"] < dirty["low"])
        | (dirty["high"] < dirty["open"])
        | (dirty["high"] < dirty["close"])
        | (dirty["low"] > dirty["open"])
        | (dirty["low"] > dirty["close"])
    )
    stats["invalid_ohlc_rows"] = int(invalid_ohlc_mask.sum())
    save_log_df(dirty.loc[invalid_ohlc_mask], "dirty_invalid_ohlc.csv")

    duplicate_key_mask = dirty.duplicated(subset=KEY_COLUMNS, keep=False)
    stats["duplicate_symbol_date_rows"] = int(duplicate_key_mask.sum())
    save_log_df(
        dirty.loc[duplicate_key_mask].sort_values(KEY_COLUMNS),
        "dirty_duplicate_symbol_date.csv",
    )

    compared = dirty.merge(
        reference,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_dirty", "_reference"),
    )
    changed_cells = []
    for column in NUMERIC_COLUMNS:
        dirty_values = compared[f"{column}_dirty"]
        reference_values = compared[f"{column}_reference"]
        mismatch = dirty_values.isna() != reference_values.isna()
        both_present = dirty_values.notna() & reference_values.notna()
        mismatch |= both_present & ~np.isclose(
            dirty_values,
            reference_values,
            rtol=0,
            atol=1e-9,
        )
        if mismatch.any():
            changed = compared.loc[mismatch, KEY_COLUMNS].copy()
            changed["column"] = column
            changed["dirty_value"] = dirty_values.loc[mismatch].to_numpy()
            changed["reference_value"] = reference_values.loc[mismatch].to_numpy()
            changed_cells.append(changed)

    if changed_cells:
        changed_df = pd.concat(changed_cells, ignore_index=True)
        stats["restored_numeric_cells"] = len(changed_df)
        save_log_df(changed_df, "restored_numeric_cells.csv")
    else:
        stats["restored_numeric_cells"] = 0

    return dirty


def restore_from_reference(
    dirty: pd.DataFrame,
    reference: pd.DataFrame,
    stats: Dict[str, int],
) -> pd.DataFrame:
    dirty_unique = dirty.drop_duplicates(subset=KEY_COLUMNS, keep="last").copy()
    reference_max_date = reference["date"].max()

    merged = dirty_unique.merge(
        reference,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("_dirty", "_reference"),
        indicator=True,
    )

    matched_reference_mask = merged["_merge"] == "both"
    new_data_mask = (~matched_reference_mask) & (merged["date"] > reference_max_date)
    removed_unmatched_mask = (~matched_reference_mask) & (~new_data_mask)

    removed_unmatched = merged.loc[removed_unmatched_mask, KEY_COLUMNS]
    save_log_df(removed_unmatched, "removed_keys_not_in_reference.csv")
    stats["removed_keys_not_in_reference"] = int(len(removed_unmatched))

    kept_new_keys = merged.loc[new_data_mask, KEY_COLUMNS]
    save_log_df(kept_new_keys, "kept_new_keys_not_in_reference.csv")
    stats["new_keys_not_in_reference_kept"] = int(len(kept_new_keys))

    merged = merged.loc[matched_reference_mask | new_data_mask].copy()
    output = merged[KEY_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        output[column] = merged[f"{column}_reference"].where(
            matched_reference_mask.loc[merged.index],
            merged[f"{column}_dirty"],
        )

    return output[OUTPUT_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)


def filter_final_clean_rows(df: pd.DataFrame, stats: Dict[str, int]) -> pd.DataFrame:
    clean = df.copy()

    missing_numeric_mask = clean[NUMERIC_COLUMNS].isna().any(axis=1)
    stats["final_missing_numeric_rows_removed"] = int(missing_numeric_mask.sum())
    save_log_df(clean.loc[missing_numeric_mask], "final_removed_missing_numeric.csv")
    clean = clean.loc[~missing_numeric_mask].copy()

    non_positive_price_mask = (clean[PRICE_COLUMNS] <= 0).any(axis=1)
    stats["final_non_positive_price_rows_removed"] = int(
        non_positive_price_mask.sum()
    )
    save_log_df(
        clean.loc[non_positive_price_mask],
        "final_removed_non_positive_prices.csv",
    )
    clean = clean.loc[~non_positive_price_mask].copy()

    negative_volume_mask = clean["volume"].notna() & (clean["volume"] < 0)
    stats["final_negative_volume_rows_removed"] = int(negative_volume_mask.sum())
    save_log_df(clean.loc[negative_volume_mask], "final_removed_negative_volume.csv")
    clean = clean.loc[~negative_volume_mask].copy()

    invalid_ohlc_mask = (
        (clean["high"] < clean["low"])
        | (clean["high"] < clean["open"])
        | (clean["high"] < clean["close"])
        | (clean["low"] > clean["open"])
        | (clean["low"] > clean["close"])
    )
    stats["final_invalid_ohlc_rows_removed"] = int(invalid_ohlc_mask.sum())
    save_log_df(clean.loc[invalid_ohlc_mask], "final_removed_invalid_ohlc.csv")
    clean = clean.loc[~invalid_ohlc_mask].copy()

    duplicate_key_mask = clean.duplicated(subset=KEY_COLUMNS, keep="first")
    stats["final_duplicate_keys_removed"] = int(duplicate_key_mask.sum())
    save_log_df(clean.loc[duplicate_key_mask], "final_removed_duplicate_keys.csv")
    clean = clean.loc[~duplicate_key_mask].copy()

    return clean.sort_values(KEY_COLUMNS).reset_index(drop=True)


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    output = df[OUTPUT_COLUMNS].copy()
    for column in PRICE_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce").round(2)
    output["volume"] = (
        pd.to_numeric(output["volume"], errors="coerce").round(0).astype("Int64")
    )
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return output


def main() -> None:
    dirty_raw = read_stock_csv(INPUT_PATH)
    reference = build_reference()
    valid_symbols = set(reference["symbol"].dropna().astype(str))

    stats: Dict[str, int] = {
        "dirty_rows_in": len(dirty_raw),
        "reference_rows": len(reference),
        "reference_symbols": len(valid_symbols),
    }

    dirty = log_dirty_artifacts(dirty_raw, reference, valid_symbols, stats)
    stats["valid_dirty_rows_after_symbol_date_filter"] = len(dirty)
    stats["valid_dirty_unique_keys"] = len(dirty[KEY_COLUMNS].drop_duplicates())

    restored = restore_from_reference(dirty, reference, stats)
    stats["rows_restored_before_final_rules"] = len(restored)
    final_clean = filter_final_clean_rows(restored, stats)
    output = format_output(final_clean)

    stats["rows_out"] = len(output)
    stats["rows_removed_vs_dirty"] = len(dirty_raw) - len(output)
    stats["rows_missing_vs_reference"] = len(reference) - len(output)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, chunksize=5_000)
    write_summary(stats)

    print(f"Saved clean file: {OUTPUT_PATH}")
    print(f"Saved clean logs: {LOG_DIR}")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
