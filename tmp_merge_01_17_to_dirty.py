from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_CSV = PROJECT_ROOT / "data" / "dirty" / "Data_500_stocks_01-17.csv"
DIRTY_CSV = PROJECT_ROOT / "data" / "dirty" / "Data_500_stocks_dirty.csv"
DIRTY_COLUMNS = ["time", "open", "high", "low", "close", "volume", "symbol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporarily merge Data_500_stocks_01-17.csv into dirty dataset."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print merge counts; do not modify the dirty CSV.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file before appending rows.",
    )
    return parser.parse_args()


def normalize(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df.columns = [str(column).strip() for column in df.columns]
    missing_columns = [column for column in DIRTY_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{source_name} missing columns: {missing_columns}")

    df = df[DIRTY_COLUMNS].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time", "symbol", "open", "high", "low", "close"])
    df = df[df["symbol"] != ""]
    df = df.drop_duplicates(subset=["symbol", "time"], keep="last")
    return df


def make_keys(df: pd.DataFrame) -> pd.Series:
    return (
        df["symbol"].astype(str).str.strip().str.upper()
        + "|"
        + df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    )


def main() -> None:
    args = parse_args()

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")
    if not DIRTY_CSV.exists():
        raise FileNotFoundError(f"Dirty CSV not found: {DIRTY_CSV}")

    print(f"[tmp-merge] Source: {SOURCE_CSV}")
    print(f"[tmp-merge] Dirty:  {DIRTY_CSV}")

    source_df = normalize(pd.read_csv(SOURCE_CSV), SOURCE_CSV.name)
    dirty_index = pd.read_csv(DIRTY_CSV, usecols=["time", "symbol"])
    dirty_index["time"] = pd.to_datetime(dirty_index["time"], errors="coerce")
    dirty_index = dirty_index.dropna(subset=["time", "symbol"])

    existing_keys = set(make_keys(dirty_index))
    source_df["_key"] = make_keys(source_df)

    new_rows = source_df[~source_df["_key"].isin(existing_keys)].copy()
    new_rows = new_rows.drop(columns=["_key"])
    new_rows = new_rows.sort_values(["time", "symbol"]).reset_index(drop=True)

    print(f"[tmp-merge] Source valid rows: {len(source_df):,}")
    print(f"[tmp-merge] Existing dirty keys: {len(existing_keys):,}")
    print(f"[tmp-merge] New rows to append: {len(new_rows):,}")

    if new_rows.empty:
        print("[tmp-merge] Nothing to merge. Dirty CSV is unchanged.")
        return

    print(
        "[tmp-merge] New date range: "
        f"{new_rows['time'].min().date()} -> {new_rows['time'].max().date()}"
    )

    if args.dry_run:
        print("[tmp-merge] Dry run only. Dirty CSV was not changed.")
        return

    if not args.no_backup:
        backup_path = DIRTY_CSV.with_suffix(DIRTY_CSV.suffix + ".bak")
        shutil.copy2(DIRTY_CSV, backup_path)
        print(f"[tmp-merge] Backup created: {backup_path}")

    new_rows["time"] = new_rows["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    new_rows[DIRTY_COLUMNS].to_csv(
        DIRTY_CSV,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
    )
    print(f"[tmp-merge] Appended rows to dirty CSV: {len(new_rows):,}")


if __name__ == "__main__":
    main()
