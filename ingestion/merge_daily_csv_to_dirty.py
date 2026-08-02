from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAILY_DIR = PROJECT_ROOT / "data" / "dirty"
DEFAULT_DIRTY_CSV = PROJECT_ROOT / "data" / "dirty" / "Data_500_stocks_dirty.csv"
DIRTY_COLUMNS = ["time", "open", "high", "low", "close", "volume", "symbol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge new daily crawled stock CSV files into the dirty dataset."
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=DEFAULT_DAILY_DIR,
        help="Folder containing stock_*.csv files created by main_cao_dl.py.",
    )
    parser.add_argument(
        "--dirty-csv",
        type=Path,
        default=DEFAULT_DIRTY_CSV,
        help="Dirty dataset CSV that receives new daily rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be merged without writing to the dirty CSV.",
    )
    return parser.parse_args()


def load_existing_index(dirty_csv: Path):
    if not dirty_csv.exists():
        return set(), set()

    existing = pd.read_csv(dirty_csv, usecols=["time", "symbol"])
    existing["time"] = pd.to_datetime(existing["time"], errors="coerce")
    existing = existing.dropna(subset=["time", "symbol"])
    existing["symbol"] = existing["symbol"].astype(str).str.strip().str.upper()

    existing_dates = set(existing["time"].dt.date)
    existing_keys = set(
        zip(
            existing["symbol"],
            existing["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    return existing_dates, existing_keys


def normalize_daily_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip() for column in df.columns]

    if "status" in df.columns and set(df["status"].dropna().astype(str)) == {"no_data"}:
        print(f"[merge] Skip no_data file: {csv_path.name}")
        return pd.DataFrame(columns=DIRTY_COLUMNS)

    if "time" not in df.columns and "trading_date" in df.columns:
        df = df.rename(columns={"trading_date": "time"})

    missing_columns = [column for column in DIRTY_COLUMNS if column not in df.columns]
    if missing_columns:
        print(f"[merge] Skip {csv_path.name}: missing columns {missing_columns}")
        return pd.DataFrame(columns=DIRTY_COLUMNS)

    df = df[DIRTY_COLUMNS].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time", "symbol", "open", "high", "low", "close"])
    df = df[df["symbol"] != ""]
    df = df.drop_duplicates(subset=["symbol", "time"], keep="last")
    return df


def collect_new_rows(daily_dir: Path, existing_dates: set, existing_keys: set):
    daily_files = sorted(daily_dir.glob("stock_*.csv"))
    if not daily_files:
        print(f"[merge] No stock_*.csv files found in: {daily_dir}")
        return pd.DataFrame(columns=DIRTY_COLUMNS), []

    frames = []
    merged_files = []

    for csv_path in daily_files:
        df = normalize_daily_csv(csv_path)
        if df.empty:
            continue

        df["_date"] = df["time"].dt.date
        new_dates = sorted(set(df["_date"]) - existing_dates)
        if not new_dates:
            print(f"[merge] Skip {csv_path.name}: dates already exist in dirty CSV")
            continue

        df = df[df["_date"].isin(new_dates)].copy()
        df["_key"] = list(
            zip(
                df["symbol"],
                df["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        df = df[~df["_key"].isin(existing_keys)].copy()

        if df.empty:
            print(f"[merge] Skip {csv_path.name}: rows already exist")
            continue

        df = df.drop(columns=["_date", "_key"])
        df = df[DIRTY_COLUMNS]
        frames.append(df)
        merged_files.append((csv_path.name, new_dates, len(df)))

        existing_dates.update(new_dates)
        existing_keys.update(
            zip(
                df["symbol"],
                df["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

    if not frames:
        return pd.DataFrame(columns=DIRTY_COLUMNS), merged_files

    new_rows = pd.concat(frames, ignore_index=True)
    new_rows = new_rows.drop_duplicates(subset=["symbol", "time"], keep="last")
    new_rows = new_rows.sort_values(["time", "symbol"]).reset_index(drop=True)
    return new_rows, merged_files


def append_to_dirty(dirty_csv: Path, rows: pd.DataFrame) -> None:
    dirty_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.copy()
    rows["time"] = rows["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows.to_csv(
        dirty_csv,
        mode="a",
        header=not dirty_csv.exists(),
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    daily_dir = args.daily_dir.resolve()
    dirty_csv = args.dirty_csv.resolve()

    print(f"[merge] Daily CSV dir: {daily_dir}")
    print(f"[merge] Dirty CSV: {dirty_csv}")

    existing_dates, existing_keys = load_existing_index(dirty_csv)
    print(f"[merge] Existing dirty dates: {len(existing_dates):,}")

    new_rows, merged_files = collect_new_rows(
        daily_dir=daily_dir,
        existing_dates=existing_dates,
        existing_keys=existing_keys,
    )

    for file_name, dates, row_count in merged_files:
        date_text = ", ".join(str(date) for date in dates)
        print(f"[merge] New data from {file_name}: {row_count:,} rows ({date_text})")

    if new_rows.empty:
        print("[merge] No new daily rows to merge. Dirty CSV is unchanged.")
        return

    print(f"[merge] Total new rows: {len(new_rows):,}")
    if args.dry_run:
        print("[merge] Dry run only. Dirty CSV was not changed.")
        return

    append_to_dirty(dirty_csv, new_rows)
    print(f"[merge] Appended new rows to: {dirty_csv}")


if __name__ == "__main__":
    main()
