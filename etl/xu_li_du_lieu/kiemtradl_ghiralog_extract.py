from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "dirty" / "Data_500_stocks_dirty.csv"
REFERENCE_PATH = PROJECT_ROOT / "data" / "clean" / "Data_500_stocks_2015-2026.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "khaosatdata"

EXPECTED_COLUMNS = ["time", "open", "high", "low", "close", "volume", "symbol"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume"]
DIRTY_SYMBOLS = {"???", "123", "NULL", "A@", "ZZZZZZ"}


def ensure_output_dir() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, filename: str) -> Path:
	path = OUTPUT_DIR / filename
	df.to_csv(path, index=False)
	return path


def to_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
	return pd.to_numeric(df[column], errors="coerce")


def load_valid_symbols() -> set[str]:
	if not REFERENCE_PATH.exists():
		return set()
	reference = pd.read_csv(REFERENCE_PATH, usecols=["symbol"])
	return set(reference["symbol"].astype("string").str.strip().str.upper().dropna())


def add_section(report_lines: List[str], title: str) -> None:
	report_lines.append("=" * 72)
	report_lines.append(title)
	report_lines.append("-" * 72)


def check_missing(df: pd.DataFrame, report_lines: List[str]) -> None:
	add_section(report_lines, "Missing values")
	missing_summary = df.isna().sum().sort_values(ascending=False).reset_index()
	missing_summary.columns = ["column", "missing_count"]
	missing_summary["missing_ratio"] = (
		missing_summary["missing_count"] / len(df)
	).round(6)
	save_df(missing_summary, "missing_summary.csv")

	missing_rows = df[df.isna().any(axis=1)]
	if not missing_rows.empty:
		save_df(missing_rows, "rows_with_missing.csv")
	report_lines.append(f"Rows with any missing: {len(missing_rows)}")
	report_lines.append("Saved: missing_summary.csv, rows_with_missing.csv")


def check_duplicates(df: pd.DataFrame, report_lines: List[str]) -> None:
	add_section(report_lines, "Duplicates")
	dup_full_mask = df.duplicated(keep=False)
	full_duplicates = df.loc[dup_full_mask]
	if not full_duplicates.empty:
		save_df(full_duplicates, "duplicate_full_rows.csv")
	report_lines.append(f"Duplicate full rows: {dup_full_mask.sum()}")

	if {"symbol", "time"}.issubset(df.columns):
		dup_key_mask = df.duplicated(subset=["symbol", "time"], keep=False)
		dup_key_rows = df.loc[dup_key_mask].sort_values(["symbol", "time"])
		if not dup_key_rows.empty:
			save_df(dup_key_rows, "duplicate_symbol_time.csv")
		report_lines.append(f"Duplicate rows (symbol+time): {dup_key_mask.sum()}")
	else:
		report_lines.append("Skipped symbol+time duplicates: missing columns")


def check_time(df: pd.DataFrame, report_lines: List[str]) -> Optional[pd.Series]:
	add_section(report_lines, "Time column")
	if "time" not in df.columns:
		report_lines.append("Missing column: time")
		return None

	parsed_time = pd.to_datetime(df["time"], errors="coerce")
	invalid_time_rows = df.loc[parsed_time.isna()]
	if not invalid_time_rows.empty:
		save_df(invalid_time_rows, "invalid_time.csv")
	report_lines.append(f"Invalid time values: {len(invalid_time_rows)}")

	valid_time = parsed_time.dropna()
	if not valid_time.empty:
		report_lines.append(f"Time range: {valid_time.min()} -> {valid_time.max()}")
	report_lines.append("Saved: invalid_time.csv")
	return parsed_time


def check_symbol(
	df: pd.DataFrame, report_lines: List[str], valid_symbols: set[str]
) -> None:
	add_section(report_lines, "Symbol column")
	if "symbol" not in df.columns:
		report_lines.append("Missing column: symbol")
		return

	symbol_series = df["symbol"].astype("string").str.strip().str.upper()
	invalid_format_mask = symbol_series.isna() | ~symbol_series.str.match(
		r"^[A-Z0-9]+$", na=False
	)
	dirty_marker_mask = symbol_series.isin(DIRTY_SYMBOLS)
	if valid_symbols:
		unknown_symbol_mask = ~symbol_series.isin(valid_symbols)
	else:
		unknown_symbol_mask = pd.Series(False, index=df.index)
	invalid_symbol_mask = invalid_format_mask | dirty_marker_mask | unknown_symbol_mask
	invalid_symbol_rows = df.loc[invalid_symbol_mask]
	if not invalid_symbol_rows.empty:
		save_df(invalid_symbol_rows, "invalid_symbol.csv")
	report_lines.append(f"Invalid symbol values: {len(invalid_symbol_rows)}")
	report_lines.append(f"Invalid format symbols: {int(invalid_format_mask.sum())}")
	report_lines.append(f"Known dirty marker symbols: {int(dirty_marker_mask.sum())}")
	report_lines.append(f"Symbols outside reference: {int(unknown_symbol_mask.sum())}")

	symbol_counts = (
		symbol_series.value_counts().rename_axis("symbol").reset_index(name="row_count")
	)
	save_df(symbol_counts, "symbol_counts.csv")
	report_lines.append(f"Unique symbols: {symbol_counts.shape[0]}")
	report_lines.append("Saved: invalid_symbol.csv, symbol_counts.csv")


def check_numeric(df: pd.DataFrame, report_lines: List[str]) -> Dict[str, pd.Series]:
	add_section(report_lines, "Numeric columns")
	numeric_data: Dict[str, pd.Series] = {}
	invalid_numeric_rows = []

	for col in NUMERIC_COLUMNS:
		if col not in df.columns:
			report_lines.append(f"Missing column: {col}")
			continue

		numeric_series = to_numeric_series(df, col)
		numeric_data[col] = numeric_series

		invalid_mask = numeric_series.isna() & df[col].notna()
		if invalid_mask.any():
			temp = pd.DataFrame()
			if "time" in df.columns:
				temp["time"] = df.loc[invalid_mask, "time"]
			if "symbol" in df.columns:
				temp["symbol"] = df.loc[invalid_mask, "symbol"]
			temp["column"] = col
			temp["raw_value"] = df.loc[invalid_mask, col].astype(str)
			invalid_numeric_rows.append(temp)

	if invalid_numeric_rows:
		invalid_numeric_df = pd.concat(invalid_numeric_rows, ignore_index=True)
		save_df(invalid_numeric_df, "invalid_numeric_values.csv")
		report_lines.append(f"Invalid numeric values: {len(invalid_numeric_df)}")
	else:
		report_lines.append("Invalid numeric values: 0")

	numeric_df = pd.DataFrame(numeric_data)
	if not numeric_df.empty:
		stats = numeric_df.describe().transpose().reset_index()
		stats.rename(columns={"index": "column"}, inplace=True)
		save_df(stats, "numeric_stats.csv")
		report_lines.append("Saved: invalid_numeric_values.csv, numeric_stats.csv")
	return numeric_data


def check_price_rules(
	df: pd.DataFrame, numeric_data: Dict[str, pd.Series], report_lines: List[str]
) -> None:
	add_section(report_lines, "Price rules")
	missing_price_cols = [col for col in PRICE_COLUMNS if col not in numeric_data]
	if missing_price_cols:
		report_lines.append(f"Skipped price checks, missing: {missing_price_cols}")
		return

	price_df = pd.DataFrame({col: numeric_data[col] for col in PRICE_COLUMNS})
	valid_prices = price_df.notna().all(axis=1)

	non_positive_mask = (price_df <= 0).any(axis=1)
	non_positive_rows = df.loc[non_positive_mask]
	if not non_positive_rows.empty:
		save_df(non_positive_rows, "non_positive_prices.csv")
	report_lines.append(f"Rows with non-positive prices: {len(non_positive_rows)}")

	ohlc_invalid_mask = valid_prices & (
		(price_df["high"] < price_df["low"])
		| (price_df["high"] < price_df["open"])
		| (price_df["high"] < price_df["close"])
		| (price_df["low"] > price_df["open"])
		| (price_df["low"] > price_df["close"])
	)
	ohlc_invalid_rows = df.loc[ohlc_invalid_mask]
	if not ohlc_invalid_rows.empty:
		save_df(ohlc_invalid_rows, "invalid_ohlc_rows.csv")
	report_lines.append(f"Rows with invalid OHLC relations: {len(ohlc_invalid_rows)}")
	report_lines.append("Saved: non_positive_prices.csv, invalid_ohlc_rows.csv")


def check_volume(
	df: pd.DataFrame, numeric_data: Dict[str, pd.Series], report_lines: List[str]
) -> None:
	add_section(report_lines, "Volume rules")
	if "volume" not in numeric_data:
		report_lines.append("Skipped volume checks: missing volume column")
		return

	volume = numeric_data["volume"]
	negative_volume_mask = volume.notna() & (volume < 0)
	negative_volume_rows = df.loc[negative_volume_mask]
	if not negative_volume_rows.empty:
		save_df(negative_volume_rows, "negative_volume.csv")
	report_lines.append(f"Rows with negative volume: {len(negative_volume_rows)}")

	non_integer_mask = volume.notna() & (np.abs(volume - volume.round()) > 1e-6)
	non_integer_rows = df.loc[non_integer_mask]
	if not non_integer_rows.empty:
		save_df(non_integer_rows, "non_integer_volume.csv")
	report_lines.append(f"Rows with non-integer volume: {len(non_integer_rows)}")
	report_lines.append("Saved: negative_volume.csv, non_integer_volume.csv")


def check_outliers(
	df: pd.DataFrame, numeric_data: Dict[str, pd.Series], report_lines: List[str]
) -> None:
	add_section(report_lines, "Outliers (IQR)")
	if not numeric_data:
		report_lines.append("Skipped outlier checks: no numeric data")
		return

	outlier_summaries = []
	for col, series in numeric_data.items():
		series = series.dropna()
		if series.empty:
			continue
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		outlier_mask = (numeric_data[col] < lower) | (numeric_data[col] > upper)
		outlier_count = int(outlier_mask.sum())
		outlier_summaries.append(
			{
				"column": col,
				"q1": q1,
				"q3": q3,
				"iqr": iqr,
				"lower": lower,
				"upper": upper,
				"outlier_count": outlier_count,
			}
		)

		if outlier_count:
			cols_to_save = [c for c in ["time", "symbol", col] if c in df.columns]
			outlier_rows = df.loc[outlier_mask, cols_to_save]
			save_df(outlier_rows, f"outliers_{col}.csv")

	outlier_summary_df = pd.DataFrame(outlier_summaries)
	if not outlier_summary_df.empty:
		save_df(outlier_summary_df, "outlier_summary.csv")
	report_lines.append("Saved: outlier_summary.csv and outliers_<column>.csv")


def main() -> None:
	ensure_output_dir()
	df = pd.read_csv(CSV_PATH)
	valid_symbols = load_valid_symbols()

	report_lines: List[str] = []
	add_section(report_lines, "Dataset overview")
	report_lines.append(f"Source: {CSV_PATH}")
	report_lines.append(f"Reference symbols: {REFERENCE_PATH}")
	report_lines.append(f"Valid symbol count: {len(valid_symbols)}")
	report_lines.append(f"Rows: {len(df)}")
	report_lines.append(f"Columns: {df.shape[1]}")
	report_lines.append(f"Column list: {list(df.columns)}")
	report_lines.append(
		f"Missing expected columns: {[c for c in EXPECTED_COLUMNS if c not in df.columns]}"
	)
	report_lines.append(
		"Dtypes: " + ", ".join([f"{k}={v}" for k, v in df.dtypes.items()])
	)

	check_missing(df, report_lines)
	check_duplicates(df, report_lines)
	check_time(df, report_lines)
	check_symbol(df, report_lines, valid_symbols)
	numeric_data = check_numeric(df, report_lines)
	check_price_rules(df, numeric_data, report_lines)
	check_volume(df, numeric_data, report_lines)
	check_outliers(df, numeric_data, report_lines)

	report_path = OUTPUT_DIR / "khaosat_summary.txt"
	report_path.write_text("\n".join(report_lines), encoding="utf-8")
	print(f"Saved report: {report_path}")


if __name__ == "__main__":
	main()
