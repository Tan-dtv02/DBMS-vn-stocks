from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VN_TZ = timezone(timedelta(hours=7))

DIRTY_DIR = PROJECT_ROOT / "data" / "dirty"
CLEAN_CSV = PROJECT_ROOT / "data" / "clean" / "Data_500_stocks_clean_ver2.csv"
DIRTY_CSV = PROJECT_ROOT / "data" / "dirty" / "Data_500_stocks_dirty.csv"
PIPELINE_SUMMARY = PROJECT_ROOT / "data" / "pipeline_run_summary.json"

MODEL_STEPS = {
    "train_model1": "model1",
    "train_model2": "model2",
    "train_model3": "model3",
    "train_model4": "model4",
    "train_model5": "model5",
}

CLICKHOUSE_STEPS = {
    "load_prices",
    "load_symbols",
    "upload_features_all",
    "upload_model4_outputs",
    "upload_model5_outputs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send workflow alerts to Telegram.")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--phase", choices=["start", "end"], required=True)
    parser.add_argument("--status", default="")
    return parser.parse_args()


def target_date() -> datetime.date:
    value = os.getenv("TARGET_DATE")
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return datetime.now(VN_TZ).date() - timedelta(days=1)


def daily_csv_path() -> Path:
    date = target_date()
    return DIRTY_DIR / f"stock_{date.day}_{date.month}.csv"


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return 0

        if "status" in reader.fieldnames:
            return sum(
                1
                for row in reader
                if str(row.get("status", "")).strip().lower() != "no_data"
            )
        return sum(1 for _ in reader)


def load_pipeline_summary() -> dict:
    if not PIPELINE_SUMMARY.exists():
        return {}

    try:
        return json.loads(PIPELINE_SUMMARY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_url() -> str:
    server_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if repository and run_id:
        return f"{server_url}/{repository}/actions/runs/{run_id}"
    return ""


def format_count(value: int | None) -> str:
    return "khong tim thay file" if value is None else f"{value:,}"


def status_text(phase: str, status: str) -> str:
    if phase == "start":
        return "bat dau"
    return f"ket thuc - {status or 'unknown'}"


def build_cao_dl_message(phase: str, status: str) -> str:
    date = target_date()
    csv_path = daily_csv_path()
    rows = count_csv_rows(csv_path) if phase == "end" else None

    lines = [
        f"Workflow: cao_dl",
        f"Trang thai: {status_text(phase, status)}",
        f"Ngay cao: {date.isoformat()}",
        f"File CSV: data/dirty/{csv_path.name}",
    ]

    if phase == "end":
        lines.append(f"So dong cao: {format_count(rows)}")
        if status == "success":
            lines.append("Commit CSV: thanh cong neu co thay doi")

    url = run_url()
    if url:
        lines.append(f"Run: {url}")
    return "\n".join(lines)


def build_pipeline_message(phase: str, status: str) -> str:
    lines = [
        "Workflow: merge_daily_csv_and_run_pipeline",
        f"Trang thai: {status_text(phase, status)}",
    ]

    if phase == "end":
        summary = load_pipeline_summary()
        completed_steps = set(summary.get("completed_steps", []))
        completed_models = [
            model_name
            for step, model_name in MODEL_STEPS.items()
            if step in completed_steps
        ]
        clean_rows = count_csv_rows(CLEAN_CSV)
        dirty_rows = count_csv_rows(DIRTY_CSV)

        lines.extend(
            [
                "Model da chay xong: "
                + (", ".join(completed_models) if completed_models else "chua co"),
                f"File clean cuoi: data/clean/{CLEAN_CSV.name}",
                f"Tong dong clean: {format_count(clean_rows)}",
                f"Tong dong dirty: {format_count(dirty_rows)}",
            ]
        )

        clickhouse_done = CLICKHOUSE_STEPS.issubset(completed_steps)
        if status == "success" and clickhouse_done:
            lines.append("ClickHouse: upload thanh cong")
        else:
            failed_step = summary.get("failed_step")
            lines.append(
                "ClickHouse: chua hoan tat"
                + (f" (dung o step {failed_step})" if failed_step else "")
            )

    url = run_url()
    if url:
        lines.append(f"Run: {url}")
    return "\n".join(lines)


def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[telegram] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Skip alert.")
        print(text)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")

    request = urllib.request.Request(url, data=payload, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
        print("[telegram] Alert sent.")
    except Exception as exc:
        print(f"[telegram] Could not send alert: {exc}")
        print(text)


def main() -> None:
    args = parse_args()
    workflow = args.workflow.strip()
    status = args.status.strip().lower()

    if workflow == "cao_dl":
        message = build_cao_dl_message(args.phase, status)
    elif workflow == "merge_daily_csv_and_run_pipeline":
        message = build_pipeline_message(args.phase, status)
    else:
        message = f"Workflow: {workflow}\nTrang thai: {args.phase} - {status}"

    send_telegram_message(message)


if __name__ == "__main__":
    main()
