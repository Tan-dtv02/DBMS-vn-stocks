import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPORTS_DIR = Path("reports")
DASHBOARD_PATH = REPORTS_DIR / "model3_dashboard.html"

METRICS_PATH = REPORTS_DIR / "metrics.json"
BACKTEST_METRICS_PATH = REPORTS_DIR / "backtest_metrics.json"
BACKTEST_PATH = REPORTS_DIR / "backtest.csv"
BACKTEST_SWEEP_PATH = REPORTS_DIR / "backtest_sweep.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"

SIGNAL_NAMES = ["SELL", "HOLD", "BUY"]
SIGNAL_COLORS = {
    "SELL": "#d1495b",
    "HOLD": "#6c757d",
    "BUY": "#2a9d8f",
}
SIGNAL_MIN_ACTION_PROBABILITY = 0.60
SIGNAL_MIN_ACTION_MARGIN = 0.00


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path, **kwargs):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def read_existing_columns(path, columns):
    if not path.exists():
        return pd.DataFrame()
    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [col for col in columns if col in available_columns]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols)


def format_pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def format_number(value, digits=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def format_volume(value):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def signal_badge(signal):
    signal = str(signal) if signal is not None else "N/A"
    color = SIGNAL_COLORS.get(signal, "#6c757d")
    return f'<span class="signal-badge" style="--badge-color:{color}">{html.escape(signal)}</span>'


def probability_meter(value):
    if value is None or pd.isna(value):
        return '<span class="empty">N/A</span>'
    pct = min(max(float(value), 0), 1) * 100
    return (
        '<div class="prob-meter">'
        f'<span style="width:{pct:.1f}%"></span>'
        f"<b>{pct:.1f}%</b>"
        "</div>"
    )


def prepare_predictions():
    columns = [
        "trading_date",
        "symbol",
        "close",
        "volume",
        "target_signal",
        "predicted_signal",
        "adjusted_signal",
        "sell_probability",
        "hold_probability",
        "buy_probability",
        "predicted_signal_score",
        "buy_sell_margin",
        "signal_confidence",
    ]
    predictions = read_existing_columns(PREDICTIONS_PATH, columns)
    if predictions.empty:
        return predictions

    for col in ["close", "volume", "sell_probability", "hold_probability", "buy_probability"]:
        if col in predictions:
            predictions[col] = pd.to_numeric(predictions[col], errors="coerce")

    if {"sell_probability", "hold_probability", "buy_probability"}.issubset(predictions.columns):
        if "buy_sell_margin" not in predictions:
            predictions["buy_sell_margin"] = (
                predictions["buy_probability"] - predictions["sell_probability"]
            )
        if "signal_confidence" not in predictions:
            predictions["signal_confidence"] = predictions[
                ["sell_probability", "hold_probability", "buy_probability"]
            ].max(axis=1)
        if "adjusted_signal" not in predictions:
            predictions["adjusted_signal"] = "HOLD"
            buy_mask = (
                (predictions["buy_probability"] >= SIGNAL_MIN_ACTION_PROBABILITY)
                & (predictions["buy_probability"] >= predictions["hold_probability"])
                & (predictions["buy_sell_margin"] >= SIGNAL_MIN_ACTION_MARGIN)
            )
            sell_mask = (
                (predictions["sell_probability"] >= SIGNAL_MIN_ACTION_PROBABILITY)
                & (predictions["sell_probability"] >= predictions["hold_probability"])
                & ((-predictions["buy_sell_margin"]) >= SIGNAL_MIN_ACTION_MARGIN)
            )
            predictions.loc[buy_mask, "adjusted_signal"] = "BUY"
            predictions.loc[sell_mask, "adjusted_signal"] = "SELL"

    if "trading_date" in predictions:
        predictions["trading_date"] = pd.to_datetime(predictions["trading_date"], errors="coerce")

    return predictions


def signal_precision(predictions, signal_col, signal):
    if predictions.empty or {signal_col, "target_signal"}.difference(predictions.columns):
        return None, 0
    selected = predictions[predictions[signal_col] == signal]
    if selected.empty:
        return None, 0
    precision = selected["target_signal"].eq(signal).mean() * 100
    return precision, len(selected)


def metric_card(title, value, caption=""):
    caption_html = f"<span>{html.escape(caption)}</span>" if caption else ""
    return (
        '<section class="metric-card">'
        f"<p>{html.escape(title)}</p>"
        f"<strong>{html.escape(str(value))}</strong>"
        f"{caption_html}"
        "</section>"
    )


def make_confusion_matrix(metrics):
    matrix = np.array(metrics.get("Confusion_Matrix", []), dtype=float)
    if matrix.shape != (3, 3):
        return '<p class="empty">Confusion matrix is not available.</p>'

    max_value = matrix.max() or 1
    rows = []
    for row_index, row_name in enumerate(SIGNAL_NAMES):
        cells = [f"<th>{row_name}</th>"]
        row_total = matrix[row_index].sum() or 1
        for col_index, value in enumerate(matrix[row_index]):
            intensity = value / max_value
            recall_share = value / row_total * 100
            bg = f"rgba(42, 157, 143, {0.12 + intensity * 0.68:.3f})"
            cells.append(
                '<td style="background: {bg}">'
                '<b>{value:,.0f}</b><small>{share:.1f}% row</small>'
                "</td>".format(bg=bg, value=value, share=recall_share)
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    header = "".join(f"<th>{name}</th>" for name in ["Actual / Pred", *SIGNAL_NAMES])
    return (
        '<table class="confusion-matrix">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def make_bar_chart(df, label_col, value_col, color="#2a9d8f", limit=12):
    if df.empty or label_col not in df or value_col not in df:
        return '<p class="empty">No data available.</p>'

    chart_df = df[[label_col, value_col]].dropna().head(limit).copy()
    if chart_df.empty:
        return '<p class="empty">No data available.</p>'

    max_value = chart_df[value_col].max() or 1
    rows = []
    for _, row in chart_df.iterrows():
        label = html.escape(str(row[label_col]))
        value = float(row[value_col])
        width = max(2, value / max_value * 100)
        rows.append(
            '<div class="bar-row">'
            f'<span title="{label}">{label}</span>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.2f}%;background:{color}"></div>'
            "</div>"
            f"<b>{value:.4f}</b>"
            "</div>"
        )
    return '<div class="bar-chart">' + "".join(rows) + "</div>"


def make_line_chart(df):
    required = {"cumulative_return", "cumulative_return_net", "benchmark_cumulative_return"}
    if df.empty or not required.issubset(df.columns):
        return '<p class="empty">Backtest curve is not available.</p>'

    values = df[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if values.empty:
        return '<p class="empty">Backtest curve is not available.</p>'

    width, height, pad = 780, 300, 34
    min_y = float(values.min().min())
    max_y = float(values.max().max())
    if min_y == max_y:
        max_y = min_y + 1

    def points_for(series):
        y_values = series.to_numpy(dtype=float)
        x_values = np.linspace(pad, width - pad, len(y_values))
        y_scaled = height - pad - ((y_values - min_y) / (max_y - min_y)) * (height - 2 * pad)
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(x_values, y_scaled))

    lines = [
        ("cumulative_return", "#2a9d8f", "Gross"),
        ("cumulative_return_net", "#1d3557", "Net"),
        ("benchmark_cumulative_return", "#d1495b", "Benchmark"),
    ]
    polylines = []
    legend = []
    for column, color, label in lines:
        polylines.append(
            f'<polyline points="{points_for(values[column])}" '
            f'fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" />'
        )
        legend.append(f'<span><i style="background:{color}"></i>{label}</span>')

    y0 = height - pad
    return (
        '<div class="line-chart">'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Backtest curve">'
        f'<line x1="{pad}" y1="{y0}" x2="{width - pad}" y2="{y0}" stroke="#cfd8dc" />'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{y0}" stroke="#cfd8dc" />'
        f'<text x="{pad}" y="22">{max_y:.1%}</text>'
        f'<text x="{pad}" y="{height - 8}">{min_y:.1%}</text>'
        f'{"".join(polylines)}'
        "</svg>"
        f'<div class="legend">{"".join(legend)}</div>'
        "</div>"
    )


def make_signal_distribution(predictions=None):
    if predictions is None:
        usecols = ["target_signal", "predicted_signal", "adjusted_signal"]
        predictions = read_existing_columns(PREDICTIONS_PATH, usecols)
    if predictions.empty:
        return '<p class="empty">Prediction distribution is not available.</p>'

    rows = []
    series = [("target_signal", "Actual"), ("predicted_signal", "Model")]
    if "adjusted_signal" in predictions:
        series.append(("adjusted_signal", "Adjusted"))
    for column, title in series:
        counts = predictions[column].value_counts().reindex(SIGNAL_NAMES, fill_value=0)
        total = counts.sum() or 1
        items = []
        for signal, count in counts.items():
            pct = count / total * 100
            color = SIGNAL_COLORS.get(signal, "#6c757d")
            items.append(
                '<div class="dist-item">'
                f'<span><i style="background:{color}"></i>{signal}</span>'
                f"<b>{count:,}</b><small>{pct:.1f}%</small>"
                "</div>"
            )
        rows.append(f'<div class="dist-card"><h3>{title}</h3>{"".join(items)}</div>')
    return '<div class="distribution-grid">' + "".join(rows) + "</div>"


def make_market_data_table(predictions, signal, limit=10):
    if predictions.empty or "adjusted_signal" not in predictions:
        return '<p class="empty">Market data is not available.</p>'

    if signal == "BUY":
        sort_col = "buy_probability"
        signal_df = predictions[
            predictions["buy_probability"].notna()
            & (predictions["buy_probability"] >= predictions["sell_probability"])
        ].copy()
    else:
        sort_col = "sell_probability"
        signal_df = predictions[
            predictions["sell_probability"].notna()
            & (predictions["sell_probability"] >= predictions["buy_probability"])
        ].copy()

    if signal_df.empty:
        return f'<p class="empty">No {html.escape(signal)} candidates available.</p>'

    signal_df = signal_df.sort_values(sort_col, ascending=False).head(limit)
    rows = []
    for _, row in signal_df.iterrows():
        date = row.get("trading_date")
        date_text = date.strftime("%Y-%m-%d") if pd.notna(date) else "N/A"
        qualified = row.get("adjusted_signal") == signal
        status = signal if qualified else "WATCH"
        status_color = SIGNAL_COLORS.get(signal, "#6c757d") if qualified else "#6c757d"
        rows.append(
            "<tr>"
            f"<td>{html.escape(date_text)}</td>"
            f"<td><b>{html.escape(str(row.get('symbol', 'N/A')))}</b></td>"
            f'<td><span class="signal-badge" style="--badge-color:{status_color}">{status}</span></td>'
            f"<td>{format_number(row.get('close'), 2)}</td>"
            f"<td>{format_volume(row.get('volume'))}</td>"
            f"<td>{probability_meter(row.get('buy_probability'))}</td>"
            f"<td>{probability_meter(row.get('sell_probability'))}</td>"
            f"<td>{format_number(row.get('buy_sell_margin'), 3)}</td>"
            "</tr>"
        )

    header = (
        "<th>Date</th><th>Symbol</th><th>Signal</th><th>Close</th>"
        "<th>Volume</th><th>P(BUY)</th><th>P(SELL)</th><th>Margin</th>"
    )
    return (
        '<div class="table-wrap"><table class="data-table market-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def make_market_data_section(predictions):
    if predictions.empty:
        return '<p class="empty">Market data is not available.</p>'

    latest_date = predictions["trading_date"].max() if "trading_date" in predictions else pd.NaT
    if pd.notna(latest_date):
        latest_df = predictions[predictions["trading_date"] == latest_date].copy()
        date_text = latest_date.strftime("%Y-%m-%d")
    else:
        latest_df = predictions.copy()
        date_text = "N/A"

    counts = latest_df.get("adjusted_signal", pd.Series(dtype=str)).value_counts()
    summary = (
        '<div class="market-strip">'
        f'<span><b>Latest date</b>{html.escape(date_text)}</span>'
        f'<span><b>Symbols</b>{len(latest_df):,}</span>'
        f'<span><b>BUY</b>{int(counts.get("BUY", 0)):,}</span>'
        f'<span><b>HOLD</b>{int(counts.get("HOLD", 0)):,}</span>'
        f'<span><b>SELL</b>{int(counts.get("SELL", 0)):,}</span>'
        "</div>"
    )
    return (
        summary
        + '<div class="market-grid">'
        + '<section><h3>Top BUY Candidates / Watchlist</h3>'
        + make_market_data_table(latest_df, "BUY")
        + "</section>"
        + '<section><h3>Top SELL / Risk Watchlist</h3>'
        + make_market_data_table(latest_df, "SELL")
        + "</section>"
        + "</div>"
    )


def make_insights_panel(predictions, metrics, backtest_metrics):
    items = []
    accuracy = metrics.get("Accuracy")
    adjusted_accuracy = metrics.get("Adjusted_Accuracy")
    if adjusted_accuracy is not None:
        items.append(
            "Adjusted signal accuracy is "
            f"{format_pct(adjusted_accuracy)}, compared with raw model accuracy "
            f"{format_pct(accuracy)}."
        )
    elif accuracy is not None:
        items.append(f"Raw model accuracy is {format_pct(accuracy)}.")

    net_return = backtest_metrics.get("Cumulative_Return_Net")
    sharpe = backtest_metrics.get("Sharpe_Ratio_Net")
    if net_return is not None:
        items.append(
            "Backtest net return is "
            f"{float(net_return):.2%} with net Sharpe {format_number(sharpe, 3)}."
        )

    if not predictions.empty and "adjusted_signal" in predictions:
        buy_df = predictions[predictions["adjusted_signal"] == "BUY"].copy()
        if not buy_df.empty:
            top_buy = buy_df.sort_values("buy_sell_margin", ascending=False).iloc[0]
            items.append(
                "Strongest BUY candidate is "
                f"{html.escape(str(top_buy.get('symbol')))} with P(BUY) "
                f"{float(top_buy.get('buy_probability')):.1%}."
            )
        sell_df = predictions[predictions["adjusted_signal"] == "SELL"].copy()
        if not sell_df.empty:
            top_sell = sell_df.sort_values("signal_confidence", ascending=False).iloc[0]
            items.append(
                "Highest-risk SELL signal is "
                f"{html.escape(str(top_sell.get('symbol')))} with P(SELL) "
                f"{float(top_sell.get('sell_probability')):.1%}."
            )

    if not items:
        return '<p class="empty">Insights are not available.</p>'

    return '<ul class="insight-list">' + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def make_backtest_sweep_table():
    sweep = read_csv(BACKTEST_SWEEP_PATH)
    if sweep.empty:
        return '<p class="empty">Backtest sweep is not available.</p>'

    columns = [
        "Top_K",
        "Min_Volume",
        "Min_Close",
        "Min_Buy_Probability",
        "Cumulative_Return_Net",
        "Sharpe_Ratio_Net",
        "Trade_Days",
    ]
    table_df = sweep[[col for col in columns if col in sweep.columns]].head(8)
    if table_df.empty:
        return '<p class="empty">Backtest sweep is not available.</p>'

    head = "".join(f"<th>{html.escape(col)}</th>" for col in table_df.columns)
    body_rows = []
    for _, row in table_df.iterrows():
        cells = []
        for col in table_df.columns:
            value = row[col]
            if "Return" in col:
                value = f"{float(value):.2%}"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def build_dashboard_html():
    metrics = load_json(METRICS_PATH)
    backtest_metrics = load_json(BACKTEST_METRICS_PATH)
    feature_importance = read_csv(FEATURE_IMPORTANCE_PATH)
    backtest = read_csv(BACKTEST_PATH)
    predictions = prepare_predictions()
    if (
        "Adjusted_Accuracy" not in metrics
        and not predictions.empty
        and {"target_signal", "adjusted_signal"}.issubset(predictions.columns)
    ):
        metrics["Adjusted_Accuracy"] = (
            predictions["target_signal"].eq(predictions["adjusted_signal"]).mean() * 100
        )
    buy_precision, buy_count = signal_precision(predictions, "adjusted_signal", "BUY")
    sell_precision, sell_count = signal_precision(predictions, "adjusted_signal", "SELL")

    cards = [
        metric_card(
            "Accuracy",
            format_pct(metrics.get("Accuracy")),
            f"Baseline: {format_pct(metrics.get('Baseline_Accuracy'))}",
        ),
        metric_card("BUY Precision", format_pct(buy_precision), f"Adjusted BUY rows: {buy_count:,}"),
        metric_card("SELL Precision", format_pct(sell_precision), f"Adjusted SELL rows: {sell_count:,}"),
        metric_card("Macro F1", format_number(metrics.get("Macro_F1"), 4)),
        metric_card(
            "Net Return",
            f"{float(backtest_metrics.get('Cumulative_Return_Net', 0)):.2%}",
            "After transaction cost and slippage",
        ),
        metric_card(
            "Net Sharpe",
            format_number(backtest_metrics.get("Sharpe_Ratio_Net"), 3),
            f"Trade days: {backtest_metrics.get('Trade_Days', 'N/A')}",
        ),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model 3 Trading Signal Dashboard</title>
  <style>
    :root {{
      --ink: #18202a;
      --muted: #5f6c7b;
      --line: #d9e1e8;
      --bg: #f5f7f9;
      --panel: #ffffff;
      --accent: #2a9d8f;
      --risk: #d1495b;
      --warn: #f4a261;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 44px) 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    header h1 {{ margin: 0; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 6px 0 0; color: var(--muted); }}
    main {{ padding: 22px clamp(18px, 4vw, 44px) 40px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }}
    .metric-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric-card {{ padding: 16px; min-height: 116px; }}
    .metric-card p {{ margin: 0 0 10px; color: var(--muted); font-size: 13px; }}
    .metric-card strong {{ display: block; font-size: 28px; }}
    .metric-card span {{ display: block; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
      gap: 14px;
      margin-top: 14px;
    }}
    .panel {{ padding: 18px; overflow: hidden; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .confusion-matrix, .data-table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px; text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
    .confusion-matrix td {{ text-align: center; border: 1px solid #fff; }}
    .confusion-matrix b, .confusion-matrix small {{ display: block; }}
    .confusion-matrix small {{ color: var(--muted); margin-top: 3px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(92px, 150px) 1fr 72px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
      font-size: 13px;
    }}
    .bar-row span {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 12px; background: #e8eef2; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    .line-chart svg {{ width: 100%; height: auto; display: block; }}
    .line-chart text {{ fill: var(--muted); font-size: 12px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 13px; }}
    .legend i, .dist-item i {{
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 50%;
    }}
    .distribution-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .dist-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .dist-item {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
      align-items: center;
    }}
    .dist-item small {{ color: var(--muted); }}
    .market-strip {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .market-strip span {{ padding: 12px; border-right: 1px solid var(--line); }}
    .market-strip span:last-child {{ border-right: 0; }}
    .market-strip b {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .market-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .market-grid h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .table-wrap {{ overflow-x: auto; }}
    .signal-badge {{
      display: inline-flex;
      align-items: center;
      min-width: 58px;
      justify-content: center;
      padding: 4px 8px;
      border-radius: 6px;
      color: var(--badge-color);
      background: color-mix(in srgb, var(--badge-color) 12%, white);
      border: 1px solid color-mix(in srgb, var(--badge-color) 26%, white);
      font-size: 12px;
      font-weight: 700;
    }}
    .prob-meter {{
      display: grid;
      grid-template-columns: 72px 48px;
      align-items: center;
      gap: 8px;
      min-width: 128px;
    }}
    .prob-meter span {{
      display: block;
      height: 8px;
      border-radius: 999px;
      background: var(--accent);
    }}
    .prob-meter:before {{
      content: "";
      grid-column: 1;
      grid-row: 1;
      height: 8px;
      border-radius: 999px;
      background: #e8eef2;
      z-index: 0;
    }}
    .prob-meter span {{ grid-column: 1; grid-row: 1; z-index: 1; }}
    .prob-meter b {{ font-size: 12px; font-weight: 700; }}
    .insight-list {{ margin: 0; padding-left: 18px; }}
    .insight-list li {{ margin: 9px 0; color: var(--ink); }}
    .empty {{ color: var(--muted); margin: 0; }}
    @media (max-width: 900px) {{
      .metric-grid, .grid-2 {{ grid-template-columns: 1fr; }}
      .distribution-grid {{ grid-template-columns: 1fr; }}
      .market-strip, .market-grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 96px 1fr 56px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Model 3 Trading Signal Dashboard</h1>
    <p>XGBoost classifier for SELL / HOLD / BUY signals, generated from the latest files in reports/.</p>
  </header>
  <main>
    <section class="metric-grid">{"".join(cards)}</section>
    <section class="grid-2">
      <div class="panel">
        <h2>Backtest Curve</h2>
        {make_line_chart(backtest)}
      </div>
      <div class="panel">
        <h2>Signal Distribution</h2>
        {make_signal_distribution(predictions)}
      </div>
    </section>
    <section class="panel" style="margin-top:14px">
      <h2>Market Data</h2>
      {make_market_data_section(predictions)}
    </section>
    <section class="panel" style="margin-top:14px">
      <h2>Insights For Report</h2>
      {make_insights_panel(predictions, metrics, backtest_metrics)}
    </section>
    <section class="grid-2">
      <div class="panel">
        <h2>Confusion Matrix</h2>
        {make_confusion_matrix(metrics)}
      </div>
      <div class="panel">
        <h2>Top Feature Importance</h2>
        {make_bar_chart(feature_importance, "feature", "importance")}
      </div>
    </section>
    <section class="panel" style="margin-top:14px">
      <h2>Best Backtest Configurations</h2>
      {make_backtest_sweep_table()}
    </section>
  </main>
</body>
</html>
"""


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(build_dashboard_html(), encoding="utf-8")
    print(f"Dashboard saved to {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
