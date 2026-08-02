import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL2_ROOT = Path(__file__).resolve().parents[0]
REPORTS_DIR =  MODEL2_ROOT / "reports"
DASHBOARD_PATH = REPORTS_DIR / "model2_dashboard.html"

METRICS_PATH = REPORTS_DIR / "metrics.json"
BACKTEST_METRICS_PATH = REPORTS_DIR / "backtest_metrics.json"
BACKTEST_PATH = REPORTS_DIR / "backtest.csv"
BACKTEST_SWEEP_PATH = REPORTS_DIR / "backtest_sweep.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
PREDICTIONS_PATH = REPORTS_DIR / "prediction_error.csv"
INSIGHTS_PATH = REPORTS_DIR / "insights.json"


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


def format_number(value, digits=4):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def format_pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def format_volume(value):
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def return_badge(value):
    if value is None or pd.isna(value):
        return '<span class="return-badge neutral">N/A</span>'

    value = float(value)

    if value >= 0.03:
        css_class = "strong-positive"
    elif value >= 0.01:
        css_class = "positive"
    elif value > -0.01:
        css_class = "neutral"
    elif value > -0.03:
        css_class = "negative"
    else:
        css_class = "strong-negative"

    return f'<span class="return-badge {css_class}">{value:.2%}</span>'


def metric_card(title, value, caption=""):
    caption_html = f"<span>{html.escape(caption)}</span>" if caption else ""

    return (
        '<section class="metric-card">'
        f"<p>{html.escape(title)}</p>"
        f"<strong>{html.escape(str(value))}</strong>"
        f"{caption_html}"
        "</section>"
    )


def prepare_predictions():
    columns = [
        "trading_date",
        "symbol",
        "actual_future_return_5d",
        "predicted_future_return_5d",
        "prediction_error",
        "abs_error",
        "signal",
        "close",
        "volume",
        "encode_sector",
    ]

    predictions = read_existing_columns(PREDICTIONS_PATH, columns)

    if predictions.empty:
        return predictions

    numeric_cols = [
        "actual_future_return_5d",
        "predicted_future_return_5d",
        "prediction_error",
        "abs_error",
        "close",
        "volume",
        "encode_sector",
    ]

    for col in numeric_cols:
        if col in predictions:
            predictions[col] = pd.to_numeric(predictions[col], errors="coerce")

    if "trading_date" in predictions:
        predictions["trading_date"] = pd.to_datetime(
            predictions["trading_date"],
            errors="coerce"
        )

    return predictions


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


def make_backtest_line_chart(backtest):
    if backtest.empty or "cumulative_return" not in backtest.columns:
        return '<p class="empty">Backtest curve is not available.</p>'

    series = pd.to_numeric(
        backtest["cumulative_return"],
        errors="coerce"
    ).dropna()

    if series.empty:
        return '<p class="empty">Backtest curve is not available.</p>'

    width, height, pad = 780, 300, 34

    min_y = float(series.min())
    max_y = float(series.max())

    if min_y == max_y:
        max_y = min_y + 1

    x_values = np.linspace(pad, width - pad, len(series))
    y_scaled = height - pad - (
        (series.to_numpy(dtype=float) - min_y) / (max_y - min_y)
    ) * (height - 2 * pad)

    points = " ".join(
        f"{x:.2f},{y:.2f}"
        for x, y in zip(x_values, y_scaled)
    )

    y0 = height - pad

    return (
        '<div class="line-chart">'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Backtest curve">'
        f'<line x1="{pad}" y1="{y0}" x2="{width - pad}" y2="{y0}" stroke="#cfd8dc" />'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{y0}" stroke="#cfd8dc" />'
        f'<text x="{pad}" y="22">{max_y:.1%}</text>'
        f'<text x="{pad}" y="{height - 8}">{min_y:.1%}</text>'
        f'<polyline points="{points}" fill="none" stroke="#2a9d8f" '
        f'stroke-width="3" stroke-linejoin="round" />'
        "</svg>"
        '<div class="legend"><span><i style="background:#2a9d8f"></i>Cumulative Return</span></div>'
        "</div>"
    )


def make_prediction_distribution(predictions):
    if predictions.empty or "predicted_future_return_5d" not in predictions.columns:
        return '<p class="empty">Prediction distribution is not available.</p>'

    values = predictions["predicted_future_return_5d"].dropna()

    if values.empty:
        return '<p class="empty">Prediction distribution is not available.</p>'

    bins = [
        -np.inf,
        -0.05,
        -0.03,
        -0.01,
        0.01,
        0.03,
        0.05,
        np.inf,
    ]

    labels = [
        "< -5%",
        "-5% to -3%",
        "-3% to -1%",
        "-1% to 1%",
        "1% to 3%",
        "3% to 5%",
        "> 5%",
    ]

    distribution = pd.cut(values, bins=bins, labels=labels).value_counts()
    distribution = distribution.reindex(labels, fill_value=0)

    max_count = distribution.max() or 1

    rows = []

    for label, count in distribution.items():
        width = max(2, count / max_count * 100)

        rows.append(
            '<div class="bar-row">'
            f"<span>{html.escape(str(label))}</span>"
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.2f}%;background:#2a9d8f"></div>'
            "</div>"
            f"<b>{int(count)}</b>"
            "</div>"
        )

    return '<div class="bar-chart">' + "".join(rows) + "</div>"


def make_top_prediction_table(predictions, ascending=False, limit=10):
    if predictions.empty:
        return '<p class="empty">Prediction data is not available.</p>'

    required = {
        "trading_date",
        "symbol",
        "predicted_future_return_5d",
        "actual_future_return_5d",
        "abs_error",
    }

    if not required.issubset(predictions.columns):
        return '<p class="empty">Required prediction columns are missing.</p>'

    table_df = predictions.sort_values(
        "predicted_future_return_5d",
        ascending=ascending
    ).head(limit)

    rows = []

    for _, row in table_df.iterrows():
        date = row.get("trading_date")
        date_text = date.strftime("%Y-%m-%d") if pd.notna(date) else "N/A"

        rows.append(
            "<tr>"
            f"<td>{html.escape(date_text)}</td>"
            f"<td><b>{html.escape(str(row.get('symbol', 'N/A')))}</b></td>"
            f"<td>{return_badge(row.get('predicted_future_return_5d'))}</td>"
            f"<td>{return_badge(row.get('actual_future_return_5d'))}</td>"
            f"<td>{format_pct(row.get('abs_error'))}</td>"
            "</tr>"
        )

    header = (
        "<th>Date</th>"
        "<th>Symbol</th>"
        "<th>Predicted Return 5D</th>"
        "<th>Actual Return 5D</th>"
        "<th>Abs Error</th>"
    )

    return (
        '<div class="table-wrap">'
        '<table class="data-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def make_error_table(predictions, limit=10):
    if predictions.empty or "abs_error" not in predictions.columns:
        return '<p class="empty">Error data is not available.</p>'

    table_df = predictions.sort_values(
        "abs_error",
        ascending=False
    ).head(limit)

    rows = []

    for _, row in table_df.iterrows():
        date = row.get("trading_date")
        date_text = date.strftime("%Y-%m-%d") if pd.notna(date) else "N/A"

        rows.append(
            "<tr>"
            f"<td>{html.escape(date_text)}</td>"
            f"<td><b>{html.escape(str(row.get('symbol', 'N/A')))}</b></td>"
            f"<td>{return_badge(row.get('predicted_future_return_5d'))}</td>"
            f"<td>{return_badge(row.get('actual_future_return_5d'))}</td>"
            f"<td>{format_pct(row.get('prediction_error'))}</td>"
            f"<td>{format_pct(row.get('abs_error'))}</td>"
            "</tr>"
        )

    header = (
        "<th>Date</th>"
        "<th>Symbol</th>"
        "<th>Predicted</th>"
        "<th>Actual</th>"
        "<th>Error</th>"
        "<th>Abs Error</th>"
    )

    return (
        '<div class="table-wrap">'
        '<table class="data-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def make_backtest_sweep_table():
    sweep = read_csv(BACKTEST_SWEEP_PATH)

    if sweep.empty:
        return '<p class="empty">Backtest sweep is not available.</p>'

    columns = [
        "top_n",
        "avg_return",
        "win_rate",
        "max_drawdown",
        "sharpe_ratio",
    ]

    table_df = sweep[[col for col in columns if col in sweep.columns]].copy()

    if table_df.empty:
        return '<p class="empty">Backtest sweep is not available.</p>'

    head = "".join(
        f"<th>{html.escape(col)}</th>"
        for col in table_df.columns
    )

    body_rows = []

    for _, row in table_df.iterrows():
        cells = []

        for col in table_df.columns:
            value = row[col]

            if col in ["avg_return", "win_rate", "max_drawdown"]:
                value = f"{float(value):.2%}"
            elif isinstance(value, float):
                value = f"{value:.4f}"

            cells.append(
                f"<td>{html.escape(str(value))}</td>"
            )

        body_rows.append(
            "<tr>" + "".join(cells) + "</tr>"
        )

    return (
        '<table class="data-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def make_insights_panel(predictions, metrics, backtest_metrics, insights):
    items = []

    r2 = metrics.get("r2_score")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")

    if r2 is not None:
        items.append(
            f"Model đạt R² = {format_number(r2, 4)}, "
            f"MAE = {format_number(mae, 6)}, "
            f"RMSE = {format_number(rmse, 6)}."
        )

    avg_return = backtest_metrics.get("avg_return")
    win_rate = backtest_metrics.get("win_rate")
    sharpe = backtest_metrics.get("sharpe_ratio")

    if avg_return is not None:
        items.append(
            f"Backtest có lợi suất trung bình {format_pct(avg_return)}, "
            f"win rate {format_pct(win_rate)}, "
            f"Sharpe Ratio {format_number(sharpe, 4)}."
        )

    if not predictions.empty and "predicted_future_return_5d" in predictions.columns:
        top_row = predictions.sort_values(
            "predicted_future_return_5d",
            ascending=False
        ).iloc[0]

        items.append(
            f"Cổ phiếu có predicted return cao nhất là "
            f"{html.escape(str(top_row.get('symbol')))} "
            f"với mức dự đoán {float(top_row.get('predicted_future_return_5d')):.2%}."
        )

    top_features = insights.get("top_10_features", [])

    if top_features:
        first_feature = top_features[0]
        items.append(
            f"Feature quan trọng nhất là "
            f"{html.escape(str(first_feature.get('feature')))}."
        )

    if not items:
        return '<p class="empty">Insights are not available.</p>'

    return (
        '<ul class="insight-list">'
        + "".join(f"<li>{item}</li>" for item in items)
        + "</ul>"
    )


def build_dashboard_html():
    metrics = load_json(METRICS_PATH)
    backtest_metrics = load_json(BACKTEST_METRICS_PATH)
    insights = load_json(INSIGHTS_PATH)

    feature_importance = read_csv(FEATURE_IMPORTANCE_PATH)
    backtest = read_csv(BACKTEST_PATH)
    predictions = prepare_predictions()

    cards = [
        metric_card(
            "MAE",
            format_number(metrics.get("mae"), 6),
            "Mean Absolute Error"
        ),
        metric_card(
            "RMSE",
            format_number(metrics.get("rmse"), 6),
            "Root Mean Squared Error"
        ),
        metric_card(
            "R² Score",
            format_number(metrics.get("r2_score"), 4),
            "Regression fitness"
        ),
        metric_card(
            "MAPE",
            format_pct(metrics.get("mape")),
            "Mean Absolute Percentage Error"
        ),
        metric_card(
            "Avg Backtest Return",
            format_pct(backtest_metrics.get("avg_return")),
            "Average actual return"
        ),
        metric_card(
            "Win Rate",
            format_pct(backtest_metrics.get("win_rate")),
            "Positive return ratio"
        ),
        metric_card(
            "Max Drawdown",
            format_pct(backtest_metrics.get("max_drawdown")),
            "Worst drawdown"
        ),
        metric_card(
            "Sharpe Ratio",
            format_number(backtest_metrics.get("sharpe_ratio"), 4),
            "Return / Risk"
        ),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model 2 Future Return Prediction Dashboard</title>
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
      --blue: #1d3557;
    }}

    * {{
      box-sizing: border-box;
    }}

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

    header h1 {{
      margin: 0;
      font-size: 28px;
    }}

    header p {{
      margin: 6px 0 0;
      color: var(--muted);
    }}

    main {{
      padding: 22px clamp(18px, 4vw, 44px) 40px;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }}

    .metric-card,
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}

    .metric-card {{
      padding: 16px;
      min-height: 116px;
    }}

    .metric-card p {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 13px;
    }}

    .metric-card strong {{
      display: block;
      font-size: 28px;
    }}

    .metric-card span {{
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}

    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
      gap: 14px;
      margin-top: 14px;
    }}

    .panel {{
      padding: 18px;
      overflow: hidden;
    }}

    .panel h2 {{
      margin: 0 0 14px;
      font-size: 18px;
    }}

    .bar-row {{
      display: grid;
      grid-template-columns: minmax(92px, 160px) 1fr 72px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
      font-size: 13px;
    }}

    .bar-row span {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .bar-track {{
      height: 12px;
      background: #e8eef2;
      border-radius: 999px;
      overflow: hidden;
    }}

    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}

    .line-chart svg {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .line-chart text {{
      fill: var(--muted);
      font-size: 12px;
    }}

    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      color: var(--muted);
      font-size: 13px;
    }}

    .legend i {{
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 50%;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
    }}

    th,
    td {{
      border-bottom: 1px solid var(--line);
      padding: 10px;
      text-align: left;
    }}

    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}

    .table-wrap {{
      overflow-x: auto;
    }}

    .return-badge {{
      display: inline-flex;
      align-items: center;
      min-width: 78px;
      justify-content: center;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
    }}

    .strong-positive {{
      color: #0f766e;
      background: #ccfbf1;
      border: 1px solid #99f6e4;
    }}

    .positive {{
      color: #15803d;
      background: #dcfce7;
      border: 1px solid #bbf7d0;
    }}

    .neutral {{
      color: #6c757d;
      background: #f1f3f5;
      border: 1px solid #dee2e6;
    }}

    .negative {{
      color: #b45309;
      background: #fef3c7;
      border: 1px solid #fde68a;
    }}

    .strong-negative {{
      color: #b91c1c;
      background: #fee2e2;
      border: 1px solid #fecaca;
    }}

    .insight-list {{
      margin: 0;
      padding-left: 18px;
    }}

    .insight-list li {{
      margin: 9px 0;
      color: var(--ink);
    }}

    .empty {{
      color: var(--muted);
      margin: 0;
    }}

    @media (max-width: 900px) {{
      .metric-grid,
      .grid-2 {{
        grid-template-columns: 1fr;
      }}

      .bar-row {{
        grid-template-columns: 96px 1fr 56px;
      }}
    }}
  </style>
</head>

<body>
  <header>
    <h1>Model 2 Future Return Prediction Dashboard</h1>
    <p>LightGBM Regressor for predicting 5-day future stock return, generated from the latest files in reports/.</p>
  </header>

  <main>
    <section class="metric-grid">
      {"".join(cards)}
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Backtest Curve</h2>
        {make_backtest_line_chart(backtest)}
      </div>

      <div class="panel">
        <h2>Predicted Return Distribution</h2>
        {make_prediction_distribution(predictions)}
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Top Predicted Return Stocks</h2>
        {make_top_prediction_table(predictions, ascending=False)}
      </div>

      <div class="panel">
        <h2>Lowest Predicted Return Stocks</h2>
        {make_top_prediction_table(predictions, ascending=True)}
      </div>
    </section>

    <section class="panel" style="margin-top:14px">
      <h2>Highest Prediction Errors</h2>
      {make_error_table(predictions)}
    </section>

    <section class="panel" style="margin-top:14px">
      <h2>Insights For Report</h2>
      {make_insights_panel(predictions, metrics, backtest_metrics, insights)}
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Top Feature Importance</h2>
        {make_bar_chart(feature_importance, "feature", "importance")}
      </div>

      <div class="panel">
        <h2>Best Backtest Configurations</h2>
        {make_backtest_sweep_table()}
      </div>
    </section>
  </main>
</body>
</html>
"""


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(
        build_dashboard_html(),
        encoding="utf-8"
    )

    print(f"Dashboard saved to {DASHBOARD_PATH}")


def create_dashboard():
    main()


if __name__ == "__main__":
    main()