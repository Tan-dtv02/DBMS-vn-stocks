import json

import numpy as np
import pandas as pd


def compute_top_k_backtest(
    result_df,
    top_k=10,
    min_volume=None,
    min_close=None,
    min_predicted_return=0.0,
    transaction_cost_rate=0.0,
    slippage_rate=0.0,
):
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if transaction_cost_rate < 0 or slippage_rate < 0:
        raise ValueError("transaction_cost_rate and slippage_rate must be non-negative")

    if min_predicted_return < 0:
        raise ValueError("min_predicted_return must be non-negative")

    required_columns = ["trading_date", "symbol", "predicted_return", "target_return"]
    if min_volume is not None:
        required_columns.append("volume")
    if min_close is not None:
        required_columns.append("close")

    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise ValueError("Missing backtest columns: " + ", ".join(missing_columns))

    df = result_df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.dropna(subset=required_columns)

    if min_volume is not None:
        df = df[df["volume"] >= min_volume].copy()

    if min_close is not None:
        df = df[df["close"] >= min_close].copy()

    if df.empty:
        raise ValueError("No rows available for backtest after filtering")

    daily_rows = []
    for trading_date, day_df in df.groupby("trading_date", sort=True):
        selected_df = day_df[
            day_df["predicted_return"] >= min_predicted_return
        ].sort_values("predicted_return", ascending=False).head(top_k)

        selected_symbols = ""
        daily_return = 0.0
        avg_predicted_return = 0.0
        if not selected_df.empty:
            selected_symbols = ",".join(selected_df["symbol"].astype(str))
            daily_return = selected_df["target_return"].mean()
            avg_predicted_return = selected_df["predicted_return"].mean()

        daily_rows.append(
            {
                "trading_date": trading_date,
                "selected_symbols": selected_symbols,
                "selected_count": len(selected_df),
                "daily_return": daily_return,
                "benchmark_return": day_df["target_return"].mean(),
                "avg_predicted_return": avg_predicted_return,
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    round_trip_cost_rate = 2 * (transaction_cost_rate + slippage_rate)
    daily_df["daily_cost_rate"] = np.where(
        daily_df["selected_count"] > 0,
        round_trip_cost_rate,
        0.0,
    )
    daily_df["daily_return_net"] = (
        daily_df["daily_return"] - daily_df["daily_cost_rate"]
    )
    daily_df["cumulative_return"] = (1 + daily_df["daily_return"]).cumprod() - 1
    daily_df["cumulative_return_net"] = (
        (1 + daily_df["daily_return_net"]).cumprod() - 1
    )
    daily_df["benchmark_cumulative_return"] = (
        (1 + daily_df["benchmark_return"]).cumprod() - 1
    )

    daily_return_std = daily_df["daily_return"].std(ddof=1)
    sharpe_ratio = 0.0
    if pd.notna(daily_return_std) and daily_return_std != 0:
        sharpe_ratio = daily_df["daily_return"].mean() / daily_return_std * np.sqrt(252)

    daily_return_net_std = daily_df["daily_return_net"].std(ddof=1)
    sharpe_ratio_net = 0.0
    if pd.notna(daily_return_net_std) and daily_return_net_std != 0:
        sharpe_ratio_net = (
            daily_df["daily_return_net"].mean()
            / daily_return_net_std
            * np.sqrt(252)
        )

    metrics = {
        "Top_K": top_k,
        "Min_Volume": min_volume,
        "Min_Close": min_close,
        "Min_Predicted_Return": min_predicted_return,
        "Transaction_Cost_Rate": transaction_cost_rate,
        "Slippage_Rate": slippage_rate,
        "Round_Trip_Cost_Rate": round_trip_cost_rate,
        "Eligible_Rows": len(df),
        "Eligible_Symbols": df["symbol"].nunique(),
        "Backtest_Days": len(daily_df),
        "Trade_Days": int((daily_df["selected_count"] > 0).sum()),
        "Average_Daily_Return": float(daily_df["daily_return"].mean()),
        "Average_Daily_Return_Net": float(daily_df["daily_return_net"].mean()),
        "Cumulative_Return": float(daily_df["cumulative_return"].iloc[-1]),
        "Cumulative_Return_Net": float(daily_df["cumulative_return_net"].iloc[-1]),
        "Benchmark_Cumulative_Return": float(
            daily_df["benchmark_cumulative_return"].iloc[-1]
        ),
        "Hit_Rate": float((daily_df["daily_return"] > 0).mean() * 100),
        "Sharpe_Ratio": float(sharpe_ratio),
        "Sharpe_Ratio_Net": float(sharpe_ratio_net),
    }

    return daily_df, metrics


def run_backtest_sweep(
    result_df,
    top_k_values,
    min_volume_values,
    min_close_values,
    min_predicted_return_values,
    transaction_cost_rate=0.0,
    slippage_rate=0.0,
):
    sweep_rows = []

    for top_k in top_k_values:
        for min_volume in min_volume_values:
            for min_close in min_close_values:
                for min_predicted_return in min_predicted_return_values:
                    _, metrics = compute_top_k_backtest(
                        result_df=result_df,
                        top_k=top_k,
                        min_volume=min_volume,
                        min_close=min_close,
                        min_predicted_return=min_predicted_return,
                        transaction_cost_rate=transaction_cost_rate,
                        slippage_rate=slippage_rate,
                    )
                    sweep_rows.append(metrics)

    return pd.DataFrame(sweep_rows).sort_values(
        ["Cumulative_Return_Net", "Sharpe_Ratio_Net"],
        ascending=False,
    )


def save_backtest_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
