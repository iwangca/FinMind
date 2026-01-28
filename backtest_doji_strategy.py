#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def tick_size(price: float) -> float:
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.1
    if price < 500:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


def load_price_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "close", "min"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def run_backtest(df: pd.DataFrame, max_ticks: int, stop_ticks: int):
    trades = []
    if df.empty:
        return trades

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        open_price = float(row["open"])
        close_price = float(row["close"])
        low_next = float(next_row["min"])
        close_next = float(next_row["close"])

        if close_price < open_price:
            continue
        ticks = tick_size(close_price)
        if close_price - open_price > max_ticks * ticks:
            continue

        entry = close_price
        if entry <= 0:
            continue
        stop_price = entry - stop_ticks * ticks
        if low_next <= stop_price:
            exit_price = stop_price
            exit_type = "stop"
        else:
            exit_price = close_next
            exit_type = "close"

        ret = (exit_price - entry) / entry
        trades.append(
            {
                "entry_date": row["date"].date().isoformat(),
                "exit_date": next_row["date"].date().isoformat(),
                "entry": entry,
                "exit": exit_price,
                "return": ret,
                "exit_type": exit_type,
            }
        )

    return trades


def summarize(trades):
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }
    df = pd.DataFrame(trades)
    wins = df[df["return"] > 0]
    losses = df[df["return"] <= 0]
    return {
        "trades": len(df),
        "win_rate": (len(wins) / len(df)) * 100,
        "avg_return": df["return"].mean() * 100,
        "avg_win": wins["return"].mean() * 100 if not wins.empty else 0.0,
        "avg_loss": losses["return"].mean() * 100 if not losses.empty else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/finmind/price",
        help="Directory with price CSV files",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=4,
        help="Max ticks between open and close for doji",
    )
    parser.add_argument(
        "--stop-ticks",
        type=int,
        default=2,
        help="Stop loss ticks below entry",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV for all trades",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    all_trades = []
    for path in files:
        df = load_price_csv(path)
        trades = run_backtest(df, args.max_ticks, args.stop_ticks)
        for t in trades:
            t["stock_id"] = path.stem
        all_trades.extend(trades)

    stats = summarize(all_trades)
    print(f"trades: {stats['trades']}")
    print(f"win_rate: {stats['win_rate']:.2f}%")
    print(f"avg_return: {stats['avg_return']:.4f}%")
    print(f"avg_win: {stats['avg_win']:.4f}%")
    print(f"avg_loss: {stats['avg_loss']:.4f}%")

    if args.output:
        pd.DataFrame(all_trades).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
