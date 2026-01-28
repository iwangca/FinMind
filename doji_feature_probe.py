#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


FEATURE_COLS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "kd_k",
    "kd_d",
    "bias_5",
    "bias_10",
    "bias_20",
    "bias_60",
    "bias_120",
    "volume_ratio_5",
    "atr_14",
    "volatility_20",
]

CHIP_DATASETS = [
    "institutional",
    "margin",
    "shareholding",
    "holding_shares_per",
    "securities_lending",
    "per_pbr",
]


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


def load_chip_csv(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    keep = {"date", "stock_id"}
    rename = {}
    for col in df.columns:
        if col in keep:
            continue
        rename[col] = f"{prefix}_{col}"
    df = df.rename(columns=rename)
    df = df.dropna(subset=["date"])
    return df


def merge_chip_data(df: pd.DataFrame, stock_id: str, base_dir: Path) -> pd.DataFrame:
    out = df
    for ds in CHIP_DATASETS:
        path = base_dir / ds / f"{stock_id}.csv"
        if not path.exists():
            continue
        chip_df = load_chip_csv(path, ds)
        if chip_df.empty:
            continue
        out = out.merge(chip_df, on=["date", "stock_id"], how="left")
    return out


def run_trades_with_features(df: pd.DataFrame, max_ticks: int, stop_ticks: int):
    rows = []
    if df.empty:
        return rows
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        open_price = float(row["open"])
        close_price = float(row["close"])
        if close_price < open_price:
            continue
        ticks = tick_size(close_price)
        if close_price - open_price > max_ticks * ticks:
            continue
        entry = close_price
        if entry <= 0:
            continue
        stop_price = entry - stop_ticks * ticks
        low_next = float(next_row["min"])
        close_next = float(next_row["close"])
        if low_next <= stop_price:
            exit_price = stop_price
            exit_type = "stop"
        else:
            exit_price = close_next
            exit_type = "close"
        ret = (exit_price - entry) / entry
        record = {
            "entry_date": row["date"].date().isoformat(),
            "exit_date": next_row["date"].date().isoformat(),
            "entry": entry,
            "exit": exit_price,
            "return": ret,
            "exit_type": exit_type,
        }
        for col in FEATURE_COLS:
            if col in df.columns:
                record[col] = row[col]
        rows.append(record)
    return rows


def summarize_feature_diff(df: pd.DataFrame):
    numeric_cols = [c for c in FEATURE_COLS if c in df.columns]
    for col in df.columns:
        if col in ("entry_date", "exit_date", "entry", "exit", "return", "exit_type", "stock_id"):
            continue
        if col in FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    if not numeric_cols:
        return pd.DataFrame()
    win = df[df["return"] > 0]
    loss = df[df["return"] <= 0]
    stats = []
    for col in numeric_cols:
        win_mean = pd.to_numeric(win[col], errors="coerce").mean()
        loss_mean = pd.to_numeric(loss[col], errors="coerce").mean()
        diff = win_mean - loss_mean
        stats.append((col, win_mean, loss_mean, diff))
    out = pd.DataFrame(stats, columns=["feature", "win_mean", "loss_mean", "diff"])
    out = out.sort_values("diff", ascending=False)
    return out


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
        "--limit",
        type=int,
        default=10,
        help="Top N features to show",
    )
    parser.add_argument(
        "--chip-dir",
        default="data/finmind",
        help="Base directory with chip datasets",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    all_rows = []
    for path in files:
        df = load_price_csv(path)
        df = merge_chip_data(df, path.stem, Path(args.chip_dir))
        trades = run_trades_with_features(df, args.max_ticks, args.stop_ticks)
        for t in trades:
            t["stock_id"] = path.stem
        all_rows.extend(trades)

    if not all_rows:
        print("No trades found")
        return

    trades_df = pd.DataFrame(all_rows)
    stats = summarize_feature_diff(trades_df)
    print(f"trades: {len(trades_df)}")
    print(f"win_rate: {(trades_df['return'] > 0).mean() * 100:.2f}%")
    if stats.empty:
        print("no feature columns available for analysis")
        return
    print("\nTop features (win_mean - loss_mean):")
    for _, row in stats.head(args.limit).iterrows():
        print(
            f"{row['feature']}: win_mean={row['win_mean']:.4f} "
            f"loss_mean={row['loss_mean']:.4f} diff={row['diff']:.4f}"
        )


if __name__ == "__main__":
    main()
