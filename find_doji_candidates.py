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


def load_latest_date(files):
    latest = None
    for path in files:
        try:
            df = pd.read_csv(path, usecols=["date"])
        except Exception:
            continue
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        d = df["date"].max()
        if pd.isna(d):
            continue
        if latest is None or d > latest:
            latest = d
    return latest.date() if latest is not None else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/finmind/price",
        help="Directory with price CSV files",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Target date YYYY-MM-DD (default: latest available)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=4,
        help="Max ticks between open and close to be included",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    target_date = None
    if args.date:
        target_date = pd.to_datetime(args.date).date()
    else:
        target_date = load_latest_date(files)
    if target_date is None:
        raise SystemExit("No valid dates found")

    results = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        row = df.loc[df["date"] == target_date]
        if row.empty:
            continue
        row = row.iloc[-1]
        try:
            open_price = float(row["open"])
            close_price = float(row["close"])
        except Exception:
            continue
        if close_price < open_price:
            continue
        ticks = tick_size(close_price)
        if close_price - open_price <= args.max_ticks * ticks:
            results.append((row["stock_id"], open_price, close_price))

    results.sort(key=lambda x: x[0])
    print(f"date {target_date} candidates {len(results)}")
    for stock_id, open_price, close_price in results:
        print(f"{stock_id} open={open_price} close={close_price}")


if __name__ == "__main__":
    main()
