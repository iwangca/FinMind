#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from FinMind.data import DataLoader

DATASET_CONFIG = {
    "price": {"method": "taiwan_stock_daily", "with_tech": True},
    "price_adj": {"method": "taiwan_stock_daily_adj", "with_tech": True},
    "institutional": {"method": "taiwan_stock_institutional_investors"},
    "margin": {"method": "taiwan_stock_margin_purchase_short_sale", "with_ratios": True},
    "shareholding": {"method": "taiwan_stock_shareholding"},
    "holding_shares_per": {"method": "taiwan_stock_holding_shares_per"},
    "securities_lending": {"method": "taiwan_stock_securities_lending"},
    "per_pbr": {"method": "taiwan_stock_per_pbr"},
}


def _load_env_tokens(dotenv_path: str):
    env_map = {}
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                env_map[key] = value
    env_map = {**os.environ, **env_map}

    if env_map.get("FINMIND_TOKENS"):
        return [t.strip() for t in env_map["FINMIND_TOKENS"].split(",") if t.strip()]
    if env_map.get("FINMIND_TOKEN"):
        return [env_map["FINMIND_TOKEN"].strip()]
    return []


def _five_years_ago(today: date) -> date:
    try:
        return today.replace(year=today.year - 5)
    except ValueError:
        return today.replace(year=today.year - 5, month=2, day=28)


def _month_chunks(start: date, end: date, months: int = 1):
    if months < 1:
        raise ValueError("months must be >= 1")
    current = start
    while current <= end:
        anchor = current.replace(day=1)
        for _ in range(months):
            anchor = (anchor + timedelta(days=32)).replace(day=1)
        period_end = min(end, anchor - timedelta(days=1))
        yield current, period_end
        current = period_end + timedelta(days=1)


def _read_stock_ids(path: str):
    with open(path, "r") as f:
        items = [line.strip() for line in f if line.strip()]
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_progress(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_progress(path: str, progress: dict):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=True)


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _df_last_date(df: pd.DataFrame):
    if "date" not in df.columns:
        return None
    parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.isna().all():
        return None
    return parsed.max().date()


def _rsi(series: pd.Series, window: int = 14):
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _kd(df: pd.DataFrame, window: int = 9, k_smooth: int = 3, d_smooth: int = 3):
    low = df["min"].astype(float)
    high = df["max"].astype(float)
    close = pd.to_numeric(df["close"], errors="coerce")
    low_n = low.rolling(window).min()
    high_n = high.rolling(window).max()
    rsv = ((close - low_n) / (high_n - low_n).replace(0, pd.NA)) * 100
    k = rsv.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def _atr(df: pd.DataFrame, window: int = 14):
    high = df["max"].astype(float)
    low = df["min"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _volatility(series: pd.Series, window: int = 20):
    returns = series.pct_change()
    return returns.rolling(window).std()


def _add_tech_indicators(df: pd.DataFrame):
    if "date" not in df.columns or "close" not in df.columns:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    close = df["close"].astype(float)
    for window in (5, 10, 20, 60, 120):
        ma = close.rolling(window).mean()
        df[f"ma_{window}"] = ma.round(4)
        df[f"bias_{window}"] = ((close - ma) / ma).round(6)
    df["rsi_14"] = _rsi(close, 14).round(4)
    macd_line, signal_line, hist = _macd(close)
    df["macd"] = macd_line.round(6)
    df["macd_signal"] = signal_line.round(6)
    df["macd_hist"] = hist.round(6)
    if {"min", "max"}.issubset(df.columns):
        k, d = _kd(df)
        df["kd_k"] = k.round(4)
        df["kd_d"] = d.round(4)
        df["atr_14"] = _atr(df, 14).round(6)
    df["volatility_20"] = _volatility(close, 20).round(6)
    if "Trading_Volume" in df.columns:
        vol = df["Trading_Volume"].astype(float)
        df["volume_ratio_5"] = (vol / vol.rolling(5).mean()).round(6)
    return df


def _add_margin_ratios(df: pd.DataFrame):
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    if {"MarginPurchaseTodayBalance", "ShortSaleTodayBalance"}.issubset(df.columns):
        mp = df["MarginPurchaseTodayBalance"].astype(float)
        ss = df["ShortSaleTodayBalance"].astype(float)
        denom = (mp + ss).replace(0, pd.NA)
        df["margin_short_ratio"] = (mp / denom).round(6)
    if {"MarginPurchaseBuy", "MarginPurchaseSell"}.issubset(df.columns):
        mp_buy = df["MarginPurchaseBuy"].astype(float)
        mp_sell = df["MarginPurchaseSell"].astype(float)
        denom = (mp_buy + mp_sell).replace(0, pd.NA)
        df["margin_buy_ratio"] = (mp_buy / denom).round(6)
    if {"ShortSaleBuy", "ShortSaleSell"}.issubset(df.columns):
        ss_buy = df["ShortSaleBuy"].astype(float)
        ss_sell = df["ShortSaleSell"].astype(float)
        denom = (ss_buy + ss_sell).replace(0, pd.NA)
        df["short_sell_ratio"] = (ss_sell / denom).round(6)
    return df


class TokenRotator:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0

    def current(self):
        if not self.tokens:
            return ""
        return self.tokens[self.index % len(self.tokens)]

    def rotate(self):
        if not self.tokens:
            return ""
        self.index = (self.index + 1) % len(self.tokens)
        return self.current()


def _fetch_with_rotation(
    data_loader: DataLoader,
    rotator: TokenRotator,
    fetch_fn,
    *args,
    max_retries_per_token: int = 3,
    context: str = "",
    skip_on_error: bool = False,
    error_log: str = "",
    **kwargs,
):
    last_exc = None
    attempts = max(1, len(rotator.tokens))
    for _ in range(attempts):
        for retry in range(max_retries_per_token):
            try:
                return fetch_fn(*args, **kwargs)
            except KeyError as exc:
                last_exc = exc
                wait = min(5 + retry * 2, 15)
                print(
                    f"warning: response missing 'data', retry in {wait}s"
                    f"{' (' + context + ')' if context else ''}"
                )
                time.sleep(wait)
            except Exception as exc:
                last_exc = exc
                break
        if not rotator.tokens:
            break
        data_loader.login_by_token(rotator.rotate())
        time.sleep(1)
    if skip_on_error:
        if error_log and context:
            _ensure_dir(os.path.dirname(error_log))
            with open(error_log, "a") as f:
                f.write(f"{datetime.now().isoformat()} {context} {last_exc}\n")
        return pd.DataFrame()
    raise last_exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock-file",
        default="stock_ids_futures.txt",
        help="Path to stock id list file",
    )
    parser.add_argument(
        "--datasets",
        default="price,price_adj,per_pbr,institutional,margin,shareholding,holding_shares_per,securities_lending",
        help="Comma-separated datasets",
    )
    parser.add_argument(
        "--start-date",
        default="",
        help="YYYY-MM-DD (default: 5 years ago)",
    )
    parser.add_argument(
        "--end-date",
        default="",
        help="YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/finmind",
        help="Output directory",
    )
    parser.add_argument(
        "--progress-file",
        default="data/finmind/progress.json",
        help="Progress state file",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=("csv", "parquet"),
        help="Output format",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=6,
        help="Months per request chunk",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between requests",
    )
    parser.add_argument(
        "--dotenv",
        default=".env",
        help="Path to .env with FINMIND_TOKENS",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Skip failed requests and continue",
    )
    parser.add_argument(
        "--error-log",
        default="data/finmind/error.log",
        help="Error log path when skipping failures",
    )
    args = parser.parse_args()

    tokens = _load_env_tokens(args.dotenv)
    rotator = TokenRotator(tokens)
    data_loader = DataLoader(token=rotator.current())

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {d}")

    stock_ids = _read_stock_ids(args.stock_file)
    if not stock_ids:
        raise ValueError("No stock ids found")

    today = date.today()
    start = _parse_date(args.start_date) if args.start_date else _five_years_ago(today)
    end = _parse_date(args.end_date) if args.end_date else today

    progress = _load_progress(args.progress_file)
    progress.setdefault("datasets", {})

    base_interval = 6.0 / max(len(tokens), 1)
    min_interval = max(args.min_interval, base_interval if tokens else 0.0)
    last_request_time = 0.0

    for dataset in datasets:
        dataset_cfg = DATASET_CONFIG[dataset]
        method = getattr(data_loader, dataset_cfg["method"])
        progress["datasets"].setdefault(dataset, {})
        dataset_dir = os.path.join(args.output_dir, dataset)
        _ensure_dir(dataset_dir)

        for stock_id in stock_ids:
            last_done = progress["datasets"][dataset].get(stock_id, "")
            start_date = _parse_date(last_done) + timedelta(days=1) if last_done else start
            if start_date > end:
                continue

            for chunk_start, chunk_end in _month_chunks(
                start_date, end, months=args.chunk_months
            ):
                wait = min_interval - (time.time() - last_request_time)
                if wait > 0:
                    time.sleep(wait)

                df = _fetch_with_rotation(
                    data_loader,
                    rotator,
                    method,
                    stock_id=stock_id,
                    start_date=chunk_start.strftime("%Y-%m-%d"),
                    end_date=chunk_end.strftime("%Y-%m-%d"),
                    context=f"{dataset} {stock_id} {chunk_start} {chunk_end}",
                    skip_on_error=args.skip_on_error,
                    error_log=args.error_log,
                )
                last_request_time = time.time()

                if df is None or df.empty:
                    continue

                if dataset_cfg.get("with_tech"):
                    df = _add_tech_indicators(df)
                if dataset_cfg.get("with_ratios"):
                    df = _add_margin_ratios(df)

                out_path = os.path.join(dataset_dir, f"{stock_id}.{args.format}")
                if args.format == "csv":
                    write_header = not os.path.exists(out_path)
                    df.to_csv(out_path, mode="a", header=write_header, index=False)
                else:
                    try:
                        if os.path.exists(out_path):
                            existing = pd.read_parquet(out_path)
                            df = pd.concat([existing, df], ignore_index=True)
                            df = df.drop_duplicates()
                        df.to_parquet(out_path, index=False)
                    except ImportError as exc:
                        raise SystemExit(
                            "Parquet requires pyarrow or fastparquet."
                        ) from exc

                last_date = _df_last_date(df)
                if last_date:
                    progress["datasets"][dataset][stock_id] = last_date.strftime(
                        "%Y-%m-%d"
                    )
                    progress["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    _save_progress(args.progress_file, progress)


if __name__ == "__main__":
    main()
