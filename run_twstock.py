#!/usr/bin/env python3
import sys

from twstock import BestFourPoint, Stock
from twstock import stock as stock_mod


def _patch_twse_fetcher():
    """Trim unexpected extra fields from TWSE data rows."""
    original = stock_mod.TWSEFetcher._make_datatuple

    def _make_datatuple_compat(self, data):
        if len(data) > len(stock_mod.DATATUPLE._fields):
            data = data[: len(stock_mod.DATATUPLE._fields)]
        return original(self, data)

    stock_mod.TWSEFetcher._make_datatuple = _make_datatuple_compat


def _linear_regression(xs, ys):
    n = len(xs)
    if n < 2:
        return None, None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None, None
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _local_extrema(points, window=3, mode="high"):
    extrema = []
    for i in range(window, len(points) - window):
        center = points[i]
        left = points[i - window : i]
        right = points[i + 1 : i + 1 + window]
        if mode == "high":
            if center > max(left) and center > max(right):
                extrema.append((i, center))
        else:
            if center < min(left) and center < min(right):
                extrema.append((i, center))
    return extrema


def _detect_patterns(prices, lookback=60):
    if len(prices) < lookback:
        return {"error": "not_enough_data"}

    window = prices[-lookback:]
    highs = _local_extrema(window, window=3, mode="high")
    lows = _local_extrema(window, window=3, mode="low")

    result = {"trendline": None, "triangle": None}
    if len(highs) >= 2:
        xs, ys = zip(*highs[-6:])
        slope, _ = _linear_regression(xs, ys)
        result["trendline"] = ("downtrend" if slope < 0 else "uptrend", slope)
    if len(lows) >= 2:
        xs, ys = zip(*lows[-6:])
        slope, _ = _linear_regression(xs, ys)
        if result["trendline"] is None:
            result["trendline"] = ("downtrend" if slope < 0 else "uptrend", slope)

    if len(highs) >= 2 and len(lows) >= 2:
        hx, hy = zip(*highs[-6:])
        lx, ly = zip(*lows[-6:])
        h_slope, h_int = _linear_regression(hx, hy)
        l_slope, l_int = _linear_regression(lx, ly)
        if h_slope is not None and l_slope is not None:
            mid = len(window) - 1
            gap_start = (h_slope * 0 + h_int) - (l_slope * 0 + l_int)
            gap_end = (h_slope * mid + h_int) - (l_slope * mid + l_int)
            if h_slope < 0 and l_slope > 0 and gap_end < gap_start:
                result["triangle"] = ("symmetrical", gap_start, gap_end)
            elif h_slope < 0 and l_slope >= 0 and gap_end < gap_start:
                result["triangle"] = ("descending", gap_start, gap_end)
            elif h_slope <= 0 and l_slope > 0 and gap_end < gap_start:
                result["triangle"] = ("ascending", gap_start, gap_end)

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_twstock.py <stock_id>")
        sys.exit(1)

    stock_id = sys.argv[1]
    _patch_twse_fetcher()

    stock = Stock(stock_id)
    bfp = BestFourPoint(stock)

    print("latest_price", stock.price[-5:])
    print("latest_capacity", stock.capacity[-5:])
    print("best_four_point_buy", bfp.best_four_point_to_buy())
    print("best_four_point_sell", bfp.best_four_point_to_sell())
    print("best_four_point", bfp.best_four_point())
    patterns = _detect_patterns(stock.price, lookback=60)
    if patterns.get("error") == "not_enough_data":
        print("pattern_error not_enough_data (need 60 data points)")
    else:
        print("trendline", patterns.get("trendline"))
        print("triangle", patterns.get("triangle"))


if __name__ == "__main__":
    main()
