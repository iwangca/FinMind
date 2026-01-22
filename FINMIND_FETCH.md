FinMind Daily Fetcher

Setup
- Create `.env` with your tokens:
  - `FINMIND_TOKENS=token1,token2,token3`
  - Or use a single `FINMIND_TOKEN=token`
- Stock list: `stock_ids_futures.txt`

Basic run (5 years, daily + chip datasets)
```bash
python finmind_fetch_daily.py
```

Common options
```bash
# Choose datasets
python finmind_fetch_daily.py --datasets price,institutional,margin,shareholding

# Custom date range
python finmind_fetch_daily.py --start-date 2020-01-01 --end-date 2024-12-31

# Chunk size (months per request)
python finmind_fetch_daily.py --chunk-months 6

# Output format
python finmind_fetch_daily.py --format csv
```

Datasets
- price (TaiwanStockPrice) + technicals
- price_adj (TaiwanStockPriceAdj) + technicals
- per_pbr (TaiwanStockPER/PBR)
- institutional (TaiwanStockInstitutionalInvestors)
- margin (TaiwanStockMarginPurchaseShortSale) + ratios
- shareholding (TaiwanStockShareholding)
- holding_shares_per (TaiwanStockHoldingSharesPer)
- securities_lending (TaiwanStockSecuritiesLending)

Technical indicators (price/price_adj)
- MA: 5/10/20/60/120
- Bias: 5/10/20/60/120
- RSI(14)
- MACD(12,26,9): macd/macd_signal/macd_hist
- KD(9,3,3): kd_k/kd_d
- ATR(14): atr_14
- Volatility(20): volatility_20 (rolling std of returns)
- Volume ratio(5): volume_ratio_5

Margin ratios (margin)
- margin_short_ratio = MarginPurchaseTodayBalance / (MarginPurchaseTodayBalance + ShortSaleTodayBalance)
- margin_buy_ratio = MarginPurchaseBuy / (MarginPurchaseBuy + MarginPurchaseSell)
- short_sell_ratio = ShortSaleSell / (ShortSaleBuy + ShortSaleSell)

Resume behavior
- Progress is saved to `data/finmind/progress.json`.
- Safe to stop/restart; it continues from the last saved date per stock/dataset.
