# Dataset

This project uses `aapl_2014_2023.csv`: Apple (AAPL) daily stock data, 2014–2023
(2,516 rows), with precomputed technical indicators and a `target` label.

**Columns:** `date, open, high, low, close, volume, rsi_7, rsi_14, cci_7, cci_14,
sma_50, ema_50, sma_100, ema_100, macd, bollinger, TrueRange, atr_7, atr_14, target`

`target` is `1` (bullish), `0` (bearish), or `neutral` (rows removed before training).

Price data originates from Yahoo Finance. Place the CSV in this folder:

```
data/aapl_2014_2023.csv
```

The CSV is excluded from git via `.gitignore`.
