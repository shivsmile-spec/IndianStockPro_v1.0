# Indian Stock Pro v1.0 — Ranking Engine

This adds a transparent research engine to the PWA.

## What it calculates
- 5/20/60 trading-day momentum
- 20/50/200-day trend
- Relative strength versus NIFTY 50
- Volume expansion
- RSI quality
- 20-day-high proximity
- ATR-based volatility
- ATR-based risk/reward proxy
- Explainable 0–100 combined score

## Important
The engine uses Yahoo Finance data through `yfinance` as a free-data prototype. It is not an NSE-certified real-time feed. NSE states that its real-time and historical market-data products are supplied under its data products/licensing framework.

The model is a research signal and must not be represented as a guaranteed prediction.

## GitHub Action
`update_rankings.yml` runs on weekdays and writes `data/rankings.json`.
