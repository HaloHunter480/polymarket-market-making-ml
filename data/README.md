# Training Data

Raw data files are excluded from git (too large / proprietary).
Place them here before running training scripts.

---

## Dataset 1 — BTC/USDT 1-Second Level-2 Order Book

**File:** `BTC_1sec.csv` (place in project root or `data/`)  
**Size:** ~1.6 GB, ~2.6M rows  
**Date range:** April 2021 onwards  
**Used by:** `train_probability_model.py`, `train_hmm_regime.py`

### Format (150+ columns):
```
system_time, midpoint, spread, buys, sells,
bids_distance_0..14, bids_notional_0..14,
bids_cancel_notional_0..14, bids_limit_notional_0..14, bids_market_notional_0..14,
asks_distance_0..14, asks_notional_0..14, ...
```

### Key columns used for training:
| Column | Description |
|--------|-------------|
| `system_time` | UTC timestamp at 1-second resolution |
| `midpoint` | BTC/USDT mid price |
| `spread` | bid-ask spread |
| `buys` | taker buy volume (1s bucket) |
| `sells` | taker sell volume (1s bucket) |

### Source:
This dataset was obtained from Kaggle. Search for BTC/USDT 1-second order book datasets, e.g.:
- https://www.kaggle.com/datasets (search: "BTC 1 second order book")

Alternatively, generate equivalent data using Binance 1s klines:
```bash
python3 download_btc_1s.py --days 30 --merge
```
This downloads from `https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/`
and produces `btc_1sec.csv` in standard Binance OHLCV format (compatible with all training scripts).

---

## Dataset 2 — BTC/USDT 5-Minute OHLCV (2017–2025)

**File:** Downloaded automatically via `kaggle.py`  
**Used by:** `train_hmm_regime.py` (HMM regime classifier)

### Download:
```bash
python3 kaggle.py
# Downloads: shivaverse/btcusdt-5-minute-ohlc-volume-data-2017-2025
# Requires: pip install kagglehub
```

---

## Quick Start

```bash
# Option A: Use Binance public API (free, no account needed)
python3 download_btc_1s.py --days 30 --merge
# → produces btc_1sec.csv (~500MB for 30 days)

# Option B: Use BTC_1sec.csv (full L2 order book, richer OFI features)
# Place BTC_1sec.csv in project root, then:

# Train all models
python3 train_probability_model.py --data btc_1sec.csv
python3 train_hmm_regime.py --data btc_1sec.csv
python3 warmup.py
```

---

## Feature Engineering (13 features → ML ensemble input)

From raw 1s data, `train_probability_model.py` computes:

| Feature | Description |
|---------|-------------|
| `ret_1s` | 1-second log return (%) |
| `ret_10s` | 10-second return (%) |
| `ret_30s` | 30-second return (%) |
| `ret_60s` | 60-second return (%) |
| `ret_120s` | 120-second return (%) |
| `realized_vol_60s` | Realized volatility over 60s (std of log returns) |
| `realized_vol_10s` | Short-horizon vol (10s) |
| `vol_30s` | Realized vol over 30s |
| `vol_120s` | Realized vol over 120s |
| `ofi` | Order Flow Imbalance = (buys − sells) / (buys + sells) |
| `spread_pct` | Bid-ask spread as % of mid price |
| `dist_strike` | (price − strike) / strike × 100 |
| `time_to_expiry` | Seconds remaining in 5-min window (0–300) |

**Target:** `Y = 1 if BTC_close[t+300s] > strike else 0`
