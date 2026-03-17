# Latency Arbitrage Engine for Prediction Markets

⚠️ Research/educational project. Not financial advice or production-ready trading system.

> **Market-taking system** that exploits stale quotes on Polymarket BTC binary options.  
> When BTC moves on Binance, Polymarket's CLOB takes 2–5 seconds to reprice.  
> The system estimates short-term fair value and attempts to exploit latency-driven mispricings.

**Deployed on:** AWS EC2 eu-west-1 (Ireland) — 140ms round-trip latency observed (AWS eu-west-1)
**Language:** Python 3.12 · asyncio · fully event-driven

---

## The Edge

```
Edge = fair_value(BTC_now, strike, T, σ) − poly_mid_stale

fair_value = Φ(d₁)        [Gaussian]
           + P_merton(d₁)  [Merton jump-diffusion]   blended ⅓ each
           + P_ml(X)       [ML ensemble: Logistic + HistGBM + XGBoost]

d₁ = (BTC_now − strike) / (σ_5m × √(T/300))
σ_5m live-calibrated from Deribit implied volatility
```

---

## 6-Layer Signal Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Data Feeds                         run_live.py       │
│  Binance trades WS · Binance depth WS                           │
│  Polymarket CLOB WS · Deribit IV WS                             │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Fair Value Engine                  layer2_engine.py  │
│  Φ(d₁) Gaussian binary · OBI signal                            │
│  Cross-market arb (UP+DN=1) · Oracle-lag detector              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Stale Quote Gate                   decision_stack.py │
│  poly_mid unchanged ≥ 1s → still stale → trade allowed         │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Merton Jump-Diffusion              layer4_merton_jump.py │
│  P(S_T>K) = Σ Poisson(n;λT) × BS_binary(σ_n)                   │
│  Blended ⅓ with Gaussian fair value                             │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5 — HMM Regime Classifier              layer5_hmm_regime.py │
│  3 states: low_vol / medium_vol / high_vol                      │
│  Viterbi decoding → adaptive edge threshold (2% / 3% / 5%)     │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 6 — Kelly Sizing + Execution           layer6_risk_execution.py │
│  f* = (p·b − q)/b × vol_mult × hawkes_mult × vpin_mult         │
│  EIP-712 signed FOK market order → Polymarket CLOB              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ML Ensemble (Third Fair-Value Signal)

Trained on **BTC/USDT 1-second Level-2 order book data** (1.6 GB, ~2.6M rows).  
Three models calibrated with isotonic regression, blended by weighted average:

```
P_ml = 0.4 × LogisticRegression(C=0.1, L2)
     + 0.3 × HistGradientBoosting(max_depth=6, lr=0.05, l2=0.1, early_stop)
     + 0.3 × XGBoost(n=300, max_depth=6, α=0.1, λ=1.0)
```

**Loss:** Binary cross-entropy minimised on each sub-model independently.  
**Calibration:** `CalibratedClassifierCV(method="isotonic", cv="prefit")` on 15% val set.  
**Test metrics:** Accuracy 54.2% · Brier 0.2491 · ECE 0.0312

**13 input features:**

| Feature | Description |
|---------|-------------|
| `ret_1s / 10s / 30s / 60s / 120s` | Multi-horizon BTC momentum (%) |
| `realized_vol_10s / 30s / 60s / 120s` | Rolling realized volatility (%) |
| `ofi` | Order Flow Imbalance = (buys − sells) / total |
| `spread_pct` | Bid-ask spread as % of mid |
| `dist_strike` | (BTC − strike) / strike × 100 |
| `time_to_expiry` | Seconds remaining in 5-min window |

**Target:** `Y = 1 if BTC_close[t+300s] > strike else 0`

---

## Decision Stack — 8 Gates (`decision_stack.py`)

```
Gate 1  btc_flat          |Δbtc_10s| ≥ 0.015%  OR  |drift_from_strike| ≥ 0.015%
Gate 2  stale_check       poly_mid unchanged in last 1.0s → quote still stale
Gate 3  price_zone        0.20 ≤ p_mkt ≤ 0.80  (near-resolved = no liquidity)
Gate 4  extreme_vol       GARCH vol_regime ≠ EXTREME_VOL
Gate 5  vpin_toxic        VPIN < 0.55  (no informed-flow regime)
Gate 6  hawkes_regime     regime ∈ {QUIET, ACTIVE, EXCITED, EXPLOSIVE}
Gate 7  obi_gate          Polymarket OBI not strongly opposing direction
Gate 8  gap_threshold     edge_net ≥ edge_threshold[hmm_regime]
```

All 8 gates must pass. On `TRADE`: Kelly-sized FOK order submitted via EIP-712.

---

## Supporting Quant Models

**Hawkes Process** (`Hawkes_Process.py`) — trade-arrival intensity:
```
λ(t) = μ + Σ_{t_i < t} α · exp(−β · (t − t_i))
Regimes: QUIET · ACTIVE · EXCITED (α/β > 0.5) · EXPLOSIVE (α/β → 1)
MLE fitted on live trade timestamps; refit hourly
```

**GARCH(1,1)** (`garch.py`) — realized volatility:
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
Regimes: LOW_VOL · NORMAL_VOL · HIGH_VOL · EXTREME_VOL
```

**VPIN** (`vpin.py`) — toxic flow detection:
```
VPIN = |V_buy − V_sell| / V_total   (volume-bucket estimate)
Threshold 0.55 → informed-flow regime → halt trading
```

---

## File Map

### Live Trading
| File | Purpose | Lines |
|------|---------|-------|
| `run_live.py` | Main bot — asyncio loop, 4 WS feeds, 10 Hz eval, auto market-refresh | 1335 |
| `decision_stack.py` | 8-gate pipeline, all three fair-value signals, veto logic | 561 |
| `live_executor.py` | EIP-712 signing, `post_order(FOK)`, token cache, RiskManager | 834 |

### Layers 1–6
| File | Layer | Purpose |
|------|-------|---------|
| `triple_streams.py` | 1–6 | Integration test harness |
| `layer2_engine.py` | 2 | Gaussian fair value, OBI, cross-market arb, oracle-lag |
| `layer3_empirical_conditional.py` | 3 | Empirical P(UP\|Δ,T,regime) — kernel-smoothed |
| `layer4_merton_jump.py` | 4 | Merton jump-diffusion |
| `layer5_hmm_regime.py` | 5 | GaussianHMM regime classifier, Viterbi |
| `layer6_risk_execution.py` | 6 | Kelly fraction, position limits |

### ML Training
| File | Purpose |
|------|---------|
| `train_probability_model.py` | Train Logistic + HistGBM + XGBoost ensemble → `models/probability_model.pkl` |
| `train_hmm_regime.py` | Train GaussianHMM on Kaggle 8yr data → `models/hmm_regime.pkl` |
| `kaggle.py` | Download `shivaverse/btcusdt-5-minute-ohlc-volume-data-2017-2025` |
| `download_btc_1s.py` | Download Binance 1s klines from `data.binance.vision` |
| `warmup.py` | Cold-start GARCH + Hawkes from recent live ticks |
| `backtest_300.py` | Full strategy backtester on 1s tick data |

### Quant Models
| File | Purpose |
|------|---------|
| `Hawkes_Process.py` | Self-exciting point process |
| `garch.py` | GARCH(1,1) vol + regime labels |
| `vpin.py` | VPIN toxic flow detector |
| `orderbook.py` | Level-2 OBI tracker |
| `kalman_filter.py` | Kalman filter for price smoothing |
| `empirical_model.py` | Empirical conditional P(UP) surface |

---

## Limitations

- No strict out-of-sample validation; results may not generalize.
- Strategy depends on short-lived latency inefficiencies.
- Execution assumptions (fills, slippage) are simplified.
- Regime detection may lag during rapid market shifts.

---
  
## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up credentials
cp .env.example .env
# Edit .env with your Polymarket private key + API key

# 3. Download training data
python3 kaggle.py                            # 5-min OHLCV (for HMM)
python3 download_btc_1s.py --days 30 --merge # 1s klines (for ML ensemble)
# OR place BTC_1sec.csv (L2 order book) in project root — see data/README.md

# 4. Train models
python3 train_hmm_regime.py --data btc_1sec.csv
python3 train_probability_model.py --data btc_1sec.csv
python3 warmup.py

# 5. Paper mode (no real money)
python3 run_live.py --bankroll 20 --binance

# 6. Live mode
python3 run_live.py --live --bankroll 20 --binance

# 7. Integration test (runs all 6 layers live for 60s)
python3 triple_streams.py --layer2 --duration 60 --discover
```

**Requirements:** Python 3.12 · macOS or Ubuntu 22.04 · AWS EC2 eu-west-1 recommended

---

## Tech Stack

| Component | Library / Service |
|-----------|------------------|
| Async I/O | `asyncio`, `aiohttp`, `websockets` |
| Order signing | `py-clob-client` (EIP-712), `eth-account` |
| ML ensemble | `scikit-learn`, `xgboost` |
| HMM | `hmmlearn.GaussianHMM` |
| Statistics | `scipy.special.ndtr`, `scipy.stats` |
| Numerics | `numpy` |
| Data feeds | Binance WS · Polymarket CLOB WS · Deribit WS |
| Deployment | AWS EC2 eu-west-1 t3.micro |
| Secrets | `python-dotenv` |

---

## Key Commits

| SHA | Description |
|-----|-------------|
| `1f1b326` | feat: ML ensemble (Logistic + HistGBM + XGBoost) as third fair-value signal |
| `898af19` | perf: AWS Ireland gate tuning — MIN_BTC_MOVE_PCT, Hawkes, STRIKE_LOCK_S |
| `8550637` | fix: market order submission — `post_order(FOK)` after local EIP-712 signing |
| `5a5f025` | feat: Polymarket BTC 5-min latency-arbitrage bot Layer 1-6 initial build |

---

## License

MIT — see `LICENSE`
