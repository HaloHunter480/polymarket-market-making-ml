
# Polymarket BTC Latency-Arbitrage — Market-Taking Bot

A live, real-money trading system that exploits **stale quotes on Polymarket BTC 5-minute binary options** using a 6-layer probabilistic signal architecture, deployed on AWS eu-west-1 (Ireland, ~140ms to Polymarket CLOB).

---

## The Edge

Polymarket prices BTC UP/DOWN binary options over rolling 5-minute windows.  
When BTC moves on Binance, Polymarket's CLOB takes **2–5 seconds** to reprice.  
We compute a fair value **faster** than the repricing delay and buy the stale side:

```
Edge = Φ(d₁)  −  poly_mid_stale
  d₁ = (BTC_now − strike) / (σ_5m × √(T/300))
  σ_5m calibrated live from Deribit implied volatility
```

Confirmed live `Status: matched` order on Polymarket:  
`ID: 0xdc9ff92dd273d60348ff0d53dfe6d040f92de07bec2cfce804dfd1189c252fba`

---

## Architecture — 6-Layer Signal Pipeline

```
BTC price (Binance WS) ──┐
Poly CLOB    (WS)        ├──► Layer 1: Data Feeds     (run_live.py)
Deribit IV   (WS)        ┘         │
Binance Depth (WS) ───────────────►│
                                   ▼
                         Layer 2: Fair Value Engine    (layer2_engine.py)
                           Φ(d₁) + Deribit σ calibration
                           OBI + cross-market arb + oracle-lag
                                   │
                                   ▼
                         Layer 3: Stale-Quote Gate     (decision_stack.py)
                           poly_mid unchanged in last 1s → still stale
                                   │
                                   ▼
                         Layer 4: Merton Jump-Diffusion (layer4_merton_jump.py)
                           P(S_T>K) with Poisson jump component
                           blended 50/50 with Layer 2
                                   │
                                   ▼
                         Layer 5: HMM Regime Classifier (layer5_hmm_regime.py)
                           3 states: low_vol / medium_vol / high_vol
                           Viterbi decoding → adaptive edge threshold
                                   │
                                   ▼
                         Layer 6: Kelly Sizing + Risk   (layer6_risk_execution.py)
                           f* = (p·b − q)/b × vol_mult × hawkes_mult
                           FOK market order via EIP-712 (live_executor.py)
```

---

## File Map

| File | Role | Lines |
|------|------|-------|
| `run_live.py` | **Main entry point** — asyncio event loop, all feed tasks, eval loop at 10 Hz | 1335 |
| `decision_stack.py` | **8-gate trading pipeline** — all veto logic, fair value, gate chain | 465 |
| `live_executor.py` | **Order execution** — EIP-712 signing, `post_order(FOK)`, risk manager | 834 |
| `layer2_engine.py` | Layer 2 — Gaussian fair value, OBI, cross-market arb, oracle lag | 282 |
| `layer4_merton_jump.py` | Layer 4 — Merton jump-diffusion model | 216 |
| `layer5_hmm_regime.py` | Layer 5 — HMM volatility regime classifier | 278 |
| `layer6_risk_execution.py` | Layer 6 — Kelly sizing, position limits, execution rules | 111 |
| `Hawkes_Process.py` | Self-exciting point process for trade-arrival intensity | 126 |
| `garch.py` | GARCH(1,1) realized volatility estimator | 115 |
| `vpin.py` | VPIN — Volume-Synchronized Probability of Informed Trading | 123 |
| `orderbook.py` | Order-book imbalance tracker (Binance + Polymarket) | — |
| `triple_streams.py` | Layer 1–6 integration test harness | — |
| `train_hmm_regime.py` | HMM training on 8-year Kaggle BTC data | — |
| `warmup.py` | Cold-start GARCH + Hawkes from recent live data | — |
| `backtest_300.py` | Strategy backtester on 1-second BTC tick data | — |

---

## Mathematical Models

### Layer 2 — Gaussian Binary Option

```python
# layer2_engine.py  _compute_fair_value()
pct_diff = (BTC_now - strike) / strike * 100
sigma_5m  = max(deribit_iv * 0.01, SIGMA_FLOOR_PCT)   # Deribit IV → 5-min σ
sigma_t   = sigma_5m * sqrt(time_remaining / 300)
d         = pct_diff / sigma_t
fair_up   = Φ(d)     # scipy.special.ndtr  or  math.erf(d/√2)/2 + 0.5
```

### Layer 4 — Merton Jump-Diffusion

```python
# layer4_merton_jump.py
# P(S_T > K) = Σ_{n=0}^{N}  Poisson(n; λT) × BS_call(σ_n)
# σ_n² = σ_diff² + n·σ_jump²/T
# λ, μ_jump, σ_jump estimated from 8-yr BTC 1-min returns (Kaggle)
p_up_merton = sum(
    poisson.pmf(n, lam*T) * black_scholes_binary(S, K, T, sigma_n)
    for n in range(N_max)
)
# Final signal: blend L2 + L4
fair_up = 0.5 * fair_value_gaussian + 0.5 * p_up_merton
```

### Layer 5 — HMM Regime Classifier

```python
# layer5_hmm_regime.py  (trained in train_hmm_regime.py)
# 3 states: low_vol, medium_vol, high_vol
# Emission:  N(μ_k, σ_k²) on 30-second BTC log-returns
# Transition: learned A matrix via Baum-Welch on 8yr data
# Inference:  Viterbi decoding for MAP state sequence
regime = hmm_model.predict(recent_returns[-20:])  # hmmlearn GaussianHMM
edge_threshold = {"high_vol": 0.05, "medium_vol": 0.03, "low_vol": 0.02}[regime]
```

### Layer 6 — Kelly Fraction + Sizing

```python
# layer6_risk_execution.py
b = 1.0 / entry_price - 1.0          # net odds (bet $p, win $1)
f_star = (p_model * b - (1 - p_model)) / b   # Kelly fraction
size_usd = bankroll * f_star
           * vol_regime_mult          # 1.2x LOW_VOL, 0.6x HIGH_VOL
           * hawkes_mult              # 1.2x EXCITED, 0.5x QUIET
           * vpin_mult                # 1 - 0.5*(vpin/vpin_max)
size_usd = max(size_usd, MIN_TRADE_USD)       # Polymarket $1 floor
size_usd = min(size_usd, bankroll * MAX_PCT)  # 10% bankroll cap
```

### Decision Stack — 8 Gates (decision_stack.py)

```
Gate 1  btc_flat        |Δbtc_10s| ≥ 0.015%  OR  |drift_from_strike| ≥ 0.015%
Gate 2  stale_check     poly_mid unchanged ≥ 1s  → quote still stale
Gate 3  price_zone      0.20 ≤ p_mkt ≤ 0.80  (near-resolved = no liquidity)
Gate 4  extreme_vol     GARCH vol_regime ≠ EXTREME_VOL
Gate 5  vpin_toxic      VPIN < 0.55  (no informed-flow regime)
Gate 6  hawkes_regime   regime ∈ {QUIET, ACTIVE, EXCITED, EXPLOSIVE}
Gate 7  obi_gate        Polymarket OBI not strongly opposing direction
Gate 8  gap_threshold   edge_net ≥ edge_threshold[hmm_regime]
```

### Supporting Models

**Hawkes Process** (`Hawkes_Process.py`)
```
λ(t) = μ + Σ_{t_i < t} α·exp(−β·(t − t_i))
Regimes: QUIET (λ≈μ), ACTIVE, EXCITED (α/β>0.5), EXPLOSIVE (α/β→1)
Fitted by MLE on live trade timestamps; refit hourly
```

**GARCH(1,1)** (`garch.py`)
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
Vol regimes: LOW_VOL (σ<0.1%), NORMAL_VOL, HIGH_VOL, EXTREME_VOL
```

**VPIN** (`vpin.py`)
```
VPIN = |V_buy - V_sell| / V_total   (volume-bucket estimate)
Toxic flow threshold: VPIN > 0.55 → halt (informed traders dominating)
```

---

## Live Results (2026-03-15, Mac → Polymarket direct)

| Window | Side | Entry | Edge | Result | P&L |
|--------|------|-------|------|--------|-----|
| 10:20–10:25 ET | DOWN | 0.570 | +15.3% | **WIN** | +$1.29 |
| 10:30–10:35 ET | DOWN | 0.740 | +7.6%  | **WIN** | +$0.34 |
| 10:55–11:00 ET | DOWN | 0.570 | +13.8% | matched | — |

Balance: $20.00 → $20.68 after 2 resolved trades.

---

## Setup & Run

```bash
pip install -r requirements.txt

# 1. Train HMM regime model (once, on historical data)
python3 train_hmm_regime.py

# 2. Warm up GARCH + Hawkes from recent ticks
python3 warmup.py

# 3. Paper mode (no real money)
python3 run_live.py --bankroll 20 --binance

# 4. Live mode
python3 run_live.py --live --bankroll 20 --binance
```

**Environment:** Python 3.12, asyncio — runs on macOS or Ubuntu 22.04 (AWS eu-west-1).

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Async I/O | `asyncio`, `aiohttp`, `websockets` |
| Order signing | `py-clob-client` (EIP-712), `eth-account` |
| Statistics | `scipy.special.ndtr`, `scipy.stats` |
| HMM | `hmmlearn.GaussianHMM` |
| Numerics | `numpy` |
| Env / secrets | `python-dotenv` |
| Data feeds | Binance WS, Polymarket CLOB WS, Deribit WS |
| Deployment | AWS EC2 eu-west-1 (t3.micro) |
