# Polymarket BTC Latency-Arbitrage Bot

**Real-money live trading system** exploiting stale quotes on Polymarket BTC 5-minute binary option markets using a 6-layer probabilistic signal architecture.

## Core Idea

Polymarket's binary options price BTC UP/DOWN over 5-minute windows.  
When BTC moves on Binance, Polymarket's CLOB takes 2–5 seconds to reprice.  
We compute a fair value **faster** than the repricing delay and trade the gap.

```
Edge = fair_value(BTC_now, strike, T, σ) − poly_mid
     = Φ(d₁) − stale_quote
```

With AWS Ireland latency (~140ms to Polymarket), the edge is real and measurable.

---

## Architecture: 6-Layer Signal Pipeline

| Layer | File | Purpose |
|-------|------|---------|
| **L1** | `run_live.py` | Live data feeds: Binance trades + depth, Polymarket CLOB WebSocket, Deribit IV |
| **L2** | `layer2_engine.py` | Fair value engine: Gaussian binary option model + Deribit IV σ calibration |
| **L3** | `decision_stack.py` | Stale-quote gate: detects if Polymarket has already repriced |
| **L4** | `layer4_merton_jump.py` | Merton jump-diffusion model (second probability estimate, averaged with L2) |
| **L5** | `layer5_hmm_regime.py` | Hidden Markov Model volatility regime classifier (low/medium/high_vol) |
| **L6** | `layer6_risk_execution.py` | Kelly sizing, position limits, execution type (MARKET vs LIMIT) |

---

## Mathematical Models

### Layer 2 — Gaussian Binary Option Fair Value
```
fair_value = Φ(d₁)
d₁ = (BTC_now − strike) / (σ_5m × √(T/300))
σ_5m  calibrated from Deribit IV or empirical BTC return distribution
```

### Layer 4 — Merton Jump-Diffusion
```
P(S_T > K) with jump component:
V = Σ_{n=0}^{N} [e^{-λT}(λT)^n / n!] × BS(S, K, T, σ_n)
σ_n² = σ² + n·σ_J² / T,  λ = jump intensity from 8yr BTC data
```

### Layer 5 — HMM Regime Classifier
```
States: low_vol, medium_vol, high_vol
Emission: N(μ_k, σ_k²) on 30s BTC log-returns
Viterbi decoding for MAP state sequence
Edge threshold: {high_vol: 5%, medium_vol: 3%, low_vol: 2%}
```

### Layer 6 — Kelly Position Sizing
```
f* = (p × b − q) / b   where b = (1/price − 1)
size = bankroll × f* × vol_regime_mult × hawkes_mult × vpin_mult
```

### Decision Stack Gates (decision_stack.py)
1. **BTC Flat**: `|Δbtc_10s| ≥ 0.015%` OR `|drift_from_strike| ≥ 0.015%`
2. **Stale Check**: `poly_mid` hasn't moved in last 1s → still stale
3. **Price Zone**: `0.20 ≤ p_mkt ≤ 0.80` (near-certain markets have no liquidity)
4. **GARCH**: not `EXTREME_VOL` regime
5. **VPIN**: Volume-Synchronized Probability of Informed Trading `< 0.55`
6. **Hawkes Process**: event-arrival regime in {QUIET, ACTIVE, EXCITED, EXPLOSIVE}
7. **OBI**: Polymarket order-book imbalance must not strongly oppose our direction

---

## Live Trading Results (2026-03-15)

| Window | Side | Entry | Edge | Outcome | P&L |
|--------|------|-------|------|---------|-----|
| 10:20–10:25 AM ET | DOWN | 0.570 | +15.3% | **WIN** | +$1.29 |
| 10:30–10:35 AM ET | DOWN | 0.740 | +7.6%  | **WIN** | +$0.34 |
| 10:55–11:00 AM ET | DOWN | 0.570 | +13.8% | live    | — |

Order confirmed on Polymarket CLOB:  
`ID: 0xdc9ff92dd273d60348ff0d53dfe6d040f92de07bec2cfce804dfd1189c252fba | Status: matched`

---

## Key Files

```
run_live.py            — Main entry point: asyncio feeds + eval loop (10Hz)
decision_stack.py      — All 8 trading gates + fair value computation
live_executor.py       — Polymarket CLOB order signing + FOK submission (EIP-712)
layer2_engine.py       — Binary option pricing with Deribit IV calibration
layer4_merton_jump.py  — Jump-diffusion second signal
layer5_hmm_regime.py   — HMM volatility regime + edge threshold adaptation
layer6_risk_execution.py — Kelly sizing, position limits, execution rules
Hawkes_Process.py      — Self-exciting point process for trade arrival intensity
garch.py               — GARCH(1,1) realized volatility estimation
vpin.py                — VPIN toxic flow detector
orderbook.py           — Order book imbalance tracker
triple_streams.py      — Layer 1–6 pipeline integration test harness
```

## Running

```bash
pip install -r requirements.txt

# Paper mode (simulated)
python3 run_live.py --bankroll 20 --binance

# Live mode (real money)
python3 run_live.py --live --bankroll 20 --binance
```

## Tech Stack
- **Python 3.12**, asyncio, aiohttp, websockets
- **py-clob-client** (EIP-712 order signing, Polymarket CLOB API)
- **scipy** (Gaussian CDF, HMM Viterbi), **numpy**, **hmmlearn**
- **Binance WS** (trade feed + depth), **Deribit WS** (IV feed), **Polymarket CLOB WS**
- **AWS eu-west-1** (Ireland) for ~140ms round-trip latency
