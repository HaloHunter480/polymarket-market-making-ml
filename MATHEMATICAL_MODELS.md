# Mathematical Models and Algorithms for High-Frequency Binary Options Trading
## A Statistical Arbitrage System for Polymarket Bitcoin Markets

**Author**: Harjot  
**Institution**: French University Submission  
**Date**: February 2026  
**Field**: Quantitative Finance, Market Microstructure, Statistical Arbitrage

---

## Executive Summary

This document presents a comprehensive statistical arbitrage system for trading binary options on Bitcoin price movements in the Polymarket prediction market. The system employs advanced mathematical models from financial econometrics, market microstructure theory, and information theory to identify and exploit pricing inefficiencies while managing toxic flow and adverse selection.

**Key Components (v4 — Paper Trading)**:
1. Empirical probability surface calibration
2. Kelly Criterion for optimal position sizing
3. Hawkes process for volume clustering detection
4. VPIN (Volume-Synchronized Probability of Informed Trading)
5. Kyle's Lambda for market impact estimation
6. Order flow pressure analysis
7. Multi-signal fusion and execution optimization

**Key Components (v5 — Research-Grade Live System)**:
1. Uncertainty-Adjusted Kelly with Monte Carlo Beta posterior sampling
2. Calibration Curve with online MLE (favorite-longshot bias correction)
3. Monte Carlo Equity Simulation for session-level ruin probability
4. ~~Order Flow Imbalance (OFI)~~ — *disabled* (not predictive on OOS)
5. Regime Classification via EWMA volatility and momentum (replaces Hurst)
6. Full outcome distribution tracking (variance, skewness, CI)

**Key Components (v6 — Validated Live System)**:
1. Walk-Forward Empirical Surface — train/test split, OOS conditional win rate (target ≥60%)
2. Regime-Stratified Bins — regime used only if OOS-validated (≥58% confident WR)
3. Empirical Calibration Tracker — Brier score, reliability diagram at runtime
4. Mud Zone — no trade when prob ∈ [0.38, 0.62]
5. Hold-to-Expiry Precision Mode — enter only in first 120s, price zone 0.35–0.57
6. Coinbase Price Feed — matches Polymarket settlement source
7. Recency-Weighted Surface — last 25% of train counts double

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Empirical Probability Model](#2-empirical-probability-model)
3. [Position Sizing: Kelly Criterion](#3-position-sizing-kelly-criterion)
4. [Market Microstructure Models](#4-market-microstructure-models)
5. [Toxic Flow Detection](#5-toxic-flow-detection)
6. [Order Flow Analysis](#6-order-flow-analysis)
7. [Execution Optimization](#7-execution-optimization)
8. [Risk Management](#8-risk-management)
9. [Performance Metrics](#9-performance-metrics)
10. [Conclusion](#10-conclusion)
11. [Research-Grade Enhancements (v5)](#11-research-grade-enhancements-v5)
    - [11.8 v6 Enhancements (Validated Live System)](#118-v6-enhancements-validated-live-system)
    - 11.1 Uncertainty-Adjusted Kelly (Monte Carlo Posterior)
    - 11.2 Calibration Curve (Favorite-Longshot Bias Correction)
    - 11.3 Monte Carlo Equity Simulation
    - 11.4 Order Flow Imbalance (OFI) — disabled in v6
    - 11.5 Regime Classification (EWMA + Momentum)
    - 11.6 Full Distribution Tracking
    - 11.7 Integrated Signal Pipeline
    - 11.8 Walk-Forward, Regime OOS, Mud Zone, Hold-to-Expiry

---

## 1. Problem Statement

### 1.1 Market Structure

We trade binary options on Bitcoin price movements with the following characteristics:

- **Contract Type**: Binary options paying $1 if condition is met, $0 otherwise
- **Time Horizon**: 5-minute windows
- **Strike Price**: Bitcoin spot price at window open (K)
- **Resolution**: Binary outcome - UP if BTC_close > K, DOWN otherwise
- **Market**: Polymarket CLOB (Central Limit Order Book)

### 1.2 Objective Function

Maximize risk-adjusted returns while managing:
```
max E[R] - λ·σ²(R)
```

Subject to constraints:
- Transaction costs (maker: 0%, taker: 2%)
- Slippage (est. 1.5% on taker orders)
- Adverse selection risk
- Market impact
- Liquidity constraints

---

## 2. Empirical Probability Model

### 2.1 Theoretical Foundation

Classical option pricing models (Black-Scholes) assume:
- Continuous trading
- No transaction costs
- Log-normal price distribution
- Constant volatility

**Problem**: These assumptions fail for 5-minute binary options:
- High transaction costs (2% taker fee)
- Non-normal short-term returns
- Time-varying volatility
- Discrete outcomes

**Solution**: Build an empirical probability surface from historical data.

### 2.2 Mathematical Formulation

Let:
- `S(t)` = Bitcoin spot price at time t
- `K` = Strike price (price at window open)
- `T` = Time remaining (seconds)
- `Δ = (S(t) - K) / K` = Percentage difference from strike

Define the empirical probability:

```
P̂(UP | Δ, T) = ∑[i=1 to N] 𝟙{UP_i} · K(Δ - Δ_i, T - T_i) / ∑[i=1 to N] K(Δ - Δ_i, T - T_i)
```

Where:
- `𝟙{UP_i}` = Indicator function (1 if window i resolved UP, 0 otherwise)
- `K(·, ·)` = Kernel function for smoothing
- `N` = Number of historical observations

### 2.3 Implementation: Binned Probability Surface

For computational efficiency, we discretize the state space:

**State Space**:
- Price difference: `Δ ∈ [-2%, +2%]` in 0.02% bins (200 bins)
- Time remaining: `T ∈ [0, 300]` seconds in 60-second bins (5 bins)

**Data Structure**:
```python
prob_surface[Δ_bin][T_bin] = {
    "up_count": int,      # Number of UP resolutions
    "down_count": int,    # Number of DOWN resolutions
    "total": int          # Total observations
}
```

**Probability Calculation**:
```
P(UP | Δ, T) = up_count / total
```

**Interpolation** for sub-minute time intervals:
```
T_lo = ⌊T / 60⌋ · 60
T_hi = ⌈T / 60⌉ · 60
α = (T - T_lo) / (T_hi - T_lo)

P(UP | Δ, T) = (1 - α) · P(UP | Δ, T_lo) + α · P(UP | Δ, T_hi)
```

### 2.4 Momentum Adjustment

Short-term price momentum affects outcomes. We compute:

```
momentum_3m = (S(t) - S(t-180)) / S(t-180)
```

Separate probability surfaces for momentum buckets:
- Strong bearish: momentum < -0.1%
- Weak bearish: -0.1% ≤ momentum < 0%
- Weak bullish: 0% ≤ momentum < +0.1%
- Strong bullish: momentum ≥ +0.1%

**Final probability** blends base and momentum estimates:
```
P_final = w_base · P_base + w_momentum · P_momentum

where:
w_momentum = min(1.0, total_momentum_obs / 30)
w_base = 1 - w_momentum
```

### 2.5 Data Requirements

**Training Dataset**:
- Source: Binance BTC/USDT 1-minute candles (historical)
- Period: 31 days (45,000 1-minute observations)
- Windows: overlapping 1-candle-step (9,000+ windows)
- Split: 70% train, 30% hold-out for OOS validation

**Live Price Feed (v6)**:
- Source: Coinbase WebSocket (`wss://ws-feed.exchange.coinbase.com`, BTC-USD ticker)
- Rationale: Matches Polymarket settlement reference; minimizes basis risk

---

## 3. Position Sizing: Kelly Criterion

### 3.1 Theoretical Foundation

The **Kelly Criterion** maximizes logarithmic utility of wealth:

```
max E[log(W_final)]
```

For a binary outcome:
```
f* = (p·b - q) / b

where:
f* = Optimal fraction of bankroll to bet
p = Probability of winning
q = 1 - p = Probability of losing
b = Odds (payout / stake - 1)
```

### 3.2 Application to Binary Options

For binary options priced at `M` (market price):

**Odds calculation**:
```
b = (1 / M) - 1
```

**Kelly fraction**:
```
f* = (P_fair · b - (1 - P_fair)) / b

where:
P_fair = Our estimated probability (from empirical model)
```

**Example**:
- Market price: M = 0.60 (60¢)
- Our estimate: P_fair = 0.70 (70%)
- Odds: b = (1 / 0.60) - 1 = 0.667
- Kelly: f* = (0.70 × 0.667 - 0.30) / 0.667 = 0.25 (25% of bankroll)

### 3.3 Fractional Kelly for Risk Management

Full Kelly is too aggressive (high volatility). We use:

```
f_actual = λ · f* · C

where:
λ = 0.08 = Fractional Kelly parameter (8% of full Kelly)
C = Confidence factor (0 to 1)
```

**Confidence calculation**:
```
C = w_sample · C_sample + w_edge · C_edge + w_spread · C_spread + w_toxic · C_toxic

where:
C_sample = min(1, N_observations / 50)    # Sample size confidence
C_edge = min(1, |edge| / 0.05)           # Edge magnitude confidence
C_spread = 1 - min(1, spread / 0.15)     # Spread tightness confidence
C_toxic = 1 if no_toxic else 0.5        # Toxic flow penalty

Weights: w_sample = 0.3, w_edge = 0.3, w_spread = 0.2, w_toxic = 0.2
```

### 3.4 Edge Capping for Anomaly Protection

To prevent over-betting on data anomalies:

```
edge_capped = min(|edge|, MAX_EDGE)

where:
MAX_EDGE = 0.15 (15%)
```

**Rationale**: In liquid BTC markets, true edges >15% are almost always data errors or empty order books.

### 3.5 Position Limits

```
Position_size = max(MIN_BET, min(MAX_BET, f_actual · Bankroll))

where:
MIN_BET = $4
MAX_BET = $30
```

---

## 4. Market Microstructure Models

### 4.1 Kyle's Lambda: Market Impact

**Kyle's Model** (1985) describes price impact of informed trading:

```
ΔP = λ · Q

where:
ΔP = Price change
Q = Order size (signed, positive for buy)
λ = Kyle's lambda (price impact coefficient)
```

**Estimation** using rolling window regression:

```
ΔP_t = λ · ΔQ_t + ε_t

where:
ΔP_t = P_t - P_{t-1}
ΔQ_t = Signed volume (buy - sell)
```

**Implementation**:
```python
class KyleLambda:
    def estimate(self, window_size=100):
        if len(self._history) < window_size:
            return 0.0
        
        price_changes = [ΔP_i for i in range(window_size)]
        volume_imbalances = [ΔQ_i for i in range(window_size)]
        
        # Linear regression: ΔP = λ · ΔQ
        λ = cov(price_changes, volume_imbalances) / var(volume_imbalances)
        
        return λ
```

**Usage**: High λ indicates high market impact → use limit orders (maker) instead of market orders (taker).

### 4.2 Bid-Ask Spread Decomposition

The bid-ask spread compensates market makers for:

```
Spread = Cost_inventory + Cost_adverse_selection + Cost_order_processing

Empirically:
spread_bps = (ask - bid) / mid

where:
mid = (ask + bid) / 2
```

**Decision rule**:
```
if spread_bps > MAX_SPREAD_BPS (0.15):
    skip_trade()  # Market too illiquid or toxic
```

---

## 5. Toxic Flow Detection

### 5.1 Hawkes Process for Volume Clustering

**Definition**: A self-exciting point process where events increase the probability of future events.

```
λ(t) = μ + α ∑_{t_i < t} exp(-β(t - t_i))

where:
λ(t) = Intensity (expected rate) at time t
μ = Background intensity
α = Self-excitation strength
β = Decay rate
t_i = Time of past events
```

**Interpretation**:
- High intensity → Clustering of trades → Potential informed trading
- Normal intensity → Random arrivals → Uninformed flow

**Implementation**:
```python
class HawkesProcess:
    def get_intensity(self, t):
        intensity = self.mu
        for (t_i, vol_i) in self._events:
            if t > t_i:
                intensity += self.alpha * vol_i * exp(-self.beta * (t - t_i))
        return intensity
    
    def is_clustering(self, t, threshold=5.0):
        return self.get_intensity(t) > threshold * self.mu
```

**Parameters**:
- μ = 1.0 (baseline rate)
- α = 0.5 (moderate self-excitation)
- β = 0.1 (slow decay)
- Threshold = 5.0x baseline

### 5.2 VPIN: Volume-Synchronized Probability of Informed Trading

**Theory** (Easley et al., 2011): Measures order flow toxicity using volume imbalance.

**Traditional VPIN** (volume-bucket based):
```
VPIN = (1/n) ∑_{i=1}^n |Buy_Vol_i - Sell_Vol_i| / Total_Vol_i
```

**Our Implementation** (time-windowed for HFT):
```
VPIN(t) = |Buy_Vol(t-Δt, t) - Sell_Vol(t-Δt, t)| / Total_Vol(t-Δt, t)

where:
Δt = 10 seconds (rolling window)
```

**Trade Classification** (using tick rule):
```
Trade_sign = sign(Price - Price_prev)

If sign > 0: Buy (aggressive buyer hit the ask)
If sign < 0: Sell (aggressive seller hit the bid)
If sign = 0: Use previous trade sign
```

**Implementation**:
```python
class VPIN:
    def __init__(self, window_seconds=10.0):
        self.window_seconds = window_seconds
        self._trades = deque(maxlen=2000)  # (timestamp, volume, is_buy)
    
    def calculate(self):
        now = time.time()
        cutoff = now - self.window_seconds
        
        buy_vol = sum(vol for (t, vol, is_buy) in self._trades 
                      if t >= cutoff and is_buy)
        sell_vol = sum(vol for (t, vol, is_buy) in self._trades 
                       if t >= cutoff and not is_buy)
        
        total_vol = buy_vol + sell_vol
        if total_vol < 0.01:
            return 0.0
        
        return abs(buy_vol - sell_vol) / total_vol
    
    def is_toxic(self, threshold=0.80):
        return self.calculate() > threshold
```

**Threshold**: VPIN > 0.80 (80% one-sided flow over 10 seconds) indicates toxic flow.

**Action**: Cancel all resting limit orders immediately.

### 5.3 Price Jump Detection

Large price jumps signal news arrival or informed trading:

```
ΔP_t = |P_t - P_{t-1}|

If ΔP_t > THRESHOLD ($50):
    toxic_flow_detected()
```

**Rolling History**:
```python
self._price_history = deque(maxlen=10)  # Last 10 price changes
```

**Detection**:
```python
if len(self._price_history) >= 2:
    latest_jump = self._price_history[-1]
    if latest_jump > TOXIC_PRICE_JUMP:
        return True, f"PRICE_JUMP_${latest_jump:.1f}"
```

### 5.4 Volume Spike Detection

Sudden volume increases indicate institutional activity:

```
If Vol_t > THRESHOLD × Vol_avg:
    toxic_flow_detected()

where:
Vol_avg = (1/n) ∑_{i=t-n}^{t-1} Vol_i
THRESHOLD = 5.0x
n = 20 observations
```

### 5.5 Unified Toxic Flow Detector

```python
class ToxicFlowDetector:
    def is_toxic(self):
        # 1. Price jump (most reliable)
        if latest_jump > TOXIC_PRICE_JUMP:
            return True, "PRICE_JUMP"
        
        # 2. VPIN (time-windowed)
        if self.vpin.calculate() > 0.80:
            return True, "VPIN"
        
        # 3. Volume spike
        if recent_vol > 5.0 × avg_vol and recent_vol > 50:
            return True, "VOL_SPIKE"
        
        return False, "CLEAN"
```

---

## 6. Order Flow Analysis

### 6.1 Order Flow Pressure

**Definition**: Directional bias in recent trades.

```
Pressure(t) = (Buy_Volume - Sell_Volume) / Total_Volume

Over window Δt = 30 seconds
```

**Normalization**:
```
Pressure ∈ [-1, +1]

where:
-1 = 100% sell pressure
 0 = Balanced
+1 = 100% buy pressure
```

**Implementation**:
```python
class OrderFlowPressure:
    def get_pressure(self):
        buy_vol = sum(v for (_, v, is_buy) in recent_trades if is_buy)
        sell_vol = sum(v for (_, v, is_buy) in recent_trades if not is_buy)
        
        total = buy_vol + sell_vol
        if total < 1.0:
            return 0.0
        
        return (buy_vol - sell_vol) / total
```

### 6.2 Quote Skewing Based on Flow

**Asymmetric Market Making**: Adjust bid/ask quotes based on flow pressure to avoid adverse selection.

```
If buying UP and flow is bullish (+0.8):
    Place bid LOWER (defensive)
    → Avoid getting run over by momentum

If buying UP and flow is bearish (-0.8):
    Place bid HIGHER (aggressive)
    → Flow is in our favor
```

**Skew Calculation**:
```
skew = -sign(side) × pressure × MAX_SKEW

where:
MAX_SKEW = 0.04 (4%)
sign(UP) = +1
sign(DOWN) = -1
```

**Quote Adjustment**:
```
If side == "UP":
    limit_price = fair_value × (1 + skew)
    
If side == "DOWN":
    limit_price = fair_value × (1 + skew)
```

**Example**:
- Fair value: 0.50
- Side: UP
- Pressure: +0.80 (strong buy pressure)
- Skew: -1 × 0.80 × 0.04 = -0.032 (-3.2%)
- Limit price: 0.50 × (1 - 0.032) = 0.484 (defensive bid)

### 6.3 Lead-Lag Detector

**Cross-Correlation** between exchanges to detect leading indicators:

```
ρ(τ) = Corr(ΔP_Binance(t), ΔP_Polymarket(t + τ))

where:
τ = Time lag
```

**If ρ(τ) peaks at τ > 0**: Binance leads Polymarket → Use Binance as signal

**Implementation**:
```python
class LeadLagDetector:
    def detect(self, max_lag=5):
        correlations = []
        for lag in range(max_lag + 1):
            corr = np.corrcoef(
                binance_returns[:-lag] if lag > 0 else binance_returns,
                polymarket_returns[lag:] if lag > 0 else polymarket_returns
            )[0, 1]
            correlations.append(corr)
        
        best_lag = np.argmax(correlations)
        return best_lag, correlations[best_lag]
```

---

## 7. Execution Optimization

### 7.1 Maker vs. Taker Decision

**Cost Analysis**:

| Type | Fee | Slippage | Fill Rate | Total Cost |
|------|-----|----------|-----------|------------|
| Maker | 0% | 0% | 40% | 0% |
| Taker | 2% | 1.5% | 100% | 3.5% |

**Decision Tree**:

```
IF edge > MIN_TAKER_EDGE (4.5%) AND (time < 60s OR edge > 8%):
    Execute TAKER (market order)
ELSE IF edge > MIN_MAKER_EDGE (0.8%) AND no_toxic_flow AND spread < 15%:
    Execute MAKER (limit order)
ELSE:
    Skip trade
```

**Rationale**:
- Taker: Only when edge >> cost (4.5% > 3.5% cost)
- Maker: When we can afford to wait and avoid fees

### 7.2 Realistic Fill Simulation (Paper Trading)

**Maker Orders**:
```
P(fill) = MAKER_FILL_RATE = 0.40 (40%)

Simulation:
fill_roll = random()
if fill_roll <= 0.40:
    order_filled()
else:
    order_expired()
```

**Rationale**: 60% of maker orders expire due to:
- Adverse selection (informed traders pick us off)
- Market moves away (no fill)
- Order cancelled due to toxic flow

**Taker Orders**:
```
effective_price = market_price × (1 + TAKER_SLIPPAGE)

where:
TAKER_SLIPPAGE = 0.015 (1.5%)
```

**Rationale**: Thin order books cause slippage beyond mid price.

### 7.3 Order Book Reality Checks

**Problem**: Early testing revealed exploitation of empty order books at window opens.

**Solutions**:

1. **Window Initialization Delay**:
```
IF time_since_window_open < 15 seconds:
    skip_trade()
```

2. **Ghost Town Filter**:
```
IF best_ask_volume < $50 OR best_bid_volume < $50:
    skip_trade()  # Order book too thin
```

3. **Maximum Edge Cap**:
```
edge_for_kelly = min(|edge|, 0.15)  # Cap at 15%
```

Prevents betting the entire bankroll on anomalous edges.

### 7.4 Pre-Emptive Order Cancellation

**Kill Switch** for toxic flow:

```
WHILE bot_running:
    IF toxic_flow_detected():
        cancel_all_orders()
        wait(CANCEL_ALL_DELAY = 1.0 second)
```

**Monitoring Loop** (runs every 200ms):
```python
async def _toxic_monitor_loop(self):
    while self.running:
        is_toxic, reason = self.feed.toxic_detector.is_toxic()
        if is_toxic:
            if self.executor.open_orders:
                await self.executor.cancel_all_orders(f"TOXIC: {reason}")
            await asyncio.sleep(1.0)  # Cooldown
        await asyncio.sleep(0.2)
```

---

## 8. Risk Management

### 8.1 Position Limits

```
MAX_TRADES_PER_WINDOW = 2
SIGNAL_COOLDOWN = 25 seconds
```

**Rationale**: Prevents over-trading and allows time for signal validation.

### 8.2 Bankroll Management

```
BANKROLL = $500 (starting)
MIN_BET = $4
MAX_BET = $30
KELLY_FRACTION = 0.08 (8% of full Kelly)
```

**Fractional Kelly** reduces volatility:
```
Volatility(fractional) ≈ (f_fractional / f_full) × Volatility(full)
```

### 8.3 Edge Requirements

```
MIN_MAKER_EDGE = 0.008 (0.8%)
MIN_TAKER_EDGE = 0.045 (4.5%)
```

**Break-Even Analysis**:

Maker:
```
Required win rate: p = fee / (1 + fee) = 0% / (1 + 0%) = 50%
With 0.8% edge: p = 50.8% (profitable)
```

Taker:
```
Required win rate: p = 3.5% / (1 + 3.5%) = 3.38%
Min edge: 4.5% > 3.5% cost ✓
```

### 8.4 Sample Size Requirements

```
MIN_SAMPLES = 20 observations per (Δ, T) bin
```

**Statistical Significance**:
```
Standard error: SE = σ / √N = 0.5 / √20 ≈ 0.11 (11%)

95% CI: p̂ ± 1.96 × SE ≈ p̂ ± 22%
```

Insufficient for tight spreads → require larger sample.

### 8.5 Spread Limits

```
MIN_SPREAD_BPS = 0.05 (5%)
MAX_SPREAD_BPS = 0.15 (15%)
```

**Wide spreads** indicate:
- Low liquidity
- High uncertainty
- Toxic market conditions

---

## 9. Performance Metrics

### 9.1 Risk-Adjusted Returns

**Sharpe Ratio**:
```
SR = (μ_R - r_f) / σ_R

where:
μ_R = Mean return
r_f = Risk-free rate (≈0 for 5-min periods)
σ_R = Standard deviation of returns
```

**Target**: SR > 1.0 (after all costs)

### 9.2 Win Rate and Edge

```
Win_Rate = N_wins / N_total

Expected_Return = Win_Rate × Payout - (1 - Win_Rate) × Loss
```

**Target**: Win Rate > 50% with positive expected return after fees.

### 9.3 Maximum Drawdown

```
MDD = max_{t} (Peak_value_{s≤t} - Value_t) / Peak_value_{s≤t}
```

**Target**: MDD < 25%

### 9.4 Profit Factor

```
PF = Gross_Profit / Gross_Loss = ∑Wins / |∑Losses|
```

**Target**: PF > 1.5

### 9.5 Fill Rate Analysis

```
Maker_Fill_Rate = N_maker_fills / N_maker_attempts

Expected: 40% (with adverse selection)
```

---

## 10. Conclusion

### 10.1 Model Summary

This system integrates multiple mathematical frameworks:

1. **Empirical Finance**: Data-driven probability estimation
2. **Information Theory**: VPIN for measuring informed trading
3. **Point Processes**: Hawkes for volume clustering
4. **Market Microstructure**: Kyle's lambda, bid-ask spreads
5. **Optimal Control**: Kelly Criterion for position sizing
6. **Time Series Analysis**: Momentum, lead-lag correlation

### 10.2 Theoretical Contributions

- **Empirical over parametric**: Avoids mis-specification of short-term distributions
- **Multi-scale toxic flow**: Combines time-domain (VPIN) and frequency-domain (Hawkes) measures
- **Asymmetric quoting**: Dynamic adjustment based on order flow pressure
- **Reality-constrained optimization**: Explicit modeling of fill rates, slippage, and empty books

### 10.3 Performance Summary (Paper Trading)

**Test Period**: 75 minutes, 15 five-minute windows

| Metric | Value |
|--------|-------|
| Total Trades | 28 |
| Win Rate | 71.4% |
| Return | +148% |
| Sharpe Ratio (annualized) | ~2.5 |
| Max Drawdown | ~8% |

**Note**: Paper trading results are optimistic. Expected live performance:
- Daily return: 0.5-2%
- Win rate: 48-52%
- Sharpe: 0.8-1.5

### 10.4 Limitations and Future Work

**Current Limitations**:
1. Paper trading doesn't capture full adverse selection
2. Small sample size (28 trades)
3. Single market (BTC 5-minute)
4. No portfolio effects

**Future Enhancements**:
1. Multi-asset portfolio optimization
2. Regime-switching models (trending vs. ranging)
3. Deep learning for pattern recognition
4. Real-time volatility forecasting
5. Cross-market arbitrage

### 10.5 Academic References

1. **Kelly Criterion**: Kelly, J.L. (1956). "A New Interpretation of Information Rate"
2. **Kyle's Lambda**: Kyle, A.S. (1985). "Continuous Auctions and Insider Trading"
3. **Hawkes Process**: Hawkes, A.G. (1971). "Spectra of some self-exciting point processes"
4. **VPIN**: Easley, D., López de Prado, M.M., O'Hara, M. (2011). "The Microstructure of the Flash Crash"
5. **Market Microstructure**: O'Hara, M. (1995). "Market Microstructure Theory"
6. **High-Frequency Trading**: Aldridge, I. (2013). "High-Frequency Trading: A Practical Guide"

---

## Appendix A: Implementation Notes

### A.1 Technology Stack

- **Language**: Python 3.11
- **Async Framework**: `asyncio` for concurrent data feeds
- **HTTP Client**: `aiohttp` for REST APIs
- **WebSocket**: `websockets` for real-time market data
- **Numerical**: NumPy for array operations
- **Statistical**: SciPy for statistical functions
- **Data Storage**: Pickle for empirical probability surfaces

### A.2 Data Sources

- **Binance**: BTC/USDT 1-minute candles (training data)
- **Polymarket**: CLOB order book, Gamma API for markets
- **Historical**: 31 days × 1,440 minutes = 44,640 observations

### A.3 Computational Complexity

- **Probability lookup**: O(1) - Direct array indexing with interpolation
- **Hawkes intensity**: O(N) - Sum over recent events (N ≤ 100)
- **VPIN calculation**: O(M) - Sum over time window (M ≤ 2,000 trades)
- **Kyle's lambda**: O(W) - Regression over window (W = 100)

**Total latency**: < 1ms per signal evaluation

### A.4 Memory Footprint

- **Probability surface**: 200 bins × 5 time bins × 2 bytes ≈ 2KB
- **Momentum surface**: 200 × 5 × 4 momentum states × 2 bytes ≈ 8KB
- **Trade history**: 2,000 trades × 32 bytes ≈ 64KB
- **Total**: < 1MB

---

## Appendix B: Mathematical Proofs

### B.1 Kelly Criterion Derivation

**Objective**: Maximize expected log wealth

```
max E[log(W_1)]

where:
W_1 = W_0 · (1 + f·(b·X - (1-X)))

X = Bernoulli(p) (1 if win, 0 if lose)
f = Fraction bet
b = Odds
```

**Expected log wealth**:
```
E[log(W_1)] = p·log(W_0(1 + f·b)) + (1-p)·log(W_0(1 - f))
```

**First-order condition**:
```
dE/df = p·b/(1 + f·b) - (1-p)/(1 - f) = 0
```

**Solving for f**:
```
p·b·(1 - f) = (1 - p)·(1 + f·b)
p·b - p·b·f = 1 - p + f·b - p·f·b
p·b - 1 + p = f·b
f* = (p·b - q) / b

where q = 1 - p
```

### B.2 VPIN Theoretical Foundation

Based on **PIN model** (Easley et al., 1996):

```
P(informed trade) = α·μ / (α·μ + 2·ε)

where:
α = Probability of information event
μ = Informed trader arrival rate
ε = Uninformed trader arrival rate
```

**VPIN approximation**:
```
VPIN ≈ E[|Buy - Sell|] / E[Total Volume]
```

Under information event:
- Informed traders trade one-sided → High imbalance
- Uninformed traders trade random → Low imbalance

Therefore: **High VPIN → High probability of informed trading**

### B.3 Hawkes Process MLE Estimation

**Log-likelihood**:
```
log L(μ, α, β) = ∑_{i=1}^N log(λ(t_i)) - ∫_0^T λ(t)dt
```

**For our parameters** (μ=1.0, α=0.5, β=0.1), we use **calibration** rather than MLE due to computational constraints in real-time trading.

---

## Appendix C: Code Architecture

### C.1 Main Components

```
professional_strategy.py
│
├── EmpiricalEngine          # Probability surface lookup
├── HawkesProcess            # Volume clustering detection
├── VPIN                     # Toxic flow via imbalance
├── KyleLambda              # Market impact estimation
├── OrderFlowPressure       # Directional flow measurement
├── ToxicFlowDetector       # Unified toxic detection
├── EnhancedMultiExchangeFeed  # Multi-source data aggregation
├── AdvancedSignalGenerator # Signal evaluation with reality checks
├── ProfessionalExecutor    # Order execution with fill simulation
└── ProfessionalStrategy    # Main orchestration
```

### C.2 Data Flow

```
Binance WebSocket → Price Feed → Hawkes/VPIN → Toxic Detection
                                     ↓
Polymarket CLOB → Order Book → Signal Generator → Kelly Sizing
                                     ↓
                           Executor → Fill Simulation → P&L Tracking
```

### C.3 Async Architecture

```python
async def main():
    strategy = ProfessionalStrategy()
    
    await asyncio.gather(
        strategy.feed.connect_all(),          # WebSocket connections
        strategy._market_monitor_loop(),       # Market discovery
        strategy._strategy_loop(),             # Trading logic
        strategy._toxic_monitor_loop(),        # Toxic flow monitoring
        strategy._display_loop(),              # Live dashboard
    )
```

---

## 11. Research-Grade Enhancements (v5)

This section documents six additional research-grade modules introduced in the live trading system (v5) to address the empirical shortcomings exposed in real trading conditions.

---

### 11.1 Uncertainty-Adjusted Kelly Criterion (Monte Carlo Posterior)

**Motivation**: The classical Kelly formula `f* = (p·b − q)/b` treats `p` as a known constant. In practice, `p` is estimated from a finite sample of `n` historical outcomes, introducing estimation uncertainty. Ignoring this uncertainty leads to systematic over-betting, especially in data-sparse regimes.

**Model**: Three independent sources of uncertainty are resampled per Monte Carlo path:

**Dimension 1 — Win-probability uncertainty (all order types)**

```
P(p | data)  ~  Beta(α, β)
     α = p̂ · n
     β = (1 − p̂) · n
```

When `n` is small the Beta distribution is wide → conservative sizing automatically.

**Dimension 2 — Entry-price / slippage uncertainty (taker orders)**

```
slip_s  ~  clip(Normal(μ_slip, σ_slip),  0,  ∞)
     μ_slip = 1.5%  (empirical taker slippage)
     σ_slip = 0.3 · μ_slip

effective_price_s = price · (1 + slip_s)
R_s = (1 − fee) / effective_price_s − 1     ← payout ratio varies per path
```

**Dimension 3 — Fill-rate uncertainty (maker orders only)**

```
fill_s  ~  Beta(fα, fβ)
     fα = fill_rate · n_fill_obs     (prior: 40% fill rate, n=50)
     fβ = (1 − fill_rate) · n_fill_obs

3-outcome Kelly derivation (win / lose / no-fill):
     E[log(1+r)] = fill_p·p·log(1+f·R) + fill_p·(1-p)·log(1-f) + (1-fill_p)·log(1)
     d/df = fill_p · [p·R/(1+f·R) − (1-p)/(1-f)] = 0
     
     fill_p cancels → f* is theoretically independent of fill_rate.
     BUT: fill-rate *uncertainty* inflates cross-path variance.
     
Solution: multiply f_s by fill_s, so the 25th-percentile cut penalises paths
where both win-probability and fill-rate are simultaneously unfavourable.
```

**Full per-path Kelly calculation**:

```
For each path s = 1 … N (N = 5,000):
    p_s     ~ Beta(α, β)                        [Dimension 1]
    slip_s  ~ clip(Normal(μ, σ), 0, ∞)          [Dimension 2 — taker only]
    R_s     = (1 − fee) / (price · (1 + slip_s)) − 1
    f_s     = max(0,  (p_s · R_s − (1−p_s)) / R_s)
    
    if MAKER:
        fill_s ~ Beta(fα, fβ)                   [Dimension 3]
        f_s    = f_s · fill_s

Conservative Kelly  f* = Quantile₂₅({f_s})
f_capped = min(f*, 15%)
size_usd = f_capped · bankroll
```

**Observed behaviour** (prob=0.60, price=0.45, n=200, bankroll=$7):

| Scenario | kelly₂₅ | kelly₅₀ | Trade size |
|:---|:---:|:---:|:---:|
| Taker (1.5% slippage) | 0.209 | 0.252 | $1.00 (capped) |
| Maker (40% fill rate) | 0.081 | 0.101 | $0.57 |
| Sparse n=25 taker | 0.130 | 0.253 | $0.91 |

The maker size ($0.57) is automatically ~35% smaller than the taker size, correctly reflecting the double uncertainty of fill rate + payout.

**Implementation**: `MonteCarloKelly.compute()` — 5,000 paths, 3 sampling dimensions, 25th percentile, automatic regime scaling applied after.

---

### 11.2 Calibration Curve (Favorite-Longshot Bias Correction)

**Motivation**: Prediction markets systematically misprice extreme probabilities. The *favorite-longshot bias* (FLB) is well-documented: longshots (low-probability events) are overpriced while favorites (high-probability events) are underpriced. If we use raw market prices as our prior, we will systematically underestimate our true edge.

**Model**: The power-law calibration transform:

```
p_calibrated = p^(1/γ) / [p^(1/γ) + (1−p)^(1/γ)]
```

where `γ > 1` corrects for FLB. At `γ = 1.0`, no correction. At `γ = 1.08` (empirical default for mature prediction markets):

| Market Price | Calibrated Price |
|:---:|:---:|
| 0.10 | 0.116 (longshot corrected up) |
| 0.30 | 0.313 |
| 0.50 | 0.500 (symmetric) |
| 0.70 | 0.687 |
| 0.90 | 0.884 (favourite corrected down) |

**Rolling Out-of-Sample Fitting (walk-forward)**: After every `FOLD_SIZE = 10` new observations the system runs a walk-forward OOS update:

```
train  = all observations BEFORE the current fold   (fitting set)
val    = current fold of 10 observations            (held-out OOS test)

γ* = argmax_{γ ∈ [0.90, 1.30]}  Σ_{i ∈ train} log L(γ; p_i, y_i)

Applied γ ← γ*  only if len(train) ≥ 20
```

The γ is **never fitted on the validation fold** — it is only evaluated there.  This eliminates the in-sample bias of fitting and scoring on the same data.  The OOS log-likelihood on each fold is tracked; if it deteriorates, the model is drifting out-of-distribution.

This means the calibration curve adapts over time to the actual FLB present in each market, with clean OOS validation at each step.

**Effect on edge calculation**:
```
edge = prob_model − p_calibrated − fee   (instead of using raw mid)
```

**Implementation**: `CalibrationCurve.calibrate()`, `CalibrationCurve.record_outcome()`, `CalibrationCurve._update_gamma()`.

---

### 11.3 Monte Carlo Equity Simulation (Forward-Looking Risk)

**Motivation**: Standard risk metrics (VaR, Kelly) are computed *per trade* and do not account for path-dependency — the fact that losses early in a session reduce the bankroll available for subsequent trades. We need a forward-looking estimate of ruin probability and expected drawdown over the remaining trading session.

**Model**: Given current bankroll `W₀`, remaining trades `T`, probability `p`, and Kelly fraction `f`:

```
For each path k = 1 … N_paths:
    W_{t+1} = W_t + f·W_t·R       with probability p     (win)
    W_{t+1} = W_t − f·W_t         with probability 1−p   (loss)
    
    Stop if  W_t < W_min  (ruin condition)
```

**Per-path statistics recorded**:
```
final equity W_T^k
peak equity   P_t^k  = max_{s≤t} W_s^k      (high-water mark)
max drawdown  DD^k   = max_t (P_t^k − W_t^k) (peak-to-trough)
loss streak   LS^k   = max consecutive losing trades
time u/water  TUW^k  = #{t : W_t^k < W_0}   (trades below starting bankroll)
```

**Aggregated metrics returned**:
```
P(ruin)           = #{k : W_T^k < W_min} / N_paths
W_median, W_5, W_95
E[return]         = mean(W_T^k) − W_0

DD_mean           = mean(DD^k)
DD_p95            = 95th pct of {DD^k}   ← worst-5% drawdown
DD_p99            = 99th pct of {DD^k}
DD_pct_p95        = DD_p95 / W_0         (as % of bankroll)

streak_mean       = mean(LS^k)
streak_p95        = 95th pct of {LS^k}
streak_dist       = {length: count}      (full frequency distribution)

tuw_mean          = mean(TUW^k)          (trades)
tuw_p95           = 95th pct of {TUW^k}
tuw_pct_mean      = tuw_mean / n_trades  (fraction of session)
```

**Decision rules**:
- If `P(ruin) > 20%`: reduce position size by 50% and recheck
- If `DD_p95 > 60%` of bankroll: flag as high-risk session

**Parameters**: 2,000 simulation paths, 10 forward trades, `W_min = $0.50`.

**Implementation**: `EquitySimulator.simulate()`.

---

### 11.4 Order Flow Imbalance (OFI) Signal — *Disabled in v6*

**Motivation**: The best available summary of current buying/selling pressure in a CLOB is the *Order Flow Imbalance* (OFI), introduced by Cont, Kukanov & Stoikov (2014). OFI measures net quote changes at the best bid/ask and has been shown to be a leading indicator of short-term price movements.

**v6 Status**: OFI was disabled after OOS validation showed it did not improve predictive accuracy. The conditional win rate (confident predictions only) was ~50% with OFI and ~77% without — OFI was adding noise. The implementation remains in code but is not used in the live signal pipeline.

**Model**: Let `B_t` = best bid price, `A_t` = best ask price, `V_t^B` = bid volume, `V_t^A` = ask volume. The OFI increment at time `t`:

```
δOFI_t = ΔV^B_t − ΔV^A_t

plus level-change contributions:
    if B_t > B_{t−1}:  δOFI_t += V^B_t       (new bid level: buying pressure)
    if A_t < A_{t−1}:  δOFI_t −= V^A_t       (new ask level: selling pressure)
```

**Normalised signal**: The cumulative OFI over a rolling window of `W = 60` ticks is Z-scored:

```
OFI_signal = clip(ΣOFI / (σ_{OFI} · √W),  −1,  +1)
```

**Use in strategy**:
1. **Gate filter**: If `|OFI_signal| > 0.2` and OFI disagrees with the model's direction, skip the trade.
2. **Edge boost**: When OFI confirms direction, add `OFI_signal × 0.15 × 0.05` to `best_edge`.

```
if side = UP  and  OFI_signal > 0.2:
    edge += OFI_signal × w_ofi × scale
```

**Rationale**: Binary option market makers reprice quickly. A strong OFI signal means the underlying order book is moving in our direction *before* the mid-price updates — a micro-latency edge.

**Implementation**: `OFITracker.update()`, `OFITracker.signal()`.

---

### 11.5 Regime Classification (EWMA Volatility + Momentum) — v6

**Motivation**: Markets cycle between different statistical regimes. Hurst exponent requires 256+ samples and has upward bias on short windows. v6 replaces it with two signals reliable from ~10 ticks.

**Model**: Two statistics jointly classify regime:

**1. EWMA Realized Volatility** (RiskMetrics, λ = 0.94):
```
σ²_t = λ·σ²_{t−1} + (1−λ)·r²_t
vol_ratio = σ_ewma / σ_baseline
```

**2. Directional Momentum Score**:
```
M = Σ sign(r_i) / n  ∈ [-1, +1]
```
Trending declared only when |M| > 2/√n (95% binomial noise band).

**Regime Table**:

| Regime | Condition | Position Scaling |
|:---|:---|:---:|
| Trending | |M| > 2/√n | 1.00× |
| Mean-reverting | vol_ratio < 0.6 | 1.15× |
| Volatile | vol_ratio > 1.8 | 0.70× |
| Neutral | otherwise | 1.00× |
| Unknown | < 10 returns | 0.85× |

**v6 OOS Validation**: Each regime is used for stratification only if its OOS win rate on confident predictions (prob ≥ 0.62 or ≤ 0.38) is ≥ 58%. Non-predictive regimes fall back to pooled surface.

**Implementation**: `RegimeClassifier.classify()` — rolling 120-tick window.

---

### 11.6 Full Distribution Tracking (Beyond the Mean)

**Motivation**: The empirical probability surface stores only the win/loss count per `(pct_diff, time_remaining)` cell. This discards all information about *how much* BTC moves — essential for understanding variance, tail risk, and the skewness of outcomes.

**Model**: For each cell `(p_bin, t_bin)`, we track the full vector of final percentage moves `{Δ_i}`:

```
dist_surface[(p_bin, t_bin)] = [Δ₁, Δ₂, ..., Δₙ]
    where  Δᵢ = (BTC_close − BTC_open) / BTC_open × 100
```

**Statistics computed on demand**:
```
μ  = (1/n) Σ Δᵢ                                (mean final move)
σ² = (1/n) Σ (Δᵢ − μ)²                         (variance)
γ₁ = (1/n) Σ [(Δᵢ − μ)/σ]³                     (skewness)
SE = σ / √n                                      (standard error)
CI = [μ − 1.96·SE,  μ + 1.96·SE]               (95% confidence interval)
P₅, P₉₅                                         (5th/95th percentiles)
```

**Use in strategy**: Displayed in the live dashboard at signal trigger time, giving the trader a complete statistical picture of the current market state:

```
Dist: CI=[−0.005, +0.023] skew=+0.34 ✓   ← CI > 0, positive skew favours UP
Dist: CI=[−0.031, +0.002] skew=−0.12     ← CI crosses zero, uncertainty high
```

**Why skewness matters**: A positive skew `γ₁ > 0` means there are occasional large upward moves in the distribution — increasing the expected payout of UP tokens beyond what the mean alone suggests.

**Implementation**: `EmpiricalEngine.dist_surface`, `EmpiricalEngine.get_distribution()`.

---

### 11.7 Integrated Signal Pipeline (v5)

The six modules interact in a sequential pipeline at each decision point. See v6 pipeline below for current live system.

---

### 11.8 v6 Enhancements (Validated Live System)

**11.8.1 Walk-Forward Empirical Surface**

The empirical surface is built from the first 70% of windows (chronological). The last 30% is held out for OOS validation. No data leakage.

**OOS Metric**: Conditional win rate on *confident* predictions only:
- Predict UP when prob ≥ 0.62
- Predict DOWN when prob ≤ 0.38
- Win rate on these predictions must be ≥ 60% (achieved ~77% in validation)

**11.8.2 Regime-Stratified Bins**

Surface key: `(pct_bin, t_bin, regime)`. Regime from `RegimeClassifier` (EWMA vol + momentum). Regimes merged: volatile + unknown → "uncertain".

A regime is used only if its OOS confident win rate ≥ 58%. Otherwise fall back to pooled `(pct, t)`.

**11.8.3 Empirical Calibration Tracker**

At runtime, `EmpiricalCalibrator` records `(predicted_prob, outcome)` for each resolved trade. Computes:
- **Brier score**: (1/n) Σ(pred − actual)²
- **Reliability diagram**: per-bucket mean predicted vs mean actual

**11.8.4 Mud Zone**

When raw empirical prob ∈ [0.38, 0.62], return 0.5 (no edge). No trade in uncertain regime.

**11.8.5 Hold-to-Expiry Precision Mode**

- Enter only in first 120s of window (before market discovers direction)
- Price zone: 0.35–0.57 (market must be uncertain)
- BTC distance gate: 0.01%–0.20% from strike in signal direction
- No mid-window sells (except catastrophic emergency)
- Min edge: 6%

**11.8.6 Recency Weighting**

Last 25% of training windows get weight 1.0; older windows get weight 0.5. Recent data counts more.

**11.8.7 v6 Pipeline**

```
Tick Data (Coinbase WebSocket — BTC-USD)
        │
        ▼
[5] Regime Classifier ──────────► regime_key  (for lookup)
        │
        ▼
EmpiricalEngine.lookup(pct, t, regime)  ──► prob (mud zone → 0.5)
        │
        ▼
[2] Calibration Curve ───────────► cal_mid
        │
        ▼
  Edge = prob − cal_mid − fee   (conviction: prob ∉ [0.38, 0.62])
        │
        ▼
[1] MC Kelly Sizing ─────────────► trade_size_usd
        │
        ▼
[3] Equity Simulator ────────────► P(ruin) check
        │
        ▼
    EXECUTE (hold to expiry)
```

OFI removed. Entry filters: price zone, window cutoff, BTC distance, min edge.

---

**End of Document**

*This document represents the complete mathematical and statistical framework for a professional-grade statistical arbitrage system targeting Polymarket binary options. All models have been implemented and tested in both paper trading conditions with realistic constraints and live trading on Polygon mainnet.*
# Layer 1-6 Pipeline: Key Design Decisions and Trade-offs
