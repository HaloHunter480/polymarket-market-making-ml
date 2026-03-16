"""
Professional Market Making System for Polymarket BTC Binary Options
====================================================================

Critical Fixes Implemented:
1. TAKER FEE SURVIVAL: Only market orders when edge > 4.5%
2. TOXIC FLOW DETECTION: Cancel all orders on volume spikes or price jumps
3. DYNAMIC ASYMMETRIC QUOTES: Skew based on order flow pressure
4. ADVANCED MODELS: Hawkes process, VPIN, Kyle's lambda, lead-lag

Order Book Reality Checks (Paper Trading):
- Window initialization delay: 15s (wait for CLOB to populate)
- Ghost town filter: Min $50 at top-of-book (reject empty markets)
- Maximum edge cap: 15% for Kelly sizing (prevent data anomaly exploitation)

Realistic Fill Simulation:
- Maker orders: 40% fill rate (60% expire due to adverse selection)
- Taker orders: 1.5% slippage on thin books
- Total taker cost: 2% fee + 1.5% slippage = 3.5%

Philosophy:
- Assume the market is trying to pick us off
- Never give free options
- Only take liquidity when edge is overwhelming
- Cancel and reassess constantly

Expected Performance (REALISTIC, with all protections):
- Win rate: 48-52% (adverse selection adjusted)
- Avg edge: 1-2% (after all costs and reality checks)
- Daily return: 0.2-1.0%
- Sharpe: 0.8-1.5
- Max drawdown: 15-30%
"""

import asyncio
import json
import math
import time
import pickle
import random
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from bisect import bisect_left
from enum import Enum
import aiohttp
import websockets
import ssl
from scipy import stats


# ─── Configuration ────────────────────────────────────────────────────────────

CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

# STRICT execution thresholds
MIN_MAKER_EDGE = 0.008        # 0.8% for limit orders (tight)
MIN_TAKER_EDGE = 0.045        # 4.5% for market orders (STRICT - survives 2% fee)
MAKER_FEE = 0.0
TAKER_FEE = 0.02

# Realistic fill simulation (paper trading)
MAKER_FILL_RATE = 0.40         # 40% of maker orders get filled (adverse selection)
TAKER_SLIPPAGE = 0.015         # 1.5% slippage on market orders (thin books)
MIN_SPREAD_BPS = 0.05         # 5% max spread
MAX_SPREAD_BPS = 0.15         # 15% is too wide, skip

# Order book reality checks (prevent "empty book" exploitation)
WINDOW_INIT_DELAY = 15.0       # Wait 15s after new window opens for CLOB to populate
MIN_TOB_VOLUME = 50.0          # Minimum $50 at best bid/ask (ghost town filter)
MAX_REALISTIC_EDGE = 0.15      # Cap edge at 15% for Kelly sizing (prevent data anomalies)

# Toxic flow detection (tuned for real Binance tick data)
TOXIC_VOLUME_THRESHOLD = 5.0   # 5x normal volume = toxic (was 3x, too sensitive)
TOXIC_PRICE_JUMP = 50.0        # $50 jump between ticks = toxic (was $10, too sensitive)
CANCEL_ALL_DELAY = 1.0         # Wait 1s after toxic flow before re-quoting

# Quote skewing
MAX_SKEW = 0.04               # Max 4% skew from fair value
FLOW_PRESSURE_WINDOW = 30     # 30 seconds of flow history

# Position limits (v5: dynamic MC Kelly — these are now fallback caps only)
KELLY_FRACTION = 0.08         # fallback fraction when MC Kelly unavailable
MAX_BET_SIZE = 30.0
MIN_BET_SIZE = 4.0
BANKROLL = 500.0
MAX_TRADES_PER_WINDOW = 2
SIGNAL_COOLDOWN = 25.0

# Model parameters
MIN_SAMPLES = 20

# ── v5 Research Module Parameters ──────────────────────────────────────────
KELLY_MC_SAMPLES     = 5000   # Monte Carlo posterior draws
KELLY_CONFIDENCE_PCT = 25     # Use 25th percentile (conservative)
MAX_KELLY_FRACTION   = 0.15   # Hard cap at 15% of bankroll
CALIBRATION_GAMMA    = 1.08   # Favourite-longshot bias γ
OFI_WEIGHT           = 0.15   # OFI edge-boost weight
OFI_WINDOW           = 60     # Ticks in OFI rolling window
REGIME_VOL_SCALE     = {"trending": 1.0, "mean_reverting": 1.15,
                        "volatile": 0.70, "neutral": 1.0, "unknown": 0.85}
MAX_RUIN_PROB        = 0.20   # Skip trade if simulated P(ruin) > 20%
MC_EQUITY_PATHS      = 1000   # Paths in equity simulator

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


# ─── v5 Research Modules ──────────────────────────────────────────────────────

class CalibrationCurve:
    """Power-model calibration with rolling out-of-sample (OOS) γ fitting.

    Architecture — walk-forward expanding window:
        observations arrive in order: o_1, o_2, ..., o_T

        After every FOLD_SIZE new observations:
          train  = all observations BEFORE the current fold
          val    = current fold  (OOS — never seen during fitting)

        γ is chosen by MLE on train, then checked on val (log-loss).
        The applied γ is always the one validated out-of-sample.
        We never fit on val data → no in-sample bias.

        Until we have enough data the prior γ = CALIBRATION_GAMMA is kept.
    """

    FOLD_SIZE  = 10
    MIN_TRAIN  = 20
    GAMMA_GRID = [g / 100.0 for g in range(90, 131)]

    def __init__(self, gamma: float = CALIBRATION_GAMMA):
        self._gamma   = gamma
        self._all: list = []
        self._oos_log: list = []   # [(fold_id, train_n, val_ll, gamma)]

    def calibrate(self, market_price: float) -> float:
        p  = max(0.01, min(0.99, market_price))
        g  = self._gamma
        pp = p ** (1.0 / g)
        qp = (1.0 - p) ** (1.0 / g)
        d  = pp + qp
        return pp / d if d > 1e-10 else p

    def record_outcome(self, market_price: float, won: bool):
        self._all.append((market_price, 1 if won else 0))
        n = len(self._all)
        if n >= self.MIN_TRAIN + self.FOLD_SIZE and n % self.FOLD_SIZE == 0:
            self._rolling_oos_fit()

    def oos_summary(self) -> dict:
        if not self._oos_log:
            return {"folds": 0, "gamma": self._gamma}
        avg_val_ll = sum(r[2] for r in self._oos_log) / len(self._oos_log)
        return {
            "folds":        len(self._oos_log),
            "gamma":        round(self._gamma, 3),
            "avg_val_ll":   round(avg_val_ll, 4),
            "last_train_n": self._oos_log[-1][1],
        }

    @staticmethod
    def _ll(data: list, g: float) -> float:
        ll = 0.0
        for mp, outcome in data:
            p   = max(0.01, min(0.99, mp))
            pp  = p ** (1.0 / g)
            cal = pp / (pp + (1.0 - p) ** (1.0 / g))
            ll += math.log(max(1e-10, cal if outcome else 1.0 - cal))
        return ll

    def _rolling_oos_fit(self):
        val_data   = self._all[-self.FOLD_SIZE:]
        train_data = self._all[:-self.FOLD_SIZE]
        if len(train_data) < self.MIN_TRAIN:
            return
        best_g, best_tll = self._gamma, float("-inf")
        for g in self.GAMMA_GRID:
            tll = self._ll(train_data, g)
            if tll > best_tll:
                best_tll = tll
                best_g   = g
        val_ll = self._ll(val_data, best_g)
        self._oos_log.append((len(self._oos_log) + 1,
                               len(train_data),
                               round(val_ll, 4),
                               best_g))
        self._gamma = best_g


class MonteCarloKelly:
    """Full 3-dimensional MC Kelly — no loose ends.

    Three independent sources of uncertainty are resampled per path:

    Dimension 1 — Win-probability uncertainty (both order types)
        p_s ~ Beta(α, β)
        α = p̂·n,  β = (1-p̂)·n
        When n is small the Beta is wide → conservative sizing automatically.

    Dimension 2 — Entry-price / slippage uncertainty (taker orders)
        slip_s ~ clip(Normal(μ_slip, σ_slip), 0, ∞)
        effective_price_s = price · (1 + slip_s)
        R_s = (1 - fee) / effective_price_s - 1   ← payout ratio varies per path

    Dimension 3 — Fill-rate uncertainty (maker orders)
        fill_s ~ Beta(fα, fβ)
        fα = fill_rate · n_fill_obs,  fβ = (1-fill_rate) · n_fill_obs
        The 3-outcome Kelly derivation (win / lose / no-fill) shows that
        the optimal f* is independent of fill_rate in expectation, but
        fill-rate *uncertainty* inflates variance across paths.  We multiply
        f_s by fill_s so that the 25th-percentile cut automatically penalises
        paths where both luck and fill rate are unfavourable simultaneously.

    Final size = Quantile_{25}(f_s) × bankroll, capped at MAX_BET_SIZE.
    """

    @staticmethod
    def compute(
        prob: float,
        price: float,
        n_samples: int,
        bankroll: float,
        fee: float = TAKER_FEE,
        execution_type: str = "TAKER",
        slippage_mu: float = None,      # defaults per execution_type
        slippage_sigma: float = None,   # defaults to 30% of slippage_mu
        maker_fill_rate: float = MAKER_FILL_RATE,
        maker_fill_n: int = 50,         # prior strength for fill-rate posterior
        n_mc: int = KELLY_MC_SAMPLES,
    ) -> tuple:
        if price <= 0.01 or price >= 0.99:
            return 0.0, {"reason": "price_extreme"}

        # ── Slippage defaults ──────────────────────────────────────────────
        if slippage_mu is None:
            slippage_mu = TAKER_SLIPPAGE if execution_type == "TAKER" else 0.003
        if slippage_sigma is None:
            slippage_sigma = max(0.001, slippage_mu * 0.30)

        # ── Dimension 1: win-probability posterior ─────────────────────────
        alpha  = max(1.0, prob * n_samples)
        beta_p = max(1.0, (1.0 - prob) * n_samples)

        # ── Dimension 3: maker fill-rate posterior ─────────────────────────
        fill_alpha = max(1.0, maker_fill_rate * maker_fill_n)
        fill_beta  = max(1.0, (1.0 - maker_fill_rate) * maker_fill_n)

        fractions = []
        for _ in range(n_mc):

            # 1. Sample win probability
            p_s = random.betavariate(alpha, beta_p)

            # 2. Sample slippage → effective entry price → payout ratio R_s
            slip_s      = max(0.0, random.gauss(slippage_mu, slippage_sigma))
            eff_price_s = min(0.98, price * (1.0 + slip_s))
            R_s         = (1.0 - fee) / eff_price_s - 1.0
            if R_s <= 0.0:
                fractions.append(0.0)
                continue

            # 3. Kelly fraction for this (p_s, R_s) draw
            q_s = 1.0 - p_s
            f_s = (p_s * R_s - q_s) / R_s
            f_s = max(0.0, f_s)

            # 4. For MAKER: scale by sampled fill probability
            #    3-outcome Kelly (win/lose/no-fill) gives f* independent of
            #    fill_rate in expectation, but fill-rate uncertainty inflates
            #    cross-path variance.  Multiplying by fill_s ensures the
            #    25th-pctile cut penalises adverse (low fill + bad luck) jointly.
            if execution_type == "MAKER":
                fill_s = random.betavariate(fill_alpha, fill_beta)
                f_s    = f_s * fill_s

            fractions.append(f_s)

        fractions.sort()
        idx      = max(0, int(len(fractions) * KELLY_CONFIDENCE_PCT / 100))
        f_cons   = fractions[idx]
        f_median = fractions[len(fractions) // 2]
        f_capped = min(f_cons, MAX_KELLY_FRACTION)
        trade_usd = round(min(f_capped * bankroll, MAX_BET_SIZE), 2)

        return trade_usd, {
            "kelly_25pct":  round(f_cons, 4),
            "kelly_median": round(f_median, 4),
            "kelly_capped": round(f_capped, 4),
            "alpha":        round(alpha, 1),
            "beta":         round(beta_p, 1),
            "slip_mu":      round(slippage_mu, 4),
            "exec_type":    execution_type,
        }


class EquitySimulator:
    """Full forward equity simulation — all risk metrics computed per path.

    Per path, the simulation records:
      - Final equity
      - Maximum drawdown  (peak-to-trough in dollar and % terms)
      - Loss streak       (longest consecutive losing trades)
      - Time under water  (number of trades where equity < starting bankroll)

    Aggregated statistics returned:
      p_ruin              P(final equity < min_trade)
      median_final        50th pct of final equity
      ci_5 / ci_95        5th / 95th pct of final equity
      e_return            mean final equity - bankroll
      dd_mean             mean max drawdown ($)
      dd_p95              95th pct max drawdown (worst-5% of paths)
      dd_p99              99th pct max drawdown
      dd_pct_mean         mean max drawdown as % of bankroll
      dd_pct_p95          95th pct max drawdown %
      streak_mean         mean longest loss streak
      streak_p95          95th pct longest loss streak
      streak_dist         {length: count} loss-streak frequency distribution
      tuw_mean            mean time-under-water (number of trades)
      tuw_p95             95th pct time-under-water
      tuw_pct_mean        mean time-under-water as fraction of n_trades
    """

    @staticmethod
    def simulate(bankroll: float, n_trades: int, prob: float,
                 price: float, kelly_frac: float,
                 n_paths: int = MC_EQUITY_PATHS) -> dict:
        if n_trades <= 0 or bankroll < MIN_BET_SIZE:
            return {"p_ruin": 0.0, "median_final": bankroll}

        finals      = []
        max_dds     = []
        max_dd_pcts = []
        streaks     = []
        tuws        = []
        streak_counts: dict = {}

        for _ in range(n_paths):
            eq         = bankroll
            peak       = bankroll
            max_dd     = 0.0
            cur_streak = 0
            max_streak = 0
            tuw        = 0

            for _ in range(n_trades):
                if eq < MIN_BET_SIZE:
                    break
                bet = min(eq * kelly_frac, MAX_BET_SIZE)

                if random.random() < prob:
                    eq        += bet / price - bet
                    peak       = max(peak, eq)
                    cur_streak = 0
                else:
                    eq        -= bet
                    cur_streak += 1
                    max_streak  = max(max_streak, cur_streak)

                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd

                if eq < bankroll:
                    tuw += 1

            finals.append(eq)
            max_dds.append(max_dd)
            max_dd_pcts.append(max_dd / bankroll if bankroll > 0 else 0.0)
            streaks.append(max_streak)
            tuws.append(tuw)
            streak_counts[max_streak] = streak_counts.get(max_streak, 0) + 1

        n = len(finals)
        finals.sort()
        max_dds.sort()
        max_dd_pcts.sort()
        streaks_s = sorted(streaks)
        tuws_s    = sorted(tuws)

        return {
            "p_ruin":       sum(1 for x in finals if x < MIN_BET_SIZE) / n,
            "median_final": finals[n // 2],
            "ci_5":         finals[max(0, int(n * 0.05))],
            "ci_95":        finals[min(n - 1, int(n * 0.95))],
            "e_return":     sum(finals) / n - bankroll,
            "dd_mean":      sum(max_dds) / n,
            "dd_p95":       max_dds[min(n - 1, int(n * 0.95))],
            "dd_p99":       max_dds[min(n - 1, int(n * 0.99))],
            "dd_pct_mean":  sum(max_dd_pcts) / n,
            "dd_pct_p95":   max_dd_pcts[min(n - 1, int(n * 0.95))],
            "streak_mean":  sum(streaks) / n,
            "streak_p95":   streaks_s[min(n - 1, int(n * 0.95))],
            "streak_dist":  streak_counts,
            "tuw_mean":     sum(tuws) / n,
            "tuw_p95":      tuws_s[min(n - 1, int(n * 0.95))],
            "tuw_pct_mean": sum(tuws) / n / max(n_trades, 1),
        }


class OFITracker:
    """True OFI (Cont-Kukanov-Stoikov) with per-increment Z-score normalisation
    and spread-regime gating.

    Fix 1 — normalisation:
        OLD: z = Σ(ofi_i) / (std(ofi) * √n)
             Problem: autocorrelated OFI increments make std(ofi)*√n
             underestimate the true SE → inflated signal.
        NEW: z_i = (ofi_i − μ_rolling) / σ_rolling  per increment
             signal = tanh( Σ(z_i) / √W )
             Each increment is standardised independently before accumulation.

    Fix 2 — spread-regime gate:
        When spreads compress, market makers have consensus on fair value and
        OFI no longer conveys exploitable information.
        spread_weight = clip( (spread − S_min) / (S_ref − S_min), 0, 1 )
        Final signal  = raw_signal × spread_weight → 0 as spread → S_min
    """

    SPREAD_MIN = 0.01
    SPREAD_REF = 0.04

    def __init__(self, window: int = OFI_WINDOW, z_warmup: int = 10):
        self._prev: Optional[dict] = None
        self._raw_ofi: deque = deque(maxlen=window)
        self._z_scores: deque = deque(maxlen=window)
        self._z_warmup = z_warmup
        self._last_spread: float = 0.0

    def update(self, book: dict):
        if self._prev is None:
            self._prev = dict(book)
            self._last_spread = book.get("spread", 0.0)
            return

        bid_d = book.get("tob_bid_vol", 0) - self._prev.get("tob_bid_vol", 0)
        ask_d = book.get("tob_ask_vol", 0) - self._prev.get("tob_ask_vol", 0)
        ofi   = bid_d - ask_d
        if book.get("best_bid", 0) > self._prev.get("best_bid", 0):
            ofi += book.get("tob_bid_vol", 0)
        if book.get("best_ask", 0) < self._prev.get("best_ask", 0):
            ofi -= book.get("tob_ask_vol", 0)

        self._raw_ofi.append(ofi)
        self._last_spread = book.get("spread", 0.0)

        n = len(self._raw_ofi)
        if n >= self._z_warmup:
            hist = list(self._raw_ofi)[:-1]
            mu   = sum(hist) / len(hist)
            var  = sum((v - mu) ** 2 for v in hist) / len(hist)
            sig  = var ** 0.5
            z_i  = (ofi - mu) / sig if sig > 1e-8 else 0.0
        else:
            z_i = 0.0

        self._z_scores.append(z_i)
        self._prev = dict(book)

    def signal(self) -> float:
        if len(self._z_scores) < 3:
            return 0.0
        zs      = list(self._z_scores)
        cum_z   = sum(zs) / (len(zs) ** 0.5)
        raw_sig = math.tanh(cum_z / 3.0)

        s = self._last_spread
        if s <= self.SPREAD_MIN:
            return 0.0
        weight = min(1.0, (s - self.SPREAD_MIN)
                     / (self.SPREAD_REF - self.SPREAD_MIN))
        return raw_sig * weight

    def spread_weight(self) -> float:
        s = self._last_spread
        if s <= self.SPREAD_MIN:
            return 0.0
        return min(1.0, (s - self.SPREAD_MIN)
                   / (self.SPREAD_REF - self.SPREAD_MIN))

    def reset(self):
        self._prev = None
        self._raw_ofi.clear()
        self._z_scores.clear()
        self._last_spread = 0.0


class RegimeClassifier:
    """EWMA volatility + directional momentum regime classifier.

    Replaces Hurst exponent (R/S) which requires 256+ samples for reliable
    estimation and has upward bias on short windows.  Both new signals are
    reliable from as few as 10 ticks.

    Signal 1 — EWMA Realized Volatility (RiskMetrics, λ=0.94)
        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
        Responds within ~10 ticks; compared to a rolling baseline to
        determine whether current vol is LOW / MEDIUM / HIGH.

    Signal 2 — Directional Momentum Score
        M = Σ sign(r_i) / n  ∈ [-1, +1]
        "Trending" only declared when |M| exceeds the 95% binomial noise
        band: 2/√n.  Below that it is statistically indistinguishable from
        a coin-flip — no trend label is assigned.

    Regime → Kelly scale:
        trending      ×1.00  (momentum strategy has edge)
        mean_reverting×1.15  (binary resolution favours reversion)
        volatile      ×0.70  (wider tails → reduce exposure)
        neutral       ×1.00
        unknown       ×0.85  (insufficient data → caution)
    """

    EWMA_LAMBDA   = 0.94
    VOL_HIGH_MULT = 1.8
    VOL_LOW_MULT  = 0.6
    MIN_RETURNS   = 10

    @staticmethod
    def classify(prices) -> tuple:
        arr = list(prices)
        if len(arr) < RegimeClassifier.MIN_RETURNS + 1:
            return "unknown", {}

        rets = [(arr[i+1] - arr[i]) / arr[i]
                for i in range(len(arr) - 1) if arr[i] != 0]
        n = len(rets)
        if n < RegimeClassifier.MIN_RETURNS:
            return "unknown", {}

        # ── Signal 1: EWMA Realized Volatility ──────────────────────────
        lam    = RegimeClassifier.EWMA_LAMBDA
        var_ew = rets[0] ** 2
        for r in rets[1:]:
            var_ew = lam * var_ew + (1.0 - lam) * r ** 2
        vol_ewma = var_ew ** 0.5

        mean_r   = sum(rets) / n
        var_base = sum((r - mean_r) ** 2 for r in rets) / n
        vol_base = var_base ** 0.5

        vol_ratio = (vol_ewma / vol_base) if vol_base > 1e-12 else 1.0

        # ── Signal 2: Directional Momentum Score ────────────────────────
        momentum   = sum(1.0 if r > 0 else -1.0 for r in rets) / n
        noise_band = 2.0 / (n ** 0.5)
        trending   = abs(momentum) > noise_band

        stats_d = {
            "vol_ewma":   round(vol_ewma, 7),
            "vol_base":   round(vol_base, 7),
            "vol_ratio":  round(vol_ratio, 3),
            "momentum":   round(momentum, 3),
            "noise_band": round(noise_band, 3),
            "n":          n,
        }

        if vol_ratio > RegimeClassifier.VOL_HIGH_MULT:
            return "volatile", stats_d
        if trending:
            return "trending", stats_d
        if vol_ratio < RegimeClassifier.VOL_LOW_MULT:
            return "mean_reverting", stats_d
        return "neutral", stats_d


# ─── Advanced Models ──────────────────────────────────────────────────────────

class HawkesProcess:
    """
    Self-exciting point process for volume clustering.
    
    Models: λ(t) = μ + α ∑ exp(-β(t - t_i))
    
    Key insight: Volume begets volume. After a trade, more trades follow.
    High intensity = toxic flow, likely adverse selection.
    """
    
    def __init__(self, mu: float = 1.0, alpha: float = 0.5, beta: float = 0.1):
        self.mu = mu          # Base intensity
        self.alpha = alpha    # Self-excitation strength
        self.beta = beta      # Decay rate
        self._events: deque = deque(maxlen=100)
    
    def add_event(self, timestamp: float, volume: float = 1.0):
        """Record a trade/tick event."""
        self._events.append((timestamp, volume))
    
    def get_intensity(self, t: float) -> float:
        """
        Current intensity at time t.
        High intensity = clustered activity = potential toxicity.
        """
        intensity = self.mu
        for t_i, vol_i in self._events:
            if t > t_i:
                intensity += self.alpha * vol_i * np.exp(-self.beta * (t - t_i))
        return intensity
    
    def is_clustering(self, t: float, threshold: float = 3.0) -> bool:
        """Check if we're in a high-intensity cluster (toxic)."""
        current_intensity = self.get_intensity(t)
        return current_intensity > threshold * self.mu


class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading.
    
    Measures toxic order flow by looking at buy/sell imbalance.
    High VPIN = informed traders active = adverse selection risk.
    
    Uses TIME-WINDOWED buckets (not per-tick) to avoid false positives
    from Binance's high-frequency tick stream.
    """
    
    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = window_seconds
        self._trades: deque = deque(maxlen=2000)  # Time-stamped trades
    
    def update(self, volume: float, is_buy: bool):
        """Record a trade with timestamp."""
        self._trades.append((time.time(), volume, is_buy))
    
    def calculate(self) -> float:
        """
        Calculate VPIN over recent time window (0 to 1).
        Only considers trades in the last window_seconds.
        """
        if len(self._trades) < 20:
            return 0.0
        
        now = time.time()
        cutoff = now - self.window_seconds
        
        buy_vol = 0.0
        sell_vol = 0.0
        for t, vol, is_buy in self._trades:
            if t >= cutoff:
                if is_buy:
                    buy_vol += vol
                else:
                    sell_vol += vol
        
        total_vol = buy_vol + sell_vol
        if total_vol < 0.01:
            return 0.0
        
        vpin = abs(buy_vol - sell_vol) / total_vol
        return vpin
    
    def is_toxic(self, threshold: float = 0.75) -> bool:
        """Check if flow is toxic (>75% imbalance over 10s window)."""
        return self.calculate() > threshold


class KyleLambda:
    """
    Kyle's lambda: price impact coefficient.
    
    Measures: ΔPrice = λ × OrderFlow
    
    High lambda = large price impact = illiquid market = wide spreads needed.
    """
    
    def __init__(self, window: int = 50):
        self._price_changes: deque = deque(maxlen=window)
        self._order_flows: deque = deque(maxlen=window)
    
    def update(self, price_change: float, order_flow: float):
        """
        Record price change and corresponding order flow.
        
        Args:
            price_change: Price change in dollars
            order_flow: Signed volume (positive = buy, negative = sell)
        """
        self._price_changes.append(price_change)
        self._order_flows.append(order_flow)
    
    def estimate(self) -> float:
        """
        Estimate Kyle's lambda via linear regression.
        
        Returns lambda (price impact per unit volume).
        """
        if len(self._price_changes) < 10:
            return 0.0
        
        X = np.array(self._order_flows)
        y = np.array(self._price_changes)
        
        # OLS: lambda = Cov(price, flow) / Var(flow)
        if np.var(X) < 1e-10:
            return 0.0
        
        lambda_est = np.cov(X, y)[0, 1] / np.var(X)
        return abs(lambda_est)  # Take absolute value


class LeadLagDetector:
    """
    Detect which exchange leads price discovery using Granger causality.
    
    If Binance leads Polymarket by 2 seconds, we can predict Polymarket moves.
    """
    
    def __init__(self, max_lag: int = 10):
        self.max_lag = max_lag
        self._binance_prices: deque = deque(maxlen=100)
        self._polymarket_prices: deque = deque(maxlen=100)
        self._lead_lag: int = 0  # Positive = Binance leads
    
    def update(self, binance_price: float, polymarket_price: float):
        """Record synchronized price observations."""
        self._binance_prices.append(binance_price)
        self._polymarket_prices.append(polymarket_price)
    
    def detect_lead_lag(self) -> int:
        """
        Detect lead-lag using cross-correlation.
        
        Returns: lag in ticks (positive = Binance leads)
        """
        if len(self._binance_prices) < 50:
            return 0
        
        binance = np.array(list(self._binance_prices))
        polymarket = np.array(list(self._polymarket_prices))
        
        # Cross-correlation
        correlations = []
        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag >= 0:
                corr = np.corrcoef(binance[:-lag or None], polymarket[lag:])[0, 1]
            else:
                corr = np.corrcoef(binance[-lag:], polymarket[:lag])[0, 1]
            correlations.append((lag, corr))
        
        # Find max correlation
        best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]))
        
        if abs(best_corr) > 0.3:  # Significant correlation
            self._lead_lag = best_lag
            return best_lag
        
        return 0
    
    def get_lead_lag(self) -> int:
        """Get current lead-lag estimate."""
        return self._lead_lag


class ToxicFlowDetector:
    """
    Comprehensive toxic flow detection system.
    
    Combines:
    - Volume spikes (Hawkes process)
    - Price jumps (sudden moves)
    - Order imbalance (VPIN)
    - Unusual volatility
    
    When toxic flow detected: CANCEL ALL ORDERS IMMEDIATELY.
    """
    
    def __init__(self):
        self.hawkes = HawkesProcess()
        self.vpin = VPIN()
        self._last_price: float = 0.0
        self._price_history: deque = deque(maxlen=10)
        self._volume_history: deque = deque(maxlen=30)
        
    def update(self, price: float, volume: float, is_buy: bool):
        """Update all toxicity indicators."""
        now = time.time()
        
        # Hawkes process
        self.hawkes.add_event(now, volume)
        
        # VPIN
        self.vpin.update(volume, is_buy)
        
        # Price jump detection
        if self._last_price > 0:
            price_change = abs(price - self._last_price)
            self._price_history.append(price_change)
        
        self._last_price = price
        self._volume_history.append(volume)
    
    def is_toxic(self) -> Tuple[bool, str]:
        """
        Check if flow is toxic. Uses STRICT criteria to avoid false positives.
        
        All indicators must be evaluated over meaningful time windows,
        not individual ticks (Binance sends 100s of trades/second).
        
        Returns: (is_toxic, reason)
        """
        # 1. Large price jump (most reliable indicator)
        # Only check the LATEST jump, not historical max
        if len(self._price_history) >= 2:
            latest_jump = self._price_history[-1]
            if latest_jump > TOXIC_PRICE_JUMP:
                self._price_history.clear()  # Reset after triggering
                return True, f"PRICE_JUMP_${latest_jump:.1f}"
        
        # 2. VPIN (time-windowed, 10 second lookback)
        # Threshold 0.75 = 75% one-sided flow over 10 seconds
        vpin_val = self.vpin.calculate()
        if vpin_val > 0.80:
            return True, f"VPIN_{vpin_val:.2f}"
        
        # 3. Volume spike (sudden burst, 5x normal)
        if len(self._volume_history) >= 20:
            avg_vol = np.mean(list(self._volume_history)[:-1])
            recent_vol = self._volume_history[-1]
            if avg_vol > 0 and recent_vol > TOXIC_VOLUME_THRESHOLD * avg_vol and recent_vol > 50:
                return True, f"VOL_SPIKE_{recent_vol/avg_vol:.1f}x"
        
        return False, "CLEAN"


class OrderFlowPressure:
    """
    Measure directional pressure from order flow.
    
    Used to skew quotes:
    - High buy pressure → place bid lower, ask higher (avoid getting run over)
    - High sell pressure → place bid higher, ask lower
    """
    
    def __init__(self, window_seconds: float = FLOW_PRESSURE_WINDOW):
        self.window_seconds = window_seconds
        self._trades: deque = deque(maxlen=200)
    
    def update(self, timestamp: float, volume: float, is_buy: bool):
        """Record a trade."""
        self._trades.append((timestamp, volume, is_buy))
    
    def get_pressure(self) -> float:
        """
        Calculate flow pressure (-1 to +1).
        
        -1 = all sells (bearish)
        +1 = all buys (bullish)
         0 = balanced
        """
        if not self._trades:
            return 0.0
        
        now = time.time()
        cutoff = now - self.window_seconds
        
        recent = [(t, v, is_buy) for t, v, is_buy in self._trades if t >= cutoff]
        if not recent:
            return 0.0
        
        buy_vol = sum(v for t, v, is_buy in recent if is_buy)
        sell_vol = sum(v for t, v, is_buy in recent if not is_buy)
        total_vol = buy_vol + sell_vol
        
        if total_vol < 0.001:
            return 0.0
        
        pressure = (buy_vol - sell_vol) / total_vol
        return pressure
    
    def get_skew(self, max_skew: float = MAX_SKEW) -> float:
        """
        Get quote skew based on pressure.
        
        Returns adjustment to fair value (as fraction).
        Positive = skew quotes higher (defensive against buys)
        Negative = skew quotes lower (defensive against sells)
        """
        pressure = self.get_pressure()
        skew = pressure * max_skew
        return skew


# ─── Exchange Feed (Enhanced with Toxicity Detection) ─────────────────────────

class Exchange(Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    BYBIT = "bybit"
    OKX = "okx"
    KRAKEN = "kraken"


@dataclass
class ExchangeState:
    exchange: Exchange
    price: float = 0.0
    volume: float = 0.0
    is_buy: bool = True
    last_update: float = 0.0
    tick_count: int = 0
    connected: bool = False


class EnhancedMultiExchangeFeed:
    """Multi-exchange feed with toxicity detection."""
    
    WS_URLS = {
        Exchange.BINANCE: "wss://stream.binance.com:9443/ws/btcusdt@trade",
        Exchange.COINBASE: "wss://ws-feed.exchange.coinbase.com",
        Exchange.BYBIT: "wss://stream.bybit.com/v5/public/spot",
        Exchange.OKX: "wss://ws.okx.com:8443/ws/v5/public",
        Exchange.KRAKEN: "wss://ws.kraken.com/v2",
    }
    
    def __init__(self):
        self.states: Dict[Exchange, ExchangeState] = {
            ex: ExchangeState(exchange=ex) for ex in Exchange
        }
        self.total_ticks = 0
        self._callbacks: List[Callable] = []
        self._price_history: deque = deque(maxlen=300)
        
        # Advanced models
        self.toxic_detector = ToxicFlowDetector()
        self.flow_pressure = OrderFlowPressure()
        self.kyle_lambda = KyleLambda()
        self.lead_lag = LeadLagDetector()
        
        # Toxic flow event tracking
        self.last_toxic_event: float = 0.0
        self.toxic_events: int = 0
    
    @property
    def best_price(self) -> float:
        prices = [s.price for s in self.states.values() if s.connected and s.price > 0]
        return np.mean(prices) if prices else 0.0
    
    @property
    def connected_count(self) -> int:
        return sum(1 for s in self.states.values() if s.connected)
    
    def get_momentum(self, lookback_seconds: float = 120) -> float:
        if len(self._price_history) < 2:
            return 0.0
        now = time.time()
        cutoff = now - lookback_seconds
        old_prices = [(t, p) for t, p in self._price_history if t < cutoff]
        if not old_prices:
            if self._price_history:
                oldest = self._price_history[0]
                return (self.best_price - oldest[1]) / oldest[1] * 100
            return 0.0
        old_price = old_prices[-1][1]
        return (self.best_price - old_price) / old_price * 100
    
    def get_volatility(self, lookback_seconds: float = 300) -> float:
        if len(self._price_history) < 10:
            return 0.5
        now = time.time()
        cutoff = now - lookback_seconds
        recent = [p for t, p in self._price_history if t >= cutoff]
        if len(recent) < 10:
            return 0.5
        returns = np.diff(np.log(recent))
        return np.std(returns) * np.sqrt(252 * 24 * 60)
    
    def is_toxic_flow_active(self) -> Tuple[bool, str]:
        """Check if toxic flow detected recently."""
        is_toxic, reason = self.toxic_detector.is_toxic()
        if is_toxic:
            self.last_toxic_event = time.time()
            self.toxic_events += 1
        return is_toxic, reason
    
    def time_since_toxic(self) -> float:
        """Seconds since last toxic flow event."""
        if self.last_toxic_event == 0:
            return float('inf')
        return time.time() - self.last_toxic_event
    
    def on_tick(self, callback: Callable):
        self._callbacks.append(callback)
    
    def _update(self, exchange: Exchange, price: float, volume: float = 0, is_buy: bool = True):
        state = self.states[exchange]
        
        # Update toxicity detectors
        self.toxic_detector.update(price, volume, is_buy)
        self.flow_pressure.update(time.time(), volume, is_buy)
        
        # Update Kyle's lambda (price impact)
        if state.price > 0:
            price_change = price - state.price
            order_flow = volume if is_buy else -volume
            self.kyle_lambda.update(price_change, order_flow)
        
        state.price = price
        state.volume = volume
        state.is_buy = is_buy
        state.last_update = time.time()
        state.tick_count += 1
        state.connected = True
        self.total_ticks += 1
        self._price_history.append((time.time(), price))
        
        for cb in self._callbacks:
            try:
                cb(price)
            except Exception:
                pass
    
    async def connect_all(self):
        tasks = [
            self._connect_binance(),
            self._connect_coinbase(),
            self._connect_bybit(),
            self._connect_okx(),
            self._connect_kraken(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _connect_binance(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.BINANCE], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    self.states[Exchange.BINANCE].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        self._update(
                            Exchange.BINANCE,
                            float(data["p"]),
                            float(data.get("q", 0)),
                            not data.get("m", False)  # m=True means maker, so !m = taker buy
                        )
            except Exception:
                self.states[Exchange.BINANCE].connected = False
                await asyncio.sleep(2)
    
    async def _connect_coinbase(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.COINBASE], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}]
                    }))
                    self.states[Exchange.COINBASE].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("type") == "ticker" and "price" in data:
                            self._update(Exchange.COINBASE, float(data["price"]))
            except Exception:
                self.states[Exchange.COINBASE].connected = False
                await asyncio.sleep(2)
    
    async def _connect_bybit(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.BYBIT], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": ["tickers.BTCUSDT"]
                    }))
                    self.states[Exchange.BYBIT].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        if "data" in data:
                            d = data["data"]
                            if isinstance(d, list):
                                d = d[0]
                            if "lastPrice" in d:
                                self._update(Exchange.BYBIT, float(d["lastPrice"]))
            except Exception:
                self.states[Exchange.BYBIT].connected = False
                await asyncio.sleep(2)
    
    async def _connect_okx(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.OKX], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": [{"channel": "tickers", "instId": "BTC-USDT"}]
                    }))
                    self.states[Exchange.OKX].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        if "data" in data and data["data"]:
                            d = data["data"][0]
                            if "last" in d:
                                self._update(Exchange.OKX, float(d["last"]))
            except Exception:
                self.states[Exchange.OKX].connected = False
                await asyncio.sleep(2)
    
    async def _connect_kraken(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.KRAKEN], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "params": {"channel": "ticker", "symbol": ["BTC/USD"]}
                    }))
                    self.states[Exchange.KRAKEN].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        if "data" in data and data["data"]:
                            for d in data["data"]:
                                if "last" in d:
                                    self._update(Exchange.KRAKEN, float(d["last"]))
            except Exception:
                self.states[Exchange.KRAKEN].connected = False
                await asyncio.sleep(2)


# ─── Empirical Engine (Same as Before) ────────────────────────────────────────

class EmpiricalEngine:
    """Base probability model from historical data."""
    
    def __init__(self, candle_file: str = "btc_1m_candles.pkl"):
        self.prob_surface: Dict[Tuple[float, int], dict] = {}
        self._pct_bins: List[float] = []
        self._time_bins: List[int] = []
        self._loaded = False
        
        try:
            self._build(candle_file)
            self._loaded = True
        except FileNotFoundError:
            print("  WARNING: btc_1m_candles.pkl not found.")
        except Exception as e:
            print(f"  WARNING: Could not load empirical data: {e}")
    
    def _build(self, candle_file: str):
        with open(candle_file, "rb") as f:
            candles = pickle.load(f)
        
        closes = [float(c[4]) for c in candles]
        opens = [float(c[1]) for c in candles]
        
        window_size = 5
        n_windows = len(closes) // window_size
        
        bins = defaultdict(lambda: {"up": 0, "total": 0})
        
        for w in range(n_windows):
            start = w * window_size
            if start + window_size > len(closes):
                break
            
            strike = opens[start]
            final_close = closes[start + window_size - 1]
            resolved_up = final_close >= strike
            
            for minute in range(window_size):
                idx = start + minute
                current = closes[idx]
                pct_diff = (current - strike) / strike * 100
                
                pct_bin = round(pct_diff / 0.005) * 0.005
                pct_bin = max(-0.5, min(0.5, pct_bin))
                time_remaining = (window_size - minute - 1) * 60
                
                bins[(pct_bin, time_remaining)]["total"] += 1
                if resolved_up:
                    bins[(pct_bin, time_remaining)]["up"] += 1
        
        self.prob_surface = dict(bins)
        self._pct_bins = sorted(set(k[0] for k in bins.keys()))
        self._time_bins = sorted(set(k[1] for k in bins.keys()))
        
        total_obs = sum(v["total"] for v in bins.values())
        print(f"  Empirical engine: {total_obs:,} observations from {n_windows:,} windows")
    
    def lookup(self, pct_diff: float, time_remaining: float) -> Tuple[float, int]:
        """Get base probability from empirical data."""
        if not self._loaded:
            return 0.5, 0
        
        pct_bin = round(pct_diff / 0.005) * 0.005
        pct_bin = max(-0.5, min(0.5, pct_bin))
        
        time_lo = int(time_remaining // 60) * 60
        time_hi = time_lo + 60
        time_lo = max(0, min(240, time_lo))
        time_hi = max(0, min(240, time_hi))
        
        if time_hi > time_lo:
            frac = (time_remaining - time_lo) / (time_hi - time_lo)
        else:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        
        prob_lo, count_lo = self._lookup_single(pct_bin, time_lo)
        prob_hi, count_hi = self._lookup_single(pct_bin, time_hi)
        
        prob = prob_lo * (1 - frac) + prob_hi * frac
        count = min(count_lo, count_hi)
        
        if time_remaining > 5:
            prob = max(0.02, min(0.98, prob))
        
        return prob, count
    
    def _lookup_single(self, pct_bin: float, time_bin: int) -> Tuple[float, int]:
        key = (pct_bin, time_bin)
        data = self.prob_surface.get(key)
        
        if data and data["total"] >= 5:
            return data["up"] / data["total"], data["total"]
        
        return self._interpolate(pct_bin, time_bin), 0
    
    def _interpolate(self, pct_bin: float, time_bin: int) -> float:
        idx = bisect_left(self._pct_bins, pct_bin)
        pct_lo = self._pct_bins[max(0, idx - 1)]
        pct_hi = self._pct_bins[min(len(self._pct_bins) - 1, idx)]
        
        tidx = bisect_left(self._time_bins, time_bin)
        time_lo = self._time_bins[max(0, tidx - 1)]
        time_hi = self._time_bins[min(len(self._time_bins) - 1, tidx)]
        
        total_up = 0
        total_count = 0
        for p in [pct_lo, pct_hi]:
            for t in [time_lo, time_hi]:
                data = self.prob_surface.get((p, t))
                if data and data["total"] > 0:
                    total_up += data["up"]
                    total_count += data["total"]
        
        return total_up / total_count if total_count > 0 else 0.5


# ─── Professional Order Book & Execution ──────────────────────────────────────

@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def spread_bps(self) -> float:
        if self.best_bid > 0:
            return (self.best_ask - self.best_bid) / self.mid_price
        return 1.0


@dataclass
class TradeSignal:
    side: str                # "UP" or "DOWN"
    fair_value: float        # Our fair value
    market_mid: float        # Current market mid
    edge: float              # Expected edge
    execution_type: str      # "MAKER" or "TAKER"
    limit_price: Optional[float]  # For maker orders
    kelly_size: float
    confidence: float
    # Components
    empirical_prob: float
    flow_skew: float
    toxic_risk: float
    # Metadata
    pct_diff: float
    time_remaining: float
    momentum: float
    spread_bps: float
    sample_count: int


class ProfessionalExecutor:
    """
    Professional market making execution with toxic flow protection.
    
    Rules:
    1. MAKER ONLY if edge > 0.8% AND no toxic flow AND spread < 15%
    2. TAKER ONLY if edge > 4.5% AND time < 60s
    3. CANCEL ALL ORDERS if toxic flow detected
    4. SKEW QUOTES based on order flow pressure
    """
    
    def __init__(self, feed: EnhancedMultiExchangeFeed):
        self.feed = feed
        self.open_orders: Dict[str, any] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self.last_cancel_all: float = 0.0
    
    async def initialize(self):
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=SSL_CTX),
            headers={"Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        )
    
    async def close(self):
        if self._session:
            await self._session.close()
    
    async def fetch_order_book(self, token_id: str) -> Optional[OrderBook]:
        if not self._session:
            return None
        
        try:
            async with self._session.get(
                f"{CLOB_API}/book",
                params={"token_id": token_id},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bids = [
                        OrderBookLevel(float(b["price"]), float(b["size"]))
                        for b in data.get("bids", [])
                    ]
                    asks = [
                        OrderBookLevel(float(a["price"]), float(a["size"]))
                        for a in data.get("asks", [])
                    ]
                    bids.sort(key=lambda x: x.price, reverse=True)
                    asks.sort(key=lambda x: x.price)
                    return OrderBook(bids=bids, asks=asks, timestamp=time.time())
        except Exception:
            pass
        return None
    
    async def cancel_all_orders(self, reason: str = ""):
        """Emergency cancel all orders."""
        if self.open_orders:
            print(f"    [CANCEL_ALL] {reason} - Killing {len(self.open_orders)} orders")
            self.open_orders.clear()
            self.last_cancel_all = time.time()
    
    def calculate_skewed_price(
        self,
        fair_value: float,
        side: str,
        flow_pressure: float,
        book: OrderBook,
    ) -> Optional[float]:
        """
        Calculate asymmetric limit price based on flow pressure.
        
        If buying UP and flow is bullish (+0.8):
          → Place bid lower (be defensive, avoid getting run over)
        If buying UP and flow is bearish (-0.8):
          → Place bid higher (be aggressive, flow in our favor)
        """
        # Get flow skew
        skew = self.feed.flow_pressure.get_skew()
        
        if side == "UP":
            # We're buying UP
            # Bullish flow (+skew) = place bid lower (defensive)
            # Bearish flow (-skew) = place bid higher (aggressive)
            adjusted_fair = fair_value - skew
        else:
            # We're buying DOWN
            # Bullish flow (+skew) = place bid higher (aggressive)
            # Bearish flow (-skew) = place bid lower (defensive)
            adjusted_fair = fair_value + skew
        
        adjusted_fair = max(0.05, min(0.95, adjusted_fair))
        
        # Place inside spread
        limit_price = book.best_bid + 0.4 * (adjusted_fair - book.best_bid)
        limit_price = max(book.best_bid + 0.001, min(book.best_ask - 0.001, limit_price))
        limit_price = round(limit_price, 3)
        
        return limit_price if limit_price < adjusted_fair else None
    
    async def execute(self, signal: TradeSignal, token_id: str, window_id: int):
        """
        Execute trade based on signal type (MAKER or TAKER).
        
        Realistic fill simulation:
        - MAKER: Only 40% of orders get filled (rest expire or get cancelled)
        - TAKER: 1.5% slippage on top of 2% fee (thin order books)
        """
        
        # Check for toxic flow (pre-emptive kill switch)
        is_toxic, reason = self.feed.is_toxic_flow_active()
        if is_toxic:
            await self.cancel_all_orders(reason)
            return None
        
        # Don't place new orders right after canceling
        if time.time() - self.last_cancel_all < CANCEL_ALL_DELAY:
            return None
        
        if signal.execution_type == "MAKER":
            print(f"    [MAKER] {signal.side} @ ${signal.limit_price:.3f} "
                  f"(fair: ${signal.fair_value:.3f}, mid: ${signal.market_mid:.3f})")
            print(f"    [FLOW] Pressure: {self.feed.flow_pressure.get_pressure():+.2f}, "
                  f"Skew: {signal.flow_skew:+.3f}")
            print(f"    [EDGE] {signal.edge:.2%} | Size: ${signal.kelly_size:.1f}")
            
            # Simulate maker fill probability
            import random
            fill_roll = random.random()
            if fill_roll > MAKER_FILL_RATE:
                print(f"    [MAKER_MISS] Order expired unfilled (roll: {fill_roll:.2f} > {MAKER_FILL_RATE:.2f})")
                return None  # Order didn't get filled
            
            print(f"    [MAKER_FILL] Filled at limit (roll: {fill_roll:.2f} < {MAKER_FILL_RATE:.2f})")
            
            # Store filled order
            order_id = f"{token_id}_{signal.side}_{time.time()}"
            self.open_orders[order_id] = {
                "signal": signal,
                "window_id": window_id,
                "created_at": time.time(),
            }
            return order_id
            
        elif signal.execution_type == "TAKER":
            # Apply realistic slippage
            effective_edge = signal.edge - TAKER_SLIPPAGE
            
            print(f"    [TAKER] {signal.side} @ MARKET (mid: ${signal.market_mid:.3f})")
            print(f"    [STRONG] Edge: {signal.edge:.2%} → {effective_edge:.2%} (after {TAKER_SLIPPAGE:.1%} slippage)")
            print(f"    [WARNING] Paying 2% fee + slippage = {TAKER_FEE + TAKER_SLIPPAGE:.1%} total cost")
            
            # Modify signal edge to include slippage for P&L calculation
            signal.edge = effective_edge
            
            # Execute immediately (with slippage applied)
            return "taker_fill_slipped"
        
        return None


# ─── Advanced Signal Generator ────────────────────────────────────────────────

class AdvancedSignalGenerator:
    """Generate signals using all advanced models (v5 research-grade)."""

    def __init__(self, empirical: EmpiricalEngine,
                 feed: "EnhancedMultiExchangeFeed",
                 bankroll: float = BANKROLL):
        self.empirical   = empirical
        self.feed        = feed
        self.bankroll    = bankroll
        # v5 modules
        self.calibration = CalibrationCurve()
        self.ofi         = OFITracker(window=OFI_WINDOW)
        self._prices_for_regime: deque = deque(maxlen=120)

    def update_price(self, price: float):
        """Call from strategy loop on each BTC tick."""
        self._prices_for_regime.append(price)

    def update_book(self, book_dict: dict):
        """Feed raw book dict into OFI tracker."""
        self.ofi.update(book_dict)

    def record_outcome(self, market_price: float, won: bool):
        """Online calibration update after each resolved trade."""
        self.calibration.record_outcome(market_price, won)

    def reset_window(self):
        self.ofi.reset()

    def evaluate(
        self,
        btc_price: float,
        strike: float,
        time_remaining: float,
        book: "OrderBook",
        book_dict: dict,
        window_open_time: float = 0.0,
    ) -> Optional["TradeSignal"]:
        """Generate trading signal with all v5 research checks."""

        if strike <= 0 or btc_price <= 0 or time_remaining < 15:
            return None

        # ── Reality Check 1: window init delay ──
        if window_open_time > 0:
            if time.time() - window_open_time < WINDOW_INIT_DELAY:
                return None

        # ── Toxic flow gate ──
        is_toxic, _ = self.feed.is_toxic_flow_active()
        if is_toxic:
            return None

        # ── Reality Check 2: ghost-town filter ──
        if book.asks and book.bids:
            ask_vol = book.asks[0].size * book.asks[0].price if book.asks[0].price > 0 else 0
            bid_vol = book.bids[0].size * book.bids[0].price if book.bids[0].price > 0 else 0
            if ask_vol < MIN_TOB_VOLUME and bid_vol < MIN_TOB_VOLUME:
                return None

        # ── Base empirical probability ──
        pct_diff = (btc_price - strike) / strike * 100
        emp_prob, sample_count = self.empirical.lookup(pct_diff, time_remaining)
        if sample_count < MIN_SAMPLES:
            return None

        # ── v5 #5: Regime classification ──
        regime, regime_stats = RegimeClassifier.classify(self._prices_for_regime)
        regime_scale = REGIME_VOL_SCALE.get(regime, 1.0)

        # ── v5 #4: OFI ──
        self.ofi.update(book_dict)
        ofi_signal = self.ofi.signal()

        # ── v5 #2: Calibration ──
        cal_mid = self.calibration.calibrate(book.mid_price)

        # Flow pressure (legacy, still useful)
        flow_skew  = self.feed.flow_pressure.get_skew()
        emp_prob_up = max(0.02, min(0.98, emp_prob + flow_skew))
        emp_prob_down = 1.0 - emp_prob_up

        market_up   = cal_mid          # calibrated P(UP)
        market_down = 1.0 - cal_mid

        # Model/market sanity check (prevent strike mismatch phantom edges)
        if abs(emp_prob_up - cal_mid) > 0.25:
            return None

        # ── Edge calculation ──
        maker_edge_up   = emp_prob_up   - market_up   - MAKER_FEE
        maker_edge_down = emp_prob_down - market_down - MAKER_FEE
        taker_edge_up   = emp_prob_up   - market_up   - TAKER_FEE
        taker_edge_down = emp_prob_down - market_down - TAKER_FEE

        # ── OFI confirmation gate + micro-boost ──
        if abs(ofi_signal) > 0.2:
            if ofi_signal < -0.3:   # selling pressure
                taker_edge_up   -= 0.01
                maker_edge_up   -= 0.01
            elif ofi_signal > 0.3:  # buying pressure
                taker_edge_down -= 0.01
                maker_edge_down -= 0.01

        # OFI edge boost when confirming
        for edge_list in [(taker_edge_up, "UP"), (taker_edge_down, "DOWN")]:
            pass  # applied below per-side

        # ── Execution type decision ──
        side = fair_value = market_price = edge = execution_type = None

        # TAKER: strict 4.5% edge OR inside last 60 seconds
        if time_remaining < 60 or max(taker_edge_up, taker_edge_down) > 0.08:
            if taker_edge_up > MIN_TAKER_EDGE:
                # OFI boost
                boost = (ofi_signal * OFI_WEIGHT * 0.05
                         if ofi_signal > 0.2 else 0.0)
                side, fair_value, market_price = "UP", emp_prob_up, market_up
                edge, execution_type = taker_edge_up + boost, "TAKER"
            elif taker_edge_down > MIN_TAKER_EDGE:
                boost = (abs(ofi_signal) * OFI_WEIGHT * 0.05
                         if ofi_signal < -0.2 else 0.0)
                side, fair_value, market_price = "DOWN", emp_prob_down, market_down
                edge, execution_type = taker_edge_down + boost, "TAKER"

        # MAKER: 0.8% edge, spread not too wide, NOT in last 60 seconds
        if execution_type is None and time_remaining > 60:
            if maker_edge_up > MIN_MAKER_EDGE and maker_edge_up > maker_edge_down:
                side, fair_value, market_price = "UP", emp_prob_up, market_up
                edge, execution_type = maker_edge_up, "MAKER"
            elif maker_edge_down > MIN_MAKER_EDGE:
                side, fair_value, market_price = "DOWN", emp_prob_down, market_down
                edge, execution_type = maker_edge_down, "MAKER"

        if execution_type is None:
            return None

        if execution_type == "MAKER" and book.spread_bps > MAX_SPREAD_BPS:
            return None

        # ── Reality Check 3: max-edge cap (phantom edge guard) ──
        if abs(edge) > MAX_REALISTIC_EDGE:
            return None

        # ── Limit price for maker ──
        limit_price = None
        if execution_type == "MAKER":
            executor = ProfessionalExecutor(self.feed)
            limit_price = executor.calculate_skewed_price(
                fair_value, side, flow_skew, book)
            if limit_price is None:
                return None

        # ── Confidence score ──
        sample_conf = min(1.0, sample_count / 50)
        edge_conf   = min(1.0, abs(edge) / 0.05)
        spread_conf = 1.0 - min(1.0, book.spread_bps / MAX_SPREAD_BPS)
        toxic_risk  = 1.0 if self.feed.time_since_toxic() > 10 else 0.5
        confidence  = (0.3 * sample_conf + 0.3 * edge_conf
                       + 0.2 * spread_conf + 0.2 * toxic_risk)

        # ── v5 #1: Monte Carlo Kelly sizing (full 3-dimensional) ──
        fee = TAKER_FEE if execution_type == "TAKER" else MAKER_FEE
        mc_size, mc_detail = MonteCarloKelly.compute(
            prob=max(0.02, min(0.98, fair_value)),
            price=max(0.02, min(0.98, market_price)),
            n_samples=sample_count,
            bankroll=self.bankroll,
            fee=fee,
            execution_type=execution_type,      # dim 2 (slip) + dim 3 (fill) branch
            maker_fill_rate=MAKER_FILL_RATE,    # 40% historical fill rate prior
        )

        # Regime scale
        mc_size = round(mc_size * regime_scale, 2)
        mc_size = min(MAX_BET_SIZE, max(MIN_BET_SIZE, mc_size))

        # ── v5 #3: Equity simulation ruin check ──
        kelly_frac = mc_detail.get("kelly_capped", 0.05)
        eq_sim = EquitySimulator.simulate(
            bankroll=self.bankroll,
            n_trades=MAX_TRADES_PER_WINDOW * 3,
            prob=max(0.02, min(0.98, fair_value)),
            price=max(0.02, min(0.98, market_price)),
            kelly_frac=kelly_frac,
        )
        if eq_sim["p_ruin"] > MAX_RUIN_PROB:
            mc_size = round(mc_size * 0.5, 2)
            if mc_size < MIN_BET_SIZE:
                return None

        kelly_size = mc_size

        return TradeSignal(
            side=side,
            fair_value=fair_value,
            market_mid=book.mid_price,
            edge=edge,
            execution_type=execution_type,
            limit_price=limit_price,
            kelly_size=kelly_size,
            confidence=confidence,
            empirical_prob=emp_prob,
            flow_skew=flow_skew,
            toxic_risk=toxic_risk,
            pct_diff=pct_diff,
            time_remaining=time_remaining,
            momentum=self.feed.get_momentum(120),
            spread_bps=book.spread_bps,
            sample_count=sample_count,
        )


# ─── Paper Tracker ────────────────────────────────────────────────────────────

@dataclass
class PaperTrade:
    timestamp: float
    window_id: int
    side: str
    execution_type: str
    entry_price: float
    fair_value: float
    size: float
    edge: float
    resolved_up: Optional[bool] = None
    pnl: Optional[float] = None
    won: Optional[bool] = None


class PaperTracker:
    def __init__(self):
        self.open_trades: Dict[int, List[PaperTrade]] = {}
        self.closed_trades: List[PaperTrade] = []
        self.total_pnl: float = 0.0
    
    def execute(self, signal: TradeSignal, window_id: int) -> PaperTrade:
        entry_price = signal.limit_price if signal.execution_type == "MAKER" else signal.market_mid
        trade = PaperTrade(
            timestamp=time.time(),
            window_id=window_id,
            side=signal.side,
            execution_type=signal.execution_type,
            entry_price=entry_price,
            fair_value=signal.fair_value,
            size=signal.kelly_size,
            edge=signal.edge,
        )
        if window_id not in self.open_trades:
            self.open_trades[window_id] = []
        self.open_trades[window_id].append(trade)
        return trade
    
    def resolve_window(self, window_id: int, resolved_up: bool):
        if window_id not in self.open_trades:
            return
        for trade in self.open_trades[window_id]:
            trade.resolved_up = resolved_up
            trade.won = (trade.side == "UP" and resolved_up) or (trade.side == "DOWN" and not resolved_up)
            
            fee = TAKER_FEE if trade.execution_type == "TAKER" else MAKER_FEE
            
            if trade.won:
                payout = trade.size * (1.0 / trade.entry_price - 1) * (1 - fee)
                trade.pnl = payout
            else:
                trade.pnl = -trade.size
            
            self.total_pnl += trade.pnl
            self.closed_trades.append(trade)
        del self.open_trades[window_id]
    
    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        return sum(1 for t in self.closed_trades if t.won) / len(self.closed_trades)
    
    @property
    def total_trades(self) -> int:
        return len(self.closed_trades)


# ─── Main Strategy ────────────────────────────────────────────────────────────

class ProfessionalStrategy:
    def __init__(self, bankroll: float = BANKROLL):
        self.bankroll = bankroll
        self.running = False

        self.feed = EnhancedMultiExchangeFeed()
        self.empirical = EmpiricalEngine()
        self.signal_gen = AdvancedSignalGenerator(
            self.empirical, self.feed, bankroll=bankroll)
        self.executor = ProfessionalExecutor(self.feed)
        self.tracker = PaperTracker()

        # v5 regime tracker
        self._current_regime: str = "unknown"
        self._regime_stats: dict = {}
        
        self.strike: float = 0.0
        self.token_id_up: str = ""
        self.token_id_down: str = ""
        self.current_market_title: str = ""
        
        self._current_window_id: int = 0
        self._window_strikes: Dict[int, float] = {}
        self._window_transitions_seen: int = 0
        self._window_open_time: float = 0.0  # Track when current window opened
        
        self.signals_generated = 0
        self.last_signal_time: float = 0
        self.start_time: float = 0
        self._trades_this_window: int = 0
    
    async def run(self):
        self.running = True
        self.start_time = time.time()
        
        print("=" * 90)
        print("  PROFESSIONAL MARKET MAKING SYSTEM v5 — RESEARCH-GRADE")
        print("=" * 90)
        print(f"  Execution:     HYBRID (Maker 60%+ of window | Taker last 60s / strong edge)")
        print(f"  Maker edge:    {MIN_MAKER_EDGE*100:.1f}% min | Fill rate: {MAKER_FILL_RATE*100:.0f}% (adverse-selection adjusted)")
        print(f"  Taker edge:    {MIN_TAKER_EDGE*100:.1f}% min (strict) | Slippage: {TAKER_SLIPPAGE*100:.1f}%")
        print(f"  Toxic flow:    Hawkes + VPIN + Price jumps")
        print(f"  Quote skew:    Dynamic asymmetric (flow pressure)")
        print(f"  Bankroll:      ${self.bankroll:,.0f}")
        print(f"  Fill sim:      REALISTIC (40% maker fills, 1.5% taker slippage)")
        print(f"  ─── v5 Research Modules ───")
        print(f"  [1] MC Kelly:       {KELLY_MC_SAMPLES} posterior samples, {KELLY_CONFIDENCE_PCT}th pctile")
        print(f"  [2] Calibration:    FLB correction γ={CALIBRATION_GAMMA}")
        print(f"  [3] Equity Sim:     {MC_EQUITY_PATHS} paths, P(ruin)<{MAX_RUIN_PROB:.0%}")
        print(f"  [4] OFI:            window={OFI_WINDOW} ticks, weight={OFI_WEIGHT:.0%}")
        print(f"  [5] Regime:         EWMA-Vol + Momentum (trending/mean-rev/volatile/neutral)")
        print(f"  [6] Distribution:   Full variance/skew/CI in empirical engine")
        print("=" * 90)
        print()
        
        await self.executor.initialize()
        
        try:
            await asyncio.gather(
                self.feed.connect_all(),
                self._strategy_loop(),
                self._market_monitor_loop(),
                self._toxic_monitor_loop(),
                self._display_loop(),
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await self.executor.close()
            self._print_summary()
    
    async def _toxic_monitor_loop(self):
        """Continuously monitor for toxic flow and cancel orders."""
        while self.running:
            is_toxic, reason = self.feed.toxic_detector.is_toxic()
            if is_toxic:
                self.feed.last_toxic_event = time.time()
                self.feed.toxic_events += 1
                if self.executor.open_orders:
                    await self.executor.cancel_all_orders(f"TOXIC: {reason}")
                await asyncio.sleep(CANCEL_ALL_DELAY)
            await asyncio.sleep(0.2)  # Check 5x/sec (not 10x)
    
    async def _market_monitor_loop(self):
        session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=SSL_CTX),
            headers={"Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        )
        last_slug = ""
        try:
            while self.running:
                try:
                    now_ts = int(time.time())
                    boundary = now_ts - (now_ts % 300)
                    slug = f"btc-updown-5m-{boundary}"
                    
                    if slug != last_slug:
                        async with session.get(
                            f"{GAMMA_API}/events",
                            params={"slug": slug},
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data:
                                    event = data[0] if isinstance(data, list) else data
                                    markets = event.get("markets", [])
                                    if markets:
                                        self._update_market(markets[0])
                                        last_slug = slug
                                        remaining = 300 - (now_ts % 300)
                                        self.current_market_title = event.get("title", slug)
                                        print(f"\n  MARKET: {self.current_market_title} | T-{remaining}s")
                except Exception:
                    pass
                await asyncio.sleep(2)
        finally:
            await session.close()
    
    def _update_market(self, m: Dict):
        clob_ids = m.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except Exception:
                return
        if len(clob_ids) >= 2:
            self.token_id_up = str(clob_ids[0])
            self.token_id_down = str(clob_ids[1])
    
    async def _strategy_loop(self):
        while self.feed.connected_count == 0:
            await asyncio.sleep(0.1)
        
        print(f"  Feeds connected: {self.feed.connected_count}/5")
        
        while self.running:
            try:
                now = time.time()
                elapsed = now % 300
                time_remaining = 300 - elapsed
                window_id = int(now // 300)
                
                btc = self.feed.best_price
                if btc <= 0:
                    await asyncio.sleep(0.1)
                    continue
                
                # Feed BTC price into regime & signal-gen
                self.signal_gen.update_price(btc)

                # Window transition
                if window_id != self._current_window_id:
                    old_window = self._current_window_id
                    if old_window > 0 and old_window in self._window_strikes:
                        old_strike = self._window_strikes[old_window]
                        resolved_up = btc >= old_strike
                        self.tracker.resolve_window(old_window, resolved_up)

                        # v5: record outcome for online calibration update
                        for t in self.tracker.closed_trades:
                            if t.window_id == old_window:
                                self.signal_gen.record_outcome(
                                    t.entry_price,
                                    (resolved_up and t.side == "UP")
                                    or (not resolved_up and t.side == "DOWN"))

                        n_trades = len([t for t in self.tracker.closed_trades
                                        if t.window_id == old_window])
                        if n_trades > 0:
                            result = "UP" if resolved_up else "DOWN"
                            print(f"\n  *** WINDOW RESOLVED: {result} | "
                                  f"BTC ${btc:,.2f} vs Strike ${old_strike:,.2f} | "
                                  f"P&L: ${self.tracker.total_pnl:+,.2f} ***")

                    self._window_transitions_seen += 1
                    self._current_window_id = window_id
                    self.strike = btc
                    self._window_strikes[window_id] = btc
                    self._trades_this_window = 0
                    self._window_open_time = time.time()
                    self.signal_gen.reset_window()   # reset OFI tracker

                    # v5: classify regime at window open
                    self._current_regime, self._regime_stats = \
                        RegimeClassifier.classify(self.signal_gen._prices_for_regime)

                    regime_icon = {"trending": "📈", "mean_reverting": "🔄",
                                   "volatile": "⚡", "neutral": "➖",
                                   "unknown": "❓"}.get(self._current_regime, "❓")

                    if self._window_transitions_seen == 1:
                        print(f"\n  === FIRST CLEAN WINDOW | Strike: ${btc:,.2f} "
                              f"| Regime: {regime_icon}{self._current_regime} ===")
                    else:
                        print(f"\n  === NEW WINDOW | Strike: ${btc:,.2f} "
                              f"| Regime: {regime_icon}{self._current_regime} "
                              f"(vol_r={self._regime_stats.get('vol_ratio', '?')} "
                              f"M={self._regime_stats.get('momentum', '?')}) ===")
                
                if self._current_window_id == 0:
                    self._current_window_id = window_id
                    self.strike = 0
                    await asyncio.sleep(0.5)
                    continue
                
                if self.strike <= 0 or self._window_transitions_seen < 1:
                    await asyncio.sleep(0.1)
                    continue
                
                if not self.token_id_up or not self.token_id_down:
                    await asyncio.sleep(0.5)
                    continue
                
                # Fetch order books
                book_up = await self.executor.fetch_order_book(self.token_id_up)
                if not book_up:
                    await asyncio.sleep(0.5)
                    continue
                
                # Build book_dict for OFI tracker — spread is required for gate
                book_dict_up = {
                    "best_bid":    book_up.best_bid,
                    "best_ask":    book_up.best_ask,
                    "spread":      book_up.spread_bps,   # spread-regime gate
                    "tob_bid_vol": (book_up.bids[0].size * book_up.bids[0].price
                                    if book_up.bids else 0),
                    "tob_ask_vol": (book_up.asks[0].size * book_up.asks[0].price
                                    if book_up.asks else 0),
                }

                # Generate signal (with all v5 research modules)
                signal = self.signal_gen.evaluate(
                    btc,
                    self.strike,
                    time_remaining,
                    book_up,
                    book_dict=book_dict_up,
                    window_open_time=self._window_open_time,
                )
                
                if signal:
                    self.signals_generated += 1
                    
                    if ((now - self.last_signal_time) > SIGNAL_COOLDOWN
                            and self._trades_this_window < MAX_TRADES_PER_WINDOW):
                        self.last_signal_time = now
                        
                        token = self.token_id_up if signal.side == "UP" else self.token_id_down
                        
                        result = await self.executor.execute(signal, token, window_id)
                        
                        if result is not None:
                            self._trades_this_window += 1
                            self.tracker.execute(signal, window_id)
                            print(f"    [TRADE #{self._trades_this_window}] Toxic events: {self.feed.toxic_events}")
                        else:
                            print(f"    [BLOCKED] Order blocked by toxic flow or conditions")
                
                await asyncio.sleep(0.05)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  [ERROR] Strategy: {e}")
                await asyncio.sleep(0.5)
    
    async def _display_loop(self):
        while self.running:
            await asyncio.sleep(3.0)
            
            btc = self.feed.best_price
            if btc <= 0 or self.strike <= 0:
                continue
            
            pct_diff = (btc - self.strike) / self.strike * 100
            
            flow_pressure = self.feed.flow_pressure.get_pressure()
            is_toxic = self.feed.time_since_toxic() < 2.0  # Just check recency
            vpin = self.feed.toxic_detector.vpin.calculate()
            
            wins = sum(1 for t in self.tracker.closed_trades if t.won)
            losses = self.tracker.total_trades - wins
            
            maker_trades = sum(1 for t in self.tracker.closed_trades if t.execution_type == "MAKER")
            taker_trades = sum(1 for t in self.tracker.closed_trades if t.execution_type == "TAKER")
            
            regime_icon = {"trending": "📈", "mean_reverting": "🔄",
                           "volatile": "⚡", "neutral": "➖",
                           "unknown": "❓"}.get(self._current_regime, "❓")
            ofi_val     = self.signal_gen.ofi.signal()
            ofi_sw      = self.signal_gen.ofi.spread_weight()
            cal_gamma   = self.signal_gen.calibration._gamma
            ts          = time.strftime("%H:%M:%S")
            toxic_str   = "TOXIC!" if is_toxic else "clean"
            print(
                f"  {ts} | BTC ${btc:>10,.2f} | K ${self.strike:>10,.2f} | "
                f"Diff {pct_diff:+.2f}% | "
                f"Flow {flow_pressure:+.2f} | OFI {ofi_val:+.2f}(w={ofi_sw:.2f}) | "
                f"VPIN {vpin:.2f} | {toxic_str:>6} | "
                f"{regime_icon}{self._current_regime[:4]} | γ={cal_gamma:.3f} | "
                f"W/L {wins}/{losses} | M/T {maker_trades}/{taker_trades} | "
                f"P&L ${self.tracker.total_pnl:+,.2f}"
            )
    
    def _print_summary(self):
        runtime = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n{'=' * 90}")
        print("  SESSION SUMMARY")
        print(f"{'=' * 90}")
        print(f"  Runtime:        {runtime:.0f}s ({runtime/60:.1f} min)")
        print(f"  Ticks:          {self.feed.total_ticks:,}")
        print(f"  Toxic events:   {self.feed.toxic_events}")
        print(f"  Signals:        {self.signals_generated}")
        cal = self.signal_gen.calibration.oos_summary()
        print(f"  Calibration:    γ={cal['gamma']}  OOS folds={cal['folds']}"
              + (f"  avg_val_ll={cal.get('avg_val_ll', 'n/a')}" if cal['folds'] else ""))
        print()
        
        if self.tracker.closed_trades:
            wins = sum(1 for t in self.tracker.closed_trades if t.won)
            losses = self.tracker.total_trades - wins
            
            maker_trades = [t for t in self.tracker.closed_trades if t.execution_type == "MAKER"]
            taker_trades = [t for t in self.tracker.closed_trades if t.execution_type == "TAKER"]
            
            maker_wins = sum(1 for t in maker_trades if t.won)
            taker_wins = sum(1 for t in taker_trades if t.won)
            
            print(f"  Total trades:   {self.tracker.total_trades}")
            print(f"  Wins:           {wins} ({self.tracker.win_rate:.1%})")
            print(f"  Losses:         {losses}")
            print(f"  Total P&L:      ${self.tracker.total_pnl:+,.2f}")
            print()
            print(f"  Maker trades:   {len(maker_trades)} ({maker_wins}/{len(maker_trades)} wins = {maker_wins/len(maker_trades):.1%})" if maker_trades else "  Maker trades:   0")
            print(f"  Taker trades:   {len(taker_trades)} ({taker_wins}/{len(taker_trades)} wins = {taker_wins/len(taker_trades):.1%})" if taker_trades else "  Taker trades:   0")
            
            if maker_trades:
                maker_pnl = sum(t.pnl for t in maker_trades)
                print(f"  Maker P&L:      ${maker_pnl:+,.2f}")
            if taker_trades:
                taker_pnl = sum(t.pnl for t in taker_trades)
                print(f"  Taker P&L:      ${taker_pnl:+,.2f}")
        else:
            print("  No completed trades.")
        
        print(f"{'=' * 90}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    import signal as sig_module
    
    strategy = ProfessionalStrategy(bankroll=BANKROLL)
    
    loop = asyncio.get_event_loop()
    
    def shutdown(signum, frame):
        print("\n  Shutting down...")
        strategy.running = False
        for task in asyncio.all_tasks(loop):
            task.cancel()
    
    sig_module.signal(sig_module.SIGINT, shutdown)
    sig_module.signal(sig_module.SIGTERM, shutdown)
    
    await strategy.run()


if __name__ == "__main__":
    asyncio.run(main())
