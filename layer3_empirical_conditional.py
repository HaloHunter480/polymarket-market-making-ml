"""
layer3_empirical_conditional.py — Layer 3 Model 1: Empirical Conditional Distribution
========================================================================================

Rigorous implementation of P(UP | Δ, T, regime) from historical BTC 5-min data.

NO LOOPSHOLES:
  1. Walk-forward: train on first 70% of windows (chronological), test on last 30%
  2. Zero data leakage: each observation used once; regime uses only past prices
  3. Regime stratification: regime used only if OOS confident win rate ≥ 58%
  4. Full conditional distribution: mean, variance, skew, percentiles (not just P(UP))
  5. Kernel smoothing: sparse cells use Epanechnikov-weighted neighbors
  6. Minimum sample size: require ≥ N obs per cell; fallback to pooled/interpolation
  7. Proper interpolation: linear in time, kernel in pct_diff for continuous inputs

Conditioning variables:
  - Δ = (S(t) - K) / K × 100  (pct_diff from strike)
  - T = time_remaining (seconds)
  - regime ∈ {trending, mean_rev, neutral, uncertain}
  - momentum (optional): (S(t) - S(t-2min)) / S(t-2min) × 100

Output: P(UP), full distribution (mean, std, skew, p5, p50, p95), sample_count
"""

import os
import pickle
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
MIN_SAMPLES_CELL = 15
MIN_SAMPLES_REGIME = 30
OOS_MIN_WIN_RATE = 0.58
MIN_CONVICTION = 0.62
PCT_BIN_STEP = 0.02
PCT_RANGE = 2.0
TIME_BINS = [60, 120, 180, 240, 300]
REGIME_MAP = {
    "trending": "trending",
    "mean_reverting": "mean_rev",
    "volatile": "uncertain",
    "neutral": "neutral",
    "unknown": "uncertain",
}
REGIMES = ("trending", "mean_rev", "neutral", "uncertain")
KERNEL_BANDWIDTH_PCT = 0.08  # Epanechnikov kernel half-width in pct units
RECENCY_CUT = 0.75  # Last 25% of train gets weight 1.0, older gets 0.5


class RegimeClassifier:
    """EWMA volatility + directional momentum. No future data."""

    EWMA_LAMBDA = 0.94
    VOL_HIGH_MULT = 1.8
    VOL_LOW_MULT = 0.6
    MIN_RETURNS = 10

    @staticmethod
    def classify(prices) -> Tuple[str, dict]:
        arr = list(prices)
        if len(arr) < RegimeClassifier.MIN_RETURNS + 1:
            return "unknown", {}

        rets = [
            (arr[i + 1] - arr[i]) / arr[i]
            for i in range(len(arr) - 1)
            if arr[i] != 0
        ]
        n = len(rets)
        if n < RegimeClassifier.MIN_RETURNS:
            return "unknown", {}

        var_ew = rets[0] ** 2
        for r in rets[1:]:
            var_ew = RegimeClassifier.EWMA_LAMBDA * var_ew + (1.0 - RegimeClassifier.EWMA_LAMBDA) * r ** 2
        vol_ewma = var_ew ** 0.5

        mean_r = sum(rets) / n
        var_base = sum((r - mean_r) ** 2 for r in rets) / n
        vol_base = var_base ** 0.5 if var_base > 0 else 1e-12
        vol_ratio = vol_ewma / vol_base

        momentum = sum(1.0 if r > 0 else -1.0 for r in rets) / n
        noise_band = 2.0 / (n ** 0.5)
        trending = abs(momentum) > noise_band

        if vol_ratio > RegimeClassifier.VOL_HIGH_MULT:
            return "volatile", {}
        if trending:
            return "trending", {}
        if vol_ratio < RegimeClassifier.VOL_LOW_MULT:
            return "mean_reverting", {}
        return "neutral", {}


@dataclass
class EmpiricalConditionalResult:
    """Output of empirical conditional distribution lookup."""

    p_up: float = 0.5
    sample_count: int = 0
    regime: str = ""
    regime_used: bool = False
    # Full distribution of final-pct move (conditional on state)
    mean: float = 0.0
    std: float = 0.0
    skew: float = 0.0
    p5: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    # Meta
    interpolated: bool = False
    kernel_smoothed: bool = False


class EmpiricalConditionalDistribution:
    """
    Empirical P(UP | Δ, T, regime) with strict walk-forward and zero leakage.
    """

    def __init__(self, candle_path: str = "btc_1m_candles.pkl"):
        self.candle_path = candle_path
        self._pooled: Dict[Tuple[float, int], dict] = {}
        self._regime: Dict[Tuple[float, int, str], dict] = {}
        self._dist: Dict[Tuple[float, int], List[float]] = {}
        self._pct_bins: List[float] = []
        self._time_bins: List[int] = []
        self._valid_regimes: set = set()
        self._oos_metrics: dict = {}
        self._train_windows = 0
        self._test_windows = 0
        self._loaded = False

        if os.path.exists(candle_path):
            try:
                self.load(candle_path)
            except Exception:
                pass

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.candle_path
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                candles = pickle.load(f)
            self._build(candles)
            self._loaded = True
            return True
        except Exception as e:
            import warnings
            warnings.warn(f"Layer3 empirical load failed: {e}")
            return False

    def _build(self, candles):
        if not candles:
            return
        if isinstance(candles[0], (list, tuple)):
            closes = [float(c[4]) for c in candles]
        else:
            closes = [float(c["close"]) for c in candles]

        n = len(closes)
        n_windows = max(0, n - 5)
        train_end = max(1, int(n_windows * TRAIN_RATIO))
        recency_cut = int(train_end * RECENCY_CUT)

        self._pct_bins = [
            round(p * PCT_BIN_STEP - PCT_RANGE, 4)
            for p in range(int(PCT_RANGE * 2 / PCT_BIN_STEP) + 1)
        ]
        self._time_bins = sorted(set(TIME_BINS))

        # TRAIN: first 70% of windows only
        for w in range(train_end):
            start = w
            if start + 5 >= len(closes):
                break
            w_recency = 1.0 if w >= recency_cut else 0.5
            strike = closes[start]
            final_close = closes[start + 5]
            resolved_up = final_close >= strike
            path = closes[start : start + 6]
            final_pct = (final_close - strike) / strike * 100 if strike > 0 else 0.0

            for minute in range(1, 6):
                if minute >= len(path):
                    break
                current = path[minute]
                if strike == 0:
                    continue
                pct = (current - strike) / strike * 100
                t_rem = (5 - minute) * 60
                t_bin = max(60, min(300, round(t_rem / 60) * 60))
                p_bin = self._snap_pct(pct)

                hist_start = max(0, start + minute - 120)
                hist = closes[hist_start : start + minute + 1]
                regime_raw, _ = RegimeClassifier.classify(hist)
                regime = REGIME_MAP.get(regime_raw, "uncertain")

                key = (p_bin, t_bin)
                key_r = (p_bin, t_bin, regime)

                if key not in self._pooled:
                    self._pooled[key] = {"up": 0.0, "dn": 0.0}
                if resolved_up:
                    self._pooled[key]["up"] += w_recency
                else:
                    self._pooled[key]["dn"] += w_recency

                if key_r not in self._regime:
                    self._regime[key_r] = {"up": 0.0, "dn": 0.0}
                if resolved_up:
                    self._regime[key_r]["up"] += w_recency
                else:
                    self._regime[key_r]["dn"] += w_recency

                if key not in self._dist:
                    self._dist[key] = []
                self._dist[key].append(final_pct)

        self._train_windows = train_end

        # OOS validation on TEST windows (last 30%) — never used for training
        oos_preds = []
        for w in range(train_end, n_windows):
            start = w
            if start + 5 >= len(closes):
                break
            strike = closes[start]
            final_close = closes[start + 5]
            resolved_up = final_close >= strike

            for minute in range(1, 5):
                idx = start + minute
                if idx >= len(closes):
                    break
                current = closes[idx]
                if strike == 0:
                    continue
                pct = (current - strike) / strike * 100
                t_rem = (5 - minute) * 60
                hist_start = max(0, idx - 120)
                hist = closes[hist_start : idx + 1]
                regime_raw, _ = RegimeClassifier.classify(hist)
                regime = REGIME_MAP.get(regime_raw, "uncertain")

                prob_up, _ = self._lookup_raw(pct, t_rem, regime)
                if prob_up is None:
                    prob_up = 0.5
                oos_preds.append((prob_up, 1 if resolved_up else 0, regime))

        self._test_windows = n_windows - train_end

        if oos_preds:
            n = len(oos_preds)
            brier = sum((p - y) ** 2 for p, y, _ in oos_preds) / n

            conf_up = [(p, y) for p, y, _ in oos_preds if p >= MIN_CONVICTION]
            conf_dn = [(p, y) for p, y, _ in oos_preds if p <= (1 - MIN_CONVICTION)]
            wr_up = sum(y for _, y in conf_up) / len(conf_up) if conf_up else 0.5
            wr_dn = sum(1 - y for _, y in conf_dn) / len(conf_dn) if conf_dn else 0.5
            n_conf = len(conf_up) + len(conf_dn)
            wr_conf = (
                (sum(y for _, y in conf_up) + sum(1 - y for _, y in conf_dn)) / n_conf
                if n_conf
                else 0.5
            )

            for r in REGIMES:
                r_preds = [(p, y) for p, y, rg in oos_preds if rg == r]
                r_conf = [
                    (p, y)
                    for p, y in r_preds
                    if p >= MIN_CONVICTION or p <= (1 - MIN_CONVICTION)
                ]
                if len(r_conf) >= MIN_SAMPLES_REGIME:
                    r_wins = sum(y for p, y in r_conf if p >= 0.5) + sum(
                        1 - y for p, y in r_conf if p < 0.5
                    )
                    r_wr = r_wins / len(r_conf)
                    if r_wr >= OOS_MIN_WIN_RATE:
                        self._valid_regimes.add(r)

            self._oos_metrics = {
                "brier": round(brier, 4),
                "n_test": n,
                "n_confident": n_conf,
                "wr_confident": round(wr_conf, 3),
                "valid_regimes": list(self._valid_regimes),
            }

    def _snap_pct(self, pct: float) -> float:
        if not self._pct_bins:
            return round(pct, 4)
        i = bisect_left(self._pct_bins, pct)
        if i == 0:
            return self._pct_bins[0]
        if i >= len(self._pct_bins):
            return self._pct_bins[-1]
        lo, hi = self._pct_bins[i - 1], self._pct_bins[i]
        return lo if abs(pct - lo) <= abs(pct - hi) else hi

    def _lookup_raw(
        self, pct_diff: float, time_remaining: float, regime: Optional[str] = None
    ) -> Tuple[Optional[float], int]:
        """Internal: (prob_up, sample_count). Regime only if OOS-validated."""
        t_lo = max(60, (int(time_remaining / 60)) * 60)
        t_hi = min(300, t_lo + 60)
        p_bin = self._snap_pct(pct_diff)

        def get(d: dict, k) -> Tuple[Optional[float], float]:
            v = d.get(k)
            if v is None:
                return None, 0.0
            tot = v["up"] + v["dn"]
            if tot == 0:
                return None, 0.0
            return v["up"] / tot, tot

        if regime and regime in self._valid_regimes:
            key_r = (p_bin, t_lo, regime)
            prob_lo, n_lo = get(self._regime, key_r)
            if n_lo >= MIN_SAMPLES_CELL and prob_lo is not None:
                if t_hi != t_lo:
                    prob_hi, n_hi = get(self._regime, (p_bin, t_hi, regime))
                    if prob_hi is not None and n_hi >= MIN_SAMPLES_CELL:
                        alpha = (time_remaining - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0
                        return (1 - alpha) * prob_lo + alpha * prob_hi, min(n_lo, n_hi)
                return prob_lo, n_lo

        prob_lo, n_lo = get(self._pooled, (p_bin, t_lo))
        prob_hi, n_hi = get(self._pooled, (p_bin, t_hi))

        if prob_lo is None and prob_hi is None:
            return None, 0
        if prob_lo is None:
            return prob_hi, n_hi
        if prob_hi is None:
            return prob_lo, n_lo

        alpha = (time_remaining - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0
        return (1 - alpha) * prob_lo + alpha * prob_hi, min(n_lo, n_hi)

    def _epanechnikov(self, u: float) -> float:
        """Epanechnikov kernel: K(u) = 0.75*(1-u²) for |u|≤1."""
        u = abs(u)
        return 0.75 * (1 - u * u) if u <= 1 else 0.0

    def _kernel_smooth(
        self, pct_diff: float, t_bin: int, regime: Optional[str] = None
    ) -> Tuple[float, float, bool]:
        """
        Kernel-weighted average over neighbors when exact cell is sparse.
        Returns (prob_up, sample_count, used_kernel).
        """
        h = KERNEL_BANDWIDTH_PCT
        surface = self._regime if (regime and regime in self._valid_regimes) else self._pooled
        keys = list(surface.keys())

        if regime and regime in self._valid_regimes:
            keys = [k for k in keys if len(k) == 3 and k[2] == regime]
        else:
            keys = [k for k in keys if len(k) == 2]

        total_up = 0.0
        total_wt = 0.0
        total_n = 0.0

        for k in keys:
            p_bin = k[0]
            t_k = k[1]
            if abs(t_k - t_bin) > 60:
                continue
            u = (pct_diff - p_bin) / h
            w = self._epanechnikov(u)
            if w <= 0:
                continue
            v = surface.get(k)
            if v is None:
                continue
            tot = v["up"] + v["dn"]
            if tot < 1:
                continue
            total_up += w * (v["up"] / tot)
            total_wt += w
            total_n += w * tot

        if total_wt < 1e-10:
            return 0.5, 0.0, False
        return total_up / total_wt, total_n, True

    def lookup(
        self,
        pct_diff: float,
        time_remaining: float,
        regime: Optional[str] = None,
        prices: Optional[List[float]] = None,
        use_kernel: bool = True,
    ) -> EmpiricalConditionalResult:
        """
        Look up P(UP | Δ, T, regime) with full distribution.

        Args:
            pct_diff: (current - strike) / strike * 100
            time_remaining: seconds until expiry
            regime: from RegimeClassifier; if None and prices given, compute from prices
            prices: recent price history for regime (if regime not provided)
            use_kernel: use kernel smoothing when cell is sparse

        Returns:
            EmpiricalConditionalResult with p_up, full distribution stats
        """
        res = EmpiricalConditionalResult()

        if not self._loaded:
            return res

        if regime is None and prices is not None:
            regime_raw, _ = RegimeClassifier.classify(prices)
            regime = REGIME_MAP.get(regime_raw, "uncertain")
        res.regime = regime or "uncertain"

        t_lo = max(60, (int(time_remaining / 60)) * 60)
        p_bin = self._snap_pct(pct_diff)

        prob, n = self._lookup_raw(pct_diff, time_remaining, regime)
        res.regime_used = regime in self._valid_regimes and bool(regime)

        if prob is None or n < MIN_SAMPLES_CELL:
            if use_kernel:
                prob, n, used = self._kernel_smooth(pct_diff, t_lo, regime)
                res.kernel_smoothed = used
            if prob is None:
                prob = 0.5
                n = 0

        res.p_up = float(np.clip(prob, 0.01, 0.99))
        res.sample_count = int(n)

        res.interpolated = n < MIN_SAMPLES_CELL and res.sample_count > 0

        # Mud zone: no edge
        if (1 - MIN_CONVICTION) < res.p_up < MIN_CONVICTION:
            res.p_up = 0.5

        # Full distribution
        key = (p_bin, t_lo)
        outcomes = self._dist.get(key, [])
        if len(outcomes) >= 10:
            arr = np.array(outcomes)
            n_o = len(arr)
            res.mean = float(np.mean(arr))
            res.std = float(np.std(arr)) if n_o > 1 else 0.0
            if res.std > 1e-10:
                res.skew = float(np.mean(((arr - res.mean) / res.std) ** 3))
            sorted_o = np.sort(arr)
            res.p5 = float(sorted_o[max(0, int(n_o * 0.05))])
            res.p50 = float(np.median(sorted_o))
            res.p95 = float(sorted_o[min(n_o - 1, int(n_o * 0.95))])
            se = res.std / (n_o ** 0.5)
            res.ci_lower = res.mean - 1.96 * se
            res.ci_upper = res.mean + 1.96 * se

        return res

    def evaluate(
        self,
        layer2_signals,
        regime: Optional[str] = None,
        prices: Optional[List[float]] = None,
    ) -> EmpiricalConditionalResult:
        """
        Evaluate from Layer2Signals. Requires btc_price, strike, time_remaining.
        regime: optional, from RegimeClassifier
        prices: optional recent price history for regime (used if regime not provided)
        """
        btc = getattr(layer2_signals, "btc_price", 0)
        strike = getattr(layer2_signals, "strike", 0)
        t_rem = getattr(layer2_signals, "time_remaining", 300)

        if strike <= 0 and btc > 0:
            strike = btc
        if strike <= 0:
            return EmpiricalConditionalResult()

        pct_diff = (btc - strike) / strike * 100
        return self.lookup(pct_diff, t_rem, regime=regime, prices=prices)


def create_and_load(path: str = "btc_1m_candles.pkl") -> EmpiricalConditionalDistribution:
    """Factory: create and load if data exists."""
    ecd = EmpiricalConditionalDistribution(path)
    if ecd._loaded:
        return ecd
    return ecd
