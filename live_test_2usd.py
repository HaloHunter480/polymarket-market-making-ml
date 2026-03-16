"""
Polymarket Scalper v5 — Research-Grade Strategy
================================================

6 Research Improvements:
  1. Uncertainty-Adjusted Kelly (Monte Carlo posterior sampling)
  2. Calibration Curve (corrects market price → true probability bias)
  3. Monte Carlo Equity Simulation (forward-looking risk assessment)
  4. Order Flow Imbalance (OFI) confirmation signal
  5. Regime Classification (trending / mean-reverting / volatile)
  6. Full Distribution Tracking (variance, skew, confidence intervals)

Bug Fixes:
  - Dynamic Kelly sizing (replaces hardcoded $2)
  - Robust sell with retry + on-chain confirmation wait
  - Actual token balance query before selling

Run: python3 live_test_2usd.py
"""

import asyncio
import aiohttp
import websockets
import json
import ssl
import time
import os
import math
import random
import pickle
from bisect import bisect_left
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
import pathlib

load_dotenv(pathlib.Path(__file__).parent / ".env", override=True)


###############################################################################
# LIVE LOGGER — structured JSONL + console, instruments every decision
###############################################################################

class LiveLogger:
    """Writes one JSON object per line to logs/live_TIMESTAMP.jsonl.

    Events logged:
      SESSION_START   boot-up state, backtest benchmark loaded
      WINDOW_START    new 5-min window, strike, regime
      WINDOW_END      resolution + realised P&L for this window
      TICK            every 3-second BTC heartbeat
      SIGNAL          signal candidate found (before sizing)
      REJECT          signal rejected and why
      ORDER_SENT      order submitted to CLOB, with latency_ms
      ORDER_RESULT    exchange response (filled / rejected / partial)
      ADVERSE_SEL     price snapshot 10 / 30 / 60 s post-entry
      REGIME_SHIFT    regime label changed
      SESSION_END     full session stats + live-vs-backtest comparison
    """

    # Backtest benchmarks (from our 300-trade run)
    BACKTEST = {
        "win_rate":        0.743,
        "profit_factor":   3.547,
        "ev_per_trade_usd":2.52,
        "edge_mean_pct":   2.73,
        "max_dd_pct":      4.97,
        "fill_rate_pct":   7.3,
        "sharpe":          0.210,
        "maker_win_rate":  0.735,
        "taker_win_rate":  1.000,
    }

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"logs/live_{ts}.jsonl"
        self._fh  = open(self.path, "a", buffering=1)  # line-buffered
        self.session_start = time.time()
        # Runtime accumulators for comparison
        self.ticks_total    = 0
        self.signals_seen   = 0
        self.rejects        = {}   # reason → count
        self.orders_sent    = []   # [{time, size, price, side, latency_ms}]
        self.orders_filled  = []   # [{...}]
        self.adv_sel_snaps  = []   # [{entry, snap_10s, snap_30s, snap_60s}]
        self.regime_history = []   # [(timestamp, regime)]
        self.window_results = []   # [{window, result, pnl}]
        self._last_regime   = None
        print(f"  📝 Live log → {self.path}")

    def _emit(self, event_type: str, payload: dict):
        rec = {"t": round(time.time(), 3),
               "event": event_type,
               **payload}
        self._fh.write(json.dumps(rec) + "\n")

    def session_start_log(self, bankroll, limits):
        self._emit("SESSION_START", {
            "bankroll": bankroll,
            "limits":   limits,
            "backtest_benchmark": self.BACKTEST,
        })

    def tick(self, btc, strike, pct_diff, edge, ofi, regime, time_remaining):
        self.ticks_total += 1
        if self.ticks_total % 10 == 0:   # log every 10th tick (not every 3s)
            self._emit("TICK", {
                "btc": btc, "strike": strike, "pct_diff": round(pct_diff,4),
                "edge": round(edge,4), "ofi": round(ofi,4),
                "regime": regime, "t_rem": round(time_remaining,1),
            })
        # Regime shift detection
        if regime != self._last_regime and self._last_regime is not None:
            self.regime_shift(self._last_regime, regime)
        self._last_regime = regime

    def signal(self, side, edge, prob, cal_mid, size, regime, ofi, spread,
               kelly_detail, eq_sim):
        self.signals_seen += 1
        self._emit("SIGNAL", {
            "side": side, "edge": round(edge,4), "prob": round(prob,4),
            "cal_mid": round(cal_mid,4), "size_usd": round(size,2),
            "regime": regime, "ofi": round(ofi,4),
            "spread": round(spread,4),
            "kelly_25pct": kelly_detail.get("kelly_25pct"),
            "kelly_median": kelly_detail.get("kelly_median"),
            "p_ruin": eq_sim.get("p_ruin"),
            "dd_p95": eq_sim.get("dd_p95"),
        })

    def reject(self, reason: str, detail: dict = None):
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        self._emit("REJECT", {"reason": reason, **(detail or {})})

    def order_sent(self, token_id, side, size_usd, price, latency_ms,
                   response_raw):
        rec = {
            "token": token_id[:20], "side": side,
            "size_usd": size_usd, "price": price,
            "latency_ms": round(latency_ms, 1),
            "response": str(response_raw)[:200],
        }
        self.orders_sent.append(rec)
        self._emit("ORDER_SENT", rec)

    def order_result(self, order_id, status, filled_size, filled_price,
                     slippage_bps):
        rec = {
            "order_id": str(order_id)[:20],
            "status": status,
            "filled_size": filled_size,
            "filled_price": filled_price,
            "slippage_bps": round(slippage_bps, 1),
        }
        self.orders_filled.append(rec)
        self._emit("ORDER_RESULT", rec)

    def adverse_selection(self, entry_price, entry_side,
                          snap_10s, snap_30s, snap_60s):
        """Record BTC price drift 10/30/60s after entry."""
        def adv(snap):
            if snap is None or entry_price <= 0:
                return None
            # For BUY UP: we want BTC to go up. positive = favourable
            # snap is BTC price relative to strike
            return round(snap - entry_price, 4)

        rec = {
            "entry_price": entry_price,
            "side": entry_side,
            "drift_10s":  adv(snap_10s),
            "drift_30s":  adv(snap_30s),
            "drift_60s":  adv(snap_60s),
        }
        self.adv_sel_snaps.append(rec)
        self._emit("ADVERSE_SEL", rec)

    def regime_shift(self, old_regime: str, new_regime: str):
        self.regime_history.append((time.time(), old_regime, new_regime))
        self._emit("REGIME_SHIFT", {"from": old_regime, "to": new_regime})

    def window_end(self, window_num, strike, final_btc, resolution,
                   window_pnl, cum_pnl):
        rec = {
            "window": window_num, "strike": strike,
            "final_btc": final_btc, "result": resolution,
            "window_pnl": round(window_pnl, 4),
            "cum_pnl": round(cum_pnl, 4),
        }
        self.window_results.append(rec)
        self._emit("WINDOW_END", rec)

    def session_end(self, trades_placed, total_spent, total_received,
                    runtime_min, empirical_calibrator=None):
        net_pnl  = total_received - total_spent
        win_rate = (sum(1 for o in self.orders_filled
                        if o.get("status") == "won") /
                    max(1, len(self.orders_filled)))

        # Latency stats
        lats = [o["latency_ms"] for o in self.orders_sent if "latency_ms" in o]
        lat_mean = sum(lats)/len(lats) if lats else 0
        lat_p95  = sorted(lats)[int(len(lats)*0.95)] if lats else 0

        # Adverse selection: fraction where price moved against us 30s post
        adv_count = len(self.adv_sel_snaps)
        adv_against = sum(1 for s in self.adv_sel_snaps
                          if s.get("drift_30s") is not None
                          and s["drift_30s"] < 0)

        # Slippage
        slippages = [o["slippage_bps"] for o in self.orders_filled
                     if "slippage_bps" in o]
        avg_slip = sum(slippages)/len(slippages) if slippages else 0

        live = {
            "trades":          trades_placed,
            "total_spent":     round(total_spent, 4),
            "total_received":  round(total_received, 4),
            "net_pnl":         round(net_pnl, 4),
            "signals_seen":    self.signals_seen,
            "fill_rate_pct":   round(trades_placed/max(1,self.signals_seen)*100, 1),
            "latency_mean_ms": round(lat_mean, 1),
            "latency_p95_ms":  round(lat_p95, 1),
            "adv_sel_pct":     round(adv_against/max(1,adv_count)*100, 1),
            "avg_slippage_bps":round(avg_slip, 1),
            "regime_shifts":   len(self.regime_history),
            "reject_reasons":  self.rejects,
            "runtime_min":     round(runtime_min, 1),
        }

        # ── Live vs Backtest comparison ──
        BT = self.BACKTEST
        comparison = {
            "fill_rate_pct":  {"live": live["fill_rate_pct"],
                               "backtest": BT["fill_rate_pct"],
                               "delta": live["fill_rate_pct"] - BT["fill_rate_pct"]},
            "edge_mean_pct":  {"live": "see_signals",
                               "backtest": BT["edge_mean_pct"]},
            "latency_ms":     {"live": round(lat_mean,1), "backtest": "N/A (simulated)"},
            "adverse_sel_pct":{"live": live["adv_sel_pct"], "backtest": "~46.7%"},
        }

        emp = empirical_calibrator.reliability() if empirical_calibrator else {}
        oos_records = empirical_calibrator.get_oos_records() if empirical_calibrator else []
        self._emit("SESSION_END", {
            "live": live,
            "comparison": comparison,
            "empirical_calibration": emp,
            "oos_signals": oos_records,  # Step 1: p_model, p_market, direction, outcome
        })

        # Print comparison table to console
        print()
        print("=" * 70)
        print("  LIVE vs BACKTEST COMPARISON")
        print("=" * 70)
        print(f"  {'Metric':<28} {'Live':>12}  {'Backtest':>12}  {'Delta':>10}")
        print(f"  {'-'*28} {'-'*12}  {'-'*12}  {'-'*10}")
        print(f"  {'Net P&L':<28} ${net_pnl:>+10.2f}  {'N/A':>12}  {'':>10}")
        print(f"  {'Signals generated':<28} {self.signals_seen:>12}  {'4130/875W':>12}")
        print(f"  {'Fill rate %':<28} {live['fill_rate_pct']:>11.1f}%  {BT['fill_rate_pct']:>11.1f}%"
              f"  {live['fill_rate_pct']-BT['fill_rate_pct']:>+9.1f}%")
        print(f"  {'Order latency mean':<28} {lat_mean:>9.0f}ms  {'~0ms (sim)':>12}")
        print(f"  {'Adverse sel 30s':<28} {live['adv_sel_pct']:>11.1f}%  {'~46.7%':>12}")
        print(f"  {'Avg slippage':<28} {avg_slip:>9.1f}bps  {'150bps (sim)':>12}")
        print(f"  {'Regime shifts':<28} {len(self.regime_history):>12}  {'N/A':>12}")
        print(f"  {'Reject: no edge':<28} {self.rejects.get('no_edge',0):>12}  {'N/A':>12}")
        print(f"  {'Reject: toxic':<28} {self.rejects.get('toxic_flow',0):>12}  {'N/A':>12}")
        print(f"  {'Reject: price zone':<28} {self.rejects.get('price_zone',0):>12}  {'N/A':>12}")
        print(f"  {'Reject: phantom edge':<28} {self.rejects.get('phantom_edge',0):>12}  {'N/A':>12}")
        print(f"  {'Reject: entry cutoff':<28} {self.rejects.get('entry_cutoff',0):>12}  {'N/A':>12}")
        print(f"  {'Reject: BTC wrong side':<28} {self.rejects.get('btc_wrong_side',0):>12}  {'N/A':>12}")
        print(f"  {'Reject: BTC too close':<28} {self.rejects.get('btc_too_close',0):>12}  {'N/A':>12}")
        print(f"  {'Reject: BTC too far':<28} {self.rejects.get('btc_too_far',0):>12}  {'N/A':>12}")
        print()
        if lats:
            print(f"  Latency distribution (n={len(lats)}):")
            print(f"    mean={lat_mean:.0f}ms  p50={sorted(lats)[len(lats)//2]:.0f}ms  "
                  f"p95={lat_p95:.0f}ms  max={max(lats):.0f}ms")
        if self.adv_sel_snaps:
            drift_30s = [s["drift_30s"] for s in self.adv_sel_snaps
                         if s.get("drift_30s") is not None]
            if drift_30s:
                print(f"\n  Adverse selection (BTC drift 30s post-entry, n={len(drift_30s)}):")
                m = sum(drift_30s)/len(drift_30s)
                print(f"    mean drift: {m:+.2f}  "
                      f"against: {adv_against}/{len(drift_30s)} "
                      f"({live['adv_sel_pct']:.1f}%)")
                print(f"    (backtest assumed ~46.7% adverse — real microstructure "
                      f"may differ)")
        print(f"\n  Structured log saved: {self.path}")
        print("=" * 70)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

# ─── HARD SAFETY LIMITS ──────────────────────────────────────────────────────
# Hold-to-expiry mode.  Precision entry only.  No mid-window sells.

MAX_SINGLE_TRADE_USD = 1.00    # $1 per trade (Polymarket min; you have $2.02)
MAX_SESSION_SPEND_USD = 2.00   # $2 session budget (2 × $1 trades max)
MAX_TRADES_SESSION   = 2
MIN_EDGE_TO_TRADE    = 0.060   # 6% edge floor — higher bar = fewer, better trades
MAX_EDGE_BELIEVABLE  = 0.15    # phantom edge cap
WINDOW_INIT_WAIT     = 15.0
MIN_TOB_VOL          = 5.0
MIN_SAMPLES_REQUIRED = 50   # was 20 — require more support for prediction
EXIT_POLL_INTERVAL   = 0.3    # position monitor: check every 300ms
LOOP_INTERVAL_FAST   = 0.10   # hot path (no signal): 10 Hz
LOOP_INTERVAL_RETRY  = 0.15   # reject (price zone, btc gate, etc): fast retry
LOOP_INTERVAL_RESET  = 0.25   # recalibrate (phantom, disagree)

# ─── PRECISION ENTRY ZONE ────────────────────────────────────────────────────
# The entire edge lives at 0.35-0.57.  At 0.65+, the market already agrees
# with you — there's no edge left, payout barely covers a loss.
#
#   price 0.40 → need only 40% wins to break even, payout 150%
#   price 0.55 → need 55% wins, payout 82%
#   price 0.68 → need 68% wins (our real rate ~65%) → LOSING TRADE
#
MIN_TOKEN_PRICE      = 0.35    # never buy above 57-cent certainty
MAX_TOKEN_PRICE      = 0.57    # market must still be uncertain

# ─── WINDOW ENTRY CUTOFF ─────────────────────────────────────────────────────
# Only enter in the FIRST 120 seconds of a window.  After that the market has
# already discovered direction and priced it in — no edge remains.
WINDOW_ENTRY_CUTOFF  = 120     # seconds from window open

# ─── BTC DISTANCE GATE ───────────────────────────────────────────────────────
# BTC must be slightly off-strike in our direction (signal confirmed) but NOT
# already so far that the market has priced it in.
#
#   < 0.01% from strike → too noisy, coin-flip
#   0.01–0.20%          → sweet spot: moving but not obvious yet
#   > 0.20%             → market already knows, token price will be outside zone
#
MIN_BTC_DIFF_PCT     = 0.010   # BTC must be ≥ 0.01% from strike our direction
MAX_BTC_DIFF_PCT     = 0.200   # BTC must be ≤ 0.20% from strike

# ─── OFI CONFIRMATION ────────────────────────────────────────────────────────
# OFI must ACTIVELY confirm direction.  0.00 = no signal = skip.
OFI_MIN_WEIGHT       = 0.30    # OFI spread_weight must be ≥ 0.30 to count
OFI_MIN_CONFIRM      = 0.05    # |ofi_signal| must be ≥ 0.05 in correct direction

# ─── EMERGENCY EXIT THRESHOLDS (hold-to-expiry mode) ────────────────────────
# The only sell trigger is catastrophic adverse selection.
# ALL conditions must be true simultaneously to trigger.
EMERGENCY_BTC_WRONG_PCT  = 0.12   # BTC must be ≥ 0.12% on the wrong side of strike
EMERGENCY_SECS_REMAIN    = 90     # must still have >90s left (otherwise hold anyway)
EMERGENCY_OFI_THRESHOLD  = 0.80   # |OFI| must be strongly against us
EMERGENCY_WIN_PROB_MAX   = 0.22   # re-evaluated P(win) must be < 0.22

# ─── RESEARCH CONSTANTS ──────────────────────────────────────────────────────

KELLY_MC_SAMPLES     = 300     # reduced from 5000 — 300 draws is fast & sufficient
KELLY_CONFIDENCE_PCT = 25      # 25th percentile (conservative)
MAX_KELLY_FRACTION   = 0.25    # 25% of bankroll cap per trade
MIN_KELLY_TRADE_USD  = 1.00    # Polymarket $1 minimum
MC_EQUITY_PATHS      = 300     # reduced from 2000 for speed
MAX_RUIN_PROB        = 0.20
OFI_WEIGHT           = 0.15
CALIBRATION_GAMMA    = 1.08
REGIME_VOL_SCALE     = {"trending": 1.0, "mean_reverting": 1.15,
                        "volatile": 0.70, "neutral": 1.0, "unknown": 0.85}
SELL_RETRY_MAX       = 2
SELL_RETRY_DELAY     = 2.0

# ─── APIs ────────────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
CLOB_WS   = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


###############################################################################
# 1. EMPIRICAL ENGINE — Walk-Forward + Regime-Stratified + Calibration Tracking
###############################################################################
#
# Three validation pillars:
#   1. WALK-FORWARD: Train on first 70% of windows, hold out last 30% for OOS
#      validation. Surface is built from train only — no data leakage.
#   2. REGIME-STRATIFIED: Bins by (pct_diff, time_remaining, regime). When regime
#      bin is sparse, falls back to pooled (pct, t). Regime from RegimeClassifier.
#   3. CALIBRATION TRACKING: EmpiricalCalibrator records (predicted_prob, outcome)
#      at runtime. Brier score + reliability buckets validate forecast accuracy.
#

EMPIRICAL_TRAIN_RATIO   = 0.70   # first 70% for training
EMPIRICAL_MIN_REGIME   = 40     # min samples for regime bin (was 15 — too noisy)
EMPIRICAL_RELIABILITY_BUCKETS = 10
# Only trade when model is confident. Mud zone 0.38–0.62 = no edge.
EMPIRICAL_MIN_CONVICTION = 0.62  # prob must be ≥ 0.62 (UP) or ≤ 0.38 (DOWN)
EMPIRICAL_OOS_MIN_WR    = 0.58   # regime used only if OOS confident win rate ≥ 58%
# Map classifier outputs to 3 predictive regimes (volatile+unknown → uncertain)
REGIME_MAP = {"trending": "trending", "mean_reverting": "mean_rev",
              "volatile": "uncertain", "neutral": "neutral",
              "unknown": "uncertain"}
EMPIRICAL_REGIMES = ("trending", "mean_rev", "neutral", "uncertain")


class EmpiricalCalibrator:
    """Tracks (predicted_prob, outcome) to validate empirical forecast accuracy.

    Brier score: (1/n) Σ (pred - actual)²  — lower is better, 0.25 = no skill
    Reliability: bucket predictions [0,0.1), [0.1,0.2), ..., check mean pred vs
                 mean actual per bucket. Well-calibrated → diagonal.

    Step 1 OOS extraction: record_full() stores (P_model, P_market, direction, outcome)
    for each signal so we can analyze edge vs market-implied price.
    """
    def __init__(self, n_buckets: int = EMPIRICAL_RELIABILITY_BUCKETS):
        self._records: list[tuple[float, int]] = []  # (prob, 0|1)
        self._full_records: list[dict] = []  # {p_model, p_market, direction, outcome}
        self._n_buckets = n_buckets

    def record(self, predicted_prob: float, won: bool):
        p = max(0.01, min(0.99, predicted_prob))
        self._records.append((p, 1 if won else 0))

    def record_full(self, p_model: float, p_market: float | None, direction: str,
                   outcome: int):
        """Store full OOS signal for Step 1 extraction: P_model, P_market, Direction, Outcome."""
        p = max(0.01, min(0.99, p_model))
        self._records.append((p, outcome))
        self._full_records.append({
            "p_model": round(p, 4),
            "p_market": round(p_market, 4) if p_market is not None else None,
            "direction": direction,
            "outcome": outcome,
        })

    def get_oos_records(self) -> list[dict]:
        """Return list of OOS confident signals: {p_model, p_market, direction, outcome}."""
        return list(self._full_records)

    def brier(self) -> float:
        if not self._records:
            return float("nan")
        n = len(self._records)
        return sum((p - y) ** 2 for p, y in self._records) / n

    def reliability(self) -> dict:
        """Reliability diagram: per-bucket mean predicted vs mean actual."""
        if not self._records:
            return {"buckets": [], "brier": float("nan"), "n": 0}
        lo, hi = 0.0, 1.0
        bucket_width = (hi - lo) / self._n_buckets
        buckets = []
        for i in range(self._n_buckets):
            b_lo = lo + i * bucket_width
            b_hi = lo + (i + 1) * bucket_width
            pts = [(p, y) for p, y in self._records if b_lo <= p < b_hi]
            if not pts:
                buckets.append({"bin": i, "mid": (b_lo + b_hi) / 2,
                               "mean_pred": (b_lo + b_hi) / 2,
                               "mean_actual": float("nan"),
                               "count": 0})
            else:
                mean_p = sum(p for p, _ in pts) / len(pts)
                mean_a = sum(y for _, y in pts) / len(pts)
                buckets.append({"bin": i, "mid": (b_lo + b_hi) / 2,
                               "mean_pred": round(mean_p, 4),
                               "mean_actual": round(mean_a, 4),
                               "count": len(pts)})
        return {
            "buckets": buckets,
            "brier": round(self.brier(), 4),
            "n": len(self._records),
        }

    def summary_str(self) -> str:
        r = self.reliability()
        s = f"Brier={r['brier']:.4f} n={r['n']}"
        if r["buckets"]:
            cal_errs = [abs(b["mean_pred"] - b["mean_actual"])
                        for b in r["buckets"] if b["count"] > 0 and b["mean_actual"] == b["mean_actual"]]
            if cal_errs:
                s += f" mean_cal_err={sum(cal_errs)/len(cal_errs):.3f}"
        return s


def _classify_regime_from_prices(prices: list) -> str:
    """Classify regime from a list of prices. Uses RegimeClassifier logic."""
    if len(prices) < RegimeClassifier.MIN_RETURNS + 1:
        return "unknown"
    regime, _ = RegimeClassifier.classify(prices)
    return regime


class EmpiricalEngine:
    """
    Walk-forward, regime-stratified empirical probability surface.

    - TRAIN: first 70% of windows (chronological)
    - TEST: last 30% — used only for OOS validation at load time
    - Surface key: (pct_bin, t_bin, regime) with fallback to (pct_bin, t_bin)
    """
    def __init__(self):
        self.prob_surface  = {}   # (p_bin, t_bin) or (p_bin, t_bin, regime) -> {up, dn}
        self.pooled_surface = {}   # (p_bin, t_bin) -> {up, dn} for fallback
        self.dist_surface  = {}   # (p_bin, t_bin) -> list of final-pct moves
        self._pct_bins     = []
        self._time_bins    = [60, 120, 180, 240, 300]
        self._loaded       = False
        self._oos_metrics  = {}  # Brier, win_rate, n_test from held-out set
        self._train_windows = 0
        self._test_windows  = 0

    def load(self, path="btc_1m_candles.pkl") -> bool:
        if not os.path.exists(path):
            print(f"  [!] No training data at {path}")
            return False
        try:
            with open(path, "rb") as f:
                candles = pickle.load(f)
            self._build(candles)
            self._loaded = True
            print(f"  [OK] Empirical engine: {len(candles):,} candles "
                  f"(train {self._train_windows}w, test {self._test_windows}w)")
            if self._oos_metrics:
                o = self._oos_metrics
                b = o.get("brier", float("nan"))
                wr = o.get("wr_confident", 0.5)
                nc = o.get("n_confident", 0)
                vr = o.get("valid_regimes", [])
                print(f"      OOS: Brier={b:.4f} | confident WR={wr:.1%} (n={nc}) "
                      f"| regimes: {vr or 'pooled'}")
            return True
        except Exception as e:
            print(f"  [!] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _build(self, candles):
        if candles and isinstance(candles[0], (list, tuple)):
            closes = [float(c[4]) for c in candles]
        else:
            closes = [float(c["close"]) for c in candles]

        n_windows = max(0, (len(closes) - 5))
        train_end = int(n_windows * EMPIRICAL_TRAIN_RATIO)
        train_end = max(train_end, 1)

        pct_step = 0.02
        pct_range = 2.0
        self._pct_bins = [round(p * pct_step - pct_range, 4)
                          for p in range(int(pct_range * 2 / pct_step) + 1)]

        # Build from TRAIN windows only. Recency weight: last 25% of train = 1.0, older = 0.5
        recency_cut = train_end * 0.75
        for w in range(train_end):
            start = w
            if start + 5 >= len(closes):
                break
            w_recency = 1.0 if w >= recency_cut else 0.5
            strike = closes[start]
            final_close = closes[start + 5]
            resolved_up = final_close >= strike
            path = closes[start : start + 6]

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
                key = (p_bin, t_bin)

                # Regime: use history of closes ending at current minute
                hist_start = max(0, start + minute - 120)
                hist = closes[hist_start : start + minute + 1]
                regime_raw = _classify_regime_from_prices(hist)
                regime = REGIME_MAP.get(regime_raw, "uncertain")

                # Pooled (fallback) — use float for recency weighting
                if key not in self.pooled_surface:
                    self.pooled_surface[key] = {"up": 0.0, "dn": 0.0}
                if resolved_up:
                    self.pooled_surface[key]["up"] += w_recency
                else:
                    self.pooled_surface[key]["dn"] += w_recency

                # Regime-stratified
                key_r = (p_bin, t_bin, regime)
                if key_r not in self.prob_surface:
                    self.prob_surface[key_r] = {"up": 0.0, "dn": 0.0}
                if resolved_up:
                    self.prob_surface[key_r]["up"] += w_recency
                else:
                    self.prob_surface[key_r]["dn"] += w_recency

                # Distribution (pooled)
                final_pct = (final_close - strike) / strike * 100
                if key not in self.dist_surface:
                    self.dist_surface[key] = []
                self.dist_surface[key].append(final_pct)

        self._train_windows = train_end

        # OOS validation on TEST windows (same overlapping structure)
        oos_preds = []   # (prob, outcome, regime)
        for w in range(train_end, n_windows):
            start = w
            if start + 5 >= len(closes):
                break
            strike = closes[start]
            final_close = closes[start + 5]
            resolved_up = final_close >= strike

            for minute in range(1, 5):  # skip last minute (trivial)
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
                regime_raw = _classify_regime_from_prices(hist)
                regime = REGIME_MAP.get(regime_raw, "uncertain")

                prob_up, _ = self._lookup_raw(pct, t_rem, regime)
                oos_preds.append((prob_up, 1 if resolved_up else 0, regime))

        self._test_windows = n_windows - train_end

        # OOS: conditional win rate (confident predictions only)
        self._valid_regimes = set()
        if oos_preds:
            n = len(oos_preds)
            brier = sum((p - y) ** 2 for p, y, _ in oos_preds) / n

            conf_up = [(p, y) for p, y, _ in oos_preds if p >= EMPIRICAL_MIN_CONVICTION]
            conf_dn = [(p, y) for p, y, _ in oos_preds if p <= (1 - EMPIRICAL_MIN_CONVICTION)]
            wr_up = sum(y for _, y in conf_up) / len(conf_up) if conf_up else 0.5
            wr_dn = sum(1 - y for _, y in conf_dn) / len(conf_dn) if conf_dn else 0.5
            n_conf = len(conf_up) + len(conf_dn)
            wr_conf = ((sum(y for _, y in conf_up) + sum(1 - y for _, y in conf_dn))
                       / n_conf) if n_conf else 0.5

            # Per-regime OOS: only use regime if its confident win rate ≥ 58%
            for r in EMPIRICAL_REGIMES:
                r_preds = [(p, y) for p, y, rg in oos_preds if rg == r]
                r_conf = [(p, y) for p, y in r_preds
                          if p >= EMPIRICAL_MIN_CONVICTION or p <= (1 - EMPIRICAL_MIN_CONVICTION)]
                if len(r_conf) >= 30:
                    r_wins = sum(y for p, y in r_conf if p >= 0.5) + sum(1 - y for p, y in r_conf if p < 0.5)
                    r_wr = r_wins / len(r_conf)
                    if r_wr >= EMPIRICAL_OOS_MIN_WR:
                        self._valid_regimes.add(r)
            if not self._valid_regimes:
                self._valid_regimes = set()  # use pooled only when no regime validates

            self._oos_metrics = {
                "brier": round(brier, 4),
                "n_test": n,
                "n_confident": n_conf,
                "wr_confident": round(wr_conf, 3),
                "wr_up": round(wr_up, 3),
                "wr_dn": round(wr_dn, 3),
                "n_up": len(conf_up),
                "n_dn": len(conf_dn),
                "valid_regimes": list(self._valid_regimes),
            }

            # Step 1 OOS extraction: store (P_model, P_market, direction, outcome)
            # Candle-based has no historical order book → p_market = None
            self._oos_confident_signals = []
            for p, y in conf_up:
                self._oos_confident_signals.append({
                    "p_model": round(p, 4),
                    "p_market": None,  # not available in candle-only backtest
                    "direction": "UP",
                    "outcome": y,
                })
            for p, y in conf_dn:
                self._oos_confident_signals.append({
                    "p_model": round(1 - p, 4),  # P(DOWN) for DOWN signals
                    "p_market": None,
                    "direction": "DOWN",
                    "outcome": 1 - y,  # outcome for DOWN: 1 if BTC went down
                })
        else:
            self._oos_confident_signals = []

    def get_oos_confident_signals(self) -> list[dict]:
        """Return OOS confident signals: [{p_model, p_market, direction, outcome}, ...].
        p_market is None for candle-based training (no historical order book)."""
        return getattr(self, "_oos_confident_signals", [])

    def _snap_pct(self, pct: float) -> float:
        if not self._pct_bins:
            return round(pct, 2)
        i = bisect_left(self._pct_bins, pct)
        if i == 0:
            return self._pct_bins[0]
        if i >= len(self._pct_bins):
            return self._pct_bins[-1]
        lo, hi = self._pct_bins[i - 1], self._pct_bins[i]
        return lo if abs(pct - lo) <= abs(pct - hi) else hi

    def _lookup_raw(self, pct_diff: float, time_remaining: float,
                    regime: str | None = None) -> tuple[float, int]:
        """Internal: (prob_up, sample_count). Only use regime if OOS-validated."""
        t_lo = max(60, (int(time_remaining / 60)) * 60)
        t_hi = min(300, t_lo + 60)
        p_bin = self._snap_pct(pct_diff)

        def get(d: dict, k) -> tuple[float | None, int]:
            v = d.get(k)
            if v is None:
                return None, 0
            tot = v["up"] + v["dn"]
            if tot == 0:
                return None, 0
            return v["up"] / tot, tot

        # Try regime only if OOS-validated
        if regime and getattr(self, "_valid_regimes", set()) and regime in self._valid_regimes:
            key_r = (p_bin, t_lo, regime)
            prob_lo, n_lo = get(self.prob_surface, key_r)
            if n_lo >= EMPIRICAL_MIN_REGIME and prob_lo is not None:
                if t_hi != t_lo:
                    prob_hi, n_hi = get(self.prob_surface, (p_bin, t_hi, regime))
                    if prob_hi is not None and n_hi >= EMPIRICAL_MIN_REGIME:
                        alpha = (time_remaining - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0
                        return (1 - alpha) * prob_lo + alpha * prob_hi, min(n_lo, n_hi)
                return prob_lo, n_lo

        # Fallback: pooled (no regime)
        prob_lo, n_lo = get(self.pooled_surface, (p_bin, t_lo))
        prob_hi, n_hi = get(self.pooled_surface, (p_bin, t_hi))

        if prob_lo is None and prob_hi is None:
            return 0.5, 0
        if prob_lo is None:
            return prob_hi, n_hi
        if prob_hi is None:
            return prob_lo, n_lo

        alpha = (time_remaining - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0
        return (1 - alpha) * prob_lo + alpha * prob_hi, min(n_lo, n_hi)

    def lookup(self, pct_diff: float, time_remaining: float,
               regime: str | None = None) -> tuple[float, int]:
        """Returns (prob_up, sample_count). Mud zone 0.38–0.62 → 0.5 (no edge)."""
        prob, n = self._lookup_raw(pct_diff, time_remaining, regime)
        # No trade in mud zone — return 0.5 so edge = 0
        if prob is not None and (1 - EMPIRICAL_MIN_CONVICTION) < prob < EMPIRICAL_MIN_CONVICTION:
            return 0.5, n
        return prob if prob is not None else 0.5, n

    # ── FULL DISTRIBUTION STATS (#6) ──
    def get_distribution(self, pct_diff: float, time_remaining: float) -> Optional[dict]:
        """Full outcome distribution: mean, variance, skew, confidence intervals."""
        t_lo = max(60, (int(time_remaining / 60)) * 60)
        p_bin = self._snap_pct(pct_diff)
        key = (p_bin, t_lo)

        outcomes = self.dist_surface.get(key, [])
        if len(outcomes) < 10:
            return None

        n = len(outcomes)
        mean = sum(outcomes) / n
        var  = sum((x - mean) ** 2 for x in outcomes) / n
        std  = var ** 0.5

        skew = 0.0
        if std > 1e-10:
            skew = sum(((x - mean) / std) ** 3 for x in outcomes) / n

        se = std / (n ** 0.5)
        sorted_o = sorted(outcomes)
        p_up = sum(1 for x in outcomes if x > 0) / n

        return {
            "mean": mean, "std": std, "var": var, "skew": skew, "n": n,
            "ci_lower": mean - 1.96 * se,
            "ci_upper": mean + 1.96 * se,
            "p_up": p_up,
            "p5":  sorted_o[max(0, int(n * 0.05))],
            "p95": sorted_o[min(n - 1, int(n * 0.95))],
        }


###############################################################################
# 2. CALIBRATION CURVE — corrects market price → true probability (#2)
###############################################################################

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

    FOLD_SIZE  = 10   # new observations per fold
    MIN_TRAIN  = 20   # minimum training points before first OOS update
    GAMMA_GRID = [g / 100.0 for g in range(90, 131)]

    def __init__(self, gamma: float = CALIBRATION_GAMMA):
        self._gamma   = gamma           # currently applied (OOS-validated) γ
        self._all: list = []            # full chronological buffer
        self._oos_log: list = []        # [(fold_id, train_n, val_ll, gamma)]

    # ── public ────────────────────────────────────────────────────────────

    def calibrate(self, market_price: float) -> float:
        p = max(0.01, min(0.99, market_price))
        g = self._gamma
        pp = p ** (1.0 / g)
        qp = (1.0 - p) ** (1.0 / g)
        d  = pp + qp
        return pp / d if d > 1e-10 else p

    def record_outcome(self, market_price: float, won: bool):
        self._all.append((market_price, 1 if won else 0))
        n = len(self._all)
        # trigger OOS refitting at each fold boundary
        if n >= self.MIN_TRAIN + self.FOLD_SIZE and n % self.FOLD_SIZE == 0:
            self._rolling_oos_fit()

    def oos_summary(self) -> dict:
        if not self._oos_log:
            return {"folds": 0, "gamma": self._gamma}
        avg_val_ll = sum(r[2] for r in self._oos_log) / len(self._oos_log)
        return {
            "folds":      len(self._oos_log),
            "gamma":      round(self._gamma, 3),
            "avg_val_ll": round(avg_val_ll, 4),
            "last_train_n": self._oos_log[-1][1],
        }

    # ── internals ─────────────────────────────────────────────────────────

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
        n         = len(self._all)
        val_data  = self._all[-self.FOLD_SIZE:]    # current fold = OOS
        train_data = self._all[:-self.FOLD_SIZE]   # everything before = train

        if len(train_data) < self.MIN_TRAIN:
            return

        # fit γ on train only
        best_g, best_train_ll = self._gamma, float("-inf")
        for g in self.GAMMA_GRID:
            tll = self._ll(train_data, g)
            if tll > best_train_ll:
                best_train_ll = tll
                best_g = g

        # evaluate on held-out val (OOS)
        val_ll = self._ll(val_data, best_g)

        fold_id = len(self._oos_log) + 1
        self._oos_log.append((fold_id, len(train_data), round(val_ll, 4), best_g))
        self._gamma = best_g


###############################################################################
# 3. ORDER FLOW IMBALANCE (OFI) — trade confirmation signal (#4)
###############################################################################

class OFITracker:
    """True OFI (Cont-Kukanov-Stoikov) with per-increment Z-score normalisation
    and spread-regime gating.

    Fix 1 — normalisation:
        OLD: z = Σ(ofi_i) / (std(ofi) * √n)
             Problem: if OFI increments are autocorrelated the denominator
             understates the true SE, inflating the signal.
        NEW: z_i = (ofi_i − μ_rolling) / σ_rolling   per increment
             signal = tanh( Σ(z_i[-W:]) / √W )
             Each increment is standardised against its own recent distribution
             before accumulation, so autocorrelation does not inflate the result.

    Fix 2 — spread-regime gate:
        OFI edge is only meaningful when the spread is wide enough that
        market makers are pricing-in uncertainty.  When spreads compress
        (consensus on fair value) the OFI edge approaches zero.

        spread_weight = clip( (spread − S_min) / (S_ref − S_min), 0, 1 )

        Final signal = raw_signal × spread_weight
        → signal collapses to 0 as spread → S_min
        → signal is fully weighted once spread ≥ S_ref
    """

    # Spread gate thresholds (fraction of price, e.g. 0.02 = 2 cents on $1 token)
    SPREAD_MIN = 0.01   # below this the OFI edge is gone
    SPREAD_REF = 0.04   # at/above this the OFI edge is fully on

    def __init__(self, window: int = 60, z_warmup: int = 10):
        self._prev_book: Optional[dict] = None
        self._raw_ofi: deque = deque(maxlen=window)   # raw OFI increments
        self._z_scores: deque = deque(maxlen=window)  # per-increment Z-scores
        self._z_warmup = z_warmup                     # ticks before Z-scoring starts
        self._last_spread: float = 0.0

    def update(self, book: dict):
        if self._prev_book is None:
            self._prev_book = dict(book)
            self._last_spread = book.get("spread", 0.0)
            return

        # ── True OFI increment (Cont-Kukanov-Stoikov) ──────────────────
        bid_d = book.get("tob_bid_vol", 0) - self._prev_book.get("tob_bid_vol", 0)
        ask_d = book.get("tob_ask_vol", 0) - self._prev_book.get("tob_ask_vol", 0)
        ofi   = bid_d - ask_d

        # price-level crossing contributions
        if book.get("best_bid", 0) > self._prev_book.get("best_bid", 0):
            ofi += book.get("tob_bid_vol", 0)
        if book.get("best_ask", 0) < self._prev_book.get("best_ask", 0):
            ofi -= book.get("tob_ask_vol", 0)

        self._raw_ofi.append(ofi)
        self._last_spread = book.get("spread", 0.0)

        # ── Per-increment Z-score ──────────────────────────────────────
        n = len(self._raw_ofi)
        if n >= self._z_warmup:
            hist = list(self._raw_ofi)[:-1]        # exclude current point
            mu   = sum(hist) / len(hist)
            var  = sum((v - mu) ** 2 for v in hist) / len(hist)
            sig  = var ** 0.5
            z_i  = (ofi - mu) / sig if sig > 1e-8 else 0.0
        else:
            z_i  = 0.0

        self._z_scores.append(z_i)
        self._prev_book = dict(book)

    def signal(self) -> float:
        """Spread-gated cumulative Z-score signal in [-1, +1].

        Steps:
          1. Accumulate per-increment Z-scores (autocorrelation-robust).
          2. Apply tanh to bound the signal.
          3. Multiply by spread_weight → 0 when spread is compressed.
        """
        if len(self._z_scores) < 3:
            return 0.0

        # 1. Cumulative Z-score (mean-reverting walk in Z-space)
        zs = list(self._z_scores)
        n  = len(zs)
        cum_z = sum(zs) / (n ** 0.5)      # scaled by √n to keep it O(1)

        # 2. Bound with tanh (softer than hard clip, preserves gradient near 0)
        raw_signal = math.tanh(cum_z / 3.0)

        # 3. Spread-regime gate
        spread = self._last_spread
        if spread <= self.SPREAD_MIN:
            return 0.0
        spread_weight = min(1.0, (spread - self.SPREAD_MIN)
                           / (self.SPREAD_REF - self.SPREAD_MIN))

        return raw_signal * spread_weight

    def spread_weight(self) -> float:
        """Expose the current spread gate weight for display."""
        s = self._last_spread
        if s <= self.SPREAD_MIN:
            return 0.0
        return min(1.0, (s - self.SPREAD_MIN) / (self.SPREAD_REF - self.SPREAD_MIN))

    def reset(self):
        self._prev_book = None
        self._raw_ofi.clear()
        self._z_scores.clear()
        self._last_spread = 0.0


###############################################################################
# 4. REGIME CLASSIFIER — adapts sizing to market environment (#5)
###############################################################################

class RegimeClassifier:
    """EWMA volatility + directional momentum regime classifier.

    Replaces Hurst exponent (R/S) which requires 256+ samples for reliable
    estimation and has upward bias on short windows.  Both new signals are
    reliable from as few as 10 ticks.

    Signal 1 — EWMA Realized Volatility (RiskMetrics, λ=0.94)
        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
        Responds within ~10 ticks; compared to a rolling percentile to
        determine whether current vol is LOW / MEDIUM / HIGH.

    Signal 2 — Directional Momentum Score
        M = Σ sign(r_i) / n  ∈ [-1, +1]
        "Trending" only declared when |M| exceeds the 95% binomial noise
        band: 2/√n.  Below that threshold it is statistically indistinguishable
        from a coin-flip — no trend label is assigned.

    Regime → Kelly scale:
        trending      ×1.00  (momentum strategy has edge)
        mean_reverting×1.15  (binary resolution favours reversion)
        volatile      ×0.70  (wider tails → reduce exposure)
        neutral       ×1.00
        unknown       ×0.85  (insufficient data → caution)
    """

    EWMA_LAMBDA    = 0.94      # RiskMetrics decay factor
    VOL_HIGH_MULT  = 1.8       # σ_current > 1.8 × σ_baseline → high vol
    VOL_LOW_MULT   = 0.6       # σ_current < 0.6 × σ_baseline → low vol
    MIN_RETURNS    = 10

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

        # Simple baseline: equal-weight std over the same window
        mean_r   = sum(rets) / n
        var_base = sum((r - mean_r) ** 2 for r in rets) / n
        vol_base = var_base ** 0.5

        vol_ratio = (vol_ewma / vol_base) if vol_base > 1e-12 else 1.0

        # ── Signal 2: Directional Momentum Score ────────────────────────
        momentum = sum(1.0 if r > 0 else -1.0 for r in rets) / n
        # 95% binomial noise band: reject H₀: M=0 if |M| > 2/√n
        noise_band = 2.0 / (n ** 0.5)
        trending   = abs(momentum) > noise_band

        # ── Classify ────────────────────────────────────────────────────
        stats = {
            "vol_ewma":   round(vol_ewma, 7),
            "vol_base":   round(vol_base, 7),
            "vol_ratio":  round(vol_ratio, 3),
            "momentum":   round(momentum, 3),
            "noise_band": round(noise_band, 3),
            "n":          n,
        }

        # High volatility overrides everything — risk first
        if vol_ratio > RegimeClassifier.VOL_HIGH_MULT:
            return "volatile", stats

        # Clear directional momentum
        if trending:
            return "trending", stats

        # Low-vol, no trend → classic mean-reversion territory for binaries
        if vol_ratio < RegimeClassifier.VOL_LOW_MULT:
            return "mean_reverting", stats

        return "neutral", stats


###############################################################################
# 5. MONTE CARLO KELLY — uncertainty-adjusted position sizing (#1)
###############################################################################

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

    Final size = Quantile_{25}(f_s) × bankroll, capped at MAX_SINGLE_TRADE_USD.
    """

    @staticmethod
    def compute(
        prob: float,
        price: float,
        n_samples: int,
        bankroll: float,
        fee: float = 0.02,
        execution_type: str = "TAKER",
        slippage_mu: float = None,      # defaults to TAKE_PROFIT_PCT*0.015 for taker, 0 for maker
        slippage_sigma: float = None,   # defaults to 30% of slippage_mu
        maker_fill_rate: float = 0.40,
        maker_fill_n: int = 50,         # prior strength for fill-rate posterior
        n_mc: int = KELLY_MC_SAMPLES,
    ) -> tuple:
        """Returns (trade_size_usd, detail_dict)."""
        if price <= 0.01 or price >= 0.99:
            return 0.0, {"reason": "price_extreme"}

        # ── Slippage defaults ──────────────────────────────────────────────
        if slippage_mu is None:
            slippage_mu = 0.015 if execution_type == "TAKER" else 0.003
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
            slip_s          = max(0.0, random.gauss(slippage_mu, slippage_sigma))
            eff_price_s     = min(0.98, price * (1.0 + slip_s))
            R_s             = (1.0 - fee) / eff_price_s - 1.0
            if R_s <= 0.0:
                fractions.append(0.0)
                continue

            # 3. Kelly fraction for this (p_s, R_s) draw
            q_s = 1.0 - p_s
            f_s = (p_s * R_s - q_s) / R_s
            f_s = max(0.0, f_s)

            # 4. For MAKER: scale by sampled fill probability
            #    Derivation: 3-outcome Kelly (win/lose/no-fill) gives f* independent
            #    of fill_rate in expectation, but fill-rate uncertainty inflates
            #    cross-path variance.  Multiplying by fill_s ensures the 25th-pctile
            #    cut penalises adverse (low fill, bad luck) paths jointly.
            if execution_type == "MAKER":
                fill_s = random.betavariate(fill_alpha, fill_beta)
                f_s   = f_s * fill_s

            fractions.append(f_s)

        fractions.sort()
        idx           = max(0, int(len(fractions) * KELLY_CONFIDENCE_PCT / 100))
        f_conservative = fractions[idx]
        f_median       = fractions[len(fractions) // 2]
        f_capped       = min(f_conservative, MAX_KELLY_FRACTION)

        trade_usd = min(f_capped * bankroll, MAX_SINGLE_TRADE_USD)
        trade_usd = round(trade_usd, 2)

        detail = {
            "kelly_25pct":  round(f_conservative, 4),
            "kelly_median": round(f_median, 4),
            "kelly_capped": round(f_capped, 4),
            "alpha":        round(alpha, 1),
            "beta":         round(beta_p, 1),
            "slip_mu":      round(slippage_mu, 4),
            "exec_type":    execution_type,
            "trade_usd":    trade_usd,
        }
        return trade_usd, detail


###############################################################################
# 6. MONTE CARLO EQUITY SIMULATION — forward-looking risk (#3)
###############################################################################

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
      dd_mean             mean max drawdown
      dd_p95              95th pct max drawdown  (worst 5% of paths)
      dd_p99              99th pct max drawdown
      streak_mean         mean longest loss streak
      streak_p95          95th pct longest loss streak
      streak_dist         {0:n, 1:n, ...} loss-streak frequency distribution
      tuw_mean            mean time-under-water (trades)
      tuw_p95             95th pct time-under-water
    """

    @staticmethod
    def simulate(bankroll: float, n_trades: int, prob: float, price: float,
                 kelly_frac: float, n_paths: int = MC_EQUITY_PATHS) -> dict:
        if n_trades <= 0 or bankroll < MIN_KELLY_TRADE_USD:
            return {"p_ruin": 0, "median_final": bankroll}

        finals      = []
        max_dds     = []   # dollar max drawdown per path
        max_dd_pcts = []   # % max drawdown per path
        streaks     = []   # longest loss streak per path
        tuws        = []   # time under water per path
        streak_counts: dict = {}

        for _ in range(n_paths):
            eq         = bankroll
            peak       = bankroll       # high-water mark
            max_dd     = 0.0
            cur_streak = 0
            max_streak = 0
            tuw        = 0             # trades where eq < bankroll

            for _ in range(n_trades):
                if eq < MIN_KELLY_TRADE_USD:
                    break
                bet = min(eq * kelly_frac, MAX_SINGLE_TRADE_USD)

                if random.random() < prob:
                    eq      += bet / price - bet
                    peak     = max(peak, eq)
                    cur_streak = 0
                else:
                    eq        -= bet
                    cur_streak += 1
                    max_streak  = max(max_streak, cur_streak)

                # drawdown from current peak
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd

                # time under water
                if eq < bankroll:
                    tuw += 1

            finals.append(eq)
            max_dds.append(max_dd)
            max_dd_pcts.append(max_dd / bankroll if bankroll > 0 else 0.0)
            streaks.append(max_streak)
            tuws.append(tuw)

            # streak frequency
            streak_counts[max_streak] = streak_counts.get(max_streak, 0) + 1

        # ── aggregate ────────────────────────────────────────────────────
        n = len(finals)

        def pct(lst, p):
            s = sorted(lst)
            return s[min(len(s) - 1, max(0, int(len(s) * p)))]

        finals.sort()
        max_dds.sort()
        max_dd_pcts.sort()
        streaks_s = sorted(streaks)
        tuws_s    = sorted(tuws)

        return {
            # final equity
            "p_ruin":        sum(1 for x in finals if x < MIN_KELLY_TRADE_USD) / n,
            "median_final":  finals[n // 2],
            "ci_5":          finals[max(0, int(n * 0.05))],
            "ci_95":         finals[min(n - 1, int(n * 0.95))],
            "e_return":      sum(finals) / n - bankroll,
            # drawdown
            "dd_mean":       sum(max_dds) / n,
            "dd_p95":        max_dds[min(n - 1, int(n * 0.95))],
            "dd_p99":        max_dds[min(n - 1, int(n * 0.99))],
            "dd_pct_mean":   sum(max_dd_pcts) / n,
            "dd_pct_p95":    max_dd_pcts[min(n - 1, int(n * 0.95))],
            # loss streaks
            "streak_mean":   sum(streaks) / n,
            "streak_p95":    streaks_s[min(n - 1, int(n * 0.95))],
            "streak_dist":   streak_counts,
            # time under water
            "tuw_mean":      sum(tuws) / n,
            "tuw_p95":       tuws_s[min(n - 1, int(n * 0.95))],
            "tuw_pct_mean":  sum(tuws) / n / max(n_trades, 1),
        }


###############################################################################
# SESSION STATE
###############################################################################

@dataclass
class SessionState:
    trades_placed:     int   = 0
    total_spent_usd:   float = 0.0
    total_received_usd: float = 0.0
    session_start:     float = 0.0
    halted:            bool  = False
    halt_reason:       str   = ""

    def can_trade(self, size_usd: float) -> tuple[bool, str]:
        if self.halted:
            return False, f"HALTED: {self.halt_reason}"
        if self.trades_placed >= MAX_TRADES_SESSION:
            return False, f"Max trades reached ({MAX_TRADES_SESSION})"
        if self.total_spent_usd + size_usd > MAX_SESSION_SPEND_USD:
            return False, (f"Session limit: ${self.total_spent_usd:.2f} + "
                           f"${size_usd:.2f} > ${MAX_SESSION_SPEND_USD:.2f}")
        return True, "OK"

    def record(self, size_usd: float):
        self.trades_placed += 1
        self.total_spent_usd += size_usd

    def record_sell(self, received_usd: float):
        self.total_received_usd += received_usd

    @property
    def bankroll(self) -> float:
        return max(0.0, MAX_SESSION_SPEND_USD - self.total_spent_usd
                   + self.total_received_usd)

    def halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason
        print(f"\n  🛑 SESSION HALTED: {reason}")


###############################################################################
# MARKET / BOOK HELPERS
###############################################################################

def _map_tokens_from_outcomes(outcomes_raw: str, tokens: list) -> tuple[str, str]:
    """Map tokens to UP/DOWN using API outcomes. Tokens and outcomes share index.
    Returns (token_up, token_down). For btc-updown: outcomes are ['Up','Down'].
    Default: tokens[0]=UP, tokens[1]=DOWN if outcomes unknown."""
    up_idx, down_idx = 0, 1
    try:
        if isinstance(outcomes_raw, str) and outcomes_raw.strip():
            raw = outcomes_raw.strip().strip("[]").replace('"', "'")
            parts = [p.strip().strip("'\"") for p in raw.split(",")]
            outcomes = [p.lower() for p in parts if p]
            if len(outcomes) >= 2:
                for i, o in enumerate(outcomes):
                    if o == "up":
                        up_idx = i
                        down_idx = 1 - i
                        break
                    if o == "down":
                        down_idx = i
                        up_idx = 1 - i
                        break
    except Exception:
        pass
    return tokens[up_idx], tokens[down_idx]


async def find_btc_5min_market(session: aiohttp.ClientSession) -> Optional[dict]:
    """Find the currently active BTC 5-min market on Polymarket."""
    now = int(time.time())

    for offset in range(0, 7):
        window_start = (now // 300) * 300 - (offset * 300)
        slug = f"btc-updown-5m-{window_start}"
        try:
            async with session.get(f"{GAMMA_API}/events",
                                   params={"slug": slug}) as r:
                if r.status == 200:
                    data = await r.json()
                    if data:
                        events = (data if isinstance(data, list)
                                  else data.get("events", [data]))
                        for event in events:
                            if not event:
                                continue
                            markets = event.get("markets", [])
                            if not markets:
                                continue
                            m = markets[0]
                            tokens = m.get("clobTokenIds", "")
                            if isinstance(tokens, str):
                                tokens = (tokens.strip().strip("[]")
                                          .replace('"', '').replace("'", ""))
                                tokens = [t.strip() for t in tokens.split(",")
                                          if t.strip()]
                            if len(tokens) >= 2:
                                token_up, token_down = _map_tokens_from_outcomes(
                                    m.get("outcomes", ""), tokens)
                                return {
                                    "title": event.get("title", slug),
                                    "token_up":   token_up,
                                    "token_down": token_down,
                                    "end_date":   m.get("endDate", ""),
                                    "window_start": window_start,
                                }
        except Exception:
            pass

    try:
        async with session.get(
            f"{GAMMA_API}/events",
            params={"tag": "bitcoin", "active": "true", "limit": "20"},
        ) as r:
            if r.status == 200:
                events = await r.json()
                events = (events if isinstance(events, list)
                          else events.get("events", []))
                for event in events:
                    title = (event.get("title") or "").lower()
                    if "5" in title and ("up" in title or "down" in title):
                        markets = event.get("markets", [])
                        if markets:
                            m = markets[0]
                            tokens = m.get("clobTokenIds", "")
                            if isinstance(tokens, str):
                                tokens = (tokens.strip().strip("[]")
                                          .replace('"', '').replace("'", ""))
                                tokens = [t.strip() for t in tokens.split(",")
                                          if t.strip()]
                            if len(tokens) >= 2:
                                token_up, token_down = _map_tokens_from_outcomes(
                                    m.get("outcomes", ""), tokens)
                                return {
                                    "title": event.get("title", ""),
                                    "token_up":   token_up,
                                    "token_down": token_down,
                                    "end_date":   m.get("endDate", ""),
                                    "window_start": now,
                                }
    except Exception:
        pass

    return None


async def fetch_book(session: aiohttp.ClientSession,
                     token_id: str) -> Optional[dict]:
    """Fetch order book for a token."""
    try:
        async with session.get(f"{CLOB_API}/book",
                               params={"token_id": token_id}) as r:
            if r.status == 200:
                data = await r.json()
                bids = sorted(data.get("bids", []),
                              key=lambda x: -float(x["price"]))
                asks = sorted(data.get("asks", []),
                              key=lambda x:  float(x["price"]))
                if not bids or not asks:
                    return None
                best_bid    = float(bids[0]["price"])
                best_ask    = float(asks[0]["price"])
                tob_bid_vol = float(bids[0].get("size", 0)) * best_bid
                tob_ask_vol = float(asks[0].get("size", 0)) * best_ask
                return {
                    "best_bid":    best_bid,
                    "best_ask":    best_ask,
                    "mid":         (best_bid + best_ask) / 2,
                    "spread":      best_ask - best_bid,
                    "tob_bid_vol": tob_bid_vol,
                    "tob_ask_vol": tob_ask_vol,
                }
    except Exception:
        pass
    return None


async def poly_clob_ws_feed(book_cache: dict, token_up: str, token_down: str,
                            subscribe_queue: asyncio.Queue):
    """Polymarket CLOB WebSocket: real-time order book (no REST polling)."""
    last_ping = 0.0
    current_tokens = []

    async def do_subscribe(ws, tokens, is_initial=False):
        payload = {"assets_ids": tokens, "custom_feature_enabled": True}
        if is_initial:
            payload["type"] = "market"
        else:
            payload["operation"] = "subscribe"
        await ws.send(json.dumps(payload))

    async def unsubscribe(ws, tokens):
        if tokens:
            await ws.send(json.dumps({"assets_ids": tokens, "operation": "unsubscribe"}))

    def parse_book(msg, asset_id):
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        if not bids or not asks:
            return None
        # Handle both {price, size} dicts and [price, size] lists
        def _p(x):
            return float(x.get("price", 0) if isinstance(x, dict) else (x[0] if x else 0))
        def _s(x):
            return float(x.get("size", 0) if isinstance(x, dict) else (x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else 0))
        best_bid = _p(bids[0])
        best_ask = _p(asks[0])
        sb = _s(bids[0])
        sa = _s(asks[0])
        tob_bid_vol = sb * best_bid
        tob_ask_vol = sa * best_ask
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": (best_bid + best_ask) / 2,
            "spread": best_ask - best_bid,
            "tob_bid_vol": tob_bid_vol,
            "tob_ask_vol": tob_ask_vol,
        }

    while True:
        try:
            async with websockets.connect(
                CLOB_WS, ssl=SSL_CTX,
                ping_interval=None, ping_timeout=None,
            ) as ws:
                tokens = [token_up, token_down]
                await do_subscribe(ws, tokens, is_initial=True)
                current_tokens = tokens

                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.5)
                        if not msg:
                            continue
                        # Server may send PONG (plain text) - skip non-JSON
                        try:
                            raw = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        # Server may send single object or batch array
                        items = raw if isinstance(raw, list) else [raw]
                        for data in items:
                            if not isinstance(data, dict):
                                continue
                            if data.get("event_type") == "book":
                                aid = data.get("asset_id", "")
                                book = parse_book(data, aid)
                                if book and aid:
                                    book_cache[aid] = book
                            elif data.get("event_type") == "best_bid_ask":
                                aid = data.get("asset_id", "")
                                bb = data.get("best_bid")
                                ba = data.get("best_ask")
                                if aid and bb is not None and ba is not None:
                                    prev = book_cache.get(aid, {})
                                    book_cache[aid] = {
                                        "best_bid": float(bb),
                                        "best_ask": float(ba),
                                        "mid": (float(bb) + float(ba)) / 2,
                                        "spread": float(ba) - float(bb),
                                        "tob_bid_vol": prev.get("tob_bid_vol", 0),
                                        "tob_ask_vol": prev.get("tob_ask_vol", 0),
                                    }
                    except asyncio.TimeoutError:
                        pass

                    # Check for subscription update
                    try:
                        new_tokens = subscribe_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        new_tokens = None
                    if new_tokens is not None and new_tokens != current_tokens:
                        await unsubscribe(ws, current_tokens)
                        await do_subscribe(ws, new_tokens, is_initial=False)
                        current_tokens = new_tokens

                    # Ping every 10s
                    if time.time() - last_ping > 10:
                        await ws.send("PING")
                        last_ping = time.time()

        except Exception as e:
            print(f"  Polymarket WS error: {e}")
        await asyncio.sleep(2)


###############################################################################
# ORDER SUBMISSION
###############################################################################

def _get_clob_client():
    """Build an authenticated ClobClient from env vars."""
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds
    return ClobClient(
        host="https://clob.polymarket.com",
        key=os.getenv("POLY_PRIVATE_KEY", ""),
        chain_id=137,
        signature_type=1,
        funder=os.getenv("PROXY_ADDRESS", ""),
        creds=ApiCreds(
            api_key=os.getenv("POLY_API_KEY", ""),
            api_secret=os.getenv("POLY_API_SECRET", ""),
            api_passphrase=os.getenv("POLY_API_PASSPHRASE", ""),
        ),
    )


async def submit_taker_order(
    token_id: str,
    side: str,
    amount_usd: float,
    price: float,
    session_state: SessionState,
) -> bool:
    """Submit a real market (taker) order to Polymarket CLOB."""
    if amount_usd > MAX_SINGLE_TRADE_USD:
        print(f"  🛑 SAFETY BLOCK: ${amount_usd:.2f} > max "
              f"${MAX_SINGLE_TRADE_USD:.2f}")
        return False

    allowed, reason = session_state.can_trade(amount_usd)
    if not allowed:
        print(f"  🛑 RISK BLOCK: {reason}")
        return False

    private_key   = os.getenv("POLY_PRIVATE_KEY", "")
    proxy_address = os.getenv("PROXY_ADDRESS", "")
    if not private_key or private_key == "0xYOUR_PRIVATE_KEY_HERE":
        print("  🛑 No private key in .env")
        return False
    if not proxy_address:
        print("  🛑 No PROXY_ADDRESS in .env")
        return False

    try:
        from py_clob_client.clob_types import (MarketOrderArgs,
                                                PartialCreateOrderOptions)

        print(f"\n  📡 SUBMITTING LIVE ORDER:")
        print(f"     Token:  {token_id[:20]}...")
        print(f"     Side:   {side} (direction: BUY this outcome token)")
        print(f"     Amount: ${amount_usd:.2f}")
        print(f"     Price:  {price:.4f}")
        print(f"     Time:   {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        client = _get_clob_client()
        t0 = time.perf_counter()
        order_args = MarketOrderArgs(
            token_id=token_id, amount=amount_usd, side=side)
        options = PartialCreateOrderOptions(tick_size="0.01")

        signed_order = client.create_market_order(order_args, options)
        response = client.post_order(signed_order, orderType="FOK")
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"     Latency: {latency_ms:.0f}ms")
        print(f"     Response: {response}")

        if isinstance(response, dict):
            order_id = response.get("orderID", response.get("id", ""))
            status   = response.get("status", "")
            success  = (bool(order_id) or
                        status in ("matched", "live", "delayed"))
        else:
            success  = bool(response)
            order_id = str(response)
            status   = ""

        if success:
            session_state.record(amount_usd)
            print(f"\n  ✅ ORDER SUBMITTED TO EXCHANGE")
            print(f"     Order ID: {order_id}")
            print(f"     Status:   {status}")
            print(f"     Total spent: ${session_state.total_spent_usd:.2f}")
            print(f"     Trades: {session_state.trades_placed}/"
                  f"{MAX_TRADES_SESSION}")
            return True
        else:
            print(f"  ❌ Order rejected: {response}")
            return False

    except ImportError:
        print("  🛑 py-clob-client not installed")
        return False
    except Exception as e:
        print(f"  ❌ Order error: {e}")
        return False


###############################################################################
# SELL FUNCTIONS — fixed with retry + on-chain confirmation
###############################################################################

def get_token_balance(token_id: str) -> float:
    """Query actual conditional token balance (in shares)."""
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        client = _get_clob_client()
        result = client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL, token_id=token_id))
        if isinstance(result, dict):
            raw = result.get("balance", "0")
            balance = float(raw)
            # Polymarket uses 6 decimal places (USDC standard)
            if balance > 1e12:
                balance = balance / 1e6
            return balance
    except Exception as e:
        print(f"  ⚠️  Balance query error: {e}")
    return 0.0


def approve_conditional_token(token_id: str) -> bool:
    """Approve CLOB to spend our conditional tokens, with verification."""
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        client = _get_clob_client()

        # Check if already approved
        try:
            result = client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL, token_id=token_id))
            if isinstance(result, dict):
                allowance_raw = result.get("allowance", "0")
                if float(allowance_raw) > 1e15:
                    print(f"  ✅ Conditional token already approved")
                    return True
        except Exception:
            pass

        resp = client.update_balance_allowance(
            BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL, token_id=token_id))
        print(f"  ✅ Conditional token allowance requested")

        time.sleep(SELL_RETRY_DELAY)

        # Verify
        try:
            result = client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL, token_id=token_id))
            if isinstance(result, dict):
                allowance_raw = result.get("allowance", "0")
                print(f"  ✅ Allowance confirmed: {float(allowance_raw):.0f}")
                return True
        except Exception:
            pass

        return True
    except Exception as e:
        print(f"  ⚠️  Conditional approval error: {e}")
        return False


def place_limit_sell(token_id: str, shares: float,
                     price: float) -> Optional[str]:
    """Place a GTC limit sell with retry on allowance errors."""
    from py_clob_client.clob_types import OrderArgs, OrderType

    for attempt in range(SELL_RETRY_MAX):
        try:
            client = _get_clob_client()
            tick = client.get_tick_size(token_id)
            price_rounded = round(
                round(price / float(tick)) * float(tick), 4)
            price_rounded = max(float(tick),
                                min(price_rounded, 1.0 - float(tick)))

            sell_shares = round(shares, 2)
            if sell_shares < 0.01:
                print(f"  ⚠️  Shares too small to sell: {sell_shares}")
                return None

            args = OrderArgs(
                token_id=token_id,
                price=price_rounded,
                size=sell_shares,
                side="SELL",
            )
            signed = client.create_order(args)
            resp = client.post_order(signed, OrderType.GTC)

            if isinstance(resp, dict):
                oid = resp.get("orderID", "")
                ok = (resp.get("success") or
                      resp.get("status") == "matched" or oid)
                if ok:
                    return oid

                error_msg = str(resp.get("error_message",
                                         resp.get("errorMessage", "")))
                if "allowance" in error_msg.lower():
                    print(f"  ⚠️  Attempt {attempt+1}: allowance not ready. "
                          f"Re-approving...")
                    approve_conditional_token(token_id)
                    continue

                print(f"  ⚠️  Limit sell response: {resp}")
            else:
                print(f"  ⚠️  Limit sell raw: {resp}")
            return None

        except Exception as e:
            error_str = str(e).lower()
            if "allowance" in error_str and attempt < SELL_RETRY_MAX - 1:
                print(f"  ⚠️  Attempt {attempt+1} failed (allowance). "
                      f"Re-approving...")
                approve_conditional_token(token_id)
                continue
            print(f"  ❌ Limit sell error: {e}")
            return None

    return None


def sell_at_market(token_id: str, shares: float,
                   reason: str = "") -> tuple[bool, float]:
    """Submit a FOK market SELL order for `shares` tokens.

    For a SELL, MarketOrderArgs.amount is in *shares*, not USD.
    Returns (success, fill_price_approx).
    """
    from py_clob_client.clob_types import MarketOrderArgs, OrderType

    shares_rounded = round(shares * 0.98, 2)   # 2 % buffer for rounding
    if shares_rounded < 0.50:
        print(f"  ⚠️  Shares too small to sell: {shares_rounded:.2f}")
        return False, 0.0

    # Approve token (non-blocking best-effort; already approved at entry)
    try:
        approve_conditional_token(token_id)
    except Exception:
        pass

    label = f" [{reason}]" if reason else ""
    print(f"\n  📤 MARKET SELL{label}: {shares_rounded:.2f} shares of {token_id[:12]}…")

    for attempt in range(SELL_RETRY_MAX):
        try:
            client = _get_clob_client()
            args = MarketOrderArgs(
                token_id=token_id,
                amount=shares_rounded,
                side="SELL",
            )
            neg_risk = os.getenv("NEG_RISK", "false").lower() == "true"
            signed = client.create_market_order(args)
            resp = client.post_order(signed, OrderType.FOK)

            if isinstance(resp, dict):
                ok = (resp.get("success")
                      or resp.get("status") == "matched"
                      or resp.get("orderID"))
                if ok:
                    matched_sz = float(resp.get("size_matched", shares_rounded))
                    fill_price = (matched_sz / shares_rounded
                                  if matched_sz > 0 else 0.0)
                    print(f"  ✅ SELL FILLED: ~{matched_sz:.2f} shares, "
                          f"approx ${matched_sz:.2f} recovered")
                    return True, fill_price
                err = str(resp.get("error_message",
                                   resp.get("errorMessage", resp)))
                if "allowance" in err.lower() and attempt < SELL_RETRY_MAX - 1:
                    print(f"  ⚠️  Attempt {attempt+1}: allowance. Re-approving…")
                    approve_conditional_token(token_id)
                    time.sleep(SELL_RETRY_DELAY)
                    continue
                print(f"  ❌ Market sell rejected: {err}")
            else:
                print(f"  ❌ Market sell raw response: {resp}")
            return False, 0.0

        except Exception as e:
            err_s = str(e).lower()
            if "allowance" in err_s and attempt < SELL_RETRY_MAX - 1:
                print(f"  ⚠️  Attempt {attempt+1} allowance error. Re-approving…")
                approve_conditional_token(token_id)
                time.sleep(SELL_RETRY_DELAY)
                continue
            print(f"  ❌ sell_at_market error: {e}")
            return False, 0.0

    return False, 0.0


def cancel_order(order_id: str):
    """Cancel an open order."""
    try:
        client = _get_clob_client()
        client.cancel(order_id)
    except Exception as e:
        print(f"  ⚠️  Cancel error (may already be filled): {e}")


###############################################################################
# MONITOR & EXIT  —  Autonomous sell engine (pure-taker, all models active)
###############################################################################

def _fetch_book_prices(token_id: str) -> tuple[float, float, float]:
    """Return (best_bid, best_ask, mid) from live CLOB. 0s on failure."""
    import requests as req
    try:
        r = req.get(f"{CLOB_API}/book?token_id={token_id}", timeout=4)
        book = r.json()
        bid = float(book["bids"][0]["price"]) if book.get("bids") else 0.0
        ask = float(book["asks"][0]["price"]) if book.get("asks") else 0.0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        return bid, ask, mid
    except Exception:
        return 0.0, 0.0, 0.0


def _reeval_win_prob(btc_now: float, strike: float, side_label: str,
                     secs_left: float, entry_btc: float,
                     ofi_tracker: "OFITracker",
                     regime_classifier: "RegimeClassifier",
                     btc_history: list) -> float:
    """Re-evaluate P(win) post-entry using BTC movement + OFI + regime.

    Logic:
      1. Base: fraction of time BTC was on the right side of strike
         over the rolling observation window.
      2. OFI nudge: each 0.1 unit of aligned OFI adds +1.5 pp probability.
      3. Regime penalty: volatile regime → discount probability toward 0.50.
      4. Time decay: within 60s the market price is a better estimate;
         blend toward 0.5 to not over-commit to our stale entry model.
    """
    if btc_now <= 0 or strike <= 0 or not btc_history:
        return 0.50

    # 1. Base probability from BTC position relative to strike
    in_money = (btc_now > strike if side_label == "UP"
                else btc_now < strike)
    base_p = 0.62 if in_money else 0.38

    # 2. OFI adjustment (directional market pressure)
    ofi_z = 0.0
    try:
        sig = ofi_tracker.signal()
        # sig is already normalised; positive = buying pressure
        ofi_aligned = sig if side_label == "UP" else -sig
        ofi_z = max(-1.0, min(1.0, ofi_aligned))
        base_p = base_p + ofi_z * 0.08  # 8 pp max adjustment
    except Exception:
        pass

    # 3. Regime adjustment
    try:
        regime, _stats = RegimeClassifier.classify(btc_history)
        if regime == "volatile":
            base_p = 0.5 + (base_p - 0.5) * 0.70   # pull toward 0.5 by 30%
        elif regime == "trending":
            # trending in our direction reinforces the signal
            momentum_aligned = (entry_btc < btc_now if side_label == "UP"
                                 else entry_btc > btc_now)
            if momentum_aligned:
                base_p = min(base_p + 0.05, 0.92)
    except Exception:
        pass

    # 4. Time decay — with < 60s left the binary price is nearly decided;
    #    trust CLOB market price over our model
    if secs_left < 60:
        blend_w = (60 - secs_left) / 60   # 0 at T=60s → 1.0 at T=0s
        base_p  = base_p * (1 - blend_w * 0.30)  # reduce by up to 30% confidence

    return max(0.05, min(0.95, base_p))


async def monitor_and_exit(
    token_id: str,
    entry_price: float,
    shares: float,
    cost_usd: float,
    side_label: str,
    window_end: float,
    btc_ref: list = None,
    strike_ref: float = 0.0,
    session_state: SessionState = None,
    ofi_tracker=None,
    btc_history: list = None,
    live_log=None,
    book_cache: dict = None,
):
    """
    HOLD-TO-EXPIRY monitor.

    Strategy: Polymarket BTC binaries are illiquid after fill — market makers
    pull quotes and mid-window exits are unreliable.  The correct strategy is
    to enter only when the signal is strong and hold to expiry.

    Decision hierarchy (evaluated every EXIT_POLL_INTERVAL seconds):
      EMERGENCY  BTC crossed strike AND is >0.12% wrong AND >60s remaining
                 AND OFI strongly against AND P(win) < 0.25  → market sell
                 (Catastrophic adverse selection: position is near-certainly lost)
      HOLD       Everything else → hold to window expiry
    """
    print(f"\n  ⏳ HOLD-TO-EXPIRY  |  entry={entry_price:.4f}  "
          f"shares={shares:.2f}  cost=${cost_usd:.2f}  "
          f"side={side_label}  strike=${strike_ref:,.0f}")
    print(f"  Strategy: hold to window end.  Emergency exit only on catastrophic adverse.")
    print(f"  {'─' * 60}")

    entry_btc  = btc_ref[0] if btc_ref else 0.0
    btc_hist   = btc_history or []

    sold        = False
    sell_reason = "expiry"
    in_money    = None

    while time.time() < window_end:
        await asyncio.sleep(EXIT_POLL_INTERVAL)
        secs_left = max(0.0, window_end - time.time())

        # ── Live data snapshot ────────────────────────────────────────────
        btc_now = btc_ref[0] if btc_ref else 0.0
        if book_cache and token_id in book_cache:
            b = book_cache[token_id]
            bid, ask, mid = b["best_bid"], b["best_ask"], b["mid"]
        else:
            bid, ask, mid = _fetch_book_prices(token_id)

        in_money = (btc_now > strike_ref if side_label == "UP"
                    else btc_now < strike_ref) if btc_now > 0 else None
        diff_pct = ((btc_now - strike_ref) / strike_ref * 100
                    if btc_now > 0 and strike_ref > 0 else 0.0)
        icon     = ("🟢" if in_money else ("🔴" if in_money is False else "⚪"))

        # Update rolling BTC history for regime/OFI
        if btc_now > 0:
            btc_hist.append(btc_now)
            if len(btc_hist) > 120:
                btc_hist.pop(0)

        # ── OFI signal ────────────────────────────────────────────────────
        ofi_signal = 0.0
        if ofi_tracker is not None:
            try:
                ofi_signal = ofi_tracker.signal()
            except Exception:
                pass

        # ── Re-evaluate model probability ─────────────────────────────────
        win_prob = _reeval_win_prob(
            btc_now, strike_ref, side_label, secs_left,
            entry_btc, ofi_tracker, RegimeClassifier, btc_hist)

        # ── Regime ────────────────────────────────────────────────────────
        regime_label = "?"
        try:
            regime_label, _ = RegimeClassifier.classify(btc_hist)
        except Exception:
            pass

        # ── P&L display (model-implied when book is thin) ─────────────────
        if bid > 0 and ask > 0 and (ask - bid) > 0.40:
            eff_mid = win_prob
            src_tag = "model"
        else:
            eff_mid = mid if mid > 0 else win_prob
            src_tag = "CLOB"

        pnl_pct = ((shares * eff_mid - cost_usd) / cost_usd
                   if cost_usd > 0 else 0.0)

        # Expiry value: if in-money → shares × $1.00; else → $0
        expiry_value = shares if in_money else 0.0
        expiry_pnl   = expiry_value - cost_usd

        print(f"\r  {icon} {side_label} | BTC ${btc_now:,.0f} "
              f"({'IN' if in_money else 'OUT'} {diff_pct:+.3f}%) | "
              f"P(win)={win_prob:.2f} | OFI={ofi_signal:+.2f} | "
              f"regime={regime_label} | "
              f"expiry_pnl=${expiry_pnl:+.2f} | T-{secs_left:.0f}s",
              end="", flush=True)

        # ── Log tick to file ──────────────────────────────────────────────
        if live_log:
            live_log._emit("HOLD_TICK", {
                "secs_left":   round(secs_left, 1),
                "btc":         btc_now,
                "diff_pct":    round(diff_pct, 4),
                "in_money":    in_money,
                "win_prob":    round(win_prob, 3),
                "ofi":         round(ofi_signal, 3),
                "regime":      regime_label,
                "eff_mid":     round(eff_mid, 4),
                "expiry_pnl":  round(expiry_pnl, 3),
            })

        # ── EMERGENCY EXIT: catastrophic adverse selection only ───────────
        # Conditions (ALL must be true — this is intentionally very tight):
        #   1. BTC crossed to the wrong side of strike
        #   2. AND is now ≥ 0.12% in the wrong direction (not just noise)
        #   3. AND there is still >90s of window left (losing early = certain loss)
        #   4. AND OFI is strongly against us (|OFI| > 0.80)
        #   5. AND model says P(win) < 0.22 (near-certain loss)
        btc_wrong_side = (in_money is False) if in_money is not None else False
        btc_distance_wrong = abs(diff_pct)
        ofi_strongly_against = (
            (side_label == "UP"   and ofi_signal < -EMERGENCY_OFI_THRESHOLD) or
            (side_label == "DOWN"  and ofi_signal >  EMERGENCY_OFI_THRESHOLD))

        if (btc_wrong_side
                and btc_distance_wrong >= EMERGENCY_BTC_WRONG_PCT
                and secs_left > EMERGENCY_SECS_REMAIN
                and ofi_strongly_against
                and win_prob < EMERGENCY_WIN_PROB_MAX):

            pnl_est = shares * eff_mid - cost_usd
            print(f"\n\n  🚨 EMERGENCY EXIT: catastrophic adverse selection")
            print(f"     BTC {diff_pct:+.3f}% wrong | P(win)={win_prob:.2f} | "
                  f"OFI={ofi_signal:+.2f} | T-{secs_left:.0f}s")
            print(f"     Est. loss if held: ${shares * 0 - cost_usd:+.2f}")
            print(f"     Est. salvage now:  ${pnl_est:+.2f}")

            if live_log:
                live_log._emit("EMERGENCY_EXIT_DECISION", {
                    "reason":      "catastrophic_adverse",
                    "secs_left":   secs_left,
                    "diff_pct":    round(diff_pct, 4),
                    "win_prob":    round(win_prob, 3),
                    "ofi_signal":  round(ofi_signal, 3),
                    "regime":      regime_label,
                    "pnl_est":     round(pnl_est, 3),
                })

            ok, fill_p = sell_at_market(token_id, shares * 0.98, "emergency_adverse")
            if ok:
                realized_usd = shares * 0.98 * (fill_p if fill_p > 0 else eff_mid)
                realized_pnl = realized_usd - cost_usd
                print(f"\n  ✅ EMERGENCY SOLD | realized ${realized_usd:.2f}  "
                      f"P&L ${realized_pnl:+.2f}")
                if session_state:
                    session_state.record_sell(realized_usd)
                if live_log:
                    live_log.order_result(
                        order_id="emergency_sell",
                        status="sold",
                        filled_size=shares * 0.98,
                        filled_price=fill_p if fill_p > 0 else eff_mid,
                        slippage_bps=abs(fill_p - eff_mid) * 10000
                                     if fill_p > 0 and eff_mid > 0 else 0,
                    )
                sold        = True
                sell_reason = "emergency_adverse"
                break
            else:
                print(f"  ⚠️  Emergency sell failed — holding. Book is illiquid.")

    # ── End-of-window outcome ──────────────────────────────────────────────
    print()  # newline after carriage-return heartbeat
    if not sold:
        outcome  = "WIN ✅" if in_money else "LOSS ❌"
        pay_at_expiry = shares if in_money else 0.0
        expiry_pnl    = pay_at_expiry - cost_usd
        print(f"\n  ⏱  EXPIRY  |  {outcome}")
        print(f"     BTC ${btc_now:,.0f} vs strike ${strike_ref:,.0f} "
              f"({diff_pct:+.3f}%)")
        print(f"     Shares: {shares:.2f}  Cost: ${cost_usd:.2f}  "
              f"Expiry payout: ${pay_at_expiry:.2f}  P&L: ${expiry_pnl:+.2f}")
        if in_money:
            print(f"     Polymarket will credit ${shares:.2f} USDC to your wallet.")
        else:
            print(f"     Tokens expire worthless.  Loss: -${cost_usd:.2f}")

        if live_log:
            live_log._emit("EXPIRY_OUTCOME", {
                "in_money":     in_money,
                "btc_final":    btc_now,
                "strike":       strike_ref,
                "diff_pct":     round(diff_pct, 4),
                "shares":       shares,
                "cost_usd":     cost_usd,
                "expiry_pnl":   round(expiry_pnl, 3),
            })

    print()
    return sold, sell_reason


###############################################################################
# MAIN LOOP
###############################################################################

async def run_live_test():
    # ── Initialise live logger first ──
    live_log = LiveLogger()

    print()
    print("=" * 70)
    print("  ⚡ POLYMARKET SCALPER v6 — HOLD-TO-EXPIRY PRECISION MODE")
    print("=" * 70)
    print(f"  Strategy:       Enter with high precision, hold to expiry.")
    print(f"                  Exits mid-window ONLY on catastrophic adverse.")
    print(f"  ─── Entry Filters (ALL must pass) ───")
    print(f"  Price zone:     ${MIN_TOKEN_PRICE:.2f}-${MAX_TOKEN_PRICE:.2f}  "
          f"(market must be uncertain, not already priced)")
    print(f"  Window cutoff:  First {WINDOW_ENTRY_CUTOFF}s only "
          f"(market discovers direction after this)")
    print(f"  Min edge:       {MIN_EDGE_TO_TRADE:.1%}  (was 4.5%)")
    print(f"  BTC distance:   {MIN_BTC_DIFF_PCT:.2f}%-{MAX_BTC_DIFF_PCT:.2f}% "
          f"from strike in signal direction")
    print(f"  Conviction:     prob ∉ [0.38, 0.62] (no trade in mud zone)")
    print(f"  ─── Hold Logic ───")
    print(f"  Normal:         Hold all the way to window expiry")
    print(f"  Emergency only: BTC >0.12% wrong + OFI>0.80 against + P(win)<0.22 + T>90s")
    print(f"  ─── Why this works ───")
    print(f"  Buying at 0.40 → need 40% wins to break even, payout 150%")
    print(f"  Buying at 0.57 → need 57% wins, payout 75%  ← edge still there")
    print(f"  Buying at 0.68 → need 68% wins (we win ~65%) ← LOSING TRADE")
    print(f"  ─── Session Budget ───")
    print(f"  Max per trade:  ${MAX_SINGLE_TRADE_USD:.2f}")
    print(f"  Session limit:  ${MAX_SESSION_SPEND_USD:.2f}")
    print(f"  Max trades:     {MAX_TRADES_SESSION}")
    print(f"  ─── Research Modules ───")
    print(f"  [1] MC Kelly:       {KELLY_MC_SAMPLES} samples, "
          f"{KELLY_CONFIDENCE_PCT}th pctile")
    print(f"  [2] Calibration:    γ={CALIBRATION_GAMMA} (FLB correction)")
    print(f"  [3] Equity Sim:     {MC_EQUITY_PATHS} paths, "
          f"max P(ruin)={MAX_RUIN_PROB:.0%}")
    print(f"  [4] OFI:            disabled (not predictive on OOS)")
    print(f"  [5] Regime:         EWMA-Vol + Momentum classifier")
    print(f"  [6] Distribution:   Full variance/skew/CI tracking")
    print(f"  [7] Data feeds:     Coinbase WS (BTC) + Polymarket CLOB WS (book)")
    print(f"  ─── Live Logging ───")
    print(f"  Log file:       {live_log.path}")
    print(f"  Backtest ref:   win={LiveLogger.BACKTEST['win_rate']:.1%} "
          f"PF={LiveLogger.BACKTEST['profit_factor']:.2f} "
          f"fill={LiveLogger.BACKTEST['fill_rate_pct']:.1f}%")
    print("=" * 70)
    print()

    # ── Load empirical engine ──
    engine = EmpiricalEngine()
    if not engine.load():
        print("  Cannot run without training data. Exiting.")
        return

    # Step 1 OOS extraction: candle-based (p_market=None, no historical order book)
    candle_oos = engine.get_oos_confident_signals()
    if candle_oos:
        os.makedirs("logs", exist_ok=True)
        path = "logs/oos_confident_signals_candle.json"
        with open(path, "w") as f:
            json.dump(candle_oos, f, indent=2)
        print(f"  [OK] Candle OOS: {len(candle_oos)} confident → {path}")

    live_log.session_start_log(
        bankroll=MAX_SESSION_SPEND_USD,
        limits={
            "max_trade": MAX_SINGLE_TRADE_USD,
            "max_session": MAX_SESSION_SPEND_USD,
            "max_trades": MAX_TRADES_SESSION,
            "min_edge": MIN_EDGE_TO_TRADE,
        }
    )

    # ── Build research modules ──
    calibration = CalibrationCurve()
    empirical_calibrator = EmpiricalCalibrator()
    ofi_tracker = OFITracker(window=60)
    print(f"  [OK] Calibration curve: γ={calibration._gamma}")
    print(f"  [OK] Empirical: walk-forward + regime-stratified + live calibration")
    print(f"  [OK] OFI tracker: window=60")
    print(f"  [OK] Regime classifier: ready")
    print(f"  [OK] MC Kelly: {KELLY_MC_SAMPLES} posterior samples")
    print(f"  [OK] Equity simulator: {MC_EQUITY_PATHS} paths")
    dist_count = sum(1 for v in engine.dist_surface.values() if len(v) >= 10)
    print(f"  [OK] Distribution surface: {dist_count:,} valid bins")
    print()

    # ── Check credentials ──
    if (not os.getenv("POLY_PRIVATE_KEY")
            or os.getenv("POLY_PRIVATE_KEY") == "0xYOUR_PRIVATE_KEY_HERE"):
        print("  ❌ No credentials. Please fill in .env file first.")
        return

    session_state = SessionState(session_start=time.time())

    btc_price         = 0.0
    market            = None
    strike            = 0.0
    strike_calibrated = False
    window_id         = 0
    window_open_t     = 0.0
    transitions       = 0
    btc_prices        = deque(maxlen=120)
    current_regime    = "unknown"
    regime_stats      = {}

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=SSL_CTX),
        headers={"Accept": "application/json"},
        timeout=aiohttp.ClientTimeout(total=10),
    ) as http:

        # ── Coinbase WebSocket (matches Polymarket settlement source) ──
        print("  Connecting to Coinbase WebSocket...")
        btc_ready = asyncio.Event()

        async def btc_feed():
            nonlocal btc_price
            try:
                async with websockets.connect(
                    COINBASE_WS, ssl=SSL_CTX,
                    ping_interval=20, ping_timeout=10,
                ) as ws:
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}],
                    }))
                    print("  ✅ Coinbase connected")
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("type") == "ticker" and "price" in data:
                            btc_price = float(data["price"])
                            btc_prices.append(btc_price)
                            if not btc_ready.is_set():
                                btc_ready.set()
            except Exception as e:
                print(f"  Coinbase feed error: {e}")
                btc_ready.set()

        asyncio.create_task(btc_feed())
        await asyncio.wait_for(btc_ready.wait(), timeout=10.0)
        await asyncio.sleep(0.2)

        if btc_price <= 0:
            print("  ❌ Could not get BTC price from Coinbase. Exiting.")
            return

        print(f"  BTC price (Coinbase): ${btc_price:,.2f}")

        # ── Find active market ──
        print("\n  Searching for active BTC 5-min market...")
        market = await find_btc_5min_market(http)
        if not market:
            print("  ❌ No active BTC 5-min market found.")
            return

        print(f"  ✅ Market: {market['title']}")
        print(f"     UP token:   {market['token_up'][:20]}...")
        print(f"     DOWN token: {market['token_down'][:20]}...")

        # ── Polymarket CLOB WebSocket (order book, no REST polling) ──
        book_cache = {}
        clob_subscribe_queue = asyncio.Queue()
        asyncio.create_task(poly_clob_ws_feed(
            book_cache, market["token_up"], market["token_down"], clob_subscribe_queue,
        ))
        print("  Connecting to Polymarket CLOB WebSocket...")
        await asyncio.sleep(1.5)  # allow first book snapshot
        if not book_cache.get(market["token_up"]):
            print("  ⚠️ No CLOB book yet, waiting...")
            for _ in range(30):
                await asyncio.sleep(0.2)
                if book_cache.get(market["token_up"]):
                    break
        print("  ✅ Polymarket CLOB WebSocket connected")
        print("\n  Waiting for clean window...\n")

        # ── Main loop ──
        while not session_state.halted:
            now = time.time()
            cur_window = int(now // 300)

            # ── Window transition ──
            if cur_window != window_id:
                window_id     = cur_window
                window_open_t = now
                transitions  += 1
                strike_calibrated = False

                strike = btc_price
                ofi_tracker.reset()
                print(f"\n  {'─' * 50}")
                print(f"  ⏱  WINDOW {transitions} | "
                      f"Initial strike est: ${strike:,.2f}")
                print(f"  ⏱  Waiting {WINDOW_INIT_WAIT:.0f}s for CLOB...")

                new_market = await find_btc_5min_market(http)
                if new_market:
                    market = new_market
                    try:
                        clob_subscribe_queue.put_nowait(
                            [market["token_up"], market["token_down"]],
                        )
                    except Exception:
                        pass
                    print(f"  ✅ Market: {market['title']}")

                if transitions > 20:
                    print(f"\n  🏁 Session complete after "
                          f"{transitions - 1} window(s).")
                    session_state.halt("Session complete")
                    break

            if transitions == 0:
                time_remaining = 300 - (now % 300)
                print(f"  ⏳ Waiting for window boundary... "
                      f"T-{time_remaining:.0f}s", end="\r")
                await asyncio.sleep(LOOP_INTERVAL_FAST)
                continue

            if transitions < 1:
                await asyncio.sleep(LOOP_INTERVAL_FAST)
                continue

            if session_state.trades_placed >= MAX_TRADES_SESSION:
                print(f"\n  ✅ Max trades reached ({MAX_TRADES_SESSION}). "
                      f"Waiting...")
                await asyncio.sleep(LOOP_INTERVAL_RESET)
                continue

            window_end_time = (window_id + 1) * 300
            time_remaining = max(0, window_end_time - now)
            secs_since_open = now - window_open_t

            if time_remaining < 30:
                await asyncio.sleep(LOOP_INTERVAL_FAST)
                continue

            if secs_since_open < WINDOW_INIT_WAIT:
                wait_left = WINDOW_INIT_WAIT - secs_since_open
                print(f"  ⏳ CLOB init wait: {wait_left:.0f}s...", end="\r")
                await asyncio.sleep(LOOP_INTERVAL_FAST)
                continue

            # ── Order book (from WebSocket; no REST polling) ──
            book = book_cache.get(market["token_up"])
            if not book:
                await asyncio.sleep(LOOP_INTERVAL_FAST)
                continue

            mid = book["mid"]

            # ── (#4) UPDATE OFI ──
            ofi_tracker.update(book)
            ofi_signal = ofi_tracker.signal()

            # ── (#5) REGIME CLASSIFICATION ──
            if len(btc_prices) >= 30:
                current_regime, regime_stats = RegimeClassifier.classify(
                    btc_prices)

            # ── STRIKE CALIBRATION ──
            if not strike_calibrated and 0.01 < mid < 0.99:
                market_implied_pup = mid
                if 0.40 < market_implied_pup < 0.60:
                    strike = btc_price
                    strike_calibrated = True
                    print(f"  📐 Strike calibrated: ${strike:,.2f} "
                          f"(market ~50/50)")
                else:
                    prob_offset = market_implied_pup - 0.50
                    est_pct_diff = prob_offset * 0.5
                    strike = btc_price / (1 + est_pct_diff / 100)
                    strike_calibrated = True
                    print(f"  📐 Strike calibrated: ${strike:,.2f} "
                          f"(market P(UP)={mid:.2f})")

            # ── Ghost town filter ──
            min_tob = min(book["tob_bid_vol"], book["tob_ask_vol"])
            if min_tob < MIN_TOB_VOL:
                print(f"  💀 Ghost town: TOB vol ${min_tob:.2f} "
                      f"< ${MIN_TOB_VOL:.2f}", end="\r")
                live_log.reject("ghost_town", {"tob_vol": round(min_tob, 2)})
                await asyncio.sleep(LOOP_INTERVAL_RETRY)
                continue

            # ── Edge calculation ──
            pct_diff = (btc_price - strike) / strike * 100
            regime_key = REGIME_MAP.get(current_regime, "uncertain")
            prob_up, n_samples = engine.lookup(pct_diff, time_remaining,
                                               regime=regime_key)
            prob_down = 1.0 - prob_up

            if n_samples < MIN_SAMPLES_REQUIRED:
                await asyncio.sleep(LOOP_INTERVAL_FAST)
                continue

            # ── (#2) CALIBRATION CURVE ──
            cal_mid = calibration.calibrate(mid)
            cal_diff = abs(cal_mid - mid)

            # ── Model vs market sanity check ──
            model_market_diff = abs(prob_up - cal_mid)
            if model_market_diff > 0.25:
                print(f"\r  ⚠️  Model/market disagree: "
                      f"model={prob_up:.2f} vs cal_mkt={cal_mid:.2f}. "
                      f"Recalibrating...", end="")
                strike_calibrated = False
                await asyncio.sleep(LOOP_INTERVAL_RESET)
                continue

            edge_up   = prob_up   - cal_mid       - 0.02
            edge_down = prob_down - (1 - cal_mid)  - 0.02

            best_edge = max(edge_up, edge_down)
            best_side = "UP" if edge_up > edge_down else "DOWN"
            best_prob = prob_up if best_side == "UP" else prob_down
            best_token = (market["token_up"] if best_side == "UP"
                          else market["token_down"])
            best_price = cal_mid if best_side == "UP" else (1 - cal_mid)

            # (OFI removed — not predictive on OOS)

            # ── Trend filter ──
            if len(btc_prices) >= 20:
                recent = list(btc_prices)
                trend_20 = ((recent[-1] - recent[-20])
                            / recent[-20] * 100)
                if best_side == "DOWN" and trend_20 > 0.03:
                    print(f"\r  📈 Trend filter: BTC +{trend_20:.3f}%. "
                          f"Skipping DOWN.", end="")
                    live_log.reject("trend_filter",
                                    {"side": "DOWN", "trend_20": round(trend_20,4)})
                    await asyncio.sleep(LOOP_INTERVAL_RETRY)
                    continue
                if best_side == "UP" and trend_20 < -0.03:
                    print(f"\r  📉 Trend filter: BTC {trend_20:.3f}%. "
                          f"Skipping UP.", end="")
                    live_log.reject("trend_filter",
                                    {"side": "UP", "trend_20": round(trend_20,4)})
                    await asyncio.sleep(LOOP_INTERVAL_RETRY)
                    continue

            # ── (#6) FULL DISTRIBUTION CHECK ──
            dist = engine.get_distribution(pct_diff, time_remaining)
            dist_label = ""
            if dist:
                ci_range = dist["ci_upper"] - dist["ci_lower"]
                if dist["ci_lower"] < 0 < dist["ci_upper"]:
                    dist_label = f"CI=[{dist['ci_lower']:+.3f}, " \
                                 f"{dist['ci_upper']:+.3f}] " \
                                 f"skew={dist['skew']:+.2f}"
                else:
                    dist_label = f"CI=[{dist['ci_lower']:+.3f}, " \
                                 f"{dist['ci_upper']:+.3f}] " \
                                 f"skew={dist['skew']:+.2f} ✓"

            # ── Display ──
            regime_icon = {"trending": "📈", "mean_reverting": "🔄",
                           "volatile": "⚡", "neutral": "➖",
                           "unknown": "❓"}.get(current_regime, "❓")
            print(
                f"\r  {datetime.now().strftime('%H:%M:%S')} | "
                f"BTC ${btc_price:,.0f} K ${strike:,.0f} "
                f"diff {pct_diff:+.3f}% | "
                f"P={prob_up:.2f} edge={best_edge:+.1%} "
                f"{regime_icon}{regime_key[:4]} | T-{time_remaining:.0f}s",
                end=""
            )

            # ── Log every tick (every 10th logged to file) ──
            live_log.tick(btc_price, strike, pct_diff, best_edge,
                          ofi_signal, current_regime, time_remaining)

            # ── Edge threshold ──
            if best_edge < MIN_EDGE_TO_TRADE:
                live_log.reject("no_edge", {"edge": round(best_edge, 4)})
                await asyncio.sleep(LOOP_INTERVAL_FAST)
                continue

            # ── Phantom edge rejection ──
            if best_edge > MAX_EDGE_BELIEVABLE:
                print(f"\r  🚫 Edge {best_edge:.1%} > "
                      f"{MAX_EDGE_BELIEVABLE:.0%}. Recalibrating.", end="")
                live_log.reject("phantom_edge", {"edge": round(best_edge, 4)})
                strike_calibrated = False
                await asyncio.sleep(LOOP_INTERVAL_RESET)
                continue

            # ── Price zone filter (precision: uncertain-market zone only) ──
            if best_price < MIN_TOKEN_PRICE or best_price > MAX_TOKEN_PRICE:
                print(f"\r  ⚠️  Price ${best_price:.3f} outside zone "
                      f"[{MIN_TOKEN_PRICE:.2f}-{MAX_TOKEN_PRICE:.2f}]. "
                      f"Skipping.", end="")
                live_log.reject("price_zone", {"price": round(best_price, 4)})
                await asyncio.sleep(LOOP_INTERVAL_RETRY)
                continue

            # ── Window entry cutoff: first 120s only ─────────────────
            if secs_since_open > WINDOW_ENTRY_CUTOFF:
                print(f"\r  🕐 Window {secs_since_open:.0f}s > {WINDOW_ENTRY_CUTOFF}s "
                      f"cutoff. Waiting for next window.", end="")
                live_log.reject("entry_cutoff",
                                {"secs_open": round(secs_since_open, 1)})
                await asyncio.sleep(LOOP_INTERVAL_RETRY)
                continue

            # ── BTC distance gate ─────────────────────────────────────
            btc_diff_abs = abs(pct_diff)
            btc_on_our_side = ((best_side == "UP"   and pct_diff > 0) or
                               (best_side == "DOWN"  and pct_diff < 0))
            if not btc_on_our_side:
                print(f"\r  ↔️  BTC on wrong side ({pct_diff:+.3f}%). "
                      f"Skipping {best_side}.", end="")
                live_log.reject("btc_wrong_side",
                                {"diff": round(pct_diff, 4), "side": best_side})
                await asyncio.sleep(LOOP_INTERVAL_RETRY)
                continue
            if btc_diff_abs < MIN_BTC_DIFF_PCT:
                print(f"\r  ↔️  BTC too close to strike ({pct_diff:+.3f}%). "
                      f"Too noisy.", end="")
                live_log.reject("btc_too_close",
                                {"diff": round(pct_diff, 4)})
                await asyncio.sleep(LOOP_INTERVAL_RETRY)
                continue
            if btc_diff_abs > MAX_BTC_DIFF_PCT:
                print(f"\r  ↔️  BTC too far from strike ({pct_diff:+.3f}%). "
                      f"Market already knows.", end="")
                live_log.reject("btc_too_far",
                                {"diff": round(pct_diff, 4)})
                await asyncio.sleep(LOOP_INTERVAL_RETRY)
                continue

            # (OFI filter removed — not predictive)

            # ══════════════════════════════════════════════════════════
            # SIGNAL FOUND — run research-grade sizing pipeline
            # ══════════════════════════════════════════════════════════
            print()
            print(f"\n  ⚡ SIGNAL FOUND!")
            print(f"     Side:       {best_side}")
            print(f"     Model prob: {best_prob:.2%}")
            print(f"     Market mid: {mid:.4f} → calibrated: {cal_mid:.4f}")
            print(f"     Edge:       {best_edge:+.2%}")
            print(f"     Regime:     {current_regime} "
                  f"(vol_r={regime_stats.get('vol_ratio', '?')} "
                  f"M={regime_stats.get('momentum', '?')})")
            if dist:
                print(f"     Dist:       {dist_label}")
            print(f"     Samples:    {n_samples:,}")
            print(f"     TOB vol:    ${min_tob:.2f}")
            print(f"     Spread:     {book['spread']:.4f}")

            # ── (#1) MONTE CARLO KELLY SIZING ──
            bankroll = session_state.bankroll
            kelly_prob = best_prob
            kelly_price = max(0.02, min(0.98, best_price))

            trade_size, kelly_detail = MonteCarloKelly.compute(
                prob=kelly_prob,
                price=kelly_price,
                n_samples=n_samples,
                bankroll=bankroll,
                execution_type="TAKER",   # live_test_2usd only places taker orders
            )

            # (#5) Regime scaling
            regime_scale = REGIME_VOL_SCALE.get(current_regime, 1.0)
            trade_size = round(trade_size * regime_scale, 2)

            # Respect session budget
            remaining = MAX_SESSION_SPEND_USD - session_state.total_spent_usd
            trade_size = min(trade_size, remaining)
            # Floor to Polymarket exchange minimum ($1) — Kelly may go lower
            # on a tiny bankroll; we accept the marginal extra risk
            trade_size = max(MIN_KELLY_TRADE_USD, trade_size)
            trade_size = round(trade_size, 2)

            print(f"\n  🧮 KELLY SIZING (Monte Carlo):")
            print(f"     Bankroll:     ${bankroll:.2f}")
            print(f"     Kelly 25pct:  {kelly_detail.get('kelly_25pct', 0):.4f}")
            print(f"     Kelly median: "
                  f"{kelly_detail.get('kelly_median', 0):.4f}")
            print(f"     Kelly capped: "
                  f"{kelly_detail.get('kelly_capped', 0):.4f}")
            print(f"     Posterior:    Beta("
                  f"{kelly_detail.get('alpha', 0):.0f}, "
                  f"{kelly_detail.get('beta', 0):.0f})")
            print(f"     Regime adj:   ×{regime_scale:.2f} "
                  f"({current_regime})")
            print(f"     Trade size:   ${trade_size:.2f}")

            if trade_size < MIN_KELLY_TRADE_USD:
                print(f"  ⚠️  Kelly says ${trade_size:.2f} < "
                      f"${MIN_KELLY_TRADE_USD:.2f} min. Skipping.")
                live_log.reject("kelly_too_small",
                                {"kelly_size": trade_size,
                                 "min": MIN_KELLY_TRADE_USD})
                await asyncio.sleep(LOOP_INTERVAL_RETRY)
                continue

            # ── (#3) MONTE CARLO EQUITY SIMULATION ──
            trades_remaining = (MAX_TRADES_SESSION
                                - session_state.trades_placed)
            kelly_frac = kelly_detail.get("kelly_capped", 0.05)

            eq_sim = EquitySimulator.simulate(
                bankroll=bankroll,
                n_trades=min(trades_remaining, 10),
                prob=kelly_prob,
                price=kelly_price,
                kelly_frac=kelly_frac)

            print(f"\n  📈 EQUITY SIMULATION ({MC_EQUITY_PATHS} paths):")
            print(f"     P(ruin):       {eq_sim['p_ruin']:.1%}")
            print(f"     Median final:  ${eq_sim['median_final']:.2f}")
            print(f"     95% CI:        [${eq_sim['ci_5']:.2f}, "
                  f"${eq_sim['ci_95']:.2f}]")
            print(f"     E[return]:     ${eq_sim['e_return']:+.2f}")
            print(f"     DD mean/p95:   ${eq_sim['dd_mean']:.2f} / "
                  f"${eq_sim['dd_p95']:.2f}  "
                  f"({eq_sim['dd_pct_p95']:.1%} worst-5%)")
            print(f"     Loss streak:   mean={eq_sim['streak_mean']:.1f}  "
                  f"p95={eq_sim['streak_p95']}")
            print(f"     Time u/water:  mean={eq_sim['tuw_mean']:.1f} trades  "
                  f"({eq_sim['tuw_pct_mean']:.0%} of session)")

            if eq_sim["p_ruin"] > MAX_RUIN_PROB:
                print(f"  🛑 P(ruin)={eq_sim['p_ruin']:.1%} > "
                      f"{MAX_RUIN_PROB:.0%}. Reducing size by 50%.")
                trade_size = round(trade_size * 0.5, 2)
                if trade_size < MIN_KELLY_TRADE_USD:
                    print(f"  ⚠️  Reduced to ${trade_size:.2f}. Skipping.")
                    live_log.reject("ruin_prob",
                                    {"p_ruin": eq_sim["p_ruin"],
                                     "trade_size": trade_size})
                    await asyncio.sleep(LOOP_INTERVAL_RETRY)
                    continue

            # ── Log signal ──
            live_log.signal(
                side=best_side, edge=best_edge, prob=best_prob,
                cal_mid=cal_mid, size=trade_size, regime=current_regime,
                ofi=ofi_signal, spread=book["spread"],
                kelly_detail=kelly_detail, eq_sim=eq_sim,
            )

            # ── Final order summary ──
            print(f"\n  📋 ORDER SUMMARY:")
            print(f"     Amount:    ${trade_size:.2f}")
            print(f"     Token:     {best_token[:20]}...")
            print(f"     Direction: {best_side}")
            print(f"     Price:     {best_price:.4f}")
            print(f"     Max loss:  ${trade_size:.2f} (binary)")
            print(f"     Max gain:  "
                  f"${trade_size * (1 / best_price - 1):.2f}")
            print(f"     Budget after: "
                  f"${session_state.total_spent_usd + trade_size:.2f}"
                  f"/${MAX_SESSION_SPEND_USD:.2f}")

            # ── PLACE THE REAL ORDER ──
            order_t0 = time.perf_counter()
            success = await submit_taker_order(
                token_id=best_token,
                side="BUY",
                amount_usd=trade_size,
                price=best_price,
                session_state=session_state,
            )
            order_latency_ms = (time.perf_counter() - order_t0) * 1000

            if success:
                entry_shares = trade_size / best_price
                print(f"\n  ✅ Trade #{session_state.trades_placed} placed. "
                      f"Monitoring...")

                # Log order sent
                live_log.order_sent(
                    token_id=best_token, side=best_side,
                    size_usd=trade_size, price=best_price,
                    latency_ms=order_latency_ms,
                    response_raw="submitted",
                )

                # Fire token approval in background so it's ready when
                # the sell engine triggers (removes allowance bottleneck)
                async def _approve_bg():
                    await asyncio.get_event_loop().run_in_executor(
                        None, approve_conditional_token, best_token)

                asyncio.create_task(_approve_bg())

                # Adverse selection tracking — snapshot BTC price 10/30/60s post
                entry_btc    = btc_price
                adv_snaps    = [None, None, None]   # [10s, 30s, 60s]

                async def _adv_sel_tracker():
                    for i, delay in enumerate([10, 30, 60]):
                        await asyncio.sleep(delay if i == 0
                                            else delay - [10, 30][i-1])
                        adv_snaps[i] = btc_price

                adv_task = asyncio.create_task(_adv_sel_tracker())

                btc_ref = [btc_price]

                async def _update_btc_ref():
                    while time.time() < window_end_time:
                        btc_ref[0] = btc_price
                        await asyncio.sleep(0.1)

                ref_task = asyncio.create_task(_update_btc_ref())

                sold, sell_reason = await monitor_and_exit(
                    token_id=best_token,
                    entry_price=best_price,
                    shares=entry_shares,
                    cost_usd=trade_size,
                    side_label=best_side,
                    window_end=window_end_time,
                    btc_ref=btc_ref,
                    strike_ref=strike,
                    session_state=session_state,
                    ofi_tracker=ofi_tracker,
                    btc_history=list(btc_prices),
                    live_log=live_log,
                    book_cache=book_cache,
                )
                ref_task.cancel()
                adv_task.cancel()

                # Log adverse selection (use whatever snaps arrived)
                live_log.adverse_selection(
                    entry_price=entry_btc,
                    entry_side=best_side,
                    snap_10s=adv_snaps[0],
                    snap_30s=adv_snaps[1],
                    snap_60s=adv_snaps[2],
                )

                # Log final order result (only if not already logged by sell engine)
                if not sold:
                    in_money_final = (btc_ref[0] > strike if best_side == "UP"
                                      else btc_ref[0] < strike)
                    live_log.order_result(
                        order_id="submitted",
                        status="won" if in_money_final else "expired_loss",
                        filled_size=entry_shares,
                        filled_price=best_price,
                        slippage_bps=0,
                    )

                # Record outcome for calibration learning
                in_money_final = (btc_ref[0] > strike if best_side == "UP"
                                  else btc_ref[0] < strike)
                calibration.record_outcome(best_price, in_money_final)
                p_market = mid if best_side == "UP" else (1 - mid)
                empirical_calibrator.record_full(
                    best_prob, p_market, best_side, 1 if in_money_final else 0,
                )

                if sold:
                    remaining = (MAX_SESSION_SPEND_USD
                                 - session_state.total_spent_usd)
                    if remaining >= MIN_KELLY_TRADE_USD:
                        print(f"  🔄 Scalper: next trade immediately...")
                        await asyncio.sleep(LOOP_INTERVAL_RETRY)
                    else:
                        session_state.halt("Budget exhausted after sell")
                        break
            else:
                print(f"\n  ❌ Trade failed. Will retry if edge persists.")

            await asyncio.sleep(LOOP_INTERVAL_RETRY)

        # ── Final summary ──
        runtime_min = (time.time() - session_state.session_start) / 60
        net_pnl = (session_state.total_received_usd
                   - session_state.total_spent_usd)

        print()
        print("=" * 70)
        print("  SESSION COMPLETE")
        print("=" * 70)
        print(f"  Trades placed:  {session_state.trades_placed}")
        print(f"  Total spent:    ${session_state.total_spent_usd:.2f}")
        print(f"  Total received: ${session_state.total_received_usd:.2f}")
        print(f"  Net P&L:        ${net_pnl:+.2f}")
        print(f"  Duration:       {runtime_min:.1f} min")
        print(f"  Halt reason:    {session_state.halt_reason}")
        print(f"  Calibration γ:  {calibration._gamma:.3f}")
        print(f"  Final regime:   {current_regime}")
        emp_cal = empirical_calibrator.reliability()
        print(f"  Empirical cal:   {empirical_calibrator.summary_str()}")
        if emp_cal.get("n", 0) >= 5 and emp_cal.get("buckets"):
            buck_str = " ".join(f"[{b['mean_pred']:.2f}→{b['mean_actual']:.2f}]"
                         for b in emp_cal["buckets"] if b["count"] > 0)
            print(f"  Reliability:     {buck_str}")
        print()
        print("  ℹ️  Check polymarket.com to see your open positions.")
        print("     They will resolve automatically at window end.")
        print("=" * 70)

        # ── Live vs Backtest report ──
        live_log.session_end(
            trades_placed   = session_state.trades_placed,
            total_spent     = session_state.total_spent_usd,
            total_received  = session_state.total_received_usd,
            runtime_min     = runtime_min,
            empirical_calibrator = empirical_calibrator,
        )
        live_log.close()


if __name__ == "__main__":
    asyncio.run(run_live_test())
