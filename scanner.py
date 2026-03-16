"""
scanner.py — Polymarket Market Scanner
=======================================

For every active market, collects and scores 6 signals:

  1. Spread (cents)      best_ask - best_bid  — profit per round-trip
  2. 24h volume ($)      flow that drives fills
  3. Trade frequency     estimated trades/hr from volume ÷ avg_trade_size
  4. L1 depth ($)        $ sitting at best bid + best ask (from CLOB book)
  5. L2 depth ($)        $ at 2nd level each side — confirms real book
  6. Price zone          mid between 0.10–0.90 (near 0/1 = frozen market)

Scoring (100 points):
  Spread       35 pts  (key profit driver, cap at 15c)
  Trade freq   25 pts  (key fill driver — low freq = days waiting)
  Volume       20 pts  (corroborates freq)
  L1 depth     10 pts  (can we actually fill?)
  L2 depth      5 pts  (is the book real or one order?)
  Price zone    5 pts  (near 0/1 = no counterparties)

Hard gates (market rejected if ANY fail):
  spread       ≥ 3 cents              (from Gamma bestBid/bestAsk — free!)
  price zone   0.05 ≤ mid ≤ 0.95     (exclude near-zero / near-certain)
  volume       ≥ $30/day
  L1 depth     ≥ $10 each side        (from CLOB book, fetched for survivors only)
  end date     ≥ 7 days away

Pipeline efficiency:
  Phase 1 (Gamma API, fast): broad fetch + spread/zone pre-filter
    → rejects ~95% of markets without any CLOB call
  Phase 2 (CLOB book, slow): fetch only pre-filtered survivors
    → get L1/L2 depth for the small promising set
  Phase 3: score, rank, output top N

Run:
  python3 scanner.py                 # scan every 60s, show top 5
  python3 scanner.py --interval 30   # faster
  python3 scanner.py --top 3         # stricter
  python3 scanner.py --debug         # verbose gate failures
"""

import asyncio
import aiohttp
import math
import ssl
import json
import time
import os
import random
import argparse
import logging
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import datetime, timezone

from prob_engine import fetch_recent_trades, estimate_true_probability
from cross_market import run_cross_market_analysis
from fundamental_sources import (
    get_news_for_markets,
    merge_fundamental_into_prob,
    FundamentalSignal,
)

log = logging.getLogger("scanner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── constants ─────────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"
DATA_API  = "https://data-api.polymarket.com"

LOG_DIR    = "logs"
TOP_LOG    = os.path.join(LOG_DIR, "scanner_top.json")
ALERTS_LOG = os.path.join(LOG_DIR, "scanner_alerts.jsonl")

DEFAULT_INTERVAL = 60    # seconds between full scans
DEFAULT_TOP      = 5     # show and trade only top N

# ── hard gates ────────────────────────────────────────────────────────────────
GATE_MIN_SPREAD_CENTS     = 1.0   # cents  (or spread_avg_last_5 >= 1)
GATE_SPREAD_AVG_LAST_5    = 1.0   # if avg spread last 5 scans >= this, quote
GATE_MID_MIN              = 0.03  # price zone
GATE_MID_MAX              = 0.97  # price zone
GATE_MIN_VOLUME_24H       = 1_000 # USD/day
EDGE_MIN_CENTS            = 0.05  # min |edge_cents| to quote (relaxed for slow markets)
GATE_MIN_L1_DEPTH         = 1     # USD on EACH side at L1 (was 2)
GATE_MIN_DAYS_TO_END      = 2     # days
GATE_MIN_LAST_TRADE_S     = 86400 # 24h — prediction markets are slow; 1hr blocked everything
GATE_MIN_ORDER_COUNT      = 2     # min orders per side — OR top3 depth (was 4)
GATE_MIN_TOP3_DEPTH_USD   = 80    # bid_top3 + ask_top3 > this → market_ok (was 200)

# Estimated average trade size in USD — used to infer trade frequency
# from volume when the authenticated /trades endpoint is unavailable.
# Sports/politics markets typically trade in $20–$100 lots.
AVG_TRADE_SIZE_USD     = 40.0

# ── scoring caps (normalisation denominators) ─────────────────────────────────
CAP_SPREAD_CENTS  = 15      # 15c = 100% of spread score
CAP_VOLUME        = 3_000   # $3k/day
CAP_TRADES_HR     = 10      # 10/hr
CAP_L1_DEPTH      = 500     # $500 at min(bid_l1, ask_l1)
CAP_L2_DEPTH      = 200     # $200 at min(bid_l2, ask_l2)

BOOK_DELAY        = 0.25    # secs between CLOB book calls (rate limit)

# ── rolling activity tracker ──────────────────────────────────────────────────
HISTORY_WINDOW_S        = 1800   # 30-minute long window
HISTORY_WINDOW_SHORT_S  = 300    # 5-minute short window — detect death quickly
MIN_MOVE_TICKS          = 0.001  # price change ≥ 0.1c counts as a "move"
MIN_SNAPSHOTS_HIST      = 5      # need ≥5 snapshots before trusting history
MIN_SNAPSHOTS_HIGH_CONF = 15     # ≥15 snapshots → high-confidence label
MAX_HISTORY_PER_TOKEN   = 120    # cap buffer at 120 snapshots (≈2 h at 60s)

# ── micro-fair-value (print proxy) ────────────────────────────────────────────
# On thin markets, mid = (bid+ask)/2 is often fake. Track where prints happen.
PRINT_EMA_ALPHA         = 0.35   # EMA smoothing for inferred print prices
MIN_PRINTS_FOR_FAIR     = 2      # need ≥2 inferred prints before using print_ema
MID_STALE_THRESHOLD     = 0.02   # |print_ema - mid| > 2% → mid is stale

# ── market death detection ────────────────────────────────────────────────────
# Markets die suddenly. Short window must see activity or we flag death.
MARKET_DEATH_MIN_SNAPS_SHORT = 0   # 0 = only flag if truly no short-window data
SPREAD_EXPLOSION_MULT   = 6.0      # spread must explode a lot to flag death (was 4)

# ── toxic flow detection ──────────────────────────────────────────────────────
# "Wide spreads during momentum = trap."  When informed traders hit you,
# price doesn't drift slowly — it gaps and continues.  Do NOT quote when:
#   |imbalance| > TOXIC_IMBALANCE  AND
#   (velocity_accelerating  OR  spread_widening_suddenly)
TOXIC_IMBALANCE           = 0.90   # very high — only skip extreme one-sided flow
TOXIC_VELOCITY_ACCEL_C_HR2 = 12.0  # need very sharp gapping to skip
TOXIC_SPREAD_WIDEN_C       = 5.0   # spread must widen a lot to skip

# ── correlation clustering ─────────────────────────────────────────────────
# One news shock (election result, game score) can move every market in the
# same cluster simultaneously, turning independent P&L into correlated loss.
# We cap active quotes per cluster to MAX_CLUSTER_EXPOSURE.
MAX_CLUSTER_EXPOSURE = 2   # max simultaneous active quotes in one cluster

# Each rule: (cluster_id, cluster_type, [keyword_fragments])
# Matching is case-insensitive substring search on question + slug.
CLUSTER_RULES: list[tuple[str, str, list[str]]] = [
    # ── Sports leagues ──────────────────────────────────────────────
    ("nfl",              "sports", ["nfl", "super bowl", "quarterback",
                                    "touchdown", "chiefs", "eagles", "49ers"]),
    ("nba",              "sports", ["nba", "basketball", "lakers", "celtics",
                                    "nuggets", "bucks", "knicks"]),
    ("premier_league",   "sports", ["premier league", "epl", "arsenal", "chelsea",
                                    "liverpool", "man city", "man united",
                                    "tottenham", "newcastle"]),
    ("champions_league", "sports", ["champions league", "ucl", "europa league"]),
    ("bundesliga",       "sports", ["bundesliga", "bayern", "dortmund", "leverkusen",
                                    "rb leipzig", "eintracht", "top 4", "top 2"]),
    ("soccer_intl",      "sports", ["world cup", "copa america", "euro 2024",
                                    "euro 2025", "euro 2026", "nations league"]),
    ("mlb",              "sports", ["mlb", "world series", "baseball", "yankees",
                                    "dodgers", "cubs"]),
    ("nhl",              "sports", ["nhl", "stanley cup", "hockey"]),
    ("college_football", "sports", ["ncaa", "college football", "cfp",
                                    "sec championship", "big ten"]),
    ("cricket",          "sports", ["ipl", "cricket", "test match", "odi",
                                    "t20", "bcci"]),
    ("mma_boxing",       "sports", ["ufc", "boxing", "mma", "knockout",
                                    "bout", "heavyweight"]),
    ("tennis",           "sports", ["wimbledon", "us open tennis", "australian open",
                                    "french open", "roland garros", "atp", "wta"]),
    ("formula1",         "sports", ["formula 1", "f1 race", "grand prix",
                                    "verstappen", "hamilton f1"]),
    # ── Politics by geography ────────────────────────────────────────
    ("us_politics",   "politics", ["trump", "biden", "harris", "democrat",
                                   "republican", "senate", "congress",
                                   "presidential", "white house", "scotus",
                                   "electoral college", "us election",
                                   "presidential election"]),
    ("uk_politics",   "politics", ["uk election", "labour party", "conservative",
                                   "tory", "starmer", "sunak",
                                   "prime minister uk", "westminster"]),
    ("eu_politics",   "politics", ["european union", "macron", "scholz",
                                   "european parliament", "nato"]),
    ("mideast",       "geo",      ["israel", "gaza", "iran", "iranian regime",
                                   "supreme leader iran", "us forces enter",
                                   "hamas", "hezbollah", "middle east", "west bank",
                                   "geopolitics"]),
    ("russia_ukraine","geo",      ["russia", "ukraine", "putin", "zelensky",
                                   "nato ukraine", "kyiv", "moscow"]),
    ("china_taiwan",  "geo",      ["china", "taiwan", "xi jinping",
                                   "taiwan strait", "beijing"]),
    # ── Financial / macro ────────────────────────────────────────────
    ("crypto",        "crypto",   ["bitcoin", "ethereum", "btc", "eth ",
                                   "crypto", "defi", "solana", "xrp",
                                   "crypto predict", "crypto price"]),
    ("fed_macro",     "macro",    ["federal reserve", "fed rate", "interest rate",
                                   "inflation", "gdp", "jobs report", "cpi",
                                   "fomc", "rate cut", "rate hike", "recession",
                                   "rate prediction"]),
    # ── Entertainment ────────────────────────────────────────────────
    ("entertainment", "entertainment", ["oscars", "oscars 2026", "grammy", "emmys",
                                        "box office", "golden globe",
                                        "academy award"]),
    # ── Live sports (high liquidity) ─────────────────────────────────────
    ("live_football", "sports",   ["live football", "live match", "match day",
                                   "kickoff", "in play", "live soccer"]),
]


def tag_cluster(question: str, slug: str) -> tuple[str, str]:
    """
    Return (cluster_id, cluster_type) for a market.
    First-wins against CLUSTER_RULES; falls back to ("uncategorized", "other").
    """
    text = (question + " " + slug).lower()
    for cluster_id, cluster_type, keywords in CLUSTER_RULES:
        if any(kw in text for kw in keywords):
            return cluster_id, cluster_type
    return "uncategorized", "other"


SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE


# ── market regime watcher (24h baseline vs rolling activity) ──────────────────
#
# Compares Gamma volume_24h (24h ground truth) against rolling activity_per_hr
# (~30 min of quote history).  When rolling activity drops to < REGIME_CHANGE_RATIO
# of baseline, OR spread_cv > REGIME_SPREAD_CV_MAX, we flag a regime change and
# reduce quote size by 50% until REGIME_RECOVERY_SCANS consecutive healthy scans.

REGIME_CHANGE_RATIO   = 0.20   # activity < 20% of 24h baseline -> degraded
REGIME_RECOVERY_RATIO = 0.50   # activity > 50% of baseline + stable -> recovery
REGIME_SPREAD_CV_MAX  = 0.60   # spread_cv > 0.60 alone -> degraded (chaotic spread)
REGIME_RECOVERY_SCANS = 3      # consecutive healthy scans needed to exit degraded
REGIME_SIZE_MULT      = 1.0    # size multiplier when degraded (1.0 = no reduction)


@dataclass
class _TokenRegime:
    size_mult:      float = 1.0
    recovery_count: int   = 0
    is_degraded:    bool  = False
    last_ratio:     float = 1.0
    last_spread_cv: float = 0.0
    degraded_since: float = 0.0


class MarketRegimeWatcher:
    """
    Detects regime changes by comparing rolling activity (last ~30 min) against
    the 24h volume baseline from Gamma API.

      degraded  : activity_ratio < REGIME_CHANGE_RATIO OR spread_cv > REGIME_SPREAD_CV_MAX
      recovery  : REGIME_RECOVERY_SCANS consecutive scans above REGIME_RECOVERY_RATIO
                  AND spread_cv < REGIME_SPREAD_CV_MAX * 0.80

    When degraded, size_multiplier = 0.50 (50% size reduction).
    """

    def __init__(self):
        self._state: dict[str, _TokenRegime] = {}

    def update(self, token: str, activity_per_hr: float,
               trades_per_hr_baseline: float, spread_cv: float) -> tuple[bool, float]:
        """
        Returns (newly_degraded, size_multiplier).
          newly_degraded  True only on the scan degradation first occurs.
          size_multiplier 0.5 while degraded, 1.0 otherwise.
        """
        s        = self._state.setdefault(token, _TokenRegime())
        baseline = max(trades_per_hr_baseline, 0.1)
        ratio    = activity_per_hr / baseline
        s.last_ratio     = round(ratio, 3)
        s.last_spread_cv = round(spread_cv, 3)

        newly_degraded = False
        if not s.is_degraded:
            bad = ratio < REGIME_CHANGE_RATIO or spread_cv > REGIME_SPREAD_CV_MAX
            if bad:
                s.is_degraded    = True
                s.size_mult      = REGIME_SIZE_MULT
                s.recovery_count = 0
                s.degraded_since = time.time()
                newly_degraded   = True
                log.warning(
                    "REGIME CHANGE: token=%.8s  activity_ratio=%.2f  "
                    "spread_cv=%.2f  -> size*%.1f",
                    token, ratio, spread_cv, REGIME_SIZE_MULT)
        else:
            recovering = (ratio >= REGIME_RECOVERY_RATIO and
                          spread_cv < REGIME_SPREAD_CV_MAX * 0.80)
            if recovering:
                s.recovery_count += 1
                if s.recovery_count >= REGIME_RECOVERY_SCANS:
                    s.is_degraded    = False
                    s.size_mult      = 1.0
                    held = (time.time() - s.degraded_since) / 60
                    log.info("REGIME RECOVERED: token=%.8s  ratio=%.2f  held=%.0fmin",
                             token, ratio, held)
            else:
                s.recovery_count = 0

        return newly_degraded, s.size_mult

    def get_mult(self, token: str) -> float:
        return self._state.get(token, _TokenRegime()).size_mult

    def degraded_tokens(self) -> list:
        return [t for t, s in self._state.items() if s.is_degraded]

    def summary_line(self) -> str:
        deg  = self.degraded_tokens()
        n_d  = len(deg)
        n_ok = len(self._state) - n_d
        if not self._state:
            return "no markets watched"
        return (f"{n_d} degraded (x{REGIME_SIZE_MULT})  {n_ok} healthy  "
                + (f"degraded: {', '.join(deg[:3])}" if deg else ""))


# ── quote history ───────────────────────────────────────────────────────────── ─────────────────────────────────────────────────────────────

@dataclass
class QuoteSnapshot:
    ts:           float
    bid:          float
    ask:          float
    spread_c:     float = 0.0   # (ask - bid) * 100
    bid_l1_size:  float = 0.0   # size at best bid (for queue position)
    ask_l1_size:  float = 0.0   # size at best ask (for queue position)

# QuoteHistoryStore: token_yes (str) → deque[QuoteSnapshot]
QuoteHistoryStore = dict


def record_snapshot(history: QuoteHistoryStore,
                    token: str, bid: float, ask: float,
                    bid_l1_size: float = 0.0, ask_l1_size: float = 0.0) -> None:
    """Append current bid/ask and L1 sizes to the rolling window."""
    if token not in history:
        history[token] = deque(maxlen=MAX_HISTORY_PER_TOKEN)
    spread_c = (ask - bid) * 100 if bid > 0 and ask < 1 else 0.0
    history[token].append(QuoteSnapshot(
        ts=time.time(), bid=bid, ask=ask,
        spread_c=spread_c, bid_l1_size=bid_l1_size, ask_l1_size=ask_l1_size,
    ))
    # Trim entries older than HISTORY_WINDOW_S
    cutoff = time.time() - HISTORY_WINDOW_S
    while history[token] and history[token][0].ts < cutoff:
        history[token].popleft()


def compute_activity(history: QuoteHistoryStore,
                     token: str) -> tuple[int, float, str]:
    """
    Count quote moves in the last HISTORY_WINDOW_S seconds.

    A "move" = bid OR ask changed by ≥ MIN_MOVE_TICKS between two consecutive
    snapshots.  Extrapolated to moves/hr for scoring.

    Returns:
        (moves_30m, activity_per_hr, source)
    where source is one of:
        "hist_high"  ≥ MIN_SNAPSHOTS_HIGH_CONF snapshots (reliable)
        "hist_low"   MIN_SNAPSHOTS_HIST – MIN_SNAPSHOTS_HIGH_CONF snaps (early)
        "vol_est"    not enough history yet (caller should fall back to volume)
    """
    snaps = history.get(token)
    if not snaps or len(snaps) < MIN_SNAPSHOTS_HIST:
        return 0, 0.0, "vol_est"

    moves = 0
    prev  = snaps[0]
    for snap in list(snaps)[1:]:
        if (abs(snap.bid - prev.bid) >= MIN_MOVE_TICKS or
                abs(snap.ask - prev.ask) >= MIN_MOVE_TICKS):
            moves += 1
        prev = snap

    # Extrapolate to per-hour over the actual observed window
    window_s  = snaps[-1].ts - snaps[0].ts
    window_hr = max(window_s / 3600, 1 / 120)   # floor at 30 s to avoid ÷0
    rate_hr   = round(moves / window_hr, 3)

    n      = len(snaps)
    source = "hist_high" if n >= MIN_SNAPSHOTS_HIGH_CONF else "hist_low"
    return moves, rate_hr, source


# PrintEMAStore: token → {"ema": float, "n_prints": int}
# Persists across scan cycles so print_ema accumulates.
PrintEMAStore = dict


def compute_micro_fair_value(
    history: QuoteHistoryStore,
    token: str,
    current_bid: float,
    current_ask: float,
    print_ema_store: PrintEMAStore,
) -> dict:
    """
    Micro-fair-value: on thin markets, mid is often fake. Track where prints
    actually happen and use that as fair value.

    Print proxy: when we detect a quote move (bid/ask changed), we infer a
    trade happened. The print price = midpoint of the PREVIOUS snapshot
    (the level that got consumed). EMA of these inferred prints = print_ema.

    If print_ema diverges from mid consistently → mid is stale.
      print_ema > mid  → buyers lifting (mid stale low)
      print_ema < mid  → sellers hitting (mid stale high)

    Returns:
      fair_value     use print_ema when n_prints ≥ MIN_PRINTS_FOR_FAIR, else weighted_mid
      print_ema      EMA of inferred prints
      mid_stale      True if |print_ema - mid| > MID_STALE_THRESHOLD
      stale_direction "HIGH" | "LOW" | "" (print_ema > mid = mid stale low = direction HIGH)
      n_prints       number of inferred prints in window
    """
    snaps = list(history.get(token, []))
    mid   = (current_bid + current_ask) / 2 if current_bid > 0 and current_ask < 1 else 0.5

    if token not in print_ema_store:
        print_ema_store[token] = {"ema": mid, "n_prints": 0}

    store = print_ema_store[token]
    alpha = PRINT_EMA_ALPHA

    # Infer prints from quote moves
    for i in range(1, len(snaps)):
        prev, curr = snaps[i - 1], snaps[i]
        if (abs(curr.bid - prev.bid) >= MIN_MOVE_TICKS or
                abs(curr.ask - prev.ask) >= MIN_MOVE_TICKS):
            # Print happened at the midpoint of the previous snapshot
            print_price = (prev.bid + prev.ask) / 2
            store["ema"] = alpha * print_price + (1 - alpha) * store["ema"]
            store["n_prints"] += 1

    print_ema = store["ema"]
    n_prints  = store["n_prints"]

    # Fair value: use print_ema when we have enough prints, else fall back to mid
    if n_prints >= MIN_PRINTS_FOR_FAIR:
        fair_value = print_ema
    else:
        fair_value = mid

    # Stale mid detection
    diff = print_ema - mid
    mid_stale = abs(diff) > MID_STALE_THRESHOLD
    if mid_stale:
        stale_direction = "HIGH" if diff > 0 else "LOW"  # print > mid → mid stale low
    else:
        stale_direction = ""

    return {
        "fair_value":      round(fair_value, 4),
        "print_ema":       round(print_ema, 4),
        "mid_stale":       mid_stale,
        "stale_direction": stale_direction,
        "n_prints":        n_prints,
    }


def _regime_for_snaps(
    snaps: list,
    activity_per_hr: float,
    scan_interval_s: float,
) -> dict:
    """
    Compute regime signals for a given list of snapshots.
    Used by compute_market_regime_dual for both short and long windows.
    """
    n = len(snaps)
    if n < 2:
        return {
            "spread_cv": 0.0, "spread_stability": 1.0,
            "trade_imbalance": 0.0, "quote_velocity_c_hr": 0.0,
            "velocity_acceleration": 0.0, "spread_change_c": 0.0,
            "spread_widening_suddenly": False, "toxic_flow_detected": False,
            "adverse_sel_risk": 0.0, "mean_spread_c": 0.0,
        }

    sp_list = [(s.ask - s.bid) * 100 for s in snaps]
    mean_sp = sum(sp_list) / len(sp_list)
    cv = 0.0
    if mean_sp > 0.001:
        var_sp = sum((x - mean_sp) ** 2 for x in sp_list) / len(sp_list)
        cv = (var_sp ** 0.5) / mean_sp
    spread_cv = round(cv, 4)
    spread_stability = round(max(0.0, 1.0 - min(cv, 1.0)), 4)

    # Trade imbalance
    buy_sig = sell_sig = 0
    prev = snaps[0]
    for snap in snaps[1:]:
        db, da = snap.bid - prev.bid, snap.ask - prev.ask
        if db >= MIN_MOVE_TICKS: buy_sig += 1
        if da >= MIN_MOVE_TICKS: buy_sig += 1
        if db <= -MIN_MOVE_TICKS: sell_sig += 1
        if da <= -MIN_MOVE_TICKS: sell_sig += 1
        prev = snap
    total = buy_sig + sell_sig
    imbalance = (buy_sig - sell_sig) / total if total else 0.0
    trade_imbalance = round(imbalance, 4)

    # Velocity
    mid_first = (snaps[0].bid + snaps[0].ask) / 2
    mid_last  = (snaps[-1].bid + snaps[-1].ask) / 2
    window_hr = max((snaps[-1].ts - snaps[0].ts) / 3600, 1 / 120)
    velocity  = (mid_last - mid_first) * 100 / window_hr
    quote_velocity_c_hr = round(velocity, 3)

    # Acceleration
    velocity_acceleration = 0.0
    if n >= 6:
        mid_idx = n // 2
        mid_mid = (snaps[mid_idx].bid + snaps[mid_idx].ask) / 2
        dt1_hr = max((snaps[mid_idx].ts - snaps[0].ts) / 3600, 1 / 120)
        dt2_hr = max((snaps[-1].ts - snaps[mid_idx].ts) / 3600, 1 / 120)
        v1 = (mid_mid - mid_first) * 100 / dt1_hr
        v2 = (mid_last - mid_mid) * 100 / dt2_hr
        dt_hr = (dt1_hr + dt2_hr) / 2
        velocity_acceleration = round((v2 - v1) / dt_hr, 3) if dt_hr > 0 else 0.0

    # Spread change
    spread_prev = getattr(snaps[-2], "spread_c", 0.0) or (snaps[-2].ask - snaps[-2].bid) * 100
    spread_now  = getattr(snaps[-1], "spread_c", 0.0) or (snaps[-1].ask - snaps[-1].bid) * 100
    spread_change_c = round(spread_now - spread_prev, 2)
    spread_widening_suddenly = spread_change_c > TOXIC_SPREAD_WIDEN_C

    # Toxic flow
    velocity_accelerating = abs(velocity_acceleration) > TOXIC_VELOCITY_ACCEL_C_HR2
    toxic_flow_detected = (
        abs(imbalance) > TOXIC_IMBALANCE
        and (velocity_accelerating or spread_widening_suddenly)
    )

    # Adverse selection risk
    if toxic_flow_detected:
        adverse_sel_risk = 1.0
    else:
        as_imbalance = abs(trade_imbalance)
        as_momentum  = min(abs(quote_velocity_c_hr) / 10.0, 1.0)
        as_chaos     = 1.0 - spread_stability
        adverse_sel_risk = round(min(0.35 * as_imbalance + 0.45 * as_momentum + 0.20 * as_chaos, 1.0), 4)

    return {
        "spread_cv": spread_cv,
        "spread_stability": spread_stability,
        "trade_imbalance": trade_imbalance,
        "quote_velocity_c_hr": quote_velocity_c_hr,
        "velocity_acceleration": velocity_acceleration,
        "spread_change_c": spread_change_c,
        "spread_widening_suddenly": spread_widening_suddenly,
        "toxic_flow_detected": toxic_flow_detected,
        "adverse_sel_risk": adverse_sel_risk,
        "mean_spread_c": mean_sp,
    }


def compute_market_regime(history: QuoteHistoryStore,
                          token: str,
                          activity_per_hr: float,
                          scan_interval_s: float = 60.0) -> dict:
    """
    Dual-window regime: short (5 min) + long (30 min). Combines both so that
    fast death or toxic flow in the short window triggers immediately, while
    the long window provides stability context.

    Market death: had activity before (long window) but short window is empty
    or spread exploded (short-window spread > 3× long-window mean).
    """
    snaps = list(history.get(token, []))
    now   = time.time()
    cutoff_short = now - HISTORY_WINDOW_SHORT_S
    cutoff_long = now - HISTORY_WINDOW_S

    snaps_long  = [s for s in snaps if s.ts >= cutoff_long]
    snaps_short = [s for s in snaps if s.ts >= cutoff_short]

    # Regime for each window
    reg_long  = _regime_for_snaps(snaps_long, activity_per_hr, scan_interval_s)
    reg_short = _regime_for_snaps(snaps_short, activity_per_hr, scan_interval_s)

    # ── combine: use WORST of both ────────────────────────────────────────
    # Short window detects death and fast toxic flow; long provides baseline.
    toxic_flow_detected = reg_long["toxic_flow_detected"] or reg_short["toxic_flow_detected"]
    spread_cv           = max(reg_long["spread_cv"], reg_short["spread_cv"])
    spread_stability    = min(reg_long["spread_stability"], reg_short["spread_stability"])
    adverse_sel_risk    = max(reg_long["adverse_sel_risk"], reg_short["adverse_sel_risk"])
    # Use short-window values for momentum (they react faster)
    trade_imbalance     = reg_short["trade_imbalance"] if len(snaps_short) >= 2 else reg_long["trade_imbalance"]
    quote_velocity_c_hr = reg_short["quote_velocity_c_hr"] if len(snaps_short) >= 3 else reg_long["quote_velocity_c_hr"]
    velocity_acceleration = reg_short["velocity_acceleration"] if len(snaps_short) >= 6 else reg_long["velocity_acceleration"]
    spread_change_c     = reg_short["spread_change_c"] if len(snaps_short) >= 2 else reg_long["spread_change_c"]
    spread_widening_suddenly = reg_short["spread_widening_suddenly"] or reg_long["spread_widening_suddenly"]

    # ── market death detection ────────────────────────────────────────────
    # Markets die suddenly. Short window must see activity or we flag death.
    market_dead = False
    spread_explosion = False

    if len(snaps_long) >= MARKET_DEATH_MIN_SNAPS_SHORT:
        # Had activity in the long window
        if len(snaps_short) < MARKET_DEATH_MIN_SNAPS_SHORT:
            market_dead = True
            log.info("MARKET DEATH: %s  short_n=%d (no activity in 5 min)",
                     token[-12:], len(snaps_short))
        elif reg_long["mean_spread_c"] > 0.001 and reg_short["mean_spread_c"] > 0:
            if reg_short["mean_spread_c"] >= SPREAD_EXPLOSION_MULT * reg_long["mean_spread_c"]:
                spread_explosion = True
                market_dead = True
                log.info("MARKET DEATH: %s  spread explosion %.1fc → %.1fc (%.1f×)",
                         token[-12:], reg_long["mean_spread_c"], reg_short["mean_spread_c"],
                         reg_short["mean_spread_c"] / reg_long["mean_spread_c"])

    if market_dead:
        toxic_flow_detected = True
        adverse_sel_risk = 1.0

    if toxic_flow_detected:
        log.info("TOXIC FLOW: imb=%.2f  accel=%.1f  spreadΔ=%.1fc  dead=%s",
                 trade_imbalance, velocity_acceleration, spread_change_c, market_dead)

    # Staleness risk (unchanged — from activity rate)
    lam = activity_per_hr * (scan_interval_s / 3600)
    staleness_risk_pct = round(1.0 - math.exp(-lam), 4)

    return {
        "spread_cv":             spread_cv,
        "spread_stability":      spread_stability,
        "trade_imbalance":       trade_imbalance,
        "quote_velocity_c_hr":   quote_velocity_c_hr,
        "velocity_acceleration": velocity_acceleration,
        "spread_change_c":       spread_change_c,
        "spread_widening_suddenly": spread_widening_suddenly,
        "toxic_flow_detected":   toxic_flow_detected,
        "adverse_sel_risk":      adverse_sel_risk,
        "staleness_risk_pct":    staleness_risk_pct,
        "market_dead":           market_dead,
        "spread_explosion":      spread_explosion,
        "n_snaps_short":         len(snaps_short),
        "n_snaps_long":          len(snaps_long),
    }


# ── data class ────────────────────────────────────────────────────────────────

@dataclass
class MarketOpp:
    # Identity
    question:         str   = ""
    slug:             str   = ""
    condition_id:     str   = ""   # for data-api trades filter
    token_yes:        str   = ""
    token_no:         str   = ""
    end_date:         str   = ""
    days_to_end:      float = 999.0

    # From Gamma API (no CLOB call needed)
    volume_24h:       float = 0.0
    mid_gamma:        float = 0.5     # mid price from outcomePrices
    best_bid_gamma:   float = 0.0     # Gamma bestBid field
    best_ask_gamma:   float = 1.0     # Gamma bestAsk field
    spread_cents_gamma: float = 0.0   # (ask - bid) * 100 from Gamma

    # From CLOB book (fetched only after Gamma pre-filter passes)
    best_bid:         float = 0.0
    best_ask:         float = 0.0
    spread_cents:     float = 0.0
    # L1
    bid_l1_price:     float = 0.0
    bid_l1_size:      float = 0.0
    bid_l1_depth:     float = 0.0    # bid_l1_price × bid_l1_size  ($)
    ask_l1_price:     float = 0.0
    ask_l1_size:      float = 0.0
    ask_l1_depth:     float = 0.0
    # L2
    bid_l2_price:     float = 0.0
    bid_l2_size:      float = 0.0
    bid_l2_depth:     float = 0.0
    ask_l2_price:     float = 0.0
    ask_l2_size:      float = 0.0
    ask_l2_depth:     float = 0.0
    bid_order_count:  int   = 0    # number of bid levels
    ask_order_count:  int   = 0
    bid_top3_ask_top3_usd: float = 0.0  # sum of top 3 levels each side ($)
    last_trade_ts:    float = 0.0  # Unix timestamp of most recent trade (0 = unknown)
    spread_avg_last_5: float = 0.0  # avg spread over last 5 scans (0 = no history)

    # Volume-based frequency estimate (fallback when history is thin)
    trades_per_hr_est: float = 0.0

    # Rolling 30-min activity from quote-move tracking
    quote_moves_30m:   int   = 0     # bid/ask changes observed in last 30 min
    activity_per_hr:   float = 0.0   # extrapolated rate (moves/hr)
    activity_source:   str   = "vol_est"  # "hist_high"|"hist_low"|"vol_est"
    snapshots_n:       int   = 0     # how many history points exist for this token

    # Market regime signals (computed from quote history)
    spread_cv:           float = 0.0   # coeff. of variation of spread
    spread_stability:    float = 1.0   # 1 − min(spread_cv, 1)
    trade_imbalance:     float = 0.0   # −1 … +1
    quote_velocity_c_hr: float = 0.0   # directional mid drift, c/hr
    velocity_acceleration: float = 0.0 # d(velocity)/dt, c/hr² (gapping)
    spread_change_c:     float = 0.0   # Δ spread this scan vs previous
    spread_widening_suddenly: bool = False  # spread jumped > 2c
    toxic_flow_detected: bool = False  # imb>0.6 + (accel OR spread widen) → PAUSE
    adverse_sel_risk:    float = 0.0   # 0–1 composite; 1.0 when toxic
    staleness_risk_pct:  float = 0.0   # P(quote moved since we last looked)
    regime_mult:         float = 1.0   # multiplier applied to base score
    market_dead:         bool = False # short window empty or spread explosion
    spread_explosion:    bool = False  # short-window spread >> long-window
    n_snaps_short:       int  = 0     # snapshots in last 5 min
    n_snaps_long:        int  = 0     # snapshots in last 30 min

    # Micro-fair-value (print proxy — mid is often fake on thin markets)
    fair_value:          float = 0.5   # print_ema when n_prints≥2, else mid
    print_ema:           float = 0.5   # EMA of inferred prints
    mid_stale:           bool = False # |print_ema - mid| > 2%
    stale_direction:     str  = ""    # "HIGH" | "LOW" | ""
    n_prints:            int  = 0     # inferred prints in window

    # Score and breakdown
    score:            float = 0.0
    score_spread:     float = 0.0
    score_freq:       float = 0.0
    score_volume:     float = 0.0
    score_l1:         float = 0.0
    score_l2:         float = 0.0
    score_zone:       float = 0.0

    gate_fail:        str   = ""    # non-empty = which hard gate failed

    # Correlation cluster (for exposure capping)
    cluster_id:       str   = "uncategorized"
    cluster_type:     str   = "other"

    # Market regime (24h baseline vs rolling activity)
    regime_change:    bool  = False  # rolling activity << 24h baseline
    size_multiplier:  float = 1.0    # 0.5 when degraded, 1.0 healthy

    # ── Event / competition grouping (from Gamma API) ────────────────────────
    event_slug:       str   = ""     # Gamma eventSlug — groups markets in same event

    # ── True probability estimate (prob_engine.py) ────────────────────────────
    p_est:            float = 0.5    # estimated true probability of YES
    p_confidence:     float = 0.0    # 0–1, how much we trust p_est
    edge_cents:       float = 0.0    # (p_est − mid) × 100, signed
    alpha_mode:       str   = "PASSIVE_MM"  # PASSIVE_MM | SKEWED_MM | DIRECTIONAL
    trade_vwap:       float = 0.0    # exp-weighted VWAP of recent trades
    n_recent_trades:  int   = 0      # number of trades used in estimate
    signal_breakdown: str   = ""     # human-readable signal summary for logging

    # ── Cross-market constraint engine (cross_market.py) ─────────────────────
    cross_edge_cents:  float = 0.0   # (p_fair - p_mid)*100 from group constraint
    cross_group_id:    str   = ""    # e.g. "bundesliga:TOP_N:4"
    cross_group_size:  int   = 0     # number of markets in constraint group
    cross_violation:   float = 0.0   # observed_sum - target_sum for the group

    # ── Fundamental / news (fundamental_sources.py) ───────────────────────────
    fund_p_adjust_cents: float = 0.0   # news-driven adjustment to p_est
    fund_direction:      str   = ""     # BULLISH | BEARISH | NEUTRAL
    fund_n_headlines:    int   = 0      # number of matched headlines


# ── Gamma API helpers ─────────────────────────────────────────────────────────

def _parse_tokens(m: dict) -> tuple[str, str]:
    raw = m.get("clobTokenIds", "")
    if isinstance(raw, str):
        raw = raw.strip().strip("[]").replace('"', "").replace("'", "")
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
    else:
        tokens = list(raw or [])
    return (tokens[0], tokens[1]) if len(tokens) >= 2 else \
           (tokens[0], "") if tokens else ("", "")


def _parse_mid(m: dict) -> float:
    raw = m.get("outcomePrices", "")
    if isinstance(raw, str) and raw.strip():
        try:
            return float(json.loads(raw)[0])
        except Exception:
            pass
    return 0.5


def _days_to_end(end_date_str: str) -> float:
    if not end_date_str:
        return 999.0
    try:
        ed = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return max(0.0, (ed - datetime.now(timezone.utc)).total_seconds() / 86400)
    except Exception:
        return 999.0


async def fetch_all_markets(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch all active markets from Gamma, sorted by volume desc."""
    all_markets = []
    for offset in range(0, 1200, 100):
        try:
            async with session.get(
                f"{GAMMA_API}/markets",
                params={
                    "active": "true", "closed": "false",
                    "limit": "100", "offset": str(offset),
                    "_sort": "volume24hr", "_order": "desc",
                },
            ) as r:
                if r.status != 200:
                    break
                batch = await r.json(content_type=None)
                if not batch:
                    break
                all_markets.extend(batch)
                if len(batch) < 100:
                    break
        except Exception as e:
            log.warning("Gamma page error offset=%d: %s", offset, e)
            break
    return all_markets


# ── CLOB book helpers ─────────────────────────────────────────────────────────

async def fetch_book(session: aiohttp.ClientSession,
                     token_id: str) -> Optional[dict]:
    try:
        async with session.get(
            f"{CLOB_API}/book", params={"token_id": token_id}
        ) as r:
            return await r.json(content_type=None) if r.status == 200 else None
    except Exception:
        return None


async def fetch_last_trade_ts(session: aiohttp.ClientSession,
                              condition_id: str) -> float:
    """Fetch most recent trade timestamp for a market. Returns 0 if unknown."""
    if not condition_id or not condition_id.startswith("0x"):
        return 0.0
    try:
        async with session.get(
            f"{DATA_API}/trades",
            params={"market": condition_id, "limit": 1},
        ) as r:
            if r.status != 200:
                return 0.0
            data = await r.json()
            if data and isinstance(data, list) and len(data) > 0:
                t = data[0]
                return float(t.get("timestamp", 0) or 0)
    except Exception:
        pass
    return 0.0


def _price(x) -> float:
    return float(x["price"] if isinstance(x, dict) else x[0])

def _size(x) -> float:
    return float(x["size"] if isinstance(x, dict) else x[1])


def _enrich_book(opp: MarketOpp, book: dict):
    """Parse CLOB book and fill all L1/L2 depth fields."""
    raw_bids = book.get("bids", [])
    raw_asks = book.get("asks", [])
    if not raw_bids or not raw_asks:
        return

    opp.bid_order_count = len(raw_bids)
    opp.ask_order_count = len(raw_asks)

    bids = sorted(raw_bids, key=lambda x: -_price(x))
    asks = sorted(raw_asks, key=lambda x:  _price(x))

    # L1
    opp.bid_l1_price = _price(bids[0])
    opp.bid_l1_size  = _size(bids[0])
    opp.bid_l1_depth = round(opp.bid_l1_price * opp.bid_l1_size, 2)
    opp.ask_l1_price = _price(asks[0])
    opp.ask_l1_size  = _size(asks[0])
    opp.ask_l1_depth = round(opp.ask_l1_price * opp.ask_l1_size, 2)

    # L2
    if len(bids) >= 2:
        opp.bid_l2_price = _price(bids[1])
        opp.bid_l2_size  = _size(bids[1])
        opp.bid_l2_depth = round(opp.bid_l2_price * opp.bid_l2_size, 2)
    if len(asks) >= 2:
        opp.ask_l2_price = _price(asks[1])
        opp.ask_l2_size  = _size(asks[1])
        opp.ask_l2_depth = round(opp.ask_l2_price * opp.ask_l2_size, 2)

    opp.best_bid     = opp.bid_l1_price
    opp.best_ask     = opp.ask_l1_price
    opp.spread_cents = round((opp.best_ask - opp.best_bid) * 100, 2)

    # Top 3 depth (USD) each side — pass if sum > 200
    bid_top3 = sum(_price(b) * _size(b) for b in bids[:3])
    ask_top3 = sum(_price(a) * _size(a) for a in asks[:3])
    opp.bid_top3_ask_top3_usd = round(bid_top3 + ask_top3, 2)


# ── Phase 1 pre-filter (Gamma data only, no CLOB call) ───────────────────────

def gamma_prefilter(m: dict) -> Optional[MarketOpp]:
    """
    Fast pre-filter using only Gamma API data.
    Returns a partial MarketOpp if the market passes, else None.
    Rejects ~95% of markets before any CLOB call.
    """
    tok_yes, tok_no = _parse_tokens(m)
    if not tok_yes or not tok_no:
        return None

    vol = m.get("volume24hr", 0) or 0
    if vol < GATE_MIN_VOLUME_24H:
        return None

    days = _days_to_end(m.get("endDate", ""))
    if days < GATE_MIN_DAYS_TO_END:
        return None

    mid = _parse_mid(m)
    if mid < GATE_MID_MIN or mid > GATE_MID_MAX:
        return None    # near-zero or near-certain — frozen market

    # Gamma returns bestBid / bestAsk directly — check spread without CLOB
    bb = float(m.get("bestBid", 0) or 0)
    ba = float(m.get("bestAsk", 1) or 1)
    spread_c = round((ba - bb) * 100, 2) if bb > 0 and ba < 1 else 0.0

    if spread_c < GATE_MIN_SPREAD_CENTS:
        return None

    # Also apply price zone gate to the actual bid/ask (not just mid)
    if bb < GATE_MID_MIN or ba > GATE_MID_MAX:
        return None

    q = m.get("question", "")[:80]
    slug = m.get("slug", "")
    cond_id = m.get("conditionId", "") or m.get("condition_id", "")
    event_slug = m.get("eventSlug", "") or m.get("event_slug", "") or ""
    cluster_id, cluster_type = tag_cluster(q, slug)

    opp = MarketOpp(
        question          = q,
        slug              = slug,
        condition_id      = cond_id,
        event_slug        = event_slug,
        token_yes         = tok_yes,
        token_no          = tok_no,
        end_date          = m.get("endDate", ""),
        days_to_end       = days,
        volume_24h        = vol,
        mid_gamma         = mid,
        best_bid_gamma    = bb,
        best_ask_gamma    = ba,
        spread_cents_gamma= spread_c,
        trades_per_hr_est = round(vol / AVG_TRADE_SIZE_USD / 24, 3),
        cluster_id        = cluster_id,
        cluster_type      = cluster_type,
    )
    return opp


# ── Phase 2 hard gates (after CLOB book) ─────────────────────────────────────

def passes_book_gates(opp: MarketOpp) -> bool:
    """
    Called after CLOB book is fetched.
    Rejects thin books that can't sustain market-making.
    """
    if opp.best_bid <= 0 or opp.best_ask >= 1:
        opp.gate_fail = "no valid book"
        return False

    # Edge: need |edge_cents| >= EDGE_MIN to quote
    if abs(opp.edge_cents) < EDGE_MIN_CENTS:
        opp.gate_fail = f"|edge|={abs(opp.edge_cents):.2f}c < {EDGE_MIN_CENTS}c"
        return False

    # Spread: current >= 1c OR spread_avg_last_5 >= 1c
    spread_ok = opp.spread_cents >= GATE_MIN_SPREAD_CENTS
    if not spread_ok and opp.spread_avg_last_5 >= GATE_SPREAD_AVG_LAST_5:
        spread_ok = True
    if not spread_ok:
        opp.gate_fail = (f"spread {opp.spread_cents:.1f}c (avg5={opp.spread_avg_last_5:.1f}c) "
                         f"< {GATE_MIN_SPREAD_CENTS}c")
        return False

    if opp.bid_l1_depth < GATE_MIN_L1_DEPTH:
        opp.gate_fail = f"bid L1 ${opp.bid_l1_depth:.0f} < ${GATE_MIN_L1_DEPTH}"
        return False
    if opp.ask_l1_depth < GATE_MIN_L1_DEPTH:
        opp.gate_fail = f"ask L1 ${opp.ask_l1_depth:.0f} < ${GATE_MIN_L1_DEPTH}"
        return False

    # Depth: order count >= 4 each side OR bid_top3+ask_top3 > 200
    depth_ok = (opp.bid_order_count >= GATE_MIN_ORDER_COUNT and
                opp.ask_order_count >= GATE_MIN_ORDER_COUNT)
    if not depth_ok and opp.bid_top3_ask_top3_usd > GATE_MIN_TOP3_DEPTH_USD:
        depth_ok = True
    if not depth_ok:
        opp.gate_fail = (f"depth: orders bid={opp.bid_order_count} ask={opp.ask_order_count}, "
                         f"top3=${opp.bid_top3_ask_top3_usd:.0f} (need 4+ each or >${GATE_MIN_TOP3_DEPTH_USD})")
        return False

    # Last trade: allow if within 120s (fail open if unknown)
    if opp.last_trade_ts > 0:
        age = time.time() - opp.last_trade_ts
        if age > GATE_MIN_LAST_TRADE_S:
            opp.gate_fail = f"last trade {age:.0f}s ago > {GATE_MIN_LAST_TRADE_S}s"
            return False
    return True


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_market(opp: MarketOpp):
    """
    Score = spread × volume × activity. Prioritizes highly liquid markets.

    Tight spread = high activity. Ignore tiny markets (volume gate = $5k).
    Multiplicative: need all three — wide spread alone is worthless without fills.
    """
    if opp.toxic_flow_detected:
        opp.regime_mult = 0.0
        opp.score       = 0.0
        return

    # spread (cents), volume ($/day), activity (trades/hr)
    spread = max(0.1, opp.spread_cents)
    volume = opp.volume_24h / 1000.0   # scale to thousands
    activity = max(0.1, opp.activity_per_hr)

    base = spread * volume * activity

    # Regime multiplier — penalise adverse/turbulent markets
    mult = (1.0 - opp.adverse_sel_risk * 0.35) * (0.75 + opp.spread_stability * 0.25)
    opp.regime_mult = round(mult, 4)
    opp.score       = round(base * mult, 1)

    # Keep breakdown for display (optional)
    opp.score_spread  = spread
    opp.score_volume  = volume
    opp.score_freq    = activity
    opp.score_l1      = min(opp.bid_l1_depth, opp.ask_l1_depth)
    opp.score_l2      = 0.0
    opp.score_zone    = 0.0


# ── Full scan pipeline ────────────────────────────────────────────────────────

async def scan_all(session: aiohttp.ClientSession,
                   top_n: int,
                   history: Optional[QuoteHistoryStore] = None,
                   print_ema_store: Optional[PrintEMAStore] = None,
                   regime_watcher: Optional["MarketRegimeWatcher"] = None) -> list[MarketOpp]:
    """
    Phase 1: Gamma API — fetch all markets, pre-filter (spread, zone, vol, days)
    Phase 2: CLOB book — fetch book only for pre-filter survivors
              After each book fetch: record snapshot, compute rolling activity,
              micro-fair-value (print proxy), dual-window regime, regime-change check
    Phase 3: Gate check on book data, score, rank, return top N
    """
    if history is None:
        history = {}
    if print_ema_store is None:
        print_ema_store = {}
    # ── phase 1: Gamma pre-filter ──────────────────────────────────────────
    raw = await fetch_all_markets(session)
    log.info("Gamma: %d markets fetched", len(raw))

    pre: list[MarketOpp] = []
    for m in raw:
        opp = gamma_prefilter(m)
        if opp:
            pre.append(opp)

    log.info("Phase-1 pre-filter: %d candidates passed "
             "(spread≥%dc, zone %.2f–%.2f, vol≥$%d, days≥%d)",
             len(pre), GATE_MIN_SPREAD_CENTS, GATE_MID_MIN, GATE_MID_MAX,
             GATE_MIN_VOLUME_24H, GATE_MIN_DAYS_TO_END)

    if not pre:
        log.warning("No markets passed Phase-1 filter")
        return []

    # ── sampling: prioritize liquid markets ──────────────────────────────
    # Sort by volume descending — fetch books for highest-volume markets first.
    # Score = spread × volume × activity; liquid markets get priority.
    pre_sorted = sorted(pre, key=lambda o: -o.volume_24h)
    candidates = pre_sorted[:35]
    # Deduplicate
    seen: set = set()
    unique = []
    for c in candidates:
        if c.token_yes not in seen:
            seen.add(c.token_yes)
            unique.append(c)

    log.info("Phase-2 CLOB fetch: %d candidates (highest volume first)",
             len(unique))

    # ── phase 2: fetch CLOB book + record activity snapshot ───────────────
    enriched: list[MarketOpp] = []
    for opp in unique:
        book = await fetch_book(session, opp.token_yes)
        if book:
            _enrich_book(opp, book)
        else:
            opp.best_bid     = opp.best_bid_gamma
            opp.best_ask     = opp.best_ask_gamma
            opp.spread_cents = opp.spread_cents_gamma

        # Record snapshot with L1 sizes for queue-position and toxic-flow logic.
        if opp.best_bid > 0 and opp.best_ask < 1:
            record_snapshot(history, opp.token_yes, opp.best_bid, opp.best_ask,
                           opp.bid_l1_size, opp.ask_l1_size)

        # spread_avg_last_5_scans: if >= 1.5c, quote market
        snaps = history.get(opp.token_yes, [])
        if len(snaps) >= 5:
            last5 = list(snaps)[-5:]
            opp.spread_avg_last_5 = round(
                sum(s.spread_c for s in last5) / len(last5), 2
            )

        # True probability estimation + last trade recency
        if opp.condition_id:
            trades = await fetch_recent_trades(session, opp.condition_id, n=20)
            opp.last_trade_ts = trades[0]["ts"] if trades else 0.0

            prob = estimate_true_probability(
                trades          = trades,
                print_ema       = opp.print_ema,
                n_prints        = opp.n_prints,
                trade_imbalance = opp.trade_imbalance,
                bid_l1_size     = opp.bid_l1_size,
                ask_l1_size     = opp.ask_l1_size,
                current_bid     = opp.best_bid,
                current_ask     = opp.best_ask,
            )
            opp.p_est            = prob.p_est
            opp.p_confidence     = prob.confidence
            opp.edge_cents       = prob.edge_cents
            opp.alpha_mode       = prob.alpha_mode
            opp.trade_vwap       = prob.trade_vwap
            opp.n_recent_trades  = prob.n_trades
            opp.signal_breakdown = prob.signal_breakdown

            if abs(prob.edge_cents) >= 3:
                log.info("EDGE %-45s  p_est=%.3f  mid=%.3f  edge=%+.1fc  "
                         "mode=%-12s conf=%.2f  [%s]",
                         opp.question[:45],
                         opp.p_est, (opp.best_bid + opp.best_ask) / 2,
                         opp.edge_cents, opp.alpha_mode,
                         opp.p_confidence, opp.signal_breakdown[:60])

        await asyncio.sleep(0.1)  # light throttle for data-api

        # Rolling activity
        moves, rate_hr, source = compute_activity(history, opp.token_yes)
        opp.quote_moves_30m  = moves
        opp.activity_per_hr  = rate_hr if source != "vol_est" else opp.trades_per_hr_est
        opp.activity_source  = source
        opp.snapshots_n      = len(history.get(opp.token_yes, []))

        # Market regime (dual-window: short 5min + long 30min)
        regime = compute_market_regime(
            history, opp.token_yes,
            activity_per_hr  = opp.activity_per_hr,
            scan_interval_s  = 60.0,
        )
        opp.spread_cv             = regime["spread_cv"]
        opp.spread_stability      = regime["spread_stability"]
        opp.trade_imbalance       = regime["trade_imbalance"]
        opp.quote_velocity_c_hr   = regime["quote_velocity_c_hr"]
        opp.velocity_acceleration = regime["velocity_acceleration"]
        opp.spread_change_c       = regime["spread_change_c"]
        opp.spread_widening_suddenly = regime["spread_widening_suddenly"]
        opp.toxic_flow_detected   = regime["toxic_flow_detected"]
        opp.adverse_sel_risk      = regime["adverse_sel_risk"]
        opp.staleness_risk_pct    = regime["staleness_risk_pct"]
        opp.market_dead           = regime.get("market_dead", False)
        opp.spread_explosion      = regime.get("spread_explosion", False)
        opp.n_snaps_short         = regime.get("n_snaps_short", 0)
        opp.n_snaps_long          = regime.get("n_snaps_long", 0)

        # Micro-fair-value (print proxy — mid is often fake on thin markets)
        fair = compute_micro_fair_value(
            history, opp.token_yes,
            opp.best_bid, opp.best_ask,
            print_ema_store,
        )
        opp.fair_value      = fair["fair_value"]
        opp.print_ema       = fair["print_ema"]
        opp.mid_stale       = fair["mid_stale"]
        opp.stale_direction = fair["stale_direction"]
        opp.n_prints        = fair["n_prints"]

        # Correlation cluster — used by run_mm to cap per-cluster exposure.
        opp.cluster_id, opp.cluster_type = tag_cluster(opp.question, opp.slug)

        # 24h vs rolling activity regime comparison
        if regime_watcher is not None:
            newly_degraded, size_mult = regime_watcher.update(
                token                 = opp.token_yes,
                activity_per_hr       = opp.activity_per_hr,
                trades_per_hr_baseline= opp.trades_per_hr_est,
                spread_cv             = opp.spread_cv,
            )
            opp.regime_change   = newly_degraded or regime_watcher.get_mult(opp.token_yes) < 1.0
            opp.size_multiplier = size_mult
        else:
            opp.regime_change   = False
            opp.size_multiplier = 1.0

        enriched.append(opp)
        await asyncio.sleep(BOOK_DELAY)

    # ── phase 2.5: fundamental news (cache only — no fetch, <1ms) ───────────
    try:
        fund_signals = get_news_for_markets(enriched)
        for opp in enriched:
            if opp.slug not in fund_signals:
                continue
            sig = fund_signals[opp.slug]
            mid = (opp.best_bid + opp.best_ask) / 2 if opp.best_bid > 0 else opp.mid_gamma
            half_spread_c = (opp.best_ask - opp.best_bid) * 50 if opp.best_bid > 0 else 0.5
            p_new, edge_new = merge_fundamental_into_prob(
                opp.p_est, opp.edge_cents, sig, half_spread_c,
            )
            opp.p_est = p_new
            opp.edge_cents = edge_new
            opp.fund_p_adjust_cents = round(sig.p_adjustment_cents * sig.confidence, 2)
            opp.fund_direction = sig.direction
            opp.fund_n_headlines = sig.n_headlines
            # Re-check alpha_mode after fundamental merge
            if abs(edge_new) >= 15 and opp.alpha_mode == "SKEWED_MM":
                opp.alpha_mode = "DIRECTIONAL"
            elif abs(edge_new) >= 3 and opp.alpha_mode == "PASSIVE_MM":
                opp.alpha_mode = "SKEWED_MM"
            if sig.n_headlines > 0:
                log.info("FUND %-45s %s adj=%+.1fc => p=%.3f edge=%+.1fc [%d headlines]",
                         opp.question[:45], sig.direction, opp.fund_p_adjust_cents,
                         opp.p_est, opp.edge_cents, sig.n_headlines)
    except Exception as e:
        log.debug("Fundamental fetch failed: %s", e)

    # ── phase 2.6: event-time dynamics — scale edge by time to resolution ───
    # Far-future events: less confident. edge *= exp(-days_to_event / 60)
    for opp in enriched:
        scale = math.exp(-opp.days_to_end / 60)
        opp.edge_cents = round(opp.edge_cents * scale, 2)
        # Re-check alpha_mode after time scaling
        if abs(opp.edge_cents) >= 15 and opp.alpha_mode == "SKEWED_MM":
            opp.alpha_mode = "DIRECTIONAL"
        elif abs(opp.edge_cents) >= 3 and opp.alpha_mode == "PASSIVE_MM":
            opp.alpha_mode = "SKEWED_MM"
        elif abs(opp.edge_cents) < 3:
            opp.alpha_mode = "PASSIVE_MM"

    # ── phase 3: gate + score ──────────────────────────────────────────────
    passed = [o for o in enriched if passes_book_gates(o)]
    failed = [o for o in enriched if o.gate_fail]

    log.info("Phase-2 gates: %d passed, %d failed", len(passed), len(failed))
    if failed:
        for o in sorted(failed, key=lambda x: -x.spread_cents_gamma)[:5]:
            log.info("  FAIL %-45s %s", o.question[:45], o.gate_fail)

    for o in passed:
        score_market(o)

    # ── phase 4: cross-market constraint analysis ──────────────────────────
    # Run over ALL enriched markets (not just passed) so we can detect
    # violations using the full group even when some members failed gates.
    # Edge is only applied to markets that passed gates.
    all_for_cross = enriched  # includes failed-gate markets as reference points
    groups = run_cross_market_analysis(all_for_cross)
    if groups:
        n_upgraded = sum(1 for o in passed if o.alpha_mode != "PASSIVE_MM"
                         and o.cross_edge_cents != 0)
        log.info("Cross-market: %d constraint groups  %d markets with edge",
                 len(groups), n_upgraded)

    ranked = sorted(passed, key=lambda o: -o.score)
    return ranked[:top_n]


# ── Display ───────────────────────────────────────────────────────────────────

_BAR  = "═"
_DIV  = "─"
W     = 118

def display(top: list[MarketOpp], cycle: int, elapsed: float):
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    print("\033[2J\033[H", end="")
    print(_BAR * W)
    print(f"  POLYMARKET SCANNER  │  cycle {cycle}  │  {now}  │  "
          f"scan {elapsed:.1f}s  │  top {len(top)}/{DEFAULT_TOP}")
    print(_BAR * W)

    if not top:
        print("\n  No markets passed all gates this cycle.\n")
        print(f"  Hard gates:")
        print(f"    spread     ≥ {GATE_MIN_SPREAD_CENTS}c (or avg5 ≥ {GATE_SPREAD_AVG_LAST_5}c)")
        print(f"    price zone {GATE_MID_MIN:.2f} – {GATE_MID_MAX:.2f}")
        print(f"    volume     ≥ ${GATE_MIN_VOLUME_24H}/day")
        print(f"    L1 depth   ≥ ${GATE_MIN_L1_DEPTH} each side")
        print(f"    depth     4+ orders each OR top3 > ${GATE_MIN_TOP3_DEPTH_USD}")
        print(f"    last trade within {GATE_MIN_LAST_TRADE_S}s")
        print(f"    expiry     ≥ {GATE_MIN_DAYS_TO_END} days out")
        print()
        return

    # Activity source legend
    # ★ = history high-confidence (≥15 scans = ≥15 min of data)
    # ◐ = history low-confidence  (5–14 scans)
    # ~ = volume estimate         (< 5 scans, first few cycles)
    def _src_icon(o: MarketOpp) -> str:
        return {"hist_high": "★", "hist_low": "◐", "vol_est": "~"}.get(
            o.activity_source, "~")

    # ── table header ──────────────────────────────────────────────────────
    print(f"\n  {'RK':>2}  {'SCORE':>5}  "
          f"{'SPREAD':>7}  {'BID':>5}  {'ASK':>5}  "
          f"{'VOL24$':>8}  {'ACT/HR':>7}  {'MV30M':>6}  "
          f"{'BID L1$':>8}  {'ASK L1$':>8}  "
          f"{'BID L2$':>8}  {'ASK L2$':>8}  "
          f"{'DAYS':>5}  {'CLUSTER':>16}  QUESTION")
    print(f"  {'─' * 6}  {'─'*5}  "
          f"{'─'*7}  {'─'*5}  {'─'*5}  "
          f"{'─'*8}  {'─'*7}  {'─'*6}  "
          f"{'─'*8}  {'─'*8}  "
          f"{'─'*8}  {'─'*8}  "
          f"{'─'*5}  {'─'*16}  {'─'*38}")

    for i, o in enumerate(top, 1):
        icon = _src_icon(o)
        cluster_label = f"{o.cluster_id}/{o.cluster_type}"
        regime_tag = f"×{o.size_multiplier:.1f}" if o.size_multiplier < 1.0 else "  ok"
        print(
            f"  {i:2d}  {o.score:5.1f}  "
            f"{o.spread_cents:6.1f}c  {o.best_bid:5.3f}  {o.best_ask:5.3f}  "
            f"${o.volume_24h:>7,.0f}  {o.activity_per_hr:6.2f}{icon}  "
            f"{o.quote_moves_30m:>5d}  "
            f"${o.bid_l1_depth:>7,.0f}  ${o.ask_l1_depth:>7,.0f}  "
            f"${o.bid_l2_depth:>7,.0f}  ${o.ask_l2_depth:>7,.0f}  "
            f"{o.days_to_end:>5.0f}  {cluster_label:>16}  "
            f"{regime_tag}  {o.question[:36]}"
        )

    print(f"\n  Activity key: ★=measured≥15scans  ◐=measured<15scans  ~=vol-estimate"
          f"  |  Regime: ×0.5=degraded (24h↓)  ok=healthy")

    # ── score breakdown for #1 ─────────────────────────────────────────────
    best = top[0]
    print(f"\n  ── Score breakdown #1: {best.question[:60]}")
    src_label = (f"({best.activity_source}, n={best.snapshots_n})")
    print(f"     Spread  {best.score_spread:5.1f}/35  "
          f"Freq {src_label}  {best.score_freq:5.1f}/25  "
          f"Vol    {best.score_volume:5.1f}/20  "
          f"L1 {best.score_l1:5.1f}/10  "
          f"L2 {best.score_l2:5.1f}/5  "
          f"Zone {best.score_zone:5.1f}/5  "
          f"→ TOTAL {best.score:.1f}/100")

    # ── rolling activity detail ────────────────────────────────────────────
    print(f"\n  ── Rolling 30-min activity (quote moves = proxy for trades):")
    for i, o in enumerate(top, 1):
        icon   = _src_icon(o)
        bar_w  = min(int(o.quote_moves_30m * 2), 30)
        bar    = "█" * bar_w
        note   = (f"n={o.snapshots_n} scans" if o.activity_source != "vol_est"
                  else "vol÷$40÷24")
        print(f"  [{i}] {o.activity_per_hr:5.2f}/hr {icon}  "
              f"moves={o.quote_moves_30m:3d}  {bar:<30s}  "
              f"{note}  │  {o.question[:38]}")

    # ── regime signals ─────────────────────────────────────────────────────
    def _risk_icon(v: float, warn: float = 0.30, danger: float = 0.60) -> str:
        return "✗" if v >= danger else "⚠" if v >= warn else "✓"

    def _imb_label(v: float) -> str:
        if v >  0.50: return f"+{v:.2f} BUY↑"
        if v < -0.50: return f"{v:.2f} SELL↓"
        return f"{v:+.2f} balanced"

    # ── micro fair value + regime  ────────────────────────────────────────
    print(f"\n  ── Micro-fair-value  (★=stale mid  ✗=toxic/dead  ◐=warn)")
    print(f"  {'RK':>2}  {'MID':>6}  {'FAIR':>6}  {'STALE':>5}  "
          f"{'NPRINTS':>7}  {'TOXIC':>7}  {'DEAD':>4}  "
          f"{'AS':>5}  {'n5m':>4} {'n30m':>5}  "
          f"{'ACCEL':>7}  {'ΔSPR':>5}  {'IMB':>12}  "
          f"{'VEL c/hr':>9}  QUESTION")
    print(f"  {'─'*2}  {'─'*6}  {'─'*6}  {'─'*5}  "
          f"{'─'*7}  {'─'*7}  {'─'*4}  "
          f"{'─'*5}  {'─'*4} {'─'*5}  "
          f"{'─'*7}  {'─'*5}  {'─'*12}  "
          f"{'─'*9}  {'─'*36}")
    for i, o in enumerate(top, 1):
        mid        = (o.best_bid + o.best_ask) / 2 if o.best_bid > 0 else o.mid_gamma
        stale_tag  = f"★{o.stale_direction}" if o.mid_stale else "  -"
        toxic_tag  = "✗TOXIC" if o.toxic_flow_detected else "  -"
        dead_tag   = "✗" if o.market_dead else "-"
        print(
            f"  {i:2d}  {mid:6.3f}  {o.fair_value:6.3f}  {stale_tag:>5}  "
            f"{o.n_prints:7d}  {toxic_tag:>7}  {dead_tag:>4}  "
            f"{o.adverse_sel_risk:5.2f}  {o.n_snaps_short:4d} {o.n_snaps_long:5d}  "
            f"{o.velocity_acceleration:>+7.1f}  "
            f"{o.spread_change_c:>+4.1f}c  "
            f"{_imb_label(o.trade_imbalance):>12s}  "
            f"{o.quote_velocity_c_hr:>+8.2f}  "
            f"{o.question[:36]}"
        )

    # ── quote recommendations ──────────────────────────────────────────────
    print(f"\n  ── Recommended quotes  (fair_value-adjusted when mid is stale):")
    for i, o in enumerate(top, 1):
        mid = (o.best_bid + o.best_ask) / 2 if o.best_bid > 0 else 0.5
        # When mid is stale, shift quotes toward the fair value so we don't
        # post a two-sided quote centred on a price nobody will trade at.
        if o.mid_stale and o.n_prints >= 2:
            shift     = round(o.fair_value - mid, 3)
            our_bid   = round(o.best_bid + 0.01 + shift, 3)
            our_ask   = round(o.best_ask - 0.01 + shift, 3)
            fv_note   = f"  [fv-adj shift={shift:+.3f} dir={o.stale_direction}]"
        else:
            our_bid   = round(o.best_bid + 0.01, 3)
            our_ask   = round(o.best_ask - 0.01, 3)
            fv_note   = ""
        our_spread = round(our_ask - our_bid, 3)
        if our_spread < 0.01 or our_bid <= 0 or our_ask >= 1:
            rec = "  [spread too narrow / out of range after adjustment — skip]"
        else:
            profit = round(our_spread * 15, 2)
            rec    = (f"  bid={our_bid:.3f}  ask={our_ask:.3f}  "
                      f"spread={our_spread*100:.1f}c  profit/fill=${profit:.2f}{fv_note}")
        print(f"  [{i}]{rec}  │  {o.question[:40]}")

    print()


# ── Persistence ───────────────────────────────────────────────────────────────

def save(top: list[MarketOpp], cycle: int, prev_slugs: set) -> set:
    os.makedirs(LOG_DIR, exist_ok=True)

    data = {
        "cycle": cycle,
        "ts":    time.time(),
        "iso":   datetime.now(timezone.utc).isoformat(),
        "top":   [asdict(o) for o in top],
    }
    with open(TOP_LOG, "w") as f:
        json.dump(data, f, indent=2)

    current_slugs = {o.slug for o in top}
    new = current_slugs - prev_slugs

    if new:
        with open(ALERTS_LOG, "a") as f:
            for o in top:
                if o.slug in new:
                    alert = {
                        "ts":              time.time(),
                        "slug":            o.slug,
                        "question":        o.question,
                        "score":           o.score,
                        "spread_cents":    o.spread_cents,
                        "vol_24h":         o.volume_24h,
                        "activity_per_hr": o.activity_per_hr,
                        "activity_source": o.activity_source,
                        "quote_moves_30m": o.quote_moves_30m,
                        "snapshots_n":     o.snapshots_n,
                        "bid_l1":          o.bid_l1_depth,
                        "ask_l1":          o.ask_l1_depth,
                        "bid_l2":          o.bid_l2_depth,
                        "ask_l2":          o.ask_l2_depth,
                        "days_to_end":     o.days_to_end,
                    }
                    f.write(json.dumps(alert) + "\n")
                    log.info(
                        "NEW → %-50s score=%.1f  spread=%.1fc  "
                        "freq=%.2f/hr  L1=($%.0f/$%.0f)",
                        o.question[:50], o.score, o.spread_cents,
                        o.trades_per_hr_est, o.bid_l1_depth, o.ask_l1_depth,
                    )

    return current_slugs


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run(interval: int, top_n: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    prev_slugs: set              = set()
    history: QuoteHistoryStore   = {}
    print_ema_store: PrintEMAStore = {}
    regime_watcher               = MarketRegimeWatcher()
    cycle = 0

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=SSL_CTX, limit=8),
        timeout=aiohttp.ClientTimeout(total=30),
        headers={"User-Agent": "poly-scanner/1.0"},
    ) as session:
        while True:
            cycle += 1
            start = time.time()
            try:
                top     = await scan_all(session, top_n, history,
                                        print_ema_store, regime_watcher)
                elapsed = time.time() - start
                display(top, cycle, elapsed)
                prev_slugs = save(top, cycle, prev_slugs)

                # Prune stale tokens from history + print_ema_store
                active_tokens = {o.token_yes for o in top}
                stale = [t for t in history if t not in active_tokens and
                         len(history[t]) > 0 and
                         time.time() - history[t][-1].ts > HISTORY_WINDOW_S * 2]
                for t in stale:
                    del history[t]
                    print_ema_store.pop(t, None)
                if stale:
                    log.debug("Pruned %d stale tokens from history", len(stale))

            except Exception as e:
                log.error("Cycle %d error: %s", cycle, e, exc_info=True)

            sleep = max(5, interval - (time.time() - start))
            log.info("Next scan in %.0fs  |  history tokens: %d", sleep, len(history))
            await asyncio.sleep(sleep)


def main():
    p = argparse.ArgumentParser(description="Polymarket opportunity scanner")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    p.add_argument("--top",      type=int, default=DEFAULT_TOP)
    p.add_argument("--debug",    action="store_true")
    args = p.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"Scanner starting  interval={args.interval}s  top={args.top}")
    print(f"Hard gates: spread≥{GATE_MIN_SPREAD_CENTS}c | "
          f"zone {GATE_MID_MIN:.2f}–{GATE_MID_MAX:.2f} | "
          f"vol≥${GATE_MIN_VOLUME_24H} | L1≥${GATE_MIN_L1_DEPTH} | "
          f"days≥{GATE_MIN_DAYS_TO_END}")
    print(f"Output: {TOP_LOG}  alerts: {ALERTS_LOG}\n")

    try:
        asyncio.run(run(args.interval, args.top))
    except KeyboardInterrupt:
        print("\nScanner stopped.")


# ── Exported symbols for run_mm.py ───────────────────────────────────────────

def parse_tokens(m: dict) -> tuple[str, str]:
    return _parse_tokens(m)

def parse_prices(m: dict) -> tuple[float, float]:
    mid = _parse_mid(m)
    return mid, 1.0 - mid


if __name__ == "__main__":
    main()
