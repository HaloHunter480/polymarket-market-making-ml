"""
market_maker.py — Multi-Market Passive Market Maker for Polymarket
===================================================================

Core strategy:
  For each wide-spread market the scanner identifies:
    1. Post BID (buy YES) at best_bid + 1 tick
    2. Post ASK (sell YES) at best_ask - 1 tick
    3. Both legs fill → profit = spread - 2 ticks
    4. One leg fills  → inventory management

New defensive layers added on top of the basic MM loop:

  AdverseSelectionGuard  — inspects scanner regime signals before quoting
    NORMAL  → post 1 tick inside (standard)
    WIDEN   → post at best bid/ask (no narrowing, protect margin)
    PAUSE   → skip market entirely (too risky right now)

  StalenessGuard         — adds extra ticks when the market moves faster
    than our scan interval, reducing the chance of being picked off on
    a stale quote.

  InventorySkewController — when one leg fills, time-skews the remaining
    quote price to accelerate exit and reduce mark-to-market exposure.

  PostFillDriftMonitor    — tracks mid-price after one leg fills.  If the
    price drifts adversely beyond ADVERSE_THRESHOLD_C, triggers an
    immediate force exit rather than waiting for the other leg.
"""

import json
import math
import os
import random
import time
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger("mm")


# ── config ────────────────────────────────────────────────────────────────────

TICK                 = 0.01   # Polymarket minimum tick
MIN_SPREAD_TO_QUOTE  = 1.5    # skip markets < 1.5 cents spread (align with scanner)
QUOTE_INSIDE_TICKS   = 1      # default: post 1 tick inside best bid/ask
MAX_INVENTORY_AGE_S  = 600    # 10 min hold → force exit at market
HEDGE_CROSS_SPREAD_S = 20     # if filled, no hedge in 20s → cross spread (market order)
REQUOTE_INTERVAL_S   = 2      # check every 2s — provide liquidity, reprice constantly
MAX_QUOTE_AGE_S      = 20     # cancel & refresh if quote stays > 20s without fill
MID_MOVE_CANCEL      = 0.005  # cancel when |mid - our_mid| > 0.5c
MAX_LOSS_PER_MARKET  = 5.0    # USD hard stop per market
SHARES_PER_QUOTE     = 15     # shares per side (Polymarket min = 15)
MIN_QUOTE_SIZE       = 8     # Polymarket enforces minimum 8 shares per order
MIN_QUOTE_SIZE_SMALL = 8     # same — cannot go below 8
MAX_MARKETS          = 10     # top 5–10 markets, quote all in same cycle
TARGET_CAPITAL_PER_MARKET = 1.5  # $1–2 per market when bankroll is small
MIN_ORDER_USD        = 0.20   # min order value (Polymarket still enforces 8 shares)

# ── AdverseSelectionGuard thresholds ─────────────────────────────────────────
AS_RISK_WIDEN        = 0.50   # relaxed from 0.40 — widen to best bid/ask
AS_RISK_PAUSE        = 0.85   # relaxed from 0.70 — skip only at very high risk
IMBALANCE_WIDEN      = 0.75   # relaxed from 0.65 — need stronger imbalance to widen

# ── StalenessGuard thresholds ─────────────────────────────────────────────────
STALENESS_MAX_EXTRA  = 3      # up to 3 extra ticks inside spread when stale

# ── InventorySkewController params ───────────────────────────────────────────
SKEW_PER_SHARE       = 0.002  # skew = inventory * 0.2 (cents) → 0.002 price per share
MAX_SKEW             = 0.02   # cap at 2 c skew

# ── Quote bias (SKEWED_MM) ───────────────────────────────────────────────────
# Instead of symmetric around p_est, shift toward the side we want filled.
# Example: p_est=0.60, symmetric bid=0.58 ask=0.62 → biased bid=0.57 ask=0.61 (sell)
# or bid=0.59 ask=0.63 (buy). 1 cent shift per side.
QUOTE_BIAS_CENTS     = 1.0    # shift bid/ask by this when we have edge

# ── Queue priority: step ahead of the touch ────────────────────────────────────
# Matching best_bid puts you behind 50k shares (FIFO). Step 1 tick inside → front of queue.
# bid = best_bid + step, ask = best_ask - step. Fill rate >> spread capture.
# 0.1c step needs tick_size=0.001 markets; 1c step uses TICK (0.01).
STEP_AHEAD_CENTS     = 1.0    # improve by this many cents to jump queue (0.1 = sub-tick)

# ── Multi-level quoting (Citadel style) ───────────────────────────────────────
# bid1, bid2, bid3 / ask1, ask2, ask3 — more chances to fill
QUOTE_LEVELS         = 3      # levels per side (1 = single level, 3 = ladder)

# ── PostFillDriftMonitor params ───────────────────────────────────────────────
ADVERSE_THRESHOLD_C  = 3.0    # 3 c adverse drift → log warning
EMERGENCY_EXIT_C     = 5.0    # 5 c adverse drift → force exit now
DRIFT_SAMPLE_S       = 20     # sample interval for drift tracking

# ── Per-market MTM loss cutoff ───────────────────────────────────────────────
# Auto-pause (force exit) if unrealized loss on a single ONE_LEG position
# exceeds this % of position value.  Per market, not daily.
MTM_LOSS_CUTOFF_PCT  = 5.0    # e.g. 5% = exit if bid filled and mid dropped 5%

# ── Max deployment cap ───────────────────────────────────────────────────────
# Only stop when deployed >= bankroll (prediction markets are slow)
MAX_DEPLOYMENT_PCT   = 100.0  # deploy up to 100%; skip only when fully deployed

# ── Order sizing (bankroll-based) ─────────────────────────────────────────────
ORDER_SIZE_PCT       = 0.05   # order_size = bankroll * 5%
ORDER_SIZE_CAP_PCT   = 0.20   # hard cap: min(order_size, bankroll * 20%)
# Liquidity-based (provide liquidity goal): size = min(10, liquidity_usd * 0.05 / mid)
SIZE_LIQUIDITY_PCT   = 0.05   # 5% of L1 depth
MAX_QUOTE_SIZE       = 10     # cap shares per side

# ── Max notional per market ───────────────────────────────────────────────────
# Cap single-market exposure (bid_price + 1-ask_price) * shares.
MAX_NOTIONAL_PER_MARKET = 100.0  # USD per quote

# ── Alpha / directional mode ──────────────────────────────────────────────────
DIRECTIONAL_KELLY_FRAC   = 0.20   # fraction of bankroll to Kelly-size directional bets
MAX_DIRECTIONAL_SHARES   = 25     # hard cap on directional order size

# ── Real fill analytics log ──────────────────────────────────────────────────
FILL_ANALYTICS_LOG   = os.path.join(os.environ.get("ARBPOLY_LOG_DIR", "logs"),
                                    "fill_analytics.jsonl")

# ── Monitoring alerts (toxic fills, large losses) ─────────────────────────────
ALERTS_LOG           = os.path.join(os.environ.get("ARBPOLY_LOG_DIR", "logs"),
                                    "mm_alerts.jsonl")
LARGE_LOSS_THRESHOLD = 2.0   # log alert when single-fill loss exceeds $X

# ── Time-to-resolution scaling ───────────────────────────────────────────────
# Liquidity dies before resolution. Auto-reduce size as market nears close.
# days_to_end >= 14: full size; 7–14: 80%; 3–7: 60%; 1–3: 40%; <1: 25%
RESOLUTION_SCALE_TIERS: list[tuple[float, float]] = [
    (14.0, 1.00),   # days >= 14: 100%
    (7.0,  0.80),   # 7–14: 80%
    (3.0,  0.60),   # 3–7: 60%
    (1.0,  0.40),   # 1–3: 40%
    (0.0,  0.25),   # <1: 25%
]


def resolution_multiplier(days_to_end: float) -> float:
    """Return size multiplier (0.25–1.0) based on days until resolution."""
    for threshold, mult in RESOLUTION_SCALE_TIERS:
        if days_to_end >= threshold:
            return mult
    return RESOLUTION_SCALE_TIERS[-1][1]

# ── Fees (Polymarket: makers 0%, takers 0.10% on US; conservative model) ───────
# Every fill PnL is NET = gross − fees.  No gross PnL delusion.
FEE_MAKER_PCT        = 0.0    # we post limit orders = maker; Polymarket 0%
FEE_TAKER_PCT        = 0.001  # 0.1% if we ever take; conservative
FEE_DEFAULT_PCT      = 0.0    # applied when maker/taker unknown (assume maker)


def alert_event(event: str, pnl: float) -> None:
    """Log kill_switch or daily_loss to ALERTS_LOG and optional webhook."""
    rec = type("_Rec", (), {"slug": "", "attribution": "", "spread_capture_pct": 0.0, "worst_drift_c": 0.0})()
    _alert_monitor(rec, pnl, event)


def _alert_monitor(rec: "FillRecord", pnl: float, event: str = "fill") -> None:
    """
    Log toxic fills and large losses to ALERTS_LOG. Optionally POST to
    ALERT_WEBHOOK_URL (Discord/Telegram) when kill switch or daily loss.
    """
    entry = {
        "ts": time.time(),
        "event": event,
        "slug": getattr(rec, "slug", ""),
        "attribution": getattr(rec, "attribution", ""),
        "pnl": round(pnl, 4),
        "capture_pct": getattr(rec, "spread_capture_pct", 0.0),
        "worst_drift_c": getattr(rec, "worst_drift_c", 0.0),
    }
    should_log = (
        rec.attribution in TOXIC_ATTRIBUTIONS
        or pnl < -LARGE_LOSS_THRESHOLD
        or event in ("kill_switch", "daily_loss")
    )
    if should_log:
        try:
            os.makedirs(os.path.dirname(ALERTS_LOG) or ".", exist_ok=True)
            with open(ALERTS_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.debug("Alert log write failed: %s", e)
        if rec.attribution in TOXIC_ATTRIBUTIONS:
            log.warning("TOXIC FILL: %s  attr=%s  pnl=$%.4f  drift=%.1fc",
                        rec.slug, rec.attribution, pnl, rec.worst_drift_c)
        if pnl < -LARGE_LOSS_THRESHOLD:
            log.warning("LARGE LOSS: %s  pnl=$%.4f  attr=%s",
                        rec.slug, pnl, rec.attribution)

    # Optional webhook (Discord/Telegram) for critical events
    webhook = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    if webhook and event in ("kill_switch", "daily_loss"):
        try:
            import urllib.request
            body = json.dumps({"content": f"[MM] {event}: pnl=${pnl:.2f}"}).encode()
            req = urllib.request.Request(webhook, data=body, method="POST",
                                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            log.debug("Webhook failed: %s", e)


def _compute_fee(qp: "QuotePair", maker_pct: float = FEE_MAKER_PCT) -> float:
    """
    Fee on filled notional. We post limit orders = maker.
    Fee = (bid_fill_notional + ask_fill_notional) * maker_pct.
    """
    bid_notional = qp.bid_fill_price * qp.bid_size if qp.bid_filled else 0.0
    ask_notional = qp.ask_fill_price * qp.ask_size if qp.ask_filled else 0.0
    return round((bid_notional + ask_notional) * maker_pct, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Defensive components
# ═══════════════════════════════════════════════════════════════════════════════

class AdverseSelectionGuard:
    """
    Evaluate whether it is safe to quote a market, based on scanner-computed
    regime signals.

    The guard runs BEFORE a quote is placed.  It uses:
      adverse_sel_risk   (0–1) composite: momentum + imbalance + spread chaos
      trade_imbalance    (−1…+1) directional flow pressure

    Returns one of:
      "NORMAL"  post 1 tick inside the spread (default behaviour)
      "WIDEN"   post at the best bid/ask — don't narrow at all
                (protects margin when flow is one-sided)
      "PAUSE"   don't quote this market at all this cycle
    """

    def evaluate(self, adverse_sel_risk: float,
                 trade_imbalance: float) -> str:
        if adverse_sel_risk >= AS_RISK_PAUSE:
            log.info("AS_GUARD PAUSE  as_risk=%.2f", adverse_sel_risk)
            return "PAUSE"
        if (adverse_sel_risk >= AS_RISK_WIDEN or
                abs(trade_imbalance) >= IMBALANCE_WIDEN):
            log.info("AS_GUARD WIDEN  as_risk=%.2f  imb=%.2f",
                     adverse_sel_risk, trade_imbalance)
            return "WIDEN"
        return "NORMAL"


class StalenessGuard:
    """
    When a market's quotes move faster than our scan interval, a filled order
    may have been picked off on a stale price.

    staleness_risk_pct = P(≥1 quote move between scans) [0–1].
    We add extra ticks INSIDE the spread proportional to this risk, so that
    even if the market moved against us while our quote was live, we still
    capture a positive margin.

    Extra ticks:
      0.0–0.25  → 0 extra ticks
      0.25–0.50 → 1 extra tick
      0.50–0.75 → 2 extra ticks
      0.75+     → 3 extra ticks (STALENESS_MAX_EXTRA)
    """

    def extra_ticks(self, staleness_risk_pct: float) -> int:
        n = math.floor(staleness_risk_pct * STALENESS_MAX_EXTRA)
        return min(n, STALENESS_MAX_EXTRA)


class InventorySkewController:
    """
    Small inventory skew: skew = inventory * 0.2 (cents).
    bid = mid - spread/2 - skew, ask = mid + spread/2 - skew.

    When one leg fills, skew the remaining leg by inventory size.
    """

    def compute_skew(self, inventory: float) -> float:
        """Returns skew in price units. skew = inventory * 0.2 cents → inventory * 0.002."""
        skew_cents = inventory * 0.2
        skew = min(skew_cents / 100.0, MAX_SKEW)
        return skew

    def skewed_exit_price(self, side: str,
                          base_price: float,
                          inventory: float) -> float:
        """
        Adjusted price for the open leg after a fill on the other side.
        skew = inventory * 0.2 (cents).
        """
        skew = self.compute_skew(inventory)
        if side == "YES":
            return round(base_price - skew, 3)   # lower ask → easier to fill
        else:
            return round(base_price + skew, 3)   # raise bid → easier to fill


class PostFillDriftMonitor:
    """
    Track how the mid-price moves after the first leg of our quote fills.

    Adverse selection: an informed trader filled our order because they
    knew the fair value had moved.  If the price keeps moving against our
    position, we want to exit quickly rather than wait for the second leg.

    Drift is measured in cents, signed so that negative = adverse:
      Holding YES (bid filled): mid falling = adverse (drift < 0)
      Holding NO  (ask filled): mid rising  = adverse (drift < 0)

    Triggers:
      drift ≤ −ADVERSE_THRESHOLD_C  → log warning
      drift ≤ −EMERGENCY_EXIT_C     → force exit
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.active       = False
        self.fill_side    = None     # "YES" or "NO"
        self.mid_at_fill  = None
        self.fill_time    = None
        self.samples: list = []      # [(ts, signed_drift_c)]
        self.worst_drift  = 0.0      # most adverse seen (≤ 0 when adverse)
        self._last_ts     = 0.0

    def on_fill(self, side: str, bid: float, ask: float):
        """Call immediately when the first leg is confirmed filled."""
        self.fill_side   = side
        self.mid_at_fill = (bid + ask) / 2
        self.fill_time   = time.time()
        self.samples     = []
        self.worst_drift = 0.0
        self.active      = True
        self._last_ts    = time.time()
        log.info("PostFillDrift start: side=%s  mid_at_fill=%.3f",
                 side, self.mid_at_fill)

    def update(self, bid: float, ask: float) -> float:
        """
        Feed current best bid/ask.  Returns signed drift in cents
        (negative = adverse).  Returns 0 if called too soon or not active.
        """
        if not self.active or self.mid_at_fill is None:
            return 0.0
        now = time.time()
        if now - self._last_ts < DRIFT_SAMPLE_S:
            return self.samples[-1][1] if self.samples else 0.0

        current_mid   = (bid + ask) / 2
        raw_drift_c  = round((current_mid - self.mid_at_fill) * 100, 2)
        signed_drift = raw_drift_c if self.fill_side == "YES" else -raw_drift_c

        self.samples.append((now, signed_drift))
        self.worst_drift  = min(self.worst_drift, signed_drift)
        self._last_ts     = now

        age = now - self.fill_time
        log.debug("PostFillDrift: drift=%+.2fc  worst=%+.2fc  age=%.0fs",
                  signed_drift, self.worst_drift, age)
        return signed_drift

    def should_warn(self) -> bool:
        return self.worst_drift <= -ADVERSE_THRESHOLD_C

    def should_force_exit(self) -> bool:
        return self.worst_drift <= -EMERGENCY_EXIT_C


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end execution latency telemetry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LatencyRecord:
    """
    Full timing chain for one quote placement:
      detect_ts   — scanner identified the opportunity
      send_ts     — first place_limit_order() call fired
      ack_ts      — exchange returned order_id (bid leg)
      book_vis_ts — order confirmed visible on book (first manage_active_quotes poll)

    All timestamps are Unix seconds (float).  Deltas are in milliseconds.
    """
    slug:          str   = ""
    detect_ts:     float = 0.0
    send_ts:       float = 0.0
    ack_ts:        float = 0.0
    book_vis_ts:   float = 0.0

    def detect_to_send_ms(self) -> float:
        return (self.send_ts - self.detect_ts) * 1000 if self.send_ts > 0 else 0.0

    def send_to_ack_ms(self) -> float:
        return (self.ack_ts - self.send_ts) * 1000 if self.ack_ts > 0 and self.send_ts > 0 else 0.0

    def ack_to_book_ms(self) -> float:
        return (self.book_vis_ts - self.ack_ts) * 1000 if self.book_vis_ts > 0 and self.ack_ts > 0 else 0.0

    def total_ms(self) -> float:
        return (self.book_vis_ts - self.detect_ts) * 1000 if self.book_vis_ts > 0 and self.detect_ts > 0 else 0.0

    def is_complete(self) -> bool:
        return self.book_vis_ts > 0 and self.detect_ts > 0


class LatencyTracker:
    """
    Collects LatencyRecords across all quote placements and exposes
    per-segment p50/p95 statistics.

    Call sequence per quote:
      on_detect(slug)      — called by scan_and_quote when opportunity found
      on_send(slug)        — called by place_quote just before order submission
      on_ack(slug)         — called by place_quote after order_id returned
      on_book_visible(slug)— called by manage_active_quotes on first book poll
    """

    def __init__(self, maxlen: int = 500):
        self._records: deque[LatencyRecord] = deque(maxlen=maxlen)
        self._pending: dict[str, LatencyRecord] = {}

    def on_detect(self, slug: str) -> None:
        self._pending[slug] = LatencyRecord(slug=slug, detect_ts=time.time())

    def on_send(self, slug: str) -> None:
        if slug in self._pending:
            self._pending[slug].send_ts = time.time()

    def on_ack(self, slug: str) -> None:
        if slug in self._pending:
            self._pending[slug].ack_ts = time.time()

    def on_book_visible(self, slug: str) -> None:
        """Mark this slug as visible on the book and archive the record."""
        r = self._pending.pop(slug, None)
        if r is not None:
            r.book_vis_ts = time.time()
            self._records.append(r)

    def pending_slugs(self) -> set:
        return set(self._pending.keys())

    def stats(self) -> dict:
        complete = [r for r in self._records if r.is_complete()]
        if not complete:
            return {}

        def _p(lst: list, pct: int) -> float:
            if not lst:
                return 0.0
            s = sorted(lst)
            idx = max(0, min(len(s) - 1, int(len(s) * pct / 100)))
            return round(s[idx], 2)

        d2s = [r.detect_to_send_ms() for r in complete]
        s2a = [r.send_to_ack_ms()    for r in complete if r.send_to_ack_ms() > 0]
        a2b = [r.ack_to_book_ms()    for r in complete if r.ack_to_book_ms() > 0]
        tot = [r.total_ms()          for r in complete]

        return {
            "n":          len(complete),
            "d2s_p50":    _p(d2s, 50),
            "d2s_p95":    _p(d2s, 95),
            "s2a_p50":    _p(s2a, 50),
            "s2a_p95":    _p(s2a, 95),
            "a2b_p50":    _p(a2b, 50),
            "a2b_p95":    _p(a2b, 95),
            "total_p50":  _p(tot, 50),
            "total_p95":  _p(tot, 95),
        }

    def print_report(self) -> None:
        s = self.stats()
        if not s:
            print("  Latency telemetry: no complete records yet.")
            return
        n = s["n"]
        print(f"  Latency telemetry (n={n} complete):")
        print(f"    detect → send      p50={s['d2s_p50']:6.1f}ms  p95={s['d2s_p95']:6.1f}ms"
              f"  [internal processing]")
        print(f"    send   → ack       p50={s['s2a_p50']:6.1f}ms  p95={s['s2a_p95']:6.1f}ms"
              f"  [network RTT + exchange]")
        print(f"    ack    → book vis  p50={s['a2b_p50']:6.1f}ms  p95={s['a2b_p95']:6.1f}ms"
              f"  [propagation lag]")
        print(f"    FULL LOOP          p50={s['total_p50']:6.1f}ms  p95={s['total_p95']:6.1f}ms"
              f"  ← queue priority window")


# ═══════════════════════════════════════════════════════════════════════════════
# Queue position tracker
# ═══════════════════════════════════════════════════════════════════════════════

class QueuePositionTracker:
    """
    Estimates where our order sits in the queue at the best bid or ask,
    and uses that to compute P(fill in next cycle).

    Polymarket does NOT expose per-order queue position. We reconstruct
    it from L1 size changes observed between consecutive book polls.

    ── Core idea ────────────────────────────────────────────────────────────
    When we placed the order, the book showed size_at_place shares ahead of
    us at the best price.  If the price level is still the same next time we
    look, the reduction in L1 size represents trades that have consumed the
    queue ahead of us:

      consumed = size_at_place − current_size   (if current_size < size_at_place)

    We model queue consumption as FIFO (first in, first out):
      queue_pos_remaining = size_at_place − consumed
                          = current_size  (when price unchanged)

    Fill probability in the next poll interval is estimated with a geometric
    model: given observed consumption_per_cycle, what fraction of the
    remaining queue will clear in one more cycle?

      P(fill) ≈ 1 − (1 − consumption_rate) ^ (queue_pos_remaining / our_size)

    where consumption_rate = consumed_this_cycle / size_at_place.

    ── Price-level change ────────────────────────────────────────────────────
    If current_best_price ≠ original_price, the level we were quoting at
    has disappeared (our order either filled or we were passed through).
    We signal this with fill_probable=True, letting the caller re-query.

    ── Safety fallback ──────────────────────────────────────────────────────
    If no history (first poll after placement) or if size rose (new orders
    joined queue ahead of us — queue priority reset or cancel/replace),
    we return conservative estimates.
    """

    def estimate(
        self,
        size_at_place:   float,   # L1 size when we placed the order
        current_size:    float,   # L1 size right now (same price level)
        prev_size:       float,   # L1 size in the previous poll
        our_size:        float,   # shares we submitted
        price_unchanged: bool,    # True if best_bid/ask == our quoted price
    ) -> dict:
        """
        Returns a dict with:
          queue_pos_remaining  shares estimated ahead of us in the queue
          consumed_this_cycle  shares traded since last poll
          consumption_rate     fraction of original queue consumed this cycle
          fill_prob            P(our order fills by next poll)
          fill_probable        True if level moved (likely filled or cancelled)
          size_increased       True if someone added a big order ahead of us
        """
        if not price_unchanged:
            # Price level moved away — either we filled, or the market moved
            return {
                "queue_pos_remaining": 0.0,
                "consumed_this_cycle": 0.0,
                "consumption_rate":    0.0,
                "fill_prob":          1.0,
                "fill_probable":      True,
                "size_increased":     False,
            }

        # How much queue was consumed since our last look
        consumed_since_place = max(0.0, size_at_place - current_size)
        consumed_this_cycle  = max(0.0, prev_size - current_size)
        size_increased       = current_size > prev_size * 1.05  # >5% rise

        # Our position in the remaining queue
        queue_pos_remaining  = max(0.0, current_size - our_size)

        # Consumption rate this cycle as fraction of original depth
        if size_at_place > 0:
            consumption_rate = consumed_this_cycle / size_at_place
        else:
            consumption_rate = 0.0

        # P(our order fills next cycle).
        # Geometric model: if queue drains at `consumption_rate` per cycle,
        # expected cycles to fill = queue_pos_remaining / (consumption_rate * size_at_place)
        # P(fill ≤ 1 cycle) = min(1, consumption_this_cycle / (queue_pos_remaining + our_size))
        denom = queue_pos_remaining + our_size
        if denom > 0 and consumed_this_cycle > 0:
            fill_prob = round(min(1.0, consumed_this_cycle / denom), 4)
        else:
            fill_prob = 0.0

        return {
            "queue_pos_remaining": round(queue_pos_remaining, 1),
            "consumed_this_cycle": round(consumed_this_cycle, 1),
            "consumption_rate":    round(consumption_rate, 4),
            "fill_prob":          fill_prob,
            "fill_probable":      False,
            "size_increased":     size_increased,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuotePair:
    """Two-sided quote on one market, enriched with regime / inventory state."""
    market_slug:       str   = ""
    question:          str   = ""
    token_yes:         str   = ""
    token_no:          str   = ""

    # Bid (buy YES) side
    bid_price:         float = 0.0
    bid_size:          float = 0.0
    bid_order_id:      str   = ""       # L1 order_id (backward compat)
    bid_filled:        bool  = False
    bid_fill_price:    float = 0.0
    bid_ts:            float = 0.0
    # Multi-level: [(price, size), ...], order_ids, filled flags
    bid_levels:        list  = field(default_factory=list)   # [(price, size), ...]
    bid_order_ids:     list  = field(default_factory=list)    # [id1, id2, id3]
    bid_level_filled:  list  = field(default_factory=list)   # [bool, ...]
    bid_filled_size:   float = 0.0        # sum of filled bid level sizes

    # Ask (sell YES) side
    ask_price:         float = 0.0
    ask_size:          float = 0.0
    ask_order_id:      str   = ""
    ask_filled:        bool  = False
    ask_fill_price:    float = 0.0
    ask_ts:            float = 0.0
    ask_levels:        list  = field(default_factory=list)
    ask_order_ids:     list  = field(default_factory=list)
    ask_level_filled:  list  = field(default_factory=list)
    ask_filled_size:   float = 0.0        # sum of filled ask level sizes

    # Lifecycle
    status:            str   = "PENDING"
    created_ts:        float = 0.0
    last_check_ts:     float = 0.0
    pnl:               float = 0.0
    notes:             str   = ""

    # Regime snapshot at quote-creation time (from scanner)
    as_action:         str   = "NORMAL"   # NORMAL | WIDEN | PAUSE
    adverse_sel_risk:  float = 0.0
    trade_imbalance:   float = 0.0
    staleness_risk_pct: float = 0.0
    spread_stability:  float = 1.0
    quote_velocity_c_hr: float = 0.0     # mid drift c/hr (for post-mortem)
    extra_ticks_used:  int   = 0          # extra staleness ticks applied

    # ── Queue position (updated by manage_active_quotes on each book poll) ──
    # Bid side (YES tokens we're trying to buy)
    bid_l1_size_at_place:  float = 0.0   # L1 size when we first placed the bid
    last_bid_l1_size:      float = 0.0   # L1 size at the previous poll
    bid_queue_remaining:   float = 0.0   # estimated shares ahead of us
    bid_consumed_cycle:    float = 0.0   # shares traded since last poll
    bid_fill_prob:         float = 0.0   # P(bid fills in next cycle)
    # Ask side (NO tokens we're trying to buy = selling YES)
    ask_l1_size_at_place:  float = 0.0
    last_ask_l1_size:      float = 0.0
    ask_queue_remaining:   float = 0.0
    ask_consumed_cycle:    float = 0.0
    ask_fill_prob:         float = 0.0
    # Flag: someone added a large order in front of us (queue priority reset)
    bid_size_increased:    bool  = False
    ask_size_increased:    bool  = False

    # ── Live post-fill tracking ────────────────────────────────────────────
    fill_mid:          float = 0.0        # mid at time of first leg fill
    post_fill_drift_c: float = 0.0        # latest signed drift (cents)
    worst_drift_c:     float = 0.0        # worst adverse drift seen (cents)
    inventory_skew_c:  float = 0.0        # current skew being applied (cents)

    # ── Micro-fair-value (from scanner print-proxy) ────────────────────────
    fair_value:        float = 0.0        # scanner's print-EMA fair value
    mid_stale:         bool  = False      # True if mid is offset from fair value
    stale_direction:   str   = ""         # "UP" or "DOWN"

    # ── True probability estimate (from prob_engine.py) ───────────────────
    p_est:             float = 0.0        # estimated true probability of YES
    p_confidence:      float = 0.0        # 0–1 confidence in p_est
    edge_cents:        float = 0.0        # (p_est − mid) × 100 at placement
    alpha_mode:        str   = "PASSIVE_MM"  # PASSIVE_MM | SKEWED_MM | DIRECTIONAL
    directional_side:  str   = ""         # "BID" | "ASK" | "" (for one-sided orders)

    # ── Correlation cluster ────────────────────────────────────────────────
    cluster_id:        str   = "uncategorized"
    cluster_type:      str   = "other"

    # ── Latency telemetry anchor ───────────────────────────────────────────
    detect_ts:         float = 0.0        # when scanner identified opportunity

    # ── Fill analytics (spread capture + attribution + survival) ──────────
    expected_spread_c:  float = 0.0   # (ask_price − bid_price)*100 at placement
    first_fill_ts:      float = 0.0   # epoch when first leg fill was detected
    first_fill_side:    str   = ""    # "BID" | "ASK"
    done_ts:            float = 0.0   # epoch when DONE / exited
    # Real fill analytics: queue at post, mid at placement/fill
    placement_mid:      float = 0.0   # (best_bid+best_ask)/2 when we placed


@dataclass
class FillRecord:
    """
    Immutable snapshot of one completed quote, written by complete_quote().

    Feeds three analytics systems:
      SpreadCaptureTracker  — expected vs realised spread
      FillAttributor        — why did this fill happen?
      SurvivalCurveTracker  — how long did the quote live?
    """
    slug:                 str
    question:             str
    cluster_id:           str   = ""
    cluster_type:         str   = ""

    # ── Spread capture ────────────────────────────────────────────────────
    expected_spread_c:    float = 0.0   # spread at placement (ask_price − bid_price)*100
    actual_spread_c:      float = 0.0   # (ask_fill − bid_fill)*100 — negative = loss
    spread_capture_pct:   float = 0.0   # actual / expected  (1.0 = full, <0 = loss)
    expected_profit:      float = 0.0   # expected_spread_c/100 * shares ($)
    actual_profit:        float = 0.0   # realised P&L ($)

    # ── Fill attribution ──────────────────────────────────────────────────
    attribution:          str   = "UNKNOWN"
    adverse_sel_risk:     float = 0.0
    staleness_risk_pct:   float = 0.0
    trade_imbalance:     float = 0.0   # at placement (for post-mortem)
    worst_drift_c:        float = 0.0
    as_action:            str   = ""
    spread_stability:     float = 1.0
    quote_velocity_c_hr:  float = 0.0

    # ── Timing (survival curve) ───────────────────────────────────────────
    time_to_first_fill_s: float = 0.0   # created_ts → first leg fill
    time_to_done_s:       float = 0.0   # created_ts → DONE / exited
    first_fill_side:      str   = ""    # "BID" | "ASK"

    # ── Real fill analytics (queue, mid move, fill type) ─────────────────────
    # If you don't measure this, your edge assumption is fantasy.
    queue_bid_at_post:   float = 0.0   # L1 size at best bid when we placed
    queue_ask_at_post:   float = 0.0   # L1 size at best ask when we placed
    mid_at_placement:    float = 0.0   # market mid when we placed
    mid_at_first_fill:   float = 0.0   # market mid when first leg filled
    mid_move_during_queue_c: float = 0.0  # (mid_at_fill − mid_at_place) * 100
    fill_type:           str   = ""    # "BID_FIRST" | "ASK_FIRST"


class RealFillAnalyticsLogger:
    """
    Logs every fill to JSONL for post-hoc analysis.

    Each line: queue at post, time-to-fill, mid move during queue, fill type.
    If you don't measure this, your edge assumption is fantasy.
    """
    def __init__(self, path: str = FILL_ANALYTICS_LOG):
        self._path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def log(self, rec: FillRecord) -> None:
        row = asdict(rec)
        row["ts"] = time.time()
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            log.warning("Fill analytics log write failed: %s", e)


@dataclass
class MMState:
    """Global market-maker state."""
    active_quotes: dict  = field(default_factory=dict)
    completed:     list  = field(default_factory=list)
    total_pnl:     float = 0.0
    total_trades:  int   = 0
    balance:       float = 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Fill analytics
# ═══════════════════════════════════════════════════════════════════════════════

def _attribute_fill(qp: "QuotePair") -> str:
    """
    Classify a completed quote into one of five attribution categories.

    ORGANIC    — clean two-sided fill; standard market-making income.
    ADVERSE    — informed-trader fill; post-fill drift or spread shortfall
                 indicates a directional trader picked us off.
    STALE      — our quote lagged the market; filled on a stale price
                 (high staleness_risk_pct at placement).
    DRIFT_EXIT — PostFillDriftMonitor triggered an early adverse exit.
    TIMEOUT    — max hold age, spread-narrowed cancel, or market moved out.

    Rules are checked in priority order (most specific first).
    """
    notes = (qp.notes or "").lower()

    # Forced exits (most specific — check notes written by exit handlers)
    if "adverse_exit" in notes:
        return "DRIFT_EXIT"
    if "force_exit" in notes or "stale:" in notes or "spread_narrow" in notes:
        return "TIMEOUT"

    # Significant adverse drift after one leg filled
    if qp.worst_drift_c < -3.0:
        return "ADVERSE"

    # Actual spread captured << expected → someone informed crossed us
    if qp.expected_spread_c > 0:
        actual_c = (qp.ask_fill_price - qp.bid_fill_price) * 100
        if actual_c < qp.expected_spread_c * 0.30:
            # Discriminate: AS risk dominant → ADVERSE; staleness dominant → STALE
            return "ADVERSE" if qp.adverse_sel_risk > qp.staleness_risk_pct else "STALE"

    # High staleness risk at placement → likely stale-quote fill
    if qp.staleness_risk_pct > 0.60:
        return "STALE"

    return "ORGANIC"


class SpreadCaptureTracker:
    """
    Tracks actual spread captured relative to expected spread at placement.

      capture_pct = actual_spread_c / expected_spread_c
        1.0  — perfect: earned the full quoted spread
        0.0  — break-even (e.g. force-exited at mid)
       <0.0  — loss (adverse exit past the other leg)

    Grouped by fill attribution so you can see which fill types are profitable
    and which are bleeding edge.
    """

    def __init__(self, maxlen: int = 500):
        self._records: deque[FillRecord] = deque(maxlen=maxlen)

    def record(self, rec: FillRecord) -> None:
        self._records.append(rec)

    def stats(self) -> dict:
        recs = list(self._records)
        if not recs:
            return {}
        n        = len(recs)
        captures = [r.spread_capture_pct for r in recs]
        profits  = [r.actual_profit       for r in recs]

        def _median(lst: list) -> float:
            s = sorted(lst)
            return round(s[len(s) // 2], 4)

        by_attr: dict[str, list] = {}
        for r in recs:
            by_attr.setdefault(r.attribution, []).append(r)

        return {
            "n":              n,
            "mean_capture":   round(sum(captures) / n, 4),
            "median_capture": _median(captures),
            "pct_positive":   round(sum(1 for c in captures if c > 0) / n * 100, 1),
            "total_profit":   round(sum(profits), 4),
            "mean_profit":    round(sum(profits) / n, 4),
            "by_attribution": {
                k: {
                    "n":            len(v),
                    "mean_capture": round(sum(r.spread_capture_pct for r in v) / len(v), 4),
                    "mean_profit":  round(sum(r.actual_profit       for r in v) / len(v), 4),
                    "total_profit": round(sum(r.actual_profit       for r in v), 4),
                }
                for k, v in sorted(by_attr.items())
            },
        }

    def print_report(self) -> None:
        s = self.stats()
        if not s:
            print("  Spread capture: no completed trades yet.")
            return

        print(f"\n  Spread Capture vs Expected Value  (n={s['n']})")
        print(f"  {'─'*62}")
        print(f"    Mean capture:    {s['mean_capture']*100:+6.1f}%  "
              f"(100% = full spread earned, 0% = break-even, <0 = loss)")
        print(f"    Median capture:  {s['median_capture']*100:+6.1f}%")
        print(f"    Profitable fills:{s['pct_positive']:5.1f}%")
        print(f"    Mean profit:    ${s['mean_profit']:+.4f} / trade")
        print(f"    Total profit:   ${s['total_profit']:+.4f}")

        if s["by_attribution"]:
            print(f"\n    By fill attribution:")
            hdr = f"    {'TYPE':<14} {'N':>4}  {'CAPTURE%':>9}  {'MEAN PNL':>9}  {'TOTAL PNL':>10}  bar"
            print(hdr)
            print(f"    {'─'*14} {'─'*4}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*20}")
            for attr, d in sorted(s["by_attribution"].items()):
                cap_pct = d["mean_capture"] * 100
                bar_len = max(0, min(20, int(abs(cap_pct) / 5)))
                bar = ("█" * bar_len) if cap_pct >= 0 else ("░" * bar_len)
                print(f"    {attr:<14} {d['n']:4d}  {cap_pct:+8.1f}%  "
                      f"${d['mean_profit']:+8.4f}  ${d['total_profit']:+9.4f}  {bar}")


class SurvivalCurveTracker:
    """
    Empirical fill survival curve.

    S(t) = fraction of quotes still unfilled at t seconds after placement
         = 1 − CDF(time_to_done_s)

    What this tells you:
      • S(60s) = 0.80 → 80% of your quotes are still waiting after 1 min.
        Your timeout should be >> 60s or you'll cancel before most fills arrive.
      • S(300s) drops fast → market is active, fills come quickly.
      • S(t) flat for long stretches → liquidity is episodic (match-day spikes).
      • ADVERSE fills cluster at low t → informed traders hit your quote fast.
        ORGANIC fills cluster at higher t → passive, patient counterparties.

    Supports segmentation by `attribution` or `cluster_type` to compare
    fill-speed profiles across market regimes.
    """

    BUCKETS_S: list = [30, 60, 90, 120, 180, 300, 420, 600, 900, 1800]

    def __init__(self, maxlen: int = 300):
        self._records: deque[FillRecord] = deque(maxlen=maxlen)

    def record(self, rec: FillRecord) -> None:
        if rec.time_to_done_s > 0:
            self._records.append(rec)

    def _survival_at(self, t: float, records: list) -> float:
        if not records:
            return 1.0
        return sum(1 for r in records if r.time_to_done_s > t) / len(records)

    def print_report(self, segment_by: str = "attribution") -> None:
        recs = list(self._records)
        if len(recs) < 3:
            print(f"  Survival curve: need ≥3 completed trades (have {len(recs)}).")
            return

        n = len(recs)
        times_sorted = sorted(r.time_to_done_s for r in recs)
        median_s = times_sorted[n // 2]
        p25_s    = times_sorted[n // 4]
        p75_s    = times_sorted[min(n - 1, 3 * n // 4)]

        print(f"\n  Fill Survival Curve  (n={n}  "
              f"p25={p25_s:.0f}s  median={median_s:.0f}s  p75={p75_s:.0f}s)")
        print(f"  S(t) = % of quotes still unfilled at time t after placement")
        print(f"  {'TIME':>6}  {'FILLED':>7}  {'CUMUL%':>7}  {'S(t)%':>6}  "
              f"survival bar  (each █ ≈ 2.5%)")
        print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*40}")

        prev_cum = 0
        for t in self.BUCKETS_S:
            cum      = sum(1 for r in recs if r.time_to_done_s <= t)
            new_n    = cum - prev_cum
            cum_pct  = cum  / n * 100
            surv_pct = (1 - cum / n) * 100
            bar_w    = int(surv_pct / 2.5)
            bar      = "█" * bar_w
            mins, secs = divmod(t, 60)
            t_label  = f"{mins}m" if secs == 0 else f"{t}s"
            new_tag  = f"+{new_n}" if new_n else "  "
            print(f"  {t_label:>6}  {cum:5d}{new_tag:>3}  {cum_pct:6.1f}%  "
                  f"{surv_pct:5.1f}%  {bar}")
            prev_cum = cum

        beyond = sum(1 for r in recs if r.time_to_done_s > self.BUCKETS_S[-1])
        if beyond:
            print(f"  {'30m+':>6}  {n - beyond:5d}     {(n-beyond)/n*100:6.1f}%  "
                  f"{beyond/n*100:5.1f}%  (very slow or still waiting)")

        # Segmentation: median fill time per group
        if segment_by:
            groups: dict[str, list] = {}
            for r in recs:
                key = getattr(r, segment_by, "?") or "?"
                groups.setdefault(key, []).append(r)
            if len(groups) > 1:
                print(f"\n  Median fill time by {segment_by}:")
                for key in sorted(groups, key=lambda k: -len(groups[k])):
                    grp = groups[key]
                    med = sorted(r.time_to_done_s for r in grp)[len(grp) // 2]
                    mean_cap = sum(r.spread_capture_pct for r in grp) / len(grp)
                    print(f"    {key:<18} n={len(grp):3d}  "
                          f"median={med:6.0f}s  capture={mean_cap*100:+6.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Toxic Fill Post-Mortem
# ═══════════════════════════════════════════════════════════════════════════════

TOXIC_ATTRIBUTIONS = frozenset({"ADVERSE", "DRIFT_EXIT", "STALE"})


class ToxicPostMortemTracker:
    """
    Deep autopsy of ADVERSE, DRIFT_EXIT, and STALE fills.

    Three-layer analysis:

    1. Signal snapshot table
       Compares mean regime signals at placement (toxic vs organic):
       AS risk, imbalance, staleness, velocity, spread stability.
       Shows threshold gaps — if toxic mean is near a guard threshold,
       that threshold needs tightening.

    2. Pattern detection
       For each placement signal, counts what fraction of toxic fills had
       that signal above its warning level.  High-frequency patterns
       ("7/9 toxic fills had |imbalance|>0.4") expose systematic blind spots.

    3. Cluster heatmap
       Which correlation clusters generate disproportionate toxic fills?
       Informs cluster exposure limits.

    4. Suggested threshold tightening
       Computes suggested tighter thresholds from observed signal means.

    5. Per-fill log (last 8)
       Regime → outcome → root cause for recent toxic fills.
    """

    # (field, label, warning level, guard threshold, direction)
    # direction: "hi" = higher is worse, "lo" = lower is worse
    _SIGNAL_DEFS: list = [
        ("adverse_sel_risk",   "AS risk",        0.40, AS_RISK_PAUSE,  "hi"),
        ("staleness_risk_pct", "Staleness",       0.40, 0.60,           "hi"),
        ("trade_imbalance",    "|Imbalance|",     0.40, IMBALANCE_WIDEN,"hi"),
        ("spread_stability",   "Spread stab",     0.60, 0.50,           "lo"),
        ("quote_velocity_c_hr","Velocity c/hr",   5.0,  10.0,           "hi"),
    ]

    def __init__(self, maxlen: int = 200):
        self._toxic:   deque[FillRecord] = deque(maxlen=maxlen)
        self._organic: deque[FillRecord] = deque(maxlen=maxlen)

    def record_if_toxic(self, rec: FillRecord) -> None:
        if rec.attribution in TOXIC_ATTRIBUTIONS:
            self._toxic.append(rec)
        elif rec.attribution == "ORGANIC":
            self._organic.append(rec)

    def _mean(self, recs: list, field: str, abs_val: bool = False) -> float:
        vals = [abs(getattr(r, field, 0.0)) if abs_val else getattr(r, field, 0.0)
                for r in recs]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _pct_above(self, recs: list, field: str, threshold: float,
                   abs_val: bool = False) -> float:
        """Fraction of records where field exceeds threshold."""
        if not recs:
            return 0.0
        vals = [abs(getattr(r, field, 0.0)) if abs_val else getattr(r, field, 0.0)
                for r in recs]
        return round(sum(1 for v in vals if v >= threshold) / len(recs) * 100, 1)

    def _root_cause(self, r: FillRecord) -> str:
        if r.attribution == "DRIFT_EXIT":
            return f"PostFillDriftMonitor fired — drift {r.worst_drift_c:+.1f}c exceeded threshold"
        if r.attribution == "STALE":
            if r.staleness_risk_pct > 0.60:
                return f"Quote lagged market (staleness {r.staleness_risk_pct*100:.0f}%)"
            return "Spread capture << expected; staleness signal dominant"
        if r.attribution == "ADVERSE":
            if r.worst_drift_c < -3.0:
                return f"Informed trader: adverse drift {r.worst_drift_c:+.1f}c post-fill"
            if r.time_to_first_fill_s < 30 and r.adverse_sel_risk > 0.30:
                return f"Fast fill ({r.time_to_first_fill_s:.0f}s) + AS risk {r.adverse_sel_risk:.2f} → informed hit"
            return "Spread shortfall — got less than expected spread"
        return "Unknown"

    def print_report(self) -> None:
        toxic   = list(self._toxic)
        organic = list(self._organic)
        n_t, n_o = len(toxic), len(organic)
        if not toxic:
            print("  Toxic fill post-mortem: no ADVERSE/DRIFT_EXIT/STALE fills yet.")
            return

        W = 76
        print(f"\n  Toxic Fill Post-Mortem  "
              f"({n_t} toxic  {n_o} organic  ratio={n_t/(n_t+n_o)*100:.0f}% if n_t+n_o else '-')")

        # ── 1. Signal comparison table ─────────────────────────────────────
        print(f"\n  Signal levels at placement (toxic vs organic):")
        print(f"  {'SIGNAL':<22} {'TOXIC':>8}  {'ORGANIC':>8}  {'GUARD':>7}  "
              f"{'GAP':>6}  VERDICT")
        print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*20}")
        suggestions = []
        for field, label, warn, guard, direction in self._SIGNAL_DEFS:
            abs_v = field == "trade_imbalance"
            t_m   = self._mean(toxic, field, abs_val=abs_v)
            o_m   = self._mean(organic, field, abs_val=abs_v) if organic else 0.0
            gap   = (guard - t_m) if direction == "hi" else (t_m - guard)
            if direction == "hi":
                fires = t_m >= guard
                close = gap < 0.15 and not fires
                verdict = ("✓ guard fires" if fires
                           else f"⚠ gap={gap:.2f}" if close
                           else "✗ not caught")
                if close:
                    suggestions.append((label, field, guard, t_m,
                                        f"{guard:.2f} → {round(t_m * 0.85, 2):.2f}"))
            else:  # "lo" — lower is worse (stability)
                fires = t_m <= guard
                close = gap < 0.10 and not fires
                verdict = ("✓ guard fires" if fires
                           else f"⚠ gap={gap:.2f}" if close
                           else "✗ not caught")
            print(f"  {label:<22} {t_m:8.3f}  {o_m:8.3f}  {guard:7.2f}  "
                  f"{gap:+6.3f}  {verdict}")

        # ── 2. Pattern detection ───────────────────────────────────────────
        print(f"\n  Pattern: % of toxic fills with signal above warning level:")
        patterns = []
        for field, label, warn, guard, direction in self._SIGNAL_DEFS:
            abs_v = field == "trade_imbalance"
            pct_t = self._pct_above(toxic,   field, warn, abs_val=abs_v)
            pct_o = self._pct_above(organic, field, warn, abs_val=abs_v) if organic else 0.0
            excess = pct_t - pct_o
            flag   = "⚠" if pct_t >= 50 else " "
            patterns.append((pct_t, label, pct_t, pct_o, excess, flag))
        for _, label, pct_t, pct_o, excess, flag in sorted(patterns, reverse=True):
            bar_t = "█" * int(pct_t / 5)
            print(f"  {flag} {label:<22} toxic={pct_t:4.0f}%  organic={pct_o:4.0f}%  "
                  f"excess={excess:+5.0f}pp  {bar_t}")

        # First-fill speed (informed traders hit fast)
        t_speed = self._mean(toxic,   "time_to_first_fill_s")
        o_speed = self._mean(organic, "time_to_first_fill_s") if organic else 999.0
        ratio   = t_speed / o_speed if o_speed > 0 else 1.0
        flag    = "⚠" if ratio < 0.5 else " "
        print(f"  {flag} {'First-fill speed':<22} toxic={t_speed:6.0f}s  "
              f"organic={o_speed:6.0f}s  "
              f"ratio={ratio:.2f}{'  ← informed hit fast' if ratio < 0.5 else ''}")

        # ── 3. Cluster heatmap ─────────────────────────────────────────────
        if n_t >= 3:
            print(f"\n  Toxic fills by cluster (top 5):")
            clusters: dict[str, int] = {}
            for r in toxic:
                clusters[r.cluster_id] = clusters.get(r.cluster_id, 0) + 1
            for cid, cnt in sorted(clusters.items(), key=lambda x: -x[1])[:5]:
                bar = "█" * cnt
                print(f"    {cid:<22} {cnt:3d}  {bar}")

        # ── 4. Attribution breakdown ───────────────────────────────────────
        by_attr: dict[str, int] = {}
        for r in toxic:
            by_attr[r.attribution] = by_attr.get(r.attribution, 0) + 1
        print(f"\n  Attribution breakdown: "
              + "  ".join(f"{k}:{v}" for k, v in sorted(by_attr.items())))

        # ── 5. Threshold tightening suggestions ───────────────────────────
        if suggestions:
            print(f"\n  Suggested threshold tightening (based on toxic signal means):")
            for label, field, guard, t_mean, suggestion in suggestions:
                print(f"    {label:<22} guard={guard:.2f}  toxic_mean={t_mean:.3f}  "
                      f"→ tighten to ~{suggestion}")

        # ── 6. Per-fill log (last 8) ───────────────────────────────────────
        print(f"\n  Last {min(8, n_t)} toxic fills (placement regime → outcome → cause):")
        print(f"  {'─'*W}")
        for r in list(toxic)[-8:]:
            cause = self._root_cause(r)
            regime = (f"AS={r.adverse_sel_risk:.2f} imb={r.trade_imbalance:+.2f} "
                      f"stl={r.staleness_risk_pct*100:.0f}% "
                      f"vel={r.quote_velocity_c_hr:+.1f} stab={r.spread_stability:.2f}")
            outcome = (f"drift={r.worst_drift_c:+.1f}c  cap={r.spread_capture_pct*100:+.0f}%  "
                       f"t1={r.time_to_first_fill_s:.0f}s  age={r.time_to_done_s:.0f}s  "
                       f"pnl=${r.actual_profit:+.4f}")
            print(f"  [{r.attribution:10s}] {r.cluster_id:<14} {r.slug[:22]}")
            print(f"    regime:  {regime}")
            print(f"    outcome: {outcome}")
            print(f"    cause:   {cause}")
            print()


# ═══════════════════════════════════════════════════════════════════════════════
# Capital Efficiency
# ═══════════════════════════════════════════════════════════════════════════════

class CapitalEfficiencyTracker:
    """
    Measures how hard the bankroll is working across the whole session.

    Deployed capital per quote:
      bid side: bid_price × shares  (cost to buy YES)
      ask side: (1 − ask_price) × shares  (worst-case payout if YES resolves = 1)
      total per quote: (bid_price + 1 − ask_price) × shares

    Core metrics:
      deployed_now           current $ committed (QUOTING + ONE_LEG quotes)
      idle_capital           bankroll − deployed_now
      deploy_pct             deployed / bankroll × 100
      time_weighted_deployed trapezoidal integral of deployed(t) / session_time
      return_on_capital      total_pnl / time_weighted_deployed × 100
      fills_per_hr           completed fills in last 60 min
      capital_turns          total_notional_traded / time_weighted_deployed
      notional_traded        sum of bid_price*shares + ask_fill*shares per fill
    """

    def __init__(self, bankroll: float = 100.0):
        self._bankroll    = bankroll
        self._start_ts    = time.time()
        self._history:    deque[tuple[float, float]] = deque(maxlen=2000)
        self._fill_ts:    deque[float] = deque(maxlen=500)
        self._notional:   float = 0.0

    def snapshot(self, active_quotes: dict) -> float:
        """Record current deployment level. Call once per main loop cycle."""
        deployed = sum(
            (qp.bid_price + 1.0 - qp.ask_price) * qp.bid_size
            for qp in active_quotes.values()
            if qp.status in ("QUOTING", "ONE_LEG")
        )
        self._history.append((time.time(), deployed))
        return deployed

    def record_completion(self, qp: "QuotePair") -> None:
        """Record fill notional and timestamp. Called from complete_quote."""
        n  = qp.bid_size * qp.bid_fill_price if qp.bid_filled else 0.0
        n += qp.ask_size * qp.ask_fill_price if qp.ask_filled else 0.0
        self._notional += n
        self._fill_ts.append(time.time())

    def capital_at_risk(self, active_quotes: dict) -> float:
        """Current notional at risk (open orders + one-leg inventory)."""
        total = 0.0
        for qp in active_quotes.values():
            if qp.status == "QUOTING":
                total += (qp.bid_price + 1.0 - qp.ask_price) * qp.bid_size
            elif qp.status == "ONE_LEG":
                total += (qp.bid_fill_price * qp.bid_size if qp.bid_filled
                          else qp.ask_fill_price * qp.ask_size)
        return total

    def _time_weighted_deployed(self) -> float:
        h = list(self._history)
        if len(h) < 2:
            return h[0][1] if h else 0.0
        integral = sum(
            (h[i][1] + h[i-1][1]) / 2 * (h[i][0] - h[i-1][0])
            for i in range(1, len(h))
        )
        duration = h[-1][0] - h[0][0]
        return integral / duration if duration > 0 else h[-1][1]

    def stats(self, total_pnl: float, active_quotes: dict) -> dict:
        if not self._history:
            return {}
        now         = time.time()
        session_s   = now - self._start_ts
        deployed    = self._history[-1][1]
        twd         = self._time_weighted_deployed()
        idle        = max(0.0, self._bankroll - deployed)
        deploy_pct  = deployed / self._bankroll * 100 if self._bankroll else 0.0
        roc         = total_pnl / twd * 100 if twd > 0 else 0.0
        turns       = self._notional / twd  if twd > 0 else 0.0
        hr_ago      = now - 3600
        fills_hr    = sum(1 for t in self._fill_ts if t > hr_ago)
        # annualised ROC: (roc per session) * (8760 / session_hrs)
        session_hrs = session_s / 3600
        roc_ann     = roc / session_hrs * 8760 if session_hrs > 0.01 else 0.0
        return {
            "deployed_now":     round(deployed, 2),
            "idle":             round(idle, 2),
            "deploy_pct":       round(deploy_pct, 1),
            "twd":              round(twd, 2),
            "roc_session_pct":  round(roc, 4),
            "roc_ann_pct":      round(roc_ann, 2),
            "fills_last_hr":    fills_hr,
            "capital_turns":    round(turns, 4),
            "notional_traded":  round(self._notional, 2),
            "session_hrs":      round(session_hrs, 3),
        }

    def print_report(self, total_pnl: float, active_quotes: dict) -> None:
        s = self.stats(total_pnl, active_quotes)
        if not s:
            print("  Capital efficiency: no deployment history yet.")
            return
        at_risk = self.capital_at_risk(active_quotes)
        print(f"\n  Capital Efficiency  "
              f"(bankroll=${self._bankroll:.0f}  session={s['session_hrs']*60:.0f}min)")
        print(f"  {'─'*62}")
        print(f"    Deployed now:         ${s['deployed_now']:7.2f}  "
              f"({s['deploy_pct']:.1f}% of bankroll)")
        print(f"    Idle capital:         ${s['idle']:7.2f}")
        print(f"    Capital at risk:      ${at_risk:7.2f}  "
              f"(open + one-leg exposure)")
        print(f"    Time-wtd deployed:    ${s['twd']:7.2f}")
        print(f"    Return on capital:   {s['roc_session_pct']:+7.3f}%  "
              f"session  ({s['roc_ann_pct']:+.1f}% ann.)")
        print(f"    Fills last 60min:     {s['fills_last_hr']:7d}")
        print(f"    Capital turns:        {s['capital_turns']:7.4f}  "
              f"(notional / avg_deployed)")
        print(f"    Notional traded:      ${s['notional_traded']:,.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Market maker engine
# ═══════════════════════════════════════════════════════════════════════════════

class MarketMaker:
    """
    Manages multiple two-sided quotes across Polymarket markets.

    Quote lifecycle:
      PENDING  → compute + post bid + ask
      QUOTING  → live, waiting for fills
      ONE_LEG  → one side filled, inventory skew + drift monitoring active
      DONE     → both sides filled (profit) or inventory exited (loss)
      CANCELLED → withdrawn (spread narrowed / AS guard fired / timeout)
    """

    def __init__(self, state: MMState, paper: bool = True):
        self.state        = state
        self.paper        = paper
        self._as_guard    = AdverseSelectionGuard()
        self._st_guard    = StalenessGuard()
        self._inv_skew    = InventorySkewController()
        self._queue          = QueuePositionTracker()
        self.latency         = LatencyTracker()          # end-to-end telemetry
        self._spread_capture = SpreadCaptureTracker()                    # spread capture analytics
        self._survival       = SurvivalCurveTracker()                    # time-to-fill survival curve
        self._toxic_pm       = ToxicPostMortemTracker()                  # toxic fill post-mortem
        self._cap_eff        = CapitalEfficiencyTracker(state.balance)   # capital efficiency
        self._fill_analytics = RealFillAnalyticsLogger()                  # real fill analytics JSONL
        self._drift: dict[str, PostFillDriftMonitor] = {}

    def update_queue_position(self, qp: QuotePair,
                              current_bid_price: float,
                              current_ask_price: float,
                              current_bid_l1_size: float,
                              current_ask_l1_size: float):
        """
        Call on every book poll for an active QUOTING quote.

        Uses QueuePositionTracker to update all queue fields on the QuotePair
        so the dashboard can show fill_prob, consumed size, and size-increase
        warnings in real time.

        Side-effects (written back onto qp):
          bid_queue_remaining, bid_consumed_cycle, bid_fill_prob,
          ask_queue_remaining, ask_consumed_cycle, ask_fill_prob,
          bid_size_increased, ask_size_increased
        """
        if qp.status not in ("QUOTING", "ONE_LEG"):
            return

        # ── Bid side ──────────────────────────────────────────────────────
        bid_unchanged = abs(current_bid_price - qp.bid_price) < 0.0005
        b = self._queue.estimate(
            size_at_place   = qp.bid_l1_size_at_place,
            current_size    = current_bid_l1_size,
            prev_size       = qp.last_bid_l1_size,
            our_size        = qp.bid_size,
            price_unchanged = bid_unchanged,
        )
        qp.bid_queue_remaining = b["queue_pos_remaining"]
        qp.bid_consumed_cycle  = b["consumed_this_cycle"]
        qp.bid_fill_prob       = b["fill_prob"]
        qp.bid_size_increased  = b["size_increased"]
        qp.last_bid_l1_size    = current_bid_l1_size

        if b["size_increased"] and not qp.bid_filled:
            log.warning("QUEUE RESET (bid): %s  someone added %.0f shares ahead — "
                        "queue priority lost",
                        qp.market_slug, current_bid_l1_size - qp.last_bid_l1_size)

        # ── Ask side ──────────────────────────────────────────────────────
        ask_unchanged = abs(current_ask_price - qp.ask_price) < 0.0005
        a = self._queue.estimate(
            size_at_place   = qp.ask_l1_size_at_place,
            current_size    = current_ask_l1_size,
            prev_size       = qp.last_ask_l1_size,
            our_size        = qp.ask_size,
            price_unchanged = ask_unchanged,
        )
        qp.ask_queue_remaining = a["queue_pos_remaining"]
        qp.ask_consumed_cycle  = a["consumed_this_cycle"]
        qp.ask_fill_prob       = a["fill_prob"]
        qp.ask_size_increased  = a["size_increased"]
        qp.last_ask_l1_size    = current_ask_l1_size

        if a["size_increased"] and not qp.ask_filled:
            log.warning("QUEUE RESET (ask): %s  someone added %.0f shares ahead — "
                        "queue priority lost",
                        qp.market_slug, current_ask_l1_size - qp.last_ask_l1_size)

        log.debug("QUEUE %s  bid_q=%.0f fill=%.2f  ask_q=%.0f fill=%.2f",
                  qp.market_slug,
                  qp.bid_queue_remaining, qp.bid_fill_prob,
                  qp.ask_queue_remaining, qp.ask_fill_prob)

    # ── quote pricing ──────────────────────────────────────────────────────

    def compute_quote_prices(
        self,
        best_bid: float,
        best_ask: float,
        spread_cents: float,
        adverse_sel_risk: float   = 0.0,
        trade_imbalance: float    = 0.0,
        staleness_risk_pct: float = 0.0,
        inventory: float          = 0.0,
        # Probability edge parameters
        p_est: float              = 0.0,
        p_confidence: float       = 0.0,
        alpha_mode: str           = "PASSIVE_MM",
    ) -> Optional[tuple[float, float, str]]:
        """
        Return (bid_price, ask_price, as_action) or None if not worth quoting.

        Queue priority: step ahead of touch. Matching best_bid = behind 50k shares (FIFO).
        best_bid + 1 tick = front of queue. One tick improvement → jump the line.

        Base: bid = best_bid + STEP_AHEAD_TICKS, ask = best_ask - STEP_AHEAD_TICKS
        Then: - skew (inventory), + bias (SKEWED_MM view).

        WIDEN: post at touch (best_bid, best_ask) — no stepping, protect margin.
        """
        if spread_cents < MIN_SPREAD_TO_QUOTE:
            return None

        as_action = self._as_guard.evaluate(adverse_sel_risk, trade_imbalance)
        if as_action == "PAUSE":
            return None

        mid    = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        skew   = self._inv_skew.compute_skew(inventory)

        # Bias toward the side we want: p_est > mid → buy YES → shift up; p_est < mid → sell → shift down
        bias = 0.0
        use_p_est = (p_est > 0.02 and p_confidence >= 0.20
                     and alpha_mode in ("SKEWED_MM", "DIRECTIONAL"))
        if use_p_est and alpha_mode == "SKEWED_MM":
            edge = p_est - mid
            if edge > 0:
                bias = QUOTE_BIAS_CENTS / 100.0   # want to buy YES
            elif edge < 0:
                bias = -QUOTE_BIAS_CENTS / 100.0  # want to sell YES

        # Queue priority: step ahead of touch. Matching best_bid = behind 50k shares.
        # best_bid + step = front of queue. Fill rate >> spread capture.
        step = STEP_AHEAD_CENTS / 100.0
        if as_action == "WIDEN":
            our_bid = round(best_bid, 3)
            our_ask = round(best_ask, 3)
        else:
            our_bid = round(best_bid + step - skew + bias, 3)
            our_ask = round(best_ask - step - skew + bias, 3)
            # Spread too tight (e.g. 2c): stepping would cross. Fall back to touch.
            if our_bid >= our_ask - 0.001:
                our_bid = round(best_bid - skew + bias, 3)
                our_ask = round(best_ask - skew + bias, 3)
            if our_bid >= our_ask - 0.001:
                our_bid = round(best_bid, 3)
                our_ask = round(best_ask, 3)

        if our_bid >= our_ask:
            return None
        if our_bid <= 0 or our_ask >= 1:
            return None
        if round(our_ask - our_bid, 4) < TICK - 0.001:
            return None

        return our_bid, our_ask, as_action

    def should_quote_market(self, slug: str, best_bid: float,
                            best_ask: float, volume_24h: float) -> bool:
        if slug in self.state.active_quotes:
            return False
        if len(self.state.active_quotes) >= MAX_MARKETS:
            return False
        return (best_ask - best_bid) * 100 >= MIN_SPREAD_TO_QUOTE

    # ── quote creation + placement ─────────────────────────────────────────

    def create_quote(
        self,
        slug: str,
        question: str,
        token_yes: str,
        token_no: str,
        best_bid: float,
        best_ask: float,
        # Scanner regime signals (optional — default to neutral)
        adverse_sel_risk:   float = 0.0,
        trade_imbalance:    float = 0.0,
        staleness_risk_pct: float = 0.0,
        spread_stability:   float = 1.0,
        # Queue position (L1 sizes at placement — for fill-probability tracking)
        bid_l1_size:  float = 0.0,
        ask_l1_size:  float = 0.0,
        bid_l1_depth: float = 0.0,   # USD at best bid (for liquidity-based size)
        ask_l1_depth: float = 0.0,   # USD at best ask
        # Micro-fair-value (from scanner print-proxy)
        fair_value:      float = 0.0,
        mid_stale:       bool  = False,
        stale_direction: str   = "",
        # Correlation cluster
        cluster_id:      str   = "uncategorized",
        cluster_type:    str   = "other",
        # Latency anchor (set by scan_and_quote before calling create_quote)
        detect_ts:       float = 0.0,
        # Market death detector: 24h vs 2h regime change → reduce size 50%
        size_multiplier: float = 1.0,
        # Time-to-resolution: liquidity dies before close; reduce size as market nears
        days_to_end: float = 999.0,
        # For toxic post-mortem (regime at placement)
        quote_velocity_c_hr: float = 0.0,
        # True probability edge (from prob_engine.py)
        p_est:        float = 0.0,
        p_confidence: float = 0.0,
        edge_cents:   float = 0.0,
        alpha_mode:   str   = "PASSIVE_MM",
    ) -> Optional[QuotePair]:
        """
        Create a QuotePair.  Does NOT place orders yet.

        When mid_stale=True the scanner detected that actual prints are
        consistently above (or below) the quoted mid.  We shift both bid
        and ask toward fair_value before computing the final quote prices
        so that we are not posting a two-sided quote centred on a stale price.
        """
        # Market mid at placement (for fill analytics: mid move during queue)
        placement_mid = round((best_bid + best_ask) / 2, 4)

        # Quote strictly relative to the book — no fair-value shift. Real bots sit at
        # best_bid + 1 tick, best_ask - 1 tick to be at the front of the queue.
        # Fair-value shift was moving us behind the touch and reducing fill probability.

        spread_c = (best_ask - best_bid) * 100
        result   = self.compute_quote_prices(
            best_bid, best_ask, spread_c,
            adverse_sel_risk, trade_imbalance, staleness_risk_pct,
            p_est=p_est, p_confidence=p_confidence, alpha_mode=alpha_mode,
        )
        if not result:
            return None

        bid_price, ask_price, as_action = result
        now   = time.time()
        # Freeze the expected spread at placement — used by SpreadCaptureTracker.
        expected_spread_c = round((ask_price - bid_price) * 100, 2)
        # Size: order_size = bankroll * 0.05, hard cap = min(order_size, bankroll * 0.2)
        notional_per_share = bid_price + 1.0 - ask_price
        res_mult = resolution_multiplier(days_to_end)
        bankroll = self.state.balance
        max_deploy = bankroll * (MAX_DEPLOYMENT_PCT / 100)
        floor = MIN_QUOTE_SIZE

        mid = (best_bid + best_ask) / 2

        # ── DIRECTIONAL: Kelly-sized one-sided order ──────────────────────────
        directional_side = ""
        if alpha_mode == "DIRECTIONAL" and p_confidence >= 0.25 and p_est > 0:
            if p_est > mid:
                kelly_f = max(0.0, (p_est - bid_price) / max(0.01, 1.0 - bid_price))
                kelly_shares = int(kelly_f * bankroll * DIRECTIONAL_KELLY_FRAC / bid_price)
                shares = max(floor, min(MAX_DIRECTIONAL_SHARES, kelly_shares))
                directional_side = "BID"
            else:
                no_price = 1.0 - ask_price
                kelly_f  = max(0.0, ((1.0 - p_est) - no_price) / max(0.01, 1.0 - no_price))
                kelly_shares = int(kelly_f * bankroll * DIRECTIONAL_KELLY_FRAC / max(0.01, no_price))
                shares = max(floor, min(MAX_DIRECTIONAL_SHARES, kelly_shares))
                directional_side = "ASK"
            # Apply hard cap: min(shares, bankroll*0.2 / price)
            if directional_side == "BID" and bid_price > 0:
                cap_shares = int((bankroll * ORDER_SIZE_CAP_PCT) / bid_price)
                shares = min(shares, max(floor, cap_shares))
            elif directional_side == "ASK":
                no_price = 1.0 - ask_price
                if no_price > 0:
                    cap_shares = int((bankroll * ORDER_SIZE_CAP_PCT) / no_price)
                    shares = min(shares, max(floor, cap_shares))
            log.info("DIRECTIONAL %s: %s  p_est=%.3f  mid=%.3f  edge=%+.1fc  "
                     "kelly_f=%.3f  shares=%d",
                     directional_side, slug[:40], p_est, mid, edge_cents, kelly_f, shares)
        elif bid_price > 0 and bankroll > 0:
            # Liquidity-based: size = min(10, liquidity*0.05) — provide liquidity, capture spread
            liquidity_usd = (bid_l1_depth + ask_l1_depth) / 2 if (bid_l1_depth or ask_l1_depth) else 0
            if liquidity_usd > 0 and mid > 0:
                liq_shares = int(liquidity_usd * SIZE_LIQUIDITY_PCT / mid)
                shares = max(floor, min(MAX_QUOTE_SIZE, liq_shares))
            else:
                order_usd = bankroll * ORDER_SIZE_PCT
                order_usd = min(order_usd, bankroll * ORDER_SIZE_CAP_PCT)
                shares = max(floor, int(order_usd / bid_price))
            shares = max(shares, math.ceil(MIN_ORDER_USD / bid_price) if bid_price > 0 else 0)
            shares = max(shares, math.ceil(MIN_ORDER_USD / (1.0 - ask_price)) if ask_price < 1.0 else 0)
            shares = round(shares * size_multiplier * res_mult)
            shares = max(floor, shares)
        else:
            shares = max(floor, round(SHARES_PER_QUOTE * size_multiplier * res_mult))

        # Cap by max notional per market
        if notional_per_share > 0:
            max_shares = MAX_NOTIONAL_PER_MARKET / notional_per_share
            shares = min(shares, max(floor, int(max_shares)))
            # Cap by bankroll: must fit within deployment limit
            deployed = sum(
                (q.bid_price + 1.0 - q.ask_price) * q.bid_size
                for q in self.state.active_quotes.values()
                if q.status in ("QUOTING", "ONE_LEG")
            )
            available = max(0, max_deploy - deployed)
            max_shares_by_cap = int(available / notional_per_share) if notional_per_share > 0 else 0
            if max_shares_by_cap < floor:
                log.info("BANKROLL — skip: %s  available=$%.2f  need≥%d shares (%.2f each)",
                         slug[:40], available, floor, notional_per_share)
                return None
            shares = min(shares, max_shares_by_cap)
        if res_mult < 1.0:
            log.debug("Resolution scale: days=%.1f → mult=%.2f  shares=%d",
                      days_to_end, res_mult, shares)

        # Multi-level: only use 3 levels if bankroll supports it (else 1 level = 8 shares)
        n_levels = max(1, QUOTE_LEVELS)
        min_for_multi = MIN_QUOTE_SIZE * n_levels
        if n_levels > 1 and (shares < min_for_multi or
                             (notional_per_share > 0 and shares * notional_per_share > bankroll)):
            n_levels = 1  # can't afford 3 levels — stay within bankroll
        shares_per_level = max(MIN_QUOTE_SIZE, shares // n_levels)

        # Build level ladders: bid1, bid2, bid3 / ask1, ask2, ask3
        bid_levels = []
        ask_levels = []
        for i in range(n_levels):
            bp = max(0.01, min(0.99, round(bid_price - i * TICK, 3)))
            ap = max(0.01, min(0.99, round(ask_price + i * TICK, 3)))
            bid_levels.append((bp, shares_per_level))
            ask_levels.append((ap, shares_per_level))

        total_shares = shares_per_level * n_levels
        qp = QuotePair(
            market_slug        = slug,
            question           = question[:60],
            token_yes          = token_yes,
            token_no           = token_no,
            bid_price          = bid_price,
            bid_size           = total_shares,
            ask_price          = ask_price,
            ask_size           = total_shares,
            bid_levels         = bid_levels,
            ask_levels         = ask_levels,
            bid_level_filled   = [False] * n_levels,
            ask_level_filled   = [False] * n_levels,
            status             = "PENDING",
            created_ts         = now,
            last_check_ts      = now,
            as_action          = as_action,
            adverse_sel_risk   = adverse_sel_risk,
            trade_imbalance    = trade_imbalance,
            staleness_risk_pct = staleness_risk_pct,
            spread_stability   = spread_stability,
            p_est              = p_est,
            p_confidence       = p_confidence,
            edge_cents         = edge_cents,
            alpha_mode         = alpha_mode,
            directional_side   = directional_side,
            extra_ticks_used   = 0,  # book-relative quoting: no staleness extra
            bid_l1_size_at_place = bid_l1_size,
            ask_l1_size_at_place = ask_l1_size,
            last_bid_l1_size   = bid_l1_size,
            last_ask_l1_size   = ask_l1_size,
            fair_value         = fair_value,
            mid_stale          = mid_stale,
            stale_direction    = stale_direction,
            cluster_id         = cluster_id,
            cluster_type       = cluster_type,
            detect_ts          = detect_ts if detect_ts > 0 else time.time(),
            expected_spread_c  = expected_spread_c,
            quote_velocity_c_hr= quote_velocity_c_hr,
            placement_mid      = placement_mid,
        )

        expected_profit = (ask_price - bid_price) * shares
        fv_note   = f"  fv={fair_value:.3f}★{stale_direction}" if mid_stale else ""
        edge_note = (f"  edge={edge_cents:+.1f}c p={p_est:.3f}[{alpha_mode}]"
                     if alpha_mode != "PASSIVE_MM" else "")
        log.info("Quote created [%s]: %s | bid=%.3f ask=%.3f spread=%.0fc "
                 "profit=$%.2f  as=%s  stale=%.0f%%  stab=%.2f%s%s",
                 as_action, question[:40],
                 bid_price, ask_price, spread_c, expected_profit,
                 as_action, staleness_risk_pct * 100, spread_stability,
                 fv_note, edge_note)
        return qp

    def place_quote(self, qp: QuotePair, executor) -> bool:
        """
        Submit both legs.  Returns True if both orders were accepted.

        Latency telemetry checkpoints captured here:
          on_send → just before first order submission
          on_ack  → just after bid order_id is confirmed
        The final checkpoint (on_book_visible) is set by manage_active_quotes
        in run_mm.py on the first book poll after the quote goes QUOTING.
        """
        slug = qp.market_slug

        if self.paper:
            # Paper: send/ack are instantaneous — record them together so the
            # detect→send and send→ack segments are measurable.
            self.latency.on_send(slug)
            qp.bid_order_id = f"paper-bid-{slug}-{int(time.time())}"
            qp.ask_order_id = f"paper-ask-{slug}-{int(time.time())}"
            qp.bid_ts = qp.ask_ts = time.time()
            self.latency.on_ack(slug)
            qp.status = "QUOTING"
            self.state.active_quotes[slug] = qp
            log.info("[paper] Quotes posted: %s  bid=%.3f  ask=%.3f",
                     slug, qp.bid_price, qp.ask_price)
            return True

        # ── Live path ─────────────────────────────────────────────────────
        self.latency.on_send(slug)          # timestamp: entering network call

        # DIRECTIONAL: post only the favorable leg (one-sided alpha capture)
        if qp.directional_side == "BID":
            bid_r = executor.place_limit_order(
                token_id=qp.token_yes, side="BUY",
                price=qp.bid_price, size=qp.bid_size,
            )
            if not bid_r.success:
                log.warning("Dir-BID failed %s: %s", slug, bid_r.error)
                qp.status = "CANCELLED"; qp.notes = f"dir_bid_fail:{bid_r.error}"
                return False
            qp.bid_order_id = bid_r.order_id
            qp.bid_ts       = time.time()
            qp.status       = "ONE_LEG"
            qp.bid_filled   = False
            self.latency.on_ack(slug)
            self.state.active_quotes[slug] = qp
            log.info("DIRECTIONAL BID: %s @ %.3f  %d sh  edge=%+.1fc",
                     slug, qp.bid_price, qp.bid_size, qp.edge_cents)
            return True

        if qp.directional_side == "ASK":
            no_price = round(1.0 - qp.ask_price, 3)
            ask_r = executor.place_limit_order(
                token_id=qp.token_no, side="BUY",
                price=no_price, size=qp.ask_size,
            )
            if not ask_r.success:
                log.warning("Dir-ASK failed %s: %s", slug, ask_r.error)
                qp.status = "CANCELLED"; qp.notes = f"dir_ask_fail:{ask_r.error}"
                return False
            qp.ask_order_id = ask_r.order_id
            qp.ask_ts       = time.time()
            qp.status       = "ONE_LEG"
            qp.ask_filled   = False
            self.latency.on_ack(slug)
            self.state.active_quotes[slug] = qp
            log.info("DIRECTIONAL ASK: %s @ %.3f  %d sh  edge=%+.1fc",
                     slug, qp.ask_price, qp.ask_size, qp.edge_cents)
            return True

        # ── Standard two-sided market-making (single or multi-level) ─────────
        use_levels = (qp.bid_levels and qp.ask_levels and
                      len(qp.bid_levels) > 0 and len(qp.ask_levels) > 0)

        if use_levels:
            # Multi-level: bid1, bid2, bid3 / ask1, ask2, ask3
            qp.bid_order_ids = []
            qp.ask_order_ids = []
            for i, (bp, sz) in enumerate(qp.bid_levels):
                r = executor.place_limit_order(
                    token_id=qp.token_yes, side="BUY",
                    price=bp, size=sz,
                )
                if not r.success:
                    log.warning("Bid L%d failed %s: %s — cancelling", i+1, slug, r.error)
                    executor.cancel_all(asset_id=qp.token_yes)
                    qp.status = "CANCELLED"; qp.notes = f"bid_L{i+1}_fail:{r.error}"
                    return False
                qp.bid_order_ids.append(r.order_id)
            qp.bid_order_id = qp.bid_order_ids[0] if qp.bid_order_ids else ""
            qp.bid_ts = time.time()
            self.latency.on_ack(slug)

            for i, (ap, sz) in enumerate(qp.ask_levels):
                no_price = round(1.0 - ap, 3)
                r = executor.place_limit_order(
                    token_id=qp.token_no, side="BUY",
                    price=no_price, size=sz,
                )
                if not r.success:
                    log.warning("Ask L%d failed %s: %s — cancelling bids", i+1, slug, r.error)
                    executor.cancel_all(asset_id=qp.token_yes)
                    executor.cancel_all(asset_id=qp.token_no)
                    qp.status = "CANCELLED"; qp.notes = f"ask_L{i+1}_fail:{r.error}"
                    return False
                qp.ask_order_ids.append(r.order_id)
            qp.ask_order_id = qp.ask_order_ids[0] if qp.ask_order_ids else ""
            qp.ask_ts = time.time()
            log.info("Quotes LIVE (multi): %s | bid %.3f/%.3f/%.3f  ask %.3f/%.3f/%.3f | %d sh [%s]",
                     slug,
                     qp.bid_levels[0][0], qp.bid_levels[1][0] if len(qp.bid_levels) > 1 else 0,
                     qp.bid_levels[2][0] if len(qp.bid_levels) > 2 else 0,
                     qp.ask_levels[0][0], qp.ask_levels[1][0] if len(qp.ask_levels) > 1 else 0,
                     qp.ask_levels[2][0] if len(qp.ask_levels) > 2 else 0,
                     qp.bid_size, qp.alpha_mode)
        else:
            # Single level (backward compat)
            bid_r = executor.place_limit_order(
                token_id=qp.token_yes, side="BUY",
                price=qp.bid_price, size=qp.bid_size,
            )
            if not bid_r.success:
                log.warning("Bid failed %s: %s", slug, bid_r.error)
                qp.status = "CANCELLED"; qp.notes = f"bid_fail:{bid_r.error}"
                return False

            qp.bid_order_id = bid_r.order_id
            qp.bid_ts       = time.time()
            self.latency.on_ack(slug)

            no_price = round(1.0 - qp.ask_price, 3)
            ask_r = executor.place_limit_order(
                token_id=qp.token_no, side="BUY",
                price=no_price, size=qp.ask_size,
            )
            if not ask_r.success:
                log.warning("Ask failed %s: %s — cancelling bid", slug, ask_r.error)
                executor.cancel_all(asset_id=qp.token_yes)
                qp.status = "CANCELLED"; qp.notes = f"ask_fail:{ask_r.error}"
                return False

            qp.ask_order_id = ask_r.order_id
            qp.ask_ts       = time.time()
            log.info("Quotes LIVE: %s | bid=%.3f (YES)  ask=%.3f (NO@%.3f) | %d sh  [%s]",
                     slug, qp.bid_price, qp.ask_price, no_price, qp.bid_size, qp.alpha_mode)

        qp.status = "QUOTING"
        self.state.active_quotes[slug] = qp
        return True

    # ── fill detection ─────────────────────────────────────────────────────

    def _record_leg_fill(self, qp: QuotePair, side: str) -> None:
        """
        Called when a leg fill is first detected (live or paper).
        Records `first_fill_ts` and `first_fill_side` for analytics timing.
        """
        if qp.first_fill_ts == 0.0:
            qp.first_fill_ts   = time.time()
            qp.first_fill_side = side

    def check_fills(self, qp: QuotePair, executor) -> str:
        """Poll executor for fill status.  Returns new status string."""
        if self.paper:
            return qp.status   # paper fills handled by simulator in run_mm.py

        try:
            open_ids = {str(o.get("id", o.get("orderID", "")))
                        for o in executor.get_open_orders()}
        except Exception:
            return qp.status

        # Multi-level: check each bid level
        if qp.bid_order_ids:
            for i, oid in enumerate(qp.bid_order_ids):
                if oid and oid not in open_ids and i < len(qp.bid_level_filled) and not qp.bid_level_filled[i]:
                    qp.bid_level_filled[i] = True
                    sz = qp.bid_levels[i][1] if i < len(qp.bid_levels) else 0
                    qp.bid_filled_size += sz
                    if not qp.bid_filled:
                        qp.bid_filled = True
                        qp.bid_fill_price = qp.bid_levels[i][0] if i < len(qp.bid_levels) else qp.bid_price
                        self._record_leg_fill(qp, "BID")
                    log.info("BID L%d FILLED: %s @ %.3f  %d sh", i+1, qp.market_slug,
                             qp.bid_levels[i][0] if i < len(qp.bid_levels) else qp.bid_price, int(sz))
        elif qp.bid_order_id not in open_ids and not qp.bid_filled:
            qp.bid_filled     = True
            qp.bid_fill_price = qp.bid_price
            qp.bid_filled_size = qp.bid_size
            self._record_leg_fill(qp, "BID")
            log.info("BID FILLED: %s @ %.3f", qp.market_slug, qp.bid_price)

        # Multi-level: check each ask level
        if qp.ask_order_ids:
            for i, oid in enumerate(qp.ask_order_ids):
                if oid and oid not in open_ids and i < len(qp.ask_level_filled) and not qp.ask_level_filled[i]:
                    qp.ask_level_filled[i] = True
                    sz = qp.ask_levels[i][1] if i < len(qp.ask_levels) else 0
                    qp.ask_filled_size += sz
                    if not qp.ask_filled:
                        qp.ask_filled = True
                        qp.ask_fill_price = qp.ask_levels[i][0] if i < len(qp.ask_levels) else qp.ask_price
                        self._record_leg_fill(qp, "ASK")
                    log.info("ASK L%d FILLED: %s @ %.3f  %d sh", i+1, qp.market_slug,
                             qp.ask_levels[i][0] if i < len(qp.ask_levels) else qp.ask_price, int(sz))
        elif qp.ask_order_id not in open_ids and not qp.ask_filled:
            qp.ask_filled     = True
            qp.ask_fill_price = qp.ask_price
            qp.ask_filled_size = qp.ask_size
            self._record_leg_fill(qp, "ASK")
            log.info("ASK FILLED: %s @ %.3f", qp.market_slug, qp.ask_price)

        if qp.bid_filled and qp.ask_filled:
            qp.status = "DONE"
            matched   = min(qp.bid_filled_size, qp.ask_filled_size) if (qp.bid_filled_size or qp.ask_filled_size) else min(qp.bid_size, qp.ask_size)
            qp.pnl    = (qp.ask_fill_price - qp.bid_fill_price) * matched
            log.info("BOTH LEGS: %s pnl=$%.4f (matched=%.0f)", qp.market_slug, qp.pnl, matched)
        elif qp.bid_filled or qp.ask_filled:
            qp.status = "ONE_LEG"
        else:
            qp.status = "QUOTING"

        return qp.status

    # ── one-leg inventory management ───────────────────────────────────────

    def on_first_fill(self, qp: QuotePair,
                      current_bid: float, current_ask: float):
        """
        Called once when the first leg fills.
        Starts the drift monitor and records the fill mid.
        """
        side = "YES" if qp.bid_filled else "NO"
        qp.fill_mid = (current_bid + current_ask) / 2

        monitor = PostFillDriftMonitor()
        monitor.on_fill(side, current_bid, current_ask)
        self._drift[qp.market_slug] = monitor

    def handle_one_leg(self, qp: QuotePair, executor,
                       current_bid: float, current_ask: float) -> str:
        """
        Manage the open leg after one side fills.

        Decision tree (in priority order):
          1. Per-market MTM loss cutoff  → force exit if unrealized loss > X%
          2. Emergency adverse drift     → immediate force exit
          3. Quote overtaken by market   → immediate force exit
          4. Inventory age timeout       → force exit
          5. Warn on adverse drift threshold
          6. Apply inventory skew to exit price
          7. Wait
        """
        now = time.time()

        # Initialise drift monitor on first call after fill
        if qp.market_slug not in self._drift:
            self.on_first_fill(qp, current_bid, current_ask)

        # 1. Per-market MTM loss cutoff — auto-pause if unrealized loss > X%
        if qp.bid_filled and not qp.ask_filled:
            # We hold YES; loss if current_bid < bid_fill_price
            if current_bid < qp.bid_fill_price and qp.bid_fill_price > 0:
                loss_pct = (qp.bid_fill_price - current_bid) / qp.bid_fill_price * 100
                if loss_pct >= MTM_LOSS_CUTOFF_PCT:
                    log.warning("MTM CUTOFF: %s  unrealized loss %.1f%% (bid=%.3f mkt=%.3f) → exit",
                                qp.market_slug, loss_pct, qp.bid_fill_price, current_bid)
                    qp.notes = (qp.notes or "") + "|mtm_cutoff"
                    return self._force_exit_yes(qp, executor, current_bid)
        elif qp.ask_filled and not qp.bid_filled:
            # We hold NO (sold YES); loss if current_ask > ask_fill_price
            if current_ask > qp.ask_fill_price and qp.ask_fill_price > 0:
                loss_pct = (current_ask - qp.ask_fill_price) / qp.ask_fill_price * 100
                if loss_pct >= MTM_LOSS_CUTOFF_PCT:
                    log.warning("MTM CUTOFF: %s  unrealized loss %.1f%% (ask=%.3f mkt=%.3f) → exit",
                                qp.market_slug, loss_pct, qp.ask_fill_price, current_ask)
                    qp.notes = (qp.notes or "") + "|mtm_cutoff"
                    return self._force_exit_no(qp, executor, current_ask)

        monitor = self._drift.get(qp.market_slug)
        if monitor:
            drift = monitor.update(current_bid, current_ask)
            qp.post_fill_drift_c = round(drift, 3)
            qp.worst_drift_c     = round(monitor.worst_drift, 3)

            if monitor.should_force_exit():
                log.warning("ADVERSE SELECTION EMERGENCY: %s  drift=%.2fc → exit",
                            qp.market_slug, monitor.worst_drift)
                qp.notes += " |adverse_exit"
                if qp.bid_filled:
                    return self._force_exit_yes(qp, executor, current_bid)
                else:
                    return self._force_exit_no(qp, executor, current_ask)

            if monitor.should_warn():
                log.warning("ADVERSE DRIFT WARNING: %s  worst=%.2fc",
                            qp.market_slug, monitor.worst_drift)

        # Age since the fill (for timeout)
        fill_ts = qp.first_fill_ts if qp.first_fill_ts > 0 else (qp.bid_ts if qp.bid_filled else qp.ask_ts)
        age     = now - fill_ts
        inv     = (qp.bid_filled_size if qp.bid_filled_size > 0 else qp.bid_size) if qp.bid_filled else (qp.ask_filled_size if qp.ask_filled_size > 0 else qp.ask_size)

        # Fast exit: if filled and no hedge in 20s → cross spread (market order)
        if age >= HEDGE_CROSS_SPREAD_S:
            log.info("CROSS SPREAD: %s held %.0fs → exit at market", qp.market_slug, age)
            qp.notes = (qp.notes or "") + "|cross_spread"
            if qp.bid_filled:
                return self._cross_spread_exit_yes(qp, executor, current_bid, current_ask)
            else:
                return self._cross_spread_exit_no(qp, executor, current_bid, current_ask)

        # Apply inventory skew — skew = inventory * 0.2 (cents)
        if qp.bid_filled and not qp.ask_filled:
            skewed_ask = self._inv_skew.skewed_exit_price("YES", qp.ask_price, inv)
            qp.inventory_skew_c = round((qp.ask_price - skewed_ask) * 100, 2)

            if age > MAX_INVENTORY_AGE_S:
                log.warning("TIMEOUT: %s held YES %.0fs → exit", qp.market_slug, age)
                return self._force_exit_yes(qp, executor, current_bid)

            # Market moved — our ask is above the new market ask
            if current_ask <= skewed_ask:
                log.info("Ask overtaken %s (mkt=%.3f ≤ skewed=%.3f) → exit",
                         qp.market_slug, current_ask, skewed_ask)
                return self._force_exit_yes(qp, executor, current_bid)

        elif qp.ask_filled and not qp.bid_filled:
            skewed_bid = self._inv_skew.skewed_exit_price("NO", qp.bid_price, inv)
            qp.inventory_skew_c = round((skewed_bid - qp.bid_price) * 100, 2)

            if age > MAX_INVENTORY_AGE_S:
                log.warning("TIMEOUT: %s held NO %.0fs → exit", qp.market_slug, age)
                return self._force_exit_no(qp, executor, current_ask)

            if current_bid >= skewed_bid:
                log.info("Bid overtaken %s (mkt=%.3f ≥ skewed=%.3f) → exit",
                         qp.market_slug, current_bid, skewed_bid)
                return self._force_exit_no(qp, executor, current_ask)

        return "ONE_LEG"

    # ── cross-spread exit (20s no hedge → market order) ──────────────────────

    def _cross_spread_exit_yes(self, qp: QuotePair, executor,
                               current_bid: float, current_ask: float) -> str:
        """Sell YES at market (cross spread) to close position."""
        if self.paper or executor is None:
            qp.ask_filled     = True
            qp.ask_fill_price = current_bid
        else:
            executor.cancel_all(asset_id=qp.token_no)
            size = qp.bid_filled_size if qp.bid_filled_size > 0 else qp.bid_size
            # SELL: amount = shares (Polymarket API)
            r = executor.place_market_order(
                token_id=qp.token_yes, side="SELL",
                amount=size, worst_price=max(0.01, current_bid - 0.05),
            )
            if r.success:
                qp.ask_filled     = True
                qp.ask_fill_price = current_bid  # approximate
            else:
                # Fallback to limit at bid
                r = executor.place_limit_order(
                    token_id=qp.token_yes, side="SELL",
                    price=max(0.01, current_bid), size=size,
                )
                if r.success:
                    qp.ask_filled     = True
                    qp.ask_fill_price = current_bid
                else:
                    log.error("Cross spread YES failed: %s", r.error)
                    return "ONE_LEG"

        size = qp.bid_filled_size if qp.bid_filled_size > 0 else qp.bid_size
        qp.pnl    = (qp.ask_fill_price - qp.bid_fill_price) * size
        qp.status = "DONE"
        self._drift.pop(qp.market_slug, None)
        log.info("CROSS SPREAD YES: %s @ %.3f  pnl=$%.4f",
                 qp.market_slug, qp.ask_fill_price, qp.pnl)
        return "DONE"

    def _cross_spread_exit_no(self, qp: QuotePair, executor,
                              current_bid: float, current_ask: float) -> str:
        """Close NO by buying YES at market (cross spread)."""
        if self.paper or executor is None:
            qp.bid_filled     = True
            qp.bid_fill_price = current_ask
        else:
            executor.cancel_all(asset_id=qp.token_yes)
            size = qp.ask_filled_size if qp.ask_filled_size > 0 else qp.ask_size
            # BUY: amount = dollar amount in USDC (Polymarket API)
            amount_usd = max(0.01, size * current_ask)
            r = executor.place_market_order(
                token_id=qp.token_yes, side="BUY",
                amount=amount_usd, worst_price=min(0.99, current_ask + 0.05),
            )
            if r.success:
                qp.bid_filled     = True
                qp.bid_fill_price = current_ask
            else:
                r = executor.place_limit_order(
                    token_id=qp.token_yes, side="BUY",
                    price=min(0.99, current_ask), size=size,
                )
                if r.success:
                    qp.bid_filled     = True
                    qp.bid_fill_price = current_ask
                else:
                    log.error("Cross spread NO failed: %s", r.error)
                    return "ONE_LEG"

        size = qp.ask_filled_size if qp.ask_filled_size > 0 else qp.ask_size
        qp.pnl    = (qp.ask_fill_price - qp.bid_fill_price) * size
        qp.status = "DONE"
        self._drift.pop(qp.market_slug, None)
        log.info("CROSS SPREAD NO: %s @ %.3f  pnl=$%.4f",
                 qp.market_slug, qp.bid_fill_price, qp.pnl)
        return "DONE"

    # ── force exits ────────────────────────────────────────────────────────

    def _force_exit_yes(self, qp: QuotePair, executor,
                        sell_price: float) -> str:
        """Sell YES position at the current market bid."""
        size = qp.bid_filled_size if qp.bid_filled_size > 0 else qp.bid_size
        if self.paper:
            qp.ask_filled     = True
            qp.ask_fill_price = sell_price
        else:
            executor.cancel_all(asset_id=qp.token_no)
            r = executor.place_limit_order(
                token_id=qp.token_yes, side="SELL",
                price=max(0.01, sell_price), size=size,
            )
            if r.success:
                qp.ask_filled     = True
                qp.ask_fill_price = sell_price
            else:
                log.error("Force exit YES failed: %s", r.error)
                return "ONE_LEG"

        qp.pnl    = (qp.ask_fill_price - qp.bid_fill_price) * size
        qp.status = "DONE"
        qp.notes  = qp.notes + "|force_exit_yes"
        self._drift.pop(qp.market_slug, None)
        log.info("FORCE EXIT YES: %s @ %.3f  pnl=$%.4f",
                 qp.market_slug, sell_price, qp.pnl)
        return "DONE"

    def _force_exit_no(self, qp: QuotePair, executor,
                       sell_price: float) -> str:
        """Close NO position (buy YES at current ask to close)."""
        size = qp.ask_filled_size if qp.ask_filled_size > 0 else qp.ask_size
        if self.paper:
            qp.bid_filled     = True
            qp.bid_fill_price = sell_price
        else:
            executor.cancel_all(asset_id=qp.token_yes)
            no_sell_price = round(1.0 - sell_price, 3)
            r = executor.place_limit_order(
                token_id=qp.token_no, side="SELL",
                price=max(0.01, no_sell_price), size=size,
            )
            if r.success:
                qp.bid_filled     = True
                qp.bid_fill_price = sell_price
            else:
                log.error("Force exit NO failed: %s", r.error)
                return "ONE_LEG"

        qp.pnl    = (qp.ask_fill_price - qp.bid_fill_price) * size
        qp.status = "DONE"
        qp.notes  = qp.notes + "|force_exit_no"
        self._drift.pop(qp.market_slug, None)
        log.info("FORCE EXIT NO: %s @ %.3f  pnl=$%.4f",
                 qp.market_slug, sell_price, qp.pnl)
        return "DONE"

    # ── lifecycle helpers ──────────────────────────────────────────────────

    def complete_quote(self, qp: QuotePair):
        now = time.time()
        if qp.done_ts == 0.0:
            qp.done_ts = now

        self.state.active_quotes.pop(qp.market_slug, None)
        self._drift.pop(qp.market_slug, None)
        # Subtract fees from PnL (Polymarket: makers 0%, takers 0.1%; we post limit = maker)
        fee = _compute_fee(qp)
        if fee > 0:
            qp.pnl -= fee
            log.debug("Fee deducted: $%.4f  net_pnl=$%.4f", fee, qp.pnl)
        self.state.completed.append(qp)
        self.state.total_pnl += qp.pnl
        self.state.balance  += qp.pnl   # keep internal balance in sync with realized PnL
        self.state.total_trades += 1

        # ── Build FillRecord for analytics ────────────────────────────────
        attribution   = _attribute_fill(qp)
        actual_c      = (qp.ask_fill_price - qp.bid_fill_price) * 100
        cap_pct       = (actual_c / qp.expected_spread_c
                         if qp.expected_spread_c > 0 else 0.0)
        exp_profit    = (qp.expected_spread_c / 100) * min(qp.bid_size, qp.ask_size)
        # Timing: if first_fill_ts wasn't set (paper fast-fill), approximate from done_ts
        first_fill_ts = qp.first_fill_ts if qp.first_fill_ts > 0 else qp.done_ts
        t_first       = max(0.0, first_fill_ts   - qp.created_ts)
        t_done        = max(0.0, qp.done_ts       - qp.created_ts)

        # Real fill analytics: queue at post, mid move during queue, fill type
        mid_place = getattr(qp, "placement_mid", 0.0)
        mid_fill  = getattr(qp, "fill_mid", 0.0)
        mid_move_c = round((mid_fill - mid_place) * 100, 2) if mid_place > 0 and mid_fill > 0 else 0.0
        fill_type = f"{qp.first_fill_side}_FIRST" if qp.first_fill_side else "UNKNOWN"

        rec = FillRecord(
            slug                 = qp.market_slug,
            question             = qp.question,
            cluster_id           = qp.cluster_id,
            cluster_type         = qp.cluster_type,
            expected_spread_c    = qp.expected_spread_c,
            actual_spread_c      = round(actual_c, 2),
            spread_capture_pct   = round(cap_pct, 4),
            expected_profit      = round(exp_profit, 4),
            actual_profit        = round(qp.pnl, 4),
            attribution          = attribution,
            adverse_sel_risk     = qp.adverse_sel_risk,
            staleness_risk_pct   = qp.staleness_risk_pct,
            trade_imbalance      = qp.trade_imbalance,
            worst_drift_c        = qp.worst_drift_c,
            as_action            = qp.as_action,
            spread_stability     = qp.spread_stability,
            quote_velocity_c_hr   = qp.quote_velocity_c_hr,
            time_to_first_fill_s = round(t_first, 1),
            time_to_done_s       = round(t_done,  1),
            first_fill_side      = qp.first_fill_side,
            queue_bid_at_post    = qp.bid_l1_size_at_place,
            queue_ask_at_post    = qp.ask_l1_size_at_place,
            mid_at_placement     = mid_place,
            mid_at_first_fill    = mid_fill,
            mid_move_during_queue_c = mid_move_c,
            fill_type            = fill_type,
        )
        self._spread_capture.record(rec)
        self._survival.record(rec)
        self._toxic_pm.record_if_toxic(rec)
        self._cap_eff.record_completion(qp)
        self._fill_analytics.log(rec)
        _alert_monitor(rec, qp.pnl, event="fill")

        log.info("Completed [%s]: %s  pnl=$%.4f  capture=%+.0f%%  age=%.0fs  "
                 "queue_b=%.0f queue_a=%.0f  midΔ=%.1fc  %s",
                 attribution, qp.market_slug, qp.pnl, cap_pct * 100, t_done,
                 rec.queue_bid_at_post, rec.queue_ask_at_post,
                 rec.mid_move_during_queue_c, rec.fill_type)
        log.info("Running totals: pnl=$%.4f  trades=%d",
                 self.state.total_pnl, self.state.total_trades)

    def cancel_quote(self, qp: QuotePair, executor, reason: str = ""):
        if not self.paper:
            executor.cancel_all(asset_id=qp.token_yes)
            executor.cancel_all(asset_id=qp.token_no)
        qp.status = "CANCELLED"
        qp.notes  = reason
        self.state.active_quotes.pop(qp.market_slug, None)
        self._drift.pop(qp.market_slug, None)
        log.info("Cancelled: %s  reason=%s", qp.market_slug, reason)

    def refresh_quote(self, qp: QuotePair, executor,
                      mkt_bid: float, mkt_ask: float,
                      bid_l1_size: float = 0.0, ask_l1_size: float = 0.0) -> bool:
        """
        Cancel existing orders and repost at current market. Recomputes size from liquidity.
        """
        if self.paper:
            qp.created_ts = time.time()
            qp.last_check_ts = time.time()
            return True
        executor.cancel_all(asset_id=qp.token_yes)
        executor.cancel_all(asset_id=qp.token_no)
        spread_c = (mkt_ask - mkt_bid) * 100
        result = self.compute_quote_prices(
            mkt_bid, mkt_ask, spread_c,
            qp.adverse_sel_risk, qp.trade_imbalance, qp.staleness_risk_pct,
            inventory=0.0,
            p_est=qp.p_est, p_confidence=qp.p_confidence, alpha_mode=qp.alpha_mode,
        )
        if not result:
            log.info("Refresh skip %s: no valid prices", qp.market_slug)
            return False
        bid_price, ask_price, _ = result
        qp.bid_price = bid_price
        qp.ask_price = ask_price
        mid = (mkt_bid + mkt_ask) / 2
        # Recompute size from liquidity: min(10, liquidity*0.05)
        liquidity_usd = (mkt_bid * bid_l1_size + mkt_ask * ask_l1_size) / 2 if (bid_l1_size or ask_l1_size) else 0
        if liquidity_usd > 0 and mid > 0:
            liq_shares = int(liquidity_usd * SIZE_LIQUIDITY_PCT / mid)
            shares = max(MIN_QUOTE_SIZE, min(MAX_QUOTE_SIZE, liq_shares))
        else:
            shares = max(MIN_QUOTE_SIZE, min(MAX_QUOTE_SIZE, int(qp.bid_size)))
        n_levels = max(1, QUOTE_LEVELS)
        if n_levels > 1 and shares < MIN_QUOTE_SIZE * n_levels:
            n_levels = 1
        shares_per_level = max(MIN_QUOTE_SIZE, shares // n_levels)
        total_shares = shares_per_level * n_levels
        qp.bid_size = total_shares
        qp.ask_size = total_shares
        qp.bid_levels = [(max(0.01, min(0.99, round(bid_price - i * TICK, 3))), shares_per_level) for i in range(n_levels)]
        qp.ask_levels = [(max(0.01, min(0.99, round(ask_price + i * TICK, 3))), shares_per_level) for i in range(n_levels)]
        qp.bid_level_filled = [False] * n_levels
        qp.ask_level_filled = [False] * n_levels
        qp.bid_order_ids = []
        qp.ask_order_ids = []
        qp.created_ts = time.time()
        qp.last_check_ts = time.time()
        return self.place_quote(qp, executor)

    def cancel_all(self, executor):
        for slug in list(self.state.active_quotes.keys()):
            self.cancel_quote(self.state.active_quotes[slug], executor,
                              "emergency_cancel_all")
        log.warning("ALL QUOTES CANCELLED")

    def summary(self) -> str:
        active = len(self.state.active_quotes)
        done   = len(self.state.completed)
        wins   = sum(1 for q in self.state.completed if q.pnl > 0)
        wr     = wins / done * 100 if done else 0
        return (f"Active: {active}  Completed: {done} "
                f"(W:{wins} L:{done-wins} WR:{wr:.0f}%)  "
                f"PnL: ${self.state.total_pnl:+.4f}")

    def print_analytics_report(self) -> None:
        """
        Print fill analytics:
          1. Spread Capture vs Expected Value
          2. Survival Curve (time-to-fill distribution)
          3. Toxic Fill Post-Mortem (ADVERSE/DRIFT_EXIT/STALE deep dive)
          4. Capital Efficiency (ROI on notional traded)
          5. Latency telemetry
        """
        self._spread_capture.print_report()
        self._survival.print_report(segment_by="attribution")
        self._toxic_pm.print_report()
        self._cap_eff.print_report(self.state.total_pnl, self.state.active_quotes)
        self.latency.print_report()
