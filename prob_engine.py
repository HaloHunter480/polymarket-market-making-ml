"""
prob_engine.py — True Probability Estimator for Polymarket
===========================================================

The market mid price is NOT the true probability.  Informed traders,
momentum, and book depth all cause the quoted mid to lag.

This module estimates p_est — the true probability the market should be
trading at — by combining four independent signals:

  Signal 1  Recent trade VWAP (exponentially weighted, 5-min halflife)
            The strongest signal.  Actual fills reveal fair value.
            Heavily downweights old trades so momentum shows through.

  Signal 2  Print EMA (from scanner snapshot history)
            Where the book actually moves over the last 30 min.
            Already computed by scanner; no extra API call.

  Signal 3  Book depth asymmetry (bid_L1 vs ask_L1 sizes)
            More money sitting on bid side = YES demand > supply.
            Shifts estimate toward that side by up to 40% of half-spread.

  Signal 4  Order flow imbalance (from scanner regime signals)
            Persistent directional pressure from recent quote moves.
            Adds a small nudge (25% of half-spread) toward the flow side.

  Anchor    Market mid, always included at low weight.
            Prevents the estimate from straying too far.

Quoting modes (output):
  PASSIVE_MM    |edge| < 3c or confidence < 0.2
                Pure spread capture.  bid = mid - half_spread,  ask = mid + half_spread.
                No directional view — collect the spread and go home.

  SKEWED_MM     3c ≤ |edge| < 15c
                Two-sided quotes centred on p_est instead of mid.
                The leg on the "cheap" side is better priced → fills faster.
                We still capture spread on both legs, but skewed toward alpha.

  DIRECTIONAL   |edge| ≥ 15c
                We have a real edge.  Post ONLY the favorable leg.
                Size by a Kelly-inspired fraction of bankroll.
                This is not market-making — it's a limit order that extracts alpha.

Usage (scanner.py Phase 2):
  trades = await fetch_recent_trades(session, condition_id)
  prob   = estimate_true_probability(trades, print_ema, n_prints, ...)
  opp.p_est        = prob.p_est
  opp.alpha_mode   = prob.alpha_mode   # passed into market_maker.create_quote
  opp.edge_cents   = prob.edge_cents
"""

import math
import time
import aiohttp
from dataclasses import dataclass, field

DATA_API = "https://data-api.polymarket.com"

# ── Signal weights & thresholds ───────────────────────────────────────────────
TRADE_HALFLIFE_S         = 300.0   # 5-min exponential decay for trade VWAP
MIN_TRADES_CONFIDENT     = 5       # scale VWAP weight linearly up to this
MIN_PRINTS_CONFIDENT     = 3       # scale PrintEMA weight linearly up to this
DEPTH_INFLUENCE          = 0.40    # book depth shifts estimate by ≤40% of half-spread
FLOW_INFLUENCE           = 0.25    # order flow nudges estimate by ≤25% of half-spread
MIN_CONFIDENCE           = 0.20    # below this confidence → PASSIVE_MM regardless

EDGE_SKEW_THRESHOLD      = 3.0     # cents — enter SKEWED_MM (< 3c = PASSIVE)
EDGE_DIRECTIONAL_THRESHOLD = 15.0  # cents — enter DIRECTIONAL (< 15c = SKEWED)


@dataclass
class ProbEstimate:
    p_est:            float = 0.5         # estimated true probability of YES
    confidence:       float = 0.0         # 0–1 (higher = more signal agreement)
    edge_cents:       float = 0.0         # (p_est − mid) × 100, signed
    alpha_mode:       str   = "PASSIVE_MM"  # PASSIVE_MM | SKEWED_MM | DIRECTIONAL
    trade_vwap:       float = 0.0         # raw exp-weighted VWAP (0 = no data)
    n_trades:         int   = 0           # number of recent trades used
    signal_breakdown: str   = ""          # human-readable signal breakdown for logs


async def fetch_recent_trades(
    session: aiohttp.ClientSession,
    condition_id: str,
    n: int = 20,
) -> list[dict]:
    """
    Fetch last N trades for a market from data-api.polymarket.com.
    Returns list of {price, size, ts, side}.  Empty list on failure.
    """
    if not condition_id or not condition_id.startswith("0x"):
        return []
    try:
        async with session.get(
            f"{DATA_API}/trades",
            params={"market": condition_id, "limit": n},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            if r.status != 200:
                return []
            data = await r.json()
            if not isinstance(data, list):
                return []
            return [
                {
                    "price": float(t.get("price", 0.5) or 0.5),
                    "size":  float(t.get("size", 0) or 0),
                    "ts":    float(t.get("timestamp", 0) or 0),
                    "side":  t.get("side", ""),
                }
                for t in data
                if t.get("size") and float(t.get("size", 0)) > 0
            ]
    except Exception:
        return []


def _exp_weight(ts: float, now: float) -> float:
    """Exponential decay: weight = exp(-age / halflife). Recent = 1.0, old → 0."""
    age = max(0.0, now - ts)
    return math.exp(-age / TRADE_HALFLIFE_S)


def estimate_true_probability(
    trades: list[dict],
    print_ema: float,
    n_prints: int,
    trade_imbalance: float,
    bid_l1_size: float,
    ask_l1_size: float,
    current_bid: float,
    current_ask: float,
) -> ProbEstimate:
    """
    Estimate the true probability of YES for a market.

    All four signals are combined as a weighted average.  Confidence
    reflects how much non-anchor signal mass is present; low confidence
    → bot stays in pure market-making mode.

    Parameters
    ----------
    trades          List of {price, size, ts, side} from data-api
    print_ema       EMA of inferred fill prices from scanner history
    n_prints        Number of inferred prints (scanner quality indicator)
    trade_imbalance −1..+1 directional pressure from scanner regime signals
    bid_l1_size     Shares sitting at best bid
    ask_l1_size     Shares sitting at best ask
    current_bid     Best bid from CLOB book
    current_ask     Best ask from CLOB book
    """
    mid         = (current_bid + current_ask) / 2
    half_spread = (current_ask - current_bid) / 2
    now         = time.time()

    signals: list[tuple[float, float]] = []  # (price, weight)
    details: list[str] = []

    # ── Signal 1: Exponentially-weighted trade VWAP ───────────────────────────
    n_tr   = 0
    vwap   = 0.0
    if trades:
        w_sum = sum(t["size"] * _exp_weight(t["ts"], now) for t in trades)
        if w_sum > 0:
            vwap  = sum(t["price"] * t["size"] * _exp_weight(t["ts"], now)
                        for t in trades) / w_sum
            n_tr  = len(trades)
            w_vwap = 2.0 * min(1.0, n_tr / MIN_TRADES_CONFIDENT)
            signals.append((vwap, w_vwap))
            details.append(f"vwap={vwap:.3f}(n={n_tr},w={w_vwap:.1f})")

    # ── Signal 2: Print EMA (confirmed fills from scanner history) ───────────
    if n_prints >= 2 and 0.02 < print_ema < 0.98:
        w_print = 1.5 * min(1.0, n_prints / MIN_PRINTS_CONFIDENT)
        signals.append((print_ema, w_print))
        details.append(f"print={print_ema:.3f}(n={n_prints},w={w_print:.1f})")

    # ── Signal 3: Book depth asymmetry ───────────────────────────────────────
    total_depth = bid_l1_size + ask_l1_size
    if total_depth > 0 and half_spread > 0:
        depth_ratio = (bid_l1_size - ask_l1_size) / total_depth  # +1=all bids, -1=all asks
        book_price  = mid + depth_ratio * half_spread * DEPTH_INFLUENCE
        w_book      = 0.6
        signals.append((book_price, w_book))
        details.append(f"book={book_price:.3f}(ratio={depth_ratio:+.2f})")

    # ── Anchor: market mid at low weight (prevents wild swings) ─────────────
    signals.append((mid, 0.4))

    # ── Weighted average ──────────────────────────────────────────────────────
    total_w = sum(w for _, w in signals)
    p_raw   = sum(p * w for p, w in signals) / total_w

    # ── Adjustment: order flow pressure (additive nudge) ─────────────────────
    flow_adj = trade_imbalance * half_spread * FLOW_INFLUENCE
    p_est    = max(0.03, min(0.97, p_raw + flow_adj))
    if abs(flow_adj) > 0.0001:
        details.append(f"flow={flow_adj*100:+.2f}c")

    # ── Confidence: fraction of signal weight that is NOT the mid anchor ─────
    non_anchor_w = total_w - 0.4
    confidence   = round(min(1.0, non_anchor_w / 3.5), 3)

    # ── Edge and mode ────────────────────────────────────────────────────────
    edge_cents = round((p_est - mid) * 100, 2)

    if confidence < MIN_CONFIDENCE or abs(edge_cents) < EDGE_SKEW_THRESHOLD:
        alpha_mode = "PASSIVE_MM"
    elif abs(edge_cents) < EDGE_DIRECTIONAL_THRESHOLD:
        alpha_mode = "SKEWED_MM"
    else:
        alpha_mode = "DIRECTIONAL"

    return ProbEstimate(
        p_est            = round(p_est, 4),
        confidence       = confidence,
        edge_cents       = edge_cents,
        alpha_mode       = alpha_mode,
        trade_vwap       = round(vwap, 4),
        n_trades         = n_tr,
        signal_breakdown = "  ".join(details) if details else "anchor_only",
    )
