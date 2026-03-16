"""
reprice_lag.py — Measures Polymarket stale-book persistence time.
=========================================================================

WHY YOUR 38ms RESULT WAS NOISE
───────────────────────────────
At threshold 0.005%, a BTC "move" fires on almost every tick (~every 100ms).
Polymarket also reprices continuously for its own reasons. You are measuring
the gap between two independent Poisson processes, not a causal reprice lag.
Result is dominated by the inter-event time of Poly ticks, not the latency
between BTC information and Poly reaction.

WHAT WE MEASURE INSTEAD
────────────────────────
At each significant BTC move (≥ MIN_BTC_MOVE_PCT):
  1. Compute the Gaussian fair value of the UP token given BTC's current
     position vs strike:  fair = Φ(pct_diff / sigma)
  2. Compute the gap:     gap = |fair - poly_mid|
  3. If gap > MIN_EDGE_GAP (5%) at BTC move time → open a "stale window"
  4. Track poly_mid tick-by-tick until gap falls below CLOSE_GAP (2%)
  5. Δt = time from BTC move to gap close = the actual edge window

This measures: "how long does the mispricing persist after a real BTC move?"
which is exactly the window your AWS order must fit inside.

Segments results by BTC move size:
  SMALL   0.03–0.05%   (just above noise floor)
  MEDIUM  0.05–0.10%
  LARGE   > 0.10%      (flash move — largest edge windows)

Output:
  logs/reprice_lag_raw.jsonl     one JSON line per event
  logs/reprice_lag_summary.json  full statistics

Run on AWS Ireland server:
  python3 reprice_lag.py --duration 600    # 10 min default
  python3 reprice_lag.py --duration 1800   # 30 min for quiet markets
"""

import asyncio
import aiohttp
import websockets
import json
import ssl
import time
import os
import math
import argparse
import statistics
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

# ── config ────────────────────────────────────────────────────────────────────
GAMMA_API   = "https://gamma-api.polymarket.com"
CLOB_WS     = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BINANCE_WS  = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"

# BTC move must be meaningful — below this you're measuring noise
MIN_BTC_MOVE_PCT  = 0.03    # 0.03% = $29 on $97k BTC

# At BTC move time, this gap must exist for the event to be "real arb"
MIN_EDGE_GAP      = 0.04    # fair_value - poly_mid must exceed 4% to be worth trading

# Gap is considered "closed" (Poly repriced) when it falls below this
CLOSE_GAP         = 0.02    # 2% — fair value and poly_mid agree

# Abandon tracking after this long (Poly might never reprice if move reverses)
MAX_TRACK_S       = 15.0

# Debounce: don't open a new stale window until previous one closes/expires
DEBOUNCE_S        = 3.0

SIGMA_5MIN_PCT    = 0.24    # empirical 5-min BTC sigma from Kaggle data
SIGMA_FLOOR_PCT   = 0.07    # minimum sigma (LOW_VOL regime)

DEFAULT_DURATION  = 600
WINDOW_S          = 300

LOG_DIR     = "logs"
RAW_LOG     = os.path.join(LOG_DIR, "reprice_lag_raw.jsonl")
SUMMARY_LOG = os.path.join(LOG_DIR, "reprice_lag_summary.json")

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE


# ── fair value math ───────────────────────────────────────────────────────────

def _ndtr(x: float) -> float:
    """Gaussian CDF — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_fair_value(btc: float, strike: float, time_remaining: float) -> float:
    """
    P(BTC > strike at expiry) under GBM.
    Uses empirical sigma from Kaggle 8yr dataset.
    """
    if strike <= 0 or btc <= 0 or time_remaining <= 0:
        return 0.5
    pct_diff  = (btc - strike) / strike * 100.0
    sigma_t   = max(SIGMA_5MIN_PCT * math.sqrt(time_remaining / WINDOW_S),
                    SIGMA_FLOOR_PCT)
    d         = max(-5.0, min(5.0, pct_diff / sigma_t))
    return _ndtr(d)


# ── market discovery ──────────────────────────────────────────────────────────

def current_window_start() -> int:
    return (int(time.time()) // WINDOW_S) * WINDOW_S


def seconds_remaining() -> float:
    return WINDOW_S - (time.time() % WINDOW_S)


async def fetch_btc_5min_market(offset_windows: int = 0) -> Optional[dict]:
    """
    Find active BTC 5-min UP+DOWN tokens.
    Slug pattern: btc-updown-5m-{window_start_timestamp}
    """
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode    = ssl.CERT_NONE
    timeout = aiohttp.ClientTimeout(total=10)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_ctx), timeout=timeout
    ) as session:
        for offset in range(offset_windows, offset_windows + 6):
            ws   = current_window_start() - offset * WINDOW_S
            slug = f"btc-updown-5m-{ws}"
            try:
                async with session.get(
                    f"{GAMMA_API}/events", params={"slug": slug}
                ) as r:
                    if r.status != 200:
                        continue
                    data = await r.json(content_type=None)
                    if not data:
                        continue
                    events = data if isinstance(data, list) else [data]
                    for ev in events:
                        markets = ev.get("markets", [])
                        if not markets:
                            continue
                        m       = markets[0]
                        raw_tok = m.get("clobTokenIds", "")
                        outcomes = m.get("outcomes", "")

                        if isinstance(raw_tok, str):
                            raw_tok = (raw_tok.strip().strip("[]")
                                       .replace('"', '').replace("'", ""))
                            tokens = [t.strip() for t in raw_tok.split(",") if t.strip()]
                        else:
                            tokens = list(raw_tok)

                        if len(tokens) < 2:
                            continue

                        up_idx, dn_idx = 0, 1
                        if isinstance(outcomes, str) and outcomes.strip():
                            parts = [p.strip().strip('[]"\'')
                                     for p in outcomes.split(",")]
                            for i, p in enumerate(parts):
                                if p.lower() == "up":
                                    up_idx, dn_idx = i, 1 - i
                                    break
                                if p.lower() == "down":
                                    dn_idx, up_idx = i, 1 - i
                                    break
                        elif isinstance(outcomes, list):
                            for i, p in enumerate(outcomes):
                                if str(p).lower() == "up":
                                    up_idx, dn_idx = i, 1 - i
                                    break

                        title = ev.get("title", slug)
                        print(f"  Market: {title}")
                        print(f"  UP:   {tokens[up_idx][:24]}...")
                        print(f"  DOWN: {tokens[dn_idx][:24]}...")
                        return {
                            "token_up":     tokens[up_idx],
                            "token_dn":     tokens[dn_idx],
                            "title":        title,
                            "window_start": ws,
                            "slug":         slug,
                        }
            except Exception as e:
                print(f"  Discovery error (offset={offset}): {e}")
    return None


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class StaleWindow:
    """One period of open mispricing after a BTC move."""
    open_ts:        float   # when BTC moved and gap appeared
    btc_price:      float
    strike:         float
    time_remaining: float
    btc_move_pct:   float   # signed move %
    fair_value:     float   # Φ(d) at open time
    poly_mid_open:  float   # Poly mid at open time (stale quote)
    gap_open:       float   # |fair_value - poly_mid| at open time
    direction:      str     # UP | DOWN (which token to buy)
    # filled on close:
    close_ts:       float  = 0.0
    poly_mid_close: float  = 0.0
    gap_close:      float  = 0.0
    delta_ms:       float  = 0.0  # time from open to gap close
    closed:         bool   = False
    expired:        bool   = False  # True if gap never closed within MAX_TRACK_S

    def size_bucket(self) -> str:
        a = abs(self.btc_move_pct)
        if a < 0.05:
            return "SMALL(0.03-0.05%)"
        if a < 0.10:
            return "MEDIUM(0.05-0.10%)"
        return "LARGE(>0.10%)"


class SharedData:
    def __init__(self):
        self.btc_prices:  deque  = deque(maxlen=120)   # (ts, price)
        self.btc_price:   float  = 0.0
        self.poly_mid:    float  = 0.0
        self.strike:      float  = 0.0    # inferred from early-window mid
        self.strike_set:  bool   = False

        self.open_window: Optional[StaleWindow] = None
        self.windows:     list   = []     # completed StaleWindow list
        self.last_open_ts: float = 0.0

        self.token_up:    str    = ""
        self.token_dn:    str    = ""
        self.window_start: int   = 0

        self.running: bool       = True
        self.lock = asyncio.Lock()


# ── strike inference ──────────────────────────────────────────────────────────

def try_set_strike(data: SharedData) -> bool:
    """
    Infer strike from poly_mid early in the window.
    When mid ≈ 0.50, BTC ≈ strike. Use BTC price directly.
    """
    if data.strike_set:
        return True
    secs_in = time.time() - data.window_start
    if data.btc_price <= 0 or data.poly_mid <= 0:
        return False
    if secs_in > 60:
        return False          # too late to reliably infer strike
    if 0.42 < data.poly_mid < 0.58:
        data.strike     = data.btc_price
        data.strike_set = True
        print(f"  [Strike] Locked at ${data.strike:,.2f}  "
              f"(mid={data.poly_mid:.3f}  T+{secs_in:.0f}s)")
        return True
    return False


# ── Binance feed ──────────────────────────────────────────────────────────────

async def binance_feed(data: SharedData, duration: float):
    deadline = time.time() + duration
    print("  [Binance] Connecting...")

    while time.time() < deadline and data.running:
        try:
            async with websockets.connect(
                BINANCE_WS, ssl=SSL_CTX,
                ping_interval=15, ping_timeout=10
            ) as ws:
                print("  [Binance] Connected — aggTrade stream")
                async for raw in ws:
                    if time.time() >= deadline:
                        data.running = False
                        return
                    try:
                        msg   = json.loads(raw)
                        price = float(msg.get("p", 0))
                        now   = time.time()
                        if price <= 0:
                            continue

                        async with data.lock:
                            data.btc_prices.append((now, price))
                            data.btc_price = price
                            try_set_strike(data)

                            # Find price ~1s ago
                            p1s = None
                            for ts_old, p_old in reversed(data.btc_prices):
                                if now - ts_old >= 1.0:
                                    p1s = p_old
                                    break
                            if not p1s or not data.strike_set:
                                continue

                            move_pct = (price - p1s) / p1s * 100.0

                            # ── Below threshold: check if open window expired ──
                            if abs(move_pct) < MIN_BTC_MOVE_PCT:
                                if data.open_window:
                                    age = now - data.open_window.open_ts
                                    if age > MAX_TRACK_S:
                                        w = data.open_window
                                        w.expired   = True
                                        w.delta_ms  = age * 1000
                                        data.windows.append(w)
                                        data.open_window = None
                                        print(f"  [EXPIRED] No close in "
                                              f"{age*1000:.0f}ms  gap_open={w.gap_open:.3f}")
                                continue

                            # ── Debounce ──────────────────────────────────────
                            if now - data.last_open_ts < DEBOUNCE_S:
                                continue

                            # ── Compute fair value & gap ──────────────────────
                            t_rem      = seconds_remaining()
                            fair       = compute_fair_value(price, data.strike, t_rem)
                            poly_mid   = data.poly_mid
                            if poly_mid <= 0:
                                continue

                            direction  = "UP" if move_pct > 0 else "DOWN"
                            if direction == "UP":
                                gap = fair - poly_mid
                            else:
                                gap = (1.0 - fair) - (1.0 - poly_mid)  # = poly_mid - fair

                            if gap < MIN_EDGE_GAP:
                                continue

                            # ── Open a stale window ───────────────────────────
                            data.open_window = StaleWindow(
                                open_ts        = now,
                                btc_price      = price,
                                strike         = data.strike,
                                time_remaining = t_rem,
                                btc_move_pct   = move_pct,
                                fair_value     = fair,
                                poly_mid_open  = poly_mid,
                                gap_open       = gap,
                                direction      = direction,
                            )
                            data.last_open_ts = now
                            print(f"  [OPEN]  {direction:4s}  btc={move_pct:+.4f}%  "
                                  f"fair={fair:.3f}  poly={poly_mid:.3f}  "
                                  f"gap={gap:.3f}  t_rem={t_rem:.0f}s")

                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

        except Exception as e:
            if data.running:
                print(f"  [Binance] {e} — reconnect in 2s")
                await asyncio.sleep(2)


# ── mid from book ─────────────────────────────────────────────────────────────

def _mid_from_book(bids, asks) -> float:
    if not bids or not asks:
        return 0.0
    try:
        def price(x):
            if isinstance(x, dict):
                return float(x.get("price") or x.get("p", 0))
            return float(x[0])
        best_bid = max(price(b) for b in bids)
        best_ask = min(price(a) for a in asks)
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return 0.0
        return (best_bid + best_ask) / 2.0
    except Exception:
        return 0.0


# ── Polymarket book feed ──────────────────────────────────────────────────────

async def poly_feed(data: SharedData, duration: float):
    deadline = time.time() + duration
    print("  [Poly] Starting book feed...")

    while time.time() < deadline and data.running:
        # Refresh token each window
        if not data.token_up or current_window_start() != data.window_start:
            print("  [Poly] Discovering current market...")
            market = await fetch_btc_5min_market()
            if not market:
                print("  [Poly] No market found — retry in 10s")
                await asyncio.sleep(10)
                continue
            async with data.lock:
                data.token_up     = market["token_up"]
                data.token_dn     = market["token_dn"]
                data.window_start = market["window_start"]
                data.strike_set   = False    # reset strike for new window
                data.open_window  = None
            print()

        token_up  = data.token_up
        token_dn  = data.token_dn
        subscribe = {"assets_ids": [token_up, token_dn], "type": "market"}

        try:
            async with websockets.connect(
                CLOB_WS, ssl=SSL_CTX,
                ping_interval=10, ping_timeout=8, max_size=2**22
            ) as ws:
                await ws.send(json.dumps(subscribe))
                print(f"  [Poly] Connected  UP={token_up[:16]}...")

                async for raw in ws:
                    if time.time() >= deadline:
                        data.running = False
                        return
                    if current_window_start() != data.window_start:
                        break   # reconnect with new window token

                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    raw = raw.strip()
                    if not raw or raw.upper() in ("PONG", "PING"):
                        continue
                    try:
                        msgs = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(msgs, dict):
                        msgs = [msgs]

                    for msg in msgs:
                        if not isinstance(msg, dict):
                            continue
                        etype = msg.get("event_type") or msg.get("type", "")
                        asset = msg.get("asset_id") or msg.get("market", "")
                        if asset != token_up:
                            continue
                        if etype not in ("book", "price_change", "best_bid_ask"):
                            continue

                        new_mid = _mid_from_book(
                            msg.get("bids") or [],
                            msg.get("asks") or []
                        )
                        if new_mid <= 0:
                            continue

                        now = time.time()
                        async with data.lock:
                            data.poly_mid = new_mid
                            try_set_strike(data)

                            # Check if open stale window has closed
                            w = data.open_window
                            if w is None:
                                continue

                            if w.direction == "UP":
                                fair_now = compute_fair_value(
                                    data.btc_price, data.strike, seconds_remaining())
                                gap_now  = fair_now - new_mid
                            else:
                                fair_now = compute_fair_value(
                                    data.btc_price, data.strike, seconds_remaining())
                                gap_now  = (1.0 - fair_now) - (1.0 - new_mid)

                            age = now - w.open_ts

                            if gap_now <= CLOSE_GAP:
                                # Gap closed — Poly repriced to match fair value
                                w.close_ts       = now
                                w.poly_mid_close = new_mid
                                w.gap_close      = gap_now
                                w.delta_ms       = age * 1000.0
                                w.closed         = True
                                data.windows.append(w)
                                data.open_window = None
                                print(f"  [CLOSE] {w.direction:4s}  "
                                      f"Δt={w.delta_ms:6.0f}ms  "
                                      f"gap {w.gap_open:.3f} → {gap_now:.3f}  "
                                      f"[{w.size_bucket()}]")
                            elif age > MAX_TRACK_S:
                                w.expired  = True
                                w.delta_ms = age * 1000.0
                                data.windows.append(w)
                                data.open_window = None
                                print(f"  [EXPIRED] {w.direction}  "
                                      f"gap still {gap_now:.3f} after {age*1000:.0f}ms")

        except Exception as e:
            if data.running:
                print(f"  [Poly] {e} — reconnect in 2s")
                await asyncio.sleep(2)


# ── report ────────────────────────────────────────────────────────────────────

def print_report(data: SharedData):
    os.makedirs(LOG_DIR, exist_ok=True)
    wins = data.windows

    with open(RAW_LOG, "w") as f:
        for w in wins:
            f.write(json.dumps(asdict(w)) + "\n")

    closed  = [w for w in wins if w.closed]
    expired = [w for w in wins if w.expired]

    print("\n" + "=" * 68)
    print("  REPRICE LAG (FAIR-VALUE GAP CLOSE TIME)")
    print("=" * 68)
    print(f"  Stale windows opened:       {len(wins)}")
    print(f"  Closed (Poly repriced):     {len(closed)}")
    print(f"  Expired (> {MAX_TRACK_S:.0f}s, gap never closed): {len(expired)}")

    if len(closed) < 5:
        print(f"\n  Not enough closed windows (need ≥5, got {len(closed)}).")
        print(f"  BTC may be moving too little. Try during volatile hours.")
        print(f"  Or run longer: python3 reprice_lag.py --duration 1800")
        return

    deltas = sorted(w.delta_ms for w in closed)
    n      = len(deltas)
    med    = statistics.median(deltas)
    avg    = statistics.mean(deltas)
    mn     = deltas[0]
    mx     = deltas[-1]
    p25    = deltas[max(0, int(n * 0.25))]
    p75    = deltas[min(n-1, int(n * 0.75))]
    p90    = deltas[min(n-1, int(n * 0.90))]

    print(f"\n  Gap-close time distribution (n={n}):")
    print(f"  {'min':8s}: {mn:8.0f} ms")
    print(f"  {'p25':8s}: {p25:8.0f} ms")
    print(f"  {'median':8s}: {med:8.0f} ms   ← KEY")
    print(f"  {'mean':8s}: {avg:8.0f} ms")
    print(f"  {'p75':8s}: {p75:8.0f} ms")
    print(f"  {'p90':8s}: {p90:8.0f} ms")
    print(f"  {'max':8s}: {mx:8.0f} ms")

    # By size bucket
    buckets = {}
    for w in closed:
        b = w.size_bucket()
        buckets.setdefault(b, []).append(w.delta_ms)
    if len(buckets) > 1:
        print(f"\n  By BTC move size:")
        for bname in sorted(buckets):
            ds  = sorted(buckets[bname])
            med_b = statistics.median(ds)
            print(f"  {bname:22s}: n={len(ds):3d}  median={med_b:6.0f}ms  "
                  f"p90={ds[min(len(ds)-1, int(len(ds)*0.90))]:6.0f}ms")

    # Histogram
    bounds = [0, 100, 200, 300, 500, 750, 1000, 2000, 5000, float("inf")]
    labels = ["<100ms","100-200ms","200-300ms","300-500ms",
              "500-750ms","750ms-1s","1-2s","2-5s",">5s"]
    counts = [0] * len(labels)
    for d in deltas:
        for i in range(len(bounds) - 1):
            if bounds[i] <= d < bounds[i + 1]:
                counts[i] += 1
                break

    print(f"\n  Histogram:")
    BAR = 36
    for label, count in zip(labels, counts):
        pct = count / n * 100
        bar = "█" * int(pct / 100 * BAR)
        print(f"  {label:12s} |{bar:<{BAR}} {count:4d} ({pct:5.1f}%)")

    # Expired windows (gap never closed) — these are the BEST trades
    if expired:
        exp_gaps = [w.gap_open for w in expired]
        print(f"\n  Expired windows ({len(expired)}) — gap never closed in "
              f"{MAX_TRACK_S:.0f}s:")
        print(f"  Mean gap at open: {statistics.mean(exp_gaps):.3f}  "
              f"These are the biggest edges.")

    # Verdict
    YOUR_LATENCY_MS = 150
    print(f"\n  AWS Ireland → Polymarket: ~{YOUR_LATENCY_MS}ms (measured)")
    print(f"  ─────────────────────────────────────────────────────────")

    capturable = sum(1 for d in deltas if d > YOUR_LATENCY_MS)
    capture_pct = capturable / n * 100 if n else 0

    if med < YOUR_LATENCY_MS:
        verdict = "NO_EDGE"
        msg = (f"Median reprice ({med:.0f}ms) < AWS latency ({YOUR_LATENCY_MS}ms).\n"
               f"  But {capture_pct:.0f}% of windows ({capturable}/{n}) are capturable "
               f"(Δt > {YOUR_LATENCY_MS}ms).\n"
               f"  These are the LARGE BTC moves. Focus strategy on those only.")
    elif med < 300:
        verdict = "MARGINAL"
        msg = (f"Tight window (median={med:.0f}ms). {capture_pct:.0f}% capturable.\n"
               f"  Reduce order submission latency below 50ms.")
    elif med < 750:
        verdict = "VIABLE"
        msg = (f"Clear window (median={med:.0f}ms). {capture_pct:.0f}% capturable.\n"
               f"  Strategy is sound with {YOUR_LATENCY_MS}ms AWS latency.")
    else:
        verdict = "STRONG_EDGE"
        msg = (f"Poly reprices very slowly (median={med:.0f}ms). {capture_pct:.0f}% capturable.")

    ICON = {"NO_EDGE": "❌", "MARGINAL": "⚠️", "VIABLE": "✅", "STRONG_EDGE": "✅✅"}
    print(f"\n  {ICON[verdict]} VERDICT: {verdict}")
    print(f"  {msg}")

    summary = {
        "n_windows_opened": len(wins),
        "n_closed":         len(closed),
        "n_expired":        len(expired),
        "lag_ms": {
            "min": mn, "p25": p25, "median": med, "mean": avg,
            "p75": p75, "p90": p90, "max": mx,
        },
        "capturable_pct":   round(capture_pct, 1),
        "aws_latency_ms":   YOUR_LATENCY_MS,
        "verdict":          verdict,
        "by_bucket":        {k: {
            "n": len(v),
            "median_ms": round(statistics.median(v), 1),
            "p90_ms": round(sorted(v)[min(len(v)-1, int(len(v)*0.90))], 1),
        } for k, v in buckets.items()},
    }
    with open(SUMMARY_LOG, "w") as f:
        import json as _j
        _j.dump(summary, f, indent=2)
    print(f"\n  Summary → {SUMMARY_LOG}")
    print("=" * 68)


# ── main ──────────────────────────────────────────────────────────────────────

async def run(duration: int):
    print("=" * 68)
    print("  POLYMARKET REPRICE LAG MEASUREMENT  (fair-value gap method)")
    print("=" * 68)
    print(f"  Duration:           {duration}s ({duration//60}m {duration%60}s)")
    print(f"  BTC move threshold: ≥{MIN_BTC_MOVE_PCT}% in 1s")
    print(f"  Min gap to open:    {MIN_EDGE_GAP:.0%} (fair vs poly_mid)")
    print(f"  Gap close level:    {CLOSE_GAP:.0%}")
    print(f"  Max track time:     {MAX_TRACK_S:.0f}s per window")
    print()

    os.makedirs(LOG_DIR, exist_ok=True)
    data = SharedData()

    print("  Initial market discovery...")
    market = await fetch_btc_5min_market()
    if not market:
        print("  Retrying in 15s...")
        await asyncio.sleep(15)
        market = await fetch_btc_5min_market()
    if not market:
        print("  ERROR: Cannot find BTC 5-min market. Exiting.")
        return

    data.token_up     = market["token_up"]
    data.token_dn     = market["token_dn"]
    data.window_start = market["window_start"]
    print()
    print("  Feeds starting. Press Ctrl+C to stop early.\n")
    print(f"  {'EVENT':<8}  {'DIRECTION':<6}  {'BTC MOVE':>10}  "
          f"{'FAIR':>6}  {'POLY':>6}  {'GAP':>6}  {'Δt':>8}")
    print(f"  {'-'*64}")

    try:
        await asyncio.gather(
            binance_feed(data, duration),
            poly_feed(data, duration),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n  Stopped early — computing results.")
    finally:
        data.running = False

    print_report(data)


def main():
    parser = argparse.ArgumentParser(
        description="Measure Polymarket reprice lag via fair-value gap method")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help=f"Seconds to run (default {DEFAULT_DURATION})")
    args = parser.parse_args()
    asyncio.run(run(args.duration))


if __name__ == "__main__":
    main()
