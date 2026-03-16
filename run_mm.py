"""
run_mm.py — Multi-Market Market Maker Orchestrator
====================================================

Connects:
  scanner.py      → finds wide-spread markets
  market_maker.py → manages two-sided quotes
  live_executor.py → places real orders on Polymarket

Loop (every SCAN_INTERVAL):
  1. Scan markets → rank by spread width + volume
  2. For new high-spread markets → create QuotePair and post
  3. For active quotes → check fills, manage one-sided inventory
  4. For stale/tight quotes → cancel and free up capital
  5. Print dashboard

Modes:
  --paper    Simulate fills (no real orders). DEFAULT.
  --live     Real orders via LiveExecutor. Requires .env credentials.

Run:
  python3 run_mm.py --paper              # paper trade (safe, test first)
  python3 run_mm.py --live --bankroll 50 # real money, $50 bankroll

Recovery / Restart:
  State is saved every cycle to logs/mm_state.json (cycle, total_pnl, active count,
  last 50 completed). On crash or Ctrl+C, restart with the same --bankroll. The
  process does NOT restore open orders from disk — Polymarket orders persist on the
  exchange. After restart: 1) Cancel any stale orders via Polymarket UI if needed.
  2) Run again; the scanner will find new opportunities. PnL is session-only.
"""

import asyncio
import aiohttp
import ssl
import json
import time
import os
import sys
import argparse
import logging
from datetime import datetime, timezone

from market_maker import (
    MarketMaker, MMState, QuotePair, LatencyTracker,
    MIN_SPREAD_TO_QUOTE, SHARES_PER_QUOTE, MAX_MARKETS,
    MAX_INVENTORY_AGE_S, REQUOTE_INTERVAL_S, MAX_QUOTE_AGE_S, MID_MOVE_CANCEL,
    MTM_LOSS_CUTOFF_PCT, resolution_multiplier, FILL_ANALYTICS_LOG,
    MAX_NOTIONAL_PER_MARKET,
    alert_event,
)
from scanner import (
    scan_all, fetch_all_markets, fetch_book, parse_tokens, parse_prices,
    SSL_CTX, GAMMA_API, CLOB_API,
    QuoteHistoryStore, MAX_CLUSTER_EXPOSURE, MarketRegimeWatcher,
)
from fundamental_sources import refresh_news_cache_async

log = logging.getLogger("run_mm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── config ────────────────────────────────────────────────────────────────────
SCAN_INTERVAL_S       = 45     # scan for new opportunities every 45s
QUOTE_MANAGE_INTERVAL = 2      # check every 2s — reprice constantly, capture spread
FILL_CHECK_S          = 10     # check order fills every 10s
BOOK_DELAY       = 0.25   # rate limit between CLOB API calls
MAX_BOOK_FETCHES = 25     # max books per scan cycle
LOG_DIR          = "logs"

# Hard daily kill switch — stop ALL quoting if down > Y% in a day. No exceptions.
DAILY_LOSS_LIMIT_PCT = 5.0   # e.g. 5% = stop if session PnL < −5% of start balance

# Kill switch file/env — stop quoting without restart. Touch .stop_quoting or set STOP_QUOTING=1.
KILL_SWITCH_FILE = ".stop_quoting"


class DailyKillSwitch:
    """
    Stop all quoting if session PnL drops below −Y% of start balance.
    No exceptions. No "just one more spread."
    """
    def __init__(self, start_balance: float, limit_pct: float = DAILY_LOSS_LIMIT_PCT):
        self.start_balance = start_balance
        self.limit_pct     = limit_pct
        self.tripped       = False
        self.tripped_at    = 0.0

    def check(self, total_pnl: float) -> bool:
        """Returns True if kill switch is tripped (must stop)."""
        if self.tripped:
            return True
        loss_pct = -total_pnl / self.start_balance * 100 if self.start_balance > 0 else 0
        if total_pnl < 0 and loss_pct >= self.limit_pct:
            self.tripped    = True
            self.tripped_at = time.time()
            log.warning("DAILY KILL SWITCH TRIPPED: PnL=$%.2f (%.1f%% loss) ≥ %.1f%% limit. "
                        "All quoting stopped. No exceptions.",
                        total_pnl, loss_pct, self.limit_pct)
            return True
        return False

    def status(self) -> str:
        if self.tripped:
            return f"TRIPPED (PnL limit {self.limit_pct}%)"
        return "armed"


def kill_switch_active() -> bool:
    """True if manual kill switch is active (file or env). Stops quoting without restart."""
    if os.environ.get("STOP_QUOTING", "").strip() in ("1", "true", "yes"):
        return True
    return os.path.exists(KILL_SWITCH_FILE)


# Paper fill simulation: randomly fill orders after some delay
PAPER_FILL_DELAY_S = (30, 180)   # uniform random fill time 30-180s


# ── correlation cluster exposure controller ───────────────────────────────────

class ClusterExposureController:
    """
    Caps the number of simultaneously active quotes in any one correlation
    cluster (e.g. NFL, us_politics, crypto).

    When a news shock hits a cluster — final whistle, election call,
    flash crash — every market in that cluster reprices at once.
    Capping exposure per cluster limits the simultaneous adverse
    selection hit to at most MAX_CLUSTER_EXPOSURE positions.

    Usage:
        ctrl = ClusterExposureController()
        if ctrl.can_add(opp.cluster_id, mm.state.active_quotes):
            mm.create_quote(...)
    """

    def __init__(self, max_per_cluster: int = MAX_CLUSTER_EXPOSURE):
        self.max_per_cluster = max_per_cluster

    def counts(self, active_quotes: dict) -> dict:
        c: dict[str, int] = {}
        for qp in active_quotes.values():
            cid = getattr(qp, "cluster_id", "uncategorized")
            c[cid] = c.get(cid, 0) + 1
        return c

    def can_add(self, cluster_id: str, active_quotes: dict) -> bool:
        return self.counts(active_quotes).get(cluster_id, 0) < self.max_per_cluster

    def exposure_line(self, active_quotes: dict) -> str:
        c = self.counts(active_quotes)
        if not c:
            return "(none)"
        parts = []
        for cid, n in sorted(c.items()):
            flag = " !" if n >= self.max_per_cluster else ""
            parts.append(f"{cid}:{n}{flag}")
        return "  ".join(parts)


# ── paper executor (simulates LiveExecutor) ───────────────────────────────────

class PaperExecutor:
    """Drop-in replacement for LiveExecutor that logs to console."""

    def __init__(self):
        self.connected = True
        self.total_orders = 0
        self.successful_orders = 0
        self.open_orders = []

    def connect(self) -> bool:
        log.info("[paper] Executor connected (simulation mode)")
        return True

    class _FakeResult:
        def __init__(self, success, order_id, **kw):
            self.success = success
            self.order_id = order_id
            self.error = kw.get("error", "")

    def place_limit_order(self, token_id, side, price, size):
        oid = f"paper-{side.lower()}-{int(time.time()*1000)%100000}"
        self.total_orders += 1
        self.successful_orders += 1
        self.open_orders.append({
            "id": oid, "token_id": token_id, "side": side,
            "price": price, "size": size, "ts": time.time(),
        })
        log.info("[paper] ORDER: %s %d @ %.2f | token=%s... | id=%s",
                 side, size, price, token_id[:16], oid)
        return self._FakeResult(success=True, order_id=oid)

    def cancel_all(self, market="", asset_id=""):
        before = len(self.open_orders)
        if asset_id:
            self.open_orders = [o for o in self.open_orders
                                if o["token_id"] != asset_id]
        else:
            self.open_orders = []
        log.info("[paper] Cancelled %d orders", before - len(self.open_orders))

    def get_open_orders(self):
        return self.open_orders

    def metrics(self):
        return f"[paper] orders={self.total_orders} open={len(self.open_orders)}"


# ── paper fill simulator ──────────────────────────────────────────────────────

async def paper_fill_simulator(mm: MarketMaker, session: aiohttp.ClientSession):
    """
    In paper mode, simulate fills by checking the live book periodically.
    If the market bid/ask has crossed our quote price, mark the order as filled.
    """
    while True:
        await asyncio.sleep(FILL_CHECK_S)

        for slug, qp in list(mm.state.active_quotes.items()):
            if qp.status not in ("QUOTING", "ONE_LEG"):
                continue

            try:
                book = await fetch_book(session, qp.token_yes)
                if not book:
                    continue

                bids = sorted(book.get("bids", []),
                              key=lambda x: -float(x["price"]))
                asks = sorted(book.get("asks", []),
                              key=lambda x: float(x["price"]))
                if not bids or not asks:
                    continue

                mkt_bid = float(bids[0]["price"])
                mkt_ask = float(asks[0]["price"])
                b0, a0  = bids[0], asks[0]
                bid_sz  = float(b0["size"] if isinstance(b0, dict) else b0[1])
                ask_sz  = float(a0["size"] if isinstance(a0, dict) else a0[1])
                mm.update_queue_position(qp, bid_sz, ask_sz)

                # Bid fills: check each level or single bid
                if qp.bid_levels:
                    for i, (bp, sz) in enumerate(qp.bid_levels):
                        if i < len(qp.bid_level_filled) and not qp.bid_level_filled[i] and mkt_ask <= bp:
                            qp.bid_level_filled[i] = True
                            qp.bid_filled_size += sz
                            if not qp.bid_filled:
                                qp.bid_filled = True
                                qp.bid_fill_price = bp
                                mm._record_leg_fill(qp, "BID")
                            log.info("[paper] BID L%d FILLED: %s @ %.2f (mkt_ask=%.2f)", i+1, slug, bp, mkt_ask)
                elif not qp.bid_filled and mkt_ask <= qp.bid_price:
                    qp.bid_filled = True
                    qp.bid_fill_price = qp.bid_price
                    qp.bid_filled_size = qp.bid_size
                    mm._record_leg_fill(qp, "BID")
                    log.info("[paper] BID FILLED: %s @ %.2f (mkt_ask=%.2f)", slug, qp.bid_price, mkt_ask)

                # Ask fills: check each level or single ask
                if qp.ask_levels:
                    for i, (ap, sz) in enumerate(qp.ask_levels):
                        if i < len(qp.ask_level_filled) and not qp.ask_level_filled[i] and mkt_bid >= ap:
                            qp.ask_level_filled[i] = True
                            qp.ask_filled_size += sz
                            if not qp.ask_filled:
                                qp.ask_filled = True
                                qp.ask_fill_price = ap
                                mm._record_leg_fill(qp, "ASK")
                            log.info("[paper] ASK L%d FILLED: %s @ %.2f (mkt_bid=%.2f)", i+1, slug, ap, mkt_bid)
                elif not qp.ask_filled and mkt_bid >= qp.ask_price:
                    qp.ask_filled = True
                    qp.ask_fill_price = qp.ask_price
                    qp.ask_filled_size = qp.ask_size
                    mm._record_leg_fill(qp, "ASK")
                    log.info("[paper] ASK FILLED: %s @ %.2f (mkt_bid=%.2f)", slug, qp.ask_price, mkt_bid)

                # Update status
                if qp.bid_filled and qp.ask_filled:
                    matched = min(qp.bid_filled_size, qp.ask_filled_size) if (qp.bid_filled_size or qp.ask_filled_size) else min(qp.bid_size, qp.ask_size)
                    qp.pnl = (qp.ask_fill_price - qp.bid_fill_price) * matched
                    qp.status = "DONE"
                    log.info("[paper] BOTH FILLED: %s pnl=$%.4f", slug, qp.pnl)
                    mm.complete_quote(qp)
                elif qp.bid_filled or qp.ask_filled:
                    qp.status = "ONE_LEG"
                    status = mm.handle_one_leg(qp, None, mkt_bid, mkt_ask)
                    if status == "DONE":
                        mm.complete_quote(qp)

                await asyncio.sleep(BOOK_DELAY)

            except Exception as e:
                log.debug("Paper fill check error on %s: %s", slug, e)


# ── market scanner + quote manager ────────────────────────────────────────────

async def scan_and_quote(mm: MarketMaker, executor,
                         session: aiohttp.ClientSession,
                         history: QuoteHistoryStore,
                         print_ema_store: dict,
                         cluster_ctrl: ClusterExposureController,
                         regime_watcher: MarketRegimeWatcher):
    """
    One scan cycle using the unified scanner (scanner.scan_all).
    All 6 signals — spread, activity, depth, regime, fair-value — are
    already computed.  Passes adverse_sel_risk, trade_imbalance,
    staleness_risk_pct, spread_stability, bid/ask L1 sizes, fair_value,
    mid_stale, stale_direction, cluster_id, cluster_type, and detect_ts
    directly to MarketMaker.create_quote() so every defensive layer runs
    before any order is placed.

    Cluster exposure: skips markets whose cluster is already at capacity.
    Latency telemetry: on_detect() called when opportunity identified.
    """
    slots_available = MAX_MARKETS - len(mm.state.active_quotes)
    if slots_available <= 0:
        log.info("All %d market slots full — skipping scan", MAX_MARKETS)
        return 0, []

    top = await scan_all(session, top_n=max(10, slots_available + 3),
                         history=history, print_ema_store=print_ema_store,
                         regime_watcher=regime_watcher)
    log.info("Scanner returned %d ranked markets", len(top))

    new_quotes = 0
    for opp in top:
        if len(mm.state.active_quotes) >= MAX_MARKETS:
            break
        if opp.slug in mm.state.active_quotes:
            continue

        if opp.toxic_flow_detected:
            log.info("TOXIC FLOW — skip: %-45s imb=%.2f accel=%.1f Δspread=%.1fc",
                     opp.question[:45], opp.trade_imbalance,
                     opp.velocity_acceleration, opp.spread_change_c)
            continue

        if opp.market_dead:
            log.info("MARKET DEAD — skip: %-45s n5m=%d spread_expl=%s",
                     opp.question[:45], opp.n_snaps_short, opp.spread_explosion)
            continue

        # Regime degradation: 24h baseline vs rolling activity
        if opp.regime_change:
            log.warning("REGIME DEGRADED: %-45s  size×%.1f  (activity<<24h baseline)",
                        opp.question[:45], opp.size_multiplier)

        # Correlation clustering: one shock moves all markets in same cluster
        if not cluster_ctrl.can_add(opp.cluster_id, mm.state.active_quotes):
            log.info("CLUSTER FULL — skip: %-45s cluster=%s (at %d/%d)",
                     opp.question[:45], opp.cluster_id,
                     cluster_ctrl.counts(mm.state.active_quotes).get(opp.cluster_id, 0),
                     MAX_CLUSTER_EXPOSURE)
            continue

        # Latency telemetry: anchor detection time (detect → send → ack → book)
        detect_ts = time.time()
        mm.latency.on_detect(opp.slug)

        fv_note   = (f"  fv={opp.fair_value:.3f}★{opp.stale_direction}"
                     if opp.mid_stale else "")
        edge_note  = (f"  edge={opp.edge_cents:+.1f}c p={opp.p_est:.3f}[{opp.alpha_mode}]"
                      if opp.alpha_mode != "PASSIVE_MM" else "")
        cross_note = (f"  Xedge={opp.cross_edge_cents:+.1f}c [{opp.cross_group_id[:20]}]"
                      if opp.cross_edge_cents != 0 else "")
        fund_note  = (f"  fund={opp.fund_direction} {opp.fund_p_adjust_cents:+.1f}c"
                      if getattr(opp, "fund_n_headlines", 0) > 0 else "")
        log.info("OPPORTUNITY: %-45s score=%.1f  spread=%.1fc  cluster=%s  "
                 "as=%.2f  stale=%.0f%%  imb=%+.2f  bid_sz=%.0f ask_sz=%.0f%s%s%s%s",
                 opp.question[:45], opp.score, opp.spread_cents, opp.cluster_id,
                 opp.adverse_sel_risk, opp.staleness_risk_pct * 100,
                 opp.trade_imbalance, opp.bid_l1_size, opp.ask_l1_size,
                 fv_note, edge_note, cross_note, fund_note)

        qp = mm.create_quote(
            slug                = opp.slug,
            question            = opp.question,
            token_yes           = opp.token_yes,
            token_no            = opp.token_no,
            best_bid            = opp.best_bid,
            best_ask            = opp.best_ask,
            adverse_sel_risk    = opp.adverse_sel_risk,
            trade_imbalance     = opp.trade_imbalance,
            staleness_risk_pct  = opp.staleness_risk_pct,
            spread_stability    = opp.spread_stability,
            bid_l1_size         = opp.bid_l1_size,
            ask_l1_size         = opp.ask_l1_size,
            bid_l1_depth        = getattr(opp, "bid_l1_depth", 0) or (opp.best_bid * opp.bid_l1_size if opp.bid_l1_size else 0),
            ask_l1_depth        = getattr(opp, "ask_l1_depth", 0) or (opp.best_ask * opp.ask_l1_size if opp.ask_l1_size else 0),
            fair_value          = opp.fair_value,
            mid_stale           = opp.mid_stale,
            stale_direction     = opp.stale_direction,
            cluster_id          = opp.cluster_id,
            cluster_type        = opp.cluster_type,
            detect_ts           = detect_ts,
            size_multiplier     = opp.size_multiplier,
            days_to_end         = opp.days_to_end,
            quote_velocity_c_hr = opp.quote_velocity_c_hr,
            p_est               = opp.p_est,
            p_confidence        = opp.p_confidence,
            edge_cents          = opp.edge_cents,
            alpha_mode          = opp.alpha_mode,
        )
        if qp:
            # Only skip when deployed >= bankroll (no % cap — prediction markets are slow)
            deployed = sum(
                (x.bid_price + 1.0 - x.ask_price) * x.bid_size
                for x in mm.state.active_quotes.values()
                if x.status in ("QUOTING", "ONE_LEG")
            )
            new_cap = (qp.bid_price + 1.0 - qp.ask_price) * qp.bid_size
            bankroll = mm.state.balance
            if bankroll > 0 and deployed >= bankroll:
                log.info("BANKROLL FULL — skip: %-45s deployed=$%.2f >= bankroll=$%.2f",
                         opp.question[:45], deployed, bankroll)
            elif bankroll > 0 and deployed + new_cap > bankroll:
                log.info("BANKROLL FULL — skip: %-45s deployed=$%.2f + new=$%.2f > bankroll=$%.2f",
                         opp.question[:45], deployed, new_cap, bankroll)
            else:
                cap_pct = (deployed + new_cap) / bankroll * 100 if bankroll > 0 else 0
                if not mm.paper:
                    log.info("PLACING LIVE: %-45s %.0f sh  cap=$%.2f (%.0f%%)",
                             opp.question[:45], qp.bid_size, new_cap, cap_pct)
                if mm.place_quote(qp, executor):
                    new_quotes += 1

    return new_quotes, top


# ── stale quote management ────────────────────────────────────────────────────

async def manage_active_quotes(mm: MarketMaker, executor,
                               session: aiohttp.ClientSession):
    """
    For each active quote, check if:
      - It's been filled
      - The spread has narrowed (cancel)
      - It's been quoting too long without fills (cancel)
    """
    for slug in list(mm.state.active_quotes.keys()):
        qp = mm.state.active_quotes.get(slug)
        if not qp:
            continue

        now = time.time()

        # Only re-check at intervals
        if now - qp.last_check_ts < REQUOTE_INTERVAL_S:
            continue
        qp.last_check_ts = now

        if not mm.paper:
            status = mm.check_fills(qp, executor)
            if status == "DONE":
                mm.complete_quote(qp)
                continue

        # Fetch current book to check if our quote is still good
        try:
            book = await fetch_book(session, qp.token_yes)
            await asyncio.sleep(BOOK_DELAY)
            if not book:
                continue

            bids = sorted(book.get("bids", []),
                          key=lambda x: -float(x["price"]))
            asks = sorted(book.get("asks", []),
                          key=lambda x: float(x["price"]))
            if not bids or not asks:
                continue

            mkt_bid = float(bids[0]["price"])
            mkt_ask = float(asks[0]["price"])
            b0, a0 = bids[0], asks[0]
            bid_sz  = float(b0["size"] if isinstance(b0, dict) else b0[1])
            ask_sz  = float(a0["size"] if isinstance(a0, dict) else a0[1])
            spread_c = (mkt_ask - mkt_bid) * 100

            # End-to-end telemetry: first book poll after QUOTING = order visible
            if qp.status == "QUOTING" and slug in mm.latency.pending_slugs():
                mm.latency.on_book_visible(slug)

            # Update queue position: pass current best prices and L1 sizes
            # so QueuePositionTracker can detect consumed shares and size resets.
            mm.update_queue_position(qp, mkt_bid, mkt_ask, bid_sz, ask_sz)

            # Cancel & requote when: mid moved > 0.5c, someone ahead, or max age 20s.
            # Goal: provide liquidity, cancel quickly, reprice constantly, capture spread.
            mkt_mid = (mkt_bid + mkt_ask) / 2
            age = now - qp.created_ts

            if qp.status == "QUOTING":
                our_bid = qp.bid_levels[0][0] if qp.bid_levels else qp.bid_price
                our_ask = qp.ask_levels[0][0] if qp.ask_levels else qp.ask_price
                our_mid = (our_bid + our_ask) / 2

                # Mid moved > 0.5c → cancel and requote
                if abs(mkt_mid - our_mid) > MID_MOVE_CANCEL:
                    if mm.refresh_quote(qp, executor, mkt_bid, mkt_ask, bid_sz, ask_sz):
                        log.info("MID MOVE: %s  |Δmid|=%.3f > %.3f → reposted",
                                 qp.market_slug, abs(mkt_mid - our_mid), MID_MOVE_CANCEL)
                    continue

                # Someone ahead by ≥1 tick → cancel and requote
                behind_bid = (mkt_bid - our_bid) >= 0.01
                behind_ask = (our_ask - mkt_ask) >= 0.01
                if behind_bid or behind_ask:
                    if mm.refresh_quote(qp, executor, mkt_bid, mkt_ask, bid_sz, ask_sz):
                        log.info("QUEUE JUMP: %s  someone ahead (bid:%s ask:%s) → reposted",
                                 qp.market_slug, behind_bid, behind_ask)
                    continue

                # Max quote age 20s → cancel and refresh
                if age > MAX_QUOTE_AGE_S:
                    if mm.refresh_quote(qp, executor, mkt_bid, mkt_ask, bid_sz, ask_sz):
                        log.info("MAX AGE: %s  %.0fs > %ds → refreshed",
                                 qp.market_slug, age, MAX_QUOTE_AGE_S)
                    continue

            # Spread narrowed below minimum → cancel
            if spread_c < MIN_SPREAD_TO_QUOTE and qp.status == "QUOTING":
                mm.cancel_quote(qp, executor, f"spread_narrowed:{spread_c:.0f}c")
                continue

            # Quote has been open for very long without any fill → cancel
            if qp.status == "QUOTING" and age > MAX_INVENTORY_AGE_S * 2:
                mm.cancel_quote(qp, executor, f"stale:{age:.0f}s")
                continue

            # Handle one-sided fills (includes drift + skew logic)
            if qp.status == "ONE_LEG":
                # Initialise drift monitor on first transition to ONE_LEG
                if qp.market_slug not in mm._drift:
                    mm.on_first_fill(qp, mkt_bid, mkt_ask)
                status = mm.handle_one_leg(qp, executor, mkt_bid, mkt_ask)
                if status == "DONE":
                    mm.complete_quote(qp)

        except Exception as e:
            log.debug("Quote management error on %s: %s", slug, e)


# ── dashboard ─────────────────────────────────────────────────────────────────

def print_dashboard(mm: MarketMaker, executor, cycle: int, elapsed: float,
                    regime_watcher: "MarketRegimeWatcher" = None,
                    kill_switch: "DailyKillSwitch" = None):
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    mode = "PAPER" if mm.paper else "LIVE"

    print(f"\033[2J\033[H", end="")
    print("=" * 90)
    print(f"  POLYMARKET MARKET MAKER [{mode}]  |  cycle {cycle}  |  "
          f"{now}  |  {elapsed:.1f}s")
    print("=" * 90)

    # Active quotes
    active = mm.state.active_quotes
    print(f"\n  ACTIVE QUOTES ({len(active)}/{MAX_MARKETS}):")
    if active:
        print(f"  {'STATUS':<12} {'BID':>5} {'ASK':>5} {'SPR':>4} "
              f"{'CLUSTER':>12} {'FV':>6} {'FV?':>4} "
              f"{'AGE':>5} {'AS':>4} {'STALE%':>6} {'DRIFT':>6} {'SKEW':>5}  "
              f"{'B_Q':>5} {'B_FP':>5} {'B_ΔS':>5}  "
              f"{'A_Q':>5} {'A_FP':>5} {'A_ΔS':>5}  MARKET")
        print(f"  {'-'*135}")
        for slug, qp in active.items():
            age    = int(time.time() - qp.created_ts)
            spread = (qp.ask_price - qp.bid_price) * 100
            status = qp.status
            if qp.bid_filled: status += "(B)"
            if qp.ask_filled: status += "(A)"
            drift_str = (f"{qp.post_fill_drift_c:+.1f}c"
                         if qp.status == "ONE_LEG" else "   -- ")
            skew_str  = (f"{qp.inventory_skew_c:.1f}c"
                         if qp.status == "ONE_LEG" else "  -- ")
            # Cluster (correlation risk — one shock moves all in cluster)
            cluster_str = f"{qp.cluster_id}" if getattr(qp, "cluster_id", "") else "  -- "
            cluster_str = cluster_str[:12].ljust(12)
            # Fair-value indicator
            fv_str  = f"{qp.fair_value:.3f}" if qp.fair_value > 0 else "  -- "
            fv_flag = f"★{qp.stale_direction}" if qp.mid_stale else "  - "
            # Queue fields: remaining depth, fill probability, consumed this cycle
            bq  = f"{qp.bid_queue_remaining:5.0f}"
            bfp = f"{qp.bid_fill_prob*100:4.0f}%"
            bds = f"-{qp.bid_consumed_cycle:.0f}" if qp.bid_consumed_cycle > 0 else " -- "
            aq  = f"{qp.ask_queue_remaining:5.0f}"
            afp = f"{qp.ask_fill_prob*100:4.0f}%"
            ads = f"-{qp.ask_consumed_cycle:.0f}" if qp.ask_consumed_cycle > 0 else " -- "
            # Flag queue resets (someone added large order ahead of us)
            bq_flag = "!" if qp.bid_size_increased else " "
            aq_flag = "!" if qp.ask_size_increased else " "
            print(f"  {status:<12} {qp.bid_price:5.3f} {qp.ask_price:5.3f} "
                  f"{spread:3.0f}c "
                  f"{cluster_str:>12} "
                  f"{fv_str:>6} {fv_flag:>4} "
                  f"{age:5d}s "
                  f"{qp.adverse_sel_risk:4.2f} "
                  f"{qp.staleness_risk_pct*100:5.0f}% "
                  f"{drift_str:>6} {skew_str:>5}  "
                  f"{bq}{bq_flag} {bfp} {bds:>5}  "
                  f"{aq}{aq_flag} {afp} {ads:>5}  "
                  f"{qp.question[:30]}")
    else:
        print(f"  (none — scanning for opportunities...)")

    # Completed trades — show attribution + spread capture alongside PnL
    completed = mm.state.completed[-10:]
    if completed:
        print(f"\n  RECENT COMPLETED ({len(mm.state.completed)} total):")
        print(f"  {'PNL':>8} {'CAPTURE':>8} {'ATTR':<12} {'AGE':>6} "
              f"{'BID_F':>5} {'ASK_F':>5} {'CLUSTER':>12}  MARKET")
        print(f"  {'-'*88}")
        for qp in reversed(completed):
            pnl_str  = f"${qp.pnl:+.4f}"
            # Spread capture % (actual / expected) — shows whether we earned what we expected
            if qp.expected_spread_c > 0:
                cap_pct = (qp.ask_fill_price - qp.bid_fill_price) * 100 / qp.expected_spread_c
                cap_str = f"{cap_pct*100:+.0f}%"
            else:
                cap_str = "   -- "
            # Attribution from notes (best-effort)
            notes = (qp.notes or "").lower()
            if "adverse_exit" in notes:
                attr = "DRIFT_EXIT"
            elif "force_exit" in notes or "stale:" in notes:
                attr = "TIMEOUT"
            elif qp.worst_drift_c < -3.0:
                attr = "ADVERSE"
            elif qp.staleness_risk_pct > 0.60:
                attr = "STALE"
            else:
                attr = "ORGANIC"
            age_s  = int(qp.done_ts - qp.created_ts) if qp.done_ts > 0 else 0
            clust  = getattr(qp, "cluster_id", "")[:12] or "--"
            print(f"  {pnl_str:>8} {cap_str:>8} {attr:<12} {age_s:5d}s "
                  f"{qp.bid_fill_price:5.3f} {qp.ask_fill_price:5.3f} "
                  f"{clust:>12}  {qp.question[:36]}")

    # Summary line
    print(f"\n  {mm.summary()}")
    if hasattr(executor, "metrics"):
        print(f"  Executor: {executor.metrics()}")

    # Risk limits (per-market MTM cutoff + daily kill switch + resolution scaling)
    print(f"  Risk limits:  MTM cutoff={MTM_LOSS_CUTOFF_PCT}%  |  "
          f"Daily kill={DAILY_LOSS_LIMIT_PCT}%  |  "
          f"Resolution scale (days→size)  |  Kill: {kill_switch.status() if kill_switch else 'N/A'}")

    # Inline spread-capture snapshot (fast — no full table needed every cycle)
    sc = mm._spread_capture.stats()
    if sc:
        surv_n  = len(mm._survival._records)
        med_s   = (sorted(r.time_to_done_s for r in mm._survival._records)
                   [surv_n // 2]) if surv_n >= 3 else 0
        by_attr = sc.get("by_attribution", {})
        attr_summary = "  ".join(
            f"{k[:3]}:{d['n']}({d['mean_capture']*100:+.0f}%)"
            for k, d in sorted(by_attr.items())
        )
        print(f"  Capture: mean={sc['mean_capture']*100:+.1f}%  "
              f"med={sc['median_capture']*100:+.1f}%  "
              f"pos={sc['pct_positive']:.0f}%  "
              f"n={sc['n']}  [{attr_summary}]")
        if surv_n >= 3:
            print(f"  Survival: n={surv_n}  median_fill={med_s:.0f}s")

    # Cluster exposure (correlation risk — one shock moves all in cluster)
    cluster_ctrl_tmp = ClusterExposureController()
    exp_line = cluster_ctrl_tmp.exposure_line(active)
    print(f"  Cluster exposure: {exp_line}  (max {MAX_CLUSTER_EXPOSURE}/cluster)")

    # Market regime watcher (24h baseline vs rolling activity)
    if regime_watcher is not None:
        print(f"  Regime watcher:   {regime_watcher.summary_line()}")

    # Capital efficiency (inline snapshot — one line)
    cap_s = mm._cap_eff.stats(mm.state.total_pnl, active)
    if cap_s:
        print(f"  Capital:  deployed=${cap_s['deployed_now']:.2f}  "
              f"idle=${cap_s['idle']:.2f}  "
              f"deploy={cap_s['deploy_pct']:.0f}%  "
              f"ROC={cap_s['roc_session_pct']:+.3f}%  "
              f"turns={cap_s['capital_turns']:.3f}  "
              f"notional=${cap_s['notional_traded']:,.0f}")

    # Full analytics reports (spread capture, survival curve, latency, toxics)
    mm.print_analytics_report()

    print()


# ── main loop ─────────────────────────────────────────────────────────────────

async def run(paper: bool, bankroll: float):
    os.makedirs(LOG_DIR, exist_ok=True)

    state = MMState(balance=bankroll)
    mm    = MarketMaker(state, paper=paper)

    if paper:
        executor = PaperExecutor()
    else:
        from live_executor import LiveExecutor
        executor = LiveExecutor(tick_size="0.01")
        if not executor.connect():
            log.error("Failed to connect to Polymarket. Check .env credentials.")
            return

    mode = "PAPER" if paper else "LIVE"
    log.info("Starting Market Maker [%s] | bankroll=$%.2f", mode, bankroll)
    if not paper:
        log.info(">>> LIVE MODE: Real orders will be sent to Polymarket <<<")

    # Hard daily kill switch — stop ALL quoting if down > Y% in a day. No exceptions.
    kill_switch = DailyKillSwitch(state.balance, limit_pct=DAILY_LOSS_LIMIT_PCT)
    log.info("Daily kill switch: armed (limit=%.1f%% loss)", DAILY_LOSS_LIMIT_PCT)
    log.info("Fill analytics: %s  |  Resolution scale: 14d=100%% 7d=80%% 3d=60%% 1d=40%% <1d=25%%",
             FILL_ANALYTICS_LOG)

    # Shared stores — persist across scan cycles so rolling activity,
    # regime signals, and print EMA all improve over time.
    history:         QuoteHistoryStore = {}
    print_ema_store: dict              = {}
    cluster_ctrl     = ClusterExposureController()
    regime_watcher   = MarketRegimeWatcher()

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=SSL_CTX, limit=5),
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:

        tasks = []

        async def news_refresh_loop():
            """Background: refresh news cache every 5 min. Never blocks scan."""
            while True:
                try:
                    await refresh_news_cache_async(session)
                except Exception as e:
                    log.debug("News refresh: %s", e)
                await asyncio.sleep(300)  # 5 min

        tasks.append(asyncio.create_task(news_refresh_loop()))

        async def quote_manage_loop():
            """Background: manage quotes every 5s — check fills, queue jump when someone ahead."""
            while True:
                try:
                    await manage_active_quotes(mm, executor, session)
                except Exception as e:
                    log.debug("Quote manage: %s", e)
                await asyncio.sleep(QUOTE_MANAGE_INTERVAL)

        tasks.append(asyncio.create_task(quote_manage_loop()))

        if paper:
            tasks.append(asyncio.create_task(
                paper_fill_simulator(mm, session)))

        cycle = 0
        exited_kill_switch = False
        exited_manual_stop = False
        try:
            while True:
                cycle += 1
                start = time.time()

                # 0. Hard daily kill switch — stop ALL quoting if down > Y%. No exceptions.
                if kill_switch.check(mm.state.total_pnl):
                    log.warning("DAILY KILL SWITCH TRIPPED — cancelling all quotes and stopping.")
                    alert_event("daily_loss", mm.state.total_pnl)
                    mm.cancel_all(executor)
                    exited_kill_switch = True
                    break

                # 0b. Manual kill switch — file or env. Stop quoting without restart.
                if kill_switch_active():
                    log.warning("KILL SWITCH ACTIVE (%s or STOP_QUOTING=1) — cancelling all quotes.",
                                KILL_SWITCH_FILE)
                    alert_event("kill_switch", mm.state.total_pnl)
                    mm.cancel_all(executor)
                    exited_manual_stop = True
                    break

                # 0c. Live: sync balance from Polymarket (avoids INSUFFICIENT_FUNDS)
                if not paper and hasattr(executor, "get_balance_allowance"):
                    bal, allow = executor.get_balance_allowance()
                    if bal > 0:
                        prev = mm.state.balance
                        mm.state.balance = bal
                        if abs(bal - prev) > 0.5:
                            log.info("Balance sync: $%.2f (allowance $%.2f)", bal, allow)

                # 1. Scan for new opportunities and place quotes
                try:
                    new, _ = await scan_and_quote(mm, executor, session,
                                                   history, print_ema_store,
                                                   cluster_ctrl, regime_watcher)
                    if new:
                        log.info("Placed %d new quotes", new)
                except Exception as e:
                    log.error("Scan error: %s", e)

                # 2. Quote management runs in background every 5s (fills, queue jump)

                elapsed = time.time() - start

                # 3. Capital efficiency snapshot (call every cycle before dashboard)
                mm._cap_eff.snapshot(mm.state.active_quotes)

                # 4. Print dashboard
                print_dashboard(mm, executor, cycle, elapsed,
                                regime_watcher=regime_watcher,
                                kill_switch=kill_switch)

                # 4. Save state
                save_state(mm, cycle)

                # 5. Sleep until next cycle
                sleep_time = max(5, SCAN_INTERVAL_S - elapsed)
                await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            for t in tasks:
                t.cancel()
            if not paper and not exited_kill_switch:
                mm.cancel_all(executor)
            if exited_kill_switch:
                print("\n  ⛔ DAILY KILL SWITCH TRIPPED — session stopped (loss limit exceeded)\n")
            elif exited_manual_stop:
                print("\n  ⛔ KILL SWITCH — session stopped (remove .stop_quoting or unset STOP_QUOTING to resume)\n")
            print_final_summary(mm)


def save_state(mm: MarketMaker, cycle: int):
    """
    Persist state to disk for crash recovery.
    Writes to logs/mm_state.json: cycle, total_pnl, active count, last 50 completed.
    Restart: run again with same --bankroll. Open orders live on Polymarket; cancel
    stale ones via UI if the process died mid-quote.
    """
    data = {
        "cycle": cycle,
        "ts": time.time(),
        "total_pnl": mm.state.total_pnl,
        "total_trades": mm.state.total_trades,
        "active": len(mm.state.active_quotes),
        "completed": [
            {"slug": q.market_slug, "pnl": q.pnl, "notes": q.notes}
            for q in mm.state.completed[-50:]
        ],
    }
    with open(os.path.join(LOG_DIR, "mm_state.json"), "w") as f:
        json.dump(data, f, indent=2)


def print_final_summary(mm: MarketMaker):
    print("\n" + "=" * 70)
    print("  MARKET MAKER SESSION SUMMARY")
    print("=" * 70)
    print(f"  {mm.summary()}")

    if mm.state.completed:
        pnls = [q.pnl for q in mm.state.completed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        print(f"\n  Total trades:    {len(pnls)}")
        print(f"  Wins:            {len(wins)}")
        print(f"  Losses:          {len(losses)}")
        print(f"  Win rate:        {len(wins)/len(pnls)*100:.1f}%")
        print(f"  Total PnL:       ${sum(pnls):+.4f}")
        if wins:
            print(f"  Avg win:         ${sum(wins)/len(wins):+.4f}")
        if losses:
            print(f"  Avg loss:        ${sum(losses)/len(losses):+.4f}")

        force_exits = [q for q in mm.state.completed if "force_exit" in q.notes]
        if force_exits:
            fe_pnl = sum(q.pnl for q in force_exits)
            print(f"  Force exits:     {len(force_exits)} (pnl=${fe_pnl:+.4f})")

    print("=" * 70)

    # Full analytics — spread capture + attribution + survival curve + latency
    mm.print_analytics_report()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-market market maker for Polymarket")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true",
                        help="Live trading with real money")
    parser.add_argument("--bankroll", type=float, default=100.0,
                        help="Starting bankroll in USD (default 100)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip live-mode confirmation prompt")
    args = parser.parse_args()

    paper = not args.live

    if not paper:
        print("=" * 70)
        print("  WARNING: LIVE MODE — REAL MONEY WILL BE USED")
        print("=" * 70)
        if not args.yes:
            confirm = input("  Type 'yes' to confirm: ").strip().lower()
            if confirm != "yes":
                print("  Aborted.")
                return

    asyncio.run(run(paper=paper, bankroll=args.bankroll))


if __name__ == "__main__":
    main()
