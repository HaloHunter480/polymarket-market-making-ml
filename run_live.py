"""
run_live.py — Polymarket BTC Binary Scalper  (AWS Ireland entry point)
=======================================================================

STRATEGY: LATENCY ARBITRAGE on stale Polymarket quotes
  AWS Ireland → Polymarket:  < 150 ms  (measured)
  Polymarket reprices after BTC moves:  2-5 seconds
  Window to capture the stale quote:    1.85-4.85 seconds

Signal stack (decision_stack.py):
  BTC move (10s)    → BTC moved ≥ 0.04% → fair value shifted, quote may be stale
  GARCH sigma       → instantaneous vol → scale the Gaussian fair value
  Fair value gap    → Φ(pct_diff / σ_t) - poly_mid > 7.5% (fee + min edge)
  Staleness check   → poly_mid hasn't moved in 3s → still stale
  VPIN gate         → no toxic / informed flow
  Hawkes gate       → market must be ACTIVE or EXCITED
  Vol gate          → GARCH not EXTREME_VOL
  OBI gate          → Polymarket book must lean in our direction

Run:  python3 run_live.py [--paper]
"""

import asyncio
import aiohttp
import websockets
import json
import ssl
import time
import os
import logging
import numpy as np
from collections import deque
from datetime import datetime, timezone
from dotenv import load_dotenv
import pathlib

load_dotenv(pathlib.Path(__file__).parent / ".env", override=True)

# ── local modules ──────────────────────────────────────────────────────────────
from decision_stack  import DecisionStack, MarketState, SignalResult, TOTAL_COST
from live_executor   import LiveExecutor, RiskManager as ExecRiskManager
from orderbook       import OrderBook
from risk_man        import RiskManager
from warmup          import load_warmup

# ── logging ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/run_live_{_ts}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("run_live")

# ── network constants ──────────────────────────────────────────────────────────
COINBASE_WS   = "wss://advanced-trade-api.coinbase.com/ws/public"
BINANCE_TRADE_WS  = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_DEPTH_WS  = "wss://stream.binance.com:9443/ws/btcusdt@depth"
BINANCE_DEPTH_REST = "https://api.binance.com/api/v3/depth"
DERIBIT_WS    = "wss://www.deribit.com/ws/api/v2"
CLOB_WS       = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_API     = "https://gamma-api.polymarket.com"
CLOB_API      = "https://clob.polymarket.com"
SSL_CTX       = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ── strategy constants ─────────────────────────────────────────────────────────
EVAL_INTERVAL_S   = 0.10     # evaluation frequency (100 ms)
WINDOW_SECONDS    = 300      # Polymarket 5-min windows
ENTRY_CUTOFF_S    = 180      # only enter in first 180 s of window
STRIKE_LOCK_S     = 150      # must lock strike within first 150 s of window
STRIKE_MID_RANGE  = (0.30, 0.70)  # lock when mid is reasonably near 50/50
REFIT_INTERVAL_S  = 3600     # refit GARCH + Hawkes every hour
EMIT_EVERY_N      = 10       # print status every N eval cycles

# emergency-exit thresholds (protect open position)
EMERGENCY_VPIN    = 0.72     # VPIN above this → consider exit
EMERGENCY_PMODEL  = 0.30     # our side P_model drops below this → exit
EMERGENCY_TIME_S  = 60       # only exit if >60 s remaining

# position sizing
INITIAL_BANKROLL  = 100.0    # USD
MAX_POSITION_PCT  = 0.10     # max position per market = 10% capital ($2 on $20 bankroll)
MAX_EXPOSURE_PCT  = 0.20     # max total exposure = 20% capital
MAX_TRADE_USD     = 20.0     # fallback cap
MIN_TRADE_USD     = 1.00   # Polymarket minimum market order size is $1
KELLY_FRACTION    = 0.25


# ── helpers ────────────────────────────────────────────────────────────────────

def current_window_id() -> int:
    return int(time.time() // WINDOW_SECONDS)

def seconds_in_window() -> float:
    return time.time() % WINDOW_SECONDS

def seconds_remaining() -> float:
    return WINDOW_SECONDS - seconds_in_window()


# ── market discovery ───────────────────────────────────────────────────────────

def _parse_clob_tokens(raw) -> list:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    if isinstance(raw, str):
        s = raw.strip().strip("[]").replace('"', '').replace("'", "")
        return [t.strip() for t in s.split(",") if t.strip()]
    return []


def _map_outcomes_to_tokens(outcomes, tokens: list) -> tuple:
    up_idx, dn_idx = 0, 1
    if isinstance(outcomes, str) and outcomes.strip():
        parts = [p.strip().strip('[]"\'') for p in outcomes.split(",")]
        for i, p in enumerate(parts):
            if p.lower() in ("up", "yes", "higher"):
                up_idx, dn_idx = i, 1 - i
                break
            if p.lower() in ("down", "no", "lower"):
                dn_idx, up_idx = i, 1 - i
                break
    elif isinstance(outcomes, list):
        for i, p in enumerate(outcomes):
            if str(p).lower() in ("up", "yes", "higher"):
                up_idx, dn_idx = i, 1 - i
                break
    return (tokens[up_idx], tokens[dn_idx]) if len(tokens) >= 2 else ("", "")


async def discover_btc_market(http: aiohttp.ClientSession) -> dict | None:
    """
    Find the active BTC 5-min UP/DOWN market on Polymarket.
    Returns {market_id, token_up, token_dn, condition_id, question} or None.

    Tries: 1) events API (btc-updown-5m slug), 2) markets API fallback.
    """
    window_start = (int(time.time()) // WINDOW_SECONDS) * WINDOW_SECONDS

    # 1. BTC 5-min events (slug: btc-updown-5m-{window})
    for offset in range(6):
        ws = window_start - offset * WINDOW_SECONDS
        slug = f"btc-updown-5m-{ws}"
        try:
            async with http.get(
                f"{GAMMA_API}/events", params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                if r.status != 200:
                    continue
                data = await r.json(content_type=None)
                events = data if isinstance(data, list) else ([data] if data else [])
                for ev in events:
                    markets = ev.get("markets", [])
                    if not markets:
                        continue
                    m = markets[0]
                    tokens = _parse_clob_tokens(m.get("clobTokenIds", ""))
                    if len(tokens) >= 2:
                        up_tok, dn_tok = _map_outcomes_to_tokens(m.get("outcomes", ""), tokens)
                        if up_tok and dn_tok:
                            title = ev.get("title", slug)
                            log.info("Market found: %s", title[:80])
                            return {
                                "market_id": m.get("id", ""),
                                "condition_id": m.get("condition_id", ""),
                                "token_up": up_tok,
                                "token_dn": dn_tok,
                                "question": title,
                            }
        except Exception as e:
            log.warning("Discovery error (slug=%s): %s", slug, e)

    # 2. Fallback: markets API
    for url in [
        f"{GAMMA_API}/markets?active=true&closed=false&limit=100",
        f"{GAMMA_API}/markets?active=true&tag=crypto&limit=50",
    ]:
        try:
            async with http.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    continue
                markets = await r.json(content_type=None)
                if isinstance(markets, dict):
                    markets = markets.get("data", [])
                for m in markets:
                    q = (m.get("question", "") or m.get("title", "")).lower()
                    if "btc" not in q and "bitcoin" not in q:
                        continue
                    if "higher" not in q and "up" not in q and "5" not in q:
                        continue
                    tokens = m.get("tokens", [])
                    parsed = _parse_clob_tokens(m.get("clobTokenIds", "")) if not tokens else []
                    tok_ids = [t.get("token_id") for t in tokens if t.get("token_id")] if tokens else parsed
                    if len(tok_ids) >= 2:
                        up_tok, dn_tok = _map_outcomes_to_tokens(m.get("outcomes", ""), tok_ids)
                        if up_tok and dn_tok:
                            log.info("Market found: %s", q[:80])
                            return {
                                "market_id": m.get("id", ""),
                                "condition_id": m.get("condition_id", ""),
                                "token_up": up_tok,
                                "token_dn": dn_tok,
                                "question": q,
                            }
        except Exception as e:
            log.warning("Market discovery error (%s): %s", url, e)

    log.error("Could not discover BTC market. Check gamma API or run find_proxy.py.")
    return None


# ── state shared across async tasks ────────────────────────────────────────────

class SharedState:
    def __init__(self):
        self.btc_price: float        = 0.0
        self.btc_price_10s: float    = 0.0   # BTC price ~10 seconds ago (timestamp-based)
        self.btc_prices_1s: deque    = deque(maxlen=3600)   # rolling 1-hr history
        self._clob_msg_count: int    = 0     # diagnostic: CLOB messages received
        self.btc_bar_open: float     = 0.0
        self.btc_bar_vol: float      = 0.0
        self.btc_bar_ts: float       = 0.0
        self.btc_trade_times: deque  = deque(maxlen=2000)   # for Hawkes

        self.book_cache: dict        = {}   # token → {"bid","ask","mid","obi","tob_bid","tob_ask"}
        self.poly_mid: float         = 0.5
        self.deribit_iv: float      = 0.0   # Layer 1: Deribit volatility index
        self.binance_book           = None  # OrderBook for Layer 2 OBI (set when use_binance)
        self.last_binance_trade: dict = None
        self.btc_price_history      = deque(maxlen=300)  # Layer 5 HMM

        # Active market tokens — updated automatically on each window rollover
        self.token_up: str           = ""
        self.token_dn: str           = ""
        self.market_question: str    = ""
        self.clob_refresh            = asyncio.Event()  # set to trigger CLOB reconnect with new tokens

        self.window_id: int          = 0
        self.strike: float           = 0.0
        self.strike_set: bool        = False

        self.open_position: dict | None = None  # {token,side,price,size,cost,ts}
        self.last_eval_n: int           = 0
        self.event_times_history: list  = []    # for Hawkes refit
        self.return_history: list        = []    # for GARCH refit
        self.last_refit_ts: float        = 0.0
        self.trades_log: list            = []    # OOS records


# ── Binance trade feed (fallback when Coinbase fails) ───────────────────────────

async def binance_feed(state: SharedState, stack: DecisionStack):
    """Stream BTC-USDT trades from Binance. Updates state.btc_price."""
    while True:
        try:
            async with websockets.connect(
                BINANCE_TRADE_WS, ssl=SSL_CTX,
                ping_interval=20, ping_timeout=10, max_size=2**20,
            ) as ws:
                log.info("Binance trade feed connected")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        price = float(msg.get("p", 0))
                        if price <= 0:
                            continue
                        now = time.time()
                        stack.on_btc_price(price, now)
                        state.btc_price = price
                        state.last_binance_trade = {"p": str(price)}
                        if not state.btc_price_history or now - state.btc_price_history[-1][0] >= 0.5:
                            state.btc_price_history.append((now, price))
                        state.btc_prices_1s.append(price)
                        # True 10-second lookback using timestamps
                        cutoff10 = now - 10.0
                        p10 = 0.0
                        for _ts, _p in reversed(list(state.btc_price_history)):
                            if _ts <= cutoff10:
                                p10 = _p
                                break
                        state.btc_price_10s = p10 if p10 > 0 else (state.btc_price_history[0][1] if state.btc_price_history else 0.0)
                        if state.return_history and len(state.btc_prices_1s) >= 2:
                            prev = state.btc_prices_1s[-2]
                            if prev > 0:
                                lr = float(np.log(price / prev))
                                state.return_history.append(lr)
                                if len(state.return_history) > 10000:
                                    state.return_history = state.return_history[-5000:]
                        stack.on_btc_trade(now)
                        state.btc_trade_times.append(now)
                        state.event_times_history.append(now)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
        except Exception as e:
            log.warning("Binance feed disconnected: %s — reconnecting in 2s", e)
            await asyncio.sleep(2)


# ── Binance depth feed (Layer 1: for Layer 2 OBI) ───────────────────────────────

async def binance_depth_feed(state: SharedState):
    """Stream Binance order book for Layer 2 OBI signal."""
    if state.binance_book is None:
        state.binance_book = OrderBook(depth=20)
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    BINANCE_DEPTH_REST,
                    params={"symbol": "BTCUSDT", "limit": 100},
                    ssl=SSL_CTX,
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Binance depth snapshot failed: {resp.status}")
                    data = await resp.json()
            ob = state.binance_book
            ob.bids.clear()
            ob.asks.clear()
            for bid in data.get("bids", []):
                ob.bids[float(bid[0])] = float(bid[1])
            for ask in data.get("asks", []):
                ob.asks[float(ask[0])] = float(ask[1])
            last_update_id = data.get("lastUpdateId", 0)
            ob.last_update_id = last_update_id

            async with websockets.connect(
                BINANCE_DEPTH_WS, ssl=SSL_CTX,
                ping_interval=20, ping_timeout=10,
            ) as ws:
                log.info("Binance depth feed connected")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        u, U = msg.get("u", 0), msg.get("U", 0)
                        if u <= last_update_id:
                            continue
                        if U > last_update_id + 1:
                            async with aiohttp.ClientSession() as s:
                                async with s.get(BINANCE_DEPTH_REST, params={"symbol": "BTCUSDT", "limit": 100}, ssl=SSL_CTX) as r:
                                    data = await r.json()
                            ob.bids.clear()
                            ob.asks.clear()
                            for bid in data.get("bids", []):
                                ob.bids[float(bid[0])] = float(bid[1])
                            for ask in data.get("asks", []):
                                ob.asks[float(ask[0])] = float(ask[1])
                            last_update_id = data.get("lastUpdateId", 0)
                            continue
                        for bid in msg.get("bids", []):
                            p, sz = float(bid[0]), float(bid[1])
                            if sz == 0:
                                ob.bids.pop(p, None)
                            else:
                                ob.bids[p] = sz
                        for ask in msg.get("asks", []):
                            p, sz = float(ask[0]), float(ask[1])
                            if sz == 0:
                                ob.asks.pop(p, None)
                            else:
                                ob.asks[p] = sz
                        last_update_id = u
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
        except Exception as e:
            log.warning("Binance depth disconnected: %s — reconnecting in 2s", e)
            await asyncio.sleep(2)


# ── Deribit IV feed (Layer 1: for Layer 2 fair value σ) ─────────────────────────

async def deribit_feed(state: SharedState):
    """Stream Deribit volatility index for Layer 2 fair value."""
    channel = "deribit_volatility_index.btc_usd"
    req_id = 0
    while True:
        try:
            async with websockets.connect(
                DERIBIT_WS, ssl=SSL_CTX,
                ping_interval=20, ping_timeout=10,
            ) as ws:
                req_id += 1
                await ws.send(json.dumps({
                    "jsonrpc": "2.0", "method": "public/subscribe",
                    "params": {"channels": [channel]}, "id": req_id,
                }))
                log.info("Deribit IV feed connected")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        if msg.get("method") == "subscription":
                            data = msg.get("params", {}).get("data", {})
                            if isinstance(data, dict) and "volatility" in data:
                                state.deribit_iv = float(data["volatility"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
        except Exception as e:
            log.warning("Deribit feed disconnected: %s — reconnecting in 2s", e)
            await asyncio.sleep(2)


# ── Coinbase WebSocket feed ─────────────────────────────────────────────────────

async def coinbase_feed(state: SharedState, stack: DecisionStack, use_binance: bool = False):
    """
    Streams BTC price: Coinbase (default) or Binance (--binance).
    Calls stack.on_btc_price() on every tick.
    """
    if use_binance:
        await binance_feed(state, stack)
        return
    subscribe = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channel": "ticker",
    }

    while True:
        try:
            async with websockets.connect(
                COINBASE_WS, ssl=SSL_CTX,
                ping_interval=20, ping_timeout=10,
                max_size=2**20,
            ) as ws:
                await ws.send(json.dumps(subscribe))
                log.info("Coinbase WebSocket connected")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        events = msg.get("events", [])
                        for ev in events:
                            tickers = ev.get("tickers", [])
                            for tk in tickers:
                                if tk.get("product_id") != "BTC-USD":
                                    continue
                                price_str = tk.get("price") or tk.get("best_bid")
                                if not price_str:
                                    continue
                                price = float(price_str)
                                if price <= 0:
                                    continue

                                now = time.time()

                                # Feed into GARCH
                                stack.on_btc_price(price, now)
                                state.btc_price = price
                                if not state.btc_price_history or now - state.btc_price_history[-1][0] >= 0.5:
                                    state.btc_price_history.append((now, price))
                                state.btc_prices_1s.append(price)

                                # True 10-second lookback using timestamps
                                cutoff10 = now - 10.0
                                p10 = 0.0
                                for _ts, _p in reversed(list(state.btc_price_history)):
                                    if _ts <= cutoff10:
                                        p10 = _p
                                        break
                                state.btc_price_10s = p10 if p10 > 0 else (state.btc_price_history[0][1] if state.btc_price_history else 0.0)

                                # Record log-return for GARCH refit history
                                if state.return_history and price > 0:
                                    prev = state.btc_prices_1s[-2] if len(state.btc_prices_1s) >= 2 else price
                                    if prev > 0:
                                        lr = float(np.log(price / prev))
                                        state.return_history.append(lr)
                                        if len(state.return_history) > 10000:
                                            state.return_history = state.return_history[-5000:]

                                # Accumulate 1-second bars for VPIN
                                bar_age = now - state.btc_bar_ts
                                if bar_age >= 1.0:
                                    if state.btc_bar_open > 0 and state.btc_bar_vol > 0:
                                        stack.on_btc_bar_1s(
                                            state.btc_bar_open, price,
                                            state.btc_bar_vol)
                                    state.btc_bar_open = price
                                    state.btc_bar_vol  = 0.0
                                    state.btc_bar_ts   = now
                                else:
                                    if state.btc_bar_open == 0:
                                        state.btc_bar_open = price
                                    vol = float(tk.get("volume_24h", 0) or 0)
                                    state.btc_bar_vol += vol / 86400.0

                                # Feed trade time into Hawkes
                                stack.on_btc_trade(now)
                                state.btc_trade_times.append(now)
                                state.event_times_history.append(now)

                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass

        except Exception as e:
            log.warning("Coinbase WS disconnected: %s — reconnecting in 2s", e)
            await asyncio.sleep(2)


# ── Polymarket CLOB WebSocket feed ─────────────────────────────────────────────

async def poly_clob_feed(state: SharedState, stack: DecisionStack):
    """
    Streams Polymarket L2 order book for UP + DOWN tokens.
    Reads state.token_up / state.token_dn on each (re)connect so that window
    rollovers automatically get the new tokens without restarting the process.
    When state.clob_refresh is set, the current connection is closed and a new
    one is opened with the latest tokens.
    """

    def parse_levels(raw) -> list[list[float]]:
        out = []
        for item in (raw or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append([float(item[0]), float(item[1])])
            elif isinstance(item, dict):
                p = item.get("price") or item.get("p", 0)
                s = item.get("size") or item.get("q", 0)
                if p and s:
                    out.append([float(p), float(s)])
        return out

    def book_stats(bids, asks):
        if not bids or not asks:
            return {"bid": 0.0, "ask": 1.0, "mid": 0.5,
                    "obi": 0.0, "tob_bid": 0.0, "tob_ask": 0.0, "spread": 1.0}
        best_bid = max(b[0] for b in bids)
        best_ask = min(a[0] for a in asks)
        best_bid_sz = next((b[1] for b in bids if b[0] == best_bid), 0)
        best_ask_sz = next((a[1] for a in asks if a[0] == best_ask), 0)
        mid = (best_bid + best_ask) / 2
        top5_bid = sum(b[1] for b in sorted(bids, key=lambda x: -x[0])[:5])
        top5_ask = sum(a[1] for a in sorted(asks, key=lambda x:  x[0])[:5])
        tot = top5_bid + top5_ask
        obi = (top5_bid - top5_ask) / tot if tot > 0 else 0.0
        return {
            "bid": best_bid, "ask": best_ask, "mid": mid,
            "obi": obi,
            "tob_bid": best_bid_sz,
            "tob_ask": best_ask_sz,
            "spread": best_ask - best_bid,
        }

    while True:
        # Clear the refresh event before connecting (so any set during connect is caught next loop)
        state.clob_refresh.clear()

        token_up = state.token_up
        token_dn = state.token_dn
        if not token_up or not token_dn:
            await asyncio.sleep(1)
            continue

        subscribe = {"assets_ids": [token_up, token_dn], "type": "market"}

        try:
            async with websockets.connect(
                CLOB_WS, ssl=SSL_CTX,
                ping_interval=10, ping_timeout=8,
                max_size=2**22,
            ) as ws:
                await ws.send(json.dumps(subscribe))
                log.info("Polymarket CLOB WebSocket connected (UP=%s… DN=%s…)",
                         token_up[:8], token_dn[:8])

                ping_task    = asyncio.create_task(_poly_ping(ws))
                refresh_task = asyncio.create_task(state.clob_refresh.wait())

                try:
                    while True:
                        recv_task = asyncio.create_task(ws.recv())
                        done, _ = await asyncio.wait(
                            {recv_task, refresh_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Tokens changed — close WebSocket, outer loop reconnects
                        if refresh_task in done:
                            recv_task.cancel()
                            log.info("CLOB refresh triggered — reconnecting with new tokens")
                            break

                        raw = recv_task.result()
                        if isinstance(raw, bytes):
                            raw = raw.decode()
                        raw = raw.strip()
                        if not raw or raw in ("PONG", "pong"):
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
                            token = msg.get("asset_id") or msg.get("market", "")

                            if etype in ("book", "price_change", "best_bid_ask"):
                                bids_raw = msg.get("bids") or []
                                asks_raw = msg.get("asks") or []
                                bids = parse_levels(bids_raw)
                                asks = parse_levels(asks_raw)

                                if not bids and not asks:
                                    bid_p = msg.get("bid_price") or msg.get("best_bid")
                                    ask_p = msg.get("ask_price") or msg.get("best_ask")
                                    bid_s = msg.get("bid_size", 10)
                                    ask_s = msg.get("ask_size", 10)
                                    if bid_p:
                                        bids = [[float(bid_p), float(bid_s)]]
                                    if ask_p:
                                        asks = [[float(ask_p), float(ask_s)]]

                                if bids or asks:
                                    stats = book_stats(bids, asks)
                                    state.book_cache[token] = stats
                                    state._clob_msg_count += 1
                                    if state._clob_msg_count % 100 == 0:
                                        log.info("CLOB feed alive: %d msgs | UP mid=%.3f DN mid=%.3f",
                                                 state._clob_msg_count,
                                                 state.book_cache.get(token_up, {}).get("mid", 0),
                                                 state.book_cache.get(token_dn, {}).get("mid", 0))

                                    if token == token_up:
                                        state.poly_mid = stats["mid"]
                                        stack.on_poly_book(
                                            {p: s for p, s in bids},
                                            {p: s for p, s in asks},
                                        )
                finally:
                    ping_task.cancel()
                    refresh_task.cancel()

        except Exception as e:
            log.warning("Poly CLOB WS error: %s — reconnecting in 2s", e)
            await asyncio.sleep(2)


async def _poly_ping(ws):
    while True:
        await asyncio.sleep(10)
        try:
            await ws.send("PING")
        except Exception:
            break


# ── Layer 1–6 pipeline (advanced system) ───────────────────────────────────────

def run_layer_pipeline(
    state: SharedState,
    token_up: str,
    token_dn: str,
    bankroll: float,
) -> tuple:
    """
    Run Layer 2, 4, 5, 6. Returns (fair_value, sig6, sig2, sig5) or (None, None, None, None) on error.
    Layer 1 = data feeds (already in state).
    Layer 3 = stale check (in decision_stack).
    """
    try:
        from layer2_engine import Layer2Engine, Layer2Config
        from layer4_merton_jump import Layer4MertonEngine
        from layer5_hmm_regime import Layer5HMMRegime
        from layer6_risk_execution import Layer6Engine
    except ImportError as e:
        log.warning("Layer pipeline import error: %s", e)
        return (None, None, None, None)

    # Build state for Layer 2 (expects poly_book, deribit_iv, binance_book)
    class L2State:
        pass
    l2 = L2State()
    l2.poly_book = {}
    book_up = state.book_cache.get(token_up) or {}
    book_dn = state.book_cache.get(token_dn) or {}
    if book_up:
        l2.poly_book[token_up] = {
            "mid": book_up.get("mid", 0.5),
            "order_book_imbalance": book_up.get("obi", 0.0),
            "bids": {book_up.get("bid", 0.5): 1} if book_up.get("bid") else {},
            "asks": {book_up.get("ask", 0.5): 1} if book_up.get("ask") else {},
        }
    if book_dn:
        l2.poly_book[token_dn] = {
            "mid": book_dn.get("mid", 0.5),
            "order_book_imbalance": book_dn.get("obi", 0.0),
            "bids": {book_dn.get("bid", 0.5): 1} if book_dn.get("bid") else {},
            "asks": {book_dn.get("ask", 0.5): 1} if book_dn.get("ask") else {},
        }
    l2.deribit_iv = state.deribit_iv if state.deribit_iv else None
    l2.binance_book = state.binance_book
    l2.last_binance_trade = state.last_binance_trade or (
        {"p": str(state.btc_price)} if state.btc_price else None
    )

    t_rem = seconds_remaining()
    cfg = Layer2Config(
        strike=state.strike,
        time_remaining=t_rem,
        token_up=token_up,
        token_down=token_dn,
    )
    engine2 = Layer2Engine(config=cfg)
    sig2 = engine2.evaluate(l2, strike=state.strike, time_remaining=t_rem)

    # Layer 4: Merton
    merton = Layer4MertonEngine()
    sig4 = merton.evaluate(sig2, deribit_iv=l2.deribit_iv)
    fair_value = 0.5 * sig2.fair_value + 0.5 * sig4.p_up

    # Layer 5: HMM regime
    hmm_state = type("S", (), {"btc_price_history": state.btc_price_history})()
    sig5 = Layer5HMMRegime().evaluate(sig2, state=hmm_state)
    regime = sig5.regime or "medium_vol"

    # Layer 6: risk and execution
    sig6 = Layer6Engine().evaluate(
        fair_value=fair_value,
        poly_mid=sig2.poly_mid,
        regime=regime,
        capital=bankroll,
    )

    return (fair_value, sig6, sig2, sig5)


# ── Strike calibration ─────────────────────────────────────────────────────────

def calibrate_strike(state: SharedState) -> bool:
    """
    Lock the window strike from Polymarket mid + current BTC price.

    Rules:
      - Must happen within first STRIKE_LOCK_S seconds of the window
        (after that, mid no longer reflects the opening price)
      - Only lock when mid is in [0.42, 0.58] — truly uncertain market
        means BTC ≈ strike right now
      - If we miss the lock window, skip this window entirely (safer than
        trading with a wrong strike)
    """
    if state.strike_set:
        return True

    mid  = state.poly_mid
    btc  = state.btc_price
    secs = seconds_in_window()

    if btc <= 0 or mid <= 0:
        return False

    # Missed the lock window — skip this window
    if secs > STRIKE_LOCK_S:
        return False

    lo, hi = STRIKE_MID_RANGE
    if lo < mid < hi:
        state.strike     = btc
        state.strike_set = True
        log.info("Strike locked: $%.2f  (btc=$%.2f  mid=%.3f  T+%.0fs)",
                 state.strike, btc, mid, secs)
        return True

    # mid is away from 0.5 — use Gaussian inversion to back out the implied strike
    if mid < 0.01 or mid > 0.99:
        return False   # near-resolved market, don't enter

    from scipy.special import ndtri
    t_frac = max(seconds_remaining(), 1.0) / 300.0
    sigma_est = 0.50 * (t_frac ** 0.5)   # typical 5-min BTC vol scaled by √(T/300)
    pct_est   = float(ndtri(min(max(mid, 0.02), 0.98))) * sigma_est
    state.strike     = btc / (1 + pct_est / 100)
    state.strike_set = True
    log.info("Strike inferred: $%.2f  (btc=$%.2f  mid=%.3f  T+%.0fs)",
             state.strike, btc, mid, secs)
    return True

    return False   # keep waiting for a better calibration moment


# ── Order sizing ───────────────────────────────────────────────────────────────

def compute_size(risk: RiskManager, p_model: float, p_market: float,
                 signal_mult: float, capital: float = None) -> float:
    """Kelly size, scaled by DecisionStack multiplier and vol regime.
    Capped at 5% capital per market."""
    capital = capital or INITIAL_BANKROLL
    max_per_market = capital * MAX_POSITION_PCT
    base = risk.kelly_size(p_model, p_market)
    adj  = base * signal_mult
    adj  = risk.adjust_for_vol_regime(adj, risk_manager_vol_regime(risk))
    adj  = max(MIN_TRADE_USD, min(max_per_market, MAX_TRADE_USD, round(adj, 2)))
    return adj

def risk_manager_vol_regime(risk: RiskManager) -> str:
    return "NORMAL_VOL"   # risk_man doesn't track vol separately; GARCH does


# ── Emergency exit ─────────────────────────────────────────────────────────────

async def emergency_exit(state: SharedState, executor: LiveExecutor,
                         risk: RiskManager, exec_risk, paper: bool) -> bool:
    """
    Sell our open position at market (best bid) if conditions go toxic.
    Returns True if we exited.
    """
    pos = state.open_position
    if not pos:
        return False

    t_left = seconds_remaining()
    if t_left < EMERGENCY_TIME_S:
        return False  # not worth exiting this close to expiry

    book = state.book_cache.get(pos["token"])
    if not book:
        return False

    sell_price = book["bid"]
    if sell_price <= 0:
        return False

    if paper:
        log.warning("EMERGENCY EXIT (paper) | side=%s entry=%.3f sell=%.3f t_left=%.0fs",
                    pos["side"], pos["price"], sell_price, t_left)
        realized = (sell_price - pos["price"]) * pos["size"]
        risk.record_trade(realized, pos["side"], pos["price"], sell_price, t_left)
        exec_risk.close_position(pos["cost"])
        state.open_position = None
        return True

    log.warning("EMERGENCY EXIT | selling %s at %.3f (entry %.3f) t_left=%.0fs",
                pos["token"][:16], sell_price, pos["price"], t_left)

    result = executor.place_limit_order(
        token_id=pos["token"],
        side="SELL",
        price=round(sell_price, 2),
        size=round(pos["size"], 2),
    )
    if result.success:
        realized = (sell_price - pos["price"]) * pos["size"]
        risk.record_trade(realized, pos["side"], pos["price"], sell_price, t_left)
        exec_risk.close_position(pos["cost"])
        state.open_position = None
        log.info("Emergency exit OK | realized=$%.4f", realized)
    else:
        log.error("Emergency exit FAILED: %s", result.error)
    return result.success


# ── Hourly refit task ──────────────────────────────────────────────────────────

async def refit_task(state: SharedState, stack: DecisionStack):
    """Retrain GARCH and Hawkes from accumulated live data every hour."""
    while True:
        await asyncio.sleep(REFIT_INTERVAL_S)
        try:
            rets = np.array(state.return_history[-5000:], dtype=float)
            if len(rets) > 200:
                stack.garch.fit(rets)
                log.info("GARCH refit — ω=%.2e α=%.3f β=%.3f h=%.2e",
                         stack.garch.omega, stack.garch.alpha,
                         stack.garch.beta, stack.garch.h or 0)

            evts = np.array(state.event_times_history[-2000:], dtype=float)
            if len(evts) > 100:
                stack.hawkes.fit(evts)
                log.info("Hawkes refit — μ=%.3f α=%.3f β=%.3f (br=%.3f)",
                         stack.hawkes.mu, stack.hawkes.alpha,
                         stack.hawkes.beta, stack.hawkes.branching_ratio)

            state.last_refit_ts = time.time()
        except Exception as e:
            log.warning("Refit error: %s", e)


# ── Auto-refresh market tokens on window rollover ──────────────────────────────

async def _refresh_market(state: SharedState):
    """
    Called once per window rollover. Discovers the newest BTC 5-min market
    and updates state.token_up / state.token_dn. Signals the CLOB feed to
    reconnect with the new tokens.
    """
    try:
        ssl_conn = aiohttp.TCPConnector(ssl=SSL_CTX)
        async with aiohttp.ClientSession(connector=ssl_conn) as http:
            market = await discover_btc_market(http)
        if not market:
            log.warning("Token refresh: could not discover new market — keeping old tokens")
            return
        new_up = market["token_up"]
        new_dn = market["token_dn"]
        if new_up == state.token_up and new_dn == state.token_dn:
            log.info("Token refresh: same tokens as before (%s…)", new_up[:8])
            return
        state.token_up = new_up
        state.token_dn = new_dn
        state.market_question = market.get("question", "")
        # Clear stale book entries for old tokens
        state.book_cache.clear()
        # Signal CLOB feed to reconnect with new tokens
        state.clob_refresh.set()
        log.info("Token refresh: new market → %s", state.market_question[:60])
        log.info("  UP: %s…  DN: %s…", new_up[:20], new_dn[:20])
    except Exception as e:
        log.warning("Token refresh error: %s", e)


# ── Main evaluation loop ───────────────────────────────────────────────────────

async def eval_loop(
    state: SharedState,
    stack: DecisionStack,
    executor: LiveExecutor,
    exec_risk: ExecRiskManager,
    risk: RiskManager,
    paper: bool,
    bankroll: float = None,
):
    """
    Runs every EVAL_INTERVAL_S (100 ms).
    1. Check circuit breakers
    2. Window management (detect new window, reset strike, re-discover tokens)
    3. Build MarketState → DecisionStack.evaluate()
    4. If TRADE and no open position → risk check → execute
    5. If open position and conditions deteriorate → emergency exit
    """
    token_up = state.token_up
    token_dn = state.token_dn

    n = 0
    while True:
        n += 1
        state.last_eval_n = n
        await asyncio.sleep(EVAL_INTERVAL_S)

        # ── 1. circuit breakers ──────────────────────────────────────────────
        cb_ok, cb_reason = risk.check_circuit_breakers()
        if not cb_ok:
            if n % 600 == 0:
                log.warning("Circuit breaker: %s", cb_reason)
            continue

        # ── 2. window management ─────────────────────────────────────────────
        wid = current_window_id()
        if wid != state.window_id:
            prev_wid = state.window_id
            state.window_id   = wid
            state.strike_set  = False
            state.strike      = 0.0

            if state.open_position:
                # Window ended — resolve outcome
                pos = state.open_position
                btc  = state.btc_price
                won  = (btc > pos["strike"]) if pos["side"] == "UP" else (btc < pos["strike"])
                pnl  = pos["size"] * (1.0 / pos["price"] - 1) * 0.98 if won else -pos["cost"]
                risk.record_trade(pnl, pos["side"], pos["p_model"], pos["price"],
                                  time.time() - pos["ts"])
                risk.update_balance(risk.current_balance + pnl)
                state.trades_log.append({
                    "window":    prev_wid,
                    "side":      pos["side"],
                    "p_model":   pos["p_model"],
                    "p_market":  pos["price"],
                    "outcome":   1 if won else 0,
                    "pnl":       round(pnl, 4),
                    "btc_entry": pos["btc_entry"],
                    "btc_exit":  btc,
                })
                log.info("Window %d resolved: %s | btc=%.2f strike=%.2f | %s | pnl=$%.4f | balance=$%.2f",
                         prev_wid, pos["side"], btc, pos["strike"],
                         "WIN" if won else "LOSS", pnl, risk.current_balance)
                exec_risk.close_position(pos["cost"])
                state.open_position = None

            # ── edge drift alert every 20 trades ─────────────────────────
            n_trades = len(state.trades_log)
            if n_trades > 0 and n_trades % 20 == 0:
                drift = risk.edge_drift()
                wr20  = drift.get("win_rate", 0)
                edges = [t["p_model"] - t.get("p_market", 0.5) for t in state.trades_log[-20:]]
                mean_net_edge = float(np.mean(edges)) - TOTAL_COST
                if not drift["ok"]:
                    log.warning(
                        "EDGE DRIFT ⚠️  | last 20 trades WR=%.1f%% "
                        "mean_net_edge=%+.3f | CIRCUIT BREAKER: pausing for 1 window",
                        wr20 * 100, mean_net_edge,
                    )
                    risk._pause(f"Edge drift: WR={wr20:.1%} over last 20 trades")
                else:
                    log.info(
                        "Edge check ✅  | last 20 trades WR=%.1f%% mean_net_edge=%+.3f",
                        wr20 * 100, mean_net_edge,
                    )

            log.info("New window %d (T-%.0fs)", wid, seconds_remaining())

            # ── Auto-refresh market tokens for new window ─────────────────
            asyncio.ensure_future(_refresh_market(state))

            # Pre-fetch token metadata (tick_size, neg_risk, fee_rate) so
            # the first market order doesn't incur 3 extra HTTP lookups.
            if not paper:
                asyncio.ensure_future(
                    asyncio.get_event_loop().run_in_executor(
                        None, executor.prefetch_token, state.token_up
                    )
                )
                asyncio.ensure_future(
                    asyncio.get_event_loop().run_in_executor(
                        None, executor.prefetch_token, state.token_dn
                    )
                )

        # Pull latest tokens (updated by _refresh_market)
        token_up = state.token_up
        token_dn = state.token_dn

        # ── 3. strike calibration ─────────────────────────────────────────────
        if not calibrate_strike(state):
            continue

        if state.btc_price <= 0:
            continue

        # ── 4. emergency exit check on open position ──────────────────────────
        if state.open_position:
            pos    = state.open_position
            mq     = stack.vpin.market_quality
            pct_diff = (state.btc_price - state.strike) / state.strike * 100
            fv_result = stack.compute_fair_value(pct_diff, seconds_remaining())
            our_p = fv_result["fair_value"] if pos["side"] == "UP" else (1 - fv_result["fair_value"])

            should_exit = (
                mq["vpin"] > EMERGENCY_VPIN
                and our_p < EMERGENCY_PMODEL
                and seconds_remaining() > EMERGENCY_TIME_S
            )
            if should_exit:
                await emergency_exit(state, executor, risk, exec_risk, paper)
            continue  # don't enter new trade while position is open

        # ── 5. entry cutoff ───────────────────────────────────────────────────
        if seconds_in_window() > ENTRY_CUTOFF_S:
            continue

        # ── 6. build MarketState ──────────────────────────────────────────────
        book_up = state.book_cache.get(token_up) or {}
        book_dn = state.book_cache.get(token_dn) or {}

        if not book_up or book_up.get("mid", 0) <= 0:
            continue

        p_market = book_up["mid"]
        obi      = book_up.get("obi", 0.0)
        tob_bid  = book_up.get("tob_bid", 0.0)
        tob_ask  = book_up.get("tob_ask", 0.0)

        ms = MarketState(
            btc_price      = state.btc_price,
            btc_price_10s  = state.btc_price_10s,
            strike         = state.strike,
            time_remaining = seconds_remaining(),
            p_market       = p_market,
            poly_bid_vol   = tob_bid,
            poly_ask_vol   = tob_ask,
            poly_obi       = obi,
            timestamp      = time.time(),
        )

        # ── 7. Layer 1–6 pipeline (advanced) + evaluate ───────────────────────
        cap = bankroll or INITIAL_BANKROLL
        layer_sigs = run_layer_pipeline(state, token_up, token_dn, cap)
        fair_val, sig6, sig2, sig5 = layer_sigs
        use_layer = (fair_val is not None and sig6 is not None and sig6.trade)
        sig: SignalResult = stack.evaluate(ms, use_layer_signals=layer_sigs if use_layer else None)

        if n % EMIT_EVERY_N == 0:
            pct_diff = (state.btc_price - state.strike) / state.strike * 100
            ml_tag = f" ml={sig.ml_p_up:.3f}" if sig.ml_p_up > 0 else ""
            log.info(
                "T-%.0fs | btc=%.2f diff=%+.3f%% move10s=%+.3f%% | "
                "fair=%.3f(g=%.3f m=%.3f%s) p_mkt=%.3f edge_net=%+.3f | "
                "vpin=%.3f hawkes=%s vol=%s hmm=%s | %s",
                seconds_remaining(), state.btc_price, pct_diff, sig.btc_move_10s,
                sig.fair_value, sig.merton_p_up or sig.fair_value, sig.merton_p_up or 0.0, ml_tag,
                sig.p_market, sig.edge_net,
                sig.vpin, sig.hawkes_regime, sig.vol_regime, sig.hmm_regime or "-",
                sig.action if sig.action != "WAIT" else f"WAIT({sig.veto_reason})",
            )

        if sig.action != "TRADE":
            continue

        # ── 8. size + risk gate ───────────────────────────────────────────────
        p_for_side = sig.p_market if sig.side == "UP" else (1 - sig.p_market)
        size_usd   = sig.size_usd if sig.size_usd > 0 else compute_size(
            risk, sig.fair_value, p_for_side, sig.size_multiplier, capital=cap)
        # Enforce Polymarket's $1 minimum — bump up if edge is strong enough
        size_usd = max(size_usd, MIN_TRADE_USD)

        allowed, reason = exec_risk.allow_trade(
            size_usd / p_for_side if p_for_side > 0 else size_usd,
            p_for_side,
        )
        if not allowed:
            log.info("Trade blocked by exec_risk: %s", reason)
            continue

        # ── 9. token + price ──────────────────────────────────────────────────
        token   = token_up if sig.side == "UP" else token_dn
        # For DOWN we buy the DOWN token; its ask is 1 - UP_bid
        if sig.side == "UP":
            entry_price = book_up.get("ask", p_market)
        else:
            entry_price = book_dn.get("ask", 1 - p_market) if book_dn else (1 - p_market)

        entry_price = max(0.01, min(0.99, round(entry_price, 2)))
        shares      = round(size_usd / entry_price, 2)

        log.info(
            "⚡ SIGNAL: %s | fair=%.3f [gauss+merton+ML] p_mkt=%.3f edge_net=%+.3f "
            "size=$%.2f @%.3f | mult=%.2f | vpin=%.3f hawkes=%s hmm=%s ml=%.3f",
            sig.side, sig.fair_value, sig.p_market, sig.edge_net,
            size_usd, entry_price, sig.size_multiplier, sig.vpin, sig.hawkes_regime,
            sig.hmm_regime or "-", sig.ml_p_up,
        )

        # ── 10. execute ──────────────────────────────────────────────────────
        if paper:
            success     = True
            order_id    = "PAPER"
            latency_ms  = 0.0
            log.info("PAPER TRADE | %s %.2f @ %.3f (cost=$%.2f)",
                     sig.side, shares, entry_price, size_usd)
        else:
            # Add 0.08 slippage buffer to worst_price so a market order always
            # fills immediately even if the ask moves up during the ~750ms round-trip.
            # Without this, the order becomes a resting limit that never fills.
            worst = min(entry_price + 0.08, 0.95)
            t0     = time.perf_counter()
            result = executor.place_market_order(token, "BUY", size_usd,
                                                 worst_price=worst)
            latency_ms = (time.perf_counter() - t0) * 1000
            success    = result.success
            order_id   = result.order_id
            if not success:
                log.error("Order FAILED: %s (%.0fms)", result.error, latency_ms)
                continue

        exec_risk.record_trade(
            result if not paper else type("R", (), {"success": True, "cost": size_usd,
                                                    "token_id": token,
                                                    "order_id": "PAPER",
                                                    "side": "BUY",
                                                    "price": entry_price,
                                                    "size": shares,
                                                    "timestamp": time.time()})()
        )

        state.open_position = {
            "token":      token,
            "side":       sig.side,
            "price":      entry_price,
            "p_model":    sig.fair_value,
            "p_market":   sig.p_market,
            "size":       shares,
            "cost":       size_usd,
            "ts":         time.time(),
            "btc_entry":  state.btc_price,
            "btc_move_10s": sig.btc_move_10s,
            "strike":     state.strike,
            "order_id":   order_id,
            "latency_ms": latency_ms,
        }

        log.info("Position open: %s | %s | entry=%.3f | cost=$%.2f | latency=%.0fms",
                 sig.side, token[:16], entry_price, size_usd, latency_ms)


# ── Session summary ─────────────────────────────────────────────────────────────

def print_summary(state: SharedState, risk: RiskManager):
    trades = state.trades_log
    if not trades:
        print("\n  No trades completed this session.")
        return

    wins  = [t for t in trades if t["outcome"] == 1]
    wr    = len(wins) / len(trades)
    pnl   = sum(t["pnl"] for t in trades)
    edges = [t["p_model"] - t.get("p_market", 0.5) for t in trades]

    print("\n" + "=" * 68)
    print("  SESSION SUMMARY")
    print("=" * 68)
    print(f"  Trades:           {len(trades)}")
    print(f"  Win rate:         {wr:.1%}")
    print(f"  Net P&L:          ${pnl:+.2f}")
    print(f"  Final balance:    ${risk.current_balance:.2f}")
    print(f"  Mean edge (raw):  {sum(edges)/len(edges):+.3f}")
    print(f"  Mean edge (net):  {sum(edges)/len(edges)-0.035:+.3f}  (after 3.5% fee)")
    drift = risk.edge_drift()
    print(f"  Edge drift OK:    {drift['ok']}  (last {drift['n']} trades WR={drift.get('win_rate','?')})")
    print("=" * 68)

    # Save OOS records
    import json
    os.makedirs("logs", exist_ok=True)
    path = f"logs/oos_live_{_ts}.json"
    with open(path, "w") as f:
        json.dump(trades, f, indent=2)
    print(f"  OOS records → {path}")


# ── Entry point ─────────────────────────────────────────────────────────────────

async def run(paper: bool = True, bankroll: float = None, use_binance: bool = False):
    bankroll = bankroll if bankroll is not None else INITIAL_BANKROLL
    max_per_market = bankroll * MAX_POSITION_PCT
    max_total      = bankroll * MAX_EXPOSURE_PCT

    print("=" * 68)
    print(f"  Polymarket BTC Scalper — {'PAPER' if paper else '⚠️  LIVE'}")
    print("=" * 68)
    print(f"  System:      Layer 1–6 (Deribit IV, Layer2, Merton, HMM, Layer6)")
    print(f"  Edge:        Polymarket reprices 2-5s after BTC spot move")
    print(f"  Latency:     AWS Ireland → Polymarket ~10-20ms")
    print(f"  Eval rate:   {int(1/EVAL_INTERVAL_S)} Hz")
    print(f"  Entry zone:  first {ENTRY_CUTOFF_S}s of each 5-min window")
    print(f"  Bankroll:    ${bankroll:.0f} (max ${max_per_market:.2f}/market)")
    print(f"  Limits:      max {MAX_POSITION_PCT*100:.0f}% per market, {MAX_EXPOSURE_PCT*100:.0f}% total exposure")
    print("=" * 68)

    # ── init components ──────────────────────────────────────────────────────
    state = SharedState()
    stack = DecisionStack()

    # Load pre-trained GARCH + Hawkes parameters (run warmup.py first)
    warmup_ok = load_warmup(stack.garch, stack.hawkes)
    if not warmup_ok:
        log.warning("No warmup state found. Run:  python3 warmup.py")
        log.warning("GARCH and Hawkes starting cold — first 30 min of signals unreliable.")
        log.warning("Continuing anyway; models will self-calibrate from live data.")

    risk       = RiskManager(bankroll)
    exec_risk  = ExecRiskManager(
        max_loss_per_session=bankroll * 0.10,
        max_position=max_total,           # 20% capital total exposure
        max_orders_per_window=2,
        max_order_size=max_per_market,    # 5% capital per market
        min_order_size=MIN_TRADE_USD,
        max_daily_orders=50,
        circuit_breaker_consecutive_losses=4,
    )

    executor = LiveExecutor(tick_size="0.01")
    if not paper:
        if not executor.connect():
            log.error("Cannot connect to Polymarket CLOB — aborting")
            return
        log.info("Executor connected: %s", executor.metrics())
    else:
        log.info("Paper mode — orders will be simulated")

    # ── discover market ──────────────────────────────────────────────────────
    ssl_conn = aiohttp.TCPConnector(ssl=SSL_CTX)
    async with aiohttp.ClientSession(connector=ssl_conn) as http:
        market = await discover_btc_market(http)

    if not market:
        log.error("No BTC market found. Exiting.")
        return

    log.info("Market: %s", market["question"][:80])
    log.info("Token UP: %s", market["token_up"][:20])
    log.info("Token DN: %s", market["token_dn"][:20])

    # Store initial tokens in state so CLOB feed and eval_loop read from one place
    state.token_up        = market["token_up"]
    state.token_dn        = market["token_dn"]
    state.market_question = market.get("question", "")

    # ── launch tasks ─────────────────────────────────────────────────────────
    tasks = [
        coinbase_feed(state, stack, use_binance=use_binance),
        poly_clob_feed(state, stack),
        deribit_feed(state),
        eval_loop(state, stack, executor, exec_risk, risk, paper, bankroll),
        refit_task(state, stack),
    ]
    if use_binance:
        state.binance_book = OrderBook(depth=20)
        tasks.insert(2, binance_depth_feed(state))
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        print_summary(state, risk)


if __name__ == "__main__":
    import sys
    import argparse
    p = argparse.ArgumentParser(description="Polymarket BTC 5-min scalper")
    p.add_argument("--live", action="store_true", help="Live trading (default: paper)")
    p.add_argument("--bankroll", type=float, default=None,
                   help="Bankroll USD (default 100). Use 20 for $1 max/market test.")
    p.add_argument("--binance", action="store_true",
                   help="Use Binance for BTC feed (fallback when Coinbase fails)")
    args = p.parse_args()
    paper = not args.live
    bankroll = args.bankroll
    if not paper:
        print("\n  ⚠️  LIVE TRADING MODE — real money at risk")
        if bankroll is not None:
            print(f"  Bankroll: ${bankroll:.0f} → max ${bankroll*MAX_POSITION_PCT:.2f} per trade")
        confirm = input("  Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("  Aborted.")
            sys.exit(0)
    asyncio.run(run(paper=paper, bankroll=bankroll, use_binance=args.binance))
