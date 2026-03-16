"""
triple_streams.py — Layer 1 inputs: 5 WebSocket streams simultaneously
=====================================================================

1. Binance trades        wss://stream.binance.com:9443/ws/btcusdt@trade
2. Binance order book    wss://stream.binance.com:9443/ws/btcusdt@depth (L2 + imbalance)
3. Polymarket CLOB       wss://ws-subscriptions-clob.polymarket.com/ws/market (L2 yes/no depth + imbalance)
4. Deribit IV            wss://www.deribit.com/ws/api/v2 (volatility index + options IV)

Shared state is updated in real time. All streams run as concurrent asyncio tasks.
"""

import asyncio
import aiohttp
import json
import ssl
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import websockets

from orderbook import OrderBook

# ── URLs ─────────────────────────────────────────────────────────────────────
BINANCE_TRADE_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_DEPTH_WS = "wss://stream.binance.com:9443/ws/btcusdt@depth"
BINANCE_DEPTH_REST = "https://api.binance.com/api/v3/depth"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DERIBIT_WS = "wss://www.deribit.com/ws/api/v2"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def _order_book_imbalance(bids: dict, asks: dict, levels: int = 5) -> float:
    """OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol). Range -1 to +1."""
    sorted_bids = sorted(bids.keys(), reverse=True)[:levels]
    sorted_asks = sorted(asks.keys())[:levels]
    bid_vol = sum(bids.get(p, 0) for p in sorted_bids)
    ask_vol = sum(asks.get(p, 0) for p in sorted_asks)
    total = bid_vol + ask_vol
    return (bid_vol - ask_vol) / total if total > 0 else 0.0


def _depth_ratio(bids: dict, asks: dict, levels: int = 10) -> float:
    """Bid depth / ask depth. > 1.0 = bullish lean."""
    sorted_bids = sorted(bids.keys(), reverse=True)[:levels]
    sorted_asks = sorted(asks.keys())[:levels]
    bid_depth = sum(bids.get(p, 0) for p in sorted_bids)
    ask_depth = sum(asks.get(p, 0) for p in sorted_asks)
    return bid_depth / ask_depth if ask_depth > 0 else 1.0


@dataclass
class TripleStreamState:
    """Shared state updated by all streams (Layer 1 inputs)."""

    # Binance trades
    last_binance_trade: Optional[dict] = None  # {p, q, m, T, ...}
    binance_trade_count: int = 0

    # Binance order book (OrderBook instance)
    binance_book: OrderBook = field(default_factory=lambda: OrderBook(depth=20))

    # Polymarket order book (per token) — L2 depth + imbalance
    # asset_id -> {bids, asks, best_bid, best_ask, mid, spread, order_book_imbalance, depth_ratio}
    poly_book: dict = field(default_factory=dict)

    # Deribit: IV (volatility index) and options pricing
    deribit_iv: Optional[float] = None  # BTC volatility index (DVOL-like)
    deribit_iv_ts: Optional[float] = None
    deribit_ticker: Optional[dict] = None  # {mark_iv, greeks, ...} from options ticker

    # Price history for Layer 5 HMM regime (ts, price) — populated by trade stream
    btc_price_history: deque = field(default_factory=lambda: deque(maxlen=300))

    # Connection status
    binance_trade_connected: bool = False
    binance_depth_connected: bool = False
    poly_connected: bool = False
    deribit_connected: bool = False

    # Shutdown
    running: bool = True


@dataclass
class TripleStreamConfig:
    """Configuration for triple streams."""

    token_up: str = ""
    token_down: str = ""
    deribit_vol_index: str = "btc_usd"  # e.g. btc_usd, eth_usd


# ── Binance trades stream ────────────────────────────────────────────────────

async def _binance_trade_stream(state: TripleStreamState):
    """Binance trades: @trade stream."""
    while state.running:
        try:
            async with websockets.connect(
                BINANCE_TRADE_WS,
                ssl=SSL_CTX,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                state.binance_trade_connected = True
                async for raw in ws:
                    if not state.running:
                        return
                    try:
                        msg = json.loads(raw)
                        price = float(msg.get("p", 0))
                        qty = float(msg.get("q", 0))
                        is_buyer_maker = msg.get("m", False)
                        if price <= 0:
                            continue
                        state.last_binance_trade = msg
                        state.binance_trade_count += 1
                        state.binance_book.add_trade(price, qty, is_buyer_maker)
                        ts = time.time()
                        hist = state.btc_price_history
                        if not hist or ts - hist[-1][0] >= 0.5:  # throttle: 1 sample per 0.5s
                            hist.append((ts, price))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
        except Exception as e:
            state.binance_trade_connected = False
            print(f"  [Binance trade] {e}")
        await asyncio.sleep(2)


# ── Binance order book stream ─────────────────────────────────────────────────

async def _fetch_binance_depth_snapshot() -> Tuple[dict, int]:
    """Fetch depth snapshot from REST API. Returns (bids, asks) dict and lastUpdateId."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            BINANCE_DEPTH_REST,
            params={"symbol": "BTCUSDT", "limit": 100},
            ssl=SSL_CTX,
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Binance depth snapshot failed: {resp.status}")
            data = await resp.json()
    last_update_id = data.get("lastUpdateId", 0)
    return data, last_update_id


def _apply_depth_to_orderbook(ob: OrderBook, data: dict):
    """Apply Binance depth update (bids/asks arrays) to OrderBook."""
    for bid in data.get("bids", []):
        price, size = float(bid[0]), float(bid[1])
        if size == 0:
            ob.bids.pop(price, None)
        else:
            ob.bids[price] = size
    for ask in data.get("asks", []):
        price, size = float(ask[0]), float(ask[1])
        if size == 0:
            ob.asks.pop(price, None)
        else:
            ob.asks[price] = size


async def _binance_depth_stream(state: TripleStreamState):
    """Binance order book: @depth stream with snapshot sync."""
    while state.running:
        try:
            # Fetch initial snapshot
            snapshot, last_update_id = await _fetch_binance_depth_snapshot()
            ob = state.binance_book
            ob.bids.clear()
            ob.asks.clear()
            for bid in snapshot.get("bids", []):
                ob.bids[float(bid[0])] = float(bid[1])
            for ask in snapshot.get("asks", []):
                ob.asks[float(ask[0])] = float(ask[1])
            ob.last_update_id = last_update_id

            async with websockets.connect(
                BINANCE_DEPTH_WS,
                ssl=SSL_CTX,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                state.binance_depth_connected = True
                async for raw in ws:
                    if not state.running:
                        return
                    try:
                        msg = json.loads(raw)
                        u = msg.get("u", 0)  # final update id
                        U = msg.get("U", 0)  # first update id
                        # Discard if update is before snapshot
                        if u <= last_update_id:
                            continue
                        # Discard if gap (U > last_update_id + 1)
                        if U > last_update_id + 1:
                            # Re-sync: fetch snapshot again
                            snapshot, last_update_id = await _fetch_binance_depth_snapshot()
                            ob.bids.clear()
                            ob.asks.clear()
                            for bid in snapshot.get("bids", []):
                                ob.bids[float(bid[0])] = float(bid[1])
                            for ask in snapshot.get("asks", []):
                                ob.asks[float(ask[0])] = float(ask[1])
                            ob.last_update_id = last_update_id
                            continue
                        _apply_depth_to_orderbook(ob, msg)
                        last_update_id = u
                        ob.last_update_id = u
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
        except Exception as e:
            state.binance_depth_connected = False
            print(f"  [Binance depth] {e}")
        await asyncio.sleep(2)


# ── Polymarket order book stream (L2 depth + imbalance) ───────────────────────

def _parse_poly_book(msg: dict, asset_id: str) -> Optional[dict]:
    """Parse Polymarket book message into full L2 + best_bid/ask, imbalance, depth_ratio."""
    raw_bids = msg.get("bids", [])
    raw_asks = msg.get("asks", [])
    if not raw_bids or not raw_asks:
        return None

    def _p(x):
        return float(x.get("price", 0) if isinstance(x, dict) else (x[0] if x else 0))

    def _s(x):
        return float(
            x.get("size", 0)
            if isinstance(x, dict)
            else (x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else 0)
        )

    bids_dict = {}
    for b in raw_bids:
        p, s = _p(b), _s(b)
        if s > 0:
            bids_dict[p] = s
    asks_dict = {}
    for a in raw_asks:
        p, s = _p(a), _s(a)
        if s > 0:
            asks_dict[p] = s

    best_bid = max(bids_dict.keys()) if bids_dict else 0.0
    best_ask = min(asks_dict.keys()) if asks_dict else float("inf")
    sb = bids_dict.get(best_bid, 0)
    sa = asks_dict.get(best_ask, 0)

    obi = _order_book_imbalance(bids_dict, asks_dict, levels=5)
    dr = _depth_ratio(bids_dict, asks_dict, levels=10)

    return {
        "bids": bids_dict,
        "asks": asks_dict,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": (best_bid + best_ask) / 2 if best_ask < float("inf") else best_bid,
        "spread": best_ask - best_bid if best_ask < float("inf") else 0.0,
        "tob_bid_vol": sb * best_bid,
        "tob_ask_vol": sa * best_ask,
        "order_book_imbalance": obi,
        "depth_ratio": dr,
    }


async def _poly_depth_stream(state: TripleStreamState, config: TripleStreamConfig):
    """Polymarket CLOB WebSocket: real-time order book."""
    if not config.token_up or not config.token_down:
        print("  [Poly depth] token_up/token_down not set, skipping")
        return

    last_ping = 0.0
    tokens = [config.token_up, config.token_down]

    while state.running:
        try:
            async with websockets.connect(
                CLOB_WS,
                ssl=SSL_CTX,
                ping_interval=None,
                ping_timeout=None,
            ) as ws:
                await ws.send(
                    json.dumps({"assets_ids": tokens, "type": "market", "custom_feature_enabled": True})
                )
                state.poly_connected = True

                while state.running:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.5)
                        if not msg:
                            continue
                        try:
                            raw = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        items = raw if isinstance(raw, list) else [raw]
                        for data in items:
                            if not isinstance(data, dict):
                                continue
                            if data.get("event_type") == "book":
                                aid = data.get("asset_id", "")
                                book = _parse_poly_book(data, aid)
                                if book and aid:
                                    state.poly_book[aid] = book
                            elif data.get("event_type") == "best_bid_ask":
                                aid = data.get("asset_id", "")
                                bb = data.get("best_bid")
                                ba = data.get("best_ask")
                                if aid and bb is not None and ba is not None:
                                    prev = state.poly_book.get(aid, {})
                                    state.poly_book[aid] = {
                                        "bids": prev.get("bids", {}),
                                        "asks": prev.get("asks", {}),
                                        "best_bid": float(bb),
                                        "best_ask": float(ba),
                                        "mid": (float(bb) + float(ba)) / 2,
                                        "spread": float(ba) - float(bb),
                                        "tob_bid_vol": prev.get("tob_bid_vol", 0),
                                        "tob_ask_vol": prev.get("tob_ask_vol", 0),
                                        "order_book_imbalance": prev.get("order_book_imbalance", 0.0),
                                        "depth_ratio": prev.get("depth_ratio", 1.0),
                                    }
                    except asyncio.TimeoutError:
                        pass

                    if time.time() - last_ping > 10:
                        await ws.send("PING")
                        last_ping = time.time()

        except Exception as e:
            state.poly_connected = False
            print(f"  [Poly depth] {e}")
        await asyncio.sleep(2)


# ── Deribit IV stream (options implied vol + volatility index) ─────────────────

async def _deribit_stream(state: TripleStreamState, config: TripleStreamConfig):
    """Deribit WebSocket: volatility index (IV) and options pricing."""
    channel = f"deribit_volatility_index.{config.deribit_vol_index}"
    req_id = 0

    while state.running:
        try:
            async with websockets.connect(
                DERIBIT_WS,
                ssl=SSL_CTX,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                req_id += 1
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "public/subscribe",
                            "params": {"channels": [channel]},
                            "id": req_id,
                        }
                    )
                )
                state.deribit_connected = True

                async for raw in ws:
                    if not state.running:
                        return
                    try:
                        msg = json.loads(raw)
                        if "method" in msg and msg.get("method") == "subscription":
                            params = msg.get("params", {})
                            data = params.get("data", {})
                            if isinstance(data, dict) and "volatility" in data:
                                state.deribit_iv = float(data["volatility"])
                                state.deribit_iv_ts = time.time()
                                ts_ms = data.get("timestamp")
                                if ts_ms:
                                    state.deribit_ticker = {
                                        "volatility": state.deribit_iv,
                                        "timestamp": ts_ms,
                                        "index_name": data.get("index_name", ""),
                                    }
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
        except Exception as e:
            state.deribit_connected = False
            print(f"  [Deribit] {e}")
        await asyncio.sleep(2)


# ── Main runner ───────────────────────────────────────────────────────────────

async def run_triple_streams(
    config: TripleStreamConfig,
    state: Optional[TripleStreamState] = None,
) -> TripleStreamState:
    """
    Run all 3 streams concurrently. Returns the shared state.

    Usage:
        config = TripleStreamConfig(token_up="...", token_down="...")
        state = await run_triple_streams(config)
        # state.binance_book, state.last_binance_trade, state.poly_book
    """
    if state is None:
        state = TripleStreamState()

    async def run():
        tasks = [
            _binance_trade_stream(state),
            _binance_depth_stream(state),
            _poly_depth_stream(state, config),
            _deribit_stream(state, config),
        ]
        await asyncio.gather(*tasks)

    asyncio.create_task(run())
    return state


async def run_triple_streams_blocking(
    config: TripleStreamConfig,
    duration: float = 60.0,
    print_interval: float = 5.0,
    use_layer2: bool = False,
    layer2_config: Optional[object] = None,
):
    """
    Run all streams for `duration` seconds, printing status every `print_interval`.
    If use_layer2=True, also runs Layer 2 engine and prints signals.
    """
    state = TripleStreamState()
    start = time.time()
    last_print = 0.0

    async def run_tasks():
        await asyncio.gather(
            _binance_trade_stream(state),
            _binance_depth_stream(state),
            _poly_depth_stream(state, config),
            _deribit_stream(state, config),
        )

    asyncio.create_task(run_tasks())

    while time.time() - start < duration:
        await asyncio.sleep(0.5)
        now = time.time()
        if now - last_print >= print_interval:
            last_print = now
            bt = state.last_binance_trade
            bb = state.binance_book
            poly = state.poly_book
            bb_obi = bb.order_book_imbalance(5) if bb.bids or bb.asks else 0.0
            iv_str = f"IV={state.deribit_iv:.1f}" if state.deribit_iv else "IV=-"
            print(
                f"  [Layer1] T={state.binance_trade_count} "
                f"Binance: {bb.best_bid:.2f}/{bb.best_ask:.2f} OBI={bb_obi:+.2f} "
                f"Poly: {len(poly)} tokens "
                f"{iv_str} "
                f"conn: Bt={state.binance_trade_connected} Bd={state.binance_depth_connected} P={state.poly_connected} D={state.deribit_connected}"
            )
            if use_layer2 and layer2_config:
                try:
                    from layer2_engine import Layer2Engine, Layer2Config
                    from layer4_merton_jump import Layer4MertonEngine
                    l2cfg = layer2_config if isinstance(layer2_config, Layer2Config) else Layer2Config(**layer2_config)
                    engine = Layer2Engine(config=l2cfg)
                    sig = engine.evaluate(state)
                    # Layer 4: Merton jump-diffusion (second signal)
                    merton = Layer4MertonEngine()
                    sig4 = merton.evaluate(sig, deribit_iv=getattr(state, "deribit_iv", None))
                    sig.merton_p_up = sig4.p_up
                    print(
                        f"  [Layer2] fv={sig.fair_value:.3f} edge={sig.edge_net:+.3f} "
                        f"OBI={sig.obi_signal} cross={sig.cross_sum:.3f} lag={sig.oracle_lag_est_ms:.0f}ms"
                    )
                    print(
                        f"  [Layer4] Merton P(UP)={sig.merton_p_up:.3f}"
                    )
                    # Layer 5: HMM regime classifier
                    try:
                        from layer5_hmm_regime import Layer5HMMRegime
                        hmm = Layer5HMMRegime()
                        sig5 = hmm.evaluate(sig, state=state)
                        sig.hmm_regime = sig5.regime
                        sig.hmm_confidence = sig5.confidence
                        print(
                            f"  [Layer5] HMM regime={sig.hmm_regime} conf={sig.hmm_confidence:.2f} n={sig5.n_samples}"
                        )
                    except Exception as e:
                        print(f"  [Layer5] {e}")
                    # Layer 6: risk and execution
                    try:
                        from layer6_risk_execution import Layer6Engine
                        sig6 = Layer6Engine().evaluate(
                            fair_value=sig.fair_value,
                            poly_mid=sig.poly_mid,
                            regime=sig.hmm_regime or "medium_vol",
                            capital=100,
                        )
                        print(
                            f"  [Layer6] edge={sig6.edge:+.3f} trade={sig6.trade} "
                            f"side={sig6.side} size=${sig6.size:.1f} exec={sig6.execution}"
                        )
                    except Exception as e:
                        print(f"  [Layer6] {e}")
                except Exception as e:
                    print(f"  [Layer2] {e}")

    state.running = False


GAMMA_API = "https://gamma-api.polymarket.com"
WINDOW_S = 300


def _parse_clob_tokens(raw) -> list:
    """Parse clobTokenIds (JSON string or list) into token IDs."""
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    if isinstance(raw, str):
        s = raw.strip().strip("[]").replace('"', '').replace("'", "")
        return [t.strip() for t in s.split(",") if t.strip()]
    return []


def _map_outcomes_to_tokens(outcomes, tokens: list) -> Tuple[str, str]:
    """Map tokens to UP/DOWN using outcomes. Default: tokens[0]=UP, tokens[1]=DOWN."""
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


async def _discover_btc_market() -> Tuple[str, str]:
    """Discover active BTC UP/DOWN market from Gamma API. Returns (token_up, token_down)."""
    import time as _time
    now = int(_time.time())
    window_start = (now // WINDOW_S) * WINDOW_S

    connector = aiohttp.TCPConnector(ssl=SSL_CTX)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Try BTC 5-min events (slug: btc-updown-5m-{window})
        for offset in range(6):
            ws = window_start - offset * WINDOW_S
            slug = f"btc-updown-5m-{ws}"
            try:
                async with session.get(
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
                            return _map_outcomes_to_tokens(m.get("outcomes", ""), tokens)
            except Exception:
                pass

        # 2. Fallback: markets API with tokens or clobTokenIds
        for url in [
            f"{GAMMA_API}/markets?active=true&closed=false&limit=100",
            f"{GAMMA_API}/markets?active=true&tag=crypto&limit=50",
        ]:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
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
                                return (up_tok, dn_tok)
            except Exception:
                pass
    return ("", "")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30, help="Run for N seconds")
    parser.add_argument("--token-up", type=str, default="", help="Polymarket UP token ID")
    parser.add_argument("--token-down", type=str, default="", help="Polymarket DOWN token ID")
    parser.add_argument("--discover", action="store_true", help="Auto-discover BTC market tokens from Gamma API")
    parser.add_argument("--deribit-index", type=str, default="btc_usd", help="Deribit vol index (btc_usd, eth_usd)")
    parser.add_argument("--layer2", action="store_true", help="Run Layer 2 engine (fair value, OBI, cross-arb, oracle lag)")
    parser.add_argument("--strike", type=float, default=0, help="Strike for fair value (0=use BTC as strike)")
    args = parser.parse_args()

    token_up = args.token_up or ""
    token_down = args.token_down or ""
    if args.discover:
        print("  Discovering BTC market...")
        token_up, token_down = asyncio.run(_discover_btc_market())
        if token_up and token_down:
            print(f"  Found: UP={token_up[:20]}... DOWN={token_down[:20]}...")
        else:
            print("  Could not discover BTC market. Run without --discover and pass --token-up/--token-down.")
            exit(1)

    config = TripleStreamConfig(
        token_up=token_up,
        token_down=token_down,
        deribit_vol_index=args.deribit_index or "btc_usd",
    )
    layer2_config = None
    if args.layer2:
        from layer2_engine import Layer2Config
        WINDOW_S = 300
        t_rem = WINDOW_S - (time.time() % WINDOW_S)
        layer2_config = Layer2Config(
            strike=args.strike,
            time_remaining=t_rem,
            token_up=token_up,
            token_down=token_down,
        )
    asyncio.run(run_triple_streams_blocking(config, duration=args.duration, use_layer2=args.layer2, layer2_config=layer2_config))
