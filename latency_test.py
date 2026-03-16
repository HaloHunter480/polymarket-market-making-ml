"""
Latency Test for Polymarket + Binance
======================================

Measures round-trip times to all endpoints we'll use in live trading.
No trading, no credentials required - just raw speed measurement.

Run: python3 latency_test.py
"""

import asyncio
import aiohttp
import websockets
import ssl
import time
import json
import statistics

CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
BINANCE_REST = "https://api.binance.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

N_PINGS = 10


async def measure_http(session: aiohttp.ClientSession, name: str, url: str, params: dict = None) -> list:
    """Measure HTTP round-trip latency."""
    latencies = []
    
    for i in range(N_PINGS):
        start = time.perf_counter()
        try:
            async with session.get(url, params=params, ssl=SSL_CTX) as resp:
                await resp.read()
                elapsed = (time.perf_counter() - start) * 1000
                status = resp.status
        except Exception as e:
            elapsed = -1
            status = f"ERR: {e}"
        
        latencies.append(elapsed)
        if i == 0:
            print(f"    {name}: {elapsed:.0f}ms (status: {status})")
    
    return latencies


async def measure_ws_binance() -> list:
    """Measure Binance WebSocket message latency."""
    latencies = []
    
    print(f"\n  [4] Binance WebSocket (btcusdt@trade)")
    print(f"      Connecting...")
    
    try:
        connect_start = time.perf_counter()
        async with websockets.connect(BINANCE_WS, ssl=SSL_CTX) as ws:
            connect_time = (time.perf_counter() - connect_start) * 1000
            print(f"      Connected in {connect_time:.0f}ms")
            print(f"      Receiving {N_PINGS} trade messages...")
            
            for i in range(N_PINGS):
                
                
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                recv_time = time.time()

                data = json.loads(msg)

    # Binance trade event time (ms since epoch)
                exchange_time = data.get("T", 0) / 1000

                if exchange_time == 0:
                    
                   
                 
                    continue

                lag_ms = (recv_time - exchange_time) * 1000
                latencies.append(lag_ms)

                price = float(data.get("p", 0))

                if i == 0:
                    
                    
                   
                   
                    print(f"      First trade lag: {lag_ms:.2f} ms (BTC ${price:,.2f})")
            latencies.insert(0, connect_time)
    except Exception as e:
        print(f"      ERROR: {e}")
        latencies = [-1]
    
    return latencies


async def measure_polymarket_book() -> list:
    """Measure fetching a real Polymarket order book."""
    latencies = []
    
    print(f"\n  [5] Polymarket Order Book (live market lookup)")
    
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=SSL_CTX),
        timeout=aiohttp.ClientTimeout(total=15),
    ) as session:
        # Find a live 5-min BTC market
        print(f"      Finding active BTC 5-min market...")
        
        search_start = time.perf_counter()
        try:
            async with session.get(
                f"{GAMMA_API}/events",
                params={"slug": "btc-5-minute", "active": "true", "limit": "5"},
            ) as resp:
                events = await resp.json()
            search_time = (time.perf_counter() - search_start) * 1000
            print(f"      Gamma API search: {search_time:.0f}ms")
            latencies.append(search_time)
        except Exception as e:
            print(f"      Gamma search failed: {e}")
            # Try alternative search
            try:
                async with session.get(
                    f"{GAMMA_API}/events",
                    params={"tag": "btc-5-minute", "active": "true", "limit": "5"},
                ) as resp:
                    events = await resp.json()
            except:
                events = []
        
        # Get token IDs from markets
        token_id = None
        if events:
            for event in events:
                markets = event.get("markets", [])
                for market in markets:
                    tokens = market.get("clobTokenIds")
                    if tokens:
                        token_id = tokens[0] if isinstance(tokens, list) else tokens.split(",")[0]
                        print(f"      Found market: {market.get('question', 'unknown')[:60]}")
                        break
                if token_id:
                    break
        
        if token_id:
            for i in range(N_PINGS):
                start = time.perf_counter()
                try:
                    async with session.get(
                        f"{CLOB_API}/book",
                        params={"token_id": token_id},
                    ) as resp:
                        data = await resp.json()
                        elapsed = (time.perf_counter() - start) * 1000
                        
                        if i == 0:
                            bids = data.get("bids", [])
                            asks = data.get("asks", [])
                            best_bid = float(bids[0]["price"]) if bids else 0
                            best_ask = float(asks[0]["price"]) if asks else 0
                            print(f"      Book: Bid ${best_bid:.3f} / Ask ${best_ask:.3f} ({elapsed:.0f}ms)")
                except Exception as e:
                    elapsed = -1
                    if i == 0:
                        print(f"      Book fetch error: {e}")
                
                latencies.append(elapsed)
        else:
            print(f"      No active BTC market found, testing generic endpoint...")
            for i in range(N_PINGS):
                start = time.perf_counter()
                try:
                    async with session.get(f"{CLOB_API}/time") as resp:
                        await resp.read()
                        elapsed = (time.perf_counter() - start) * 1000
                except:
                    elapsed = -1
                latencies.append(elapsed)
    
    return latencies


def print_stats(name: str, latencies: list):
    """Print latency statistics."""
    valid = [x for x in latencies if x > 0]
    if not valid:
        print(f"    {name}: ALL FAILED")
        return
    
    avg = statistics.mean(valid)
    med = statistics.median(valid)
    mn = min(valid)
    mx = max(valid)
    p95 = sorted(valid)[int(len(valid) * 0.95)] if len(valid) >= 5 else mx
    
    print(f"    {name:30s} | avg: {avg:6.0f}ms | med: {med:6.0f}ms | min: {mn:6.0f}ms | max: {mx:6.0f}ms | p95: {p95:6.0f}ms")


async def main():
    print("=" * 80)
    print("  LATENCY TEST - Polymarket + Binance")
    print("=" * 80)
    print(f"  Pings per endpoint: {N_PINGS}")
    print(f"  Location: Your machine (not server)")
    print()
    
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=SSL_CTX),
        timeout=aiohttp.ClientTimeout(total=15),
    ) as session:
        # 1. Polymarket CLOB API
        print(f"  [1] Polymarket CLOB API ({CLOB_API})")
        clob_lat = await measure_http(session, "CLOB /time", f"{CLOB_API}/time")
        
        # 2. Gamma API
        print(f"\n  [2] Gamma API ({GAMMA_API})")
        gamma_lat = await measure_http(session, "Gamma /events", f"{GAMMA_API}/events", {"limit": "1"})
        
        # 3. Binance REST
        print(f"\n  [3] Binance REST ({BINANCE_REST})")
        binance_lat = await measure_http(session, "Binance /ticker", f"{BINANCE_REST}/api/v3/ticker/price", {"symbol": "BTCUSDT"})
    
    # 4. Binance WebSocket
    ws_lat = await measure_ws_binance()
    
    # 5. Polymarket Order Book
    book_lat = await measure_polymarket_book()
    
    # Summary
    print()
    print("=" * 80)
    print("  LATENCY SUMMARY")
    print("=" * 80)
    print()
    print_stats("Polymarket CLOB API", clob_lat)
    print_stats("Gamma API", gamma_lat)
    print_stats("Binance REST", binance_lat)
    print_stats("Binance WebSocket", ws_lat)
    print_stats("Polymarket Order Book", book_lat)
    
    print()
    
    # Calculate end-to-end estimate
    clob_valid = [x for x in clob_lat if x > 0]
    binance_valid = [x for x in binance_lat if x > 0]
    
    if clob_valid and binance_valid:
        clob_avg = statistics.mean(clob_valid)
        binance_avg = statistics.mean(binance_valid)
        
        # Signal detection (Binance) + Order submission (Polymarket) = End-to-end
        e2e = binance_avg + clob_avg
        
        print("=" * 80)
        print("  ESTIMATED END-TO-END LATENCY")
        print("=" * 80)
        print(f"    Signal detection (Binance):   ~{binance_avg:.0f}ms")
        print(f"    Order submission (Polymarket): ~{clob_avg:.0f}ms")
        print(f"    ─────────────────────────────────────")
        print(f"    Total round-trip:              ~{e2e:.0f}ms")
        print()
        
        if e2e < 200:
            print(f"    ✅ EXCELLENT - Sub-200ms is competitive for Polymarket")
        elif e2e < 500:
            print(f"    ✅ GOOD - Sub-500ms is workable for 5-min windows")
        elif e2e < 1000:
            print(f"    ⚠️  FAIR - Sub-1s, usable but not ideal")
        else:
            print(f"    ❌ SLOW - >1s round-trip, consider a server closer to exchanges")
        
        print()
        print("  NEXT STEPS:")
        print("  1. Set up Polymarket API keys (deposit USDC on polymarket.com)")
        print("  2. Configure .env file with credentials")
        print("  3. Run $2 live test")


if __name__ == "__main__":
    asyncio.run(main())
