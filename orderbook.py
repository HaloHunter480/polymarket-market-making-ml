# orderbook.py
# This is arguably the most important file.
# The order book tells you WHERE the market wants to go
# before price actually moves. Think of it as X-ray vision.

import time
import numpy as np
from collections import deque


class OrderBook:
    """
    Maintains a real-time L2 order book and computes
    microstructure signals that HFT firms live and die by.
    """
    
    def __init__(self, depth=20):
        self.bids = {}  # price -> size
        self.asks = {}  # price -> size
        self.depth = depth
        self.last_update_id = 0
        
        # Rolling CVD (Cumulative Volume Delta)
        # CVD = sum of buy volume - sum of sell volume
        # Rising CVD + flat price = about to pump
        # Falling CVD + flat price = about to dump
        self.cvd = 0.0
        self.cvd_history = deque(maxlen=500)
        
        # Track last 1000 trades for micro-analysis
        self.recent_trades = deque(maxlen=1000)
        
    def update(self, data: dict):
        """Process Binance depth update (partial book, 20 levels)."""
        for bid in data.get('bids', []):
            price, size = float(bid[0]), float(bid[1])
            if size == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size
                
        for ask in data.get('asks', []):
            price, size = float(ask[0]), float(ask[1])
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size
    
    def add_trade(self, price: float, size: float, is_buyer_maker: bool):
        """
        is_buyer_maker = True means a SELL order hit the book (sell aggressor).
        is_buyer_maker = False means a BUY order hit the book (buy aggressor).
        
        This is the heart of order flow analysis. Every trade tells
        you who is WILLING TO PAY to get filled immediately.
        """
        direction = -1 if is_buyer_maker else 1
        self.cvd += direction * size
        self.cvd_history.append(self.cvd)
        self.recent_trades.append({
            'price': price, 'size': size,
            'direction': direction,
            'ts': time.time()
        })
    
    @property
    def best_bid(self) -> float:
        return max(self.bids.keys()) if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return min(self.asks.keys()) if self.asks else float('inf')
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid
    
    @property
    def spread_ratio(self) -> float:
        """Spread as % of mid price. Key liquidity gauge."""
        if self.mid_price == 0:
            return 1.0
        return self.spread / self.mid_price
    
    def order_book_imbalance(self, levels=5) -> float:
        """
        OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
        Range: -1 to +1
        OBI > 0.3  → more buying pressure, likely up move
        OBI < -0.3 → more selling pressure, likely down move
        
        This is one of the most predictive short-term signals.
        Empirically shown to predict next-tick direction at ~58% accuracy
        on its own in crypto markets.
        """
        sorted_bids = sorted(self.bids.keys(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.keys())[:levels]
        
        bid_vol = sum(self.bids.get(p, 0) for p in sorted_bids)
        ask_vol = sum(self.asks.get(p, 0) for p in sorted_asks)
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total
    
    def depth_ratio(self, levels=10) -> float:
        """
        Ratio of total bid depth to ask depth across N levels.
        > 1.0 means more liquidity on the buy side (bullish lean).
        """
        sorted_bids = sorted(self.bids.keys(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.keys())[:levels]
        
        bid_depth = sum(self.bids.get(p, 0) for p in sorted_bids)
        ask_depth = sum(self.asks.get(p, 0) for p in sorted_asks)
        
        return bid_depth / ask_depth if ask_depth > 0 else 1.0
    
    def buy_sell_ratio_recent(self, seconds=30) -> float:
        """
        Among trades in the last N seconds, what fraction was buy-side?
        Above 0.55 is meaningfully bullish on short timeframes.
        """
        now = time.time()
        recent = [t for t in self.recent_trades if now - t['ts'] < seconds]
        if not recent:
            return 0.5
        
        buy_vol = sum(t['size'] for t in recent if t['direction'] == 1)
        total_vol = sum(t['size'] for t in recent)
        return buy_vol / total_vol if total_vol > 0 else 0.5
    
    def vwap_recent(self, seconds=60) -> float:
        """Volume-weighted average price of recent trades."""
        now = time.time()
        recent = [t for t in self.recent_trades if now - t['ts'] < seconds]
        if not recent:
            return self.mid_price
        
        total_pv = sum(t['price'] * t['size'] for t in recent)
        total_v = sum(t['size'] for t in recent)
        return total_pv / total_v if total_v > 0 else self.mid_price
    
    def cvd_momentum(self, window=50) -> float:
        """
        Rate of change of CVD over last N samples.
        Accelerating CVD is a leading indicator of price movement.
        """
        if len(self.cvd_history) < window:
            return 0.0
        recent = list(self.cvd_history)[-window:]
        # Simple linear regression slope normalized by std
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        std = np.std(recent)
        return slope / std if std > 0 else 0.0
    
    def large_order_detection(self, threshold_multiplier=3.0) -> dict:
        """
        Detect unusually large orders sitting in the book.
        A 'whale wall' on the ask side is strong resistance.
        A 'whale bid' on the bid side is strong support.
        Returns location and strength of detected walls.
        """
        if not self.bids or not self.asks:
            return {'bid_wall': None, 'ask_wall': None}
        
        all_bid_sizes = list(self.bids.values())
        all_ask_sizes = list(self.asks.values())
        
        mean_bid = np.mean(all_bid_sizes)
        mean_ask = np.mean(all_ask_sizes)
        
        bid_wall = None
        ask_wall = None
        
        for price, size in self.bids.items():
            if size > threshold_multiplier * mean_bid:
                if bid_wall is None or size > bid_wall['size']:
                    bid_wall = {'price': price, 'size': size}
        
        for price, size in self.asks.items():
            if size > threshold_multiplier * mean_ask:
                if ask_wall is None or size > ask_wall['size']:
                    ask_wall = {'price': price, 'size': size}
        
        return {'bid_wall': bid_wall, 'ask_wall': ask_wall}
