# vpin.py
# Volume-synchronized Probability of Informed Trading.
# This is your "danger detector" — when VPIN > 0.7, your HFT edge
# evaporates because you're trading against informed players.
# When VPIN < 0.3, you're in noise trader territory — max aggression.

import numpy as np
from collections import deque

class VPIN:
    """
    VPIN measures order flow imbalance in volume-time (not clock-time).
    This is important: during fast markets, volume-time speeds up.
    Regular time-based measures miss the toxicity spike.
    
    Algorithm:
    1. Divide total volume into equal buckets (e.g., each = 1000 BTC)
    2. For each bucket, classify volume as buy-initiated or sell-initiated
       (using bulk volume classification since we lack individual tick data)
    3. VPIN = average |buy_vol - sell_vol| / total_vol across last N buckets
    
    High VPIN = order flow is one-sided = informed trading = danger.
    Low VPIN  = order flow is balanced = uninformed noise = opportunity.
    """
    
    def __init__(self, bucket_size: float = 500.0, window: int = 50):
        self.bucket_size = bucket_size  # Volume per bucket in BTC
        self.window = window            # Number of buckets in VPIN window
        
        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0
        self.current_bucket_total = 0.0
        
        # Rolling history of completed buckets
        self.bucket_imbalances = deque(maxlen=window)
        
        # For bulk volume classification, we need recent price bar data
        self.price_bar_closes = deque(maxlen=100)
    
    def bulk_classify(self, open_p: float, close_p: float, 
                       volume: float) -> tuple:
        """
        Bulk Volume Classification (Easley et al.):
        For a price bar, estimate buy vs sell volume using the
        fraction of the bar that was an up-move.
        
        Z = (close - open) / σ    (normalized price change)
        buy_fraction = Φ(Z)       (standard normal CDF)
        
        This is better than just "up bar = all buys" because it
        accounts for the magnitude of the move relative to volatility.
        """
        if not self.price_bar_closes or len(self.price_bar_closes) < 2:
            buy_frac = 0.5
        else:
            prices = list(self.price_bar_closes)
            sigma = np.std(np.diff(prices))
            if sigma > 0:
                z = (close_p - open_p) / sigma
                buy_frac = float(np.clip(0.5 + z / (2 * np.sqrt(2 * np.pi) * sigma + 1e-8), 0, 1))
            else:
                buy_frac = 0.5 if close_p >= open_p else 0.5
        
        return volume * buy_frac, volume * (1 - buy_frac)
    
    def update(self, open_p: float, close_p: float, volume: float):
        """Feed in each price bar and VPIN updates automatically."""
        self.price_bar_closes.append(close_p)
        buy_vol, sell_vol = self.bulk_classify(open_p, close_p, volume)
        
        remaining_volume = volume
        remaining_buy = buy_vol
        remaining_sell = sell_vol
        
        # Fill current bucket, potentially completing multiple buckets
        while remaining_volume > 0:
            space = self.bucket_size - self.current_bucket_total
            fill = min(space, remaining_volume)
            
            frac = fill / max(volume, 1e-8)
            self.current_bucket_buy  += remaining_buy  * frac
            self.current_bucket_sell += remaining_sell * frac
            self.current_bucket_total += fill
            remaining_volume -= fill
            remaining_buy  *= (1 - frac)
            remaining_sell *= (1 - frac)
            
            # Bucket complete — record imbalance and start fresh
            if self.current_bucket_total >= self.bucket_size:
                imbalance = abs(self.current_bucket_buy - self.current_bucket_sell) / self.bucket_size
                self.bucket_imbalances.append(imbalance)
                self.current_bucket_buy = 0.0
                self.current_bucket_sell = 0.0
                self.current_bucket_total = 0.0
    
    @property
    def vpin(self) -> float:
        """Current VPIN value. Range 0-1."""
        if len(self.bucket_imbalances) < 5:
            return 0.5  # Default: neutral
        return float(np.mean(list(self.bucket_imbalances)))
    
    @property
    def market_quality(self) -> dict:
        """
        Translate VPIN into actionable trading guidance.
        This is the output your main bot should consume.
        """
        v = self.vpin
        if v < 0.25:
            regime = "CLEAN"
            action = "Full aggression — noise traders dominate, exploit freely"
        elif v < 0.45:
            regime = "NORMAL"
            action = "Normal trading — standard position sizes"
        elif v < 0.65:
            regime = "ELEVATED"
            action = "Reduce size by 50% — informed flow increasing"
        else:
            regime = "TOXIC"
            action = "STOP trading — informed traders active, adverse selection risk extreme"
        
        return {'vpin': v, 'regime': regime, 'action': action}
