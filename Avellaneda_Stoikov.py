# avellaneda_stoikov.py
# The A-S model solves for optimal bid/ask quotes given current inventory.
# Mathematical foundation: stochastic control theory (HJB equation).
#
# Key insight: a market maker holding 0 BTC quotes symmetrically.
# Holding long BTC → push ask down (eager to sell), push bid up (not eager to buy more).
# Holding short BTC → opposite.
# The model calculates EXACTLY how much to skew by.

import numpy as np

class AvellanedaStoikov:
    """
    Optimal Market Making under the Avellaneda-Stoikov framework.
    
    The model gives you:
    1. Reservation price r(t): your risk-adjusted mid, shifted by inventory
    2. Optimal spread δ: the total spread to quote around r(t)
    
    These together give bid = r - δ/2, ask = r + δ/2.
    """
    
    def __init__(self, 
                 gamma: float = 0.1,    # Risk aversion (higher = more cautious)
                 sigma: float = 100.0,  # BTC price volatility ($/second)
                 k: float = 1.5,        # Order arrival rate sensitivity
                 T: float = 3600.0):    # Trading session length in seconds
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.T = T  # Total session time
        self.t = 0  # Current time into session
        
        self.inventory = 0.0     # Current BTC inventory
        self.max_inventory = 2.0 # Hard cap: never hold more than 2 BTC
    
    def update_vol(self, sigma_new: float):
        """
        Update volatility estimate dynamically. 
        This is why you run the Kalman filter — feed its output here.
        A-S with stale volatility is dangerous during regime changes.
        """
        self.sigma = sigma_new
    
    def reservation_price(self, mid_price: float) -> float:
        """
        The reservation price is the mid you'd quote if you had zero spread.
        It's different from the market mid when you have inventory because
        you're now personally exposed to price moves.
        
        r = S - q·γ·σ²·(T-t)
        
        Where q is inventory. The formula says: if you're long (q > 0),
        your personal valuation of BTC is LOWER than market price because
        you're already exposed and a further drop hurts you.
        
        This is not psychological — it's mathematically derived from
        the utility function of a risk-averse trader.
        """
        time_remaining = max(self.T - self.t, 0.01)
        return mid_price - self.inventory * self.gamma * (self.sigma**2) * time_remaining
    
    def optimal_spread(self) -> float:
        """
        δ* = γσ²(T-t) + (2/γ)·ln(1 + γ/k)
        
        The spread has two components:
        1. γσ²(T-t): risk premium for carrying inventory. Grows with vol and
           remaining time (more time = more chance of being caught offside)
        2. (2/γ)·ln(1 + γ/k): base spread from order arrival dynamics.
           This is the minimum spread needed to be profitable given how
           fast orders arrive at your quotes.
        """
        time_remaining = max(self.T - self.t, 0.01)
        inventory_risk_spread = self.gamma * (self.sigma**2) * time_remaining
        arrival_rate_spread = (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return inventory_risk_spread + arrival_rate_spread
    
    def optimal_quotes(self, mid_price: float) -> dict:
        """
        The final output: your optimal bid and ask prices right now.
        """
        r = self.reservation_price(mid_price)
        half_spread = self.optimal_spread() / 2
        
        bid = r - half_spread
        ask = r + half_spread
        
        # Inventory guard: if at inventory limits, collapse one side
        # so you can only trade in the direction that reduces risk
        if self.inventory >= self.max_inventory:
            bid = 0  # Don't take more BTC when you're already full long
        if self.inventory <= -self.max_inventory:
            ask = float('inf')  # Don't take more short
        
        return {
            'bid': bid,
            'ask': ask,
            'reservation_price': r,
            'spread': ask - bid,
            'inventory_skew': r - mid_price,  # How far we've shifted from market
        }
    
    def update(self, filled_bid: bool = False, filled_ask: bool = False,
               quantity: float = 0.001, dt: float = 1.0):
        """Update inventory and time after fills."""
        self.t += dt
        if filled_bid:
            self.inventory += quantity   # Bought, now longer
        if filled_ask:
            self.inventory -= quantity   # Sold, now shorter
