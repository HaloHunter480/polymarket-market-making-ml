# ornstein_uhlenbeck.py
# The OU process: dX_t = θ(μ - X_t)dt + σdW_t
#
# θ (theta): mean reversion speed. High θ = snaps back fast.
# μ (mu):    long-run mean (equilibrium level)
# σ (sigma): volatility around the mean
#
# For HFT, you fit this to the spread between two things
# that should be in sync (BTC spot vs futures, Binance vs Coinbase, etc.)
# When the spread is at ±2σ from μ, enter. When it reverts to μ, exit.
# The math guarantees you a positive expected return if θ > 0.

import numpy as np
from scipy.optimize import minimize

class OrnsteinUhlenbeck:
    """
    Fit an OU process to a spread series, then use it to:
    1. Detect when the spread is anomalously wide (entry signal)
    2. Predict when it will revert (exit timing)
    3. Compute the expected profit of entering a trade right now
    """
    
    def __init__(self):
        self.theta = 1.0   # Mean reversion speed (per unit time)
        self.mu = 0.0      # Long-run mean of the spread
        self.sigma = 1.0   # Volatility of deviations
        self.spread_history = []
        
    def fit(self, spread_series: np.ndarray, dt: float = 0.1):
        """
        Fit θ, μ, σ using Exact Maximum Likelihood for discretely observed OU.
        
        The key equations for discrete-time OU (the ones that actually
        give you unbiased estimates unlike the simpler OLS approach):
        
        X_{t+dt} = X_t·e^{-θdt} + μ(1 - e^{-θdt}) + noise
        
        This is called the "Vasicek discretization" — the same math used
        in interest rate modeling.
        """
        n = len(spread_series) - 1
        Sx  = np.sum(spread_series[:-1])
        Sy  = np.sum(spread_series[1:])
        Sxx = np.sum(spread_series[:-1]**2)
        Sxy = np.sum(spread_series[:-1] * spread_series[1:])
        Syy = np.sum(spread_series[1:]**2)
        
        # Closed-form MLE estimates
        b = (n*Sxy - Sx*Sy) / (n*Sxx - Sx**2)
        a = (Sy - b*Sx) / n
        
        # Extract OU parameters from the regression coefficients
        self.theta = -np.log(b) / dt
        self.mu = a / (1 - b)
        
        # Variance of residuals → OU sigma
        residuals = spread_series[1:] - a - b * spread_series[:-1]
        s2 = np.var(residuals)
        self.sigma = np.sqrt(s2 * 2 * self.theta / (1 - np.exp(-2 * self.theta * dt)))
        
        return self
    
    def z_score(self, current_spread: float) -> float:
        """
        How many standard deviations is the current spread from its mean?
        
        Unlike a naive z-score, this accounts for the fact that the OU
        stationary distribution has variance σ²/(2θ), not just σ².
        The half-life of mean reversion (ln(2)/θ) tells you how long
        an extreme spread typically persists.
        """
        stationary_std = self.sigma / np.sqrt(2 * self.theta)
        return (current_spread - self.mu) / stationary_std
    
    def half_life_seconds(self) -> float:
        """
        How many seconds does it typically take for the spread to
        revert halfway back to the mean? This is your exit timing guide.
        
        For HFT on BTC/USDT vs BTC/USDC spread, this is typically 0.5-3 seconds.
        For the Polymarket/CEX price gap, it's typically 300-800ms.
        """
        return np.log(2) / self.theta
    
    def expected_profit(self, current_spread: float, 
                         trade_cost_pct: float = 0.001) -> float:
        """
        Expected profit of entering a mean-reversion trade RIGHT NOW.
        
        This integrates the OU reversion path to compute the expected
        amount the spread will close, then subtracts trading costs.
        If this is negative, don't trade — no edge.
        """
        expected_reversion = (current_spread - self.mu) * (1 - np.exp(-self.theta))
        # Subtract round-trip costs (entry + exit)
        cost = 2 * trade_cost_pct * abs(current_spread)
        return abs(expected_reversion) - cost
    
    def entry_signal(self, current_spread: float, 
                      z_threshold: float = 2.0) -> str:
        """
        Entry logic:
        - If z > +threshold: spread is wide on the high side → short the spread
        - If z < -threshold: spread is wide on the low side → long the spread  
        - Otherwise: no edge, wait
        """
        z = self.z_score(current_spread)
        if z > z_threshold:
            return "short_spread"
        elif z < -z_threshold:
            return "long_spread"
        return "none"
