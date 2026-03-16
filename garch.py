# garch.py
# GARCH(1,1) volatility model.
# The "1,1" means we use 1 lag of past returns² and 1 lag of past variance.
# Remarkably, GARCH(1,1) beats more complex models most of the time.

import numpy as np
from scipy.optimize import minimize
from collections import deque

class GARCH11:
    """
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    
    Where:
      ω (omega): long-run variance floor (baseline volatility)
      α (alpha): ARCH term — how much last period's shock matters
      β (beta):  GARCH term — how much last period's variance persists
    
    Constraint: α + β < 1 (ensures variance doesn't explode to infinity)
    Typical crypto values: α ≈ 0.08, β ≈ 0.88, meaning shocks are absorbed
    slowly (88% of yesterday's variance carries into today).
    """
    
    def __init__(self):
        # Conservative starting values
        self.omega = 1e-6
        self.alpha = 0.08
        self.beta  = 0.88
        
        self.h = None   # Current conditional variance
        self.returns = deque(maxlen=2000)
    
    def fit(self, return_series: np.ndarray):
        """
        Fit GARCH(1,1) via MLE using a vectorised recursive filter.
        Numba-style loop replaced by a Python loop on a capped sample
        (2000 obs) for speed while keeping statistical accuracy.
        """
        rets = np.asarray(return_series, dtype=float)
        rets = rets[np.isfinite(rets)]
        # Cap at 2000 most-recent observations: statistically sufficient,
        # avoids slow MLE on 26K+ samples.
        if len(rets) > 2000:
            rets = rets[-2000:]
        if len(rets) < 20:
            return

        h0 = float(np.var(rets))

        def neg_ll(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1.0:
                return 1e10
            h = h0
            ll = 0.0
            for r in rets:
                h = omega + alpha * r * r + beta * h
                if h <= 0:
                    return 1e10
                ll += 0.5 * (np.log(h) + r * r / h)
            return ll   # minimising, so omit constants

        result = minimize(
            neg_ll,
            [self.omega, self.alpha, self.beta],
            method="L-BFGS-B",
            bounds=[(1e-10, None), (1e-6, 0.499), (1e-6, 0.989)],
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if result.success and result.x[1] + result.x[2] < 1.0:
            self.omega, self.alpha, self.beta = result.x
    
    def update(self, new_return: float):
        """Update conditional variance with each new return."""
        self.returns.append(new_return)
        if self.h is None:
            self.h = np.var(list(self.returns)) if len(self.returns) > 5 else 1e-6
        else:
            self.h = self.omega + self.alpha * new_return**2 + self.beta * self.h
    
    def forecast(self, steps_ahead: int = 5) -> np.ndarray:
        """
        Forecast conditional variance for the next N steps.
        The key formula for multi-step GARCH forecast is:
        h_{t+k} = ω·[1 + (α+β) + ... + (α+β)^{k-1}] + (α+β)^k · h_t
        
        This converges to the long-run variance ω/(1-α-β) as k→∞.
        That convergence speed tells you how long current vol conditions persist.
        """
        if self.h is None:
            return np.ones(steps_ahead) * 1e-4
        
        forecasts = np.zeros(steps_ahead)
        long_run_var = self.omega / (1 - self.alpha - self.beta)
        persist = self.alpha + self.beta
        
        for k in range(1, steps_ahead + 1):
            forecasts[k-1] = long_run_var + (persist**k) * (self.h - long_run_var)
        
        return np.sqrt(forecasts)  # Return volatility, not variance
    
    @property
    def vol_regime(self) -> str:
        """Use forecasted vol to classify the regime for position sizing."""
        if self.h is None:
            return "UNKNOWN"
        current_vol = np.sqrt(self.h * 288)  # Annualized, assuming 5-min bars
        if current_vol < 0.30:
            return "LOW_VOL"     # Scale position UP (mean reversion works well)
        elif current_vol < 0.60:
            return "NORMAL_VOL"  # Standard sizing
        elif current_vol < 1.0:
            return "HIGH_VOL"    # Scale position DOWN
        else:
            return "EXTREME_VOL" # Reduce to minimum or stop
