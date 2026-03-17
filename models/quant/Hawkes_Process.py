# hawkes_process.py
# The Hawkes Process models the "intensity" (arrival rate) of trades.
# High intensity = market is excited, likely informed flow → trade.
# Low intensity = random noise, wide spread likely → sit out.

import numpy as np
from scipy.optimize import minimize

class HawkesProcess:
    """
    The Hawkes Process intensity function is:
    λ(t) = μ + Σ α·exp(-β·(t - tᵢ))   for all past events tᵢ < t
    
    Where:
      μ (mu)   = baseline arrival rate (trades per second even in quiet market)
      α (alpha)= how much each trade excites future arrivals (jump size)
      β (beta) = how fast the excitement decays (memory decay)
    
    The ratio α/β is the "branching ratio" — the expected number of
    offspring trades each trade generates. If α/β > 1, the process
    explodes (flash crash territory). Healthy markets have α/β ≈ 0.5-0.8.
    """
    
    def __init__(self):
        # These get estimated from live data using MLE
        self.mu = 1.0    # baseline: 1 trade/second
        self.alpha = 0.5 # each trade causes 50% of another
        self.beta = 1.5  # excitement decays with half-life of ~0.7 seconds
        
        self.event_times = []  # history of trade timestamps
        self.current_intensity = self.mu
    
    def add_event(self, timestamp: float):
        """Record a new trade and update the running intensity."""
        self.event_times.append(timestamp)
        # Trim to last 500 events to keep computation fast
        if len(self.event_times) > 500:
            self.event_times = self.event_times[-500:]
        self.current_intensity = self._compute_intensity(timestamp)
    
    def _compute_intensity(self, t: float) -> float:
        """
        λ(t) = μ + α·Σ exp(-β·(t - tᵢ))
        The sum decays exponentially for older events,
        so recent trades matter far more than old ones.
        """
        intensity = self.mu
        for ti in self.event_times:
            if ti < t:
                intensity += self.alpha * np.exp(-self.beta * (t - ti))
        return intensity
    
    def fit(self, event_times: np.ndarray):
        """
        Fit parameters via vectorised MLE.  O(n) per likelihood eval
        using the recursive compensator trick — replaces the O(n²) loop.

        Ogata (1981) recursive formula:
          R_i = Σ_{j<i} exp(-β(t_i - t_j))  =  exp(-β·Δt_i) · (1 + R_{i-1})
        This makes log-likelihood O(n) instead of O(n²).
        """
        evt = np.asarray(event_times, dtype=float)
        if len(evt) < 10:
            return

        # Use at most 500 events for speed (still statistically robust)
        if len(evt) > 500:
            evt = evt[-500:]

        T   = evt[-1] - evt[0]
        dt  = np.diff(evt)          # inter-arrival times

        def neg_ll(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
                return 1e10

            # Recursive R: R_i = exp(-β·Δt) · (1 + R_{i-1}), R_0 = 0
            R    = np.zeros(len(evt))
            for i in range(1, len(evt)):
                R[i] = np.exp(-beta * dt[i - 1]) * (1.0 + R[i - 1])

            lam  = mu + alpha * R          # intensity at each event
            lam  = np.maximum(lam, 1e-12)

            # Log-likelihood: sum(log λ(t_i)) - ∫λ dt
            log_sum      = np.sum(np.log(lam))
            compensator  = mu * T + (alpha / beta) * np.sum(
                               1.0 - np.exp(-beta * (T - (evt - evt[0]))))
            return -(log_sum - compensator)

        result = minimize(
            neg_ll, [self.mu, self.alpha, self.beta],
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, 0.99), (1e-6, None)],
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if result.success and result.x[1] < result.x[2]:
            self.mu, self.alpha, self.beta = result.x
    
    @property
    def regime(self) -> str:
        """
        Classify the current market regime based on intensity.
        This is your 'when to trade' filter — only trade in
        ACTIVE regimes where your edge is actually present.
        """
        baseline = self.mu * 3  # 3x baseline = active
        if self.current_intensity < self.mu * 1.5:
            return "QUIET"       # Random noise, don't trade
        elif self.current_intensity < baseline:
            return "ACTIVE"      # Normal trading, proceed with caution
        elif self.current_intensity < self.mu * 8:
            return "EXCITED"     # High order flow, good entry conditions
        else:
            return "EXPLOSIVE"   # Possible flash event, sit out (too risky)
    
    @property
    def branching_ratio(self) -> float:
        """
        α/β is the expected number of offspring trades per parent trade.
        > 0.9 means market is near critical (flash crash risk).
        < 0.3 means market is very stable (low vol, low opportunity).
        Sweet spot for HFT entry: 0.5 to 0.8
        """
        return self.alpha / self.beta
