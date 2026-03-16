# kalman_filter.py
# Three filters running in parallel:
#   1. Price filter: removes bid-ask bounce from mid-price
#   2. OBI filter: smooths out order book noise to reveal real pressure
#   3. Spread filter: tracks the true bid-ask spread trend

import numpy as np

class KalmanFilter1D:
    """
    The simplest Kalman Filter for a 1D state (like price or OBI).
    
    State equation:  x_t = x_{t-1} + noise_process   (state evolves slowly)
    Observation:     z_t = x_t + noise_observation    (we see noisy version)
    
    The key insight: if process noise is small (state changes slowly)
    and observation noise is large (measurements are noisy), the filter
    weights history heavily. If process noise is large (state jumps around),
    the filter trusts new observations more. This ratio is what you tune.
    """
    
    def __init__(self, process_noise: float = 1e-4, obs_noise: float = 1e-2):
        # Q: How fast does the true state change?
        # Small Q = assume the thing we're tracking moves slowly
        self.Q = process_noise
        
        # R: How noisy are our observations?  
        # Large R = observations are unreliable (bid-ask bounce is ~$5 on BTC)
        self.R = obs_noise
        
        self.x = None    # Current state estimate (what we think is true)
        self.P = 1.0     # Current uncertainty (starts high, decreases as we learn)
    
    def update(self, observation: float) -> float:
        """
        Two-step process every tick:
        1. Predict: where do we think the state is now?
        2. Update: given the new observation, what's our revised estimate?
        
        The 'Kalman Gain' K is the weight given to the new observation.
        K close to 1: trust the new data. K close to 0: trust the model.
        """
        if self.x is None:
            self.x = observation
            return self.x
        
        # PREDICT step: state uncertainty grows with process noise
        P_pred = self.P + self.Q
        
        # UPDATE step: compute the Kalman Gain
        K = P_pred / (P_pred + self.R)
        
        # Revised estimate: blend prediction with observation
        # K=0.1 means "trust 10% new data, 90% our model prediction"
        self.x = self.x + K * (observation - self.x)
        
        # Update uncertainty: always decreases when we get new data
        self.P = (1 - K) * P_pred
        
        return self.x


class MultiStateKalman:
    """
    A more sophisticated 2-state Kalman Filter that tracks BOTH
    price level AND price velocity (rate of change) simultaneously.
    
    This is powerful because it lets us predict where price is GOING,
    not just where it is now. Velocity is the most direct measure of
    momentum that the filter can provide.
    
    State vector: [price, velocity]
    Think of it as: the filter knows if BTC is moving up at $50/minute
    or $200/minute, and uses that to extrapolate where it'll be in 100ms.
    """
    
    def __init__(self, dt: float = 0.1):
        # dt = default time step in seconds. Overridden by actual elapsed time
        # on every update() call so velocity is correct regardless of tick rate.
        self.dt = dt
        self._last_ts: float = 0.0   # wall-clock time of last update

        # State: [price, velocity]
        self.x = np.array([0.0, 0.0])

        # State covariance
        self.P = np.eye(2) * 100.0

        # Observation matrix: we only observe price
        self.H = np.array([[1, 0]])

        # Process noise tuning constant
        self._q = 0.1

        # Observation noise (BTC bid-ask spread ≈ $5)
        self.R = np.array([[25.0]])

    def _matrices(self, dt: float):
        """Recompute F and Q for the actual elapsed dt."""
        F = np.array([[1, dt], [0, 1]])
        q = self._q
        Q = np.array([[q * dt**3/3, q * dt**2/2],
                      [q * dt**2/2, q * dt]])
        return F, Q

    def update(self, observed_price: float) -> dict:
        # Use actual wall-clock elapsed time for correct velocity $/second.
        now = __import__('time').time()
        if self._last_ts > 0 and self.x[0] > 0:
            dt = min(max(now - self._last_ts, 0.01), 5.0)  # clamp 10ms–5s
        else:
            dt = self.dt
        self._last_ts = now

        F, Q = self._matrices(dt)

        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Innovation
        z = np.array([[observed_price]])
        y = z - self.H @ x_pred.reshape(-1, 1)
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman Gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = x_pred + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return {
            'filtered_price':   self.x[0],
            'velocity':         self.x[1],            # $/second (real time)
            'price_uncertainty': self.P[0, 0],
            'forecast_500ms':   self.x[0] + self.x[1] * 0.5,
        }
