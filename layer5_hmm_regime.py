"""
layer5_hmm_regime.py — Layer 5: HMM Regime Classifier
======================================================

Hidden Markov Model for market regime classification.
Uses Gaussian HMM on log returns to detect: low_vol, high_vol, trending_up, trending_down.

Regimes map to position scaling / strategy adaptation:
  - low_vol / mean_rev: scale up (predictable)
  - high_vol / volatile: scale down (erratic)
  - trending: directional bias

Usage:
    from layer2_engine import Layer2Signals
    from layer5_hmm_regime import Layer5HMMRegime, Layer5Signals

    hmm = Layer5HMMRegime()
    sig5 = hmm.evaluate(layer2_signals, state=triple_stream_state)
    # sig5.regime, sig5.regime_idx, sig5.confidence
"""

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ── Constants ───────────────────────────────────────────────────────────────────
MIN_SAMPLES = 60       # min samples for predict (need 120 for full features)
BUFFER_SIZE = 300      # rolling buffer
N_COMPONENTS = 4
REGIME_LABELS = ("low_vol", "trending_up", "trending_down", "high_vol")
DEFAULT_MODEL_PATH = "models/hmm_regime.pkl"

# Feature names (must match train_hmm_regime.py)
HMM_FEATURE_NAMES = [
    "returns_1s", "returns_10s", "returns_30s", "returns_60s",
    "realized_vol_30s", "realized_vol_120s",
    "order_flow_imbalance", "spread_pct", "volatility_short",
]


@dataclass
class Layer5Signals:
    """Output of Layer 5 HMM regime classifier."""

    regime: str = "unknown"       # low_vol | medium_vol | high_vol (manual mapping)
    regime_idx: int = -1         # HMM state index
    confidence: float = 0.0       # posterior prob of current state
    state_mean: float = 0.0      # mean return of current state
    state_vol: float = 0.0       # vol of current state
    n_samples: int = 0           # returns used
    # Meta
    btc_price: float = 0.0
    timestamp: float = 0.0


@dataclass
class Layer5Config:
    """Configuration for HMM regime classifier."""

    min_samples: int = MIN_SAMPLES
    buffer_size: int = BUFFER_SIZE
    n_components: int = N_COMPONENTS
    model_path: Optional[str] = None   # load pre-fitted HMM (default: models/hmm_regime.pkl)
    throttle_s: float = 1.0            # min seconds between price adds (avoid duplicates)


class Layer5HMMRegime:
    """
    HMM regime classifier trained on multi-feature 1s data.
    Uses: returns_1s/10s/30s/60s, realized_vol_30s/120s, OFI, spread, vol_short.
    At inference: computes features from price buffer; OFI=0 if not available, spread=range proxy.
    """

    def __init__(self, config: Optional[Layer5Config] = None, **kwargs):
        if config is None:
            config = Layer5Config(**kwargs)
        self.config = config
        self._prices: deque = deque(maxlen=config.buffer_size)
        self._feature_buffer: list = []  # (price, vol, buy_vol, spread) for full features
        self._last_price: Optional[float] = None
        self._last_add_ts: float = 0.0
        self._model = None
        self._scaler = None
        self._state_labels: Optional[list] = None
        path = config.model_path or DEFAULT_MODEL_PATH
        if Path(path).exists():
            self._load_model(path)
        else:
            self._model = None
            self._scaler = None

    def _load_model(self, path: str) -> bool:
        """Load pre-fitted HMM + scaler from path."""
        try:
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._model = data.get("model")
            self._scaler = data.get("scaler")
            self._state_labels = data.get("state_labels")
            return self._model is not None
        except Exception:
            return False

    def _add_price(self, price: float, ts: float):
        """Add price to buffer (throttled)."""
        if price <= 0:
            return
        if self._last_price is not None and abs(price - self._last_price) < 1e-8:
            if ts - self._last_add_ts < self.config.throttle_s:
                return
        self._prices.append(price)
        self._last_price = price
        self._last_add_ts = ts

    def _compute_features_from_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute HMM features from price array only.
        OFI=0, spread_pct=(max-min)/mid proxy. Same structure as train_hmm_regime.
        """
        n = len(prices)
        if n < 121:
            return np.zeros((0, len(HMM_FEATURE_NAMES)))
        rows = []
        for i in range(120, n):
            p = prices[i]
            ret_1s = (prices[i] / prices[i - 1] - 1) * 100
            ret_10s = (prices[i] / prices[i - 10] - 1) * 100
            ret_30s = (prices[i] / prices[i - 30] - 1) * 100
            ret_60s = (prices[i] / prices[i - 60] - 1) * 100
            rets_30 = np.diff(np.log(prices[i - 30 : i + 1] + 1e-12))
            vol_30s = np.std(rets_30) * 100 if len(rets_30) > 2 else 0.05
            rets_120 = np.diff(np.log(prices[i - 120 : i + 1] + 1e-12))
            vol_120s = np.std(rets_120) * 100 if len(rets_120) > 5 else 0.05
            rets_10 = np.diff(np.log(prices[i - 10 : i + 1] + 1e-12))
            vol_short = np.std(rets_10) * 100 if len(rets_10) > 1 else 0.05
            ofi = 0.0  # not available at inference
            window = prices[i - 10 : i + 1]
            mid = np.mean(window)
            spread_pct = (np.max(window) - np.min(window)) / mid * 100 if mid > 0 else 0.0
            rows.append([ret_1s, ret_10s, ret_30s, ret_60s, vol_30s, vol_120s, ofi, spread_pct, vol_short])
        return np.array(rows, dtype=np.float64)

    def _returns(self) -> np.ndarray:
        """Log returns from price buffer (fallback when no trained model)."""
        arr = np.array(self._prices, dtype=np.float64)
        if len(arr) < 2:
            return np.array([])
        log_rets = np.diff(np.log(arr + 1e-12)) * 100
        return log_rets.reshape(-1, 1)

    def _fit_model_fallback(self, X: np.ndarray):
        """Fit Gaussian HMM on returns when no pre-trained model (legacy)."""
        try:
            from hmmlearn.hmm import GaussianHMM
            self._model = GaussianHMM(
                n_components=self.config.n_components,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
            )
            self._model.fit(X)
            self._assign_state_labels_fallback()
        except Exception:
            self._model = None

    def _assign_state_labels_fallback(self):
        """Map HMM states to regime names (single-feature model)."""
        if self._model is None:
            return
        means = self._model.means_.flatten()
        vars_ = np.array([self._model.covars_[i].flatten()[0] for i in range(self.config.n_components)])
        med = np.median(vars_)
        self._state_labels = [""] * self.config.n_components
        for i in range(self.config.n_components):
            m, v = means[i], vars_[i]
            if v > med * 1.5:
                self._state_labels[i] = "high_vol"
            elif m > 0.02:
                self._state_labels[i] = "trending_up"
            elif m < -0.02:
                self._state_labels[i] = "trending_down"
            else:
                self._state_labels[i] = "low_vol"

    def _predict(self, X: np.ndarray) -> tuple:
        """Predict regime. X is scaled features (n, n_features) or returns (n, 1)."""
        if self._model is None or len(X) < self.config.min_samples:
            return -1, 0.0, 0.0, 0.0
        try:
            if self._scaler is not None:
                X = self._scaler.transform(X)
            posteriors = self._model.predict_proba(X)
            idx = int(np.argmax(posteriors[-1]))
            conf = float(posteriors[-1][idx])
            means = self._model.means_[idx]
            mean = float(means.flatten()[0])
            var = float(self._model.covars_[idx].flatten()[0])
            vol = np.sqrt(max(var, 1e-12))
            return idx, conf, mean, vol
        except Exception:
            return -1, 0.0, 0.0, 0.0

    def evaluate(
        self,
        layer2_signals,
        state=None,
    ) -> Layer5Signals:
        """
        Evaluate HMM regime from price history.
        Uses layer2_signals.btc_price; if state has btc_price_history, uses that.
        Otherwise appends btc_price to internal buffer each call.
        """
        out = Layer5Signals(timestamp=time.time())
        btc = getattr(layer2_signals, "btc_price", 0)
        out.btc_price = btc

        if btc <= 0:
            return out

        # Get price history: from state or internal buffer
        if state is not None:
            hist = getattr(state, "btc_price_history", None)
            if hist is not None and len(hist) > 0:
                prices = [p for _, p in list(hist)[-self.config.buffer_size:] if p > 0]
                self._prices.clear()
                self._prices.extend(prices)
            else:
                self._add_price(btc, time.time())
        else:
            self._add_price(btc, time.time())

        prices_arr = np.array(self._prices, dtype=np.float64)

        if self._model is not None and self._scaler is not None:
            # Use trained model: compute full features from prices
            X = self._compute_features_from_prices(prices_arr)
            out.n_samples = len(X)
            if len(X) < self.config.min_samples:
                return out
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -50, 50)
        else:
            # Fallback: returns only, fit on the fly
            X = self._returns()
            out.n_samples = len(X)
            if len(X) < self.config.min_samples:
                return out
            if self._model is None:
                self._fit_model_fallback(X)
            if self._model is None:
                return out

        idx, conf, mean, vol = self._predict(X)
        out.regime_idx = idx
        out.confidence = conf
        out.state_mean = mean
        out.state_vol = vol
        out.regime = self._regime_idx_to_type(idx)

        return out

    def _regime_idx_to_type(self, regime: int) -> str:
        """
        Manual mapping: regime 2 = highest vol (from means inspection), 1 = medium, 0 = low.
        Trained model (3 components): 0=low_vol, 1=medium_vol, 2=high_vol.
        """
        if regime == 2:  # highest volatility state
            return "high_vol"
        elif regime == 1:
            return "medium_vol"
        elif regime == 0:
            return "low_vol"
        return "unknown"
