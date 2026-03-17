"""
layer2_engine.py — Layer 2 signal engine
========================================

Consumes Layer 1 inputs (triple_streams.TripleStreamState) and produces:

1. Fair value model    — Gaussian binary P(BTC > strike) using σ from Deribit IV or empirical
2. OBI signal         — Order book imbalance confirms direction (Binance + Polymarket)
3. Cross-market arb   — UP + DOWN = 1.0 constraint; violation = structural edge
4. Oracle lag        — Polymarket lags Binance; estimated lag window for stale-quote arb

Usage:
    from triple_streams import TripleStreamState, TripleStreamConfig, run_triple_streams
    from layer2_engine import Layer2Engine, Layer2Signals

    state = await run_triple_streams(config)
    engine = Layer2Engine(strike=97000, time_remaining=180, token_up="...", token_down="...")
    signals = engine.evaluate(state)
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ── Fair value constants ───────────────────────────────────────────────────────
SIGMA_5MIN_PCT = 0.24   # empirical 5-min σ from Kaggle 8yr data
SIGMA_FLOOR_PCT = 0.07  # minimum σ (LOW_VOL regime)
WINDOW_S = 300          # 5-min window
DERIBIT_IV_TO_SIGMA_SCALE = 0.01  # Deribit IV ~50 → sigma_pct ~0.5% per tick


def _ndtr(x: float) -> float:
    """Gaussian CDF — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Layer2Signals:
    """Output of Layer 2 engine."""

    # 1. Fair value model
    fair_value: float = 0.5       # P(UP) = Φ(d)
    fair_value_sigma: float = 0.0  # σ used
    fair_value_d: float = 0.0      # d = pct_diff / σ_t
    edge_raw: float = 0.0          # fair_value - poly_mid (UP side)
    edge_net: float = 0.0         # after costs

    # 2. OBI signal
    binance_obi: float = 0.0      # Binance order book imbalance
    poly_obi: float = 0.0         # Polymarket OBI (UP token)
    obi_confirms: bool = False    # OBI leans in trade direction
    obi_signal: str = ""          # "BULLISH" | "BEARISH" | "NEUTRAL"

    # 3. Cross-market arb
    cross_sum: float = 0.0        # mid_up + mid_down (should = 1.0)
    cross_violation: float = 0.0   # cross_sum - 1.0
    cross_edge_cents: float = 0.0  # per-token edge from violation
    cross_arb_opportunity: bool = False

    # 4. Oracle lag
    oracle_lag_est_ms: float = 0.0   # estimated Polymarket lag behind Binance
    gap_open: float = 0.0             # |fair_value - poly_mid| when BTC moved
    gap_persists: bool = False        # gap still open (stale quote)
    last_btc_move_ts: float = 0.0
    poly_mid_at_move: float = 0.0

    # Layer 4: Merton jump-diffusion (second signal, set by caller when Layer4 runs)
    merton_p_up: float = 0.0       # P(UP) from Merton model; 0 = not computed

    # Layer 5: HMM regime (set by caller when Layer5 runs)
    hmm_regime: str = ""            # low_vol | trending_up | trending_down | high_vol
    hmm_confidence: float = 0.0     # posterior prob of regime

    # Meta
    poly_mid: float = 0.5
    btc_price: float = 0.0
    strike: float = 0.0
    time_remaining: float = 0.0
    timestamp: float = 0.0


@dataclass
class Layer2Config:
    """Configuration for Layer 2 engine."""

    strike: float = 0.0
    time_remaining: float = 300.0
    token_up: str = ""
    token_down: str = ""
    total_cost: float = 0.035      # fee + slippage
    min_edge_net: float = 0.04
    obi_confirm_threshold: float = 0.08
    cross_violation_threshold: float = 0.01   # 1% sum deviation = arb
    use_deribit_iv: bool = True    # use Deribit IV for σ when available


class Layer2Engine:
    """
    Consumes Layer 1 state, produces Layer 2 signals.
    """

    def __init__(self, config: Optional[Layer2Config] = None, **kwargs):
        if config is None:
            config = Layer2Config(**kwargs)
        self.config = config
        self._btc_prices: deque = deque(maxlen=120)  # (ts, price)
        self._last_btc_move_ts: float = 0.0
        self._last_btc_move_price: float = 0.0
        self._poly_mid_history: deque = deque(maxlen=100)  # (ts, mid)
        self._open_gap: Optional[float] = None  # gap when BTC moved

    def _get_btc_price(self, state) -> float:
        """Get current BTC price from Layer 1 state."""
        if state.last_binance_trade:
            return float(state.last_binance_trade.get("p", 0) or 0)
        if state.binance_book and (state.binance_book.bids or state.binance_book.asks):
            bb = state.binance_book.best_bid
            ba = state.binance_book.best_ask
            return (bb + ba) / 2 if (bb > 0 and ba < float("inf")) else 0
        return 0.0

    def _get_poly_mid(self, state, token_id: str) -> float:
        """Get Polymarket mid for token from Layer 1 state."""
        book = state.poly_book.get(token_id, {})
        return book.get("mid", 0.5) or 0.5

    def _compute_fair_value(
        self,
        btc: float,
        strike: float,
        time_remaining: float,
        deribit_iv: Optional[float] = None,
    ) -> dict:
        """
        P(BTC > strike at expiry) under Gaussian assumption.
        σ from Deribit IV if available, else empirical.
        """
        if strike <= 0 or btc <= 0 or time_remaining <= 0:
            return {"fair_value": 0.5, "sigma_t": SIGMA_FLOOR_PCT, "d": 0.0}

        pct_diff = (btc - strike) / strike * 100.0

        if self.config.use_deribit_iv and deribit_iv and deribit_iv > 0:
            # Deribit IV is annualized; scale to 5-min
            sigma_5m = max(deribit_iv * DERIBIT_IV_TO_SIGMA_SCALE, SIGMA_FLOOR_PCT)
        else:
            sigma_5m = max(
                SIGMA_5MIN_PCT * math.sqrt(max(time_remaining, 1) / WINDOW_S),
                SIGMA_FLOOR_PCT,
            )

        sigma_t = sigma_5m
        d = max(-5.0, min(5.0, pct_diff / sigma_t))
        p_up = _ndtr(d)

        return {
            "fair_value": p_up,
            "sigma_t": sigma_t,
            "sigma_5m": sigma_5m,
            "d": d,
        }

    def _detect_btc_move(self, btc: float, ts: float) -> bool:
        """Track BTC prices, detect significant move. Returns True if move detected."""
        self._btc_prices.append((ts, btc))
        if len(self._btc_prices) < 10:
            return False
        # Price ~10s ago
        p_old = None
        for t, p in reversed(self._btc_prices):
            if ts - t >= 10.0:
                p_old = p
                break
        if p_old is None or p_old <= 0:
            return False
        move_pct = (btc - p_old) / p_old * 100
        if abs(move_pct) >= 0.04:  # MIN_BTC_MOVE_PCT
            self._last_btc_move_ts = ts
            self._last_btc_move_price = btc
            return True
        return False

    def evaluate(self, state, strike: Optional[float] = None, time_remaining: Optional[float] = None) -> Layer2Signals:
        """
        Evaluate Layer 2 signals from Layer 1 state.
        """
        cfg = self.config
        btc = self._get_btc_price(state)
        strike = strike if strike is not None else cfg.strike
        if strike <= 0 and btc > 0:
            strike = btc  # use current BTC as strike when not set
        time_remaining = time_remaining if time_remaining is not None else cfg.time_remaining

        sig = Layer2Signals(
            strike=strike,
            time_remaining=time_remaining,
            timestamp=time.time(),
        )
        sig.btc_price = btc

        # Poly mid (UP token)
        poly_mid_up = self._get_poly_mid(state, cfg.token_up) if cfg.token_up else 0.5
        poly_book = getattr(state, "poly_book", {}) or {}
        if cfg.token_down and cfg.token_down in poly_book:
            poly_mid_dn = self._get_poly_mid(state, cfg.token_down)
        else:
            poly_mid_dn = 1.0 - poly_mid_up  # UP + DOWN = 1
        sig.poly_mid = poly_mid_up

        # ── 1. Fair value model ─────────────────────────────────────────────────
        fv = self._compute_fair_value(
            btc, strike, time_remaining,
            deribit_iv=getattr(state, "deribit_iv", None),
        )
        sig.fair_value = fv["fair_value"]
        sig.fair_value_sigma = fv.get("sigma_5m", fv["sigma_t"])
        sig.fair_value_d = fv["d"]

        sig.edge_raw = sig.fair_value - poly_mid_up
        sig.edge_net = sig.edge_raw - cfg.total_cost

        # ── 2. OBI signal ─────────────────────────────────────────────────────
        binance_book = getattr(state, "binance_book", None)
        if binance_book and (binance_book.bids or binance_book.asks):
            sig.binance_obi = binance_book.order_book_imbalance(levels=5)
        poly_book = getattr(state, "poly_book", {}) or {}
        poly_book_up = poly_book.get(cfg.token_up, {}) if cfg.token_up else {}
        sig.poly_obi = poly_book_up.get("order_book_imbalance", 0.0)

        # OBI signal: positive = bullish, negative = bearish
        obi = sig.poly_obi if (poly_book_up.get("bids") or poly_book_up.get("asks")) else sig.binance_obi
        if obi > cfg.obi_confirm_threshold:
            sig.obi_signal = "BULLISH"
            sig.obi_confirms = sig.edge_raw > 0  # UP edge + bullish OBI
        elif obi < -cfg.obi_confirm_threshold:
            sig.obi_signal = "BEARISH"
            sig.obi_confirms = sig.edge_raw < 0  # DOWN edge + bearish OBI
        else:
            sig.obi_signal = "NEUTRAL"
            sig.obi_confirms = False

        # ── 3. Cross-market arb (UP + DOWN = 1.0) ───────────────────────────────
        sig.cross_sum = poly_mid_up + poly_mid_dn
        sig.cross_violation = sig.cross_sum - 1.0
        if abs(sig.cross_violation) >= cfg.cross_violation_threshold:
            sig.cross_arb_opportunity = True
            # Per-token edge: if sum > 1, both overpriced → sell; if sum < 1, underpriced → buy
            sig.cross_edge_cents = -sig.cross_violation * 50  # rough cents per token

        # ── 4. Oracle lag ──────────────────────────────────────────────────────
        ts = time.time()
        self._poly_mid_history.append((ts, poly_mid_up))

        if self._detect_btc_move(btc, ts):
            self._open_gap = abs(sig.fair_value - poly_mid_up)
            sig.gap_open = self._open_gap
            sig.last_btc_move_ts = ts
            sig.poly_mid_at_move = poly_mid_up

        if self._open_gap is not None and self._last_btc_move_ts > 0:
            current_gap = abs(sig.fair_value - poly_mid_up)
            sig.gap_persists = current_gap >= 0.02  # CLOSE_GAP
            sig.oracle_lag_est_ms = (ts - self._last_btc_move_ts) * 1000 if sig.gap_persists else 0
            if not sig.gap_persists:
                self._open_gap = None

        return sig


def compute_fair_value_standalone(
    btc: float,
    strike: float,
    time_remaining: float,
    deribit_iv: Optional[float] = None,
) -> float:
    """
    Standalone fair value for UP token. No state required.
    """
    eng = Layer2Engine(Layer2Config(strike=strike, time_remaining=time_remaining))
    return eng._compute_fair_value(btc, strike, time_remaining, deribit_iv)["fair_value"]
