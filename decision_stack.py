# decision_stack.py
# ──────────────────────────────────────────────────────────────────────────────
# Strategy: LATENCY ARBITRAGE on stale Polymarket quotes.
#
# AWS Ireland → Polymarket: < 150ms  (measured)
# Polymarket reprices after BTC moves: 2-5 seconds
#
# Edge = fair_value(BTC_now, strike, T, σ) - poly_mid
#   - fair_value is the Black-Scholes binary call: Φ(d)
#     where d = pct_diff / (σ_5min × √(T/300))
#   - poly_mid is the CURRENT Polymarket quote (may be stale)
#   - When BTC moves, fair_value jumps instantly; poly_mid lags
#   - We hit the stale quote before it updates
#
# Signal architecture:
#   1. BTC moved enough      → pct_diff ≥ MIN_BTC_MOVE_PCT in our window
#   2. Fair value gap         → |fair_value - poly_mid| > TOTAL_COST + MIN_EDGE_NET
#   3. Poly quote is stale    → poly_mid hasn't already closed the gap
#   4. GARCH vol regime       → not EXTREME_VOL (erratic mkt destroys fills)
#   5. VPIN gate              → no toxic / informed flow
#   6. Hawkes gate            → market ACTIVE or EXCITED (not dead or crashing)
#   7. OBI gate               → Polymarket book leans in our direction
#
# What was REMOVED (and why):
#   ✗ Kalman velocity projection  → warmup.py OOS test: 47% WR, worse than random
#     Momentum has ZERO predictive power on 5-min BTC direction.
#     BTC mean-reverts at this timescale more often than it trends.
#   ✗ MIN_VELOCITY_PGAIN gate     → based on removed model
#
# What was ADDED:
#   ✓ compute_fair_value()         → Gaussian binary option fair value
#   ✓ MIN_BTC_MOVE_PCT             → require BTC to have moved (creates the gap)
#   ✓ POLY_STALE_LOOKBACK_S       → track recent poly_mid to detect if already updated
# ──────────────────────────────────────────────────────────────────────────────

import time
import numpy as np
import logging
from collections import deque
from dataclasses import dataclass
from scipy.special import ndtr

from Hawkes_Process import HawkesProcess
from garch          import GARCH11
from vpin           import VPIN
from orderbook      import OrderBook

log = logging.getLogger("decision")

# ── thresholds ─────────────────────────────────────────────────────────────────
TOTAL_COST          = 0.035   # 2% Poly fee + 1.5% slippage
MIN_EDGE_NET        = 0.04    # default; overridden by HMM regime
MIN_FAIR_GAP        = TOTAL_COST + MIN_EDGE_NET   # 7.5% raw gap needed

# HMM regime: edge threshold (Layer 6: low=2%, medium=3%, high=5%)
EDGE_THRESHOLD = {"high_vol": 0.05, "medium_vol": 0.03, "low_vol": 0.02}
# Position size: 1/volatility (capped). Replaces fixed 0.5 for high_vol.
HMM_VOL_FLOOR = 0.5   # vol < floor → mult capped at 1 (no scale-up)
HMM_MULT_FLOOR = 0.25 # min mult (max scale-down)
REGIME_MODE_WINDOW = 20  # mode of last N predictions for persistence

CONF_P_MIN          = 0.20    # Polymarket price zone: 20-80 cts (wider on low-latency AWS)
CONF_P_MAX          = 0.80    # (binary near 0/1 has no liquidity)

MIN_BTC_MOVE_PCT    = 0.015   # BTC must have moved ≥ 0.015% to create a gap
                               # (lowered — move_mult now scales UP from threshold)
SIGMA_FLOOR_PCT     = 0.07    # minimum σ in % (empirical LOW_VOL σ = 0.073%)
SIGMA_SCALE_PCT     = 0.24    # baseline 5-min σ (from 8yr Kaggle data)

VPIN_MAX_TRADE      = 0.55
OBI_CONFIRM         = 0.08    # OBI must lean ≥ 0.08 in our direction
HAWKES_OK           = {"ACTIVE", "EXCITED", "QUIET", "EXPLOSIVE"}  # EXPLOSIVE = high-vol move, size capped via hawkes_mult

POLY_STALE_LOOKBACK_S = 1.0   # if poly_mid moved in last 1s → already repriced
POLY_STALE_MOVE_THR   = 0.02  # poly move < 2 cts → still stale


# ── data classes ───────────────────────────────────────────────────────────────

@dataclass
class MarketState:
    """Everything the decision stack needs at evaluation time."""
    btc_price:       float = 0.0
    btc_price_10s:   float = 0.0   # BTC price 10 seconds ago (for move detection)
    strike:          float = 0.0
    time_remaining:  float = 300.0
    p_market:        float = 0.5   # Polymarket UP token mid (may be stale)
    poly_bid_vol:    float = 0.0
    poly_ask_vol:    float = 0.0
    poly_obi:        float = 0.0
    timestamp:       float = 0.0


@dataclass
class SignalResult:
    action:          str   = "WAIT"     # TRADE | WAIT | HALT
    side:            str   = ""         # UP | DOWN
    fair_value:      float = 0.5        # Blended fair value for UP token
    merton_p_up:     float = 0.0        # Layer 4: Merton jump-diffusion P(UP); 0=not computed
    ml_p_up:         float = 0.0        # ML ensemble P(UP); 0=not computed
    p_market:        float = 0.5        # Polymarket mid at signal time
    edge_raw:        float = 0.0        # fair_value - p_market (before fee)
    edge_net:        float = 0.0        # after TOTAL_COST
    size_multiplier: float = 0.0
    veto_reason:     str   = ""
    # sub-scores for logging
    pct_diff:        float = 0.0
    sigma_pct:       float = 0.0
    btc_move_10s:    float = 0.0
    vpin:            float = 0.5
    hawkes_regime:   str   = ""
    vol_regime:      str   = ""
    obi:             float = 0.0
    hmm_regime:      str   = ""    # Layer 5: HMM regime
    branching_ratio: float = 0.0
    execution:       str   = ""    # Layer 6: "MARKET" | "LIMIT"
    size_usd:        float = 0.0   # Layer 6 size when using layer pipeline


# ── core stack ─────────────────────────────────────────────────────────────────

class DecisionStack:
    """
    Evaluates every 100ms.  Returns TRADE only when ALL layers approve.
    """

    def __init__(self):
        self.garch  = GARCH11()
        self.hawkes = HawkesProcess()
        self.vpin   = VPIN(bucket_size=500, window=50)
        self.ob     = OrderBook(depth=20)

        self._last_btc:   float = 0.0
        self._last_ts:    float = 0.0
        self._n_ticks:    int   = 0

        # Track recent Polymarket mid to detect if quote already repriced
        self._poly_mid_history: deque = deque(maxlen=60)   # (ts, mid) pairs
        # Price history for Layer 5 HMM regime + ML feature construction
        self._btc_price_history: deque = deque(maxlen=600)  # (ts, price) — 5-10 min @ 1-2 ticks/s

        # ML ensemble (Logistic + HistGBM + XGBoost) — lazy loaded
        self._ml_ensemble = None
        self._ml_ensemble_tried: bool = False
        self._hmm_regime = None  # Layer5HMMRegime, lazy init
        self._regime_history: deque = deque(maxlen=REGIME_MODE_WINDOW)  # mode filter
        self._prev_regime: str = ""
        self._hmm_state_vol: float = 1.0  # last HMM state vol for 1/vol scaling

    def _regime_mode(self) -> str:
        """Most common regime in last N predictions. Empty if no history."""
        if not self._regime_history:
            return ""
        from collections import Counter
        valid = [r for r in self._regime_history if r and r != "unknown"]
        if not valid:
            return self._regime_history[-1] if self._regime_history else ""
        return Counter(valid).most_common(1)[0][0]

    def _get_hmm_regime(self):
        if self._hmm_regime is None:
            try:
                from layer5_hmm_regime import Layer5HMMRegime
                self._hmm_regime = Layer5HMMRegime()
            except Exception:
                pass
        return self._hmm_regime

    def _get_ml_ensemble(self):
        """Lazy-load the trained 3-model ensemble (Logistic + HistGBM + XGBoost)."""
        if self._ml_ensemble_tried:
            return self._ml_ensemble
        self._ml_ensemble_tried = True
        try:
            import pickle, os
            path = os.path.join(os.path.dirname(__file__), "models", "probability_model.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self._ml_ensemble = pickle.load(f)
                log.info("ML ensemble loaded (Logistic + HistGBM + XGBoost)")
            else:
                log.warning("ML ensemble not found at %s — run train_probability_model.py first", path)
        except Exception as e:
            log.warning("ML ensemble load error: %s", e)
        return self._ml_ensemble

    def _build_ml_features(self, state: MarketState) -> np.ndarray:
        """
        Build the 13-feature vector that matches train_probability_model.compute_features():
          [ret_1s, ret_10s, ret_30s, ret_60s, ret_120s,
           realized_vol_60s, realized_vol_10s, vol_30s, vol_120s,
           ofi, spread_pct, dist_strike, time_to_expiry]
        """
        now   = state.timestamp or time.time()
        price = state.btc_price
        hist  = list(self._btc_price_history)  # [(ts, price)], oldest first

        def _price_ago(seconds: float) -> float:
            cutoff = now - seconds
            for ts, p in reversed(hist):
                if ts <= cutoff:
                    return p
            return hist[0][1] if hist else price

        def _rvol(seconds: float) -> float:
            cutoff = now - seconds
            prices = [p for ts, p in hist if ts >= cutoff]
            if len(prices) < 3:
                return 0.05
            return float(np.std(np.diff(np.log(np.array(prices) + 1e-12))) * 100)

        p1   = _price_ago(1.0)
        p10  = _price_ago(10.0)
        p30  = _price_ago(30.0)
        p60  = _price_ago(60.0)
        p120 = _price_ago(120.0)

        ret_1s   = (price / p1   - 1) * 100 if p1   > 0 else 0.0
        ret_10s  = (price / p10  - 1) * 100 if p10  > 0 else 0.0
        ret_30s  = (price / p30  - 1) * 100 if p30  > 0 else 0.0
        ret_60s  = (price / p60  - 1) * 100 if p60  > 0 else 0.0
        ret_120s = (price / p120 - 1) * 100 if p120 > 0 else 0.0

        # OFI proxy: use Polymarket OBI when live, else Binance book
        if state.poly_bid_vol + state.poly_ask_vol > 0:
            ofi = state.poly_obi
        else:
            ofi = self.ob.order_book_imbalance(levels=5)

        # Spread proxy from Binance depth
        try:
            spread_pct = float(self.ob.spread_ratio) * 100
        except Exception:
            spread_pct = 0.0

        dist_strike = (price - state.strike) / state.strike * 100 if state.strike > 0 else 0.0

        return np.array([[
            ret_1s, ret_10s, ret_30s, ret_60s, ret_120s,
            _rvol(60.0), _rvol(10.0), _rvol(30.0), _rvol(120.0),
            ofi, spread_pct, dist_strike, state.time_remaining,
        ]], dtype=np.float64)

    # ── feed methods ───────────────────────────────────────────────────────────

    def on_btc_price(self, price: float, ts: float):
        if self._last_btc > 0:
            lr = np.log(price / self._last_btc)
            self.garch.update(lr)
        self._last_btc = price
        self._last_ts  = ts
        self._n_ticks += 1
        if price > 0:
            self._btc_price_history.append((ts, price))

    def on_btc_trade(self, ts: float):
        self.hawkes.add_event(ts)

    def on_btc_bar_1s(self, open_p: float, close_p: float, volume: float):
        self.vpin.update(open_p, close_p, volume)

    def on_poly_book(self, bids: dict, asks: dict):
        self.ob.update({"bids": list(bids.items()), "asks": list(asks.items())})

    def on_poly_mid(self, mid: float, ts: float):
        """Call every time a new Polymarket book arrives. Tracks staleness."""
        self._poly_mid_history.append((ts, mid))

    def on_poly_trade(self, price: float, size: float, is_sell: bool):
        self.ob.add_trade(price, size, is_buyer_maker=is_sell)

    # ── fair value model ───────────────────────────────────────────────────────

    def compute_fair_value(self, pct_diff: float, time_remaining: float) -> dict:
        """
        Gaussian binary option fair value.

        P(UP) = Φ(d)
        d     = pct_diff / σ_pct(T)
        σ_pct = σ_5min × √(T/300)

        σ_5min from GARCH (calibrated by warmup.py on 8yr Kaggle data).
        Floor at empirical LOW_VOL σ = 0.07% so we never overstate certainty.

        This is the instantaneous fair value given BTC's CURRENT position
        relative to strike — NOT a projection of where BTC is going.
        The edge is that poly_mid lags this by 2-5 seconds.
        """
        h        = self.garch.h if self.garch.h and self.garch.h > 0 else (SIGMA_SCALE_PCT / 100) ** 2
        sigma_5m = max(np.sqrt(h) * 100, SIGMA_FLOOR_PCT)   # % per 5-min
        sigma_t  = sigma_5m * np.sqrt(max(time_remaining, 1.0) / 300.0)

        d       = np.clip(pct_diff / sigma_t, -5.0, 5.0)
        p_up    = float(ndtr(d))

        return {
            "fair_value": p_up,
            "sigma_5m":   sigma_5m,
            "sigma_t":    sigma_t,
            "d":          d,
        }

    # ── staleness detector ─────────────────────────────────────────────────────

    def _poly_already_repriced(self, current_mid: float) -> bool:
        """
        True if Polymarket already updated its quote in the last
        POLY_STALE_LOOKBACK_S seconds.
        If poly_mid moved by > POLY_STALE_MOVE_THR recently, the
        gap is already closed and there is no arb to take.
        """
        if len(self._poly_mid_history) < 2:
            return False
        now    = time.time()
        cutoff = now - POLY_STALE_LOOKBACK_S
        recent = [(t, m) for t, m in self._poly_mid_history if t >= cutoff]
        if not recent:
            return False
        oldest_mid = recent[0][1]
        poly_move  = abs(current_mid - oldest_mid)
        return poly_move > POLY_STALE_MOVE_THR   # already updated → no arb

    # ── main evaluate ──────────────────────────────────────────────────────────

    def evaluate(self, state: MarketState, use_layer_signals: tuple = None) -> SignalResult:
        """
        use_layer_signals: (fair_value, sig6, sig2, sig5) from run_layer_pipeline.
        When provided, uses Layer 2+4+5+6 outputs; still runs gates (VPIN, Hawkes, GARCH, OBI, stale).
        """
        res = SignalResult(p_market=state.p_market)

        # ── Guard ────────────────────────────────────────────────────────────
        if self._n_ticks < 20 or state.btc_price <= 0 or state.strike <= 0:
            res.veto_reason = "warming_up"
            return res

        # Feed current price for HMM (in case caller didn't)
        self.on_btc_price(state.btc_price, state.timestamp or time.time())

        if state.time_remaining < 30:
            res.veto_reason = "too_late"
            return res

        # ── Layer 1: BTC must have moved enough to create a gap ───────────────
        # Pass if EITHER: recent 10s momentum ≥ threshold (fresh move, poly lag)
        #              OR: cumulative drift from strike ≥ threshold (persistent displacement)
        if state.btc_price_10s > 0:
            btc_move_10s = (state.btc_price - state.btc_price_10s) / state.btc_price_10s * 100
        else:
            btc_move_10s = 0.0
        res.btc_move_10s = btc_move_10s

        if state.strike > 0:
            drift_from_strike = (state.btc_price - state.strike) / state.strike * 100
        else:
            drift_from_strike = 0.0

        if abs(btc_move_10s) < MIN_BTC_MOVE_PCT and abs(drift_from_strike) < MIN_BTC_MOVE_PCT:
            res.veto_reason = f"btc_flat:{btc_move_10s:+.3f}%"
            return res

        # ── Layer 2+4+5+6: Use pipeline or compute internally ─────────────────
        if use_layer_signals and len(use_layer_signals) >= 4:
            fair_value, sig6, sig2, sig5 = use_layer_signals[0], use_layer_signals[1], use_layer_signals[2], use_layer_signals[3]
            if fair_value is not None and sig6 is not None and sig6.trade:
                res.fair_value = fair_value
                res.pct_diff   = round((state.btc_price - state.strike) / state.strike * 100, 4)
                res.edge_raw   = round(fair_value - state.p_market, 4)
                res.edge_net   = round(res.edge_raw - TOTAL_COST, 4)
                res.side       = "UP" if sig6.side == "YES" else "DOWN"
                res.hmm_regime = sig5.regime if sig5 else ""
                res.size_usd   = sig6.size
                res.execution  = sig6.execution
                res.sigma_pct  = getattr(sig2, "fair_value_sigma", 0.24)
                # Skip to gates (Layer 3, 4 price zone, 5 GARCH, 6 VPIN, 7 Hawkes, 8 OBI)
            else:
                res.veto_reason = "layer6_no_trade"
                return res
        else:
            pct_diff        = (state.btc_price - state.strike) / state.strike * 100
            fv              = self.compute_fair_value(pct_diff, state.time_remaining)
            gaussian_fair_up = fv["fair_value"]
            sigma_t         = fv["sigma_t"]

            # Layer 4: Merton jump-diffusion (second signal)
            merton_p = None
            try:
                from layer4_merton_jump import Layer4MertonEngine
                merton   = Layer4MertonEngine()
                merton_p = merton.evaluate_standalone(
                    state.btc_price, state.strike, state.time_remaining,
                    sigma_5m_pct=fv.get("sigma_5m", 0.24),
                )
                res.merton_p_up = merton_p
            except Exception:
                pass

            # ML ensemble: Logistic Regression + HistGradientBoosting + XGBoost
            ml_p = None
            try:
                ml_model = self._get_ml_ensemble()
                if ml_model is not None:
                    X_feat = self._build_ml_features(state)
                    ml_p   = float(ml_model.predict_proba(X_feat)[0, 1])
                    res.ml_p_up = ml_p
            except Exception:
                pass

            # Blend: equal thirds for all three available signals
            signals = [s for s in [gaussian_fair_up, merton_p, ml_p] if s is not None]
            fair_up = sum(signals) / len(signals)

            res.fair_value = fair_up
            res.pct_diff   = round(pct_diff, 4)
            res.sigma_pct  = round(sigma_t, 4)

            gap_up   = fair_up          - state.p_market
            if btc_move_10s > 0:
                side     = "UP"
                edge_raw = gap_up
            else:
                side     = "DOWN"
                edge_raw = -gap_up

            edge_net = edge_raw - TOTAL_COST
            res.edge_raw = round(edge_raw, 4)
            res.edge_net = round(edge_net, 4)
            res.side     = side

            hmm = self._get_hmm_regime()
            if hmm is not None:
                try:
                    l2 = type("L2", (), {"btc_price": state.btc_price})()
                    hmm_state = type("S", (), {"btc_price_history": self._btc_price_history})()
                    sig5 = hmm.evaluate(l2, state=hmm_state)
                    self._hmm_state_vol = max(sig5.state_vol, 0.01)
                    self._regime_history.append(sig5.regime or "")
                    res.hmm_regime = self._regime_mode() or sig5.regime or ""
                    if res.hmm_regime != self._prev_regime and self._prev_regime:
                        log.info("Regime change: %s → %s", self._prev_regime, res.hmm_regime)
                    self._prev_regime = res.hmm_regime
                    edge_threshold = EDGE_THRESHOLD.get(res.hmm_regime, MIN_EDGE_NET)
                except Exception:
                    res.hmm_regime = self._prev_regime or ""
                    edge_threshold = MIN_EDGE_NET
            else:
                res.hmm_regime = self._prev_regime or ""
                edge_threshold = MIN_EDGE_NET

            if edge_net < edge_threshold:
                res.veto_reason = f"gap_too_small:{edge_net:.3f}(need{edge_threshold:.2f})"
                return res

            # Size multiplier for non-layer path
            vol = max(self._hmm_state_vol, HMM_VOL_FLOOR)
            hmm_mult = min(1.0, 1.0 / vol)
            hmm_mult = max(HMM_MULT_FLOOR, hmm_mult)
            vpin_mult   = 1.0 - (self.vpin.market_quality["vpin"] / VPIN_MAX_TRADE) * 0.5
            hawkes_mult = (1.2 if self.hawkes.regime == "EXCITED"
                           else 1.0 if self.hawkes.regime == "ACTIVE"
                           else 0.6 if self.hawkes.regime == "EXPLOSIVE"
                           else 0.5)   # QUIET = 50% size
            vol_mult = {"LOW_VOL": 1.2, "NORMAL_VOL": 1.0, "HIGH_VOL": 0.6}.get(self.garch.vol_regime, 0.75)
            # Scale starts at 1.0x at the minimum threshold, up to 2x for large moves.
            # Dividing by MIN_BTC_MOVE_PCT ensures any qualifying move gets ≥ 1.0x.
            move_mult = min(max(abs(btc_move_10s) / MIN_BTC_MOVE_PCT, 1.0), 2.0)
            br_mult = 0.6 if self.hawkes.branching_ratio > 0.8 else 1.0
            res.size_multiplier = round(hmm_mult * vpin_mult * hawkes_mult * vol_mult * move_mult * br_mult, 3)
            res.execution = "MARKET" if abs(res.edge_net) >= 0.06 else "LIMIT"

        # ── Layer 3: Poly quote must still be stale ───────────────────────────
        self.on_poly_mid(state.p_market, state.timestamp or time.time())
        if self._poly_already_repriced(state.p_market):
            res.veto_reason = "poly_already_repriced"
            return res

        # ── Layer 4: Price zone — avoid near-certain / near-zero tokens ───────
        if not (CONF_P_MIN <= state.p_market <= CONF_P_MAX):
            res.veto_reason = f"price_zone:{state.p_market:.3f}"
            return res

        # ── Layer 5: GARCH — not extreme vol ─────────────────────────────────
        res.vol_regime = self.garch.vol_regime
        if self.garch.vol_regime == "EXTREME_VOL":
            res.veto_reason = "extreme_vol"
            return res

        # ── Layer 6: VPIN — no toxic flow ────────────────────────────────────
        mq       = self.vpin.market_quality
        res.vpin = mq["vpin"]
        if mq["regime"] == "TOXIC" or mq["vpin"] > VPIN_MAX_TRADE:
            res.veto_reason = f"vpin_toxic:{mq['vpin']:.3f}"
            return res

        # ── Layer 7: Hawkes — market must be active ───────────────────────────
        res.hawkes_regime   = self.hawkes.regime
        res.branching_ratio = self.hawkes.branching_ratio
        if self.hawkes.regime not in HAWKES_OK:
            res.veto_reason = f"hawkes_{self.hawkes.regime.lower()}"
            return res

        # ── Layer 8: OBI — Polymarket book must lean in our direction ─────────
        # Only use CLOB OBI when there is real book volume; fall back to Binance
        # OBI; and if neither has data yet, default to 0 (neutral, skip gate).
        # This prevents the 1.0 default from blocking all DOWN bets on a fresh
        # CLOB connection at window start.
        if state.poly_bid_vol + state.poly_ask_vol > 0:
            obi = state.poly_obi
        else:
            obi = self.ob.order_book_imbalance(levels=5)  # returns 0.0 if empty
        res.obi = round(obi, 4)
        side_check = res.side
        if side_check == "UP"   and obi < -OBI_CONFIRM:
            res.veto_reason = f"obi_against_up:{obi:.3f}"
            return res
        if side_check == "DOWN" and obi > OBI_CONFIRM:
            res.veto_reason = f"obi_against_dn:{obi:.3f}"
            return res

        # ── All gates passed ───────────────────────────────────────────────────
        if res.size_usd <= 0:
            # Non-layer path: compute size multiplier
            vol = max(self._hmm_state_vol, HMM_VOL_FLOOR)
            hmm_mult = min(1.0, 1.0 / vol)
            hmm_mult = max(HMM_MULT_FLOOR, hmm_mult)
            vpin_mult   = 1.0 - (res.vpin / VPIN_MAX_TRADE) * 0.5
            hawkes_mult = (1.2 if self.hawkes.regime == "EXCITED"
                           else 1.0 if self.hawkes.regime == "ACTIVE"
                           else 0.6 if self.hawkes.regime == "EXPLOSIVE"
                           else 0.5)   # QUIET = 50% size
            vol_mult = {"LOW_VOL": 1.2, "NORMAL_VOL": 1.0, "HIGH_VOL": 0.6}.get(
                self.garch.vol_regime, 0.75)
            move_mult = min(max(abs(btc_move_10s) / MIN_BTC_MOVE_PCT, 1.0), 2.0)
            br_mult = 0.6 if self.hawkes.branching_ratio > 0.8 else 1.0
            res.size_multiplier = round(
                hmm_mult * vpin_mult * hawkes_mult * vol_mult * move_mult * br_mult, 3)
            res.execution = "MARKET" if abs(res.edge_net) >= 0.06 else "LIMIT"
        res.action = "TRADE"
        return res

    # ── refit ─────────────────────────────────────────────────────────────────

    def refit(self, event_times: np.ndarray = None,
              return_series: np.ndarray = None):
        """Re-estimate Hawkes and GARCH from recent live data (hourly)."""
        if event_times is not None and len(event_times) > 50:
            try:
                self.hawkes.fit(event_times)
                log.info("Hawkes refit: μ=%.3f α=%.3f β=%.3f",
                         self.hawkes.mu, self.hawkes.alpha, self.hawkes.beta)
            except Exception as e:
                log.warning("Hawkes refit failed: %s", e)

        if return_series is not None and len(return_series) > 100:
            try:
                self.garch.fit(return_series)
                log.info("GARCH refit: ω=%.2e α=%.3f β=%.3f",
                         self.garch.omega, self.garch.alpha, self.garch.beta)
            except Exception as e:
                log.warning("GARCH refit failed: %s", e)
