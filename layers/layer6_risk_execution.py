"""
layer6_risk_execution.py — Layer 6: Risk and Execution
=======================================================

1. Compute edge (fair_value - poly_mid for UP side)
2. Apply regime thresholds
3. Choose direction (edge > 0 → buy YES, edge < 0 → buy NO)
4. Position sizing: size = capital × 0.25 × |edge|
5. Volatility adjustment: high_vol → size × 0.5
6. Execution rule: |edge| > 0.06 → market; 0.02–0.06 → limit

Usage:
    from layer6_risk_execution import Layer6Engine, Layer6Signals

    sig6 = Layer6Engine().evaluate(fair_value=0.58, poly_mid=0.52, regime="low_vol", capital=100)
"""

from dataclasses import dataclass
from typing import Optional

# ── Regime edge thresholds ─────────────────────────────────────────────────────
EDGE_THRESHOLD = {"low_vol": 0.02, "medium_vol": 0.03, "high_vol": 0.05}

# Position sizing
KELLY_FRAC = 0.25  # size = capital × 0.25 × |edge|
HIGH_VOL_SIZE_MULT = 0.5

# Execution
MARKET_EDGE_THRESHOLD = 0.06  # |edge| > 0.06 → market order
LIMIT_EDGE_LO = 0.02
LIMIT_EDGE_HI = 0.06

# Position limits (as fraction of capital)
MAX_POSITION_PCT = 0.05   # max position per market = 5% capital
MAX_EXPOSURE_PCT = 0.20   # max total exposure = 20% capital


@dataclass
class Layer6Signals:
    """Output of Layer 6 risk and execution engine."""

    edge: float = 0.0           # fair_value - poly_mid (UP side)
    edge_abs: float = 0.0       # |edge|
    trade: bool = False         # True if |edge| > regime threshold
    side: str = ""              # "YES" | "NO"
    size: float = 0.0           # position size in USD
    execution: str = ""         # "MARKET" | "LIMIT"
    regime: str = ""
    regime_threshold: float = 0.0


class Layer6Engine:
    """
    Layer 6: Risk and execution.
    Consumes fair_value, poly_mid, regime, capital.
    """

    def evaluate(
        self,
        fair_value: float,
        poly_mid: float,
        regime: str = "",
        capital: float = 100.0,
        total_cost: float = 0.035,
    ) -> Layer6Signals:
        """
        1. Compute edge
        2. Apply regime thresholds
        3. Choose direction
        4. Position sizing
        5. Volatility adjustment
        6. Execution rule
        """
        sig = Layer6Signals(regime=regime)

        # 1. Compute edge (fair_value - poly_mid for UP token)
        edge_raw = fair_value - poly_mid
        sig.edge = edge_raw - total_cost  # net after costs
        sig.edge_abs = abs(sig.edge)

        # 2. Regime threshold
        threshold = EDGE_THRESHOLD.get(regime, 0.03)
        sig.regime_threshold = threshold
        sig.trade = sig.edge_abs > threshold

        if not sig.trade:
            return sig

        # 3. Direction
        sig.side = "YES" if sig.edge > 0 else "NO"

        # 4. Position sizing: size = capital × 0.25 × |edge|
        size = capital * KELLY_FRAC * sig.edge_abs

        # 5. Volatility adjustment
        if regime == "high_vol":
            size *= HIGH_VOL_SIZE_MULT

        # Cap at 5% capital per market
        size = min(size, capital * MAX_POSITION_PCT)
        sig.size = max(0.0, size)

        # 6. Execution rule
        if sig.edge_abs > MARKET_EDGE_THRESHOLD:
            sig.execution = "MARKET"
        elif LIMIT_EDGE_LO <= sig.edge_abs <= LIMIT_EDGE_HI:
            sig.execution = "LIMIT"
        else:
            sig.execution = "LIMIT"  # fallback

        return sig
