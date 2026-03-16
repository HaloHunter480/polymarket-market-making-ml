# risk_man.py
# The hardest part of trading isn't finding signals — it's not losing
# everything on a bad streak. This module enforces discipline.

import time
import math
import logging
import numpy as np
from collections import deque

log = logging.getLogger("risk")

CONFIG = {
    # Circuit breakers
    "daily_loss_limit":    0.08,   # Stop after 8% daily loss
    "max_drawdown":        0.15,   # Stop after 15% drawdown from peak

    # Kelly sizing
    "kelly_fraction":      0.25,   # Quarter-Kelly (conservative)
    "max_risk_per_trade":  0.05,   # Hard cap: never risk more than 5% per trade

    # R:R assumption (used when stop/TP not explicit)
    "tp_atr_multiplier":   2.0,
    "sl_atr_multiplier":   1.0,

    # Trade limits
    "min_trade_usd":       1.0,
    "max_trade_usd":       25.0,
    "max_session_usd":     100.0,
}


class RiskManager:
    """
    Fractional Kelly criterion for position sizing plus circuit breakers.

    Quarter-Kelly is used because we don't have perfect knowledge of edge.
    Being conservative is the difference between a bot that survives for
    years and one that blows up in a week.
    """

    def __init__(self, initial_balance: float, config: dict = None):
        self.config = config or CONFIG
        self.initial_balance      = initial_balance
        self.peak_balance         = initial_balance
        self.daily_starting_balance = initial_balance
        self.current_balance      = initial_balance
        self.daily_pnl            = 0.0
        self.trade_history        = []
        self.is_paused            = False
        self.pause_reason         = ""
        self._recent_pnls         = deque(maxlen=100)

    # ── Balance tracking ────────────────────────────────────────────────────

    def update_balance(self, new_balance: float):
        self.current_balance = new_balance
        self.daily_pnl = new_balance - self.daily_starting_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    def reset_daily(self):
        """Call at midnight or start of a new session."""
        self.daily_starting_balance = self.current_balance
        self.daily_pnl = 0.0
        self.is_paused = False
        self.pause_reason = ""
        log.info("Daily reset — fresh session started. Balance: $%.2f",
                 self.current_balance)

    # ── Circuit breakers ────────────────────────────────────────────────────

    def check_circuit_breakers(self) -> tuple[bool, str]:
        """Returns (ok, reason). ok=False means: do not trade."""
        if self.is_paused:
            return False, self.pause_reason

        # 1. Daily loss limit
        if self.daily_starting_balance > 0:
            daily_loss_pct = -self.daily_pnl / self.daily_starting_balance
            if daily_loss_pct > self.config["daily_loss_limit"]:
                msg = f"Daily loss {daily_loss_pct:.1%} > limit {self.config['daily_loss_limit']:.0%}"
                self._pause(msg)
                return False, msg

        # 2. Peak drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            if drawdown > self.config["max_drawdown"]:
                msg = f"Drawdown {drawdown:.1%} > max {self.config['max_drawdown']:.0%}"
                self._pause(msg)
                return False, msg

        # 3. Minimum balance floor
        if self.current_balance < self.config["min_trade_usd"] * 2:
            msg = f"Balance ${self.current_balance:.2f} too low to trade"
            return False, msg

        return True, ""

    def _pause(self, reason: str):
        self.is_paused = True
        self.pause_reason = reason
        log.warning("CIRCUIT BREAKER FIRED: %s", reason)

    # ── Kelly position sizing ────────────────────────────────────────────────

    def kelly_size(self, p_model: float, p_market: float) -> float:
        """
        Kelly fraction for a binary market.

        If you buy the UP token at price p_market and win, payout = 1/p_market.
        Kelly: f = (p_model * payout - 1) / (payout - 1) (binary form)
             = (p_model - p_market) / (1 - p_market)

        We apply a quarter-Kelly fraction and hard caps.
        """
        if p_market <= 0 or p_market >= 1:
            return 0.0
        payout = 1.0 / p_market
        kelly_raw = (p_model * payout - 1) / (payout - 1)

        if kelly_raw <= 0:
            return 0.0

        # Apply quarter-Kelly
        kelly_safe = kelly_raw * self.config["kelly_fraction"]

        # Hard caps
        kelly_capped = min(kelly_safe, self.config["max_risk_per_trade"])

        # Convert to USD
        size_usd = self.current_balance * kelly_capped
        size_usd = max(self.config["min_trade_usd"],
                       min(self.config["max_trade_usd"], size_usd))

        log.debug(
            "Kelly: p_model=%.3f p_market=%.3f raw=%.3f safe=%.3f "
            "size=$%.2f", p_model, p_market, kelly_raw, kelly_safe, size_usd
        )
        return round(size_usd, 2)

    def adjust_for_vol_regime(self, size_usd: float, vol_regime: str) -> float:
        """Scale position down in high-vol environments."""
        multipliers = {
            "LOW_VOL":     1.25,
            "NORMAL_VOL":  1.00,
            "HIGH_VOL":    0.50,
            "EXTREME_VOL": 0.00,
            "UNKNOWN":     0.75,
        }
        mult = multipliers.get(vol_regime, 0.75)
        return round(size_usd * mult, 2)

    # ── Trade recording ──────────────────────────────────────────────────────

    def record_trade(self, pnl: float, direction: str, p_model: float,
                     p_market: float, duration_s: float):
        self.trade_history.append({
            "pnl":       pnl,
            "direction": direction,
            "p_model":   round(p_model, 4),
            "p_market":  round(p_market, 4),
            "duration":  round(duration_s, 1),
            "ts":        time.time(),
        })
        self._recent_pnls.append(pnl)
        self.update_balance(self.current_balance + pnl)

        n = len(self.trade_history)
        if n % 20 == 0:
            self._log_stats()

    def _log_stats(self):
        if not self.trade_history:
            return
        wins   = [t for t in self.trade_history if t["pnl"] > 0]
        losses = [t for t in self.trade_history if t["pnl"] <= 0]
        wr     = len(wins) / len(self.trade_history)
        avg_w  = np.mean([t["pnl"] for t in wins])   if wins   else 0
        avg_l  = np.mean([t["pnl"] for t in losses]) if losses else 0
        pf     = abs(avg_w * len(wins)) / (abs(avg_l * len(losses)) + 1e-8)
        log.info(
            "=== STATS [%d trades] WR=%.1f%% AvgW=$%.2f AvgL=$%.2f "
            "PF=%.2f Balance=$%.2f ===",
            len(self.trade_history), wr * 100, avg_w, avg_l, pf,
            self.current_balance
        )

    # ── Rolling edge analysis ────────────────────────────────────────────────

    def edge_drift(self) -> dict:
        """Check if our live edge is holding up. Warn if win rate < 55% over last 20."""
        recent = self.trade_history[-20:]
        if len(recent) < 10:
            return {"n": len(recent), "ok": True}
        wr = sum(1 for t in recent if t["pnl"] > 0) / len(recent)
        mean_pnl = np.mean([t["pnl"] for t in recent])
        return {
            "n":       len(recent),
            "win_rate": round(wr, 3),
            "mean_pnl": round(mean_pnl, 4),
            "ok":       wr >= 0.50,
            "warn":     wr < 0.50,
        }
