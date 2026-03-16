"""
layer4_merton_jump.py — Layer 4: Merton Jump Diffusion Model
=============================================================

Second signal: P(BTC > strike at expiry) under Merton (1976) jump-diffusion.

Model: dS/S = μ dt + σ dW + (J-1) dN
  - σ: diffusion volatility (from Deribit IV or empirical)
  - λ: jump intensity (jumps per year)
  - J: jump size, log(J) ~ N(γ, δ²)
  - k = E[J-1]: expected jump magnitude

Binary probability: P(S_T > K) = Σ_{n=0}^∞ e^{-λ'T}(λ'T)^n/n! × Φ(d_n)
  where d_n uses effective drift and variance conditional on n jumps.

Usage:
    from layer2_engine import Layer2Signals
    from layer4_merton_jump import Layer4MertonEngine, Layer4Signals

    merton = Layer4MertonEngine()
    sig4 = merton.evaluate(layer2_signals)
    # sig4.p_up: Merton-implied P(UP)
"""

import math
from dataclasses import dataclass
from typing import Optional

# ── Constants ───────────────────────────────────────────────────────────────────
WINDOW_S = 300
SIGMA_5MIN_PCT = 0.24
SIGMA_FLOOR_PCT = 0.07
DERIBIT_IV_TO_SIGMA_SCALE = 0.01
POISSON_TRUNCATE = 15  # sum truncated at n=15 (λT small → fast decay)


def _ndtr(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Layer4Signals:
    """Output of Layer 4 Merton jump-diffusion engine."""

    p_up: float = 0.5           # P(BTC > strike) under Merton
    sigma_pct: float = 0.0       # diffusion σ used
    lambda_jump: float = 0.0      # jump intensity (per year)
    jump_vol: float = 0.0        # δ = vol of log-jump size
    n_terms: int = 0             # Poisson terms summed
    # Meta (from Layer2)
    btc_price: float = 0.0
    strike: float = 0.0
    time_remaining: float = 0.0


@dataclass
class Layer4Config:
    """Configuration for Merton jump-diffusion."""

    sigma_5min_pct: float = SIGMA_5MIN_PCT
    sigma_floor_pct: float = SIGMA_FLOOR_PCT
    lambda_jump: float = 2.0     # ~2 jumps per year (BTC: occasional large moves)
    jump_vol_pct: float = 1.5     # δ: log-jump std ~1.5% (moderate jumps)
    jump_mean_pct: float = 0.0   # γ: mean log-jump (0 = symmetric)
    use_deribit_iv: bool = True
    poisson_truncate: int = POISSON_TRUNCATE


class Layer4MertonEngine:
    """
    Merton jump-diffusion binary probability: P(S_T > K).

    P = Σ_{n=0}^N e^{-λ'T}(λ'T)^n/n! × Φ(d_n)
    where d_n = (log(S/K) + m_n) / v_n
    m_n = (-σ²/2 - λk)*T + n*γ
    v_n² = σ²*T + n*δ²
    k = exp(γ + δ²/2) - 1
    """

    def __init__(self, config: Optional[Layer4Config] = None, **kwargs):
        if config is None:
            config = Layer4Config(**kwargs)
        self.config = config

    def _compute_p_up(
        self,
        btc: float,
        strike: float,
        time_remaining: float,
        sigma_5m_pct: float,
        deribit_iv: Optional[float] = None,
    ) -> dict:
        """
        Compute P(BTC > strike) under Merton jump-diffusion.
        Returns dict with p_up, sigma_used, lambda_jump, jump_vol, n_terms.
        """
        cfg = self.config
        if strike <= 0 or btc <= 0 or time_remaining <= 0:
            return {
                "p_up": 0.5,
                "sigma_pct": cfg.sigma_floor_pct,
                "lambda_jump": cfg.lambda_jump,
                "jump_vol": cfg.jump_vol_pct / 100.0,
                "n_terms": 0,
            }

        # σ for diffusion (scale to horizon)
        if cfg.use_deribit_iv and deribit_iv and deribit_iv > 0:
            sigma_pct = max(deribit_iv * DERIBIT_IV_TO_SIGMA_SCALE, cfg.sigma_floor_pct)
        else:
            sigma_pct = max(
                sigma_5m_pct * math.sqrt(max(time_remaining, 1) / WINDOW_S),
                cfg.sigma_floor_pct,
            )

        T = time_remaining / (365.25 * 24 * 3600)  # years
        if T <= 0:
            T = 1e-6

        sigma = sigma_pct / 100.0
        lam = cfg.lambda_jump
        delta = cfg.jump_vol_pct / 100.0
        gamma = cfg.jump_mean_pct / 100.0

        # k = E[J-1] = exp(γ + δ²/2) - 1
        k = math.exp(gamma + 0.5 * delta * delta) - 1.0

        log_ratio = math.log(btc / strike)
        p_sum = 0.0
        n_terms = 0

        for n in range(cfg.poisson_truncate):
            # Poisson weight: e^{-λT} (λT)^n / n!
            poisson_w = math.exp(-lam * T)
            for i in range(1, n + 1):
                poisson_w *= (lam * T) / i

            # Conditional on n jumps: log(S_T/S_0) ~ N(m_n, v_n²)
            # m_n = (-σ²/2 - λk)*T + n*γ
            # v_n² = σ²*T + n*δ²
            m_n = (-0.5 * sigma * sigma - lam * k) * T + n * gamma
            v_n_sq = sigma * sigma * T + n * delta * delta
            v_n = math.sqrt(v_n_sq) if v_n_sq > 0 else 1e-12

            # d_n = (log(S/K) + m_n) / v_n  (P(log(S_T/S_0) > log(K/S_0)))
            d_n = (log_ratio + m_n) / v_n
            d_n = max(-8.0, min(8.0, d_n))
            p_sum += poisson_w * _ndtr(d_n)
            n_terms = n + 1

            if poisson_w < 1e-10:
                break

        p_up = max(0.01, min(0.99, p_sum))

        return {
            "p_up": p_up,
            "sigma_pct": sigma_pct,
            "lambda_jump": lam,
            "jump_vol": delta,
            "n_terms": n_terms,
        }

    def evaluate(
        self,
        layer2_signals,
        deribit_iv: Optional[float] = None,
    ) -> Layer4Signals:
        """
        Evaluate Merton P(UP) from Layer2Signals.
        Requires: btc_price, strike, time_remaining.
        Uses fair_value_sigma from Layer2 if available for diffusion σ.
        """
        btc = getattr(layer2_signals, "btc_price", 0)
        strike = getattr(layer2_signals, "strike", 0)
        t_rem = getattr(layer2_signals, "time_remaining", 300)
        sigma_5m = getattr(layer2_signals, "fair_value_sigma", self.config.sigma_5min_pct)

        if strike <= 0 and btc > 0:
            strike = btc
        if strike <= 0:
            return Layer4Signals()

        div = getattr(layer2_signals, "deribit_iv", None) or deribit_iv

        out = self._compute_p_up(
            btc, strike, t_rem,
            sigma_5m_pct=sigma_5m,
            deribit_iv=div,
        )

        return Layer4Signals(
            p_up=out["p_up"],
            sigma_pct=out["sigma_pct"],
            lambda_jump=out["lambda_jump"],
            jump_vol=out["jump_vol"] * 100,
            n_terms=out["n_terms"],
            btc_price=btc,
            strike=strike,
            time_remaining=t_rem,
        )

    def evaluate_standalone(
        self,
        btc: float,
        strike: float,
        time_remaining: float,
        sigma_5m_pct: Optional[float] = None,
        deribit_iv: Optional[float] = None,
    ) -> float:
        """Standalone P(UP) without Layer2Signals."""
        if sigma_5m_pct is None:
            sigma_5m_pct = self.config.sigma_5min_pct
        out = self._compute_p_up(btc, strike, time_remaining, sigma_5m_pct, deribit_iv)
        return out["p_up"]
