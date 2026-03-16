"""
warmup.py — Pre-train GARCH + Hawkes + validate Gaussian model
==============================================================

Data source (preferred): Kaggle 5-min BTC dataset
  shivaverse/btcusdt-5-minute-ohlc-volume-data-2017-2025
  846,445 candles · 2017-09-01 → 2025-09-23

Fallback: btc_1m_candles.pkl (local 31-day snapshot)

What this script does:
  1. Fit GARCH(1,1) on 5-min log returns (last 3 months of training data)
  2. Fit Hawkes process parameters from volume-derived event times
  3. OOS validation: does Gaussian momentum model (Kalman vel + GARCH σ)
     actually predict 5-min direction better than 50/50?
  4. Save calibrated state to logs/model_warmup.pkl

Run BEFORE run_live.py:   python3 warmup.py
Re-run daily to stay fresh.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.optimize import minimize

os.makedirs("logs", exist_ok=True)

KAGGLE_CSV = (
    "/Users/harjot/.cache/kagglehub/datasets/"
    "shivaverse/btcusdt-5-minute-ohlc-volume-data-2017-2025/"
    "versions/1/BTCUSDT_5m_2017-09-01_to_2025-09-23.csv"
)
FALLBACK_PKL  = "btc_1m_candles.pkl"
WARMUP_OUT    = "logs/model_warmup.pkl"

GARCH_FIT_N   = 4032    # ~14 days of 5-min candles for warmup loop
HAWKES_N      = 1000
TRAIN_FRAC    = 0.80    # 80% train / 20% OOS
CONF_MIN      = 0.62    # same threshold as decision_stack.py
VEL_WINDOW    = 12      # candles (60 min) to estimate entry velocity
WINDOW_SEC    = 300     # 5-min windows


# ─── data loading ──────────────────────────────────────────────────────────────

def load_5min_kaggle() -> pd.DataFrame | None:
    if not os.path.exists(KAGGLE_CSV):
        return None
    print(f"  Loading 5-min Kaggle dataset...")
    df = pd.read_csv(KAGGLE_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  {len(df):,} candles  ({df.iloc[0]['datetime'].date()} → "
          f"{df.iloc[-1]['datetime'].date()})")
    return df


def load_1min_fallback() -> pd.DataFrame | None:
    if not os.path.exists(FALLBACK_PKL):
        return None
    print(f"  Loading fallback 1-min candles from {FALLBACK_PKL}...")
    with open(FALLBACK_PKL, "rb") as f:
        candles = pickle.load(f)
    # Group into 5-min windows
    rows = []
    for i in range(0, len(candles) - 4, 5):
        block = candles[i:i + 5]
        ts    = block[0][0]
        op    = float(block[0][1])
        hi    = max(float(c[2]) for c in block)
        lo    = min(float(c[3]) for c in block)
        cl    = float(block[-1][4])
        vol   = sum(float(c[5]) for c in block)
        rows.append({"datetime": ts, "open": op, "high": hi,
                     "low": lo, "close": cl, "volume": vol})
    df = pd.DataFrame(rows)
    print(f"  {len(df):,} 5-min candles from 1-min fallback")
    return df


# ─── GARCH(1,1) fit ────────────────────────────────────────────────────────────

def fit_garch(df: pd.DataFrame):
    from garch import GARCH11

    closes  = df["close"].values
    returns = np.diff(np.log(closes))
    returns = returns[np.isfinite(returns)]

    # Fit on most recent GARCH_FIT_N candles (avoids 2017-2020 regime)
    fit_rets = returns[-GARCH_FIT_N:]

    g = GARCH11()
    # Warm up recursive filter on full recent history
    for r in fit_rets[-2000:]:
        g.update(r)

    print(f"  Fitting GARCH(1,1) on {len(fit_rets):,} 5-min returns...")
    g.fit(fit_rets)

    # Re-warm after fit so h is correct
    h_init = np.var(fit_rets[-100:])
    g.h = h_init
    for r in fit_rets[-200:]:
        g.h = g.omega + g.alpha * r**2 + g.beta * g.h

    ann_vol    = np.sqrt(g.h * 105120) * 100   # 105120 = 5-min periods/yr
    sigma_5min = np.sqrt(g.h) * 100            # 5-min sigma in %

    print(f"  GARCH fitted:  ω={g.omega:.3e}  α={g.alpha:.4f}  β={g.beta:.4f}")
    print(f"  Current:       σ(5-min)={sigma_5min:.4f}%  "
          f"ann_vol={ann_vol:.1f}%  regime={g.vol_regime}")
    return g


# ─── Hawkes fit ────────────────────────────────────────────────────────────────

def fit_hawkes(df: pd.DataFrame):
    from Hawkes_Process import HawkesProcess

    recent  = df.tail(500)
    vols    = np.maximum(recent["volume"].values, 1e-3)
    mean_v  = vols.mean()

    # Synthetic event times: high volume → more trade events in that bar
    event_times = []
    t = 0.0
    for v in vols:
        n_ev = max(1, int(round(v / mean_v * 8)))
        for k in range(n_ev):
            event_times.append(t + k * WINDOW_SEC / n_ev)
        t += WINDOW_SEC

    event_times = np.array(event_times[-HAWKES_N:], dtype=float)
    h = HawkesProcess()

    # Seed intensity with tail of events mapped to now
    now = time.time()
    seed = event_times[-200:] - event_times[-200:][0] + (now - 120)
    for t in seed:
        h.add_event(t)

    print(f"  Fitting Hawkes on {len(event_times):,} synthetic events...")
    h.fit(event_times)
    print(f"  Hawkes fitted: μ={h.mu:.3f}  α={h.alpha:.3f}  β={h.beta:.3f}"
          f"  branching={h.branching_ratio:.3f}  regime={h.regime}")
    return h


# ─── Gaussian model OOS validation ────────────────────────────────────────────

def validate(df: pd.DataFrame, garch) -> dict:
    """
    Vectorised OOS validation — runs in <5 seconds on 846K candles.

    Velocity = linear slope of last VEL_WINDOW closes fitted in one
    strided matrix operation (no Python loop over polyfit).
    GARCH σ is approximated by rolling realised vol (avoids 169K update calls).
    """
    print(f"\n  Running OOS validation on last {1 - TRAIN_FRAC:.0%} of data "
          f"({int(len(df)*(1-TRAIN_FRAC)):,} candles)...")

    closes  = df["close"].values.astype(float)
    opens   = df["open"].values.astype(float)
    n       = len(df)
    split   = int(n * TRAIN_FRAC)
    W       = VEL_WINDOW

    # ── velocity: vectorised linear slope over rolling window ────────────────
    # Build strided view: shape (n - W, W)
    shape   = (n - W, W)
    strides = (closes.strides[0], closes.strides[0])
    windows = np.lib.stride_tricks.as_strided(closes, shape=shape, strides=strides)

    x     = np.arange(W, dtype=float) * WINDOW_SEC
    x     = x - x.mean()
    xvar  = (x ** 2).sum()
    # slope = cov(x, y) / var(x)
    y_dm  = windows - windows.mean(axis=1, keepdims=True)
    slopes = (x * y_dm).sum(axis=1) / xvar         # $/sec, shape (n-W,)

    # Indices: slopes[k] → velocity entering candle k+W
    idx_start = W + split     # first OOS candle index
    vel_all   = slopes[split:]   # velocity for OOS candles

    # ── GARCH σ: rolling realised vol (fast, no update loop) ─────────────────
    log_rets = np.diff(np.log(np.maximum(closes, 1e-10)))
    # 96-period rolling std ≈ 8-hour realised vol, converted to % per 5-min
    roll_std = pd.Series(log_rets).rolling(96, min_periods=20).std().values
    # Align: roll_std[i] → vol entering candle i+1
    # For candle i in OOS: use roll_std[i-1]
    sigma_arr = np.maximum(roll_std[split - 1 : split - 1 + len(vel_all)] * 100, 0.05)

    # ── Gaussian p_model ─────────────────────────────────────────────────────
    oos_closes = closes[idx_start : idx_start + len(vel_all)]
    oos_opens  = opens [idx_start : idx_start + len(vel_all)]

    safe_strikes = np.maximum(oos_opens, 1.0)
    vel_pct_s    = vel_all / safe_strikes * 100       # %/sec
    proj_pct     = vel_pct_s * WINDOW_SEC             # pct change at expiry

    z      = np.clip(proj_pct / sigma_arr, -4.5, 4.5)
    p_up   = ndtr(z)
    side   = p_up >= 0.5
    p_model_arr = np.where(side, p_up, 1.0 - p_up)

    # ── filter confident ─────────────────────────────────────────────────────
    conf_mask   = p_model_arr >= CONF_MIN
    outcome_up  = oos_closes >= oos_opens
    correct     = np.where(side, outcome_up, ~outcome_up)
    outcome_arr = correct.astype(int)

    p_conf   = p_model_arr[conf_mask]
    out_conf = outcome_arr [conf_mask]
    side_conf = side       [conf_mask]

    if len(p_conf) == 0:
        print("  WARNING: zero confident OOS predictions — "
              "check CONF_MIN threshold or VEL_WINDOW")
        return {}

    n_pred  = len(p_conf)
    wins    = int(out_conf.sum())
    wr      = wins / n_pred
    mean_pm = float(p_conf.mean())
    brier   = float(((p_conf - out_conf) ** 2).mean())
    n_up    = int(side_conf.sum())
    n_dn    = n_pred - n_up

    # Edge over 50/50 baseline (p_market proxy = 0.50, no real book here)
    edge_over_base = mean_pm - 0.50

    # Profitable if: win_rate > 1 / (1 + payout) where payout ≈ 1/(mean_pm)
    # At mean_pm = 0.64, payout = 0.5625x → break-even win rate = 64%
    # Simpler: WR > mean_pm (Kelly > 0 condition)
    kelly_positive = wr > mean_pm

    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  OOS GAUSSIAN MODEL VALIDATION              │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Confident signals:  {n_pred:>7,}              │")
    print(f"  │  UP / DOWN:          {n_up:>7,} / {n_dn:<7,}     │")
    print(f"  │  Win rate:           {wr:>7.1%}              │")
    print(f"  │  Mean P_model:       {mean_pm:>7.4f}              │")
    print(f"  │  Brier score:        {brier:>7.4f}   (0.25=rand) │")
    print(f"  │  Edge vs 0.5 base:   {edge_over_base:>+7.4f}              │")
    print(f"  │  Kelly > 0:          {str(kelly_positive):>7}              │")
    print(f"  └─────────────────────────────────────────────┘")

    if wr >= 0.57:
        verdict = "✅  STRONG EDGE — go to paper trading"
    elif wr >= 0.53:
        verdict = "⚠️   MARGINAL EDGE — paper trade and verify with real book data"
    else:
        verdict = "❌  NO EDGE — model does not beat 50/50 on this data"

    print(f"\n  Verdict: {verdict}")
    print(f"  Note: real edge also depends on p_market being stale (latency arb).")
    print(f"        Run python3 latency_test.py from AWS to verify the lag window.")

    return {
        "n":               n_pred,
        "win_rate":        round(wr, 4),
        "mean_p_model":    round(mean_pm, 4),
        "brier":           round(brier, 4),
        "edge_over_base":  round(edge_over_base, 4),
        "kelly_positive":  kelly_positive,
    }


# ─── sigma calibration table ───────────────────────────────────────────────────

def build_sigma_table(df: pd.DataFrame) -> dict:
    """
    Compute empirical σ of final %Δ from open as a function of:
      - vol_regime (LOW / NORMAL / HIGH based on rolling 20-candle vol)
      - time_remaining (proxied by relative position in the candle)

    This gives decision_stack.py a more accurate SIGMA_FLOOR_PCT per regime.
    """
    closes = df["close"].values
    opens  = df["open"].values
    n      = len(df)

    # Rolling 20-candle realised vol in %
    rvol = pd.Series(np.diff(np.log(closes))).rolling(20).std().values * 100

    results = {"LOW": [], "NORMAL": [], "HIGH": [], "EXTREME": []}
    split   = int(n * TRAIN_FRAC)

    for i in range(split + 20, n):
        rv = rvol[i - 1] if i - 1 < len(rvol) and np.isfinite(rvol[i - 1]) else None
        if rv is None:
            continue
        if rv < 0.10:
            regime = "LOW"
        elif rv < 0.25:
            regime = "NORMAL"
        elif rv < 0.50:
            regime = "HIGH"
        else:
            regime = "EXTREME"
        pct_change = (closes[i] - opens[i]) / opens[i] * 100
        results[regime].append(pct_change)

    table = {}
    print(f"\n  Empirical σ by vol regime (from OOS data):")
    for regime, vals in results.items():
        if len(vals) < 50:
            continue
        σ = np.std(vals)
        table[regime] = round(σ, 4)
        print(f"    {regime:<8}  n={len(vals):>6,}  σ={σ:.4f}%  "
              f"σ_ann={σ*np.sqrt(105120):.1f}%")

    return table


# ─── save / load ───────────────────────────────────────────────────────────────

def save_warmup(garch, hawkes, validation: dict, sigma_table: dict):
    state = {
        "garch": {
            "omega": garch.omega,
            "alpha": garch.alpha,
            "beta":  garch.beta,
            "h":     garch.h,
        },
        "hawkes": {
            "mu":    hawkes.mu,
            "alpha": hawkes.alpha,
            "beta":  hawkes.beta,
        },
        "validation":  validation,
        "sigma_table": sigma_table,
        "fitted_at":   time.time(),
    }
    with open(WARMUP_OUT, "wb") as f:
        pickle.dump(state, f)
    print(f"\n  Saved → {WARMUP_OUT}")
    return state


def load_warmup(garch, hawkes) -> bool:
    """Called from run_live.py at startup. Returns True if loaded OK."""
    if not os.path.exists(WARMUP_OUT):
        return False
    try:
        with open(WARMUP_OUT, "rb") as f:
            state = pickle.load(f)

        g  = state["garch"]
        hw = state["hawkes"]
        garch.omega  = g["omega"]
        garch.alpha  = g["alpha"]
        garch.beta   = g["beta"]
        garch.h      = g["h"]
        hawkes.mu    = hw["mu"]
        hawkes.alpha = hw["alpha"]
        hawkes.beta  = hw["beta"]

        age_h = (time.time() - state.get("fitted_at", 0)) / 3600
        v     = state.get("validation", {})
        st    = state.get("sigma_table", {})
        print(f"  [WARMUP] Loaded (fitted {age_h:.1f}h ago)")
        print(f"  [WARMUP] GARCH  ω={garch.omega:.3e}  "
              f"α={garch.alpha:.4f}  β={garch.beta:.4f}")
        print(f"  [WARMUP] Hawkes μ={hawkes.mu:.3f}  "
              f"branching={hawkes.branching_ratio:.3f}")
        if v:
            print(f"  [WARMUP] OOS WR={v.get('win_rate','?')}  "
                  f"edge={v.get('edge_over_base','?'):+}  "
                  f"Kelly+={v.get('kelly_positive','?')}")
        if st:
            print(f"  [WARMUP] σ table: "
                  + "  ".join(f"{k}={v:.4f}%" for k, v in st.items()))
        return True
    except Exception as e:
        print(f"  [WARMUP] Load failed: {e}")
        return False


# ─── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  WARMUP — GARCH + Hawkes pre-training")
    print("=" * 60)

    df = load_5min_kaggle()
    if df is None:
        print("  Kaggle CSV not found, trying fallback pkl...")
        df = load_1min_fallback()
    if df is None:
        print("  ERROR: no data source found. "
              "Run kaggle.py or ensure btc_1m_candles.pkl exists.")
        raise SystemExit(1)

    print()
    garch  = fit_garch(df)
    print()
    hawkes = fit_hawkes(df)

    sigma_table = build_sigma_table(df)
    validation  = validate(df, garch)

    save_warmup(garch, hawkes, validation, sigma_table)

    print()
    print("=" * 60)
    print("  Done. Start trading with:  python3 run_live.py")
    print("  Re-run warmup.py daily (or after major vol regime shift).")
    print("=" * 60)
