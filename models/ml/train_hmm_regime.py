"""
train_hmm_regime.py — Train HMM Regime Classifier on 1s Features
================================================================

Builds features from BTC_1sec.csv and trains a Gaussian HMM for regime detection.
Features: returns_1s, returns_10s, realized_vol_30s, realized_vol_120s, OFI, spread, etc.

Output: models/hmm_regime.pkl (model, scaler, feature_names, state_labels)

Usage:
    python3 train_hmm_regime.py --data BTC_1sec.csv
    python3 train_hmm_regime.py --data BTC_1sec.csv --max-rows 100000
"""

import argparse
import os
from pathlib import Path

import numpy as np

# Reuse data loading from train_probability_model
from train_probability_model import find_data, load_btc_data

# ── Feature names (must match compute_hmm_features) ─────────────────────────────
HMM_FEATURE_NAMES = [
    "returns_1s",
    "returns_10s",
    "returns_30s",
    "returns_60s",
    "realized_vol_30s",
    "realized_vol_120s",
    "order_flow_imbalance",
    "spread_pct",
    "volatility_short",  # 10s vol
]
MIN_LOOKBACK = 120  # need 120 rows for vol_120s


def compute_hmm_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    taker_buys: np.ndarray,
    spreads: np.ndarray = None,
) -> np.ndarray:
    """
    Compute HMM features for each row i (i >= MIN_LOOKBACK).
    Returns (n_rows, n_features). Uses only past data [i-lookback, i].
    """
    n = len(closes)
    rows = []
    for i in range(MIN_LOOKBACK, n):
        price = closes[i]

        # Returns (%)
        ret_1s = (closes[i] / closes[i - 1] - 1) * 100 if i >= 1 else 0.0
        ret_10s = (closes[i] / closes[i - 10] - 1) * 100 if i >= 10 else 0.0
        ret_30s = (closes[i] / closes[i - 30] - 1) * 100 if i >= 30 else 0.0
        ret_60s = (closes[i] / closes[i - 60] - 1) * 100 if i >= 60 else 0.0

        # Realized vol 30s
        start_30 = max(0, i - 30)
        rets_30 = np.diff(np.log(closes[start_30 : i + 1] + 1e-12))
        vol_30s = np.std(rets_30) * 100 if len(rets_30) > 2 else 0.05

        # Realized vol 120s
        start_120 = max(0, i - 120)
        rets_120 = np.diff(np.log(closes[start_120 : i + 1] + 1e-12))
        vol_120s = np.std(rets_120) * 100 if len(rets_120) > 5 else 0.05

        # Volatility short (10s)
        start_10 = max(0, i - 10)
        rets_10 = np.diff(np.log(closes[start_10 : i + 1] + 1e-12))
        vol_short = np.std(rets_10) * 100 if len(rets_10) > 1 else 0.05

        # Order flow imbalance (30s window)
        start_ofi = max(0, i - 30)
        buy_vol = np.sum(taker_buys[start_ofi : i + 1])
        sell_vol = np.sum(volumes[start_ofi : i + 1]) - buy_vol
        total = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total if total > 0 else 0.0

        # Spread (% of price)
        if spreads is not None and len(spreads) > i:
            sp_window = spreads[max(0, i - 10) : i + 1]
            spread_pct = np.mean(sp_window) / price * 100 if price > 0 and len(sp_window) > 0 else 0.0
        else:
            sp_window = max(0, i - 10)
            hh = np.max(highs[sp_window : i + 1])
            ll = np.min(lows[sp_window : i + 1])
            mid = (hh + ll) / 2
            spread_pct = (hh - ll) / mid * 100 if mid > 0 else 0.0

        rows.append([
            ret_1s, ret_10s, ret_30s, ret_60s,
            vol_30s, vol_120s,
            ofi,
            spread_pct,
            vol_short,
        ])

    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, len(HMM_FEATURE_NAMES)))


def _assign_state_labels(model, n_components: int) -> list:
    """Map HMM states to regime names by mean return and volatility."""
    means = model.means_
    ret_col = 0
    vol_col = min(4, means.shape[1] - 1)
    rets = np.asarray(means[:, ret_col]).flatten()
    cov = model.covars_
    vols = np.array([
        float(np.sqrt(np.maximum(np.ravel(cov[i])[vol_col], 1e-12)))
        for i in range(n_components)
    ])
    med_vol = float(np.median(vols))
    # Sort by (vol desc, then ret) to assign: high_vol first, then trending_up/down, then low_vol
    state_labels = [""] * n_components
    for i in range(n_components):
        r, v = float(rets[i]), float(vols[i])
        if v > med_vol * 1.5:
            state_labels[i] = "high_vol"
        elif r > 0.02:
            state_labels[i] = "trending_up"
        elif r < -0.02:
            state_labels[i] = "trending_down"
        else:
            state_labels[i] = "low_vol"
    return state_labels


def main():
    parser = argparse.ArgumentParser(description="Train HMM regime classifier on 1s features")
    parser.add_argument("--data", type=str, default="", help="Path to BTC_1sec.csv")
    parser.add_argument("--out", type=str, default="models/hmm_regime.pkl", help="Output model path")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to load (0=all)")
    parser.add_argument("--n-components", type=int, default=4, help="HMM states")
    args = parser.parse_args()

    path = find_data(args.data)
    if not path:
        print("  [!] No data found. Place BTC_1sec.csv in project root.")
        return 1

    print(f"  [1] Loading data from {path}")
    out = load_btc_data(path, max_rows=args.max_rows)
    ts, o, h, l, c, v, taker = out[:7]
    spreads = out[7] if len(out) > 7 else None
    if len(c) < MIN_LOOKBACK + 100:
        print(f"  [!] Need at least {MIN_LOOKBACK + 100} rows (got {len(c)})")
        return 1
    print(f"      Loaded {len(c):,} rows")

    print("  [2] Computing HMM features")
    X = compute_hmm_features(c, h, l, v, taker, spreads)
    print(f"      Features: {X.shape[0]:,} rows × {X.shape[1]} cols")
    print(f"      Names: {HMM_FEATURE_NAMES}")

    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip extremes
    X = np.clip(X, -50, 50)

    print("  [3] Standardizing and training HMM")
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=args.n_components,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
    )
    model.fit(X_scaled)
    state_labels = _assign_state_labels(model, args.n_components)
    print(f"      State labels: {state_labels}")

    # Quick validation: predict and show regime distribution
    hidden = model.predict(X_scaled)
    from collections import Counter
    counts = Counter(hidden)
    print("  [4] Regime distribution (train)")
    for i in range(args.n_components):
        pct = 100 * counts.get(i, 0) / len(hidden)
        print(f"      {state_labels[i]:12s}: {counts.get(i, 0):>8,} ({pct:.1f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import pickle
    with open(args.out, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "feature_names": HMM_FEATURE_NAMES,
            "state_labels": state_labels,
            "n_components": args.n_components,
        }, f)
    print(f"  [OK] Model saved to {args.out}")
    return 0


if __name__ == "__main__":
    exit(main())
