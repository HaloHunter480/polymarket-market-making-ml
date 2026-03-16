"""
train_probability_model.py — 5-Step Probability Model Training Pipeline
=======================================================================

Predicts P(BTC closes above strike at t+300s) from 1-second tick data.

STEP 1: Define prediction target
  For every time t: price_now = P(t), price_future = P(t+300s)
  Target: Y = 1 if price_future > strike else 0
  Strike = price at 5-min window start (t aligned to 300s boundaries)

STEP 2: Create state features (at time t, using ONLY past data)
  - MOMENTUM: ret_1s, ret_10s, ret_30s
  - VOLATILITY: realized vol over last 60s
  - ORDER-FLOW IMBALANCE: (buy_vol - sell_vol) / (buy_vol + sell_vol) over last 30s
  - SPREAD BEHAVIOUR: (high-low)/mid over last 10s (proxy when no L2)
  - DISTANCE TO STRIKE: (price_now - strike) / strike
  - TIME TO EXPIRY: seconds until window close (0-300)

STEP 3: Build training dataset
  - One row per (t, window) with valid future
  - Train/val/test chronological split (70/15/15)
  - No leakage: features use only t and earlier

STEP 4: Train probability model
  - Ensemble: P_final = 0.4*logistic + 0.3*boosting + 0.3*xgboost (all calibrated)
  - Output: P(UP | features)

STEP 5: Validate calibration
  - Reliability diagram (predicted vs actual by bucket)
  - Brier score
  - ECE (Expected Calibration Error)

Data: Place btc_1sec.csv or btc_1sec.csv.zip in project root.
      Or use --data path/to/file
      Format: timestamp,open,high,low,close,volume,taker_buy_volume
      (taker_buy_volume = buy volume for OFI; sell = volume - taker_buy)
"""

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np

# ── Data loading ─────────────────────────────────────────────────────────────

def find_data(path: str = None) -> str:
    """Find btc_1sec data: .csv, .csv.zip, or btc_1m_candles.pkl as fallback."""
    root = Path(__file__).parent
    candidates = [
        path,
        str(root / "BTC_1sec.csv"),
        str(root / "btc_1sec.csv"),
        str(root / "btc_1sec.csv.zip"),
        str(root / "data" / "btc_1sec.csv"),
        str(root / "BTCUSDT-1s-2024-01-01.csv"),
        str(root / "btc_1m_candles.pkl"),
    ]
    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            return p
    data_dir = root / "data"
    if data_dir.exists():
        for f in data_dir.iterdir():
            if "btc" in f.name.lower() and (f.suffix == ".csv" or f.suffix == ".zip"):
                return str(f)
    return ""


def load_btc_data(path: str, max_rows: int = 0) -> tuple:
    """
    Load BTC data. Returns (timestamps, opens, highs, lows, closes, volumes, taker_buys).
    Handles: BTC_1sec.csv (midpoint, spread, buys, sells), Binance CSV, .pkl.
    """
    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            csv_name = next((n for n in names if n.endswith(".csv")), names[0])
            with z.open(csv_name) as f:
                out = _parse_csv(f, path)
                return out if len(out) == 8 else out + (None,)
    elif path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            candles = pickle.load(f)
        out = _pkl_to_arrays(candles)
        return out if len(out) == 8 else out + (None,)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            out = _parse_csv(f, path, max_rows=max_rows)
            return out if len(out) == 8 else out + (None,)


def _parse_csv(f, path: str = "", max_rows: int = 0) -> tuple:
    """
    Parse CSV. Supports:
    - BTC_1sec.csv: system_time, midpoint, spread, buys, sells
    - Binance: ts,o,h,l,c,v,...,taker_buy
    """
    content = f.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
    if not lines:
        return (), (), (), (), (), (), ()

    header = lines[0].lower()
    has_header = "open" in header or "timestamp" in header or "midpoint" in header or "system_time" in header
    start = 1 if has_header else 0
    if max_rows > 0 and len(lines) > start + max_rows:
        lines = lines[: start + max_rows]

    if "midpoint" in header and "buys" in header:
        return _parse_btc_1sec(lines[start:], header)

    ts, o, h, l, c, v, taker = [], [], [], [], [], [], []
    for line in lines[start:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            t = int(float(parts[0]))
            op = float(parts[1])
            hi = float(parts[2])
            lo = float(parts[3])
            cl = float(parts[4])
            vol = float(parts[5]) if len(parts) > 5 else 0.0
            tb = float(parts[9]) if len(parts) > 9 else vol / 2
        except (ValueError, IndexError):
            continue
        ts.append(t)
        o.append(op)
        h.append(hi)
        l.append(lo)
        c.append(cl)
        v.append(vol)
        taker.append(tb)

    return (
        np.array(ts),
        np.array(o),
        np.array(h),
        np.array(l),
        np.array(c),
        np.array(v),
        np.array(taker),
        None,
    )


def _parse_btc_1sec(lines: list, header: str) -> tuple:
    """
    Parse BTC_1sec.csv: midpoint, spread, buys, sells.
    Returns (ts, o, h, l, c, v, taker, spreads) with:
    - c = midpoint (price)
    - h/l = midpoint ± spread/2 (proxy)
    - v = buys + sells, taker = buys
    - spreads = raw spread for spread behaviour feature
    """
    cols = [x.strip() for x in header.split(",")]
    idx_mid = cols.index("midpoint") if "midpoint" in cols else 2
    idx_spread = cols.index("spread") if "spread" in cols else 3
    idx_buys = cols.index("buys") if "buys" in cols else 4
    idx_sells = cols.index("sells") if "sells" in cols else 5

    ts, o, h, l, c, v, taker, spreads = [], [], [], [], [], [], [], []
    for i, line in enumerate(lines):
        parts = line.split(",")
        if len(parts) <= max(idx_mid, idx_spread, idx_buys, idx_sells):
            continue
        try:
            mid = float(parts[idx_mid])
            sp = float(parts[idx_spread])
            buys = float(parts[idx_buys])
            sells = float(parts[idx_sells])
        except (ValueError, IndexError):
            continue
        ts.append(i)
        o.append(mid)
        h.append(mid + sp / 2)
        l.append(mid - sp / 2)
        c.append(mid)
        v.append(buys + sells)
        taker.append(buys)
        spreads.append(sp)
    return (
        np.array(ts),
        np.array(o),
        np.array(h),
        np.array(l),
        np.array(c),
        np.array(v),
        np.array(taker),
        np.array(spreads),
    )


def _pkl_to_arrays(candles) -> tuple:
    """Returns 7-tuple; caller adds None for spreads."""
    """Convert 1m pickle to arrays. We treat each row as 1 'second' for structure (60x less data)."""
    if not candles:
        return (), (), (), (), (), (), ()
    if isinstance(candles[0], (list, tuple)):
        ts = np.array([int(c[0]) for c in candles])
        o = np.array([float(c[1]) for c in candles])
        h = np.array([float(c[2]) for c in candles])
        l = np.array([float(c[3]) for c in candles])
        c = np.array([float(c[4]) for c in candles])
        v = np.array([float(c[5]) if len(c) > 5 else 0 for c in candles])
        taker = np.array([float(c[9]) if len(c) > 9 else v[i]/2 for i, c in enumerate(candles)])
    else:
        ts = np.array([int(c["timestamp"]) for c in candles])
        o = np.array([float(c["open"]) for c in candles])
        h = np.array([float(c["high"]) for c in candles])
        l = np.array([float(c["low"]) for c in candles])
        c = np.array([float(c["close"]) for c in candles])
        v = np.array([float(c.get("volume", 0)) for c in candles])
        taker = np.array([float(c.get("taker_buy", v[i]/2)) for i, c in enumerate(candles)])
    return ts, o, h, l, c, v, taker  # 7 elements; load_btc_data adds None for spreads


# ── Step 1: Prediction target ───────────────────────────────────────────────────

WINDOW_S = 300  # seconds; for 1m data we use WINDOW_ROWS = 5


def build_targets(closes: np.ndarray, window_starts: np.ndarray) -> np.ndarray:
    """
    For each (t_idx, window_start_idx): Y = 1 if close[t+300] > strike else 0.
    strike = closes[window_start_idx]
    """
    n = len(closes)
    targets = []
    for ws in window_starts:
        if ws + WINDOW_S >= n:
            continue
        strike = closes[ws]
        future_close = closes[ws + WINDOW_S]
        targets.append(1.0 if future_close > strike else 0.0)
    return np.array(targets)


# ── Step 2: State features ──────────────────────────────────────────────────────

def compute_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    taker_buys: np.ndarray,
    window_starts: np.ndarray,
    t_indices: np.ndarray,
    window_rows: int = WINDOW_S,
    spreads: np.ndarray = None,
) -> np.ndarray:
    """
    At each (window_start, t): t is seconds into window. Features use only [window_start, t].
    Returns (n_samples, n_features).
    """
    n = len(closes)
    rows = []

    n = len(closes)
    for i, (ws, t_idx) in enumerate(zip(window_starts, t_indices)):
        if ws + window_rows > n or t_idx < ws or t_idx >= ws + window_rows:
            continue
        t_sec = t_idx - ws
        t_min_required = min(30, window_rows - 2)
        if t_sec < t_min_required:
            continue

        price = closes[t_idx]
        strike = closes[ws]

        lb1 = min(1, t_idx - ws)
        lb10 = min(10, t_idx - ws)
        lb30 = min(30, t_idx - ws)
        lb60 = min(60, t_idx - ws)
        lb120 = min(120, t_idx - ws)
        ret_1s = (closes[t_idx] / closes[t_idx - lb1] - 1) * 100 if lb1 > 0 else 0.0
        ret_10s = (closes[t_idx] / closes[t_idx - lb10] - 1) * 100 if lb10 > 0 else 0.0
        ret_30s = (closes[t_idx] / closes[t_idx - lb30] - 1) * 100 if lb30 > 0 else 0.0
        ret_60s = (closes[t_idx] / closes[t_idx - lb60] - 1) * 100 if lb60 > 0 else 0.0
        ret_120s = (closes[t_idx] / closes[t_idx - lb120] - 1) * 100 if lb120 > 0 else 0.0

        vol_window = min(60, window_rows - 1)
        start_vol = max(ws, t_idx - vol_window)
        rets = np.diff(np.log(closes[start_vol : t_idx + 1] + 1e-12))
        realized_vol = np.std(rets) * 100 if len(rets) > 5 else 0.05

        vol_short_window = min(10, window_rows - 1)
        start_vol_short = max(ws, t_idx - vol_short_window)
        rets_short = np.diff(np.log(closes[start_vol_short : t_idx + 1] + 1e-12))
        realized_vol_short = np.std(rets_short) * 100 if len(rets_short) > 1 else 0.05

        vol_30_window = min(30, window_rows - 1)
        start_vol_30 = max(ws, t_idx - vol_30_window)
        rets_30 = np.diff(np.log(closes[start_vol_30 : t_idx + 1] + 1e-12))
        vol_30s = np.std(rets_30) * 100 if len(rets_30) > 2 else 0.05

        vol_120_window = min(120, window_rows - 1)
        start_vol_120 = max(ws, t_idx - vol_120_window)
        rets_120 = np.diff(np.log(closes[start_vol_120 : t_idx + 1] + 1e-12))
        vol_120s = np.std(rets_120) * 100 if len(rets_120) > 5 else 0.05

        ofi_window = min(30, window_rows - 1)
        start_ofi = max(ws, t_idx - ofi_window)
        buy_vol = np.sum(taker_buys[start_ofi : t_idx + 1])
        sell_vol = np.sum(volumes[start_ofi : t_idx + 1]) - buy_vol
        total = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total if total > 0 else 0.0

        if spreads is not None and len(spreads) > t_idx:
            spread_window = min(10, window_rows - 1)
            start_sp = max(ws, t_idx - spread_window)
            sp_vals = spreads[start_sp : t_idx + 1]
            spread_pct = np.mean(sp_vals) / price * 100 if price > 0 and len(sp_vals) > 0 else 0.0
        else:
            spread_window = min(10, window_rows - 1)
            start_sp = max(ws, t_idx - spread_window)
            hh = np.max(highs[start_sp : t_idx + 1])
            ll = np.min(lows[start_sp : t_idx + 1])
            mid = (hh + ll) / 2
            spread_pct = (hh - ll) / mid * 100 if mid > 0 else 0.0

        dist_strike = (price - strike) / strike * 100 if strike > 0 else 0.0
        time_to_expiry = (window_rows - t_sec) * (300 / window_rows) if window_rows != WINDOW_S else (WINDOW_S - t_sec)

        rows.append([
            ret_1s, ret_10s, ret_30s, ret_60s, ret_120s,
            realized_vol,
            realized_vol_short,
            vol_30s,
            vol_120s,
            ofi,
            spread_pct,
            dist_strike,
            time_to_expiry,
        ])

    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 13))


# ── Step 3: Build dataset ──────────────────────────────────────────────────────

def build_dataset(
    ts, o, h, l, c, v, taker,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    is_1s_data: bool = None,
    spreads: np.ndarray = None,
):
    """Build X, y with chronological train/val/test split."""
    ts, o, h, l, c, v, taker = [np.asarray(x) for x in (ts, o, h, l, c, v, taker)]
    n = len(c)
    if spreads is not None:
        spreads = np.asarray(spreads)

    if is_1s_data is None:
        is_1s_data = n > 10000
    window_rows = WINDOW_S if is_1s_data else 5
    min_rows = window_rows * 2
    if n < min_rows:
        return None, None, None, None, None, None

    window_starts = np.arange(0, n - window_rows, window_rows)
    if len(window_starts) < 5:
        window_starts = np.arange(0, n - window_rows, max(1, (n - window_rows) // 10))

    all_X, all_y = [], []
    t_step = 10 if is_1s_data else 1
    t_min = 30 if is_1s_data else 3
    for ws in window_starts:
        for t_offset in range(t_min, window_rows, t_step):
            t_idx = ws + t_offset
            if t_idx >= ws + window_rows or ws + window_rows >= n:
                break
            X_row = compute_features(
                c, h, l, v, taker,
                np.array([ws]),
                np.array([t_idx]),
                window_rows=window_rows,
                spreads=spreads,
            )
            if len(X_row) == 0:
                continue
            strike = c[ws]
            future = c[ws + window_rows - 1]
            y = 1.0 if future > strike else 0.0
            all_X.append(X_row[0])
            all_y.append(y)

    X = np.array(all_X)
    y = np.array(all_y)
    if len(X) < 50:
        return None, None, None, None, None, None

    n_total = len(X)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Step 4: Train model ────────────────────────────────────────────────────────

def _calibrate(clf, X_val, y_val):
    """Wrap fitted classifier with isotonic calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    except ImportError:
        try:
            from sklearn.calibration import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        except ImportError:
            cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    return cal


def train_logistic(X_train, y_train, X_val, y_val, use_calibration: bool = True):
    """Train logistic regression with optional isotonic calibration."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    if use_calibration and len(X_val) >= 50:
        return _calibrate(clf, X_val, y_val)
    return clf


def train_boosting(X_train, y_train, X_val, y_val, use_calibration: bool = True):
    """Train HistGradientBoostingClassifier with optional isotonic calibration."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    if use_calibration and len(X_val) >= 50:
        return _calibrate(clf, X_val, y_val)
    return clf


def train_xgboost(X_train, y_train, X_val, y_val, use_calibration: bool = True):
    """Train XGBoost with isotonic calibration (XGBoost probs are poorly calibrated)."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    if use_calibration and len(X_val) >= 50:
        return _calibrate(clf, X_val, y_val)
    return clf


class EnsemblePredictor:
    """P_final = 0.4*logistic + 0.3*boosting + 0.3*xgboost"""

    WEIGHTS = (0.4, 0.3, 0.3)  # logistic, boosting, xgboost

    def __init__(self, logistic, boosting, xgboost):
        self.logistic = logistic
        self.boosting = boosting
        self.xgboost = xgboost

    def predict_proba(self, X):
        p_log = self.logistic.predict_proba(X)[:, 1]
        p_boost = self.boosting.predict_proba(X)[:, 1]
        p_xgb = self.xgboost.predict_proba(X)[:, 1]
        w = self.WEIGHTS
        p_final = w[0] * p_log + w[1] * p_boost + w[2] * p_xgb
        return np.column_stack([1 - p_final, p_final])


# ── Step 5: Calibration validation ─────────────────────────────────────────────

def reliability_diagram(y_true, y_pred, n_bins: int = 10):
    """Bucket predictions, compute mean predicted vs mean actual."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    result = []
    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            result.append({"bin": i, "mid": (bins[i] + bins[i + 1]) / 2, "mean_pred": np.nan, "mean_actual": np.nan, "count": 0})
            continue
        mean_pred = np.mean(y_pred[mask])
        mean_actual = np.mean(y_true[mask])
        result.append({
            "bin": i,
            "mid": (bins[i] + bins[i + 1]) / 2,
            "mean_pred": mean_pred,
            "mean_actual": mean_actual,
            "count": int(np.sum(mask)),
        })
    return result


def brier_score(y_true, y_pred) -> float:
    return np.mean((y_pred - y_true) ** 2)


def expected_calibration_error(y_true, y_pred, n_bins: int = 10) -> float:
    """ECE = sum (|B_m|/n) * |acc(B_m) - conf(B_m)|"""
    rd = reliability_diagram(y_true, y_pred, n_bins)
    n = len(y_true)
    ece = 0.0
    for r in rd:
        if r["count"] == 0:
            continue
        acc = r["mean_actual"]
        conf = r["mean_pred"]
        ece += (r["count"] / n) * abs(acc - conf)
    return ece


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 5-min binary probability model")
    parser.add_argument("--data", type=str, default="", help="Path to btc_1sec.csv or .zip")
    parser.add_argument("--out", type=str, default="models/probability_model.pkl", help="Output model path")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to load (0=all). Use for large files.")
    parser.add_argument("--no-calibration", action="store_true", help="Skip Platt/isotonic calibration")
    args = parser.parse_args()

    path = find_data(args.data)
    if not path:
        print("  [!] No data found. Place btc_1sec.csv or btc_1sec.csv.zip in project root.")
        print("      Download 1s data: python download_btc_1s.py --days 7 --merge")
        print("      Or: python train_probability_model.py --data /path/to/btc_1sec.csv")
        print("      Fallback: btc_1m_candles.pkl (1m data, fewer samples)")
        return 1

    print(f"  [1] Loading data from {path}")
    out = load_btc_data(path, max_rows=args.max_rows)
    ts, o, h, l, c, v, taker = out[:7]
    spreads = out[7] if len(out) > 7 else None
    if len(c) < 1000:
        print("  [!] Insufficient data (need 1000+ rows)")
        return 1
    print(f"      Loaded {len(c):,} rows")

    print("  [2] Building features and targets")
    data = build_dataset(ts, o, h, l, c, v, taker, spreads=spreads)
    if data[0] is None:
        print("  [!] Could not build dataset")
        return 1
    X_train, y_train, X_val, y_val, X_test, y_test = data
    print(f"      Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    print("  [3] Training ensemble (logistic + boosting + xgboost)")
    use_cal = not args.no_calibration
    print("      Training logistic...")
    m_log = train_logistic(X_train, y_train, X_val, y_val, use_calibration=use_cal)
    print("      Training HistGradientBoosting...")
    m_boost = train_boosting(X_train, y_train, X_val, y_val, use_calibration=use_cal)
    print("      Training XGBoost...")
    m_xgb = train_xgboost(X_train, y_train, X_val, y_val, use_calibration=use_cal)
    model = EnsemblePredictor(m_log, m_boost, m_xgb)

    print("  [4] Evaluating on test set")
    y_pred = model.predict_proba(X_test)[:, 1]
    brier = brier_score(y_test, y_pred)
    ece = expected_calibration_error(y_test, y_pred)
    acc = np.mean((y_pred >= 0.5) == y_test)
    print(f"      Accuracy: {acc:.2%}")
    print(f"      Brier:    {brier:.4f} (lower is better)")
    print(f"      ECE:      {ece:.4f} (lower = better calibration)")

    print("  [5] Reliability diagram (calibration)")
    rd = reliability_diagram(y_test, y_pred)
    for r in rd:
        if r["count"] > 0:
            print(f"      Bin {r['bin']}: pred={r['mean_pred']:.3f} actual={r['mean_actual']:.3f} n={r['count']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import pickle
    with open(args.out, "wb") as f:
        pickle.dump({
            "model": model,
            "weights": {"logistic": 0.4, "boosting": 0.3, "xgboost": 0.3},
            "feature_names": [
                "ret_1s", "ret_10s", "ret_30s", "ret_60s", "ret_120s",
                "volatility", "volatility_short", "vol_30s", "vol_120s",
                "ofi", "spread_pct", "dist_strike", "time_to_expiry",
            ],
        }, f)
    print(f"  [OK] Model saved to {args.out}")
    return 0


if __name__ == "__main__":
    exit(main())
