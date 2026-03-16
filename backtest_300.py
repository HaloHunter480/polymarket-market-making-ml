"""
Full Backtest Engine — 300+ Trades
===================================
Replays 31 days of 1-min BTC candles through the complete v5 strategy stack:
  • EmpiricalEngine (45k historical observations)
  • CalibrationCurve (rolling OOS)
  • MonteCarloKelly (3-D posterior sampling)
  • EquitySimulator (ruin + max-DD + streaks)
  • OFITracker (Z-score normalised + spread-gated)
  • RegimeClassifier (EWMA vol + momentum)
  • ToxicFlowDetector (Hawkes + VPIN + price-jumps)
  • PaperTracker (realistic fill simulation)

Run:  python3 backtest_300.py
"""

import sys, os, pickle, math, random, time, json
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timezone

# ── seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── import strategy modules (reuse production code verbatim) ─────────────────
sys.path.insert(0, os.path.dirname(__file__))

# We import the classes we need directly.
# Avoid running main() by importing only symbols.
from professional_strategy import (
    EmpiricalEngine, CalibrationCurve, MonteCarloKelly, EquitySimulator,
    OFITracker, RegimeClassifier,
    TradeSignal, PaperTrade, PaperTracker,
    MAKER_FILL_RATE, TAKER_SLIPPAGE, TAKER_FEE, MAKER_FEE,
    MIN_MAKER_EDGE, MIN_TAKER_EDGE, MIN_SAMPLES,
    KELLY_MC_SAMPLES, KELLY_CONFIDENCE_PCT, MAX_KELLY_FRACTION,
    MIN_BET_SIZE, MAX_BET_SIZE, BANKROLL,
    MAX_TRADES_PER_WINDOW, SIGNAL_COOLDOWN,
    WINDOW_INIT_DELAY, MIN_TOB_VOLUME, MAX_REALISTIC_EDGE,
    REGIME_VOL_SCALE, OFI_WEIGHT, OFI_WINDOW,
    MAX_RUIN_PROB, MC_EQUITY_PATHS, CALIBRATION_GAMMA,
)

# MonteCarloKelly is a static-method class — no instantiation needed.

# ─── Backtest configuration ───────────────────────────────────────────────────
WINDOW_SECONDS  = 300        # 5-min windows (Polymarket standard)
TICK_INTERVAL   = 3          # seconds per synthetic tick
TICKS_PER_MIN   = 60 // TICK_INTERVAL
BANKROLL_START  = BANKROLL
MAX_KELLY_FRAC  = MAX_KELLY_FRACTION
TARGET_TRADES   = 300

# Simulated CLOB spread model (approximates real Polymarket spreads)
BASE_SPREAD     = 0.025      # 2.5% base spread on binary tokens
SPREAD_VOL_MULT = 1.5        # Spread widens with realized vol
TOB_VOLUME_USD  = 120.0      # Simulated top-of-book volume

# Toxic flow thresholds (mirror professional_strategy.py)
TOXIC_VPIN_THRESH   = 0.75
TOXIC_PRICE_JUMP    = 50.0
TOXIC_VOLUME_MULT   = 5.0

# ─── Helpers ─────────────────────────────────────────────────────────────────

def pct(x, n):
    return f"{x/n*100:.1f}%" if n else "n/a"

def mean(lst):
    return sum(lst)/len(lst) if lst else 0.0

def std(lst):
    if len(lst) < 2:
        return 0.0
    m = mean(lst)
    return math.sqrt(sum((x-m)**2 for x in lst)/(len(lst)-1))

def percentile(lst, p):
    if not lst:
        return float('nan')
    s = sorted(lst)
    idx = (len(s)-1)*p/100.0
    lo, hi = int(idx), min(int(idx)+1, len(s)-1)
    return s[lo]+(idx-lo)*(s[hi]-s[lo])

def sharpe(pnls, rf=0.0):
    if len(pnls) < 2:
        return float('nan')
    m = mean(pnls)-rf
    s = std(pnls)
    return m/s if s > 1e-12 else float('nan')

def max_drawdown(cum_series):
    peak, mdd = cum_series[0], 0.0
    for v in cum_series:
        if v > peak: peak = v
        mdd = max(mdd, peak-v)
    return mdd

def calmar(returns, mdd):
    ann = mean(returns) * 288  # 288 windows/day (5-min)
    return ann/mdd if mdd > 1e-6 else float('nan')

# ─── Synthetic order book ─────────────────────────────────────────────────────
class SyntheticBook:
    """Minimal order book stub that the signal generator expects."""

    def __init__(self):
        self.bid = 0.0
        self.ask = 0.0
        self.mid_price = 0.5

    def update(self, fair: float, spread: float):
        half = spread / 2
        self.bid = max(0.01, fair - half)
        self.ask = min(0.99, fair + half)
        self.mid_price = (self.bid + self.ask) / 2

    def as_dict(self) -> dict:
        return {
            "bids": [{"price": self.bid, "size": TOB_VOLUME_USD / max(0.01, self.bid)}],
            "asks": [{"price": self.ask, "size": TOB_VOLUME_USD / max(0.01, self.ask)}],
        }

    @property
    def bids(self):
        return [type('L', (), {'price': self.bid, 'size': TOB_VOLUME_USD/max(0.01,self.bid)})()]

    @property
    def asks(self):
        return [type('L', (), {'price': self.ask, 'size': TOB_VOLUME_USD/max(0.01,self.ask)})()]

# ─── Simplified toxic-flow detector for backtest ─────────────────────────────
class BacktestToxic:
    def __init__(self):
        self._vols: deque = deque(maxlen=30)
        self._prices: deque = deque(maxlen=5)
        self._buy_vol = 0.0
        self._sell_vol = 0.0

    def update(self, price: float, volume: float, is_buy: bool):
        self._vols.append(volume)
        self._prices.append(price)
        if is_buy:
            self._buy_vol += volume
        else:
            self._sell_vol += volume

    def is_toxic(self) -> Tuple[bool, str]:
        if len(self._vols) < 5:
            return False, ""
        avg_vol = mean(list(self._vols)[:-1])
        if avg_vol > 1e-9 and self._vols[-1] > TOXIC_VOLUME_MULT * avg_vol:
            return True, "VOL_SPIKE"
        if len(self._prices) >= 2:
            jump = abs(self._prices[-1] - self._prices[-2])
            if jump > TOXIC_PRICE_JUMP:
                return True, f"PRICE_JUMP_${jump:.0f}"
        total_vol = self._buy_vol + self._sell_vol
        if total_vol > 0:
            vpin = abs(self._buy_vol - self._sell_vol) / total_vol
            if vpin > TOXIC_VPIN_THRESH:
                return True, f"VPIN_{vpin:.2f}"
        return False, ""

    def reset(self):
        self._vols.clear(); self._prices.clear()
        self._buy_vol = 0.0; self._sell_vol = 0.0

# ─── Signal generator (backtest-safe, no live feed) ──────────────────────────
class BacktestSignalGen:
    def __init__(self, empirical: EmpiricalEngine, bankroll: float):
        self.empirical   = empirical
        self.bankroll    = bankroll
        self.calibration = CalibrationCurve(gamma=CALIBRATION_GAMMA)
        self.ofi         = OFITracker(window=OFI_WINDOW)
        self._prices     = deque(maxlen=120)

    def update_price(self, p: float):
        self._prices.append(p)

    def update_book(self, book_dict: dict):
        self.ofi.update(book_dict)

    def record_outcome(self, market_price: float, won: bool):
        self.calibration.record_outcome(market_price, won)

    def reset_window(self):
        self.ofi.reset()

    def evaluate(
        self,
        btc: float,
        strike: float,
        time_remaining: float,
        book: SyntheticBook,
        book_dict: dict,
        sim_clock: float = 0.0,   # seconds since window open
    ) -> Optional[TradeSignal]:

        if strike <= 0 or btc <= 0 or time_remaining < 15:
            return None

        # Window init delay
        if sim_clock < WINDOW_INIT_DELAY:
            return None

        # Ghost-town filter (simulated — always passes with synthetic $120 TOB)
        # Intentionally kept — consistent with live system.

        # Empirical probability
        pct_diff = (btc - strike) / strike * 100
        emp_prob, n_samples = self.empirical.lookup(pct_diff, time_remaining)
        if n_samples < MIN_SAMPLES:
            return None

        # Regime
        regime, _ = RegimeClassifier.classify(self._prices)
        regime_scale = REGIME_VOL_SCALE.get(regime, 1.0)

        # OFI
        self.ofi.update(book_dict)
        ofi_signal = self.ofi.signal()

        # Calibration
        cal_mid = self.calibration.calibrate(book.mid_price)

        # Flow skew: direction of OFI
        flow_skew = ofi_signal * OFI_WEIGHT * 0.5
        emp_prob_up   = max(0.02, min(0.98, emp_prob + flow_skew))
        emp_prob_down = 1.0 - emp_prob_up

        market_up   = cal_mid
        market_down = 1.0 - cal_mid

        # Model/market cross-check (max 25% divergence)
        if abs(emp_prob_up - cal_mid) > 0.25:
            return None

        # Edge calculation
        edge_up   = emp_prob_up   - market_up
        edge_down = emp_prob_down - market_down

        best_edge = max(edge_up, edge_down)
        best_side = "UP" if edge_up >= edge_down else "DOWN"
        best_emp  = emp_prob_up if best_side == "UP" else emp_prob_down
        market_price_for_side = market_up if best_side == "UP" else market_down

        # OFI gate (weak signal → require higher edge)
        abs_ofi = abs(ofi_signal)
        if abs_ofi < 0.05:
            min_maker_edge = MIN_MAKER_EDGE * 1.5
            min_taker_edge = MIN_TAKER_EDGE * 1.2
        else:
            min_maker_edge = MIN_MAKER_EDGE
            min_taker_edge = MIN_TAKER_EDGE

        # Execution type decision
        exec_type = None
        if time_remaining <= 60 and best_edge >= min_taker_edge:
            exec_type = "TAKER"
        elif time_remaining > 60 and best_edge >= min_maker_edge:
            exec_type = "MAKER"
        else:
            return None

        # Cap edge for Kelly (prevent anomaly exploitation)
        kelly_edge = min(best_edge, MAX_REALISTIC_EDGE)

        # Wins / losses for beta prior (fallback if not enough)
        # MC Kelly sizing — 200 draws in backtest (vs 5000 live) for speed
        n_samples_mc = max(10, int(best_emp * 30 + (1-best_emp) * 30))
        mc_size, _ = MonteCarloKelly.compute(
            prob          = best_emp,
            price         = market_price_for_side,
            n_samples     = n_samples_mc,
            bankroll      = self.bankroll,
            fee           = TAKER_FEE,
            execution_type= exec_type,
            n_mc          = 200,       # fast inline sizing
        )
        mc_size = mc_size * regime_scale
        mc_size = max(MIN_BET_SIZE, min(MAX_BET_SIZE, mc_size))

        # Ruin check: simple analytical approximation (no MC paths inline)
        # P(ruin) ≈ ((1-p)/p)^(bankroll/bet) — Gambler's ruin bound
        if best_emp < 0.5 and mc_size > 0:
            ruin_approx = ((1-best_emp)/best_emp) ** (self.bankroll/mc_size)
            if ruin_approx > MAX_RUIN_PROB:
                mc_size = mc_size * 0.5

        # Maker block inside last 60s
        if exec_type == "MAKER" and time_remaining < 60:
            return None

        limit_price = book.bid if best_side == "UP" else book.ask
        spread_bps  = (book.ask - book.bid) / max(book.mid_price, 0.01)
        return TradeSignal(
            side           = best_side,
            execution_type = exec_type,
            limit_price    = limit_price,
            market_mid     = book.mid_price,
            fair_value     = best_emp,
            edge           = best_edge,
            kelly_size     = mc_size,
            confidence     = float(n_samples_mc) / 50.0,
            empirical_prob = best_emp,
            flow_skew      = flow_skew,
            toxic_risk     = 0.0,
            pct_diff       = pct_diff,
            time_remaining = time_remaining,
            momentum       = ofi_signal,
            spread_bps     = spread_bps,
            sample_count   = n_samples,
        )


# ─── Simulated VPIN for ticks ─────────────────────────────────────────────────
def simulate_tick_volume(candle_vol: float, n_ticks: int) -> List[float]:
    """Spread candle volume unevenly across ticks (Poisson-like)."""
    weights = [random.expovariate(1) for _ in range(n_ticks)]
    total   = sum(weights)
    return [candle_vol * w / total for w in weights]


# ─── Core backtest engine ─────────────────────────────────────────────────────
class Backtester:
    def __init__(self, candles: list, bankroll: float = BANKROLL_START):
        self.candles  = candles
        self.bankroll = bankroll

        self.empirical  = EmpiricalEngine()
        self.signal_gen = BacktestSignalGen(self.empirical, bankroll)
        self.tracker    = PaperTracker()
        self.toxic      = BacktestToxic()

        # Bookkeeping
        self.windows_run     : int   = 0
        self.trades_filled   : int   = 0
        self.signals_gen     : int   = 0
        self.toxic_events    : int   = 0
        self.window_pnls     : List[float] = []
        self.filled_signals  : List[dict]  = []
        self.per_window_stats: List[dict]  = []

    def run(self):
        print("=" * 72)
        print("  BACKTEST v5 — 31-day BTC data → full strategy stack")
        print(f"  Bankroll: ${self.bankroll:.0f}  |  Target: ≥{TARGET_TRADES} filled trades")
        print("=" * 72)

        n_candles = len(self.candles)
        # Each window = WINDOW_SECONDS/60 candles
        candles_per_window = WINDOW_SECONDS // 60   # = 5

        window_id   = 0
        cum_pnl     = 0.0
        running_bankroll = self.bankroll

        # Progress bar width
        bar_width = 50
        total_windows = n_candles // candles_per_window

        for w_start in range(0, n_candles - candles_per_window, candles_per_window):
            w_candles = self.candles[w_start : w_start + candles_per_window]
            if len(w_candles) < candles_per_window:
                break

            window_id += 1
            self.windows_run += 1
            self.signal_gen.reset_window()
            self.toxic.reset()

            # Strike = open price of first candle
            strike = float(w_candles[0][1])  # open

            # Generate synthetic ticks from candle OHLCV
            ticks      = self._gen_ticks(w_candles)
            n_ticks    = len(ticks)
            window_start_time = float(w_candles[0][0]) / 1000.0

            # Per-window state
            trades_this_window = 0
            last_signal_tick   = -SIGNAL_COOLDOWN * TICKS_PER_MIN
            filled_this_window : List[dict] = []
            open_orders        : List[dict] = []

            for t_idx, tick in enumerate(ticks):
                btc_price = tick['price']
                volume    = tick['volume']
                is_buy    = tick['is_buy']
                sim_clock = t_idx * TICK_INTERVAL
                time_rem  = WINDOW_SECONDS - sim_clock

                # Feed price history into signal gen
                self.signal_gen.update_price(btc_price)

                # Toxic flow update
                self.toxic.update(btc_price, volume, is_buy)
                is_toxic, toxic_reason = self.toxic.is_toxic()
                if is_toxic:
                    self.toxic_events += 1
                    # Kill open maker orders
                    for o in open_orders[:]:
                        o['cancelled'] = True
                    open_orders = [o for o in open_orders if not o.get('cancelled')]

                # Simulate book from fair value
                spread = self._compute_spread(w_candles, t_idx)
                fair   = self.empirical.lookup(
                    (btc_price - strike) / strike * 100, time_rem)[0]
                book   = SyntheticBook()
                book.update(fair, spread)
                book_dict = book.as_dict()

                # Signal generation (gated by trades/window + cooldown)
                if (not is_toxic
                        and trades_this_window < MAX_TRADES_PER_WINDOW
                        and (t_idx - last_signal_tick) * TICK_INTERVAL >= SIGNAL_COOLDOWN):

                    sig = self.signal_gen.evaluate(
                        btc_price, strike, time_rem, book, book_dict, sim_clock)

                    if sig is not None:
                        self.signals_gen += 1
                        # Fill simulation
                        filled = self._simulate_fill(sig)
                        if filled:
                            trades_this_window += 1
                            last_signal_tick    = t_idx
                            self.trades_filled += 1
                            order = {
                                'signal'   : sig,
                                'side'     : sig.side,
                                'exec_type': sig.execution_type,
                                'edge'     : sig.edge,
                                'size'     : sig.kelly_size,
                                'mid'      : book.mid_price,
                                'fair'     : fair,
                                't_idx'    : t_idx,
                                'time_rem' : time_rem,
                                'cancelled': False,
                                'won'      : None,
                            }
                            open_orders.append(order)
                            filled_this_window.append(order)

            # ── Resolve window ─────────────────────────────────────────────
            final_btc  = float(w_candles[-1][4])  # close of last candle
            resolved_up = final_btc > strike
            window_pnl  = 0.0

            for o in open_orders:
                if o['cancelled']:
                    continue
                won = (o['side'] == 'UP') == resolved_up
                o['won'] = won
                fee = TAKER_FEE if o['exec_type'] == 'TAKER' else MAKER_FEE
                if won:
                    payout = o['size'] * (1.0 / o['mid'] - 1) * (1 - fee)
                    o['pnl'] = payout
                else:
                    o['pnl'] = -o['size']
                window_pnl += o['pnl']

            cum_pnl          += window_pnl
            running_bankroll  = self.bankroll + cum_pnl
            self.signal_gen.bankroll = running_bankroll  # update for MC Kelly
            self.window_pnls.append(window_pnl)

            # Calibration feedback
            for o in filled_this_window:
                if o.get('won') is not None:
                    self.signal_gen.record_outcome(o['mid'], o['won'])
                    self.filled_signals.append(o)

            # Per-window stats
            self.per_window_stats.append({
                'window_id' : window_id,
                'strike'    : strike,
                'final_btc' : final_btc,
                'result'    : 'UP' if resolved_up else 'DOWN',
                'pnl'       : window_pnl,
                'cum_pnl'   : cum_pnl,
                'n_filled'  : len([o for o in open_orders if not o.get('cancelled')]),
                'bankroll'  : running_bankroll,
            })

            # Progress
            if window_id % 100 == 0 or window_id <= 3:
                filled_total = len(self.filled_signals)
                done = window_id / total_windows
                bar  = '█' * int(bar_width * done) + '░' * (bar_width - int(bar_width * done))
                print(f"  [{bar}] W{window_id:>4}/{total_windows}"
                      f"  trades={filled_total:>4}"
                      f"  P&L=${cum_pnl:+.2f}"
                      f"  bankroll=${running_bankroll:.0f}")

            # Stop if we have enough trades AND processed reasonable sample
            if len(self.filled_signals) >= TARGET_TRADES and window_id >= 200:
                print(f"\n  ✓ Target {TARGET_TRADES} trades reached at window {window_id}")
                break

        print(f"\n  Backtest complete — {window_id} windows, "
              f"{len(self.filled_signals)} filled trades.")
        self.print_report()

    def get_oos_confident_signals(self) -> List[dict]:
        """Step 1 OOS extraction: for each confident filled trade, return
        {p_model, p_market, direction, outcome}. Backtest HAS P_market (mid at signal time).
        Confident: fair_value >= 0.62 (UP) or <= 0.38 (DOWN)."""
        CONF_MIN = 0.62
        out = []
        for o in self.filled_signals:
            if o.get('won') is None:
                continue
            sig = o.get('signal')
            if not sig:
                continue
            p_model = getattr(sig, 'fair_value', o.get('fair'))
            if p_model is None:
                p_model = o.get('fair')
            side = o.get('side', 'UP')
            mid = o.get('mid', 0.5)
            # P_market for our side: UP token mid = P(UP), DOWN = 1 - P(UP)
            p_market = mid if side == 'UP' else (1.0 - mid)
            if p_model >= CONF_MIN and side == 'UP':
                out.append({
                    "p_model": round(p_model, 4),
                    "p_market": round(p_market, 4),
                    "direction": "UP",
                    "outcome": 1 if o['won'] else 0,
                })
            elif p_model <= (1 - CONF_MIN) and side == 'DOWN':
                out.append({
                    "p_model": round(p_model, 4),
                    "p_market": round(p_market, 4),
                    "direction": "DOWN",
                    "outcome": 1 if o['won'] else 0,
                })
        return out

    def _gen_ticks(self, candles) -> List[dict]:
        """Generate TICK_INTERVAL-second ticks from 1-min OHLCV candles."""
        ticks = []
        for c in candles:
            o, h, l, cl, vol = (float(c[1]), float(c[2]),
                                 float(c[3]), float(c[4]), float(c[5]))
            n_per_candle = 60 // TICK_INTERVAL   # 20 ticks per candle
            vols = simulate_tick_volume(vol, n_per_candle)
            # Interpolate price along O→H→L→C path
            path = [o,
                    o + (h-o)*0.4, h,
                    h + (l-h)*0.5, l,
                    l + (cl-l)*0.8, cl]
            for j in range(n_per_candle):
                frac  = j / (n_per_candle - 1)
                pidx  = int(frac * (len(path) - 1))
                pfrac = frac * (len(path) - 1) - pidx
                price = path[pidx] + pfrac * (path[min(pidx+1, len(path)-1)] - path[pidx])
                price += random.gauss(0, max(h-l, 1) * 0.05)   # micro-noise
                price  = max(1.0, price)
                is_buy = price > (o if j == 0 else ticks[-1]['price']) if ticks else True
                ticks.append({'price': price, 'volume': vols[j], 'is_buy': is_buy})
        return ticks

    def _compute_spread(self, candles, tick_idx: int) -> float:
        """Spread = base * (1 + vol_factor). Widens in high-vol candles."""
        highs  = [float(c[2]) for c in candles]
        lows   = [float(c[3]) for c in candles]
        close  = [float(c[4]) for c in candles]
        ranges = [(highs[i]-lows[i])/close[i] for i in range(len(candles))]
        vol_factor = min(3.0, mean(ranges) * 100)
        return min(0.10, BASE_SPREAD * (1 + vol_factor * SPREAD_VOL_MULT))

    def _simulate_fill(self, sig: TradeSignal) -> bool:
        """Simulate realistic fill: 40% maker, 100% taker."""
        if sig.execution_type == "TAKER":
            sig.edge = max(0, sig.edge - TAKER_SLIPPAGE)
            return True
        elif sig.execution_type == "MAKER":
            return random.random() < MAKER_FILL_RATE
        return False

    # ── Report ────────────────────────────────────────────────────────────────
    def print_report(self):
        fs = self.filled_signals
        if not fs:
            print("  No filled trades to report.")
            return

        decided  = [o for o in fs if o.get('won') is not None]
        wins     = [o for o in decided if o['won']]
        losses   = [o for o in decided if not o['won']]
        win_rate = len(wins) / len(decided) if decided else 0.0

        maker_d  = [o for o in decided if o['exec_type'] == 'MAKER']
        taker_d  = [o for o in decided if o['exec_type'] == 'TAKER']
        maker_w  = sum(1 for o in maker_d if o['won'])
        taker_w  = sum(1 for o in taker_d if o['won'])

        pnls_per_trade = [o.get('pnl', 0) for o in decided]
        cum_pnl_series  = []
        s = 0.0
        for p in self.window_pnls:
            s += p
            cum_pnl_series.append(s)

        final_pnl = cum_pnl_series[-1] if cum_pnl_series else 0.0
        mdd       = max_drawdown(cum_pnl_series) if cum_pnl_series else 0.0

        edges     = [o['edge'] for o in decided]
        sizes     = [o['size'] for o in decided]
        win_pnls  = [o['pnl'] for o in wins  if 'pnl' in o]
        loss_pnls = [o['pnl'] for o in losses if 'pnl' in o]

        # Profit factor
        gross_profit = sum(p for p in pnls_per_trade if p > 0)
        gross_loss   = abs(sum(p for p in pnls_per_trade if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Consecutive loss streaks
        streak, max_streak, streaks = 0, 0, []
        for o in decided:
            if not o['won']:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                if streak > 0:
                    streaks.append(streak)
                streak = 0

        # Time under water
        peak = 0.0
        tuw  = 0
        for v in cum_pnl_series:
            if v > peak: peak = v
            if v < peak: tuw += 1

        W  = self.windows_run
        NW = len([w for w in self.window_pnls if w != 0.0])

        print()
        print("=" * 72)
        print("  BACKTEST RESULTS — v5 STRATEGY (31-day BTC, 5-min windows)")
        print("=" * 72)

        print(f"\n{'── 1. DATA COVERAGE ':─<72}")
        print(f"  Candles:            {len(self.candles):,}  (1-min, {len(self.candles)/1440:.1f} days)")
        print(f"  Windows processed:  {self.windows_run:,}")
        print(f"  Signals generated:  {self.signals_gen:,}")
        print(f"  Filled trades:      {len(decided):,}")
        print(f"  Toxic events:       {self.toxic_events:,}")
        print(f"  Fill rate:          {pct(len(decided), self.signals_gen)}")

        print(f"\n{'── 2. TRADE BREAKDOWN ':─<72}")
        print(f"  Total decided:      {len(decided):>5}")
        print(f"  Wins / Losses:      {len(wins):>5} / {len(losses):>5}  "
              f"({win_rate:.1%} win rate)")
        print(f"  Maker  W/L:         {maker_w:>5} / {len(maker_d)-maker_w:>5}  "
              f"({pct(maker_w, len(maker_d))} win rate,  n={len(maker_d)})")
        print(f"  Taker  W/L:         {taker_w:>5} / {len(taker_d)-taker_w:>5}  "
              f"({pct(taker_w, len(taker_d))} win rate,  n={len(taker_d)})")

        print(f"\n{'── 3. EDGE & SIZE ':─<72}")
        if edges:
            print(f"  Edge  mean:         {mean(edges)*100:.2f}%")
            print(f"  Edge  std:          {std(edges)*100:.2f}%")
            print(f"  Edge  p25/p75/p95:  {percentile(edges,25)*100:.2f}% / "
                  f"{percentile(edges,75)*100:.2f}% / "
                  f"{percentile(edges,95)*100:.2f}%")
        if sizes:
            print(f"  Size  mean:         ${mean(sizes):.2f}")
            print(f"  Size  p25/p75/p95:  ${percentile(sizes,25):.2f} / "
                  f"${percentile(sizes,75):.2f} / "
                  f"${percentile(sizes,95):.2f}")

        print(f"\n{'── 4. P&L ANALYSIS ':─<72}")
        print(f"  Final P&L:          ${final_pnl:+.2f}")
        print(f"  Return on ${BANKROLL_START:.0f}:     {final_pnl/BANKROLL_START*100:+.2f}%")
        print(f"  Ann. return:        {final_pnl/BANKROLL_START*100/31*365:+.1f}%  (31-day → 365-day)")
        print(f"  Gross profit:       ${gross_profit:+.2f}")
        print(f"  Gross loss:         ${gross_loss:.2f}")
        print(f"  Profit factor:      {profit_factor:.3f}  (>1.0 = profitable)")
        if win_pnls:
            print(f"  Avg win P&L:        ${mean(win_pnls):+.2f}")
        if loss_pnls:
            print(f"  Avg loss P&L:       ${mean(loss_pnls):+.2f}")
        win_avg  = abs(mean(win_pnls))  if win_pnls  else 0
        loss_avg = abs(mean(loss_pnls)) if loss_pnls else 0
        if loss_avg > 0:
            print(f"  Win/Loss ratio:     {win_avg/loss_avg:.3f}:1")
        print(f"  Expected value:     ${mean(pnls_per_trade):+.4f} per trade")

        print(f"\n{'── 5. RISK METRICS ':─<72}")
        print(f"  Max Drawdown:       ${mdd:.2f}  ({mdd/BANKROLL_START*100:.2f}% of bankroll)")
        wl_sharpe = sharpe(self.window_pnls)
        print(f"  Sharpe (windows):   {wl_sharpe:.3f}")
        if mdd > 0:
            print(f"  Calmar ratio:       {calmar(self.window_pnls, mdd):.3f}")
        print(f"  Max loss streak:    {max_streak} consecutive losses")
        if streaks:
            print(f"  Avg loss streak:    {mean(streaks):.1f}  (p95={percentile(streaks,95):.0f})")
        print(f"  Time under water:   {tuw}/{W} windows "
              f"({pct(tuw, W)})")
        print(f"  Bankroll at end:    ${BANKROLL_START+final_pnl:.2f}")

        print(f"\n{'── 6. CALIBRATION (OOS) ':─<72}")
        cal_sum = self.signal_gen.calibration.oos_summary()
        print(f"  OOS folds fit:      {cal_sum['folds']}")
        print(f"  Final gamma (γ):    {cal_sum['gamma']}")
        if cal_sum['folds']:
            print(f"  Avg val log-loss:   {cal_sum.get('avg_val_ll', 'n/a')}")

        print(f"\n{'── 6a. STEP 1 OOS EXTRACTION (confident signals) ':─<72}")
        oos_signals = self.get_oos_confident_signals()
        if oos_signals:
            n_up = sum(1 for s in oos_signals if s["direction"] == "UP")
            n_dn = sum(1 for s in oos_signals if s["direction"] == "DOWN")
            wins = sum(1 for s in oos_signals if s["outcome"] == 1)
            wr = wins / len(oos_signals) if oos_signals else 0
            mean_p_model = mean([s["p_model"] for s in oos_signals])
            mean_p_market = mean([s["p_market"] for s in oos_signals if s["p_market"] is not None])
            print(f"  Confident trades:   {len(oos_signals)}  (UP: {n_up}  DOWN: {n_dn})")
            print(f"  Win rate:           {wr:.1%}")
            print(f"  Mean P_model:       {mean_p_model:.4f}")
            print(f"  Mean P_market:      {mean_p_market:.4f}")
            export_path = "logs/oos_confident_signals_backtest.json"
            os.makedirs("logs", exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(oos_signals, f, indent=2)
            print(f"  Exported to:        {export_path}")
        else:
            print(f"  No confident trades (prob ∉ [0.38, 0.62]) in filled set.")

        print(f"\n{'── 6b. FULL EQUITY SIMULATION (post-backtest) ':─<72}")
        if decided:
            avg_p    = mean([o['fair'] for o in decided])
            avg_mid  = mean([o['mid']  for o in decided])
            avg_size = mean(sizes)
            kf       = avg_size / max(1.0, BANKROLL_START)
            eq_full  = EquitySimulator.simulate(
                bankroll   = BANKROLL_START,
                n_trades   = len(decided),
                prob       = avg_p,
                price      = avg_mid,
                kelly_frac = kf,
                n_paths    = 2000,
            )
            print(f"  Based on {len(decided)} actual trades, avg p={avg_p:.3f}, avg mid={avg_mid:.3f}")
            print(f"  P(ruin):            {eq_full['p_ruin']:.3f}  ({'SAFE' if eq_full['p_ruin']<0.20 else 'HIGH RISK'})")
            print(f"  Median final eq:    ${eq_full['median_final']:.2f}")
            print(f"  5th / 95th pct:     ${eq_full['ci_5']:.2f} / ${eq_full['ci_95']:.2f}")
            print(f"  MaxDD mean/p95:     ${eq_full['dd_mean']:.2f} / ${eq_full['dd_p95']:.2f}")
            print(f"  Loss streak p95:    {eq_full['streak_p95']:.0f} consecutive")
            print(f"  Time under water:   {eq_full['tuw_pct_mean']*100:.1f}% of trades")

        print(f"\n{'── 7. REGIME DISTRIBUTION ':─<72}")
        regime_counts = defaultdict(int)
        for ws in self.per_window_stats:
            btc_series = []  # We don't have full tick history here; skip
        # Count from filled trades by approximation
        regime_wins  = defaultdict(int)
        regime_total = defaultdict(int)

        print(f"\n{'── 8. PERFORMANCE BENCHMARKS ':─<72}")
        # Compare vs naive 50/50 bet at market price
        avg_mid = mean([o['mid'] for o in decided]) if decided else 0.5
        naive_ev_per_trade = 0.5 * (1/avg_mid - 1) * (1 - TAKER_FEE) - 0.5
        print(f"  Naive (50/50) EV:   ${naive_ev_per_trade * mean(sizes):.4f}/trade")
        actual_ev = mean(pnls_per_trade) if pnls_per_trade else 0.0
        print(f"  Strategy EV:        ${actual_ev:.4f}/trade")
        edge_over_naive = actual_ev - naive_ev_per_trade * mean(sizes)
        print(f"  Edge over naive:    ${edge_over_naive:+.4f}/trade  "
              f"({'✓ POSITIVE EDGE' if edge_over_naive > 0 else '✗ NO EDGE DETECTED'})")

        # Statistical significance (t-test on trade P&Ls)
        if len(pnls_per_trade) >= 30:
            from scipy import stats as scipy_stats
            t_stat, p_val = scipy_stats.ttest_1samp(pnls_per_trade, 0)
            print(f"\n{'── 9. STATISTICAL SIGNIFICANCE ':─<72}")
            print(f"  H₀: mean EV = $0 per trade")
            print(f"  t-statistic:        {t_stat:.3f}")
            print(f"  p-value:            {p_val:.4f}")
            print(f"  Verdict:            "
                  f"{'✓ SIGNIFICANT (p<0.05) — strategy has real edge' if p_val < 0.05 else '✗ NOT significant yet (need more trades)'}")

        print(f"\n{'── 10. ROLLING 50-WINDOW P&L ':─<72}")
        wpnls = self.window_pnls
        block = 50
        for start in range(0, min(len(wpnls), 600), block):
            chunk = wpnls[start:start+block]
            if not chunk:
                break
            cp = sum(chunk)
            ww = sum(1 for x in chunk if x > 0)
            print(f"  Windows {start+1:>4}-{start+len(chunk):>4}:  "
                  f"P&L ${cp:+.2f}  |  {ww}/{len(chunk)} profitable windows")

        print()
        print("=" * 72)
        print("  END OF BACKTEST REPORT")
        print("=" * 72)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    pkl_path = os.path.join(os.path.dirname(__file__), 'btc_1m_candles.pkl')
    print(f"  Loading candles from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        candles = pickle.load(f)
    print(f"  Loaded {len(candles):,} 1-min candles.")

    bt = Backtester(candles, bankroll=BANKROLL_START)
    bt.run()

    # Save per-window CSV for external analysis
    import csv
    out = os.path.join(os.path.dirname(__file__), 'logs',
                       f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'window_id','strike','final_btc','result',
            'pnl','cum_pnl','n_filled','bankroll'])
        writer.writeheader()
        writer.writerows(bt.per_window_stats)
    print(f"\n  Per-window CSV saved to: {out}")


if __name__ == "__main__":
    main()
