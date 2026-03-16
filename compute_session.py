"""
Session Analytics — parse paper trade log and compute full statistics.
Run:  python3 compute_session.py logs/paper_YYYYMMDD_HHMMSS.log
"""

import sys
import re
import math
from collections import defaultdict

# ── helpers ───────────────────────────────────────────────────────────────────

def pct(x, n):
    return f"{x/n*100:.1f}%" if n else "n/a"

def mean(lst):
    return sum(lst)/len(lst) if lst else 0.0

def std(lst):
    if len(lst) < 2:
        return 0.0
    m = mean(lst)
    return math.sqrt(sum((x-m)**2 for x in lst) / (len(lst)-1))

def median(lst):
    s = sorted(lst)
    n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1]+s[n//2])/2

def percentile(lst, p):
    s = sorted(lst)
    idx = (len(s)-1)*p/100
    lo, hi = int(idx), min(int(idx)+1, len(s)-1)
    return s[lo] + (idx-lo)*(s[hi]-s[lo])

def sharpe(pnls, rf=0.0):
    if len(pnls) < 2:
        return float('nan')
    m = mean(pnls) - rf
    s = std(pnls)
    return m/s if s > 1e-12 else float('nan')

def max_drawdown(cum_pnl_series):
    peak = cum_pnl_series[0]
    max_dd = 0.0
    for v in cum_pnl_series:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd

# ── parse ─────────────────────────────────────────────────────────────────────

def parse_log(path):
    windows = []        # list of {id, result, pnl_delta, trades:[...]}
    ticks   = []        # list of {ts, btc, vpin, flow, ofi, regime, is_toxic}
    trades  = []        # list of {type, side, edge, size, filled, won, exec_type}
    signals = []        # every trade signal (filled or not)

    current_window = None
    last_pnl = 0.0
    last_wl  = (0, 0)
    pending_signal = None  # accumulate multi-line trade block

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    for i, raw in enumerate(lines):
        line = raw.strip()

        # ── Window resolved ────────────────────────────────────────────────
        m = re.search(r'WINDOW RESOLVED: (\w+) \| BTC \$([\d,.]+) vs Strike \$([\d,.]+) \| P&L: \$([+-]?[\d,.]+)', line)
        if m:
            direction = m.group(1)
            final_btc = float(m.group(2).replace(',',''))
            strike    = float(m.group(3).replace(',',''))
            cum_pnl   = float(m.group(4).replace(',',''))
            delta     = cum_pnl - last_pnl
            if current_window is not None:
                current_window['result']    = direction
                current_window['pnl_delta'] = delta
                windows.append(current_window)
            last_pnl = cum_pnl
            current_window = None
            continue

        # ── New window ─────────────────────────────────────────────────────
        m = re.search(r'NEW WINDOW \| Strike: \$([\d,.]+) \| Regime: [^\(]+ \(([^)]+)\)', line)
        if m:
            strike  = float(m.group(1).replace(',',''))
            current_window = {'strike': strike, 'result': None,
                              'pnl_delta': 0.0, 'trades': []}
            continue

        # ── Tick line ──────────────────────────────────────────────────────
        m = re.match(r'\s*(\d{2}:\d{2}:\d{2}) \| BTC \$([\d, .]+) \| K \$([\d, .]+) \| '
                     r'Diff ([+-][\d.]+)% \| Flow ([+-][\d.]+) \| OFI ([+-][\d.]+)\(w=([\d.]+)\) \| '
                     r'VPIN ([\d.]+) \| (.+?) \| [^\|]+ \| .+ \| W/L (\d+)/(\d+) \| M/T (\d+)/(\d+) \| '
                     r'P&L \$([+-]?[\d,.]+)', line)
        if m:
            ts      = m.group(1)
            btc     = float(m.group(2).replace(' ','').replace(',',''))
            vpin    = float(m.group(8))
            flow    = float(m.group(5))
            ofi     = float(m.group(6))
            toxic   = 'TOXIC' in m.group(9)
            regime  = m.group(9).strip()
            wins    = int(m.group(10))
            losses  = int(m.group(11))
            pnl     = float(m.group(13).replace(',',''))
            ticks.append({'btc': btc, 'vpin': vpin, 'flow': flow,
                          'ofi': ofi, 'toxic': toxic, 'pnl': pnl,
                          'wins': wins, 'losses': losses})
            last_wl = (wins, losses)
            continue

        # ── Trade signal header ────────────────────────────────────────────
        m = re.search(r'\[(MAKER|TAKER)\] (UP|DOWN) @ .*?mid: \$([\d.]+)\)', line)
        if m:
            pending_signal = {
                'exec_type': m.group(1),
                'side':      m.group(2),
                'mid':       float(m.group(3)),
                'edge':      None,
                'size':      None,
                'filled':    None,
                'won':       None,
            }
            continue

        m = re.search(r'\[STRONG\] Edge: ([\d.]+)%.*→ ([\d.]+)%', line)
        if m and pending_signal:
            pending_signal['edge'] = float(m.group(1)) / 100
            continue

        # ── Edge / size ────────────────────────────────────────────────────
        m = re.search(r'\[EDGE\] ([\d.]+)% \| Size: \$([\d.]+)', line)
        if m and pending_signal:
            pending_signal['edge'] = float(m.group(1)) / 100
            pending_signal['size'] = float(m.group(2))
            continue

        # ── Fill result ────────────────────────────────────────────────────
        if '[MAKER_FILL]' in line and pending_signal:
            pending_signal['filled'] = True
            continue
        if '[MAKER_MISS]' in line and pending_signal:
            pending_signal['filled'] = False
            continue
        if '[TAKER]' in line and pending_signal and pending_signal['exec_type'] == 'TAKER':
            pending_signal['filled'] = True
            continue

        # ── Trade recorded ─────────────────────────────────────────────────
        m = re.search(r'\[TRADE #(\d+)\]', line)
        if m and pending_signal:
            signals.append(dict(pending_signal))
            if current_window is not None:
                current_window['trades'].append(dict(pending_signal))
            pending_signal = None
            continue

        # ── Blocked (signal was generated but killed) ──────────────────────
        if '[BLOCKED]' in line and pending_signal:
            pending_signal['filled'] = False
            signals.append(dict(pending_signal))
            pending_signal = None
            continue

    # ── Annotate won/lost after window resolution ──────────────────────────
    for w in windows:
        resolved_up = w['result'] == 'UP'
        for t in w['trades']:
            if t['filled']:
                t['won'] = (t['side'] == 'UP') == resolved_up

    all_filled = [t for w in windows for t in w['trades'] if t.get('filled')]
    return windows, ticks, signals, all_filled

# ── report ─────────────────────────────────────────────────────────────────────

def report(path):
    windows, ticks, signals, filled = parse_log(path)

    print("=" * 72)
    print("  PAPER TRADE SESSION ANALYTICS")
    print(f"  Log: {path}")
    print("=" * 72)

    # ── 1. Tick / market data ──────────────────────────────────────────────
    n_ticks  = len(ticks)
    n_toxic  = sum(1 for t in ticks if t['toxic'])
    vpins    = [t['vpin'] for t in ticks]
    flows    = [t['flow'] for t in ticks]
    ofis     = [abs(t['ofi']) for t in ticks]
    print(f"\n{'── 1. MARKET DATA ':─<72}")
    print(f"  Ticks processed:   {n_ticks:,}")
    print(f"  Windows resolved:  {len(windows)}")
    print(f"  Toxic ticks:       {n_toxic:,} ({pct(n_toxic,n_ticks)}) — VPIN ≥ threshold")
    print(f"  VPIN  mean/p95:    {mean(vpins):.3f} / {percentile(vpins,95):.3f}")
    print(f"  |OFI| mean/p95:    {mean(ofis):.3f} / {percentile(ofis,95):.3f}")
    print(f"  Flow  mean/p95:    {mean([abs(f) for f in flows]):.3f} / {percentile([abs(f) for f in flows],95):.3f}")

    # ── 2. Signal generation ───────────────────────────────────────────────
    n_sig    = len(signals)
    n_filled = len(filled)
    n_maker  = sum(1 for s in signals if s['exec_type'] == 'MAKER')
    n_taker  = sum(1 for s in signals if s['exec_type'] == 'TAKER')
    n_fill_m = sum(1 for s in signals if s['exec_type'] == 'MAKER' and s.get('filled'))
    n_fill_t = sum(1 for s in signals if s['exec_type'] == 'TAKER' and s.get('filled'))
    edges    = [s['edge'] for s in signals if s.get('edge')]
    sizes    = [s['size'] for s in signals if s.get('size')]
    print(f"\n{'── 2. SIGNAL GENERATION ':─<72}")
    print(f"  Total signals:     {n_sig}")
    print(f"    Maker signals:   {n_maker}  (filled: {n_fill_m}, fill-rate: {pct(n_fill_m,n_maker)})")
    print(f"    Taker signals:   {n_taker}  (filled: {n_fill_t})")
    print(f"  Filled trades:     {n_filled}")
    if edges:
        print(f"  Edge  mean/p95:    {mean(edges)*100:.2f}% / {percentile(edges,95)*100:.2f}%")
    if sizes:
        print(f"  Size  mean/p95:    ${mean(sizes):.1f} / ${percentile(sizes,95):.1f}")

    # ── 3. Trade outcomes ──────────────────────────────────────────────────
    decided = [t for t in filled if t.get('won') is not None]
    wins    = [t for t in decided if t['won']]
    losses  = [t for t in decided if not t['won']]
    win_rt  = len(wins)/len(decided) if decided else 0.0

    maker_d = [t for t in decided if t['exec_type'] == 'MAKER']
    taker_d = [t for t in decided if t['exec_type'] == 'TAKER']
    maker_w = sum(1 for t in maker_d if t['won'])
    taker_w = sum(1 for t in taker_d if t['won'])

    print(f"\n{'── 3. TRADE OUTCOMES ':─<72}")
    print(f"  Decided trades:    {len(decided)}")
    print(f"  Wins / Losses:     {len(wins)} / {len(losses)} ({win_rt:.1%} win rate)")
    print(f"  Maker W/L:         {maker_w}/{len(maker_d)-maker_w}  ({pct(maker_w,len(maker_d))} win rate)")
    print(f"  Taker W/L:         {taker_w}/{len(taker_d)-taker_w}  ({pct(taker_w,len(taker_d))} win rate)")

    # ── 4. P&L analysis ───────────────────────────────────────────────────
    pnl_per_window = [w['pnl_delta'] for w in windows]
    cum_pnl = []
    s = 0.0
    for p in pnl_per_window:
        s += p
        cum_pnl.append(s)

    final_pnl  = cum_pnl[-1] if cum_pnl else 0.0
    max_dd_val = max_drawdown(cum_pnl) if cum_pnl else 0.0
    win_w      = [w for w in windows if w['pnl_delta'] > 0]
    loss_w     = [w for w in windows if w['pnl_delta'] < 0]

    print(f"\n{'── 4. P&L ANALYSIS ':─<72}")
    print(f"  Final P&L:         ${final_pnl:+.2f}")
    print(f"  Return on $500:    {final_pnl/500*100:+.2f}%")
    print(f"  P&L per window:")
    for i, w in enumerate(windows):
        print(f"    W{i+1} ({w['result']:>4}):  ${w['pnl_delta']:+.2f}  (cum ${sum(pnl_per_window[:i+1]):+.2f})")
    print(f"  Avg P&L/window:    ${mean(pnl_per_window):+.2f}")
    print(f"  Std P&L/window:    ${std(pnl_per_window):.2f}")
    if len(pnl_per_window) >= 2:
        sr = sharpe(pnl_per_window)
        print(f"  Sharpe (windows):  {sr:.3f}")
    print(f"  Max Drawdown:      ${max_dd_val:.2f}")
    print(f"  Profitable windows:{len(win_w)}/{len(windows)}")

    # ── 5. OFI signal quality ─────────────────────────────────────────────
    ofi_vals = [t['ofi'] for t in ticks]
    ofi_nz   = [v for v in ofi_vals if abs(v) > 0.01]
    print(f"\n{'── 5. OFI SIGNAL QUALITY ':─<72}")
    print(f"  Active OFI ticks:  {len(ofi_nz)}/{n_ticks} ({pct(len(ofi_nz),n_ticks)})")
    if ofi_nz:
        print(f"  OFI  mean:         {mean(ofi_nz):+.4f}")
        print(f"  OFI  std:          {std(ofi_nz):.4f}")
        print(f"  OFI  p5/p95:       {percentile(ofi_nz,5):+.4f} / {percentile(ofi_nz,95):+.4f}")
        neg = sum(1 for v in ofi_nz if v < 0)
        pos = sum(1 for v in ofi_nz if v > 0)
        print(f"  OFI  neg/pos:      {neg}/{pos} — {'Sell pressure dominant' if neg>pos else 'Buy pressure dominant'}")

    # ── 6. Regime stats ───────────────────────────────────────────────────
    regimes = defaultdict(int)
    for line in open(path):
        for r in ['trending','mean_reverting','volatile','neutral','unknown']:
            if r in line and 'tren' in line[:80] or r in line and '|' in line:
                if '📈' in line or '🔄' in line or '⚡' in line or '➖' in line or '❓' in line:
                    for tok in ['tren','mean','vola','neut','unkn']:
                        if tok in line:
                            regimes[tok] += 1
                            break
                    break
    print(f"\n{'── 6. REGIME CLASSIFICATION ':─<72}")
    for k,v in sorted(regimes.items(), key=lambda x: -x[1]):
        print(f"  {k:<14}: {v:>5} ticks ({pct(v,sum(regimes.values()))})")

    # ── 7. Risk metrics ───────────────────────────────────────────────────
    bankroll = 500.0
    print(f"\n{'── 7. RISK METRICS ':─<72}")
    print(f"  Bankroll at start: ${bankroll:.2f}")
    print(f"  Bankroll at end:   ${bankroll+final_pnl:.2f}")
    print(f"  Max Drawdown:      ${max_dd_val:.2f} ({max_dd_val/bankroll*100:.2f}% of bankroll)")
    if pnl_per_window:
        worst = min(pnl_per_window)
        best  = max(pnl_per_window)
        print(f"  Worst window:      ${worst:+.2f}")
        print(f"  Best window:       ${best:+.2f}")
    print(f"  Win rate:          {win_rt:.1%}")
    if win_rt > 0 and win_rt < 1 and decided:
        avg_win  = mean([s['size'] for s in decided if s.get('won') and s.get('size')] or [0.0])
        avg_loss = mean([s['size'] for s in decided if not s.get('won') and s.get('size')] or [0.0])
        if avg_loss > 0:
            rr = avg_win / avg_loss
            print(f"  Avg win / avg loss (size): {rr:.2f}:1")
        kelly_est = win_rt - (1-win_rt)
        print(f"  Theoretical Kelly: {kelly_est:.3f} (naive, no odds adjustment)")

    # ── 8. Expected value check ───────────────────────────────────────────
    print(f"\n{'── 8. EXPECTED VALUE (per filled trade) ':─<72}")
    if decided:
        avg_edge = mean([s['edge'] for s in decided if s.get('edge')] or [0.0])
        print(f"  Avg declared edge: {avg_edge*100:.2f}%")
        actual_ev = final_pnl / len(decided) if decided else 0.0
        print(f"  Actual EV/trade:   ${actual_ev:+.2f}")
        if sizes:
            actual_ev_pct = actual_ev / mean(sizes) * 100
            print(f"  Actual EV %:       {actual_ev_pct:+.2f}% of avg stake")

    print(f"\n{'=' * 72}")
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else sorted(
        __import__('glob').glob('logs/paper_*.log'))[-1]
    report(log)
