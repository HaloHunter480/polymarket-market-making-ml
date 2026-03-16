# Setup, Sizing & Risk — Live BTC Trading

## Setup (before first run)

### 1. Credentials in `.env`

```bash
POLY_PRIVATE_KEY=0x...          # MetaMask wallet (export private key)
POLY_API_KEY=...                # Polymarket API (create at polymarket.com)
POLY_API_SECRET=...
POLY_API_PASSPHRASE=...
PROXY_ADDRESS=0x...             # From find_proxy.py
```

### 2. Get proxy address

```bash
python3 find_proxy.py
# Copy the proxy address into .env as PROXY_ADDRESS
```

### 3. Fund wallet

- USDC on Polygon in your Polymarket proxy wallet
- **Minimum:** ~$2 for live_test_2usd (2 × $1 trades)

---

## Sizing (current config)

| Parameter              | Value  | Meaning                    |
|------------------------|--------|----------------------------|
| `MAX_SINGLE_TRADE_USD` | $1.00  | Per-trade cap (Polymarket min) |
| `MAX_SESSION_SPEND_USD`| $2.00  | Session budget             |
| `MAX_TRADES_SESSION`   | 2      | Trades per session         |
| `MAX_KELLY_FRACTION`   | 25%    | Max bankroll per trade     |
| Kelly percentile       | 25th   | Conservative (not median)  |

**Effect:** With $2 bankroll, each trade is $1. Kelly and equity simulation may suggest longer sizes; we cap at $1 for safety.

---

## Risk Controls

| Control         | Threshold | Action                          |
|-----------------|-----------|---------------------------------|
| P(ruin)         | &gt; 20%  | Reject trade or halve size      |
| Edge floor      | 6%        | Skip if edge &lt; 6%            |
| Price zone      | 0.35–0.57 | Only uncertain markets         |
| BTC distance    | 0.01–0.20%| Must be near strike, our direction |
| Entry window    | First 120s| No late entries                |
| Emergency exit  | 4 conditions| Only catastrophic adverse selection |

### Emergency exit (hold-to-expiry)

Sell only if **all** true:

- BTC &ge; 0.12% wrong side of strike  
- &gt; 90 s remaining  
- OFI strongly against us (&ge; 0.80)  
- Re-eval P(win) &lt; 22%  

Otherwise: hold to expiry.

---

## Run

```bash
python3 live_test_2usd.py
```

Logs → `logs/live_YYYYMMDD_HHMMSS.jsonl`
# STRIKE_LOCK_S extended to 150s, OBI gate neutral default, EXPLOSIVE regime
