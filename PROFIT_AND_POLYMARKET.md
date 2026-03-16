# Why Bot Profit vs Polymarket Portfolio Can Look Different

## When Does the Bot Count "Profit"?

The bot only adds to `total_pnl` when **both legs** of a quote are filled:

1. **BID filled** (you bought YES) + **ASK filled** (you sold YES) = round-trip complete
2. Profit = `(ask_fill_price - bid_fill_price) × shares` — e.g. buy @ 0.69, sell @ 0.71 = $0.02/share

That profit **should** appear in your Polymarket USDC balance when both legs fill.

---

## Why Your Portfolio Might Not Show It

### 1. **INSUFFICIENT_FUNDS / Allowance Errors**

If you see `"not enough balance / allowance"` in the logs, **orders are failing**. No real trades = no real profit.

**Fix:** The bot now auto-refreshes allowance on this error. Also:
- Go to [Polymarket](https://polymarket.com) → Settings → Approve USDC for trading
- Ensure you have enough USDC in your Polymarket wallet (not just your main wallet)

### 2. **One Leg Filled = Open Position, Not Profit**

If only the BID filled (you bought) and the ASK hasn’t filled yet:
- You have an **open position** (long 8 shares)
- The bot does **not** count this as profit
- Polymarket shows this under **Positions** or **Open Orders** (as a position, not an order)
- Profit is realized only when you **sell** (ASK fills)

### 3. **"Open Orders" vs "Positions"**

- **Open orders** = unfilled limit orders (your quotes waiting to be hit)
- **Positions** = you bought but haven’t sold yet (one leg filled)
- **Trade history** = completed round-trips; may be delayed or shown differently in the UI

### 4. **Small Amounts**

Session profit of $0.10–$0.40 can be hard to notice in a larger balance. Check your Polymarket USDC balance before and after a session.

### 5. **Polymarket Proxy Wallet**

Polymarket often uses a proxy. Your balance is in the **exchange/collateral** wallet, not necessarily your main wallet. The bot reads from the CLOB balance.

---

## What the Bot Tracks

| Metric | Meaning |
|--------|---------|
| `total_pnl` | Sum of completed round-trips (both legs filled) |
| `state.balance` | Synced from Polymarket each cycle in live mode |
| Spread capture % | How much of the quoted spread you actually got |

---

## Quick Checks

1. **Logs:** Look for `Completed [ORGANIC]: ... pnl=$0.10` — that’s a real completed trade.
2. **Errors:** Search for `INSUFFICIENT_FUNDS` or `ORDER_FAILED` — those orders did not execute.
3. **Polymarket:** Check **Activity** or **Positions** for filled orders and closed positions.
