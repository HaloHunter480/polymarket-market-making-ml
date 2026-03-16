#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Polymarket Empirical Bot - Server Deployment Script
# ═══════════════════════════════════════════════════════════════════
#
# This script sets up the bot on a fresh Ubuntu server (AWS/VPS/etc).
#
# Usage:
#   1. SSH into your server
#   2. Upload this project: scp -r arbpoly/ user@server:~/
#   3. Run: bash deploy.sh
#
# After setup, use tmux or systemd to keep it running:
#   tmux new -s bot
#   python3 live_test_2usd.py
#   (Ctrl+B, D to detach)
#
# ═══════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Polymarket Empirical Bot - Server Setup"
echo "═══════════════════════════════════════════════════════════════"

# 1. System packages
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv tmux htop

# 2. Python virtual environment
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
echo "[3/5] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 4. Validate .env
echo "[4/5] Checking .env file..."
if [ ! -f .env ]; then
    echo ""
    echo "  ERROR: .env file not found!"
    echo "  Create .env with your credentials:"
    echo "    POLY_PRIVATE_KEY=0x..."
    echo "    POLY_API_KEY=..."
    echo "    POLY_API_SECRET=..."
    echo "    POLY_API_PASSPHRASE=..."
    echo ""
    exit 1
fi
echo "  .env found"

# 5. Latency benchmark
echo "[5/5] Running latency benchmark..."
python3 -c "
import time, requests

# Ping Polymarket CLOB
session = requests.Session()
session.get('https://clob.polymarket.com/', timeout=10)

lats = []
for _ in range(5):
    t0 = time.perf_counter()
    session.get('https://clob.polymarket.com/', timeout=10)
    t1 = time.perf_counter()
    lats.append((t1 - t0) * 1000)

avg = sum(lats) / len(lats)
mn = min(lats)
print(f'  Polymarket CLOB latency: {mn:.0f}ms (min), {avg:.0f}ms (avg)')

# Ping Coinbase (v6 price feed)
session2 = requests.Session()
session2.get('https://api.exchange.coinbase.com/', timeout=10)
lats2 = []
for _ in range(5):
    t0 = time.perf_counter()
    session2.get('https://api.exchange.coinbase.com/', timeout=10)
    t1 = time.perf_counter()
    lats2.append((t1 - t0) * 1000)

avg2 = sum(lats2) / len(lats2)
mn2 = min(lats2)
print(f'  Coinbase API latency:    {mn2:.0f}ms (min), {avg2:.0f}ms (avg)')
print(f'  Total round-trip budget: ~{int(mn + mn2)}ms')
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Run paper trading:"
echo "    source venv/bin/activate"
echo "    python3 professional_strategy.py --bankroll=500"
echo ""
echo "  Run live trading (v6, small size):"
echo "    python3 live_test_2usd.py"
echo ""
echo "  Run in background (survives SSH disconnect):"
echo "    tmux new -s bot"
echo "    python3 live_test_2usd.py"
echo "    # Press Ctrl+B, then D to detach"
echo "    # tmux attach -t bot to reattach"
echo "═══════════════════════════════════════════════════════════════"
