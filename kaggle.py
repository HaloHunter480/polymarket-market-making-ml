"""
kaggle.py — Download training datasets via kagglehub
=====================================================

Downloads:
  1. BTC/USDT 5-min OHLCV 2017-2025  (used by train_hmm_regime.py)

Usage:
    python3 kaggle.py

Requires:
    pip install kagglehub
"""

import kagglehub

# BTC/USDT 5-minute OHLCV 2017-2025 (for HMM regime training)
path = kagglehub.dataset_download("shivaverse/btcusdt-5-minute-ohlc-volume-data-2017-2025")
print("5-min OHLCV dataset:", path)

# For 1-second data (used by train_probability_model.py), use:
#   python3 download_btc_1s.py --days 30 --merge
# Or place BTC_1sec.csv (L2 order book, 1.6GB) in the project root.
# See data/README.md for full details.
