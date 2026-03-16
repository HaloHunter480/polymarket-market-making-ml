"""
download_btc_1s.py — Download BTC 1-second data from Binance
=============================================================

Binance provides 1s klines at: https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/

Format per file: BTCUSDT-1s-YYYY-MM-DD.zip
Each CSV row: open_time, open, high, low, close, volume, close_time,
              quote_asset_volume, number_of_trades, taker_buy_base_asset_volume,
              taker_buy_quote_asset_volume, ignore

Usage:
  python download_btc_1s.py                    # Download last 7 days
  python download_btc_1s.py --days 30        # Download last 30 days
  python download_btc_1s.py --start 2024-01-01 --end 2024-01-07
"""

import argparse
import os
import re
import zipfile
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    requests = None


BASE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s"
OUTPUT_CSV = "btc_1sec.csv"
OUTPUT_ZIP = "btc_1sec.csv.zip"


def download_date(date_str: str, out_dir: str = ".") -> str:
    """Download one day's zip, extract CSV, return path to CSV."""
    fname = f"BTCUSDT-1s-{date_str}.zip"
    url = f"{BASE_URL}/{fname}"
    path = os.path.join(out_dir, fname)
    csv_path = os.path.join(out_dir, fname.replace(".zip", ".csv"))

    if os.path.exists(csv_path):
        return csv_path

    if not requests:
        print("  [!] pip install requests")
        return ""

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return ""
        with open(path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(out_dir)
        os.remove(path)
        return csv_path
    except Exception as e:
        print(f"  [!] {date_str}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    parser.add_argument("--merge", action="store_true", help="Merge all into btc_1sec.csv")
    args = parser.parse_args()

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end = datetime.utcnow()
        start = end - timedelta(days=args.days)

    os.makedirs(args.out, exist_ok=True)
    csvs = []
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        p = download_date(ds, args.out)
        if p:
            csvs.append(p)
            print(f"  [OK] {ds}")
        d += timedelta(days=1)

    if not csvs:
        print("  [!] No data downloaded. Check dates (Binance has data from ~2020).")
        return 1

    if args.merge and len(csvs) > 1:
        out_path = os.path.join(args.out, OUTPUT_CSV)
        with open(out_path, "w") as outf:
            for i, p in enumerate(sorted(csvs)):
                with open(p, "r") as f:
                    lines = f.readlines()
                if i == 0:
                    outf.writelines(lines)
                else:
                    outf.writelines(lines[1:])
        print(f"  [OK] Merged to {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
