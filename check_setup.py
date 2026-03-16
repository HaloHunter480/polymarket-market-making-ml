#!/usr/bin/env python3
"""Pre-flight check before live trading. Verifies credentials, proxy, connectivity."""
import os
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

def main():
    pk = os.getenv("POLY_PRIVATE_KEY", "")
    ak = os.getenv("POLY_API_KEY", "")
    aks = os.getenv("POLY_API_SECRET", "")
    ap = os.getenv("POLY_API_PASSPHRASE", "")
    proxy = os.getenv("PROXY_ADDRESS", "")

    ok = True
    if not pk or pk == "0xYOUR_PRIVATE_KEY_HERE":
        print("  ❌ POLY_PRIVATE_KEY missing or placeholder")
        ok = False
    if not ak:
        print("  ❌ POLY_API_KEY missing")
        ok = False
    if not aks:
        print("  ❌ POLY_API_SECRET missing")
        ok = False
    if not ap:
        print("  ❌ POLY_API_PASSPHRASE missing")
        ok = False
    if not proxy:
        print("  ❌ PROXY_ADDRESS missing (run: python3 find_proxy.py)")
        ok = False

    if not ok:
        print("\n  Fix .env and run again.")
        sys.exit(1)

    print("  ✅ All .env vars present")
    # Verify balance & proxy via find_proxy
    try:
        import subprocess
        r = subprocess.run([sys.executable, "find_proxy.py"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print("  ❌ find_proxy failed (check balance/credentials)")
            if r.stderr:
                print(r.stderr[:500])
            sys.exit(1)
        print("  ✅ Credentials & balance OK")
    except Exception as e:
        print(f"  ⚠️ Could not verify balance: {e}")

    print("\n  Ready to trade. Run: python3 live_test_2usd.py\n")

if __name__ == "__main__":
    main()
