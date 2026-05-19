"""
test_connection

Verifies that the Alpaca paper trading connection works and that both
GLD and FXY are tradeable assets. Run this once after setting up .env.

Usage:
    python test_connection.py
"""

import sys
import os

# Allow running from the synk/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config

# Symbols Synk trades — defined once so both test and strategy share the same source of truth
_SYNK_SYMBOLS = ("GLD", "FXY")

try:
    from alpaca.trading.client import TradingClient
except ImportError:
    print("ERROR: alpaca-py not installed. Run: pip install alpaca-py")
    sys.exit(1)


def main() -> None:
    # --- Load and validate config ---
    try:
        cfg = get_config()
    except EnvironmentError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print("Config loaded OK")
    print(f"  Paper mode: {cfg.PAPER}")

    # --- Connect to Alpaca ---
    try:
        client = TradingClient(
            api_key=cfg.ALPACA_API_KEY,
            secret_key=cfg.ALPACA_SECRET_KEY,
            paper=cfg.PAPER,
        )
    except Exception as e:
        print(f"FAIL: Could not create TradingClient: {e}")
        sys.exit(1)

    # --- Check account ---
    try:
        account = client.get_account()
        print(f"\nAccount Status:  {account.status}")
        print(f"Cash:            ${float(account.cash):,.2f}")
        print(f"Buying Power:    ${float(account.buying_power):,.2f}")
    except Exception as e:
        print(f"FAIL: Could not fetch account: {e}")
        sys.exit(1)

    # --- Check assets ---
    passed = True
    for symbol in _SYNK_SYMBOLS:
        try:
            asset = client.get_asset(symbol)
            print(f"\n{symbol}:")
            print(f"  status:       {asset.status}")
            print(f"  fractionable: {asset.fractionable}")
            # Compare .value (the underlying string) to avoid fragile enum-repr checks
            # that break across Python versions (str-Enum repr changed in 3.11+)
            if asset.status.value != "active":
                print(f"  WARNING: {symbol} is not active")
                passed = False
        except Exception as e:
            print(f"\nFAIL: Could not fetch asset {symbol}: {e}")
            passed = False

    # --- Final verdict ---
    print()
    if passed:
        print("--- PASS ---")
    else:
        print("--- FAIL: see warnings above ---")
        sys.exit(1)


if __name__ == "__main__":
    main()
