#!/usr/bin/env python3
"""
Manual backfill script - run when needed for recovery or initial setup.
Usage: python -m scripts.backfill_pairs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptobot.data.kraken import KrakenAPI

PAIRS = ['XBTUSD', 'ETHUSD', 'XLMUSD', 'ZECUSD', 'ETCUSD', 'XMRUSD', 'ADAUSD']

if __name__ == "__main__":
    api = KrakenAPI()
    
    print("Updating all trading pairs from last DB entry...")
    for pair in PAIRS:
        print(f"\n{'='*40}")
        print(f"Updating {pair}...")
        try:
            rows = api.fetch_and_store(pair)
            print(f"✓ Added {rows} rows")
        except Exception as e:
            print(f"✗ Error: {e}")