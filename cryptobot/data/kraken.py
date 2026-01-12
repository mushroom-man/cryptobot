# -*- coding: utf-8 -*-
"""
CryptoBot - Kraken API Fetcher
===============================
Fetch OHLC data from Kraken REST API.
Use to backfill data or get recent updates.

Usage:
    from cryptobot.datasources.kraken_api import KrakenAPI
    
    api = KrakenAPI()
    df = api.get_ohlc("XBTUSD", interval=60)  # Hourly
    
    # Or fetch and save to database
    api.fetch_and_store("XBTUSD", interval=60, since="2024-10-01")
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from typing import Optional, List, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://cryptobot:cryptobot_dev@localhost:5432/cryptobot"
)

# Kraken API base URL
KRAKEN_API_URL = "https://api.kraken.com/0/public"

# Kraken pair name mapping (display name -> Kraken API name)
PAIR_MAPPING = {
    "XBTUSD": "XXBTZUSD",
    "ETHUSD": "XETHZUSD",
    "SOLUSD": "SOLUSD",
    "XRPUSD": "XXRPZUSD",
    "ADAUSD": "ADAUSD",
    "DOGEUSD": "XDGUSD",
    "LINKUSD": "LINKUSD",
    "MATICUSD": "MATICUSD",
    "AVAXUSD": "AVAXUSD",
    "DOTUSD": "DOTUSD",
    "LTCUSD": "XLTCZUSD",
}

# Reverse mapping
PAIR_MAPPING_REVERSE = {v: k for k, v in PAIR_MAPPING.items()}


class KrakenAPI:
    """Kraken REST API client for OHLC data."""
    
    def __init__(self, connection_url: str = None):
        """
        Initialize Kraken API client.
        
        Args:
            connection_url: Database connection string (optional)
        """
        self.connection_url = connection_url or DATABASE_URL
        self._engine = None
        self.rate_limit_delay = 2  # seconds between requests
    
    @property
    def engine(self):
        """Lazy-load database engine."""
        if self._engine is None:
            self._engine = create_engine(self.connection_url)
        return self._engine
    
    def _get_kraken_pair(self, pair: str) -> str:
        """Convert display pair name to Kraken API name."""
        return PAIR_MAPPING.get(pair, pair)
    
    def _get_display_pair(self, kraken_pair: str) -> str:
        """Convert Kraken API name to display pair name."""
        return PAIR_MAPPING_REVERSE.get(kraken_pair, kraken_pair)
    
    def get_ohlc(
        self,
        pair: str,
        interval: int = 60,
        since: int = None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Get OHLC data from Kraken API.
        
        Args:
            pair: Trading pair (e.g., "XBTUSD")
            interval: Interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Unix timestamp to get data from (optional)
        
        Returns:
            Tuple of (DataFrame, last_timestamp)
            DataFrame has columns: timestamp, open, high, low, close, vwap, volume, count
        """
        kraken_pair = self._get_kraken_pair(pair)
        
        params = {
            "pair": kraken_pair,
            "interval": interval
        }
        if since:
            params["since"] = since
        
        response = requests.get(f"{KRAKEN_API_URL}/OHLC", params=params)
        data = response.json()
        
        if data.get("error") and len(data["error"]) > 0:
            raise Exception(f"Kraken API error: {data['error']}")
        
        result = data.get("result", {})
        
        # Find the data key (Kraken uses various naming conventions)
        data_key = None
        for key in result.keys():
            if key != "last":
                data_key = key
                break
        
        if not data_key:
            return pd.DataFrame(), None
        
        ohlc_data = result[data_key]
        last_timestamp = result.get("last")
        
        if not ohlc_data:
            return pd.DataFrame(), last_timestamp
        
        # Convert to DataFrame
        # Kraken returns: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(ohlc_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
            df[col] = df[col].astype(float)
        df['count'] = df['count'].astype(int)
        
        return df, last_timestamp
    
    def get_ohlc_since_date(
        self,
        pair: str,
        since_date: str,
        interval: int = 60,
        max_requests: int = 50
    ) -> pd.DataFrame:
        """
        Get all OHLC data from a specific date to now.
        Makes multiple API calls to get past the 720 bar limit.
        
        Args:
            pair: Trading pair (e.g., "XBTUSD")
            since_date: Start date (ISO format: "2024-10-01")
            interval: Interval in minutes
            max_requests: Maximum API calls to prevent runaway loops
        
        Returns:
            DataFrame with all OHLC data
        """
        # Convert date to timestamp
        since_dt = pd.to_datetime(since_date)
        since_ts = int(since_dt.timestamp())
        
        print(f"\nFetching {pair} data from {since_date}...")
        print(f"Interval: {interval} minutes")
        
        all_data = []
        request_count = 0
        current_since = since_ts
        
        while request_count < max_requests:
            request_count += 1
            
            try:
                df, last_ts = self.get_ohlc(pair, interval, since=current_since)
            except Exception as e:
                print(f"Error on request {request_count}: {e}")
                break
            
            if len(df) == 0:
                print(f"No more data after request {request_count}")
                break
            
            all_data.append(df)
            
            # Progress update
            latest_date = df['timestamp'].max()
            print(f"  Request {request_count}: Got {len(df)} bars, latest: {latest_date}", end='\r')
            
            # Check if we've reached current time (less than 720 bars returned)
            if len(df) < 720:
                print(f"\n  Reached current data after {request_count} requests")
                break
            
            # Update since for next request
            current_since = last_ts
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates and sort
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\nTotal: {len(combined)} bars from {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        
        return combined
    
    def fetch_and_store(
        self,
        pair: str,
        interval: int = 60,
        since_date: str = None,
        if_exists: str = "append"
    ) -> int:
        """
        Fetch OHLC data from Kraken and store in database.
        
        Args:
            pair: Trading pair (e.g., "XBTUSD")
            interval: Interval in minutes (60 for hourly)
            since_date: Start date. If None, fetches from last database entry.
            if_exists: "append" or "replace"
        
        Returns:
            Number of rows inserted
        """
        # Determine start date
        if since_date is None:
            # Get last timestamp from database
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT MAX(timestamp) FROM ohlcv WHERE pair = :pair"),
                    {"pair": pair}
                )
                last_ts = result.scalar()
                
                if last_ts:
                    since_date = (last_ts + timedelta(hours=1)).strftime("%Y-%m-%d")
                    print(f"Continuing from last database entry: {last_ts}")
                else:
                    print("No existing data found. Please specify since_date.")
                    return 0
        
        # Fetch data
        df = self.get_ohlc_since_date(pair, since_date, interval)
        
        if len(df) == 0:
            print("No new data to insert")
            return 0
        
        # Prepare for database
        df_to_insert = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_to_insert['pair'] = pair
        df_to_insert['source'] = 'kraken_api'
        df_to_insert['volume_quote'] = None
        
        # Handle existing data
        if if_exists == "append":
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT MAX(timestamp) FROM ohlcv WHERE pair = :pair"),
                    {"pair": pair}
                )
                last_ts = result.scalar()
                
                if last_ts:
                    df_to_insert = df_to_insert[df_to_insert['timestamp'] > last_ts]
                    print(f"Filtering to {len(df_to_insert)} new rows after {last_ts}")
        
        if len(df_to_insert) == 0:
            print("No new data to insert after filtering")
            return 0
        
        # Insert
        print(f"Inserting {len(df_to_insert)} rows...")
        
        df_to_insert.to_sql(
            'ohlcv',
            self.engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        
        print(f"Done! Inserted {len(df_to_insert)} rows for {pair}")
        
        return len(df_to_insert)
    
    def backfill_multiple_pairs(
        self,
        pairs: List[str],
        since_date: str,
        interval: int = 60
    ) -> dict:
        """
        Backfill multiple pairs from a specific date.
        
        Args:
            pairs: List of trading pairs
            since_date: Start date (ISO format)
            interval: Interval in minutes
        
        Returns:
            Dict of pair -> rows inserted
        """
        results = {}
        
        for pair in pairs:
            print(f"\n{'='*60}")
            print(f"Processing {pair}")
            print(f"{'='*60}")
            
            try:
                rows = self.fetch_and_store(pair, interval, since_date)
                results[pair] = rows
            except Exception as e:
                print(f"Error processing {pair}: {e}")
                results[pair] = 0
            
            # Delay between pairs
            time.sleep(3)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for pair, rows in results.items():
            print(f"  {pair}: {rows:,} rows")
        
        return results
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs from Kraken."""
        response = requests.get(f"{KRAKEN_API_URL}/AssetPairs")
        data = response.json()
        
        if data.get("error"):
            raise Exception(f"Kraken API error: {data['error']}")
        
        pairs = list(data.get("result", {}).keys())
        
        # Filter to USD pairs
        usd_pairs = [p for p in pairs if p.endswith("USD") or p.endswith("ZUSD")]
        
        return sorted(usd_pairs)


# Convenience functions
def backfill_from_kraken(
    pairs: List[str],
    since_date: str = "2024-10-01"
) -> dict:
    """
    Backfill data for multiple pairs from Kraken API.
    
    Usage:
        from cryptobot.datasources.kraken_api import backfill_from_kraken
        backfill_from_kraken(["XBTUSD", "ETHUSD"], "2024-10-01")
    """
    api = KrakenAPI()
    return api.backfill_multiple_pairs(pairs, since_date)


def update_pair(pair: str) -> int:
    """
    Update a single pair with latest data from Kraken.
    Automatically continues from last database entry.
    
    Usage:
        from cryptobot.datasources.kraken_api import update_pair
        update_pair("XBTUSD")
    """
    api = KrakenAPI()
    return api.fetch_and_store(pair)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch data from Kraken API")
    parser.add_argument("--pairs", "-p", nargs="+", default=["XBTUSD"], 
                        help="Trading pairs to fetch")
    parser.add_argument("--since", "-s", default="2024-10-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--interval", "-i", type=int, default=60,
                        help="Interval in minutes (default: 60)")
    
    args = parser.parse_args()
    
    api = KrakenAPI()
    api.backfill_multiple_pairs(args.pairs, args.since, args.interval)