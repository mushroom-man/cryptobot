# -*- coding: utf-8 -*-
"""
CryptoBot - CSV to Database Loader
===================================
Loads OHLCV data from CSV files into TimescaleDB.
Automatically detects format (CryptoCompare or Kraken).

Usage:
    python -m cryptobot.datasources.csv_loader --file /path/to/file.csv --pair XBTUSD

Or from Python:
    from cryptobot.datasources import load_csv_to_db
    load_csv_to_db("path/to/file.csv", "XBTUSD")
"""

import pandas as pd
import argparse
from datetime import datetime
from sqlalchemy import create_engine, text
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Database connection

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable is not set. Check your .env file.")

def detect_format(df: pd.DataFrame) -> str:
    """
    Detect CSV format based on column names.
    
    Returns:
        'cryptocompare', 'kraken', or 'unknown'
    """
    columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    
    # CryptoCompare has 'time', 'volumefrom', 'volumeto'
    if 'time' in columns and 'volumefrom' in columns:
        return 'cryptocompare'
    
    # Kraken has numeric columns (0-6) or named: timestamp, open, high, low, close, volume, trades
    if all(isinstance(c, int) for c in df.columns):
        return 'kraken'
    
    # Kraken with headers
    if 'trades' in columns or 'trade_count' in columns:
        return 'kraken'
    
    # Check for standard OHLCV columns
    if 'open' in columns and 'high' in columns and 'low' in columns and 'close' in columns:
        return 'generic'
    
    return 'unknown'


def map_cryptocompare(df: pd.DataFrame) -> pd.DataFrame:
    """Map CryptoCompare columns to standard format."""
    column_mapping = {
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volumefrom': 'volume',
        'volumeto': 'volume_quote',
    }
    
    available_cols = [c for c in column_mapping.keys() if c in df.columns]
    df_mapped = df[available_cols].rename(columns=column_mapping)
    
    # Parse timestamp
    df_mapped['timestamp'] = pd.to_datetime(df_mapped['timestamp'], unit='s', utc=True)
    
    return df_mapped


def map_kraken(df: pd.DataFrame) -> pd.DataFrame:
    """Map Kraken columns to standard format."""
    
    # Kraken files often have no header - columns are positional:
    # 0: timestamp (unix), 1: open, 2: high, 3: low, 4: close, 5: volume, 6: trades
    if all(isinstance(c, int) for c in df.columns):
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
    
    # Lowercase all column names
    df.columns = [c.lower() for c in df.columns]
    
    # Select relevant columns
    cols_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_mapped = df[[c for c in cols_to_keep if c in df.columns]].copy()
    
    # Parse timestamp - Kraken uses Unix timestamp
    if df_mapped['timestamp'].dtype in ['int64', 'float64']:
        df_mapped['timestamp'] = pd.to_datetime(df_mapped['timestamp'], unit='s', utc=True)
    else:
        df_mapped['timestamp'] = pd.to_datetime(df_mapped['timestamp'])
    
    # Kraken doesn't have volume_quote
    df_mapped['volume_quote'] = None
    
    return df_mapped


def map_generic(df: pd.DataFrame) -> pd.DataFrame:
    """Map generic OHLCV columns to standard format."""
    df.columns = [c.lower() for c in df.columns]
    
    # Try to find timestamp column
    timestamp_cols = ['timestamp', 'time', 'date', 'datetime']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        raise ValueError("Could not find timestamp column")
    
    df_mapped = pd.DataFrame()
    df_mapped['timestamp'] = pd.to_datetime(df[timestamp_col], utc=True)
    df_mapped['open'] = df['open']
    df_mapped['high'] = df['high']
    df_mapped['low'] = df['low']
    df_mapped['close'] = df['close']
    df_mapped['volume'] = df.get('volume', df.get('volumefrom', None))
    df_mapped['volume_quote'] = df.get('volume_quote', df.get('volumeto', None))
    
    return df_mapped


def load_csv_to_db(
    file_path: str,
    pair: str,
    source: str = None,
    if_exists: str = "append"
) -> int:
    """
    Load a CSV file into the ohlcv table.
    Automatically detects format (CryptoCompare, Kraken, or generic).
    
    Args:
        file_path: Path to CSV file
        pair: Trading pair (e.g., "XBTUSD")
        source: Data source name (auto-detected if None)
        if_exists: How to handle existing data ("append" or "replace")
    
    Returns:
        Number of rows inserted
    """
    
    print(f"\n{'='*60}")
    print(f"Loading {pair} data from CSV")
    print(f"{'='*60}")
    
    # Load CSV
    print(f"\n[1/5] Reading CSV: {file_path}")
    
    # Try reading with and without header
    try:
        df = pd.read_csv(file_path)
        # Check if first row looks like data (numeric) rather than header
        if df.iloc[0].apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit()).all():
            df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"      Error reading CSV: {e}")
        raise
    
    print(f"      Rows in file: {len(df):,}")
    print(f"      Columns: {list(df.columns)}")
    
    # Detect format
    print(f"\n[2/5] Detecting format...")
    file_format = detect_format(df)
    print(f"      Format detected: {file_format}")
    
    if source is None:
        source = file_format if file_format != 'unknown' else 'csv'
    
    # Map columns
    print(f"\n[3/5] Mapping columns...")
    if file_format == 'cryptocompare':
        df_mapped = map_cryptocompare(df)
    elif file_format == 'kraken':
        df_mapped = map_kraken(df)
    elif file_format == 'generic':
        df_mapped = map_generic(df)
    else:
        raise ValueError(f"Unknown CSV format. Columns: {list(df.columns)}")
    
    # Add metadata
    df_mapped['pair'] = pair
    df_mapped['source'] = source
    
    # Process timestamps
    print(f"\n[4/5] Processing data...")
    df_mapped = df_mapped.set_index('timestamp')
    df_mapped = df_mapped.sort_index()
    
    # Remove duplicates
    before_dedup = len(df_mapped)
    df_mapped = df_mapped[~df_mapped.index.duplicated(keep='last')]
    after_dedup = len(df_mapped)
    if before_dedup != after_dedup:
        print(f"      Removed {before_dedup - after_dedup} duplicate timestamps")
    
    print(f"      Date range: {df_mapped.index.min()} to {df_mapped.index.max()}")
    print(f"      Rows to process: {len(df_mapped):,}")
    
    # Connect to database
    print(f"\n[5/5] Inserting into database...")
    engine = create_engine(DATABASE_URL)
    
    # Check for existing data
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM ohlcv WHERE pair = :pair"),
            {"pair": pair}
        )
        existing_count = result.scalar()
        
        if existing_count > 0:
            print(f"      Existing rows for {pair}: {existing_count:,}")
            
            if if_exists == "replace":
                print(f"      Deleting existing data...")
                conn.execute(
                    text("DELETE FROM ohlcv WHERE pair = :pair"),
                    {"pair": pair}
                )
                conn.commit()
            else:
                # Get date range of existing data
                result = conn.execute(
                    text("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv WHERE pair = :pair"),
                    {"pair": pair}
                )
                row = result.fetchone()
                print(f"      Existing range: {row[0]} to {row[1]}")
                
                # Filter to only new data
                df_mapped = df_mapped[df_mapped.index > row[1]]
                print(f"      New rows to append: {len(df_mapped):,}")
    
    if len(df_mapped) == 0:
        print(f"\n      No new data to insert.")
        return 0
    
    # Prepare for insert
    df_to_insert = df_mapped.reset_index()
    
    # Ensure column order matches table
    columns = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume', 'source', 'volume_quote']
    for col in columns:
        if col not in df_to_insert.columns:
            df_to_insert[col] = None
    df_to_insert = df_to_insert[columns]
    
    # Insert in chunks
    chunk_size = 10000
    total_inserted = 0
    
    for i in range(0, len(df_to_insert), chunk_size):
        chunk = df_to_insert.iloc[i:i+chunk_size]
        chunk.to_sql(
            'ohlcv',
            engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        total_inserted += len(chunk)
        print(f"      Inserted {total_inserted:,} / {len(df_to_insert):,} rows", end='\r')
    
    print(f"\n      Done! Inserted {total_inserted:,} rows.")
    
    # Verify
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM ohlcv WHERE pair = :pair"),
            {"pair": pair}
        )
        final_count = result.scalar()
        print(f"\n      Total rows for {pair} in database: {final_count:,}")
    
    return total_inserted


def load_multiple_pairs(
    directory: str,
    pairs: dict,
    interval: str = "60",
    if_exists: str = "append"
) -> dict:
    """
    Load multiple pairs from Kraken download directory.
    
    Args:
        directory: Path to extracted Kraken data directory
        pairs: Dict mapping filename pattern to pair name, e.g.:
               {"XBTUSD": "XBTUSD", "ETHUSD": "ETHUSD"}
        interval: Interval to load ("60" for hourly)
        if_exists: How to handle existing data
    
    Returns:
        Dict of pair -> rows inserted
    """
    results = {}
    directory = Path(directory)
    
    for file_pattern, pair_name in pairs.items():
        # Look for file matching pattern
        pattern = f"*{file_pattern}*{interval}*.csv"
        matches = list(directory.glob(pattern))
        
        if not matches:
            print(f"No file found for {file_pattern} with interval {interval}")
            continue
        
        file_path = matches[0]
        print(f"\nLoading {pair_name} from {file_path.name}")
        
        try:
            rows = load_csv_to_db(
                str(file_path),
                pair_name,
                source="kraken",
                if_exists=if_exists
            )
            results[pair_name] = rows
        except Exception as e:
            print(f"Error loading {pair_name}: {e}")
            results[pair_name] = 0
    
    return results


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Load CSV data into TimescaleDB")
    parser.add_argument("--file", "-f", required=True, help="Path to CSV file")
    parser.add_argument("--pair", "-p", required=True, help="Trading pair (e.g., XBTUSD)")
    parser.add_argument("--source", "-s", default=None, help="Data source name (auto-detected if not specified)")
    parser.add_argument("--replace", action="store_true", help="Replace existing data")
    
    args = parser.parse_args()
    
    if_exists = "replace" if args.replace else "append"
    
    load_csv_to_db(
        file_path=args.file,
        pair=args.pair,
        source=args.source,
        if_exists=if_exists
    )


if __name__ == "__main__":
    main()