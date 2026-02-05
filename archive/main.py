# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 11:02:50 2026

@author: John

CryptoBot - Main Entry Point
=============================
Entry point for the CryptoBot container.

Modes:
    - idle: Keep container alive for interactive use (default)
    - test: Run module tests and exit
    - info: Print system info and exit

Usage:
    python -m cryptobot.main          # Idle mode (default)
    python -m cryptobot.main --test   # Run tests
    python -m cryptobot.main --info   # Print info
"""

import argparse
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cryptobot')


def print_banner():
    """Print startup banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                      CryptoBot                            ║
║              Multi-Timeframe Trading Platform             ║
╚═══════════════════════════════════════════════════════════╝
    """)


def check_imports():
    """Check all modules can be imported."""
    modules = []
    errors = []
    
    # Core datasources
    try:
        from cryptobot.datasources import Database, DataLoader
        modules.append("datasources.Database")
        modules.append("datasources.DataLoader")
    except ImportError as e:
        errors.append(f"datasources: {e}")
    
    try:
        from cryptobot.datasources import Resampler, GapHandler, Aligner
        modules.append("datasources.Resampler")
        modules.append("datasources.GapHandler")
        modules.append("datasources.Aligner")
    except ImportError as e:
        errors.append(f"datasources components: {e}")
    
    # Features (if exists)
    try:
        from cryptobot.features import FeatureEngine
        modules.append("features.FeatureEngine")
    except ImportError:
        pass  # Optional module
    
    # Research/backtest (if exists)
    try:
        from cryptobot.research.backtest import BacktestRunner
        modules.append("research.backtest.BacktestRunner")
    except ImportError:
        pass  # Optional module
    
    return modules, errors


def check_database():
    """Check database connection."""
    try:
        from cryptobot.datasources import Database
        db = Database()
        pairs = db.get_available_pairs()
        return True, pairs
    except Exception as e:
        return False, str(e)


def print_info():
    """Print system information."""
    print_banner()
    
    print("Checking imports...")
    modules, errors = check_imports()
    
    print(f"\n✓ Loaded modules ({len(modules)}):")
    for mod in modules:
        print(f"    - {mod}")
    
    if errors:
        print(f"\n✗ Import errors ({len(errors)}):")
        for err in errors:
            print(f"    - {err}")
    
    print("\nChecking database...")
    db_ok, result = check_database()
    
    if db_ok:
        print(f"✓ Database connected")
        print(f"    Available pairs: {result}")
    else:
        print(f"✗ Database error: {result}")
    
    print("\n" + "=" * 50)


def run_tests():
    """Run basic module tests."""
    print_banner()
    print("Running tests...\n")
    
    import pandas as pd
    import numpy as np
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    df = pd.DataFrame({
        'open': 50000 + np.random.randn(500).cumsum() * 100,
        'high': 50100 + np.random.randn(500).cumsum() * 100,
        'low': 49900 + np.random.randn(500).cumsum() * 100,
        'close': 50000 + np.random.randn(500).cumsum() * 100,
        'volume': np.abs(np.random.randn(500)) * 100,
    }, index=dates)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test Resampler
    try:
        from cryptobot.datasources import Resampler
        df_24h = Resampler.resample(df, '24h')
        assert len(df_24h) > 0
        print(f"✓ Resampler: {len(df)} → {len(df_24h)} rows")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Resampler: {e}")
        tests_failed += 1
    
    # Test GapHandler
    try:
        from cryptobot.datasources import GapHandler
        gaps = GapHandler.detect(df)
        print(f"✓ GapHandler: detected {len(gaps)} gaps")
        tests_passed += 1
    except Exception as e:
        print(f"✗ GapHandler: {e}")
        tests_failed += 1
    
    # Test Aligner
    try:
        from cryptobot.datasources import Aligner, Resampler
        df_24h = Resampler.resample(df, '24h')
        df_aligned = Aligner.align(df_24h, df.index, '24h')
        assert len(df_aligned) == len(df)
        print(f"✓ Aligner: aligned to {len(df_aligned)} rows")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Aligner: {e}")
        tests_failed += 1
    
    # Test Database
    try:
        from cryptobot.datasources import Database
        db = Database()
        pairs = db.get_available_pairs()
        print(f"✓ Database: {len(pairs)} pairs available")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Database: {e}")
        tests_failed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Tests: {tests_passed} passed, {tests_failed} failed")
    print(f"{'=' * 50}")
    
    return tests_failed == 0


def idle_mode():
    """Keep container alive for interactive use."""
    print_banner()
    
    # Check imports on startup
    modules, errors = check_imports()
    print(f"✓ Loaded {len(modules)} modules")
    
    if errors:
        print(f"✗ {len(errors)} import errors (run with --info for details)")
    
    # Check database
    db_ok, result = check_database()
    if db_ok:
        print(f"✓ Database connected ({len(result)} pairs)")
    else:
        print(f"! Database not available: {result}")
    
    print("\nContainer ready for interactive use.")
    print("Connect with: docker exec -it cryptobot-app python")
    print("\n" + "=" * 50)
    
    # Keep alive
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CryptoBot Trading Platform')
    parser.add_argument('--test', action='store_true', help='Run tests and exit')
    parser.add_argument('--info', action='store_true', help='Print system info and exit')
    
    args = parser.parse_args()
    
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    elif args.info:
        print_info()
        sys.exit(0)
    else:
        idle_mode()


if __name__ == "__main__":
    main()