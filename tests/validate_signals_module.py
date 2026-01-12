# -*- coding: utf-8 -*-
"""
Validation: Compare backtest functions vs signals module
=========================================================
Ensures the extracted signals module produces IDENTICAL results
to the original backtest implementation.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cryptobot.data.database import Database
from cryptobot.signals import (
    SignalGenerator,
    resample_ohlcv,
    label_trend_binary,
    generate_32state_signals,
    calculate_expanding_hit_rates,
    MA_PERIOD_24H,
    MA_PERIOD_72H,
    MA_PERIOD_168H,
    ENTRY_BUFFER,
    EXIT_BUFFER,
)

# =============================================================================
# ORIGINAL BACKTEST FUNCTIONS (copied verbatim for comparison)
# =============================================================================

def original_resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Original from backtest line 278-282"""
    return df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


def original_label_trend_binary(df, ma_period, entry_buffer=0.015, exit_buffer=0.005):
    """Original from backtest line 285-310"""
    close = df['close']
    ma = close.rolling(ma_period).mean()
    
    labels = pd.Series(index=df.index, dtype=int)
    current = 1
    
    for i in range(len(df)):
        if pd.isna(ma.iloc[i]):
            labels.iloc[i] = current
            continue
        
        price = close.iloc[i]
        ma_val = ma.iloc[i]
        
        if current == 1:
            if price < ma_val * (1 - exit_buffer) and price < ma_val * (1 - entry_buffer):
                current = 0
        else:
            if price > ma_val * (1 + exit_buffer) and price > ma_val * (1 + entry_buffer):
                current = 1
        
        labels.iloc[i] = current
    
    return labels


def original_generate_32state_signals(df_24h, df_72h, df_168h):
    """Original from backtest line 313-338"""
    MA_24H = 16
    MA_72H = 6
    MA_168H = 2
    
    trend_24h = original_label_trend_binary(df_24h, MA_24H)
    trend_72h = original_label_trend_binary(df_72h, MA_72H)
    trend_168h = original_label_trend_binary(df_168h, MA_168H)
    
    ma_24h = df_24h['close'].rolling(MA_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_168H).mean()
    
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_resample(df_1h):
    """Test resampling produces identical results."""
    print("\n" + "="*60)
    print("TEST 1: Resampling")
    print("="*60)
    
    for tf in ['24h', '72h', '168h']:
        orig = original_resample_ohlcv(df_1h, tf)
        new = resample_ohlcv(df_1h, tf)
        
        match = orig.equals(new)
        print(f"  {tf}: {'? MATCH' if match else '? MISMATCH'} ({len(orig)} rows)")
        
        if not match:
            diff = (orig != new).sum().sum()
            print(f"    Differences: {diff} cells")
            return False
    
    return True


def test_trend_labeling(df_1h):
    """Test trend labeling produces identical results."""
    print("\n" + "="*60)
    print("TEST 2: Trend Labeling (with hysteresis)")
    print("="*60)
    
    test_cases = [
        ('24h', MA_PERIOD_24H),
        ('72h', MA_PERIOD_72H),
        ('168h', MA_PERIOD_168H),
    ]
    
    for tf, ma_period in test_cases:
        df_tf = resample_ohlcv(df_1h, tf)
        
        orig = original_label_trend_binary(df_tf, ma_period, ENTRY_BUFFER, EXIT_BUFFER)
        new = label_trend_binary(df_tf, ma_period, ENTRY_BUFFER, EXIT_BUFFER)
        
        match = orig.equals(new)
        print(f"  {tf} (MA{ma_period}): {'? MATCH' if match else '? MISMATCH'}")
        
        if not match:
            diff_count = (orig != new).sum()
            print(f"    Differences: {diff_count} rows")
            # Show first few differences
            diff_idx = orig[orig != new].index[:3]
            for idx in diff_idx:
                print(f"    {idx}: orig={orig.loc[idx]}, new={new.loc[idx]}")
            return False
    
    return True


def test_32state_signals(df_1h):
    """Test 32-state signal generation produces identical results."""
    print("\n" + "="*60)
    print("TEST 3: 32-State Signal Generation")
    print("="*60)
    
    # Resample
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    # Generate signals
    orig = original_generate_32state_signals(df_24h, df_72h, df_168h)
    new = generate_32state_signals(df_24h, df_72h, df_168h)
    
    print(f"  Original shape: {orig.shape}")
    print(f"  New shape:      {new.shape}")
    
    # Check index alignment
    if not orig.index.equals(new.index):
        print("  ? Index mismatch!")
        print(f"    Original: {orig.index[0]} to {orig.index[-1]}")
        print(f"    New:      {new.index[0]} to {new.index[-1]}")
        return False
    
    # Check each column
    all_match = True
    for col in orig.columns:
        col_match = orig[col].equals(new[col])
        print(f"  {col}: {'? MATCH' if col_match else '? MISMATCH'}")
        
        if not col_match:
            all_match = False
            diff_count = (orig[col] != new[col]).sum()
            print(f"    Differences: {diff_count} rows")
    
    return all_match


def test_signal_generator_class(df_1h):
    """Test the SignalGenerator class wrapper."""
    print("\n" + "="*60)
    print("TEST 4: SignalGenerator Class")
    print("="*60)
    
    gen = SignalGenerator()
    
    # Generate signals via class
    signals = gen.generate_signals(df_1h)
    
    # Generate signals via functions
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    direct_signals = generate_32state_signals(df_24h, df_72h, df_168h)
    
    match = signals.equals(direct_signals)
    print(f"  Class vs direct functions: {'? MATCH' if match else '? MISMATCH'}")
    
    # Test returns calculation
    returns = gen.get_returns(df_1h)
    direct_returns = df_24h['close'].pct_change()
    returns_match = returns.equals(direct_returns)
    print(f"  Returns calculation: {'? MATCH' if returns_match else '? MISMATCH'}")
    
    return match and returns_match


def test_hit_rate_calculation(df_1h):
    """Test hit rate calculation."""
    print("\n" + "="*60)
    print("TEST 5: Hit Rate Calculation")
    print("="*60)
    
    gen = SignalGenerator()
    signals = gen.generate_signals(df_1h)
    returns = gen.get_returns(df_1h)
    
    # Calculate hit rates
    hit_rates = calculate_expanding_hit_rates(returns, signals)
    
    # Check structure
    n_states = len(hit_rates)
    print(f"  States calculated: {n_states} (expected 32)")
    
    if n_states != 32:
        print("  ? Wrong number of states!")
        return False
    
    # Check data integrity
    total_samples = sum(hr['n'] for hr in hit_rates.values())
    sufficient_states = sum(1 for hr in hit_rates.values() if hr['sufficient'])
    
    print(f"  Total samples: {total_samples}")
    print(f"  Sufficient states (n>=20): {sufficient_states}/32")
    
    # Verify hit rates are valid
    for key, data in hit_rates.items():
        if not (0 <= data['hit_rate'] <= 1):
            print(f"  ? Invalid hit rate for {key}: {data['hit_rate']}")
            return False
    
    print("  ? All hit rates valid (0-1)")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("SIGNALS MODULE VALIDATION")
    print("Comparing extracted module vs original backtest functions")
    print("="*60)
    
    # Load data
    print("\nLoading data from database...")
    db = Database()
    
    # Use XBTUSD as test case (longest history)
    df_1h = db.get_ohlcv('XBTUSD')
    print(f"Loaded {len(df_1h)} 1h bars for XBTUSD")
    print(f"Date range: {df_1h.index[0]} to {df_1h.index[-1]}")
    
    # Run tests
    results = []
    
    results.append(("Resampling", test_resample(df_1h)))
    results.append(("Trend Labeling", test_trend_labeling(df_1h)))
    results.append(("32-State Signals", test_32state_signals(df_1h)))
    results.append(("SignalGenerator Class", test_signal_generator_class(df_1h)))
    results.append(("Hit Rate Calculation", test_hit_rate_calculation(df_1h)))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "? PASS" if passed else "? FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("? ALL TESTS PASSED - Module is validated!")
        print("  Safe to use signals module in backtest and live trading.")
    else:
        print("? VALIDATION FAILED - Do NOT use module until fixed!")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
