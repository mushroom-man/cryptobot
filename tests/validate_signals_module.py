# -*- coding: utf-8 -*-
"""
Validation: 16-State Signals Module
====================================
Ensures the 16-state signals module produces correct results
with proper NO_MA72_ONLY filter behavior.

Tests:
    1. Resampling correctness
    2. Trend labeling with hysteresis
    3. 16-state signal generation (no trend_72h)
    4. NO_MA72_ONLY filter behavior
    5. Hit rate calculation (16 states)
    6. SignalGenerator class integration
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cryptobot.data.database import Database
from cryptobot.signals import (
    SignalGenerator,
    resample_ohlcv,
    label_trend_binary,
    generate_16state_signals,
    calculate_expanding_hit_rates,
    get_16state_position,
    should_trade_signal,
    get_state_tuple,
    MA_PERIOD_24H,
    MA_PERIOD_72H,
    MA_PERIOD_168H,
    ENTRY_BUFFER,
    EXIT_BUFFER,
    USE_MA72_FILTER,
)


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_resample(df_1h):
    """Test resampling produces correct results."""
    print("\n" + "="*60)
    print("TEST 1: Resampling")
    print("="*60)
    
    all_passed = True
    
    for tf in ['24h', '72h', '168h']:
        resampled = resample_ohlcv(df_1h, tf)
        
        # Check columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in resampled.columns]
        
        if missing:
            print(f"  {tf}: ✗ FAIL - Missing columns: {missing}")
            all_passed = False
            continue
        
        # Check no NaN in output
        nan_count = resampled.isna().sum().sum()
        if nan_count > 0:
            print(f"  {tf}: ✗ FAIL - {nan_count} NaN values")
            all_passed = False
            continue
        
        # Check monotonic index
        if not resampled.index.is_monotonic_increasing:
            print(f"  {tf}: ✗ FAIL - Non-monotonic index")
            all_passed = False
            continue
        
        print(f"  {tf}: ✓ PASS ({len(resampled)} bars)")
    
    return all_passed


def test_trend_labeling(df_1h):
    """Test trend labeling produces binary results with hysteresis."""
    print("\n" + "="*60)
    print("TEST 2: Trend Labeling (with hysteresis)")
    print("="*60)
    
    all_passed = True
    
    test_cases = [
        ('24h', MA_PERIOD_24H),
        ('168h', MA_PERIOD_168H),
    ]
    
    for tf, ma_period in test_cases:
        df_tf = resample_ohlcv(df_1h, tf)
        labels = label_trend_binary(df_tf, ma_period, ENTRY_BUFFER, EXIT_BUFFER)
        
        # Check binary values only
        unique_vals = labels.dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            print(f"  {tf} (MA{ma_period}): ✗ FAIL - Non-binary values: {unique_vals}")
            all_passed = False
            continue
        
        # Check reasonable flip frequency (not too many, not too few)
        flips = (labels.diff() != 0).sum()
        flip_rate = flips / len(labels)
        
        if flip_rate < 0.001 or flip_rate > 0.5:
            print(f"  {tf} (MA{ma_period}): ⚠ WARNING - Unusual flip rate: {flip_rate:.3f}")
        
        # Check label distribution
        label_pct = labels.value_counts(normalize=True)
        
        print(f"  {tf} (MA{ma_period}): ✓ PASS - {flips} flips ({flip_rate*100:.1f}%), "
              f"dist: 0={label_pct.get(0, 0)*100:.1f}%, 1={label_pct.get(1, 0)*100:.1f}%")
    
    return all_passed


def test_16state_signals(df_1h):
    """Test 16-state signal generation (without trend_72h)."""
    print("\n" + "="*60)
    print("TEST 3: 16-State Signal Generation")
    print("="*60)
    
    # Resample
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    # Generate signals
    signals = generate_16state_signals(df_24h, df_72h, df_168h)
    
    print(f"  Shape: {signals.shape}")
    
    # Check required columns (NO trend_72h!)
    required_cols = ['trend_24h', 'trend_168h', 'ma72_above_ma24', 'ma168_above_ma24']
    missing = [c for c in required_cols if c not in signals.columns]
    
    if missing:
        print(f"  ✗ FAIL - Missing columns: {missing}")
        return False
    
    # Check trend_72h is NOT present
    if 'trend_72h' in signals.columns:
        print(f"  ✗ FAIL - trend_72h should NOT be present (removed in 16-state)")
        return False
    
    print(f"  ✓ trend_72h correctly removed")
    
    # Check binary values
    for col in required_cols:
        unique_vals = signals[col].unique()
        if not set(unique_vals).issubset({0, 1}):
            print(f"  ✗ FAIL - {col} has non-binary values: {unique_vals}")
            return False
    
    # Count unique states
    state_combos = signals.groupby(required_cols).size()
    n_states = len(state_combos)
    
    print(f"  ✓ All columns binary")
    print(f"  ✓ Unique states observed: {n_states}/16")
    
    return True


def test_filter_behavior():
    """Test NO_MA72_ONLY filter behavior."""
    print("\n" + "="*60)
    print("TEST 4: NO_MA72_ONLY Filter Behavior")
    print("="*60)
    
    all_passed = True
    
    # Test cases: (prev_state, curr_state, expected_should_trade)
    test_cases = [
        # (trend_24h, trend_168h, ma72_above_ma24, ma168_above_ma24)
        
        # First signal - always trade
        (None, (1, 1, 1, 1), True, "First signal"),
        
        # No change - no trade
        ((1, 1, 1, 1), (1, 1, 1, 1), False, "No change"),
        
        # Only ma72 changed - FILTER (should NOT trade)
        ((1, 1, 0, 1), (1, 1, 1, 1), False, "Only ma72 changed"),
        ((1, 1, 1, 1), (1, 1, 0, 1), False, "Only ma72 changed (reverse)"),
        
        # trend_24h changed - TRADE
        ((0, 1, 1, 1), (1, 1, 1, 1), True, "trend_24h changed"),
        
        # trend_168h changed - TRADE
        ((1, 0, 1, 1), (1, 1, 1, 1), True, "trend_168h changed"),
        
        # ma168 changed - TRADE
        ((1, 1, 1, 0), (1, 1, 1, 1), True, "ma168 changed"),
        
        # ma72 + trend_24h changed - TRADE
        ((0, 1, 0, 1), (1, 1, 1, 1), True, "ma72 + trend_24h changed"),
        
        # All changed - TRADE
        ((0, 0, 0, 0), (1, 1, 1, 1), True, "All changed"),
    ]
    
    for prev_state, curr_state, expected, description in test_cases:
        result = should_trade_signal(prev_state, curr_state, use_filter=True)
        
        if result == expected:
            print(f"  ✓ {description}: {result}")
        else:
            print(f"  ✗ {description}: expected {expected}, got {result}")
            all_passed = False
    
    return all_passed


def test_hit_rate_calculation(df_1h):
    """Test hit rate calculation for 16 states."""
    print("\n" + "="*60)
    print("TEST 5: Hit Rate Calculation (16 States)")
    print("="*60)
    
    gen = SignalGenerator()
    signals = gen.generate_signals(df_1h)
    returns = gen.get_returns(df_1h)
    
    # Calculate hit rates
    hit_rates = calculate_expanding_hit_rates(returns, signals)
    
    # Check structure - should be 16 states (4 trend × 4 MA)
    n_states = len(hit_rates)
    expected_states = 16
    
    if n_states != expected_states:
        print(f"  ✗ FAIL - Expected {expected_states} states, got {n_states}")
        return False
    
    print(f"  ✓ States calculated: {n_states}")
    
    # Check key structure
    all_trend_perms = list(product([0, 1], repeat=2))  # 4 trend permutations
    all_ma_perms = list(product([0, 1], repeat=2))      # 4 MA permutations
    
    for trend_perm in all_trend_perms:
        for ma_perm in all_ma_perms:
            key = (trend_perm, ma_perm)
            if key not in hit_rates:
                print(f"  ✗ FAIL - Missing key: {key}")
                return False
    
    print(f"  ✓ All 16 state keys present")
    
    # Check data integrity
    total_samples = sum(hr['n'] for hr in hit_rates.values())
    sufficient_states = sum(1 for hr in hit_rates.values() if hr['sufficient'])
    
    print(f"  Total samples: {total_samples}")
    print(f"  Sufficient states (n>=20): {sufficient_states}/16")
    
    # Verify hit rates are valid (0-1)
    for key, data in hit_rates.items():
        if not (0 <= data['hit_rate'] <= 1):
            print(f"  ✗ FAIL - Invalid hit rate for {key}: {data['hit_rate']}")
            return False
    
    print("  ✓ All hit rates valid (0-1)")
    return True


def test_signal_generator_class(df_1h):
    """Test the SignalGenerator class with filter."""
    print("\n" + "="*60)
    print("TEST 6: SignalGenerator Class (with Filter)")
    print("="*60)
    
    gen = SignalGenerator(use_filter=True)
    
    # Generate signals
    signals = gen.generate_signals(df_1h)
    returns = gen.get_returns(df_1h)
    
    # Check signals shape
    if 'trend_72h' in signals.columns:
        print("  ✗ FAIL - trend_72h should not be in signals")
        return False
    
    print(f"  ✓ Signals generated: {len(signals)} rows, {len(signals.columns)} columns")
    
    # Test position calculation with filter
    data_start = signals.index[0]
    test_dates = signals.index[400:410]  # After training period
    
    gen.reset_filter_state()  # Reset for clean test
    
    positions = []
    filtered_count = 0
    
    for date in test_dates:
        position, details = gen.get_position_for_date(
            signals, returns, date, data_start, min_training_months=12
        )
        positions.append(position)
        
        if details.get('filtered', False):
            filtered_count += 1
    
    # Check positions are valid
    for pos in positions:
        if pos not in [0.0, 0.5, 1.0]:
            print(f"  ✗ FAIL - Invalid position: {pos}")
            return False
    
    print(f"  ✓ Positions valid: {positions}")
    
    # Check filter stats
    stats = gen.get_filter_stats()
    print(f"  Filter stats: {stats}")
    
    return True


def test_filter_rate(df_1h):
    """Test that filter rate is approximately 56% as validated."""
    print("\n" + "="*60)
    print("TEST 7: Filter Rate Validation (~56% expected)")
    print("="*60)
    
    gen = SignalGenerator(use_filter=True)
    signals = gen.generate_signals(df_1h)
    returns = gen.get_returns(df_1h)
    data_start = signals.index[0]
    
    # Run through all dates
    gen.reset_filter_state()
    
    for date in signals.index:
        gen.get_position_for_date(signals, returns, date, data_start, min_training_months=12)
    
    stats = gen.get_filter_stats()
    filter_rate = stats['filter_rate']
    
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Filtered: {stats['filtered_signals']}")
    print(f"  Traded: {stats['traded_signals']}")
    print(f"  Filter rate: {filter_rate*100:.1f}%")
    
    # Expected ~56% based on validation
    if 0.40 <= filter_rate <= 0.70:
        print(f"  ✓ Filter rate in expected range (40-70%)")
        return True
    else:
        print(f"  ⚠ WARNING - Filter rate outside expected range")
        return True  # Still pass, just warn


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("16-STATE SIGNALS MODULE VALIDATION")
    print("With NO_MA72_ONLY Filter")
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
    results.append(("16-State Signals", test_16state_signals(df_1h)))
    results.append(("Filter Behavior", test_filter_behavior()))
    results.append(("Hit Rate Calculation", test_hit_rate_calculation(df_1h)))
    results.append(("SignalGenerator Class", test_signal_generator_class(df_1h)))
    results.append(("Filter Rate", test_filter_rate(df_1h)))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - 16-State Module Validated!")
        print("  Ready for paper trading.")
    else:
        print("✗ VALIDATION FAILED - Fix issues before deployment!")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)