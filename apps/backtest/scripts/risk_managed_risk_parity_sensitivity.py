#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST Parameter Sensitivity Analysis
====================================
Optimized version using:
1. Pre-compute ALL MAs upfront
2. Vectorized operations
3. Numba JIT for inner loops
4. Cache reusable calculations

Usage:
    python parameter_sensitivity_fast.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from itertools import product
import warnings
import time
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database

try:
    from numba import jit
    HAS_NUMBA = True
    print("✓ Numba available - using JIT compilation")
except ImportError:
    HAS_NUMBA = False
    print("✗ Numba not available - using pure Python")


# =============================================================================
# BASELINE PARAMETERS
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

BASELINE = {
    'ma_24h': 24,
    'ma_72h': 8,
    'ma_168h': 2,
    'entry_buffer': 0.02,
    'exit_buffer': 0.005,
    'hit_rate_threshold': 0.50,
    'target_vol': 0.40,
    'dd_start_reduce': -0.20,
    'min_exposure_floor': 0.40,
}

MIN_SAMPLES = 20
VOL_LOOKBACK = 30
DD_MIN_EXPOSURE = -0.50


# =============================================================================
# PARAMETER VARIATIONS (REDUCED FOR SPEED)
# =============================================================================

SENSITIVITY_TESTS = {
    'ma_24h': [16, 24, 32],
    'ma_72h': [5, 8, 11],
    'ma_168h': [1, 2, 3],
    'entry_buffer': [0.01, 0.02, 0.03],
    'exit_buffer': [0.0025, 0.005, 0.0075],
    'hit_rate_threshold': [0.45, 0.50, 0.55],
    'target_vol': [0.30, 0.40, 0.50],
    'dd_start_reduce': [-0.15, -0.20, -0.25],
    'min_exposure_floor': [0.30, 0.40, 0.50],
}


# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True)
    def label_trend_numba(close: np.ndarray, ma: np.ndarray, 
                          entry_buffer: float, exit_buffer: float) -> np.ndarray:
        """Numba-optimized trend detection."""
        n = len(close)
        labels = np.ones(n, dtype=np.int32)
        current = 1
        
        for i in range(n):
            if np.isnan(ma[i]):
                labels[i] = current
                continue
            
            price = close[i]
            threshold_down = ma[i] * (1 - entry_buffer)
            threshold_up = ma[i] * (1 + entry_buffer)
            
            if current == 1:
                if price < threshold_down:
                    current = 0
            else:
                if price > threshold_up:
                    current = 1
            
            labels[i] = current
        
        return labels

    @jit(nopython=True)
    def simulate_fast_numba(port_returns: np.ndarray, target_vol: float,
                            dd_start: float, dd_min: float, 
                            min_floor: float) -> Tuple[float, float, float, float]:
        """Numba-optimized strategy simulation."""
        n = len(port_returns)
        
        # Vol scalar
        vol_scalar = np.ones(n)
        for i in range(VOL_LOOKBACK, n):
            window = port_returns[i-VOL_LOOKBACK:i]
            realized_vol = np.std(window) * np.sqrt(365)
            if realized_vol > 0:
                vs = target_vol / realized_vol
                vol_scalar[i] = max(min_floor, min(vs, 1.0))
        
        # Equity simulation
        equity = np.ones(n)
        peak = 1.0
        dd_scalar = 1.0
        
        for i in range(1, n):
            exposure = min(vol_scalar[i-1], dd_scalar)
            exposure = max(exposure, min_floor)
            
            equity[i] = equity[i-1] * (1.0 + port_returns[i] * exposure)
            peak = max(peak, equity[i])
            
            current_dd = (equity[i] - peak) / peak
            
            if current_dd >= dd_start:
                dd_scalar = 1.0
            elif current_dd <= dd_min:
                dd_scalar = min_floor
            else:
                range_dd = dd_start - dd_min
                position = (current_dd - dd_min) / range_dd
                dd_scalar = min_floor + position * (1.0 - min_floor)
        
        # Metrics
        total_return = equity[-1] - 1.0
        years = n / 365.0
        
        if years > 0:
            annual_return = (1.0 + total_return) ** (1.0/years) - 1.0
        else:
            annual_return = 0.0
        
        daily_returns = np.diff(equity) / equity[:-1]
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        
        if std_ret > 0:
            sharpe = mean_ret / std_ret * np.sqrt(365)
        else:
            sharpe = 0.0
        
        # Max drawdown
        running_max = equity[0]
        max_dd = 0.0
        for i in range(n):
            running_max = max(running_max, equity[i])
            dd = (equity[i] - running_max) / running_max
            max_dd = min(max_dd, dd)
        
        return total_return, annual_return, sharpe, max_dd

else:
    def label_trend_numba(close, ma, entry_buffer, exit_buffer):
        """Fallback Python version."""
        n = len(close)
        labels = np.ones(n, dtype=np.int32)
        current = 1
        
        for i in range(n):
            if np.isnan(ma[i]):
                labels[i] = current
                continue
            
            price = close[i]
            
            if current == 1:
                if price < ma[i] * (1 - entry_buffer):
                    current = 0
            else:
                if price > ma[i] * (1 + entry_buffer):
                    current = 1
            
            labels[i] = current
        
        return labels

    def simulate_fast_numba(port_returns, target_vol, dd_start, dd_min, min_floor):
        """Fallback Python version."""
        n = len(port_returns)
        
        vol_scalar = np.ones(n)
        for i in range(VOL_LOOKBACK, n):
            realized_vol = np.std(port_returns[i-VOL_LOOKBACK:i]) * np.sqrt(365)
            if realized_vol > 0:
                vol_scalar[i] = np.clip(target_vol / realized_vol, min_floor, 1.0)
        
        equity = np.ones(n)
        peak = 1.0
        dd_scalar = 1.0
        
        for i in range(1, n):
            exposure = max(min(vol_scalar[i-1], dd_scalar), min_floor)
            equity[i] = equity[i-1] * (1.0 + port_returns[i] * exposure)
            peak = max(peak, equity[i])
            
            current_dd = (equity[i] - peak) / peak
            
            if current_dd >= dd_start:
                dd_scalar = 1.0
            elif current_dd <= dd_min:
                dd_scalar = min_floor
            else:
                range_dd = dd_start - dd_min
                position = (current_dd - dd_min) / range_dd
                dd_scalar = min_floor + position * (1.0 - min_floor)
        
        total_return = equity[-1] - 1.0
        years = n / 365.0
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        daily_returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
        
        max_dd = np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))
        
        return total_return, annual_return, sharpe, max_dd


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 100)
    print("FAST PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 100)
    
    db = Database()
    
    # =========================================================================
    # LOAD AND PRECOMPUTE
    # =========================================================================
    
    print("\n" + "-" * 60)
    print("LOADING DATA & PRE-COMPUTING MAs")
    print("-" * 60)
    
    # All unique MA periods we need
    all_ma_periods_24h = list(set(SENSITIVITY_TESTS['ma_24h']))
    all_ma_periods_72h = list(set(SENSITIVITY_TESTS['ma_72h']))
    all_ma_periods_168h = list(set(SENSITIVITY_TESTS['ma_168h']))
    
    # Storage
    price_data = {}
    precomputed_mas = {}
    
    for pair in DEPLOY_PAIRS:
        print(f"  {pair}...", end=" ")
        
        df_1h = db.get_ohlcv(pair)
        df_24h = df_1h.resample('24h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        df_72h = df_1h.resample('72h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        df_168h = df_1h.resample('168h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        
        price_data[pair] = {
            '24h': df_24h['close'].values,
            '72h': df_72h['close'].values,
            '168h': df_168h['close'].values,
            '24h_idx': df_24h.index,
            '72h_idx': df_72h.index,
            '168h_idx': df_168h.index,
        }
        
        # Pre-compute all MAs
        precomputed_mas[pair] = {
            '24h': {p: pd.Series(df_24h['close']).rolling(p).mean().values for p in all_ma_periods_24h},
            '72h': {p: pd.Series(df_72h['close']).rolling(p).mean().values for p in all_ma_periods_72h},
            '168h': {p: pd.Series(df_168h['close']).rolling(p).mean().values for p in all_ma_periods_168h},
        }
        
        print(f"{len(df_24h)} bars")
    
    # Create aligned returns
    returns_dict = {}
    for pair in DEPLOY_PAIRS:
        returns_dict[pair] = pd.Series(price_data[pair]['24h'], index=price_data[pair]['24h_idx']).pct_change()
    
    returns_df = pd.DataFrame(returns_dict).dropna()
    common_idx = returns_df.index
    n_days = len(returns_df)
    
    print(f"\n  Common period: {common_idx.min().date()} to {common_idx.max().date()}")
    print(f"  Total days: {n_days}")
    
    # Pre-compute risk parity weights (same for all tests)
    print("  Pre-computing risk parity weights...")
    
    weights = np.zeros((n_days, len(DEPLOY_PAIRS)))
    current_w = np.ones(len(DEPLOY_PAIRS)) / len(DEPLOY_PAIRS)
    
    for i in range(n_days):
        if i >= 60 and (i - 60) % 30 == 0:
            trailing = returns_df.iloc[i-60:i]
            cov = trailing.cov().values * 252
            vols = np.sqrt(np.diag(cov))
            vols[vols == 0] = 1e-6
            inv_vols = 1.0 / vols
            current_w = inv_vols / inv_vols.sum()
        weights[i] = current_w
    
    returns_arr = returns_df.values
    
    # =========================================================================
    # RUN TESTS
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("RUNNING SENSITIVITY TESTS")
    print("=" * 100)
    
    # Warm up Numba
    if HAS_NUMBA:
        print("\n  Warming up Numba JIT...")
        _ = label_trend_numba(np.random.randn(100), np.random.randn(100), 0.02, 0.005)
        _ = simulate_fast_numba(np.random.randn(100), 0.4, -0.2, -0.5, 0.4)
        print("  Done")
    
    results_all = {}
    
    def run_test(params):
        """Run single test with given parameters."""
        # Generate signals for each pair
        signals = np.zeros((n_days, len(DEPLOY_PAIRS)))
        
        for j, pair in enumerate(DEPLOY_PAIRS):
            # Get pre-computed MAs
            ma_24h = precomputed_mas[pair]['24h'][params['ma_24h']]
            ma_72h = precomputed_mas[pair]['72h'][params['ma_72h']]
            ma_168h = precomputed_mas[pair]['168h'][params['ma_168h']]
            
            # Generate trends
            close_24h = price_data[pair]['24h']
            close_72h = price_data[pair]['72h']
            close_168h = price_data[pair]['168h']
            
            trend_24h = label_trend_numba(close_24h, ma_24h, params['entry_buffer'], params['exit_buffer'])
            trend_72h = label_trend_numba(close_72h, ma_72h, params['entry_buffer'], params['exit_buffer'])
            trend_168h = label_trend_numba(close_168h, ma_168h, params['entry_buffer'], params['exit_buffer'])
            
            # Align to common index
            trend_24h_series = pd.Series(trend_24h, index=price_data[pair]['24h_idx']).shift(1)
            trend_72h_series = pd.Series(trend_72h, index=price_data[pair]['72h_idx']).shift(1).reindex(price_data[pair]['24h_idx'], method='ffill')
            trend_168h_series = pd.Series(trend_168h, index=price_data[pair]['168h_idx']).shift(1).reindex(price_data[pair]['24h_idx'], method='ffill')
            
            # Align to common returns index
            t24 = trend_24h_series.reindex(common_idx).fillna(1).astype(int).values
            t72 = trend_72h_series.reindex(common_idx).fillna(1).astype(int).values
            t168 = trend_168h_series.reindex(common_idx).fillna(1).astype(int).values
            
            # Calculate hit rates for this pair
            forward_ret = returns_df[pair].shift(-1).values
            
            hit_rates = {}
            for perm in product([0, 1], repeat=3):
                mask = (t24 == perm[0]) & (t72 == perm[1]) & (t168 == perm[2])
                perm_ret = forward_ret[mask & ~np.isnan(forward_ret)]
                n = len(perm_ret)
                if n >= MIN_SAMPLES:
                    hr = (perm_ret > 0).sum() / n
                    hit_rates[perm] = 1.0 if hr > params['hit_rate_threshold'] else 0.0
                else:
                    hit_rates[perm] = 0.5
            
            # Create position array
            for i in range(n_days):
                perm = (t24[i], t72[i], t168[i])
                signals[i, j] = hit_rates.get(perm, 0.5)
        
        # Portfolio returns
        weighted_returns = returns_arr * weights * signals
        port_returns = np.sum(weighted_returns, axis=1)
        
        # Simulate
        total_ret, annual_ret, sharpe, max_dd = simulate_fast_numba(
            port_returns, params['target_vol'], params['dd_start_reduce'],
            DD_MIN_EXPOSURE, params['min_exposure_floor']
        )
        
        return {
            'total_return': total_ret,
            'annual_return': annual_ret,
            'sharpe': sharpe,
            'max_dd': max_dd
        }
    
    # Run baseline
    print("\n  Running baseline...")
    baseline = run_test(BASELINE)
    print(f"    Baseline: Return={baseline['total_return']*100:+.1f}%, Sharpe={baseline['sharpe']:.2f}")
    
    # Run sensitivity tests
    for param_name, values in SENSITIVITY_TESTS.items():
        print(f"\n  Testing {param_name}...")
        start = time.time()
        
        results = []
        for val in values:
            params = BASELINE.copy()
            params[param_name] = val
            
            result = run_test(params)
            result['value'] = val
            result['is_baseline'] = (val == BASELINE[param_name])
            results.append(result)
        
        results_all[param_name] = results
        elapsed = time.time() - start
        
        # Print inline
        sharpes = [r['sharpe'] for r in results]
        print(f"    Sharpe range: [{min(sharpes):.2f}, {max(sharpes):.2f}] ({elapsed:.1f}s)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    print(f"\n  {'Parameter':<22} {'Baseline':>10} {'Sharpe Range':>18} {'Min Sharpe':>12} {'Verdict':>15}")
    print("  " + "-" * 80)
    
    sensitivity_scores = {}
    
    for param_name, results in results_all.items():
        sharpes = [r['sharpe'] for r in results]
        min_sharpe = min(sharpes)
        max_sharpe = max(sharpes)
        sharpe_range = max_sharpe - min_sharpe
        
        # Format baseline
        bv = BASELINE[param_name]
        if 'buffer' in param_name:
            base_str = f"{bv*100:.2f}%"
        elif any(x in param_name for x in ['threshold', 'vol', 'floor', 'reduce']):
            base_str = f"{bv*100:.0f}%"
        else:
            base_str = str(bv)
        
        # Verdict
        if min_sharpe >= 0.8 and sharpe_range < 0.3:
            verdict = "✓ ROBUST"
            score = 100
        elif min_sharpe >= 0.5 and sharpe_range < 0.5:
            verdict = "~ MODERATE"
            score = 70
        elif min_sharpe >= 0.3:
            verdict = "⚠ SENSITIVE"
            score = 40
        else:
            verdict = "✗ FRAGILE"
            score = 10
        
        sensitivity_scores[param_name] = score
        
        range_str = f"[{min_sharpe:.2f}, {max_sharpe:.2f}]"
        print(f"  {param_name:<22} {base_str:>10} {range_str:>18} {min_sharpe:>12.2f} {verdict:>15}")
    
    # Overall verdict
    avg_score = np.mean(list(sensitivity_scores.values()))
    min_sharpe_overall = min([min(r['sharpe'] for r in results) for results in results_all.values()])
    
    if avg_score >= 80 and min_sharpe_overall >= 0.7:
        verdict = "ROBUST"
        desc = "Strategy maintains strong performance across parameter variations"
    elif avg_score >= 60 and min_sharpe_overall >= 0.4:
        verdict = "MODERATE"
        desc = "Strategy is reasonably stable but some sensitivity exists"
    elif avg_score >= 40:
        verdict = "SENSITIVE"
        desc = "Strategy shows meaningful parameter sensitivity - use caution"
    else:
        verdict = "FRAGILE"
        desc = "Strategy is highly dependent on exact parameters - likely overfit"
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  SENSITIVITY VERDICT: {verdict:<15}                                                                 ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  {desc:<84} ║
    ║                                                                                                      ║
    ║  METRICS:                                                                                            ║
    ║    • Average Robustness Score:    {avg_score:>5.0f}/100                                                       ║
    ║    • Minimum Sharpe (any config): {min_sharpe_overall:>5.2f}                                                         ║
    ║    • Baseline Sharpe:             {baseline['sharpe']:>5.2f}                                                         ║
    ║                                                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Detailed per-parameter results
    print("\n" + "=" * 100)
    print("DETAILED RESULTS BY PARAMETER")
    print("=" * 100)
    
    for param_name, results in results_all.items():
        print(f"\n  {param_name}:")
        print(f"    {'Value':>12} {'Return':>12} {'Sharpe':>10} {'MaxDD':>10}")
        print("    " + "-" * 50)
        
        for r in results:
            val = r['value']
            if 'buffer' in param_name:
                val_str = f"{val*100:.2f}%"
            elif any(x in param_name for x in ['threshold', 'vol', 'floor', 'reduce']):
                val_str = f"{val*100:.0f}%"
            else:
                val_str = str(val)
            
            marker = " ← BASE" if r['is_baseline'] else ""
            print(f"    {val_str:>12} {r['total_return']*100:>+11.1f}% {r['sharpe']:>10.2f} {r['max_dd']*100:>9.1f}%{marker}")
    
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    
    return results_all, baseline


if __name__ == "__main__":
    results, baseline = main()