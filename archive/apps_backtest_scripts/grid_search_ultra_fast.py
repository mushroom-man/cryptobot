#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTRA-FAST Grid Search: Risk-Managed Risk Parity
=================================================
Uses Numba JIT compilation for 100x+ speedup.

Usage:
    pip install numba
    python grid_search_fast.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from itertools import product
import warnings
import time
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database

# Try to import numba, fall back to pure numpy if not available
try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("✓ Numba available - using JIT compilation")
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba not installed - using pure NumPy (slower)")
    print("  Install with: pip install numba")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']
REBALANCE_DAYS = 30
COV_LOOKBACK = 60
VOL_LOOKBACK = 30
MAX_LEVERAGE = 1.0
MAX_DD_CONSTRAINT = -0.50

# Grid parameters
TARGET_VOLS = np.array([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
DD_STARTS = np.array([-0.15, -0.20, -0.25, -0.30, -0.35])
DD_MINS = np.array([-0.30, -0.35, -0.40, -0.45, -0.50])
MIN_EXPOSURES = np.array([0.10, 0.20, 0.30, 0.40])


# =============================================================================
# NUMBA-ACCELERATED BACKTEST
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True)
    def backtest_numba(port_returns, rolling_vol, target_vol, dd_start, dd_min, min_exp):
        """Ultra-fast backtest using Numba JIT."""
        n = len(port_returns)
        
        # Pre-compute vol scalar
        vol_scalar = np.empty(n)
        for i in range(n):
            rv = rolling_vol[i]
            if rv <= 0:
                rv = target_vol
            vs = target_vol / rv
            if vs < min_exp:
                vs = min_exp
            elif vs > 1.0:
                vs = 1.0
            vol_scalar[i] = vs
        
        # Simulate with drawdown control
        equity = np.ones(n)
        peak = 1.0
        dd_scalar = 1.0
        
        for i in range(1, n):
            # Combined exposure
            exposure = vol_scalar[i-1]
            if dd_scalar < exposure:
                exposure = dd_scalar
            if exposure < min_exp:
                exposure = min_exp
            
            # Apply return
            equity[i] = equity[i-1] * (1.0 + port_returns[i] * exposure)
            
            # Update peak
            if equity[i] > peak:
                peak = equity[i]
            
            # Current drawdown
            current_dd = (equity[i] - peak) / peak
            
            # Update dd_scalar for next iteration
            if current_dd >= dd_start:
                dd_scalar = 1.0
            elif current_dd <= dd_min:
                dd_scalar = min_exp
            else:
                range_dd = dd_start - dd_min
                if range_dd != 0:
                    position = (current_dd - dd_min) / range_dd
                    dd_scalar = min_exp + position * (1.0 - min_exp)
                else:
                    dd_scalar = min_exp
        
        # Calculate metrics
        total_return = equity[-1] - 1.0
        
        # Max drawdown
        max_dd = 0.0
        peak = equity[0]
        for i in range(1, n):
            if equity[i] > peak:
                peak = equity[i]
            dd = (equity[i] - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # Daily returns for Sharpe
        daily_sum = 0.0
        daily_sq_sum = 0.0
        for i in range(1, n):
            ret = (equity[i] - equity[i-1]) / equity[i-1]
            daily_sum += ret
            daily_sq_sum += ret * ret
        
        n_days = n - 1
        mean_ret = daily_sum / n_days
        var_ret = daily_sq_sum / n_days - mean_ret * mean_ret
        std_ret = np.sqrt(var_ret) if var_ret > 0 else 1e-10
        sharpe = mean_ret / std_ret * np.sqrt(365.0)
        
        # Annual return
        years = n / 365.0
        annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        
        # Calmar
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        return total_return, annual_return, sharpe, max_dd, calmar

else:
    # Pure numpy fallback (slower but works)
    def backtest_numba(port_returns, rolling_vol, target_vol, dd_start, dd_min, min_exp):
        """NumPy fallback backtest."""
        n = len(port_returns)
        
        # Vol scalar
        rolling_vol = np.where(rolling_vol <= 0, target_vol, rolling_vol)
        vol_scalar = np.clip(target_vol / rolling_vol, min_exp, 1.0)
        
        # Simulate
        equity = np.ones(n)
        peak = 1.0
        dd_scalar = 1.0
        
        for i in range(1, n):
            exposure = min(vol_scalar[i-1], dd_scalar)
            exposure = max(exposure, min_exp)
            
            equity[i] = equity[i-1] * (1.0 + port_returns[i] * exposure)
            peak = max(peak, equity[i])
            
            current_dd = (equity[i] - peak) / peak
            
            if current_dd >= dd_start:
                dd_scalar = 1.0
            elif current_dd <= dd_min:
                dd_scalar = min_exp
            else:
                range_dd = dd_start - dd_min
                position = (current_dd - dd_min) / range_dd if range_dd != 0 else 0
                dd_scalar = min_exp + position * (1.0 - min_exp)
        
        total_return = equity[-1] - 1.0
        
        rolling_max = np.maximum.accumulate(equity)
        max_dd = np.min((equity - rolling_max) / rolling_max)
        
        daily_returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
        
        years = n / 365.0
        annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        return total_return, annual_return, sharpe, max_dd, calmar


# =============================================================================
# HELPER
# =============================================================================

def resample_ohlcv(df, timeframe):
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("ULTRA-FAST GRID SEARCH: RISK-MANAGED RISK PARITY")
    print("=" * 100)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    db = Database()
    all_returns = {}
    
    for pair in DEPLOY_PAIRS:
        print(f"  {pair}...", end=" ")
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        all_returns[pair] = df_24h['close'].pct_change()
        print(f"{len(df_24h)} bars")
    
    returns_df = pd.DataFrame(all_returns).dropna()
    n_days = len(returns_df)
    print(f"\n  Common: {returns_df.index.min().date()} to {returns_df.index.max().date()} ({n_days} bars)")
    
    # =========================================================================
    # PRE-COMPUTE WEIGHTS AND PORTFOLIO RETURNS
    # =========================================================================
    
    print("\n  Pre-computing risk parity portfolio returns...")
    
    returns_arr = returns_df.values
    n_assets = returns_arr.shape[1]
    
    # Build weight matrix
    weights_matrix = np.zeros((n_days, n_assets))
    current_weights = np.ones(n_assets) / n_assets
    
    rebalance_indices = list(range(COV_LOOKBACK, n_days, REBALANCE_DAYS))
    
    for idx in rebalance_indices:
        trailing = returns_df.iloc[idx-COV_LOOKBACK:idx]
        cov = trailing.cov().values * 252
        vols = np.sqrt(np.diag(cov))
        vols[vols == 0] = 1e-6
        inv_vols = 1.0 / vols
        current_weights = inv_vols / inv_vols.sum()
        weights_matrix[idx:] = current_weights
    
    # Fill initial period
    weights_matrix[:COV_LOOKBACK] = np.ones(n_assets) / n_assets
    
    # Portfolio returns (pre-computed for all combos)
    port_returns = np.sum(weights_matrix * returns_arr, axis=1)
    
    # Rolling volatility (pre-computed)
    port_series = pd.Series(port_returns)
    rolling_vol = (port_series.rolling(VOL_LOOKBACK).std() * np.sqrt(365)).fillna(0.5).values
    
    print(f"  Portfolio returns computed: {len(port_returns)} days")
    
    # =========================================================================
    # GRID SEARCH
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("RUNNING GRID SEARCH")
    print("=" * 100)
    
    # Generate valid combinations
    combos = []
    for tv in TARGET_VOLS:
        for ds in DD_STARTS:
            for dm in DD_MINS:
                for me in MIN_EXPOSURES:
                    if dm < ds:  # Valid: dd_min must be below dd_start
                        combos.append((tv, ds, dm, me))
    
    n_combos = len(combos)
    print(f"\n  Testing {n_combos} valid combinations...")
    
    # Warm up numba (first call compiles)
    if HAS_NUMBA:
        print("  Compiling JIT (first run)...", end=" ")
        _ = backtest_numba(port_returns, rolling_vol, 0.5, -0.25, -0.30, 0.20)
        print("done")
    
    # Run grid search
    results = []
    start_time = time.time()
    
    for i, (target_vol, dd_start, dd_min, min_exp) in enumerate(combos):
        total_ret, annual_ret, sharpe, max_dd, calmar = backtest_numba(
            port_returns, rolling_vol, target_vol, dd_start, dd_min, min_exp
        )
        
        results.append({
            'target_vol': target_vol,
            'dd_start_reduce': dd_start,
            'dd_min_exposure': dd_min,
            'min_exposure': min_exp,
            'total_return': total_ret,
            'annual_return': annual_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'calmar': calmar,
            'meets_constraint': max_dd >= MAX_DD_CONSTRAINT
        })
        
        # Progress every 50
        if (i + 1) % 50 == 0 or i == n_combos - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_combos - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/{n_combos} ({100*(i+1)/n_combos:.0f}%) - {elapsed:.1f}s elapsed, ~{eta:.1f}s remaining")
    
    total_time = time.time() - start_time
    print(f"\n  Completed in {total_time:.1f} seconds ({n_combos/total_time:.1f} combos/sec)")
    
    # =========================================================================
    # ANALYZE RESULTS
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    
    df = pd.DataFrame(results)
    valid_df = df[df['meets_constraint']].copy()
    
    print(f"\n  Valid combinations (DD >= {MAX_DD_CONSTRAINT*100:.0f}%): {len(valid_df)}/{len(df)}")
    
    if len(valid_df) == 0:
        print("\n  ⚠ No combinations meet constraint!")
        return
    
    # Sort by objectives
    by_calmar = valid_df.sort_values('calmar', ascending=False)
    by_return = valid_df.sort_values('total_return', ascending=False)
    by_sharpe = valid_df.sort_values('sharpe', ascending=False)
    
    # Top 10 by Calmar
    print("\n" + "-" * 90)
    print("TOP 10 BY CALMAR RATIO")
    print("-" * 90)
    print(f"{'#':<4} {'TargVol':>8} {'DDStart':>9} {'DDMin':>9} {'Floor':>8} │ {'Return':>10} {'MaxDD':>9} {'Calmar':>8} {'Sharpe':>8}")
    print("-" * 90)
    
    for rank, (_, row) in enumerate(by_calmar.head(10).iterrows(), 1):
        print(f"{rank:<4} {row['target_vol']*100:>7.0f}% {row['dd_start_reduce']*100:>8.0f}% "
              f"{row['dd_min_exposure']*100:>8.0f}% {row['min_exposure']*100:>7.0f}% │ "
              f"{row['total_return']*100:>+9.1f}% {row['max_dd']*100:>8.1f}% "
              f"{row['calmar']:>8.2f} {row['sharpe']:>8.2f}")
    
    # Top 10 by Return
    print("\n" + "-" * 90)
    print("TOP 10 BY TOTAL RETURN")
    print("-" * 90)
    print(f"{'#':<4} {'TargVol':>8} {'DDStart':>9} {'DDMin':>9} {'Floor':>8} │ {'Return':>10} {'MaxDD':>9} {'Calmar':>8} {'Sharpe':>8}")
    print("-" * 90)
    
    for rank, (_, row) in enumerate(by_return.head(10).iterrows(), 1):
        print(f"{rank:<4} {row['target_vol']*100:>7.0f}% {row['dd_start_reduce']*100:>8.0f}% "
              f"{row['dd_min_exposure']*100:>8.0f}% {row['min_exposure']*100:>7.0f}% │ "
              f"{row['total_return']*100:>+9.1f}% {row['max_dd']*100:>8.1f}% "
              f"{row['calmar']:>8.2f} {row['sharpe']:>8.2f}")
    
    # Optimal
    best = by_calmar.iloc[0]
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  OPTIMAL PARAMETERS (Best Calmar)                                                ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  Target Volatility:      {best['target_vol']*100:>5.0f}%                                                   ║
    ║  DD Start Reduce:        {best['dd_start_reduce']*100:>5.0f}%                                                   ║
    ║  DD Min Exposure At:     {best['dd_min_exposure']*100:>5.0f}%                                                   ║
    ║  Min Exposure Floor:     {best['min_exposure']*100:>5.0f}%                                                   ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  Total Return:           {best['total_return']*100:>+8.1f}%                                                 ║
    ║  Annual Return:          {best['annual_return']*100:>+8.1f}%                                                 ║
    ║  Max Drawdown:           {best['max_dd']*100:>8.1f}%                                                 ║
    ║  Sharpe Ratio:           {best['sharpe']:>8.2f}                                                   ║
    ║  Calmar Ratio:           {best['calmar']:>8.2f}                                                   ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Recommended by risk tolerance
    conservative = valid_df[valid_df['max_dd'] >= -0.40].sort_values('calmar', ascending=False)
    moderate = valid_df[(valid_df['max_dd'] >= -0.50) & (valid_df['max_dd'] < -0.40)].sort_values('calmar', ascending=False)
    
    print("\n  RECOMMENDED BY RISK TOLERANCE:")
    print("  " + "-" * 80)
    
    if len(conservative) > 0:
        c = conservative.iloc[0]
        print(f"  CONSERVATIVE (DD > -40%): Vol={c['target_vol']*100:.0f}%, Start={c['dd_start_reduce']*100:.0f}%, "
              f"Min={c['dd_min_exposure']*100:.0f}%, Floor={c['min_exposure']*100:.0f}%")
        print(f"                            Return: {c['total_return']*100:+.1f}% | MaxDD: {c['max_dd']*100:.1f}% | "
              f"Calmar: {c['calmar']:.2f} | Sharpe: {c['sharpe']:.2f}")
    
    if len(moderate) > 0:
        m = moderate.iloc[0]
        print(f"\n  MODERATE (DD -40% to -50%): Vol={m['target_vol']*100:.0f}%, Start={m['dd_start_reduce']*100:.0f}%, "
              f"Min={m['dd_min_exposure']*100:.0f}%, Floor={m['min_exposure']*100:.0f}%")
        print(f"                              Return: {m['total_return']*100:+.1f}% | MaxDD: {m['max_dd']*100:.1f}% | "
              f"Calmar: {m['calmar']:.2f} | Sharpe: {m['sharpe']:.2f}")
    
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()