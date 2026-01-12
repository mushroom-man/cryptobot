#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend + Volatility Combined Signal
===================================
Tests integrating volatility regime with trend signals.

Hypothesis: Volatility is predictable (60% correlation), use it to:
  - Reduce position in high vol (risk management)
  - OR increase position in high vol + uptrend (alpha seeking)

Tests across: 24h, 72h, 168h timeframes

Usage:
    python trend_vol_combined.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from enum import IntEnum
from typing import Dict, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# SETTINGS
# =============================================================================

PAIR = 'XBTUSD'
DATA_START = '2017-01-01'
DATA_END = '2024-03-31'

TRAIN_END = '2019-12-31'
TEST_START = '2020-01-01'

# Timeframes to test
TIMEFRAMES = ['1h', '4h','12h', '24h', '72h', '168h']

# Trading costs
TOTAL_COST = 0.0015

# -----------------------------------------------------------------------------
# TREND SETTINGS (same as before)
# -----------------------------------------------------------------------------
TREND_MA_PERIOD = 24
TREND_ENTRY_BUFFER = 0.02
TREND_EXIT_BUFFER = 0.01

# -----------------------------------------------------------------------------
# VOLATILITY SETTINGS
# -----------------------------------------------------------------------------
VOL_LOOKBACK = 20           # Period for realized vol calculation
VOL_QUANTILE_HIGH = 0.75    # Above this = high vol
VOL_QUANTILE_LOW = 0.25     # Below this = low vol
VOL_SMOOTH = 5              # Smoothing for vol signal

# -----------------------------------------------------------------------------
# POSITION LOOKUP TABLES (Trend × Volatility)
# -----------------------------------------------------------------------------

# Conservative: Reduce in high vol
LOOKUP_CONSERVATIVE = {
    # (Trend, Vol) → Position
    # Trend: 1=Up, 0=Flat, -1=Down
    # Vol: 1=High, 0=Normal, -1=Low
    (1, 1):   0.80,   # Up + High Vol → Reduce (risky)
    (1, 0):   1.00,   # Up + Normal Vol → Full
    (1, -1):  1.00,   # Up + Low Vol → Full (safe)
    (0, 1):   0.50,   # Flat + High Vol → Careful
    (0, 0):   0.70,   # Flat + Normal Vol → Moderate
    (0, -1):  0.80,   # Flat + Low Vol → OK to hold
    (-1, 1):  0.10,   # Down + High Vol → DANGER, exit
    (-1, 0):  0.30,   # Down + Normal Vol → Reduce
    (-1, -1): 0.50,   # Down + Low Vol → Less urgent
}

# Aggressive: Exploit high vol moves
LOOKUP_AGGRESSIVE = {
    (1, 1):   1.00,   # Up + High Vol → Ride the wave!
    (1, 0):   1.00,   # Up + Normal Vol → Full
    (1, -1):  0.80,   # Up + Low Vol → Smaller moves expected
    (0, 1):   0.60,   # Flat + High Vol → Breakout coming?
    (0, 0):   0.70,   # Flat + Normal Vol → Moderate
    (0, -1):  0.60,   # Flat + Low Vol → Boring, stay moderate
    (-1, 1):  0.00,   # Down + High Vol → CRASH, get out!
    (-1, 0):  0.20,   # Down + Normal Vol → Reduce
    (-1, -1): 0.40,   # Down + Low Vol → Slow bleed, less urgent
}

# Asymmetric: Different response to high vol based on trend
LOOKUP_ASYMMETRIC = {
    (1, 1):   1.00,   # Up + High Vol → Big opportunity
    (1, 0):   1.00,   # Up + Normal Vol → Full
    (1, -1):  0.90,   # Up + Low Vol → Safe, full
    (0, 1):   0.50,   # Flat + High Vol → Uncertain, reduce
    (0, 0):   0.75,   # Flat + Normal Vol → Moderate
    (0, -1):  0.80,   # Flat + Low Vol → Safe to hold
    (-1, 1):  0.00,   # Down + High Vol → MAXIMUM DANGER
    (-1, 0):  0.30,   # Down + Normal Vol → Exit
    (-1, -1): 0.50,   # Down + Low Vol → Slow decline
}

# Ultra Risk Manager (from previous tests, extended with vol)
LOOKUP_ULTRA = {
    (1, 1):   0.90,   # Up + High Vol → Slight caution
    (1, 0):   1.00,   # Up + Normal Vol → Full
    (1, -1):  1.00,   # Up + Low Vol → Full
    (0, 1):   0.70,   # Flat + High Vol → Reduce
    (0, 0):   0.85,   # Flat + Normal Vol → High
    (0, -1):  0.90,   # Flat + Low Vol → Safe
    (-1, 1):  0.20,   # Down + High Vol → Danger
    (-1, 0):  0.50,   # Down + Normal Vol → Moderate
    (-1, -1): 0.60,   # Down + Low Vol → Less urgent
}

# Trend Only (ignore volatility - baseline)
LOOKUP_TREND_ONLY = {
    (1, 1):   1.00,
    (1, 0):   1.00,
    (1, -1):  1.00,
    (0, 1):   0.80,
    (0, 0):   0.80,
    (0, -1):  0.80,
    (-1, 1):  0.30,
    (-1, 0):  0.30,
    (-1, -1): 0.30,
}

# Vol Only (ignore trend)
LOOKUP_VOL_ONLY = {
    (1, 1):   0.50,   # High Vol → Reduce
    (1, 0):   0.70,   # Normal Vol → Moderate
    (1, -1):  0.90,   # Low Vol → Safe
    (0, 1):   0.50,
    (0, 0):   0.70,
    (0, -1):  0.90,
    (-1, 1):  0.50,
    (-1, 0):  0.70,
    (-1, -1): 0.90,
}

ALL_LOOKUPS = {
    'Trend Only': LOOKUP_TREND_ONLY,
    'Vol Only': LOOKUP_VOL_ONLY,
    'Conservative (T+V)': LOOKUP_CONSERVATIVE,
    'Aggressive (T+V)': LOOKUP_AGGRESSIVE,
    'Asymmetric (T+V)': LOOKUP_ASYMMETRIC,
    'Ultra (T+V)': LOOKUP_ULTRA,
}


# =============================================================================
# REGIME ENUM
# =============================================================================

class Regime(IntEnum):
    DOWN = -1
    FLAT = 0
    UP = 1


class VolRegime(IntEnum):
    LOW = -1
    NORMAL = 0
    HIGH = 1


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def label_trend(df: pd.DataFrame) -> pd.Series:
    """Trend detection: Price vs MA with hysteresis."""
    close = df['close']
    ma = close.rolling(TREND_MA_PERIOD).mean()
    
    enter_up = ma * (1 + TREND_ENTRY_BUFFER)
    exit_up = ma * (1 - TREND_EXIT_BUFFER)
    enter_down = ma * (1 - TREND_ENTRY_BUFFER)
    exit_down = ma * (1 + TREND_EXIT_BUFFER)
    
    labels = pd.Series(Regime.FLAT, index=df.index)
    current = Regime.FLAT
    
    for i in range(len(df)):
        if pd.isna(ma.iloc[i]):
            continue
        
        price = close.iloc[i]
        
        if current == Regime.FLAT:
            if price > enter_up.iloc[i]:
                current = Regime.UP
            elif price < enter_down.iloc[i]:
                current = Regime.DOWN
        elif current == Regime.UP:
            if price < exit_up.iloc[i]:
                current = Regime.FLAT if price >= enter_down.iloc[i] else Regime.DOWN
        elif current == Regime.DOWN:
            if price > exit_down.iloc[i]:
                current = Regime.UP if price > enter_up.iloc[i] else Regime.FLAT
        
        labels.iloc[i] = current
    
    return labels


def label_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Volatility regime detection.
    
    Uses realized volatility with quantile-based regime assignment.
    """
    returns = df['close'].pct_change()
    
    # Realized volatility (rolling std of returns)
    realized_vol = returns.rolling(VOL_LOOKBACK).std()
    
    # Smooth to reduce noise
    smoothed_vol = realized_vol.rolling(VOL_SMOOTH).mean()
    
    # Rolling quantiles for adaptive thresholds
    vol_upper = smoothed_vol.rolling(VOL_LOOKBACK * 4).quantile(VOL_QUANTILE_HIGH)
    vol_lower = smoothed_vol.rolling(VOL_LOOKBACK * 4).quantile(VOL_QUANTILE_LOW)
    
    # Assign regimes
    labels = pd.Series(VolRegime.NORMAL, index=df.index)
    
    for i in range(len(df)):
        if pd.isna(smoothed_vol.iloc[i]) or pd.isna(vol_upper.iloc[i]):
            continue
        
        vol = smoothed_vol.iloc[i]
        
        if vol > vol_upper.iloc[i]:
            labels.iloc[i] = VolRegime.HIGH
        elif vol < vol_lower.iloc[i]:
            labels.iloc[i] = VolRegime.LOW
        else:
            labels.iloc[i] = VolRegime.NORMAL
    
    return labels, smoothed_vol


def get_positions(trend: pd.Series, vol: pd.Series, lookup: Dict) -> pd.Series:
    """Convert signals to positions using lookup table."""
    positions = pd.Series(0.7, index=trend.index)
    
    for i in range(len(positions)):
        t = int(trend.iloc[i])
        v = int(vol.iloc[i])
        positions.iloc[i] = lookup.get((t, v), 0.7)
    
    return positions


# =============================================================================
# UTILITY
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()


# =============================================================================
# BACKTEST
# =============================================================================

def backtest(df: pd.DataFrame, positions: pd.Series, tf_hours: int, cost: float = TOTAL_COST) -> dict:
    """Full backtest with all metrics."""
    returns = df['close'].pct_change()
    
    # Costs
    position_changes = positions.diff().abs().fillna(0)
    costs = position_changes * cost
    
    # Strategy returns
    strategy_returns = positions.shift(1) * returns - costs
    bh_returns = returns
    
    # Equity curves
    strategy_equity = (1 + strategy_returns).cumprod()
    bh_equity = (1 + bh_returns).cumprod()
    
    # Total returns
    total_return = strategy_equity.iloc[-1] - 1
    bh_return = bh_equity.iloc[-1] - 1
    alpha = total_return - bh_return
    
    # Sharpe
    bars_per_year = 365 * 24 / tf_hours
    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(bars_per_year) if strategy_returns.std() > 0 else 0
    bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(bars_per_year) if bh_returns.std() > 0 else 0
    
    # Max Drawdown
    rolling_max = strategy_equity.expanding().max()
    drawdowns = (strategy_equity - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    bh_rolling_max = bh_equity.expanding().max()
    bh_drawdowns = (bh_equity - bh_rolling_max) / bh_rolling_max
    bh_max_dd = bh_drawdowns.min()
    
    # Return per unit drawdown
    return_per_dd = abs(total_return / max_dd) if max_dd != 0 else 0
    
    # Exposure
    avg_position = positions.mean()
    n_trades = (position_changes > 0.1).sum()
    
    return {
        'return': total_return,
        'bh_return': bh_return,
        'alpha': alpha,
        'sharpe': strategy_sharpe,
        'bh_sharpe': bh_sharpe,
        'max_dd': max_dd,
        'bh_max_dd': bh_max_dd,
        'return_per_dd': return_per_dd,
        'avg_pos': avg_position,
        'n_trades': n_trades,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("TREND + VOLATILITY COMBINED SIGNAL")
    print("Testing across 24h, 72h, 168h timeframes")
    print("=" * 100)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    db = Database()
    df_1h = db.get_ohlcv(PAIR, start=DATA_START, end=DATA_END)
    print(f"Loaded {len(df_1h):,} 1h bars")
    print(f"Date range: {df_1h.index.min().date()} to {df_1h.index.max().date()}")
    
    # =========================================================================
    # TEST EACH TIMEFRAME
    # =========================================================================
    
    all_results = {}
    
    for tf in TIMEFRAMES:
        tf_hours = int(tf.replace('h', ''))
        
        print("\n" + "=" * 100)
        print(f"TIMEFRAME: {tf}")
        print("=" * 100)
        
        # Resample
        df = resample_ohlcv(df_1h, tf)
        print(f"Resampled to {len(df)} {tf} bars")
        
        # Split
        df_train = df[df.index <= TRAIN_END]
        df_test = df[df.index >= TEST_START]
        
        print(f"Train: {len(df_train)} bars")
        print(f"Test:  {len(df_test)} bars")
        
        # Generate signals
        print("\n" + "-" * 60)
        print("GENERATING SIGNALS")
        print("-" * 60)
        
        # Full dataset for signal generation
        trend_full = label_trend(df)
        vol_full, vol_raw = label_volatility(df)
        
        # Test period
        trend_test = trend_full[df.index >= TEST_START]
        vol_test = vol_full[df.index >= TEST_START]
        
        print(f"Trend distribution (test):")
        print(f"  Up={100*(trend_test==1).mean():.0f}%, Flat={100*(trend_test==0).mean():.0f}%, Down={100*(trend_test==-1).mean():.0f}%")
        print(f"Volatility distribution (test):")
        print(f"  High={100*(vol_test==1).mean():.0f}%, Normal={100*(vol_test==0).mean():.0f}%, Low={100*(vol_test==-1).mean():.0f}%")
        
        # Cross-tabulation
        print(f"\nTrend × Volatility Distribution (test):")
        print(f"            Low    Normal   High")
        for t in [1, 0, -1]:
            t_name = "Up  " if t == 1 else "Flat" if t == 0 else "Down"
            counts = []
            for v in [-1, 0, 1]:
                count = ((trend_test == t) & (vol_test == v)).sum()
                pct = 100 * count / len(trend_test)
                counts.append(f"{pct:5.1f}%")
            print(f"  {t_name}   {counts[0]}   {counts[1]}   {counts[2]}")
        
        # =====================================================================
        # BACKTEST ALL STRATEGIES
        # =====================================================================
        
        print("\n" + "-" * 60)
        print("BACKTESTING STRATEGIES")
        print("-" * 60)
        
        results = []
        
        for name, lookup in ALL_LOOKUPS.items():
            positions = get_positions(trend_test, vol_test, lookup)
            metrics = backtest(df_test, positions, tf_hours)
            metrics['name'] = name
            results.append(metrics)
        
        # Add B&H
        bh_metrics = results[0]
        results.append({
            'name': 'Buy & Hold',
            'return': bh_metrics['bh_return'],
            'bh_return': bh_metrics['bh_return'],
            'alpha': 0,
            'sharpe': bh_metrics['bh_sharpe'],
            'bh_sharpe': bh_metrics['bh_sharpe'],
            'max_dd': bh_metrics['bh_max_dd'],
            'bh_max_dd': bh_metrics['bh_max_dd'],
            'return_per_dd': abs(bh_metrics['bh_return'] / bh_metrics['bh_max_dd']) if bh_metrics['bh_max_dd'] != 0 else 0,
            'avg_pos': 1.0,
            'n_trades': 0,
        })
        
        # Sort by Sharpe
        results.sort(key=lambda x: x['sharpe'], reverse=True)
        
        # Store for comparison
        all_results[tf] = results
        
        # Print results
        print(f"\n{'Strategy':<25} {'Return':>10} {'Alpha':>10} │ {'Sharpe':>7} {'B&H':>7} │ {'MaxDD':>8} {'B&H DD':>8} │ {'Ret/DD':>7} {'AvgPos':>7}")
        print("-" * 115)
        
        for r in results:
            alpha_marker = "✓" if r['alpha'] > 0 else ""
            sharpe_marker = "✓" if r['sharpe'] > r['bh_sharpe'] else ""
            dd_marker = "✓" if abs(r['max_dd']) < abs(r['bh_max_dd']) else ""
            
            print(f"{r['name']:<25} {r['return']*100:>+9.0f}% {r['alpha']*100:>+9.0f}%{alpha_marker} │ "
                  f"{r['sharpe']:>7.2f}{sharpe_marker} {r['bh_sharpe']:>7.2f} │ "
                  f"{r['max_dd']*100:>7.1f}%{dd_marker} {r['bh_max_dd']*100:>7.1f}% │ "
                  f"{r['return_per_dd']:>7.1f}x {r['avg_pos']*100:>6.0f}%")
        
        # Best performers
        best_return = max([r for r in results if r['name'] != 'Buy & Hold'], key=lambda x: x['return'])
        best_sharpe = max([r for r in results if r['name'] != 'Buy & Hold'], key=lambda x: x['sharpe'])
        best_dd = min([r for r in results if r['name'] != 'Buy & Hold'], key=lambda x: abs(x['max_dd']))
        
        print(f"\n  Best Return: {best_return['name']} ({best_return['return']*100:+.0f}%)")
        print(f"  Best Sharpe: {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")
        print(f"  Best MaxDD:  {best_dd['name']} ({best_dd['max_dd']*100:.1f}%)")
    
    # =========================================================================
    # CROSS-TIMEFRAME COMPARISON
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("CROSS-TIMEFRAME COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Strategy':<25} │", end="")
    for tf in TIMEFRAMES:
        print(f" {tf:>12} │", end="")
    print(" Best TF")
    print("-" * (30 + 15 * len(TIMEFRAMES)))
    
    for strategy_name in ALL_LOOKUPS.keys():
        print(f"{strategy_name:<25} │", end="")
        sharpes = {}
        for tf in TIMEFRAMES:
            result = next((r for r in all_results[tf] if r['name'] == strategy_name), None)
            if result:
                sharpes[tf] = result['sharpe']
                marker = "✓" if result['sharpe'] > result['bh_sharpe'] else ""
                print(f"  {result['sharpe']:>5.2f}{marker}     │", end="")
        best_tf = max(sharpes, key=sharpes.get) if sharpes else "-"
        print(f" {best_tf}")
    
    # B&H row
    print(f"{'Buy & Hold':<25} │", end="")
    for tf in TIMEFRAMES:
        result = next((r for r in all_results[tf] if r['name'] == 'Buy & Hold'), None)
        if result:
            print(f"  {result['sharpe']:>5.2f}      │", end="")
    print(" (baseline)")
    
    # =========================================================================
    # FIND OVERALL BEST
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("OVERALL BEST CONFIGURATIONS")
    print("=" * 100)
    
    # Flatten all results
    all_flat = []
    for tf, results in all_results.items():
        for r in results:
            if r['name'] != 'Buy & Hold':
                all_flat.append({**r, 'tf': tf})
    
    # Best by Sharpe
    best_sharpe = max(all_flat, key=lambda x: x['sharpe'])
    
    # Best by Alpha
    best_alpha = max(all_flat, key=lambda x: x['alpha'])
    
    # Best by Return/DD
    best_ret_dd = max(all_flat, key=lambda x: x['return_per_dd'])
    
    # Best beating B&H
    beating_bh = [r for r in all_flat if r['alpha'] > 0]
    
    print(f"""
    BEST BY SHARPE:
        TF: {best_sharpe['tf']}
        Strategy: {best_sharpe['name']}
        Sharpe: {best_sharpe['sharpe']:.2f} (vs B&H {best_sharpe['bh_sharpe']:.2f})
        Return: {best_sharpe['return']*100:+.0f}%
        MaxDD: {best_sharpe['max_dd']*100:.1f}%
    
    BEST BY ALPHA:
        TF: {best_alpha['tf']}
        Strategy: {best_alpha['name']}
        Alpha: {best_alpha['alpha']*100:+.1f}%
        Return: {best_alpha['return']*100:+.0f}% (vs B&H {best_alpha['bh_return']*100:+.0f}%)
    
    BEST BY RETURN/DD:
        TF: {best_ret_dd['tf']}
        Strategy: {best_ret_dd['name']}
        Return/DD: {best_ret_dd['return_per_dd']:.1f}x
        Return: {best_ret_dd['return']*100:+.0f}%
        MaxDD: {best_ret_dd['max_dd']*100:.1f}%
    
    STRATEGIES BEATING B&H ON RETURN: {len(beating_bh)}/{len(all_flat)}
    """)
    
    if beating_bh:
        print("    Configurations with positive alpha:")
        for r in sorted(beating_bh, key=lambda x: x['alpha'], reverse=True)[:5]:
            print(f"      {r['tf']} {r['name']}: Alpha={r['alpha']*100:+.1f}%, Sharpe={r['sharpe']:.2f}")
    
    # =========================================================================
    # RECOMMENDATION
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    
    # Score each config
    for r in all_flat:
        # Score: 40% Sharpe, 30% Return/DD, 20% Alpha, 10% DD improvement
        bh_result = next(x for x in all_results[r['tf']] if x['name'] == 'Buy & Hold')
        
        sharpe_score = r['sharpe'] / max(x['sharpe'] for x in all_flat) if max(x['sharpe'] for x in all_flat) > 0 else 0
        ret_dd_score = r['return_per_dd'] / max(x['return_per_dd'] for x in all_flat) if max(x['return_per_dd'] for x in all_flat) > 0 else 0
        alpha_score = (r['alpha'] - min(x['alpha'] for x in all_flat)) / (max(x['alpha'] for x in all_flat) - min(x['alpha'] for x in all_flat)) if max(x['alpha'] for x in all_flat) != min(x['alpha'] for x in all_flat) else 0
        dd_improvement = (abs(bh_result['bh_max_dd']) - abs(r['max_dd'])) / abs(bh_result['bh_max_dd']) if bh_result['bh_max_dd'] != 0 else 0
        dd_score = max(0, dd_improvement)
        
        r['score'] = 0.4 * sharpe_score + 0.3 * ret_dd_score + 0.2 * alpha_score + 0.1 * dd_score
    
    best_overall = max(all_flat, key=lambda x: x['score'])
    bh_baseline = next(x for x in all_results[best_overall['tf']] if x['name'] == 'Buy & Hold')
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RECOMMENDED CONFIGURATION                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Timeframe:   {best_overall['tf']:<55} │
    │  Strategy:    {best_overall['name']:<55} │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Return:      {best_overall['return']*100:>+7.1f}%  (B&H: {bh_baseline['bh_return']*100:>+7.1f}%)                          │
    │  Alpha:       {best_overall['alpha']*100:>+7.1f}%                                               │
    │  Sharpe:      {best_overall['sharpe']:>7.2f}   (B&H: {bh_baseline['bh_sharpe']:>5.2f})                             │
    │  Max DD:      {best_overall['max_dd']*100:>7.1f}%  (B&H: {bh_baseline['bh_max_dd']*100:>6.1f}%)                          │
    │  Return/DD:   {best_overall['return_per_dd']:>7.1f}x                                               │
    │  Avg Pos:     {best_overall['avg_pos']*100:>7.0f}%                                               │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Print the lookup table
    lookup = ALL_LOOKUPS[best_overall['name']]
    print(f"\n    POSITION LOOKUP TABLE ({best_overall['name']}):")
    print("    " + "-" * 50)
    print(f"    {'':12} │ {'Low Vol':>8} {'Normal':>8} {'High Vol':>8}")
    print("    " + "-" * 50)
    for t in [1, 0, -1]:
        t_name = "Trend Up" if t == 1 else "Trend Flat" if t == 0 else "Trend Down"
        vals = [lookup.get((t, v), 0.7) for v in [-1, 0, 1]]
        print(f"    {t_name:12} │ {vals[0]*100:>7.0f}% {vals[1]*100:>7.0f}% {vals[2]*100:>7.0f}%")
    
    # =========================================================================
    # VOLATILITY VALUE-ADD
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("VOLATILITY VALUE-ADD ANALYSIS")
    print("=" * 100)
    
    print(f"\n    Does adding volatility improve over trend-only?")
    print(f"\n    {'TF':<8} │ {'Trend Only':>12} │ {'Best T+V':>12} │ {'Improvement':>12}")
    print("    " + "-" * 55)
    
    for tf in TIMEFRAMES:
        trend_only = next((r for r in all_results[tf] if r['name'] == 'Trend Only'), None)
        tv_results = [r for r in all_results[tf] if '(T+V)' in r['name']]
        best_tv = max(tv_results, key=lambda x: x['sharpe']) if tv_results else None
        
        if trend_only and best_tv:
            improvement = best_tv['sharpe'] - trend_only['sharpe']
            marker = "✓" if improvement > 0 else "✗"
            print(f"    {tf:<8} │ {trend_only['sharpe']:>12.2f} │ {best_tv['sharpe']:>12.2f} │ {improvement:>+11.2f} {marker}")
    
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()