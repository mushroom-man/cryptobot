#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hourly vs Daily Transition Analysis
====================================
Compare state transitions at hourly vs daily frequency.

Key Questions:
1. How often do states change hour-to-hour?
2. Does hourly analysis add signal or just noise?
3. Is path dependence stronger/weaker at hourly scale?

Usage:
    python hourly_transition_analysis.py --pair ETHUSD
    python hourly_transition_analysis.py --all-pairs
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from cryptobot.data.database import Database
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    print("Warning: Database not available.")

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked)
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# State classification
BULLISH_STATES = [8, 9, 10, 11, 12, 13, 14, 15]
BEARISH_STATES = [0, 1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# REGIME COMPUTATION - HOURLY
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_regime_states_hourly(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 16-state regime at HOURLY frequency.
    
    MAs are computed at their natural timeframes (24h, 72h, 168h)
    but states are evaluated every hour.
    """
    # Resample to get MA base data
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    # Compute MAs at their natural frequencies
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Forward-fill MAs to hourly frequency
    ma_24h_hourly = ma_24h.reindex(df_1h.index, method='ffill')
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
    # Compute trends with hysteresis at hourly frequency
    def compute_trend_hourly(close: pd.Series, ma: pd.Series, 
                             entry_buf=ENTRY_BUFFER, exit_buf=EXIT_BUFFER):
        trend = []
        current = 1
        for i in range(len(close)):
            if pd.isna(ma.iloc[i]):
                trend.append(current)
                continue
            price = close.iloc[i]
            ma_val = ma.iloc[i]
            if current == 1:
                if price < ma_val * (1 - exit_buf) and price < ma_val * (1 - entry_buf):
                    current = 0
            else:
                if price > ma_val * (1 + exit_buf) and price > ma_val * (1 + entry_buf):
                    current = 1
            trend.append(current)
        return trend
    
    states = pd.DataFrame(index=df_1h.index)
    states['close'] = df_1h['close']
    states['ma_24h'] = ma_24h_hourly
    states['ma_72h'] = ma_72h_hourly
    states['ma_168h'] = ma_168h_hourly
    
    states['trend_24h'] = compute_trend_hourly(df_1h['close'], ma_24h_hourly)
    states['trend_168h'] = compute_trend_hourly(df_1h['close'], ma_168h_hourly)
    states['ma72_above_ma24'] = (ma_72h_hourly > ma_24h_hourly).astype(int)
    states['ma168_above_ma24'] = (ma_168h_hourly > ma_24h_hourly).astype(int)
    
    states['state'] = (
        states['trend_24h'] * 8 +
        states['trend_168h'] * 4 +
        states['ma72_above_ma24'] * 2 +
        states['ma168_above_ma24'] * 1
    )
    
    return states.dropna()


def compute_regime_states_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Compute 16-state regime at DAILY frequency (for comparison)."""
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_daily = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_daily = ma_168h.reindex(df_24h.index, method='ffill')
    
    def compute_trend(close, ma, entry_buf=ENTRY_BUFFER, exit_buf=EXIT_BUFFER):
        trend = []
        current = 1
        for i in range(len(close)):
            if pd.isna(ma.iloc[i]):
                trend.append(current)
                continue
            price = close.iloc[i]
            ma_val = ma.iloc[i]
            if current == 1:
                if price < ma_val * (1 - exit_buf) and price < ma_val * (1 - entry_buf):
                    current = 0
            else:
                if price > ma_val * (1 + exit_buf) and price > ma_val * (1 + entry_buf):
                    current = 1
            trend.append(current)
        return trend
    
    states = pd.DataFrame(index=df_24h.index)
    states['close'] = df_24h['close']
    states['trend_24h'] = compute_trend(df_24h['close'], ma_24h)
    states['trend_168h'] = compute_trend(df_24h['close'], ma_168h_daily)
    states['ma72_above_ma24'] = (ma_72h_daily > ma_24h).astype(int)
    states['ma168_above_ma24'] = (ma_168h_daily > ma_24h).astype(int)
    
    states['state'] = (
        states['trend_24h'] * 8 +
        states['trend_168h'] * 4 +
        states['ma72_above_ma24'] * 2 +
        states['ma168_above_ma24'] * 1
    )
    
    return states.dropna()


# =============================================================================
# TRANSITION ANALYSIS
# =============================================================================

def analyze_transitions(states: pd.DataFrame, frequency: str) -> Dict:
    """
    Analyze state transitions.
    
    Returns dict with:
        - n_periods: total periods
        - n_transitions: number of state changes
        - transition_rate: transitions per period
        - state_stats: per-state statistics
        - transition_matrix: from->to counts
    """
    state_series = states['state'].values
    n_periods = len(state_series)
    
    # Count transitions
    transitions = []
    for i in range(1, len(state_series)):
        if state_series[i] != state_series[i-1]:
            transitions.append({
                'from': int(state_series[i-1]),
                'to': int(state_series[i]),
                'idx': i
            })
    
    n_transitions = len(transitions)
    transition_rate = n_transitions / n_periods if n_periods > 0 else 0
    
    # Transition matrix
    matrix = np.zeros((16, 16), dtype=int)
    for t in transitions:
        matrix[t['from'], t['to']] += 1
    
    # Per-state statistics
    state_stats = {}
    for state_id in range(16):
        # Count periods in this state
        periods_in_state = np.sum(state_series == state_id)
        
        # Count exits from this state
        exits = sum(1 for t in transitions if t['from'] == state_id)
        
        if periods_in_state > 0:
            exit_rate = exits / periods_in_state
            avg_duration = periods_in_state / max(exits, 1)
            
            state_stats[state_id] = {
                'periods': periods_in_state,
                'exits': exits,
                'exit_rate': exit_rate,
                'avg_duration': avg_duration
            }
    
    return {
        'frequency': frequency,
        'n_periods': n_periods,
        'n_transitions': n_transitions,
        'transition_rate': transition_rate,
        'state_stats': state_stats,
        'transition_matrix': matrix
    }


def analyze_path_dependence(states: pd.DataFrame) -> Dict:
    """Analyze duration by entry source."""
    state_series = states['state'].values
    
    # Extract visits with entry source
    visits = []
    current_state = state_series[0]
    entry_idx = 0
    prev_state = None
    
    for i in range(1, len(state_series)):
        if state_series[i] != current_state:
            duration = i - entry_idx
            visits.append({
                'from': prev_state,
                'state': current_state,
                'to': state_series[i],
                'duration': duration
            })
            prev_state = current_state
            current_state = state_series[i]
            entry_idx = i
    
    # Aggregate by (from, state) pairs
    path_stats = {}
    for v in visits:
        if v['from'] is None:
            continue
        key = (int(v['from']), int(v['state']))
        if key not in path_stats:
            path_stats[key] = []
        path_stats[key].append(v['duration'])
    
    # Calculate statistics
    results = {}
    for (from_state, to_state), durations in path_stats.items():
        if len(durations) >= 5:
            results[(from_state, to_state)] = {
                'count': len(durations),
                'avg_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'std_duration': np.std(durations)
            }
    
    return results


# =============================================================================
# DISPLAY
# =============================================================================

def display_comparison(hourly: Dict, daily: Dict, pair: str):
    """Display hourly vs daily comparison."""
    
    print(f"""
{'='*100}
{pair}: HOURLY vs DAILY TRANSITION COMPARISON
{'='*100}

OVERVIEW
{'─'*50}
                          HOURLY          DAILY
Total Periods:        {hourly['n_periods']:>10,}     {daily['n_periods']:>10,}
Total Transitions:    {hourly['n_transitions']:>10,}     {daily['n_transitions']:>10,}
Transition Rate:      {hourly['transition_rate']:>10.4f}     {daily['transition_rate']:>10.4f}
Avg Periods/Trans:    {hourly['n_periods']/max(hourly['n_transitions'],1):>10.1f}     {daily['n_periods']/max(daily['n_transitions'],1):>10.1f}
""")
    
    # Convert hourly rate to daily equivalent
    # P(no transition in 24h) = (1 - hourly_rate)^24
    # P(transition in 24h) = 1 - (1 - hourly_rate)^24
    hourly_to_daily = 1 - (1 - hourly['transition_rate']) ** 24
    
    print(f"""
RATE COMPARISON
{'─'*50}
Hourly transition rate:      {hourly['transition_rate']:.4f} ({hourly['transition_rate']*100:.2f}%)
Daily transition rate:       {daily['transition_rate']:.4f} ({daily['transition_rate']*100:.2f}%)
Hourly rate → Daily equiv:   {hourly_to_daily:.4f} ({hourly_to_daily*100:.2f}%)
Ratio (hourly_equiv/daily):  {hourly_to_daily/daily['transition_rate']:.2f}x
""")
    
    # Per-state comparison
    print(f"""
PER-STATE EXIT RATES
{'─'*100}
{'State':<8} {'Type':<6} {'Hourly Rate':<14} {'Daily Rate':<14} {'Hourly→Daily':<14} {'Ratio':<10}
{'─'*100}""")
    
    for state_id in range(16):
        if state_id not in hourly['state_stats'] or state_id not in daily['state_stats']:
            continue
        
        h_stats = hourly['state_stats'][state_id]
        d_stats = daily['state_stats'][state_id]
        
        h_rate = h_stats['exit_rate']
        d_rate = d_stats['exit_rate']
        h_to_d = 1 - (1 - h_rate) ** 24
        ratio = h_to_d / d_rate if d_rate > 0 else 0
        
        bull_bear = "BULL" if state_id >= 8 else "BEAR"
        
        print(f"{state_id:<8} {bull_bear:<6} {h_rate:>10.4f}    {d_rate:>10.4f}    {h_to_d:>10.4f}    {ratio:>8.2f}x")
    
    print()


def display_hourly_details(hourly: Dict, path_stats: Dict, pair: str):
    """Display detailed hourly analysis."""
    
    print(f"""
{'='*100}
{pair}: HOURLY STATE ANALYSIS
{'='*100}

STATE DURATION (in hours)
{'─'*80}
{'State':<8} {'Type':<6} {'Hours':<12} {'Exits':<10} {'Exit Rate':<12} {'Avg Duration':<15}
{'─'*80}""")
    
    state_list = []
    for state_id, stats in hourly['state_stats'].items():
        state_list.append((state_id, stats))
    
    state_list = sorted(state_list, key=lambda x: -x[1]['avg_duration'])
    
    for state_id, stats in state_list:
        bull_bear = "BULL" if state_id >= 8 else "BEAR"
        print(f"{state_id:<8} {bull_bear:<6} {stats['periods']:>10,}  {stats['exits']:>8,}  "
              f"{stats['exit_rate']:>10.4f}  {stats['avg_duration']:>12.1f}h")
    
    # Path dependence
    if path_stats:
        print(f"""

PATH DEPENDENCE (top paths by sample size)
{'─'*80}
{'From→To':<12} {'Count':<10} {'Avg Hours':<12} {'Median':<12} {'Std':<12}
{'─'*80}""")
        
        sorted_paths = sorted(path_stats.items(), key=lambda x: -x[1]['count'])[:20]
        
        for (from_s, to_s), stats in sorted_paths:
            print(f"{from_s:>3}→{to_s:<3}     {stats['count']:>8}  {stats['avg_duration']:>10.1f}h  "
                  f"{stats['median_duration']:>10.1f}h  {stats['std_duration']:>10.1f}h")


def display_cross_pair_summary(all_results: Dict):
    """Display cross-pair summary."""
    
    print(f"""

{'='*100}
CROSS-PAIR SUMMARY: HOURLY vs DAILY
{'='*100}

{'Pair':<10} {'Hourly Rate':<14} {'Daily Rate':<14} {'Hourly→Daily':<14} {'Ratio':<10} {'Match?':<10}
{'─'*80}""")
    
    for pair, (hourly, daily) in all_results.items():
        h_rate = hourly['transition_rate']
        d_rate = daily['transition_rate']
        h_to_d = 1 - (1 - h_rate) ** 24
        ratio = h_to_d / d_rate if d_rate > 0 else 0
        
        # Does hourly-to-daily match actual daily?
        match = "YES" if 0.8 <= ratio <= 1.2 else "NO"
        
        print(f"{pair:<10} {h_rate:>10.4f}    {d_rate:>10.4f}    {h_to_d:>10.4f}    {ratio:>8.2f}x  {match:<10}")
    
    print(f"""

INTERPRETATION
{'─'*80}
- If Ratio ≈ 1.0: Hourly and daily are consistent (good)
- If Ratio > 1.0: More transitions detected hourly (could be noise)
- If Ratio < 1.0: Fewer transitions hourly (states change within day but revert)
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hourly vs Daily Transition Analysis')
    parser.add_argument('--pair', type=str, default=None, help='Analyze specific pair')
    parser.add_argument('--all-pairs', action='store_true', help='Analyze all pairs')
    args = parser.parse_args()
    
    print("=" * 100)
    print("HOURLY vs DAILY TRANSITION ANALYSIS")
    print("=" * 100)
    
    print("""
QUESTION: Does hourly granularity add signal or noise?

If hourly rate → daily equivalent ≈ actual daily rate:
    → Transitions are "real" and persist
    → Hourly tracking adds precision

If hourly rate → daily equivalent > actual daily rate:
    → Many intraday transitions that revert
    → Hourly tracking adds noise

If hourly rate → daily equivalent < actual daily rate:
    → States can flip multiple times per day
    → Daily sampling misses some transitions
""")
    
    if args.pair:
        pairs = [args.pair]
    elif args.all_pairs:
        pairs = DEPLOY_PAIRS
    else:
        pairs = ['ETHUSD']
    
    print(f"\nPairs: {pairs}\n")
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    all_results = {}
    
    for pair in pairs:
        print(f"Processing {pair}...", end=" ", flush=True)
        
        # Load data
        df_1h = db.get_ohlcv(pair)
        
        # Compute states at both frequencies
        states_hourly = compute_regime_states_hourly(df_1h)
        states_daily = compute_regime_states_daily(df_1h)
        
        # Analyze transitions
        hourly = analyze_transitions(states_hourly, 'hourly')
        daily = analyze_transitions(states_daily, 'daily')
        
        # Path dependence at hourly scale
        path_stats = analyze_path_dependence(states_hourly)
        
        all_results[pair] = (hourly, daily)
        
        print("done")
        
        # Display comparison
        display_comparison(hourly, daily, pair)
        display_hourly_details(hourly, path_stats, pair)
    
    # Cross-pair summary
    if len(pairs) > 1:
        display_cross_pair_summary(all_results)
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    results = main()