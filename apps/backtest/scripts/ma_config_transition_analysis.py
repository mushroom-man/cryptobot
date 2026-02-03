#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Transitions Across MA Configurations
==========================================
Compare daily state transitions across different 24h MA update frequencies.

Question: Does faster MA updating change daily transition patterns?

Configs:
    A: 24h@24h - MA updates at 00:00 UTC only (baseline)
    B: 24h@12h - MA updates every 12h
    C: 24h@6h  - MA updates every 6h  
    D: 24h@1h  - MA updates every hour

For each config, we sample the state at 00:00 UTC daily and count transitions.

Usage:
    python ma_config_transition_analysis.py --pair ETHUSD
    python ma_config_transition_analysis.py --all-pairs
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

# MA update configurations to test
MA_CONFIGS = {
    'A_24h@24h': {'ma_24h_freq': '24h'},  # Baseline
    'B_24h@12h': {'ma_24h_freq': '12h'},
    'C_24h@6h':  {'ma_24h_freq': '6h'},
    'D_24h@1h':  {'ma_24h_freq': '1h'},
}


# =============================================================================
# REGIME COMPUTATION WITH CONFIGURABLE MA FREQUENCY
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_ma_at_frequency(df_1h: pd.DataFrame, ma_period: int, update_freq: str) -> pd.Series:
    """
    Compute MA with specified update frequency.
    
    Args:
        df_1h: Hourly OHLCV data
        ma_period: Number of periods for MA (e.g., 16 for 16-day MA)
        update_freq: How often to update MA ('24h', '12h', '6h', '1h')
    
    Returns:
        MA values at hourly frequency (forward-filled from update points)
    """
    # Resample to update frequency
    df_freq = resample_ohlcv(df_1h, update_freq)
    
    # For 24h MA, we need 24h bars, but update at different frequencies
    # Key insight: MA(16) of 24h bars = MA of last 16*24 hours
    # When updating more frequently, we still want a 16-day equivalent
    
    if update_freq == '24h':
        # Standard: 16-period MA of 24h bars
        ma = df_freq['close'].rolling(ma_period).mean()
    else:
        # Faster update: compute MA over equivalent window
        # 16 days = 16 * 24 hours = 384 hours
        # At 12h frequency: 384/12 = 32 periods
        # At 6h frequency: 384/6 = 64 periods
        # At 1h frequency: 384/1 = 384 periods
        hours_per_period = {'24h': 24, '12h': 12, '6h': 6, '1h': 1}[update_freq]
        equivalent_periods = (ma_period * 24) // hours_per_period
        ma = df_freq['close'].rolling(equivalent_periods).mean()
    
    # Forward-fill to hourly
    ma_hourly = ma.reindex(df_1h.index, method='ffill')
    
    return ma_hourly


def compute_regime_states_with_config(df_1h: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Compute 16-state regime with specified MA configuration.
    
    Returns states at DAILY frequency (sampled at 00:00 UTC).
    """
    ma_24h_freq = config['ma_24h_freq']
    
    # 72h and 168h MAs stay at their natural frequencies
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # 24h MA with configurable update frequency
    ma_24h = compute_ma_at_frequency(df_1h, MA_PERIOD_24H, ma_24h_freq)
    
    # Forward-fill 72h and 168h to hourly
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
    # Compute trends with hysteresis at hourly frequency
    def compute_trend(close: pd.Series, ma: pd.Series, 
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
    
    # Build hourly states
    states_hourly = pd.DataFrame(index=df_1h.index)
    states_hourly['close'] = df_1h['close']
    states_hourly['ma_24h'] = ma_24h
    states_hourly['ma_72h'] = ma_72h_hourly
    states_hourly['ma_168h'] = ma_168h_hourly
    
    states_hourly['trend_24h'] = compute_trend(df_1h['close'], ma_24h)
    states_hourly['trend_168h'] = compute_trend(df_1h['close'], ma_168h_hourly)
    states_hourly['ma72_above_ma24'] = (ma_72h_hourly > ma_24h).astype(int)
    states_hourly['ma168_above_ma24'] = (ma_168h_hourly > ma_24h).astype(int)
    
    states_hourly['state'] = (
        states_hourly['trend_24h'] * 8 +
        states_hourly['trend_168h'] * 4 +
        states_hourly['ma72_above_ma24'] * 2 +
        states_hourly['ma168_above_ma24'] * 1
    )
    
    # Sample at daily frequency (00:00 UTC)
    states_daily = states_hourly.resample('24h').first().dropna()
    
    return states_daily


# =============================================================================
# TRANSITION ANALYSIS
# =============================================================================

def analyze_transitions(states: pd.DataFrame) -> Dict:
    """Analyze state transitions."""
    state_series = states['state'].values
    n_days = len(state_series)
    
    # Count transitions
    transitions = []
    for i in range(1, len(state_series)):
        if state_series[i] != state_series[i-1]:
            transitions.append({
                'from': int(state_series[i-1]),
                'to': int(state_series[i]),
                'day': i
            })
    
    n_transitions = len(transitions)
    transition_rate = n_transitions / n_days if n_days > 0 else 0
    
    # Transition matrix
    matrix = np.zeros((16, 16), dtype=int)
    for t in transitions:
        matrix[t['from'], t['to']] += 1
    
    # Per-state stats
    state_stats = {}
    for state_id in range(16):
        days_in_state = np.sum(state_series == state_id)
        exits = sum(1 for t in transitions if t['from'] == state_id)
        
        if days_in_state > 0:
            exit_rate = exits / days_in_state
            avg_duration = days_in_state / max(exits, 1)
            state_stats[state_id] = {
                'days': days_in_state,
                'exits': exits,
                'exit_rate': exit_rate,
                'avg_duration': avg_duration
            }
    
    # Visit durations
    durations = []
    current_state = state_series[0]
    current_duration = 1
    
    for i in range(1, len(state_series)):
        if state_series[i] == current_state:
            current_duration += 1
        else:
            durations.append({'state': current_state, 'duration': current_duration})
            current_state = state_series[i]
            current_duration = 1
    
    return {
        'n_days': n_days,
        'n_transitions': n_transitions,
        'transition_rate': transition_rate,
        'matrix': matrix,
        'state_stats': state_stats,
        'durations': durations,
        'transitions': transitions
    }


def compare_state_sequences(seq_a: np.ndarray, seq_b: np.ndarray) -> Dict:
    """
    Compare two state sequences.
    
    Returns:
        agreement_rate: % of days with same state
        transitions_only_in_a: transitions in A but not B
        transitions_only_in_b: transitions in B but not A
    """
    min_len = min(len(seq_a), len(seq_b))
    seq_a = seq_a[:min_len]
    seq_b = seq_b[:min_len]
    
    agreement = np.mean(seq_a == seq_b)
    
    # Find transitions in each
    trans_a = set()
    trans_b = set()
    
    for i in range(1, min_len):
        if seq_a[i] != seq_a[i-1]:
            trans_a.add(i)
        if seq_b[i] != seq_b[i-1]:
            trans_b.add(i)
    
    only_a = trans_a - trans_b
    only_b = trans_b - trans_a
    both = trans_a & trans_b
    
    return {
        'agreement_rate': agreement,
        'trans_in_a': len(trans_a),
        'trans_in_b': len(trans_b),
        'trans_only_a': len(only_a),
        'trans_only_b': len(only_b),
        'trans_both': len(both),
        'trans_overlap': len(both) / max(len(trans_a | trans_b), 1)
    }


def analyze_transition_stability(transitions: List[Dict], states: np.ndarray) -> Dict:
    """
    Analyze how stable transitions are (do they revert quickly?).
    
    Look at transitions and check if the new state persists for at least N days.
    """
    stability = {
        'persist_1d': 0,  # Still in new state after 1 day
        'persist_3d': 0,  # Still in new state after 3 days
        'persist_7d': 0,  # Still in new state after 7 days
        'total': len(transitions)
    }
    
    for t in transitions:
        day = t['day']
        to_state = t['to']
        
        # Check persistence
        if day + 1 < len(states) and states[day + 1] == to_state:
            stability['persist_1d'] += 1
        if day + 3 < len(states) and states[day + 3] == to_state:
            stability['persist_3d'] += 1
        if day + 7 < len(states) and states[day + 7] == to_state:
            stability['persist_7d'] += 1
    
    if stability['total'] > 0:
        stability['persist_1d_pct'] = stability['persist_1d'] / stability['total']
        stability['persist_3d_pct'] = stability['persist_3d'] / stability['total']
        stability['persist_7d_pct'] = stability['persist_7d'] / stability['total']
    else:
        stability['persist_1d_pct'] = 0
        stability['persist_3d_pct'] = 0
        stability['persist_7d_pct'] = 0
    
    return stability


# =============================================================================
# DISPLAY
# =============================================================================

def display_config_comparison(results: Dict[str, Dict], pair: str):
    """Display comparison of MA configurations."""
    
    print(f"""
{'='*100}
{pair}: MA CONFIGURATION TRANSITION COMPARISON
{'='*100}

OVERVIEW
{'─'*80}
{'Config':<15} {'Days':<10} {'Transitions':<12} {'Rate':<12} {'Avg Duration':<15}
{'─'*80}""")
    
    for config_name, result in results.items():
        avg_dur = result['n_days'] / max(result['n_transitions'], 1)
        print(f"{config_name:<15} {result['n_days']:<10} {result['n_transitions']:<12} "
              f"{result['transition_rate']:<12.1%} {avg_dur:<15.1f}d")
    
    # Transition stability
    print(f"""

TRANSITION STABILITY (does new state persist?)
{'─'*80}
{'Config':<15} {'Total Trans':<12} {'Persist 1d':<12} {'Persist 3d':<12} {'Persist 7d':<12}
{'─'*80}""")
    
    for config_name, result in results.items():
        stab = result['stability']
        print(f"{config_name:<15} {stab['total']:<12} {stab['persist_1d_pct']:<12.1%} "
              f"{stab['persist_3d_pct']:<12.1%} {stab['persist_7d_pct']:<12.1%}")


def display_state_comparison(results: Dict[str, Dict], pair: str):
    """Display per-state comparison across configs."""
    
    print(f"""

PER-STATE EXIT RATES BY CONFIG
{'─'*100}
{'State':<8} {'Type':<6}""", end="")
    
    for config_name in results.keys():
        short_name = config_name.split('_')[0]
        print(f" {short_name:<12}", end="")
    print()
    print("─" * 100)
    
    for state_id in range(16):
        has_data = any(state_id in results[c]['state_stats'] for c in results)
        if not has_data:
            continue
        
        bull_bear = "BULL" if state_id >= 8 else "BEAR"
        print(f"{state_id:<8} {bull_bear:<6}", end="")
        
        for config_name, result in results.items():
            if state_id in result['state_stats']:
                rate = result['state_stats'][state_id]['exit_rate']
                print(f" {rate:<12.1%}", end="")
            else:
                print(f" {'N/A':<12}", end="")
        print()


def display_sequence_comparison(results: Dict[str, Dict], pair: str):
    """Compare state sequences between configs."""
    
    print(f"""

SEQUENCE AGREEMENT (vs Baseline A_24h@24h)
{'─'*80}
{'Config':<15} {'Agreement':<12} {'Trans Overlap':<15} {'Only Baseline':<15} {'Only This':<12}
{'─'*80}""")
    
    baseline_states = results['A_24h@24h']['states']
    
    for config_name, result in results.items():
        if config_name == 'A_24h@24h':
            print(f"{config_name:<15} {'(baseline)':<12}")
            continue
        
        comp = compare_state_sequences(baseline_states, result['states'])
        print(f"{config_name:<15} {comp['agreement_rate']:<12.1%} {comp['trans_overlap']:<15.1%} "
              f"{comp['trans_only_a']:<15} {comp['trans_only_b']:<12}")


def display_transition_timing(results: Dict[str, Dict], pair: str):
    """Analyze when faster configs detect transitions vs baseline."""
    
    print(f"""

TRANSITION TIMING ANALYSIS
{'─'*80}
Do faster MA updates detect transitions EARLIER than baseline?
""")
    
    baseline = results['A_24h@24h']
    baseline_trans_days = set(t['day'] for t in baseline['transitions'])
    
    for config_name, result in results.items():
        if config_name == 'A_24h@24h':
            continue
        
        config_trans_days = set(t['day'] for t in result['transitions'])
        
        # Find transitions that occur in both, check timing
        earlier = 0
        same_day = 0
        later = 0
        only_config = 0
        only_baseline = 0
        
        # For each baseline transition, find nearest config transition within ±3 days
        for b_day in baseline_trans_days:
            found = False
            for c_day in config_trans_days:
                if abs(c_day - b_day) <= 3:
                    if c_day < b_day:
                        earlier += 1
                    elif c_day == b_day:
                        same_day += 1
                    else:
                        later += 1
                    found = True
                    break
            if not found:
                only_baseline += 1
        
        # Transitions only in config (within ±3 days of any baseline)
        for c_day in config_trans_days:
            if not any(abs(c_day - b_day) <= 3 for b_day in baseline_trans_days):
                only_config += 1
        
        total = earlier + same_day + later
        print(f"{config_name}:")
        print(f"  Same day as baseline:    {same_day:>4} ({same_day/max(total,1):.0%})")
        print(f"  Earlier than baseline:   {earlier:>4} ({earlier/max(total,1):.0%})")
        print(f"  Later than baseline:     {later:>4} ({later/max(total,1):.0%})")
        print(f"  Only in this config:     {only_config:>4}")
        print(f"  Only in baseline:        {only_baseline:>4}")
        print()


def display_cross_pair_summary(all_results: Dict[str, Dict[str, Dict]]):
    """Display cross-pair summary."""
    
    print(f"""

{'='*100}
CROSS-PAIR SUMMARY
{'='*100}

TRANSITION RATES BY CONFIG
{'─'*80}
{'Pair':<10}""", end="")
    
    configs = list(next(iter(all_results.values())).keys())
    for config_name in configs:
        short_name = config_name.split('_')[0]
        print(f" {short_name:<12}", end="")
    print()
    print("─" * 80)
    
    for pair, results in all_results.items():
        print(f"{pair:<10}", end="")
        for config_name in configs:
            rate = results[config_name]['transition_rate']
            print(f" {rate:<12.1%}", end="")
        print()
    
    # Average persistence
    print(f"""

TRANSITION PERSISTENCE (3-day) BY CONFIG
{'─'*80}
{'Pair':<10}""", end="")
    
    for config_name in configs:
        short_name = config_name.split('_')[0]
        print(f" {short_name:<12}", end="")
    print()
    print("─" * 80)
    
    for pair, results in all_results.items():
        print(f"{pair:<10}", end="")
        for config_name in configs:
            persist = results[config_name]['stability']['persist_3d_pct']
            print(f" {persist:<12.1%}", end="")
        print()
    
    # Agreement with baseline
    print(f"""

STATE AGREEMENT WITH BASELINE (A_24h@24h)
{'─'*80}
{'Pair':<10}""", end="")
    
    for config_name in configs:
        if config_name == 'A_24h@24h':
            continue
        short_name = config_name.split('_')[0]
        print(f" {short_name:<12}", end="")
    print()
    print("─" * 80)
    
    for pair, results in all_results.items():
        baseline_states = results['A_24h@24h']['states']
        print(f"{pair:<10}", end="")
        for config_name in configs:
            if config_name == 'A_24h@24h':
                continue
            comp = compare_state_sequences(baseline_states, results[config_name]['states'])
            print(f" {comp['agreement_rate']:<12.1%}", end="")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MA Config Transition Analysis')
    parser.add_argument('--pair', type=str, default=None, help='Analyze specific pair')
    parser.add_argument('--all-pairs', action='store_true', help='Analyze all pairs')
    args = parser.parse_args()
    
    print("=" * 100)
    print("MA CONFIGURATION TRANSITION ANALYSIS")
    print("=" * 100)
    
    print("""
QUESTION: How do different 24h MA update frequencies affect daily transitions?

Configs:
    A: 24h@24h - MA updates daily at 00:00 UTC (baseline)
    B: 24h@12h - MA updates every 12 hours
    C: 24h@6h  - MA updates every 6 hours
    D: 24h@1h  - MA updates every hour

All configs sample STATE at daily frequency (00:00 UTC).
We're testing if faster MA updates change the daily regime picture.
""")
    
    if args.pair:
        pairs = [args.pair]
    elif args.all_pairs:
        pairs = DEPLOY_PAIRS
    else:
        pairs = ['ETHUSD']
    
    print(f"Pairs: {pairs}\n")
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    all_results = {}
    
    for pair in pairs:
        print(f"Processing {pair}...", end=" ", flush=True)
        
        # Load data
        df_1h = db.get_ohlcv(pair)
        
        results = {}
        
        for config_name, config in MA_CONFIGS.items():
            # Compute states
            states_df = compute_regime_states_with_config(df_1h, config)
            states_array = states_df['state'].values
            
            # Analyze transitions
            analysis = analyze_transitions(states_df)
            
            # Analyze stability
            stability = analyze_transition_stability(analysis['transitions'], states_array)
            
            results[config_name] = {
                **analysis,
                'stability': stability,
                'states': states_array
            }
        
        all_results[pair] = results
        print("done")
        
        # Display results for this pair
        display_config_comparison(results, pair)
        display_state_comparison(results, pair)
        display_sequence_comparison(results, pair)
        display_transition_timing(results, pair)
    
    # Cross-pair summary
    if len(pairs) > 1:
        display_cross_pair_summary(all_results)
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    results = main()