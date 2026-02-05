#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transition Quality Analysis
===========================
Test whether baseline's extra transitions are valuable signals or noise.

Key Question: When baseline detects a transition that faster configs miss,
is that transition PROFITABLE or is it WHIPSAW?

Method:
1. Identify transitions unique to each config
2. Measure returns in the N days AFTER each transition
3. Compare transition quality across configs

If baseline's extra transitions are valuable:
    → Returns after "baseline-only" transitions should be positive
    → These transitions catch real inflection points

If baseline's extra transitions are noise:
    → Returns after "baseline-only" transitions should be ~0 or negative
    → Faster configs correctly filtered these out

Usage:
    python transition_quality_analysis.py --pair ETHUSD
    python transition_quality_analysis.py --all-pairs
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

# Horizons to measure returns after transition (in periods of sampling frequency)
RETURN_HORIZONS = [1, 3, 5, 7, 14]

# MA update configurations
MA_CONFIGS = {
    'A_24h@24h': {'ma_24h_freq': '24h'},
    'B_24h@12h': {'ma_24h_freq': '12h'},
    'C_24h@6h':  {'ma_24h_freq': '6h'},
    'D_24h@1h':  {'ma_24h_freq': '1h'},
}

# State sampling frequencies to test
SAMPLING_FREQUENCIES = ['24h', '12h', '6h', '1h']

# State classification
BULLISH_STATES = [8, 9, 10, 11, 12, 13, 14, 15]
BEARISH_STATES = [0, 1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# REGIME COMPUTATION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_ma_at_frequency(df_1h: pd.DataFrame, ma_period: int, update_freq: str) -> pd.Series:
    df_freq = resample_ohlcv(df_1h, update_freq)
    
    if update_freq == '24h':
        ma = df_freq['close'].rolling(ma_period).mean()
    else:
        hours_per_period = {'24h': 24, '12h': 12, '6h': 6, '1h': 1}[update_freq]
        equivalent_periods = (ma_period * 24) // hours_per_period
        ma = df_freq['close'].rolling(equivalent_periods).mean()
    
    ma_hourly = ma.reindex(df_1h.index, method='ffill')
    return ma_hourly


def compute_regime_states_with_config(df_1h: pd.DataFrame, config: Dict) -> pd.DataFrame:
    ma_24h_freq = config['ma_24h_freq']
    
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_24h = compute_ma_at_frequency(df_1h, MA_PERIOD_24H, ma_24h_freq)
    
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
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
    
    states_hourly = pd.DataFrame(index=df_1h.index)
    states_hourly['close'] = df_1h['close']
    states_hourly['ma_24h'] = ma_24h
    
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
    
    states_daily = states_hourly.resample('24h').first().dropna()
    return states_daily


def compute_regime_states_at_frequency(df_1h: pd.DataFrame, sampling_freq: str, 
                                        ma_update_freq: str = '1h') -> pd.DataFrame:
    """
    Compute 16-state regime sampled at specified frequency.
    
    Args:
        df_1h: Hourly OHLCV data
        sampling_freq: How often to sample state ('1h', '6h', '12h', '24h')
        ma_update_freq: How often to update 24h MA (default '1h' for smoothest)
    
    Returns:
        States at specified sampling frequency
    """
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Use specified MA update frequency
    ma_24h = compute_ma_at_frequency(df_1h, MA_PERIOD_24H, ma_update_freq)
    
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
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
    
    states_hourly = pd.DataFrame(index=df_1h.index)
    states_hourly['close'] = df_1h['close']
    states_hourly['ma_24h'] = ma_24h
    
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
    
    # Resample to specified frequency
    if sampling_freq == '1h':
        return states_hourly.dropna()
    else:
        return states_hourly.resample(sampling_freq).first().dropna()


# =============================================================================
# TRANSITION EXTRACTION
# =============================================================================

def extract_transitions(states: pd.DataFrame) -> List[Dict]:
    """Extract all transitions with metadata."""
    state_series = states['state'].values
    close_series = states['close'].values
    dates = states.index
    
    transitions = []
    for i in range(1, len(state_series)):
        if state_series[i] != state_series[i-1]:
            from_state = int(state_series[i-1])
            to_state = int(state_series[i])
            
            # Direction of transition
            from_bull = from_state in BULLISH_STATES
            to_bull = to_state in BULLISH_STATES
            
            if from_bull and not to_bull:
                direction = 'BULL_TO_BEAR'
            elif not from_bull and to_bull:
                direction = 'BEAR_TO_BULL'
            elif from_bull and to_bull:
                direction = 'BULL_TO_BULL'
            else:
                direction = 'BEAR_TO_BEAR'
            
            transitions.append({
                'day': i,
                'date': dates[i],
                'from_state': from_state,
                'to_state': to_state,
                'direction': direction,
                'price_at_transition': close_series[i]
            })
    
    return transitions


def classify_transitions(transitions_a: List[Dict], transitions_b: List[Dict], 
                         window: int = 2) -> Dict:
    """
    Classify transitions as:
    - both: detected by both configs (within window days)
    - only_a: detected only by config A
    - only_b: detected only by config B
    """
    days_a = {t['day']: t for t in transitions_a}
    days_b = {t['day']: t for t in transitions_b}
    
    both = []
    only_a = []
    only_b = []
    
    matched_b = set()
    
    for day_a, trans_a in days_a.items():
        found_match = False
        for offset in range(-window, window + 1):
            day_b = day_a + offset
            if day_b in days_b and day_b not in matched_b:
                both.append({
                    'day_a': day_a,
                    'day_b': day_b,
                    'trans_a': trans_a,
                    'trans_b': days_b[day_b],
                    'timing_diff': day_b - day_a  # positive = B detected later
                })
                matched_b.add(day_b)
                found_match = True
                break
        
        if not found_match:
            only_a.append(trans_a)
    
    for day_b, trans_b in days_b.items():
        if day_b not in matched_b:
            only_b.append(trans_b)
    
    return {
        'both': both,
        'only_a': only_a,
        'only_b': only_b
    }


# =============================================================================
# RETURN ANALYSIS
# =============================================================================

def compute_returns_after_transition(transitions: List[Dict], prices: pd.Series, 
                                     horizons: List[int]) -> Dict:
    """
    Compute returns after each transition.
    
    For BULL_TO_BEAR transitions: we expect NEGATIVE returns (we'd be short/flat)
    For BEAR_TO_BULL transitions: we expect POSITIVE returns (we'd be long)
    
    A "good" transition is one where the market moves in the expected direction.
    """
    results = {h: [] for h in horizons}
    detailed = []
    
    price_array = prices.values
    
    for trans in transitions:
        day = trans['day']
        direction = trans['direction']
        
        for horizon in horizons:
            if day + horizon < len(price_array):
                entry_price = price_array[day]
                exit_price = price_array[day + horizon]
                raw_return = (exit_price - entry_price) / entry_price
                
                # "Correct" return depends on transition direction
                # If we went BULL_TO_BEAR, we'd exit long, so positive raw return = bad signal
                # If we went BEAR_TO_BULL, we'd enter long, so positive raw return = good signal
                
                if direction in ['BEAR_TO_BULL', 'BEAR_TO_BEAR']:
                    # We're bullish after transition, want positive returns
                    signal_return = raw_return
                else:
                    # We're bearish after transition, want negative returns (short gains)
                    signal_return = -raw_return
                
                results[horizon].append({
                    'raw_return': raw_return,
                    'signal_return': signal_return,
                    'direction': direction
                })
        
        # Store detailed info for first horizon
        if day + horizons[0] < len(price_array):
            entry_price = price_array[day]
            exit_price = price_array[day + horizons[0]]
            raw_return = (exit_price - entry_price) / entry_price
            
            detailed.append({
                **trans,
                'raw_return': raw_return,
                'direction': direction
            })
    
    return {'by_horizon': results, 'detailed': detailed}


def summarize_returns(returns_data: Dict, horizons: List[int]) -> Dict:
    """Summarize return statistics."""
    summary = {}
    
    for horizon in horizons:
        data = returns_data['by_horizon'][horizon]
        if not data:
            summary[horizon] = None
            continue
        
        raw_returns = [d['raw_return'] for d in data]
        signal_returns = [d['signal_return'] for d in data]
        
        summary[horizon] = {
            'n': len(data),
            'raw_mean': np.mean(raw_returns),
            'raw_std': np.std(raw_returns),
            'signal_mean': np.mean(signal_returns),  # This is the key metric
            'signal_std': np.std(signal_returns),
            'signal_positive_pct': np.mean([r > 0 for r in signal_returns]),
            'raw_positive_pct': np.mean([r > 0 for r in raw_returns]),
        }
    
    return summary


def analyze_by_direction(returns_data: Dict, horizons: List[int]) -> Dict:
    """Break down returns by transition direction."""
    directions = ['BEAR_TO_BULL', 'BULL_TO_BEAR', 'BULL_TO_BULL', 'BEAR_TO_BEAR']
    
    results = {}
    for horizon in horizons:
        data = returns_data['by_horizon'][horizon]
        if not data:
            continue
        
        results[horizon] = {}
        for direction in directions:
            dir_data = [d for d in data if d['direction'] == direction]
            if dir_data:
                raw_returns = [d['raw_return'] for d in dir_data]
                results[horizon][direction] = {
                    'n': len(dir_data),
                    'mean_return': np.mean(raw_returns),
                    'positive_pct': np.mean([r > 0 for r in raw_returns])
                }
    
    return results


# =============================================================================
# MULTI-FREQUENCY ANALYSIS
# =============================================================================

def analyze_sampling_frequency(df_1h: pd.DataFrame, sampling_freq: str, 
                                ma_update_freq: str = '1h') -> Dict:
    """
    Analyze transitions at a specific sampling frequency.
    
    Returns in HOURS for comparability across frequencies.
    """
    states = compute_regime_states_at_frequency(df_1h, sampling_freq, ma_update_freq)
    transitions = extract_transitions(states)
    
    price_series = states['close']
    price_array = price_series.values
    dates = states.index
    
    # Hours per period for this sampling frequency
    hours_per_period = {'1h': 1, '6h': 6, '12h': 12, '24h': 24}[sampling_freq]
    
    # Target return horizons in hours: 6h, 12h, 24h, 48h, 72h
    target_hours = [6, 12, 24, 48, 72]
    
    results_by_horizon = {h: [] for h in target_hours}
    
    for trans in transitions:
        day = trans['day']
        direction = trans['direction']
        
        for target_h in target_hours:
            # How many periods to look forward
            periods_forward = target_h // hours_per_period
            
            if periods_forward == 0:
                continue  # Can't measure this horizon at this frequency
            
            if day + periods_forward < len(price_array):
                entry_price = price_array[day]
                exit_price = price_array[day + periods_forward]
                raw_return = (exit_price - entry_price) / entry_price
                
                if direction in ['BEAR_TO_BULL', 'BEAR_TO_BEAR']:
                    signal_return = raw_return
                else:
                    signal_return = -raw_return
                
                results_by_horizon[target_h].append({
                    'raw_return': raw_return,
                    'signal_return': signal_return,
                    'direction': direction
                })
    
    # Summarize
    summary = {}
    for horizon_h, data in results_by_horizon.items():
        if not data:
            summary[horizon_h] = None
            continue
        
        raw_returns = [d['raw_return'] for d in data]
        signal_returns = [d['signal_return'] for d in data]
        
        summary[horizon_h] = {
            'n': len(data),
            'raw_mean': np.mean(raw_returns),
            'signal_mean': np.mean(signal_returns),
            'signal_win_pct': np.mean([r > 0 for r in signal_returns]),
        }
    
    # Transition statistics
    n_transitions = len(transitions)
    n_periods = len(states)
    trans_rate = n_transitions / n_periods if n_periods > 0 else 0
    
    # Persistence: does new state last at least 1 period?
    persist_1 = 0
    persist_3 = 0
    state_array = states['state'].values
    for trans in transitions:
        day = trans['day']
        to_state = trans['to_state']
        if day + 1 < len(state_array) and state_array[day + 1] == to_state:
            persist_1 += 1
        periods_for_24h = 24 // hours_per_period
        if day + periods_for_24h < len(state_array) and state_array[day + periods_for_24h] == to_state:
            persist_3 += 1
    
    return {
        'sampling_freq': sampling_freq,
        'ma_update_freq': ma_update_freq,
        'n_periods': n_periods,
        'n_transitions': n_transitions,
        'transition_rate': trans_rate,
        'transitions_per_day': trans_rate * (24 / hours_per_period),
        'persist_1_period': persist_1 / max(n_transitions, 1),
        'persist_24h': persist_3 / max(n_transitions, 1),
        'returns_by_horizon': summary
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_transition_quality(df_1h: pd.DataFrame, pair: str) -> Dict:
    """Full analysis of transition quality across configs."""
    
    # Compute states for each config
    config_states = {}
    config_transitions = {}
    
    for config_name, config in MA_CONFIGS.items():
        states = compute_regime_states_with_config(df_1h, config)
        transitions = extract_transitions(states)
        config_states[config_name] = states
        config_transitions[config_name] = transitions
    
    # Get price series (daily close)
    prices = config_states['A_24h@24h']['close']
    
    # Classify transitions: baseline vs each faster config
    classifications = {}
    for config_name in ['B_24h@12h', 'C_24h@6h', 'D_24h@1h']:
        classifications[config_name] = classify_transitions(
            config_transitions['A_24h@24h'],
            config_transitions[config_name],
            window=2
        )
    
    # Analyze returns for each category
    results = {}
    
    # 1. All transitions for each config
    for config_name, transitions in config_transitions.items():
        returns_data = compute_returns_after_transition(transitions, prices, RETURN_HORIZONS)
        results[f'{config_name}_all'] = {
            'n_transitions': len(transitions),
            'summary': summarize_returns(returns_data, RETURN_HORIZONS),
            'by_direction': analyze_by_direction(returns_data, RETURN_HORIZONS)
        }
    
    # 2. Transitions only in baseline (the "extra" ones we want to test)
    for config_name, classification in classifications.items():
        only_baseline = classification['only_a']
        only_faster = classification['only_b']
        both = classification['both']
        
        # Returns for baseline-only transitions
        if only_baseline:
            returns_data = compute_returns_after_transition(only_baseline, prices, RETURN_HORIZONS)
            results[f'only_baseline_vs_{config_name}'] = {
                'n_transitions': len(only_baseline),
                'summary': summarize_returns(returns_data, RETURN_HORIZONS),
                'by_direction': analyze_by_direction(returns_data, RETURN_HORIZONS)
            }
        
        # Returns for faster-only transitions
        if only_faster:
            returns_data = compute_returns_after_transition(only_faster, prices, RETURN_HORIZONS)
            results[f'only_{config_name}'] = {
                'n_transitions': len(only_faster),
                'summary': summarize_returns(returns_data, RETURN_HORIZONS),
                'by_direction': analyze_by_direction(returns_data, RETURN_HORIZONS)
            }
        
        # Returns for transitions detected by both
        both_transitions = [b['trans_a'] for b in both]
        if both_transitions:
            returns_data = compute_returns_after_transition(both_transitions, prices, RETURN_HORIZONS)
            results[f'both_baseline_and_{config_name}'] = {
                'n_transitions': len(both_transitions),
                'summary': summarize_returns(returns_data, RETURN_HORIZONS),
                'by_direction': analyze_by_direction(returns_data, RETURN_HORIZONS)
            }
        
        # Timing analysis for "both" transitions
        if both:
            timing_diffs = [b['timing_diff'] for b in both]
            results[f'timing_vs_{config_name}'] = {
                'n': len(timing_diffs),
                'mean_diff': np.mean(timing_diffs),
                'faster_detected_earlier': sum(1 for d in timing_diffs if d < 0),
                'same_day': sum(1 for d in timing_diffs if d == 0),
                'faster_detected_later': sum(1 for d in timing_diffs if d > 0)
            }
    
    # 3. NEW: Analyze transitions at different sampling frequencies
    # Using 1h MA updates for all, but different sampling frequencies
    results['sampling_frequency'] = {}
    for freq in SAMPLING_FREQUENCIES:
        results['sampling_frequency'][freq] = analyze_sampling_frequency(
            df_1h, freq, ma_update_freq='1h'
        )
    
    return results


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(results: Dict, pair: str):
    """Display analysis results."""
    
    print(f"""
{'='*100}
{pair}: TRANSITION QUALITY ANALYSIS
{'='*100}

QUESTION: Are baseline's extra transitions valuable signals or noise?
""")
    
    # Summary of all transitions by config
    print(f"""
ALL TRANSITIONS BY CONFIG
{'─'*80}
{'Config':<20} {'N Trans':<12} {'Avg 3d Return':<15} {'Signal Win%':<15}
{'─'*80}""")
    
    for config in ['A_24h@24h', 'B_24h@12h', 'C_24h@6h', 'D_24h@1h']:
        key = f'{config}_all'
        if key in results:
            r = results[key]
            if r['summary'][3]:
                print(f"{config:<20} {r['n_transitions']:<12} "
                      f"{r['summary'][3]['signal_mean']:>+12.2%}    "
                      f"{r['summary'][3]['signal_positive_pct']:>12.1%}")
    
    # The key comparison: baseline-only vs faster-only transitions
    print(f"""

BASELINE-ONLY TRANSITIONS (detected by baseline, missed by faster config)
{'─'*100}
{'Comparison':<30} {'N':<8} {'1d Ret':<12} {'3d Ret':<12} {'7d Ret':<12} {'Win%':<10}
{'─'*100}""")
    
    for config in ['B_24h@12h', 'C_24h@6h', 'D_24h@1h']:
        key = f'only_baseline_vs_{config}'
        if key in results:
            r = results[key]
            s = r['summary']
            if s[1] and s[3] and s[7]:
                print(f"vs {config:<27} {r['n_transitions']:<8} "
                      f"{s[1]['signal_mean']:>+10.2%}  "
                      f"{s[3]['signal_mean']:>+10.2%}  "
                      f"{s[7]['signal_mean']:>+10.2%}  "
                      f"{s[3]['signal_positive_pct']:>8.1%}")
    
    print(f"""

FASTER-ONLY TRANSITIONS (detected by faster config, missed by baseline)
{'─'*100}
{'Config':<30} {'N':<8} {'1d Ret':<12} {'3d Ret':<12} {'7d Ret':<12} {'Win%':<10}
{'─'*100}""")
    
    for config in ['B_24h@12h', 'C_24h@6h', 'D_24h@1h']:
        key = f'only_{config}'
        if key in results:
            r = results[key]
            s = r['summary']
            if s[1] and s[3] and s[7]:
                print(f"{config:<30} {r['n_transitions']:<8} "
                      f"{s[1]['signal_mean']:>+10.2%}  "
                      f"{s[3]['signal_mean']:>+10.2%}  "
                      f"{s[7]['signal_mean']:>+10.2%}  "
                      f"{s[3]['signal_positive_pct']:>8.1%}")
    
    print(f"""

TRANSITIONS DETECTED BY BOTH (agreement)
{'─'*100}
{'Comparison':<30} {'N':<8} {'1d Ret':<12} {'3d Ret':<12} {'7d Ret':<12} {'Win%':<10}
{'─'*100}""")
    
    for config in ['B_24h@12h', 'C_24h@6h', 'D_24h@1h']:
        key = f'both_baseline_and_{config}'
        if key in results:
            r = results[key]
            s = r['summary']
            if s[1] and s[3] and s[7]:
                print(f"Both A & {config:<21} {r['n_transitions']:<8} "
                      f"{s[1]['signal_mean']:>+10.2%}  "
                      f"{s[3]['signal_mean']:>+10.2%}  "
                      f"{s[7]['signal_mean']:>+10.2%}  "
                      f"{s[3]['signal_positive_pct']:>8.1%}")
    
    # Direction breakdown for baseline-only
    print(f"""

BASELINE-ONLY BY DIRECTION (3-day returns)
{'─'*80}
{'Comparison':<25} {'Direction':<15} {'N':<8} {'Mean Ret':<12} {'Positive%':<10}
{'─'*80}""")
    
    for config in ['D_24h@1h']:  # Focus on biggest difference
        key = f'only_baseline_vs_{config}'
        if key in results and results[key]['by_direction'].get(3):
            for direction, data in results[key]['by_direction'][3].items():
                print(f"vs {config:<22} {direction:<15} {data['n']:<8} "
                      f"{data['mean_return']:>+10.2%}  {data['positive_pct']:>8.1%}")
    
    # NEW: Sampling frequency analysis
    if 'sampling_frequency' in results:
        print(f"""

{'='*100}
SAMPLING FREQUENCY ANALYSIS (MA updates hourly, state sampled at different intervals)
{'='*100}

{'Freq':<8} {'Periods':<10} {'Trans':<10} {'Rate/Period':<12} {'Trans/Day':<12} {'Persist 24h':<12}
{'─'*80}""")
        
        for freq in SAMPLING_FREQUENCIES:
            sf = results['sampling_frequency'][freq]
            print(f"{freq:<8} {sf['n_periods']:<10} {sf['n_transitions']:<10} "
                  f"{sf['transition_rate']:<12.2%} {sf['transitions_per_day']:<12.2f} "
                  f"{sf['persist_24h']:<12.1%}")
        
        print(f"""

SIGNAL RETURNS BY SAMPLING FREQUENCY (using 24h MA updates hourly)
{'─'*100}
{'Freq':<8} {'N Trans':<10} {'12h Ret':<12} {'24h Ret':<12} {'48h Ret':<12} {'72h Ret':<12} {'Win% 24h':<10}
{'─'*100}""")
        
        for freq in SAMPLING_FREQUENCIES:
            sf = results['sampling_frequency'][freq]
            ret = sf['returns_by_horizon']
            
            r12 = f"{ret[12]['signal_mean']:>+.2%}" if ret.get(12) else "N/A"
            r24 = f"{ret[24]['signal_mean']:>+.2%}" if ret.get(24) else "N/A"
            r48 = f"{ret[48]['signal_mean']:>+.2%}" if ret.get(48) else "N/A"
            r72 = f"{ret[72]['signal_mean']:>+.2%}" if ret.get(72) else "N/A"
            win = f"{ret[24]['signal_win_pct']:.1%}" if ret.get(24) else "N/A"
            
            print(f"{freq:<8} {sf['n_transitions']:<10} {r12:<12} {r24:<12} {r48:<12} {r72:<12} {win:<10}")


def display_cross_pair_summary(all_results: Dict):
    """Display cross-pair summary."""
    
    print(f"""

{'='*100}
CROSS-PAIR SUMMARY: ARE BASELINE'S EXTRA TRANSITIONS VALUABLE?
{'='*100}

3-DAY SIGNAL RETURNS BY TRANSITION CATEGORY
{'─'*100}
{'Pair':<10} {'All Baseline':<15} {'Only Baseline':<15} {'Only D(1h)':<15} {'Both':<15}
{'─'*100}""")
    
    for pair, results in all_results.items():
        all_base = results.get('A_24h@24h_all', {}).get('summary', {}).get(3)
        only_base = results.get('only_baseline_vs_D_24h@1h', {}).get('summary', {}).get(3)
        only_fast = results.get('only_D_24h@1h', {}).get('summary', {}).get(3)
        both = results.get('both_baseline_and_D_24h@1h', {}).get('summary', {}).get(3)
        
        all_base_str = f"{all_base['signal_mean']:>+.2%}" if all_base else "N/A"
        only_base_str = f"{only_base['signal_mean']:>+.2%}" if only_base else "N/A"
        only_fast_str = f"{only_fast['signal_mean']:>+.2%}" if only_fast else "N/A"
        both_str = f"{both['signal_mean']:>+.2%}" if both else "N/A"
        
        print(f"{pair:<10} {all_base_str:<15} {only_base_str:<15} {only_fast_str:<15} {both_str:<15}")
    
    print(f"""

INTERPRETATION
{'─'*80}
- "Only Baseline" > 0: Baseline's extra transitions ARE valuable signals
- "Only Baseline" ≈ 0: Baseline's extra transitions are noise (whipsaw)
- "Only Baseline" < 0: Baseline's extra transitions are BAD signals

- "Only D(1h)" > 0: Faster config catches transitions baseline misses
- "Both" should have highest returns (consensus transitions)
""")
    
    # Cross-pair sampling frequency summary
    print(f"""

{'='*100}
CROSS-PAIR SUMMARY: SAMPLING FREQUENCY COMPARISON
{'='*100}

TRANSITION RATES BY SAMPLING FREQUENCY
{'─'*80}
{'Pair':<10} {'24h':<12} {'12h':<12} {'6h':<12} {'1h':<12}
{'─'*80}""")
    
    for pair, results in all_results.items():
        if 'sampling_frequency' not in results:
            continue
        row = f"{pair:<10}"
        for freq in ['24h', '12h', '6h', '1h']:
            sf = results['sampling_frequency'].get(freq)
            if sf:
                row += f" {sf['transitions_per_day']:<11.2f}"
            else:
                row += f" {'N/A':<11}"
        print(row)
    
    print(f"""

24-HOUR SIGNAL RETURNS BY SAMPLING FREQUENCY
{'─'*80}
{'Pair':<10} {'24h Sample':<12} {'12h Sample':<12} {'6h Sample':<12} {'1h Sample':<12}
{'─'*80}""")
    
    for pair, results in all_results.items():
        if 'sampling_frequency' not in results:
            continue
        row = f"{pair:<10}"
        for freq in ['24h', '12h', '6h', '1h']:
            sf = results['sampling_frequency'].get(freq)
            if sf and sf['returns_by_horizon'].get(24):
                ret = sf['returns_by_horizon'][24]['signal_mean']
                row += f" {ret:>+10.2%} "
            else:
                row += f" {'N/A':<11}"
        print(row)
    
    print(f"""

24-HOUR WIN RATE BY SAMPLING FREQUENCY
{'─'*80}
{'Pair':<10} {'24h Sample':<12} {'12h Sample':<12} {'6h Sample':<12} {'1h Sample':<12}
{'─'*80}""")
    
    for pair, results in all_results.items():
        if 'sampling_frequency' not in results:
            continue
        row = f"{pair:<10}"
        for freq in ['24h', '12h', '6h', '1h']:
            sf = results['sampling_frequency'].get(freq)
            if sf and sf['returns_by_horizon'].get(24):
                win = sf['returns_by_horizon'][24]['signal_win_pct']
                row += f" {win:>10.1%} "
            else:
                row += f" {'N/A':<11}"
        print(row)
    
    print(f"""

INTERPRETATION: SAMPLING FREQUENCY
{'─'*80}
- Higher signal returns at faster sampling = faster reaction catches real moves
- Higher signal returns at slower sampling = faster sampling adds noise
- Consistent across pairs = robust finding
- Compare trans/day: more transitions can mean more signal OR more noise
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transition Quality Analysis')
    parser.add_argument('--pair', type=str, default=None)
    parser.add_argument('--all-pairs', action='store_true')
    args = parser.parse_args()
    
    print("=" * 100)
    print("TRANSITION QUALITY ANALYSIS")
    print("=" * 100)
    
    print("""
TESTING: Are baseline's extra transitions valuable or noise?

Method:
1. Identify transitions unique to baseline (not detected by faster configs)
2. Measure returns after these transitions
3. If returns are positive → baseline catches real inflection points
4. If returns are ~0 or negative → baseline detects noise/whipsaw
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
        
        df_1h = db.get_ohlcv(pair)
        results = analyze_transition_quality(df_1h, pair)
        all_results[pair] = results
        
        print("done")
        display_results(results, pair)
    
    if len(pairs) > 1:
        display_cross_pair_summary(all_results)
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    results = main()