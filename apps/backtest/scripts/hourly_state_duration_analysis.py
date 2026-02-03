#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hourly State Duration Analysis
==============================
Compute state duration statistics at multiple frequencies (1h, 3h, daily)
and with different duration start points (raw vs confirmed).

This generates the parameters needed for position sizing models:
- Exit rates at hourly/3h/daily frequency
- P(→bearish | state) transition probabilities
- Duration distributions per state

Output feeds into position_sizing_tests.py

Usage:
    python hourly_state_duration_analysis.py
    python hourly_state_duration_analysis.py --pair ETHUSD
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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked from validation)
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

# Entry/exit buffers (validated E2.5%)
ENTRY_BUFFER = 0.025
EXIT_BUFFER = 0.005

# Confirmation hours
CONFIRMATION_HOURS = 3

# State classification
BULLISH_STATES = set([8, 9, 10, 11, 12, 13, 14, 15])
BEARISH_STATES = set([0, 1, 2, 3, 4, 5, 6, 7])

# Minimum samples for reliable statistics
MIN_SAMPLES = 20


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StateVisit:
    """Record of a single visit to a state."""
    state: int
    start_idx: int
    end_idx: int
    duration_hours: int
    duration_3h_blocks: int
    duration_days: float
    exit_state: Optional[int]
    is_to_bearish: bool
    
    # For confirmed tracking
    confirmed_start_idx: Optional[int] = None
    confirmed_duration_hours: Optional[int] = None


@dataclass
class StateStats:
    """Statistics for a single state at a given frequency."""
    state_id: int
    frequency: str  # '1h', '3h', 'daily'
    duration_start: str  # 'raw' or 'confirmed'
    
    # Duration stats
    n_visits: int = 0
    total_periods: int = 0  # Total periods (hours/3h-blocks/days) in this state
    durations: List[int] = field(default_factory=list)
    
    # Transition stats
    n_exits: int = 0
    n_to_bearish: int = 0
    exit_destinations: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def avg_duration(self) -> float:
        return np.mean(self.durations) if self.durations else 0
    
    @property
    def median_duration(self) -> float:
        return np.median(self.durations) if self.durations else 0
    
    @property
    def std_duration(self) -> float:
        return np.std(self.durations) if len(self.durations) > 1 else 0
    
    @property
    def exit_rate(self) -> float:
        """Exit rate per period (hour/3h-block/day)."""
        if self.total_periods == 0:
            return 0.5
        return self.n_exits / self.total_periods
    
    @property
    def p_to_bearish(self) -> float:
        """P(→bearish | exit from this state)."""
        if self.n_exits == 0:
            return 0.5
        return self.n_to_bearish / self.n_exits


# =============================================================================
# REGIME COMPUTATION (HOURLY)
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_trend_with_hysteresis(close: pd.Series, ma: pd.Series,
                                   entry_buf: float, exit_buf: float) -> List[int]:
    """Compute trend signal with hysteresis buffers."""
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


def compute_states_hourly(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Compute 16-state regime at hourly frequency."""
    # Resample to higher timeframes for MA calculation
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    # Compute MAs at their native timeframes
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Forward-fill to hourly frequency
    ma_24h_hourly = ma_24h.reindex(df_1h.index, method='ffill')
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
    # Compute trend signals with hysteresis
    trend_24h = compute_trend_with_hysteresis(df_1h['close'], ma_24h_hourly, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = compute_trend_with_hysteresis(df_1h['close'], ma_168h_hourly, ENTRY_BUFFER, EXIT_BUFFER)
    
    # Build state DataFrame
    states = pd.DataFrame(index=df_1h.index)
    states['close'] = df_1h['close']
    states['ma_24h'] = ma_24h_hourly
    states['ma_72h'] = ma_72h_hourly
    states['ma_168h'] = ma_168h_hourly
    
    # Compute 16-state
    states['raw_state'] = (
        pd.Series(trend_24h, index=df_1h.index) * 8 +
        pd.Series(trend_168h, index=df_1h.index) * 4 +
        (ma_72h_hourly > ma_24h_hourly).astype(int) * 2 +
        (ma_168h_hourly > ma_24h_hourly).astype(int) * 1
    )
    
    return states.dropna()


def apply_confirmation_filter(states: pd.DataFrame) -> pd.DataFrame:
    """Apply 3h confirmation filter to raw states."""
    states = states.copy()
    
    raw_states = states['raw_state'].values
    n = len(raw_states)
    
    confirmed_states = []
    confirmation_start_hours = []  # Track when confirmation period started
    
    current_confirmed = int(raw_states[0])
    pending_state = None
    pending_count = 0
    pending_start_idx = None
    
    for i in range(n):
        raw = int(raw_states[i])
        
        if raw == current_confirmed:
            # State matches confirmed - reset pending
            pending_state = None
            pending_count = 0
            pending_start_idx = None
            confirmed_states.append(current_confirmed)
            confirmation_start_hours.append(i)  # Current state started before this
        elif raw == pending_state:
            # Continue counting pending state
            pending_count += 1
            if pending_count >= CONFIRMATION_HOURS:
                # Confirmation complete
                current_confirmed = pending_state
                confirmed_states.append(current_confirmed)
                confirmation_start_hours.append(pending_start_idx)
                pending_state = None
                pending_count = 0
                pending_start_idx = None
            else:
                confirmed_states.append(current_confirmed)
                confirmation_start_hours.append(i)
        else:
            # New pending state
            pending_state = raw
            pending_count = 1
            pending_start_idx = i
            confirmed_states.append(current_confirmed)
            confirmation_start_hours.append(i)
    
    states['confirmed_state'] = confirmed_states
    states['confirmation_start_hour'] = confirmation_start_hours
    
    return states


# =============================================================================
# DURATION EXTRACTION
# =============================================================================

def extract_visits_raw(states: pd.DataFrame) -> List[StateVisit]:
    """Extract state visits using RAW state changes (before confirmation)."""
    raw_states = states['raw_state'].values
    n = len(raw_states)
    visits = []
    
    current_state = int(raw_states[0])
    start_idx = 0
    
    for i in range(1, n):
        if int(raw_states[i]) != current_state:
            # State ended
            exit_state = int(raw_states[i])
            duration_hours = i - start_idx
            
            visits.append(StateVisit(
                state=current_state,
                start_idx=start_idx,
                end_idx=i,
                duration_hours=duration_hours,
                duration_3h_blocks=duration_hours // 3,
                duration_days=duration_hours / 24,
                exit_state=exit_state,
                is_to_bearish=(current_state in BULLISH_STATES and exit_state in BEARISH_STATES)
            ))
            
            current_state = exit_state
            start_idx = i
    
    return visits


def extract_visits_confirmed(states: pd.DataFrame) -> List[StateVisit]:
    """Extract state visits using CONFIRMED state changes."""
    confirmed_states = states['confirmed_state'].values
    n = len(confirmed_states)
    visits = []
    
    current_state = int(confirmed_states[0])
    start_idx = 0
    
    for i in range(1, n):
        if int(confirmed_states[i]) != current_state:
            # Confirmed state changed
            exit_state = int(confirmed_states[i])
            duration_hours = i - start_idx
            
            visits.append(StateVisit(
                state=current_state,
                start_idx=start_idx,
                end_idx=i,
                duration_hours=duration_hours,
                duration_3h_blocks=duration_hours // 3,
                duration_days=duration_hours / 24,
                exit_state=exit_state,
                is_to_bearish=(current_state in BULLISH_STATES and exit_state in BEARISH_STATES),
                confirmed_start_idx=start_idx,
                confirmed_duration_hours=duration_hours
            ))
            
            current_state = exit_state
            start_idx = i
    
    return visits


# =============================================================================
# STATISTICS COMPUTATION
# =============================================================================

def compute_stats_hourly(visits: List[StateVisit], states: pd.DataFrame, 
                         duration_start: str) -> Dict[int, StateStats]:
    """Compute per-state statistics at hourly frequency."""
    stats = {i: StateStats(state_id=i, frequency='1h', duration_start=duration_start) 
             for i in range(16)}
    
    # Count total hours in each state
    if duration_start == 'raw':
        state_col = 'raw_state'
    else:
        state_col = 'confirmed_state'
    
    state_counts = states[state_col].value_counts()
    for state_id, count in state_counts.items():
        stats[int(state_id)].total_periods = count
    
    # Process visits
    for visit in visits:
        s = stats[visit.state]
        s.n_visits += 1
        s.durations.append(visit.duration_hours)
        
        if visit.exit_state is not None:
            s.n_exits += 1
            s.exit_destinations[visit.exit_state] += 1
            if visit.is_to_bearish:
                s.n_to_bearish += 1
    
    return stats


def compute_stats_3h(visits: List[StateVisit], states: pd.DataFrame,
                     duration_start: str) -> Dict[int, StateStats]:
    """Compute per-state statistics at 3-hour frequency."""
    stats = {i: StateStats(state_id=i, frequency='3h', duration_start=duration_start) 
             for i in range(16)}
    
    # Resample states to 3h blocks
    if duration_start == 'raw':
        state_col = 'raw_state'
    else:
        state_col = 'confirmed_state'
    
    # Count 3h blocks in each state (take state at start of each 3h block)
    states_3h = states[state_col].resample('3h').first().dropna()
    state_counts = states_3h.value_counts()
    for state_id, count in state_counts.items():
        stats[int(state_id)].total_periods = count
    
    # Process visits (convert durations to 3h blocks)
    for visit in visits:
        s = stats[visit.state]
        s.n_visits += 1
        s.durations.append(visit.duration_3h_blocks)
        
        if visit.exit_state is not None:
            s.n_exits += 1
            s.exit_destinations[visit.exit_state] += 1
            if visit.is_to_bearish:
                s.n_to_bearish += 1
    
    return stats


def compute_stats_daily(visits: List[StateVisit], states: pd.DataFrame,
                        duration_start: str) -> Dict[int, StateStats]:
    """Compute per-state statistics at daily frequency."""
    stats = {i: StateStats(state_id=i, frequency='daily', duration_start=duration_start) 
             for i in range(16)}
    
    # Resample states to daily (take state at start of each day)
    if duration_start == 'raw':
        state_col = 'raw_state'
    else:
        state_col = 'confirmed_state'
    
    states_daily = states[state_col].resample('24h').first().dropna()
    state_counts = states_daily.value_counts()
    for state_id, count in state_counts.items():
        stats[int(state_id)].total_periods = count
    
    # Process visits (convert durations to days)
    for visit in visits:
        s = stats[visit.state]
        s.n_visits += 1
        s.durations.append(visit.duration_days)
        
        if visit.exit_state is not None:
            s.n_exits += 1
            s.exit_destinations[visit.exit_state] += 1
            if visit.is_to_bearish:
                s.n_to_bearish += 1
    
    return stats


# =============================================================================
# AGGREGATION & DISPLAY
# =============================================================================

def aggregate_stats(all_pair_stats: Dict[str, Dict[str, Dict[int, StateStats]]]) -> pd.DataFrame:
    """Aggregate statistics across all pairs into a summary DataFrame."""
    rows = []
    
    for pair, freq_stats in all_pair_stats.items():
        for key, state_stats in freq_stats.items():
            freq, duration_start = key.split('_')
            
            for state_id, stats in state_stats.items():
                if stats.n_visits < MIN_SAMPLES:
                    continue
                    
                rows.append({
                    'pair': pair,
                    'state': state_id,
                    'is_bullish': state_id in BULLISH_STATES,
                    'frequency': freq,
                    'duration_start': duration_start,
                    'n_visits': stats.n_visits,
                    'total_periods': stats.total_periods,
                    'avg_duration': stats.avg_duration,
                    'median_duration': stats.median_duration,
                    'std_duration': stats.std_duration,
                    'exit_rate': stats.exit_rate,
                    'p_to_bearish': stats.p_to_bearish if state_id in BULLISH_STATES else None,
                })
    
    return pd.DataFrame(rows)


def compute_cross_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-pair averages for each state/frequency/duration_start combo."""
    
    # Only bullish states for main analysis
    df_bull = df[df['is_bullish']].copy()
    
    summary = df_bull.groupby(['state', 'frequency', 'duration_start']).agg({
        'n_visits': 'sum',
        'total_periods': 'sum',
        'avg_duration': 'mean',
        'exit_rate': 'mean',
        'p_to_bearish': 'mean',
    }).reset_index()
    
    # Recalculate exit rate from totals
    summary['exit_rate_pooled'] = summary.apply(
        lambda r: df_bull[(df_bull['state'] == r['state']) & 
                          (df_bull['frequency'] == r['frequency']) &
                          (df_bull['duration_start'] == r['duration_start'])]['exit_rate'].mean(),
        axis=1
    )
    
    return summary


def display_results(df: pd.DataFrame, summary: pd.DataFrame):
    """Display formatted results."""
    
    print("\n" + "=" * 120)
    print("CROSS-PAIR SUMMARY: BULLISH STATES")
    print("=" * 120)
    
    # For each frequency, show comparison of raw vs confirmed
    for freq in ['1h', '3h', 'daily']:
        print(f"\n{'─' * 120}")
        print(f"FREQUENCY: {freq.upper()}")
        print(f"{'─' * 120}")
        
        freq_data = summary[summary['frequency'] == freq].copy()
        
        print(f"\n  {'State':<8} │ {'Duration Start':<12} │ {'N Visits':>10} │ {'Avg Duration':>14} │ "
              f"{'Exit Rate':>12} │ {'P(→bearish)':>12}")
        print(f"  {'─' * 8}─┼─{'─' * 12}─┼─{'─' * 10}─┼─{'─' * 14}─┼─{'─' * 12}─┼─{'─' * 12}")
        
        for state in sorted(BULLISH_STATES):
            state_data = freq_data[freq_data['state'] == state]
            
            for _, row in state_data.iterrows():
                duration_unit = 'h' if freq == '1h' else ('3h' if freq == '3h' else 'd')
                print(f"  {state:<8} │ {row['duration_start']:<12} │ {row['n_visits']:>10,.0f} │ "
                      f"{row['avg_duration']:>11.1f} {duration_unit:>2} │ "
                      f"{row['exit_rate']:>11.2%} │ {row['p_to_bearish']:>11.1%}")
    
    # Summary comparison table
    print(f"\n\n{'=' * 120}")
    print("EXIT RATE COMPARISON: RAW vs CONFIRMED")
    print("=" * 120)
    
    print(f"\n  {'State':<8} │ {'1h Raw':>10} │ {'1h Conf':>10} │ {'3h Raw':>10} │ "
          f"{'3h Conf':>10} │ {'Daily Raw':>10} │ {'Daily Conf':>10}")
    print(f"  {'─' * 8}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}")
    
    for state in sorted(BULLISH_STATES):
        row_data = {'state': state}
        
        for freq in ['1h', '3h', 'daily']:
            for start in ['raw', 'confirmed']:
                key = f"{freq}_{start}"
                mask = (summary['state'] == state) & (summary['frequency'] == freq) & (summary['duration_start'] == start)
                if mask.any():
                    row_data[key] = summary.loc[mask, 'exit_rate'].values[0]
                else:
                    row_data[key] = None
        
        vals = []
        for key in ['1h_raw', '1h_confirmed', '3h_raw', '3h_confirmed', 'daily_raw', 'daily_confirmed']:
            if row_data.get(key) is not None:
                vals.append(f"{row_data[key]:>9.2%}")
            else:
                vals.append(f"{'N/A':>10}")
        
        print(f"  {state:<8} │ {vals[0]} │ {vals[1]} │ {vals[2]} │ {vals[3]} │ {vals[4]} │ {vals[5]}")
    
    # P(→bearish) summary
    print(f"\n\n{'=' * 120}")
    print("P(→BEARISH) BY STATE (Cross-pair average)")
    print("=" * 120)
    
    print(f"\n  {'State':<8} │ {'1h Raw':>10} │ {'1h Conf':>10} │ {'3h Raw':>10} │ "
          f"{'3h Conf':>10} │ {'Daily Raw':>10} │ {'Daily Conf':>10}")
    print(f"  {'─' * 8}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}")
    
    for state in sorted(BULLISH_STATES):
        row_data = {'state': state}
        
        for freq in ['1h', '3h', 'daily']:
            for start in ['raw', 'confirmed']:
                key = f"{freq}_{start}"
                mask = (summary['state'] == state) & (summary['frequency'] == freq) & (summary['duration_start'] == start)
                if mask.any():
                    row_data[key] = summary.loc[mask, 'p_to_bearish'].values[0]
                else:
                    row_data[key] = None
        
        vals = []
        for key in ['1h_raw', '1h_confirmed', '3h_raw', '3h_confirmed', 'daily_raw', 'daily_confirmed']:
            if row_data.get(key) is not None:
                vals.append(f"{row_data[key]:>9.1%}")
            else:
                vals.append(f"{'N/A':>10}")
        
        print(f"  {state:<8} │ {vals[0]} │ {vals[1]} │ {vals[2]} │ {vals[3]} │ {vals[4]} │ {vals[5]}")


def export_parameters(summary: pd.DataFrame, output_path: str):
    """Export parameters for use in position_sizing_tests.py"""
    
    params = {
        'STATE_EXIT_RATES': {},
        'STATE_P_TO_BEARISH': {},
    }
    
    for freq in ['1h', '3h', 'daily']:
        for start in ['raw', 'confirmed']:
            key = f"{freq}_{start}"
            params['STATE_EXIT_RATES'][key] = {}
            params['STATE_P_TO_BEARISH'][key] = {}
            
            for state in range(16):
                mask = (summary['state'] == state) & (summary['frequency'] == freq) & (summary['duration_start'] == start)
                if mask.any():
                    params['STATE_EXIT_RATES'][key][state] = float(summary.loc[mask, 'exit_rate'].values[0])
                    if state in BULLISH_STATES:
                        params['STATE_P_TO_BEARISH'][key][state] = float(summary.loc[mask, 'p_to_bearish'].values[0])
    
    # Write as Python module
    with open(output_path, 'w') as f:
        f.write('"""Auto-generated state parameters from hourly_state_duration_analysis.py"""\n\n')
        f.write('# Exit rates per period (hour/3h-block/day)\n')
        f.write('STATE_EXIT_RATES = {\n')
        for key, rates in params['STATE_EXIT_RATES'].items():
            f.write(f"    '{key}': {{\n")
            for state, rate in sorted(rates.items()):
                f.write(f"        {state}: {rate:.4f},\n")
            f.write(f"    }},\n")
        f.write('}\n\n')
        
        f.write('# P(→bearish | exit) for bullish states\n')
        f.write('STATE_P_TO_BEARISH = {\n')
        for key, probs in params['STATE_P_TO_BEARISH'].items():
            f.write(f"    '{key}': {{\n")
            for state, prob in sorted(probs.items()):
                f.write(f"        {state}: {prob:.4f},\n")
            f.write(f"    }},\n")
        f.write('}\n')
    
    print(f"\nParameters exported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hourly State Duration Analysis')
    parser.add_argument('--pair', type=str, default=None, help='Analyze specific pair')
    parser.add_argument('--output', type=str, default='state_parameters.py', 
                        help='Output file for parameters')
    args = parser.parse_args()
    
    print("=" * 120)
    print("HOURLY STATE DURATION ANALYSIS")
    print("=" * 120)
    print(f"""
    Analyzing state durations at multiple frequencies:
    - 1h (hourly): Exit rate per hour
    - 3h (3-hour blocks): Exit rate per 3h block  
    - daily: Exit rate per day
    
    Duration start variants:
    - raw: Duration from first raw state change
    - confirmed: Duration from 3h confirmation complete
    
    Parameters: E{ENTRY_BUFFER:.1%} entry buffer, {CONFIRMATION_HOURS}h confirmation
    """)
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    
    pairs = [args.pair] if args.pair else DEPLOY_PAIRS
    
    all_pair_stats = {}
    
    for pair in pairs:
        print(f"\nProcessing {pair}...", end=" ", flush=True)
        
        # Load hourly data
        df_1h = db.get_ohlcv(pair)
        if df_1h is None or len(df_1h) == 0:
            print(f"WARNING: No data for {pair}")
            continue
        
        print(f"loaded {len(df_1h):,} hours...", end=" ", flush=True)
        
        # Compute states
        states = compute_states_hourly(df_1h)
        states = apply_confirmation_filter(states)
        print(f"computed states...", end=" ", flush=True)
        
        # Extract visits (both raw and confirmed)
        visits_raw = extract_visits_raw(states)
        visits_confirmed = extract_visits_confirmed(states)
        
        print(f"{len(visits_raw)} raw visits, {len(visits_confirmed)} confirmed visits...", end=" ", flush=True)
        
        # Compute stats at all frequencies
        pair_stats = {}
        
        # Raw duration start
        pair_stats['1h_raw'] = compute_stats_hourly(visits_raw, states, 'raw')
        pair_stats['3h_raw'] = compute_stats_3h(visits_raw, states, 'raw')
        pair_stats['daily_raw'] = compute_stats_daily(visits_raw, states, 'raw')
        
        # Confirmed duration start
        pair_stats['1h_confirmed'] = compute_stats_hourly(visits_confirmed, states, 'confirmed')
        pair_stats['3h_confirmed'] = compute_stats_3h(visits_confirmed, states, 'confirmed')
        pair_stats['daily_confirmed'] = compute_stats_daily(visits_confirmed, states, 'confirmed')
        
        all_pair_stats[pair] = pair_stats
        print("done")
    
    # Aggregate into DataFrame
    print("\nAggregating results...")
    df = aggregate_stats(all_pair_stats)
    
    # Compute cross-pair summary
    summary = compute_cross_pair_summary(df)
    
    # Display results
    display_results(df, summary)
    
    # Export parameters
    export_parameters(summary, args.output)
    
    # Save full results to CSV
    csv_path = 'hourly_duration_analysis_full.csv'
    df.to_csv(csv_path, index=False)
    print(f"Full results saved to: {csv_path}")
    
    print(f"\n{'=' * 120}")
    print("ANALYSIS COMPLETE")
    print("=" * 120)
    
    return df, summary


if __name__ == "__main__":
    df, summary = main()