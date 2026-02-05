#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duration Distribution Analysis
==============================
Analyze raw duration distributions for each regime state.

PURPOSE:
    Understand how long the system stays in each state before transitioning.
    This is foundational data for the U-shape hazard analysis.

OUTPUT:
    - Duration histograms per state
    - Summary statistics (mean, median, percentiles)
    - Separate analysis for bullish (long model) and bearish (short model) states

Usage:
    python duration_distribution_analysis.py

Author: CryptoBot Research
Date: January 2026
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
from collections import defaultdict
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

# Buffers (validated)
ENTRY_BUFFER = 0.025
EXIT_BUFFER = 0.005

# Confirmation
CONFIRMATION_HOURS = 3

# State classification
BULLISH_STATES = {8, 9, 10, 11, 13, 15}  # Excluding 12, 14 (exhaustion)
EXHAUSTION_STATES = {12, 14}
BEARISH_STATES = {0, 1, 2, 3, 4, 5, 6, 7}
CONFIRMED_BEARISH = {0, 1, 2, 3}  # Strong bearish
TRANSITIONAL_BEARISH = {4, 5, 6, 7}  # Weak bearish

# Duration bins (hours)
DURATION_BINS = [0, 1, 3, 6, 12, 24, 48, 72, float('inf')]
DURATION_LABELS = ['0-1h', '1-3h', '3-6h', '6-12h', '12-24h', '24-48h', '48-72h', '72h+']

# States to analyze in detail
LONG_MODEL_STATES = [15, 11, 13, 8, 12, 14]  # Bullish + exhaustion
SHORT_MODEL_STATES = [0, 1, 2, 3, 7, 4]  # Bearish states

# State descriptions
STATE_DESCRIPTIONS = {
    0: "Full Bear (all MAs bearish)",
    1: "Strong Bear",
    2: "Strong Bear",
    3: "Confirmed Bear",
    4: "Transitional",
    5: "Transitional",
    6: "Transitional",
    7: "Weak Bear (almost bullish)",
    8: "Minimum Bull",
    9: "Bull",
    10: "Bull",
    11: "Momentum Bull",
    12: "Exhaustion (unstable)",
    13: "Bull",
    14: "Exhaustion (unstable)",
    15: "Full Bull (all MAs aligned)",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DurationStats:
    """Statistics for durations in a single state."""
    state: int
    description: str
    count: int = 0
    mean: float = 0
    median: float = 0
    std: float = 0
    min_val: float = 0
    max_val: float = 0
    p25: float = 0
    p75: float = 0
    p90: float = 0
    p95: float = 0
    bin_counts: Dict[str, int] = field(default_factory=dict)
    bin_pcts: Dict[str, float] = field(default_factory=dict)


@dataclass
class PairDurationData:
    """Duration data for a single pair."""
    pair: str
    state_durations: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    total_hours: int = 0


# =============================================================================
# REGIME COMPUTATION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_trend_with_hysteresis(close: pd.Series, ma: pd.Series,
                                   entry_buf: float, exit_buf: float) -> List[int]:
    trend = []
    current = 1
    for i in range(len(close)):
        if pd.isna(ma.iloc[i]):
            trend.append(current)
            continue
        price = close.iloc[i]
        ma_val = ma.iloc[i]
        if current == 1:
            if price < ma_val * (1 - exit_buf):
                current = 0
        else:
            if price > ma_val * (1 + entry_buf):
                current = 1
        trend.append(current)
    return trend


def compute_states_hourly(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Compute 16-state regime with confirmation filter."""
    if len(df_1h) == 0:
        return pd.DataFrame()
    
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    if len(df_24h) < MA_PERIOD_24H or len(df_72h) < MA_PERIOD_72H or len(df_168h) < MA_PERIOD_168H:
        return pd.DataFrame()
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_24h_hourly = ma_24h.reindex(df_1h.index, method='ffill')
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
    trend_24h = compute_trend_with_hysteresis(df_1h['close'], ma_24h_hourly, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = compute_trend_with_hysteresis(df_1h['close'], ma_168h_hourly, ENTRY_BUFFER, EXIT_BUFFER)
    
    states = pd.DataFrame(index=df_1h.index)
    states['close'] = df_1h['close']
    states['raw_state'] = (
        pd.Series(trend_24h, index=df_1h.index) * 8 +
        pd.Series(trend_168h, index=df_1h.index) * 4 +
        (ma_72h_hourly > ma_24h_hourly).astype(int) * 2 +
        (ma_168h_hourly > ma_24h_hourly).astype(int) * 1
    )
    
    return states.dropna()


def apply_confirmation_filter(states: pd.DataFrame) -> pd.DataFrame:
    """Apply 3-hour confirmation filter."""
    states = states.copy()
    raw_states = states['raw_state'].values
    n = len(raw_states)
    
    confirmed_states = []
    current_confirmed = int(raw_states[0])
    pending_state = None
    pending_count = 0
    
    for i in range(n):
        raw = int(raw_states[i])
        
        if raw == current_confirmed:
            pending_state = None
            pending_count = 0
        elif raw == pending_state:
            pending_count += 1
            if pending_count >= CONFIRMATION_HOURS:
                current_confirmed = pending_state
                pending_state = None
                pending_count = 0
        else:
            pending_state = raw
            pending_count = 1
        
        confirmed_states.append(current_confirmed)
    
    states['confirmed_state'] = confirmed_states
    return states


def extract_durations(states: pd.DataFrame) -> Dict[int, List[float]]:
    """Extract all state durations from confirmed state series."""
    confirmed = states['confirmed_state'].values
    n = len(confirmed)
    
    durations = defaultdict(list)
    
    current_state = int(confirmed[0])
    current_duration = 1
    
    for i in range(1, n):
        state = int(confirmed[i])
        
        if state == current_state:
            current_duration += 1
        else:
            # State changed - record the completed duration
            durations[current_state].append(current_duration)
            current_state = state
            current_duration = 1
    
    # Don't forget the last segment (but mark as potentially incomplete)
    # We include it for now but could filter later
    durations[current_state].append(current_duration)
    
    return dict(durations)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_duration_stats(durations: List[float], state: int) -> DurationStats:
    """Compute statistics for a list of durations."""
    if not durations:
        return DurationStats(state=state, description=STATE_DESCRIPTIONS.get(state, "Unknown"))
    
    arr = np.array(durations)
    
    # Bin the durations
    bin_counts = {}
    bin_pcts = {}
    
    for i, label in enumerate(DURATION_LABELS):
        lower = DURATION_BINS[i]
        upper = DURATION_BINS[i + 1]
        count = np.sum((arr >= lower) & (arr < upper))
        bin_counts[label] = int(count)
        bin_pcts[label] = count / len(arr) * 100 if len(arr) > 0 else 0
    
    return DurationStats(
        state=state,
        description=STATE_DESCRIPTIONS.get(state, "Unknown"),
        count=len(durations),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        bin_counts=bin_counts,
        bin_pcts=bin_pcts,
    )


def aggregate_durations(pair_data: Dict[str, PairDurationData]) -> Dict[int, List[float]]:
    """Combine duration data across all pairs."""
    combined = defaultdict(list)
    
    for pair, data in pair_data.items():
        for state, durations in data.state_durations.items():
            combined[state].extend(durations)
    
    return dict(combined)


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_header(text: str):
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}")


def print_section(text: str):
    print(f"\n  {text}")
    print(f"  {'─' * 70}")


def display_state_distribution(stats: DurationStats):
    """Display duration distribution for a single state."""
    print(f"\n  State {stats.state}: {stats.description}")
    print(f"  Count: {stats.count:,} occurrences")
    print()
    
    print(f"  {'Duration':<12} │ {'Count':>7} │ {'Pct':>6} │ Histogram")
    print(f"  {'─' * 12}─┼─{'─' * 7}─┼─{'─' * 6}─┼─{'─' * 30}")
    
    for label in DURATION_LABELS:
        count = stats.bin_counts.get(label, 0)
        pct = stats.bin_pcts.get(label, 0)
        bar = '█' * int(pct / 2)  # Scale: 2% per block
        print(f"  {label:<12} │ {count:>7,} │ {pct:>5.1f}% │ {bar}")
    
    print()
    print(f"  Mean: {stats.mean:.1f}h │ Median: {stats.median:.1f}h │ Std: {stats.std:.1f}h")
    print(f"  P25: {stats.p25:.1f}h │ P75: {stats.p75:.1f}h │ P90: {stats.p90:.1f}h │ P95: {stats.p95:.1f}h")


def display_state_comparison(all_stats: Dict[int, DurationStats], states: List[int], title: str):
    """Display comparison table across states."""
    print_section(title)
    
    print(f"\n  {'State':<8} {'Desc':<25} {'Count':>8} {'Mean':>8} {'Median':>8} {'P90':>8}")
    print(f"  {'─' * 75}")
    
    for state in states:
        if state in all_stats:
            s = all_stats[state]
            desc = s.description[:24]
            print(f"  {state:<8} {desc:<25} {s.count:>8,} {s.mean:>7.1f}h {s.median:>7.1f}h {s.p90:>7.1f}h")


def display_per_pair_summary(pair_data: Dict[str, PairDurationData], state: int):
    """Show how a specific state behaves across pairs."""
    print_section(f"State {state} by Pair")
    
    print(f"\n  {'Pair':<12} {'Count':>8} {'Mean':>10} {'Median':>10} {'P90':>10}")
    print(f"  {'─' * 55}")
    
    for pair in DEPLOY_PAIRS:
        if pair in pair_data:
            durations = pair_data[pair].state_durations.get(state, [])
            if durations:
                arr = np.array(durations)
                print(f"  {pair:<12} {len(durations):>8} {np.mean(arr):>9.1f}h {np.median(arr):>9.1f}h {np.percentile(arr, 90):>9.1f}h")
            else:
                print(f"  {pair:<12} {'N/A':>8}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  DURATION DISTRIBUTION ANALYSIS")
    print("  Understanding state persistence patterns")
    print("=" * 80)
    
    print(f"""
    Configuration:
    ─────────────────────────────────────────────────────────
    Pairs:              {', '.join(DEPLOY_PAIRS)}
    MA Periods:         24h={MA_PERIOD_24H}, 72h={MA_PERIOD_72H}, 168h={MA_PERIOD_168H}
    Entry Buffer:       {ENTRY_BUFFER:.1%}
    Confirmation:       {CONFIRMATION_HOURS}h
    Duration Bins:      {', '.join(DURATION_LABELS)}
    ─────────────────────────────────────────────────────────
    """)
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return None
    
    db = Database()
    
    # Collect duration data from all pairs
    pair_data = {}
    
    for pair in DEPLOY_PAIRS:
        print(f"  Processing {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        if df_1h is None or len(df_1h) == 0:
            print("SKIP")
            continue
        
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h.index = pd.to_datetime(df_1h.index)
        df_1h = df_1h.sort_index()
        
        print(f"loaded {len(df_1h):,} hours...", end=" ", flush=True)
        
        states = compute_states_hourly(df_1h)
        if len(states) == 0:
            print("SKIP")
            continue
        
        states = apply_confirmation_filter(states)
        durations = extract_durations(states)
        
        pair_data[pair] = PairDurationData(
            pair=pair,
            state_durations=durations,
            total_hours=len(states)
        )
        
        total_segments = sum(len(d) for d in durations.values())
        print(f"{total_segments:,} state segments")
    
    # Aggregate across all pairs
    print("\n  Aggregating across pairs...")
    combined = aggregate_durations(pair_data)
    
    # Compute statistics for each state
    all_stats = {}
    for state in range(16):
        durations = combined.get(state, [])
        all_stats[state] = compute_duration_stats(durations, state)
    
    # =============================================================================
    # DISPLAY RESULTS
    # =============================================================================
    
    print_header("BULLISH STATES (Long Model)")
    
    display_state_comparison(all_stats, LONG_MODEL_STATES, "Summary Comparison")
    
    for state in LONG_MODEL_STATES:
        if state in all_stats and all_stats[state].count > 0:
            display_state_distribution(all_stats[state])
    
    print_header("BEARISH STATES (Short Model)")
    
    display_state_comparison(all_stats, SHORT_MODEL_STATES, "Summary Comparison")
    
    for state in SHORT_MODEL_STATES:
        if state in all_stats and all_stats[state].count > 0:
            display_state_distribution(all_stats[state])
    
    print_header("PER-PAIR ANALYSIS")
    
    # Show key states by pair
    display_per_pair_summary(pair_data, 15)  # Full bull
    display_per_pair_summary(pair_data, 0)   # Full bear
    display_per_pair_summary(pair_data, 11)  # Momentum bull
    
    print_header("KEY FINDINGS")
    
    # Identify most/least stable states
    stable_states = sorted(all_stats.items(), key=lambda x: x[1].median if x[1].count > 100 else 0, reverse=True)
    
    print("\n  Most Stable States (longest median duration):")
    for state, stats in stable_states[:5]:
        if stats.count > 100:
            print(f"    State {state:>2}: {stats.median:>6.1f}h median ({stats.description})")
    
    print("\n  Least Stable States (shortest median duration):")
    for state, stats in sorted(stable_states, key=lambda x: x[1].median if x[1].count > 100 else 999)[:5]:
        if stats.count > 100:
            print(f"    State {state:>2}: {stats.median:>6.1f}h median ({stats.description})")
    
    print(f"\n{'=' * 80}")
    print("  DURATION DISTRIBUTION ANALYSIS COMPLETE")
    print("=" * 80)
    
    return all_stats, pair_data


if __name__ == "__main__":
    results = main()