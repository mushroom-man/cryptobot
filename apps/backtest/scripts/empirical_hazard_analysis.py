#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical Hazard Analysis
=========================
Analyze duration-dependent exit rates and transition probabilities.

Key questions:
1. Is exit probability constant, increasing, or decreasing with duration?
2. Does P(→bearish) depend on how long we've been in the state?
3. What is P(bearish flip in Δt | state, duration)?

Horizons tested: 3h, 6h, 24h

Usage:
    python empirical_hazard_analysis.py
    python empirical_hazard_analysis.py --pair ETHUSD
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
from scipy import stats as scipy_stats
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

# Entry/exit buffers (validated)
ENTRY_BUFFER = 0.025
EXIT_BUFFER = 0.005

# Confirmation hours
CONFIRMATION_HOURS = 3

# State classification
BULLISH_STATES = set([8, 9, 10, 11, 12, 13, 14, 15])
BEARISH_STATES = set([0, 1, 2, 3, 4, 5, 6, 7])

# Duration buckets for hazard analysis (hours)
DURATION_BUCKETS = [
    (0, 6),
    (6, 12),
    (12, 24),
    (24, 48),
    (48, 72),
    (72, 120),
    (120, 999),
]

# Prediction horizons (hours)
HORIZONS = [3, 6, 24]

# Minimum samples for reliable statistics
MIN_SAMPLES = 20


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HazardBucket:
    """Statistics for a duration bucket within a state."""
    state: int
    bucket_start: int
    bucket_end: int
    
    # Observation counts
    n_observations: int = 0  # Hours spent in this bucket
    n_exits: int = 0  # Exits from this bucket
    
    # Transition counts (when exiting from this bucket)
    n_to_bearish: int = 0
    n_to_bullish: int = 0
    exit_destinations: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def exit_rate_per_hour(self) -> float:
        """Hazard rate: P(exit in this hour | reached this bucket)."""
        if self.n_observations == 0:
            return 0.0
        return self.n_exits / self.n_observations
    
    @property
    def p_to_bearish(self) -> float:
        """P(→bearish | exit from this bucket)."""
        if self.n_exits == 0:
            return 0.5
        return self.n_to_bearish / self.n_exits
    
    @property
    def p_bearish_flip_per_hour(self) -> float:
        """P(flip to bearish in this hour | in this bucket)."""
        return self.exit_rate_per_hour * self.p_to_bearish


@dataclass
class DurationDistribution:
    """Full duration distribution for a state."""
    state: int
    durations: List[int] = field(default_factory=list)
    
    @property
    def n_visits(self) -> int:
        return len(self.durations)
    
    @property
    def mean(self) -> float:
        return np.mean(self.durations) if self.durations else 0
    
    @property
    def std(self) -> float:
        return np.std(self.durations) if len(self.durations) > 1 else 0
    
    @property
    def median(self) -> float:
        return np.median(self.durations) if self.durations else 0
    
    def percentile(self, p: float) -> float:
        return np.percentile(self.durations, p) if self.durations else 0
    
    def survival_prob(self, hours: int) -> float:
        """P(duration > hours) - empirical survival function."""
        if not self.durations:
            return 0.5
        return np.mean([d > hours for d in self.durations])


# =============================================================================
# REGIME COMPUTATION
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
    """Compute 16-state regime at hourly frequency with confirmed states."""
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
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
    
    # Apply confirmation filter
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
            confirmed_states.append(current_confirmed)
        elif raw == pending_state:
            pending_count += 1
            if pending_count >= CONFIRMATION_HOURS:
                current_confirmed = pending_state
                pending_state = None
                pending_count = 0
            confirmed_states.append(current_confirmed)
        else:
            pending_state = raw
            pending_count = 1
            confirmed_states.append(current_confirmed)
    
    states['confirmed_state'] = confirmed_states
    
    # Track duration in confirmed state
    durations = []
    current_duration = 1
    for i in range(n):
        if i > 0 and confirmed_states[i] == confirmed_states[i-1]:
            current_duration += 1
        else:
            current_duration = 1
        durations.append(current_duration)
    
    states['duration_hours'] = durations
    
    return states.dropna()


# =============================================================================
# HAZARD ANALYSIS
# =============================================================================

def get_bucket_idx(duration: int) -> int:
    """Get bucket index for a duration."""
    for i, (start, end) in enumerate(DURATION_BUCKETS):
        if start <= duration < end:
            return i
    return len(DURATION_BUCKETS) - 1


def analyze_hazards(states: pd.DataFrame) -> Tuple[Dict[int, List[HazardBucket]], Dict[int, DurationDistribution]]:
    """
    Analyze duration-dependent hazard rates for each state.
    
    Returns:
        hazard_buckets: {state: [HazardBucket, ...]} - hazard by duration bucket
        distributions: {state: DurationDistribution} - full duration distribution
    """
    confirmed = states['confirmed_state'].values
    durations = states['duration_hours'].values
    n = len(confirmed)
    
    # Initialize structures
    hazard_buckets = {}
    distributions = {}
    
    for state in range(16):
        hazard_buckets[state] = [
            HazardBucket(state=state, bucket_start=b[0], bucket_end=b[1])
            for b in DURATION_BUCKETS
        ]
        distributions[state] = DurationDistribution(state=state)
    
    # Track current visit
    current_state = int(confirmed[0])
    visit_start_idx = 0
    
    for i in range(1, n):
        state_now = int(confirmed[i])
        state_prev = int(confirmed[i-1])
        duration_now = int(durations[i])
        duration_prev = int(durations[i-1])
        
        # Record observation (we spent this hour in this bucket)
        bucket_idx = get_bucket_idx(duration_prev)
        hazard_buckets[state_prev][bucket_idx].n_observations += 1
        
        # Check for transition
        if state_now != state_prev:
            # Exit occurred
            bucket_idx = get_bucket_idx(duration_prev)
            bucket = hazard_buckets[state_prev][bucket_idx]
            bucket.n_exits += 1
            bucket.exit_destinations[state_now] += 1
            
            if state_prev in BULLISH_STATES and state_now in BEARISH_STATES:
                bucket.n_to_bearish += 1
            elif state_prev in BULLISH_STATES and state_now in BULLISH_STATES:
                bucket.n_to_bullish += 1
            
            # Record visit duration
            visit_duration = duration_prev
            distributions[state_prev].durations.append(visit_duration)
            
            # Reset for new state
            current_state = state_now
            visit_start_idx = i
    
    return hazard_buckets, distributions


def compute_horizon_probabilities(hazard_buckets: Dict[int, List[HazardBucket]], 
                                   horizons: List[int]) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Compute P(exit in horizon) and P(bearish flip in horizon) for each state/duration bucket.
    
    Returns:
        {state: {bucket_idx: {'exit_3h': p, 'exit_6h': p, 'bearish_3h': p, ...}}}
    """
    results = {}
    
    for state in range(16):
        results[state] = {}
        buckets = hazard_buckets[state]
        
        for bucket_idx, bucket in enumerate(buckets):
            bucket_results = {}
            
            # Get hourly exit rate for this bucket
            h = bucket.exit_rate_per_hour
            p_to_bearish = bucket.p_to_bearish
            
            for horizon in horizons:
                # P(exit within horizon) = 1 - P(survive horizon)
                # P(survive) = (1 - hourly_rate)^hours
                if h > 0:
                    p_exit = 1 - (1 - h) ** horizon
                else:
                    p_exit = 0
                
                # P(bearish flip) = P(exit) × P(bearish | exit)
                p_bearish = p_exit * p_to_bearish
                
                bucket_results[f'exit_{horizon}h'] = p_exit
                bucket_results[f'bearish_{horizon}h'] = p_bearish
            
            bucket_results['hourly_hazard'] = h
            bucket_results['p_to_bearish'] = p_to_bearish
            bucket_results['n_observations'] = bucket.n_observations
            bucket_results['n_exits'] = bucket.n_exits
            
            results[state][bucket_idx] = bucket_results
    
    return results


def fit_weibull(durations: List[int]) -> Tuple[float, float, float]:
    """
    Fit Weibull distribution to duration data.
    
    Returns:
        shape (β): >1 = increasing hazard, <1 = decreasing, =1 = constant
        scale (η): characteristic duration
        p_value: goodness of fit
    """
    if len(durations) < 10:
        return 1.0, np.mean(durations) if durations else 1.0, 0.0
    
    try:
        # Fit Weibull (scipy uses different parameterization)
        shape, loc, scale = scipy_stats.weibull_min.fit(durations, floc=0)
        
        # KS test for goodness of fit
        ks_stat, p_value = scipy_stats.kstest(durations, 'weibull_min', args=(shape, loc, scale))
        
        return shape, scale, p_value
    except:
        return 1.0, np.mean(durations) if durations else 1.0, 0.0


# =============================================================================
# DISPLAY
# =============================================================================

def display_hazard_analysis(hazard_buckets: Dict[int, List[HazardBucket]], 
                            distributions: Dict[int, DurationDistribution],
                            horizon_probs: Dict[int, Dict[int, Dict[str, float]]],
                            pair: str):
    """Display comprehensive hazard analysis."""
    
    print(f"\n{'=' * 120}")
    print(f"EMPIRICAL HAZARD ANALYSIS: {pair}")
    print(f"{'=' * 120}")
    
    # Duration Distribution Summary
    print(f"\n{'─' * 120}")
    print("DURATION DISTRIBUTIONS (Bullish States)")
    print(f"{'─' * 120}")
    
    print(f"\n  {'State':<8} {'N Visits':>10} {'Mean':>10} {'Std':>10} {'Median':>10} "
          f"{'P10':>8} {'P25':>8} {'P75':>8} {'P90':>8}")
    print(f"  {'─' * 100}")
    
    for state in sorted(BULLISH_STATES):
        dist = distributions[state]
        if dist.n_visits >= MIN_SAMPLES:
            print(f"  {state:<8} {dist.n_visits:>10} {dist.mean:>9.1f}h {dist.std:>9.1f}h "
                  f"{dist.median:>9.1f}h {dist.percentile(10):>7.0f}h {dist.percentile(25):>7.0f}h "
                  f"{dist.percentile(75):>7.0f}h {dist.percentile(90):>7.0f}h")
    
    # Weibull Fits
    print(f"\n{'─' * 120}")
    print("WEIBULL DISTRIBUTION FITS")
    print(f"{'─' * 120}")
    print(f"\n  β > 1: Increasing hazard (exhaustion effect)")
    print(f"  β = 1: Constant hazard (memoryless)")
    print(f"  β < 1: Decreasing hazard (momentum effect)")
    
    print(f"\n  {'State':<8} {'β (shape)':>12} {'η (scale)':>12} {'KS p-value':>12} {'Interpretation':<25}")
    print(f"  {'─' * 75}")
    
    for state in sorted(BULLISH_STATES):
        dist = distributions[state]
        if dist.n_visits >= MIN_SAMPLES:
            beta, eta, p_val = fit_weibull(dist.durations)
            
            if beta > 1.1:
                interp = "Increasing (exhaustion)"
            elif beta < 0.9:
                interp = "Decreasing (momentum)"
            else:
                interp = "~Constant (memoryless)"
            
            print(f"  {state:<8} {beta:>12.3f} {eta:>11.1f}h {p_val:>12.3f} {interp:<25}")
    
    # Hazard by Duration Bucket
    print(f"\n{'─' * 120}")
    print("HAZARD RATES BY DURATION BUCKET (Bullish States)")
    print(f"{'─' * 120}")
    
    for state in sorted(BULLISH_STATES):
        dist = distributions[state]
        if dist.n_visits < MIN_SAMPLES:
            continue
        
        print(f"\n  State {state}:")
        print(f"    {'Duration':<15} {'N Obs':>10} {'N Exits':>10} {'Hazard/h':>12} "
              f"{'P(→bear)':>10} {'P(bear)/h':>12}")
        print(f"    {'─' * 75}")
        
        buckets = hazard_buckets[state]
        for bucket in buckets:
            if bucket.n_observations >= MIN_SAMPLES:
                print(f"    {bucket.bucket_start:>3}-{bucket.bucket_end:<6}h   "
                      f"{bucket.n_observations:>10} {bucket.n_exits:>10} "
                      f"{bucket.exit_rate_per_hour:>11.2%} {bucket.p_to_bearish:>10.1%} "
                      f"{bucket.p_bearish_flip_per_hour:>11.3%}")
    
    # Horizon Probabilities
    print(f"\n{'─' * 120}")
    print("P(BEARISH FLIP) BY STATE, DURATION, AND HORIZON")
    print(f"{'─' * 120}")
    
    print(f"\n  {'State':<8} {'Duration':<15} {'P(flip 3h)':>12} {'P(flip 6h)':>12} {'P(flip 24h)':>12}")
    print(f"  {'─' * 65}")
    
    for state in sorted(BULLISH_STATES):
        dist = distributions[state]
        if dist.n_visits < MIN_SAMPLES:
            continue
        
        for bucket_idx, bucket in enumerate(hazard_buckets[state]):
            if bucket.n_observations >= MIN_SAMPLES:
                probs = horizon_probs[state][bucket_idx]
                bucket_label = f"{bucket.bucket_start}-{bucket.bucket_end}h"
                print(f"  {state:<8} {bucket_label:<15} {probs['bearish_3h']:>11.2%} "
                      f"{probs['bearish_6h']:>11.2%} {probs['bearish_24h']:>11.2%}")
        print()


def display_hazard_trend(hazard_buckets: Dict[int, List[HazardBucket]]):
    """Display whether hazard is increasing or decreasing with duration."""
    
    print(f"\n{'=' * 120}")
    print("HAZARD TREND ANALYSIS: Is Exit Rate Duration-Dependent?")
    print(f"{'=' * 120}")
    
    print(f"\n  {'State':<8} {'Early (0-12h)':>15} {'Mid (12-48h)':>15} {'Late (48h+)':>15} {'Trend':<20}")
    print(f"  {'─' * 80}")
    
    for state in sorted(BULLISH_STATES):
        buckets = hazard_buckets[state]
        
        # Aggregate into early/mid/late
        early_obs, early_exits = 0, 0
        mid_obs, mid_exits = 0, 0
        late_obs, late_exits = 0, 0
        
        for bucket in buckets:
            if bucket.bucket_end <= 12:
                early_obs += bucket.n_observations
                early_exits += bucket.n_exits
            elif bucket.bucket_end <= 48:
                mid_obs += bucket.n_observations
                mid_exits += bucket.n_exits
            else:
                late_obs += bucket.n_observations
                late_exits += bucket.n_exits
        
        early_rate = early_exits / early_obs if early_obs > 0 else 0
        mid_rate = mid_exits / mid_obs if mid_obs > 0 else 0
        late_rate = late_exits / late_obs if late_obs > 0 else 0
        
        # Determine trend
        if early_obs < MIN_SAMPLES or mid_obs < MIN_SAMPLES:
            trend = "Insufficient data"
        elif late_rate > early_rate * 1.2:
            trend = "↑ INCREASING"
        elif late_rate < early_rate * 0.8:
            trend = "↓ DECREASING"
        else:
            trend = "→ CONSTANT"
        
        print(f"  {state:<8} {early_rate:>14.2%} {mid_rate:>14.2%} {late_rate:>14.2%} {trend:<20}")


def display_conditional_transitions(hazard_buckets: Dict[int, List[HazardBucket]]):
    """Display P(→bearish) by duration bucket - does it depend on time in state?"""
    
    print(f"\n{'=' * 120}")
    print("CONDITIONAL TRANSITIONS: Does P(→bearish) Depend on Duration?")
    print(f"{'=' * 120}")
    
    print(f"\n  {'State':<8} {'Early (0-12h)':>15} {'Mid (12-48h)':>15} {'Late (48h+)':>15} {'Trend':<25}")
    print(f"  {'─' * 85}")
    
    for state in sorted(BULLISH_STATES):
        buckets = hazard_buckets[state]
        
        # Aggregate P(→bearish) by time period
        early_bear, early_exits = 0, 0
        mid_bear, mid_exits = 0, 0
        late_bear, late_exits = 0, 0
        
        for bucket in buckets:
            if bucket.bucket_end <= 12:
                early_bear += bucket.n_to_bearish
                early_exits += bucket.n_exits
            elif bucket.bucket_end <= 48:
                mid_bear += bucket.n_to_bearish
                mid_exits += bucket.n_exits
            else:
                late_bear += bucket.n_to_bearish
                late_exits += bucket.n_exits
        
        early_p = early_bear / early_exits if early_exits > 0 else 0
        mid_p = mid_bear / mid_exits if mid_exits > 0 else 0
        late_p = late_bear / late_exits if late_exits > 0 else 0
        
        # Determine trend
        if early_exits < 10 or mid_exits < 10:
            trend = "Insufficient data"
        elif late_p > early_p * 1.3:
            trend = "↑ More bearish exits late"
        elif late_p < early_p * 0.7:
            trend = "↓ Fewer bearish exits late"
        else:
            trend = "→ Constant"
        
        print(f"  {state:<8} {early_p:>14.1%} {mid_p:>14.1%} {late_p:>14.1%} {trend:<25}")


def display_cross_pair_summary(all_hazards: Dict[str, Dict[int, List[HazardBucket]]],
                                all_distributions: Dict[str, Dict[int, DurationDistribution]]):
    """Display aggregated cross-pair summary."""
    
    print(f"\n{'=' * 120}")
    print("CROSS-PAIR SUMMARY")
    print(f"{'=' * 120}")
    
    # Aggregate Weibull shape parameters
    print(f"\n{'─' * 120}")
    print("WEIBULL SHAPE (β) BY PAIR AND STATE")
    print(f"{'─' * 120}")
    
    print(f"\n  {'State':<8}", end="")
    for pair in DEPLOY_PAIRS:
        print(f" {pair:>10}", end="")
    print(f" {'Mean':>10} {'Interpretation':<20}")
    print(f"  {'─' * 100}")
    
    for state in sorted(BULLISH_STATES):
        print(f"  {state:<8}", end="")
        betas = []
        for pair in DEPLOY_PAIRS:
            dist = all_distributions[pair][state]
            if dist.n_visits >= MIN_SAMPLES:
                beta, _, _ = fit_weibull(dist.durations)
                betas.append(beta)
                print(f" {beta:>10.2f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        
        if betas:
            mean_beta = np.mean(betas)
            if mean_beta > 1.1:
                interp = "Increasing"
            elif mean_beta < 0.9:
                interp = "Decreasing"
            else:
                interp = "~Constant"
            print(f" {mean_beta:>10.2f} {interp:<20}")
        else:
            print(f" {'N/A':>10} {'Insufficient data':<20}")
    
    # Aggregate P(→bearish)
    print(f"\n{'─' * 120}")
    print("P(→BEARISH) BY PAIR AND STATE (Overall)")
    print(f"{'─' * 120}")
    
    print(f"\n  {'State':<8}", end="")
    for pair in DEPLOY_PAIRS:
        print(f" {pair:>10}", end="")
    print(f" {'Mean':>10}")
    print(f"  {'─' * 85}")
    
    for state in sorted(BULLISH_STATES):
        print(f"  {state:<8}", end="")
        probs = []
        for pair in DEPLOY_PAIRS:
            buckets = all_hazards[pair][state]
            total_exits = sum(b.n_exits for b in buckets)
            total_bearish = sum(b.n_to_bearish for b in buckets)
            if total_exits >= MIN_SAMPLES:
                p = total_bearish / total_exits
                probs.append(p)
                print(f" {p:>9.1%}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        
        if probs:
            print(f" {np.mean(probs):>9.1%}")
        else:
            print(f" {'N/A':>10}")


def export_model_parameters(all_hazards: Dict[str, Dict[int, List[HazardBucket]]],
                             all_distributions: Dict[str, Dict[int, DurationDistribution]],
                             output_path: str):
    """Export parameters for use in trading model."""
    
    # Aggregate across pairs
    agg_hazards = {state: {i: {'obs': 0, 'exits': 0, 'bearish': 0} 
                           for i in range(len(DURATION_BUCKETS))} for state in range(16)}
    agg_durations = {state: [] for state in range(16)}
    
    for pair in DEPLOY_PAIRS:
        for state in range(16):
            # Aggregate hazard buckets
            for i, bucket in enumerate(all_hazards[pair][state]):
                agg_hazards[state][i]['obs'] += bucket.n_observations
                agg_hazards[state][i]['exits'] += bucket.n_exits
                agg_hazards[state][i]['bearish'] += bucket.n_to_bearish
            
            # Aggregate durations
            agg_durations[state].extend(all_distributions[pair][state].durations)
    
    with open(output_path, 'w') as f:
        f.write('"""Auto-generated hazard model parameters."""\n\n')
        
        # Duration buckets
        f.write(f'DURATION_BUCKETS = {DURATION_BUCKETS}\n\n')
        
        # Hazard rates by bucket
        f.write('# Hourly exit rate by state and duration bucket\n')
        f.write('# Format: {state: {bucket_idx: (exit_rate, p_to_bearish, n_samples)}}\n')
        f.write('HAZARD_TABLE = {\n')
        
        for state in range(16):
            f.write(f'    {state}: {{\n')
            for i in range(len(DURATION_BUCKETS)):
                obs = agg_hazards[state][i]['obs']
                exits = agg_hazards[state][i]['exits']
                bearish = agg_hazards[state][i]['bearish']
                
                exit_rate = exits / obs if obs > 0 else 0
                p_bearish = bearish / exits if exits > 0 else 0.5
                
                f.write(f'        {i}: ({exit_rate:.6f}, {p_bearish:.4f}, {obs}),\n')
            f.write(f'    }},\n')
        f.write('}\n\n')
        
        # Weibull parameters
        f.write('# Weibull distribution parameters (shape, scale)\n')
        f.write('WEIBULL_PARAMS = {\n')
        
        for state in range(16):
            durations = agg_durations[state]
            if len(durations) >= 10:
                beta, eta, _ = fit_weibull(durations)
                f.write(f'    {state}: ({beta:.4f}, {eta:.2f}),\n')
            else:
                f.write(f'    {state}: (1.0, 24.0),  # Default (insufficient data)\n')
        f.write('}\n\n')
        
        # Overall P(→bearish)
        f.write('# Overall P(→bearish | state)\n')
        f.write('P_TO_BEARISH = {\n')
        
        for state in range(16):
            total_exits = sum(agg_hazards[state][i]['exits'] for i in range(len(DURATION_BUCKETS)))
            total_bearish = sum(agg_hazards[state][i]['bearish'] for i in range(len(DURATION_BUCKETS)))
            p = total_bearish / total_exits if total_exits > 0 else 0.5
            f.write(f'    {state}: {p:.4f},\n')
        f.write('}\n')
    
    print(f"\nModel parameters exported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Empirical Hazard Analysis')
    parser.add_argument('--pair', type=str, default=None, help='Analyze specific pair')
    parser.add_argument('--output', type=str, default='hazard_model_params.py',
                        help='Output file for model parameters')
    args = parser.parse_args()
    
    print("=" * 120)
    print("EMPIRICAL HAZARD ANALYSIS")
    print("=" * 120)
    print(f"""
    Analyzing duration-dependent exit rates and transition probabilities.
    
    Key questions:
    1. Is exit probability constant, increasing, or decreasing with duration?
    2. Does P(→bearish) depend on how long we've been in the state?
    3. What is P(bearish flip in Δt | state, duration)?
    
    Duration buckets: {DURATION_BUCKETS}
    Horizons: {HORIZONS} hours
    """)
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    
    pairs = [args.pair] if args.pair else DEPLOY_PAIRS
    
    all_hazards = {}
    all_distributions = {}
    all_horizon_probs = {}
    
    for pair in pairs:
        print(f"\nProcessing {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        if df_1h is None or len(df_1h) == 0:
            print(f"WARNING: No data for {pair}")
            continue
        
        print(f"loaded {len(df_1h):,} hours...", end=" ", flush=True)
        
        # Compute states
        states = compute_states_hourly(df_1h)
        print(f"states computed...", end=" ", flush=True)
        
        # Analyze hazards
        hazard_buckets, distributions = analyze_hazards(states)
        horizon_probs = compute_horizon_probabilities(hazard_buckets, HORIZONS)
        
        all_hazards[pair] = hazard_buckets
        all_distributions[pair] = distributions
        all_horizon_probs[pair] = horizon_probs
        
        print("done")
        
        # Display per-pair analysis
        display_hazard_analysis(hazard_buckets, distributions, horizon_probs, pair)
    
    # Display aggregate analysis
    if len(pairs) > 1:
        display_hazard_trend(hazard_buckets)  # Using last pair as example
        display_conditional_transitions(hazard_buckets)
        display_cross_pair_summary(all_hazards, all_distributions)
    
    # Export model parameters
    if len(pairs) > 1:
        export_model_parameters(all_hazards, all_distributions, args.output)
    
    print(f"\n{'=' * 120}")
    print("ANALYSIS COMPLETE")
    print("=" * 120)
    
    return all_hazards, all_distributions, all_horizon_probs


if __name__ == "__main__":
    hazards, distributions, horizon_probs = main()