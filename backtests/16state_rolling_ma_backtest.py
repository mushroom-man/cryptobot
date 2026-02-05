#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling MA Backtest v3 - Weekly MA Focus
=========================================
Test removing 72h MA entirely and/or rolling the 168h (weekly) MA.

Test Configurations:
    A  = Baseline: 24h@24h, 72h@72h, 168h@168h (16 states)
    
    D CONFIGS (Remove 72h MA - reduces to 8 states):
    D1 = 24h@24h, NO 72h, 168h@24h
    D2 = 24h@24h, NO 72h, 168h@72h
    
    E CONFIGS (Keep all MAs with rolling updates):
    E1 = 24h@24h, 72h@24h, 168h@72h

Key Changes from v2:
    - D configs remove 72h MA entirely (8-state system)
    - E config tests rolling 168h MA
    - NO_MA72_ONLY filter disabled for D configs (no 72h to filter)

Usage:
    python 16state_rolling_ma_backtest_v3.py --all
    python 16state_rolling_ma_backtest_v3.py --config D1
    python 16state_rolling_ma_backtest_v3.py --d-only
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from cryptobot.data.database import Database

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from itertools import product
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked from validation)
MA_PERIOD_24H = 16      # 16 bars on 24h = 384 hours
MA_PERIOD_72H = 6       # 6 bars on 72h = 432 hours
MA_PERIOD_168H = 2      # 2 bars on 168h = 336 hours

# Hourly equivalents for rolling MAs
MA_HOURS_24H = MA_PERIOD_24H * 24    # 384 hours
MA_HOURS_72H = MA_PERIOD_72H * 72    # 432 hours
MA_HOURS_168H = MA_PERIOD_168H * 168  # 336 hours

ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Hit rate parameters
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_STATE = 20

# Calendar alignment
MIN_TRAINING_MONTHS = 12
HIT_RATE_RECALC_MONTHS = 1
VOL_LOOKBACK_MONTHS = 1
COV_LOOKBACK_MONTHS = 2

# Exposure limits
DIR_MAX_EXPOSURE = 2.0
DIR_TARGET_VOL = 0.40
DIR_DD_START_REDUCE = -0.20
DIR_DD_MIN_EXPOSURE = -0.50
DIR_MIN_EXPOSURE_FLOOR = 0.40

# Shared parameters
INITIAL_CAPITAL = 100000.0
TRADING_COST = 0.0020
MIN_TRADE_SIZE = 100.0

# Test configurations
# Format: {
#   'ma24_freq': update freq for 24h MA,
#   'ma72_freq': update freq for 72h MA (None = removed),
#   'ma168_freq': update freq for 168h MA,
#   'use_72h': whether to include 72h MA
# }
ROLLING_CONFIGS = {
    'A':  {'name': 'Baseline (16-state)',  'ma24_freq': '24h', 'ma72_freq': '72h',  'ma168_freq': '168h', 'use_72h': True},
    'D1': {'name': 'No72h, 168h@24h',      'ma24_freq': '24h', 'ma72_freq': None,   'ma168_freq': '24h',  'use_72h': False},
    'D2': {'name': 'No72h, 168h@72h',      'ma24_freq': '24h', 'ma72_freq': None,   'ma168_freq': '72h',  'use_72h': False},
    'E1': {'name': '72h@24h, 168h@72h',    'ma24_freq': '24h', 'ma72_freq': '24h',  'ma168_freq': '72h',  'use_72h': True},
    # F CONFIGS: Keep 24h fixed, roll 72h and/or 168h
    'F1': {'name': '72h@24h only',         'ma24_freq': '24h', 'ma72_freq': '24h',  'ma168_freq': '168h', 'use_72h': True},
    'F2': {'name': '168h@72h only',        'ma24_freq': '24h', 'ma72_freq': '72h',  'ma168_freq': '72h',  'use_72h': True},
    'F3': {'name': '72h@24h, 168h@72h',    'ma24_freq': '24h', 'ma72_freq': '24h',  'ma168_freq': '72h',  'use_72h': True},
    'F4': {'name': '72h@12h, 168h@24h',    'ma24_freq': '24h', 'ma72_freq': '12h',  'ma168_freq': '24h',  'use_72h': True},
}


# =============================================================================
# CALENDAR HELPERS
# =============================================================================

def get_month_start(date: pd.Timestamp, months_back: int = 0) -> pd.Timestamp:
    """Get the 1st of the month, optionally N months back."""
    target = date - pd.DateOffset(months=months_back)
    return target.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_months_of_data(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Calculate full calendar months between two dates."""
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def has_minimum_training(current_date: pd.Timestamp, data_start: pd.Timestamp, 
                          min_months: int) -> bool:
    """Check if we have minimum training months of data."""
    months_available = get_months_of_data(data_start, current_date)
    return months_available >= min_months


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    name: str
    config: str = ""
    n_states: int = 16
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_raw: float = 0.0
    calmar_ratio: float = 0.0
    n_trades: int = 0
    total_costs: float = 0.0
    signals_filtered: int = 0
    signals_total: int = 0
    equity_curve: Optional[pd.Series] = None
    yearly_returns: Dict[int, float] = field(default_factory=dict)


# =============================================================================
# SIGNAL GENERATION - FLEXIBLE MA VERSION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def label_trend_binary_rolling(
    close_series: pd.Series,
    ma_series: pd.Series,
    entry_buffer: float,
    exit_buffer: float
) -> pd.Series:
    """
    Binary trend detection with hysteresis.
    Works on pre-computed MA series (can be at any frequency).
    """
    labels = pd.Series(index=close_series.index, dtype=float)
    labels[:] = np.nan
    
    current = 1  # Start bullish
    
    for i in range(len(close_series)):
        idx = close_series.index[i]
        price = close_series.iloc[i]
        ma = ma_series.iloc[i] if idx in ma_series.index else np.nan
        
        if pd.isna(ma):
            labels.iloc[i] = current
            continue
        
        if current == 1:
            # Currently bullish - check for bearish flip
            if price < ma * (1 - exit_buffer) and price < ma * (1 - entry_buffer):
                current = 0
        else:
            # Currently bearish - check for bullish flip
            if price > ma * (1 + exit_buffer) and price > ma * (1 + entry_buffer):
                current = 1
        
        labels.iloc[i] = current
    
    return labels.astype(int)


def generate_signals_flexible(
    df_1h: pd.DataFrame,
    ma24_update_freq: str = '24h',
    ma72_update_freq: Optional[str] = '72h',
    ma168_update_freq: str = '168h',
    use_72h: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Generate signals with flexible MA configuration.
    
    Args:
        df_1h: Hourly OHLCV data
        ma24_update_freq: Update frequency for 24h MA
        ma72_update_freq: Update frequency for 72h MA (None if not used)
        ma168_update_freq: Update frequency for 168h MA
        use_72h: Whether to include 72h MA in signals
    
    Returns:
        Tuple of (signals DataFrame, prices DataFrame, number of states)
    """
    # Resample to standard timeframes
    df_24h = resample_ohlcv(df_1h, '24h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    # =========================================================================
    # 24h MA - configurable update frequency
    # =========================================================================
    if ma24_update_freq == '24h':
        ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
        trend_24h = label_trend_binary_rolling(
            df_24h['close'], ma_24h, ENTRY_BUFFER, EXIT_BUFFER
        )
    else:
        ma_24h_hourly = df_1h['close'].rolling(MA_HOURS_24H).mean()
        ma_24h_at_freq = ma_24h_hourly.resample(ma24_update_freq).last()
        close_at_freq = df_1h['close'].resample(ma24_update_freq).last()
        
        trend_24h_freq = label_trend_binary_rolling(
            close_at_freq, ma_24h_at_freq, ENTRY_BUFFER, EXIT_BUFFER
        )
        trend_24h = trend_24h_freq.reindex(df_24h.index, method='ffill')
        ma_24h = ma_24h_at_freq.reindex(df_24h.index, method='ffill')
    
    # =========================================================================
    # 72h MA - optional, configurable update frequency
    # =========================================================================
    ma_72h_aligned = None
    if use_72h and ma72_update_freq is not None:
        df_72h = resample_ohlcv(df_1h, '72h')
        
        if ma72_update_freq == '72h':
            ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
            ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
        else:
            ma_72h_hourly = df_1h['close'].rolling(MA_HOURS_72H).mean()
            ma_72h_at_freq = ma_72h_hourly.resample(ma72_update_freq).last()
            ma_72h_aligned = ma_72h_at_freq.reindex(df_24h.index, method='ffill')
    
    # =========================================================================
    # 168h MA - configurable update frequency
    # =========================================================================
    if ma168_update_freq == '168h':
        ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
        ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    else:
        ma_168h_hourly = df_1h['close'].rolling(MA_HOURS_168H).mean()
        ma_168h_at_freq = ma_168h_hourly.resample(ma168_update_freq).last()
        ma_168h_aligned = ma_168h_at_freq.reindex(df_24h.index, method='ffill')
    
    # trend_168h: Based on 168h timeframe data
    trend_168h_raw = label_trend_binary_rolling(
        df_168h['close'], df_168h['close'].rolling(MA_PERIOD_168H).mean(), 
        ENTRY_BUFFER, EXIT_BUFFER
    )
    
    # =========================================================================
    # Build signals DataFrame
    # =========================================================================
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_168h'] = trend_168h_raw.shift(1).reindex(df_24h.index, method='ffill')
    
    if use_72h and ma_72h_aligned is not None:
        aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
        n_states = 16
    else:
        n_states = 8
    
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    signals = aligned.dropna().astype(int)
    
    return signals, df_24h, n_states


# =============================================================================
# SIGNAL FILTER
# =============================================================================

def should_trade_signal_16state(prev_state: Optional[Tuple], 
                                curr_state: Tuple,
                                use_filter: bool = True) -> bool:
    """
    NO_MA72_ONLY filter for 16-state: Skip if ONLY ma72_above_ma24 changed.
    """
    if prev_state is None:
        return True
    
    if prev_state == curr_state:
        return False
    
    if not use_filter:
        return True
    
    # For 16-state: (trend_24h, trend_168h, ma72_above_ma24, ma168_above_ma24)
    if len(curr_state) != 4:
        return True  # Can't apply filter to non-16-state
    
    trend_24h_changed = prev_state[0] != curr_state[0]
    trend_168h_changed = prev_state[1] != curr_state[1]
    ma72_changed = prev_state[2] != curr_state[2]
    ma168_changed = prev_state[3] != curr_state[3]
    
    only_ma72_changed = (ma72_changed and 
                         not trend_24h_changed and 
                         not trend_168h_changed and 
                         not ma168_changed)
    
    return not only_ma72_changed


def should_trade_signal_8state(prev_state: Optional[Tuple], 
                               curr_state: Tuple) -> bool:
    """
    For 8-state (no 72h): Always trade on state change.
    No filter since there's no ma72_above_ma24 to filter.
    """
    if prev_state is None:
        return True
    return prev_state != curr_state


# =============================================================================
# HIT RATE CALCULATION
# =============================================================================

def calculate_expanding_hit_rates_16state(returns_history: pd.Series, 
                                           signals_history: pd.DataFrame) -> Dict:
    """Calculate 16-state hit rates using ONLY historical data."""
    all_trend_perms = list(product([0, 1], repeat=2))
    all_ma_perms = list(product([0, 1], repeat=2))
    
    if len(returns_history) < MIN_SAMPLES_PER_STATE:
        return {(t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for t in all_trend_perms for m in all_ma_perms}
    
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < MIN_SAMPLES_PER_STATE:
        return {(t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for t in all_trend_perms for m in all_ma_perms}
    
    aligned_returns = returns_history.loc[common_idx]
    aligned_signals = signals_history.loc[common_idx]
    
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    hit_rates = {}
    
    for trend_perm in all_trend_perms:
        for ma_perm in all_ma_perms:
            mask = (
                (aligned_signals['trend_24h'] == trend_perm[0]) &
                (aligned_signals['trend_168h'] == trend_perm[1]) &
                (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
                (aligned_signals['ma168_above_ma24'] == ma_perm[1])
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            
            if n > 0:
                hit_rate = (perm_returns > 0).sum() / n
            else:
                hit_rate = 0.5
            
            hit_rates[(trend_perm, ma_perm)] = {
                'n': n,
                'hit_rate': hit_rate,
                'sufficient': n >= MIN_SAMPLES_PER_STATE,
            }
    
    return hit_rates


def calculate_expanding_hit_rates_8state(returns_history: pd.Series, 
                                          signals_history: pd.DataFrame) -> Dict:
    """Calculate 8-state hit rates (no 72h MA)."""
    all_trend_perms = list(product([0, 1], repeat=2))  # trend_24h, trend_168h
    all_ma_perms = list(product([0, 1], repeat=1))     # ma168_above_ma24 only
    
    if len(returns_history) < MIN_SAMPLES_PER_STATE:
        return {(t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for t in all_trend_perms for m in all_ma_perms}
    
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < MIN_SAMPLES_PER_STATE:
        return {(t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for t in all_trend_perms for m in all_ma_perms}
    
    aligned_returns = returns_history.loc[common_idx]
    aligned_signals = signals_history.loc[common_idx]
    
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    hit_rates = {}
    
    for trend_perm in all_trend_perms:
        for ma_perm in all_ma_perms:
            mask = (
                (aligned_signals['trend_24h'] == trend_perm[0]) &
                (aligned_signals['trend_168h'] == trend_perm[1]) &
                (aligned_signals['ma168_above_ma24'] == ma_perm[0])
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            
            if n > 0:
                hit_rate = (perm_returns > 0).sum() / n
            else:
                hit_rate = 0.5
            
            hit_rates[(trend_perm, ma_perm)] = {
                'n': n,
                'hit_rate': hit_rate,
                'sufficient': n >= MIN_SAMPLES_PER_STATE,
            }
    
    return hit_rates


def get_position(state_key: Tuple, hit_rates: Dict) -> float:
    """Get position signal for any state configuration."""
    data = hit_rates.get(state_key, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.00


def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Calculate risk parity weights for assets."""
    vols = returns_df.std() * np.sqrt(365)
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database, 
                  ma24_update_freq: str = '24h', 
                  ma72_update_freq: Optional[str] = '72h',
                  ma168_update_freq: str = '168h',
                  use_72h: bool = True) -> Tuple[Dict, int]:
    """Load and prepare data for all pairs with specified MA frequencies."""
    data = {}
    n_states = None
    
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        signals, df_24h, pair_n_states = generate_signals_flexible(
            df_1h, 
            ma24_update_freq=ma24_update_freq,
            ma72_update_freq=ma72_update_freq,
            ma168_update_freq=ma168_update_freq,
            use_72h=use_72h
        )
        returns = df_24h['close'].pct_change()
        
        data[pair] = {
            'prices': df_24h,
            'signals': signals,
            'returns': returns,
        }
        
        if n_states is None:
            n_states = pair_n_states
        
        print(f"{len(df_24h)} days")
    
    return data, n_states


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    data: Dict,
    dates: List[pd.Timestamp],
    data_start: pd.Timestamp,
    n_states: int = 16,
    trading_cost: float = TRADING_COST,
    name: str = "Backtest",
    config: str = "",
    use_dd_protection: bool = True
) -> BacktestResult:
    """Run backtest with specified data."""
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    equity_curve = {}
    
    total_costs = 0.0
    n_trades = 0
    signals_filtered = 0
    signals_total = 0
    
    # State tracking
    returns_dict = {pair: data[pair]['returns'] for pair in DEPLOY_PAIRS}
    returns_df = pd.DataFrame(returns_dict).dropna()
    asset_weights = pd.Series({pair: 1.0/len(DEPLOY_PAIRS) for pair in DEPLOY_PAIRS})
    prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    last_rebalance_month = None
    
    # Filter state tracking per asset
    prev_signal_states = {pair: None for pair in DEPLOY_PAIRS}
    active_states = {pair: None for pair in DEPLOY_PAIRS}
    
    # Hit rate caches
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}
    
    # Yearly tracking
    yearly_pnl = {}
    yearly_start_equity = {}
    current_year = dates[0].year
    yearly_start_equity[current_year] = equity
    yearly_pnl[current_year] = 0.0
    
    equity_curve[dates[0]] = equity
    
    # Determine if using 16-state or 8-state
    is_16state = n_states == 16
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        current_month = (date.year, date.month)
        
        # Year tracking
        if next_date.year != current_year:
            current_year = next_date.year
            yearly_start_equity[current_year] = equity
            yearly_pnl[current_year] = 0.0
        
        # Drawdown protection
        dd_scalar = 1.0
        if use_dd_protection:
            current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
            if current_dd >= DIR_DD_START_REDUCE:
                dd_scalar = 1.0
            elif current_dd <= DIR_DD_MIN_EXPOSURE:
                dd_scalar = DIR_MIN_EXPOSURE_FLOOR
            else:
                range_dd = DIR_DD_START_REDUCE - DIR_DD_MIN_EXPOSURE
                position = (current_dd - DIR_DD_MIN_EXPOSURE) / range_dd
                dd_scalar = DIR_MIN_EXPOSURE_FLOOR + position * (1.0 - DIR_MIN_EXPOSURE_FLOOR)
        
        # Monthly rebalance
        if last_rebalance_month != current_month:
            lookback_start = get_month_start(date, COV_LOOKBACK_MONTHS)
            lookback_end = get_month_start(date, 0)
            lookback_returns = returns_df.loc[lookback_start:lookback_end]
            if len(lookback_returns) >= 20:
                asset_weights = calculate_risk_parity_weights(lookback_returns)
            last_rebalance_month = current_month
        
        # Calculate exposures
        base_exposure = 0.0
        asset_exposures = {}
        
        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            
            if date not in signals.index:
                continue
            
            sig = signals.loc[date]
            
            # Build current state based on available signals
            if is_16state:
                current_state = (
                    int(sig['trend_24h']), 
                    int(sig['trend_168h']),
                    int(sig['ma72_above_ma24']), 
                    int(sig['ma168_above_ma24'])
                )
            else:
                # 8-state: no ma72_above_ma24
                current_state = (
                    int(sig['trend_24h']), 
                    int(sig['trend_168h']),
                    int(sig['ma168_above_ma24'])
                )
            
            # Apply filter
            prev_state = prev_signal_states[pair]
            
            if current_state != prev_state:
                signals_total += 1
                
                if is_16state:
                    should_trade = should_trade_signal_16state(prev_state, current_state, use_filter=True)
                else:
                    should_trade = should_trade_signal_8state(prev_state, current_state)
                
                if should_trade:
                    active_states[pair] = current_state
                else:
                    signals_filtered += 1
            
            prev_signal_states[pair] = current_state
            
            # Use active state for position
            active = active_states[pair]
            if active is None:
                active = current_state
                active_states[pair] = current_state
            
            # Build state key for hit rate lookup
            if is_16state:
                trend_perm = (active[0], active[1])
                ma_perm = (active[2], active[3])
                state_key = (trend_perm, ma_perm)
            else:
                trend_perm = (active[0], active[1])
                ma_perm = (active[2],)  # Single element tuple
                state_key = (trend_perm, ma_perm)
            
            if not has_minimum_training(date, data_start, MIN_TRAINING_MONTHS):
                pos = 0.50
            else:
                current_m = (date.year, date.month)
                if last_recalc_month[pair] != current_m:
                    cutoff = get_month_start(date, 0)
                    hist_ret = returns[returns.index < cutoff]
                    hist_sig = signals[signals.index < cutoff]
                    if len(hist_ret) > 0:
                        if is_16state:
                            hit_rate_cache[pair] = calculate_expanding_hit_rates_16state(hist_ret, hist_sig)
                        else:
                            hit_rate_cache[pair] = calculate_expanding_hit_rates_8state(hist_ret, hist_sig)
                        last_recalc_month[pair] = current_m
                
                pos = get_position(state_key, hit_rate_cache[pair])
            
            asset_exposures[pair] = pos * asset_weights[pair]
            base_exposure += asset_exposures[pair]
        
        if base_exposure > 1.0:
            for pair in asset_exposures:
                asset_exposures[pair] /= base_exposure
            base_exposure = 1.0
        
        # Vol scaling
        vol_scalar = 1.0
        if date in returns_df.index:
            vol_lookback_start = get_month_start(date, VOL_LOOKBACK_MONTHS)
            vol_lookback_end = get_month_start(date, 0)
            vol_returns = returns_df.loc[vol_lookback_start:vol_lookback_end]
            if len(vol_returns) >= 15:
                market_returns = vol_returns.mean(axis=1)
                realized_vol = market_returns.std() * np.sqrt(365)
                if realized_vol > 0:
                    vol_scalar = DIR_TARGET_VOL / realized_vol
                    vol_scalar = np.clip(vol_scalar, DIR_MIN_EXPOSURE_FLOOR, DIR_MAX_EXPOSURE)
        
        risk_scalar = min(vol_scalar, 1.0 / dd_scalar if dd_scalar > 0 else 1.0)
        risk_scalar = max(risk_scalar, DIR_MIN_EXPOSURE_FLOOR)
        
        for pair in asset_exposures:
            asset_exposures[pair] *= risk_scalar
        
        # Execute trades
        daily_pnl = 0.0
        daily_cost = 0.0
        
        for pair in DEPLOY_PAIRS:
            curr_exp = asset_exposures.get(pair, 0.0)
            prev_exp = prev_exposures.get(pair, 0.0)
            
            trade_value = abs(curr_exp - prev_exp) * equity
            
            if trade_value >= MIN_TRADE_SIZE:
                daily_cost += trade_value * trading_cost
                n_trades += 1
                prev_exposures[pair] = curr_exp
            else:
                curr_exp = prev_exp
                asset_exposures[pair] = prev_exp
            
            if curr_exp > 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    daily_pnl += equity * curr_exp * ret
        
        daily_pnl -= daily_cost
        total_costs += daily_cost
        equity += daily_pnl
        
        yearly_pnl[next_date.year] = yearly_pnl.get(next_date.year, 0) + daily_pnl
        
        equity_curve[next_date] = equity
        
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
    
    # Calculate raw DD from equity curve
    equity_series = pd.Series(equity_curve)
    running_peak = equity_series.cummax()
    drawdowns = (running_peak - equity_series) / running_peak
    max_drawdown_raw = drawdowns.max()
    
    # Calculate metrics
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    returns_series = equity_series.pct_change().dropna()
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(365) if returns_series.std() > 0 else 0
    calmar = annual_return / max_drawdown_raw if max_drawdown_raw > 0 else 0
    
    # Yearly returns
    yearly_returns = {}
    for year in yearly_start_equity:
        if year in yearly_pnl:
            start_eq = yearly_start_equity[year]
            yearly_returns[year] = yearly_pnl[year] / start_eq if start_eq > 0 else 0
    
    return BacktestResult(
        name=name,
        config=config,
        n_states=n_states,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        max_drawdown_raw=max_drawdown_raw,
        calmar_ratio=calmar,
        n_trades=n_trades,
        total_costs=total_costs,
        signals_filtered=signals_filtered,
        signals_total=signals_total,
        equity_curve=equity_series,
        yearly_returns=yearly_returns,
    )


# =============================================================================
# DISPLAY
# =============================================================================

def display_backtest_results(result: BacktestResult):
    """Display formatted backtest results."""
    filter_pct = result.signals_filtered / result.signals_total * 100 if result.signals_total > 0 else 0
    
    print(f"""
    ═══════════════════════════════════════════════════════════════
    {result.name.upper()} [{result.config}] ({result.n_states} states)
    ═══════════════════════════════════════════════════════════════
    
    Performance Metrics:
    ─────────────────────────────────────────────────────────────────
    Annual Return:      {result.annual_return*100:>+8.1f}%
    Sharpe Ratio:       {result.sharpe_ratio:>8.2f}
    Max Drawdown:       {result.max_drawdown_raw*100:>8.1f}%
    Calmar Ratio:       {result.calmar_ratio:>8.2f}
    
    Trading Statistics:
    ─────────────────────────────────────────────────────────────────
    Total Trades:       {result.n_trades:>8,}
    Total Costs:        ${result.total_costs:>10,.0f}
    Signals Filtered:   {result.signals_filtered:>8,} / {result.signals_total:>8,} ({filter_pct:.1f}%)
    """)
    
    if result.yearly_returns:
        print("    Yearly Returns:")
        print("    ─────────────────────────────────────────────────────────────────")
        for year in sorted(result.yearly_returns.keys()):
            ret = result.yearly_returns[year]
            print(f"      {year}:  {ret*100:>+8.1f}%")


def display_comparison(results: Dict[str, BacktestResult]):
    """Display side-by-side comparison of all configurations."""
    print("\n" + "=" * 95)
    print("CONFIGURATION COMPARISON")
    print("=" * 95)
    
    print(f"""
    ┌────────┬─────────────────────┬────────┬──────────┬──────────┬──────────┬──────────┐
    │ Config │ Name                │ States │ Ann.Ret  │ Sharpe   │ Max DD   │ Trades   │
    ├────────┼─────────────────────┼────────┼──────────┼──────────┼──────────┼──────────┤""")
    
    baseline_sharpe = results.get('A', results[list(results.keys())[0]]).sharpe_ratio
    
    for config in sorted(results.keys()):
        result = results[config]
        sharpe_diff = result.sharpe_ratio - baseline_sharpe
        diff_str = f"({sharpe_diff:+.2f})" if config != 'A' else ""
        
        print(f"    │ {config:6s} │ {result.name:19s} │ {result.n_states:>6} │ {result.annual_return*100:>+7.1f}% │ "
              f"{result.sharpe_ratio:>5.2f} {diff_str:>6s} │ {result.max_drawdown_raw*100:>7.1f}% │ {result.n_trades:>8,} │")
    
    print(f"""    └────────┴─────────────────────┴────────┴──────────┴──────────┴──────────┴──────────┘""")
    
    # Find best config
    best_config = max(results.keys(), key=lambda k: results[k].sharpe_ratio)
    best_result = results[best_config]
    
    print(f"""
    BEST CONFIGURATION: {best_config} ({best_result.name})
    ─────────────────────────────────────────────────────────────────
    States:             {best_result.n_states}
    Sharpe Ratio:       {best_result.sharpe_ratio:.2f}
    Annual Return:      {best_result.annual_return*100:+.1f}%
    Max Drawdown:       {best_result.max_drawdown_raw*100:.1f}%
    """)
    
    if best_config != 'A':
        improvement = best_result.sharpe_ratio - baseline_sharpe
        print(f"    Sharpe improvement over baseline: {improvement:+.2f}")
    else:
        print("    Baseline is still the best configuration.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rolling MA Backtest v3 - Weekly MA Focus')
    parser.add_argument('--config', type=str, default=None, 
                        choices=['A', 'D1', 'D2', 'E1', 'F1', 'F2', 'F3', 'F4'],
                        help='Run specific config only')
    parser.add_argument('--all', action='store_true', 
                        help='Run all configurations')
    parser.add_argument('--d-only', action='store_true',
                        help='Run only D configs (D1, D2)')
    parser.add_argument('--f-only', action='store_true',
                        help='Run only F configs (F1, F2, F3, F4)')
    args = parser.parse_args()
    
    # Determine which configs to run
    if args.config:
        configs_to_run = [args.config]
    elif args.d_only:
        configs_to_run = ['A', 'D1', 'D2']
    elif args.f_only:
        configs_to_run = ['A', 'F1', 'F2', 'F3', 'F4']
    elif args.all:
        configs_to_run = ['A', 'D1', 'D2', 'E1', 'F1', 'F2', 'F3', 'F4']
    else:
        configs_to_run = ['A', 'F1', 'F2', 'F3', 'F4']  # Default to F tests
    
    print("=" * 95)
    print("ROLLING MA BACKTEST v3 - WEEKLY MA FOCUS")
    print("=" * 95)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
    ║  TEST HYPOTHESIS                                                                          ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
    ║  Test rolling 72h/168h MAs while keeping 24h MA fixed.                                    ║
    ║                                                                                           ║
    ║  CONFIGURATIONS:                                                                          ║
    ║    A  = Baseline: 24h@24h, 72h@72h, 168h@168h (16 states)                                 ║
    ║                                                                                           ║
    ║  F CONFIGS (Keep 24h fixed, roll slower MAs):                                             ║
    ║    F1 = 24h@24h, 72h@24h, 168h@168h (roll 72h daily)                                      ║
    ║    F2 = 24h@24h, 72h@72h, 168h@72h  (roll 168h every 3 days)                              ║
    ║    F3 = 24h@24h, 72h@24h, 168h@72h  (roll both)                                           ║
    ║    F4 = 24h@24h, 72h@12h, 168h@24h  (roll both faster)                                    ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"  Configurations to test: {', '.join(configs_to_run)}")
    print()
    
    # Connect to database
    print("  Connecting to database...")
    db = Database()
    
    # Run each configuration
    results = {}
    
    for config in configs_to_run:
        cfg = ROLLING_CONFIGS[config]
        print(f"\n{'=' * 60}")
        print(f"  RUNNING CONFIG {config}: {cfg['name']}")
        ma72_str = cfg['ma72_freq'] if cfg['ma72_freq'] else 'NONE'
        print(f"  24h MA: {cfg['ma24_freq']}, 72h MA: {ma72_str}, 168h MA: {cfg['ma168_freq']}")
        print(f"{'=' * 60}")
        
        # Load data with this config's MA frequencies
        print("\n  Loading data...")
        data, n_states = load_all_data(
            db, 
            ma24_update_freq=cfg['ma24_freq'],
            ma72_update_freq=cfg['ma72_freq'],
            ma168_update_freq=cfg['ma168_freq'],
            use_72h=cfg['use_72h']
        )
        
        print(f"  Using {n_states}-state system")
        
        # Find common dates
        all_dates = None
        for pair in DEPLOY_PAIRS:
            dates = data[pair]['signals'].index
            if all_dates is None:
                all_dates = set(dates)
            else:
                all_dates = all_dates.intersection(set(dates))
        
        dates = sorted(list(all_dates))
        data_start = dates[0]
        
        print(f"  Common dates: {len(dates)} ({dates[0].date()} to {dates[-1].date()})")
        
        # Run backtest
        result = run_backtest(
            data=data,
            dates=dates,
            data_start=data_start,
            n_states=n_states,
            name=cfg['name'],
            config=config
        )
        
        results[config] = result
        display_backtest_results(result)
    
    # Show comparison if multiple configs
    if len(results) > 1:
        display_comparison(results)
    
    print("\n" + "=" * 95)
    print("BACKTEST COMPLETE")
    print("=" * 95)
    
    return results


if __name__ == "__main__":
    results = main()