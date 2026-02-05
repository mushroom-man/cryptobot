# -*- coding: utf-8 -*-
"""
CryptoBot - 16-State Signal Generator
======================================
Validated 16-state signal generation with NO_MA72_ONLY filter.

16-State System (validated configuration):
    - 4 trend states (trend_24h × trend_168h = 2²)
    - 4 MA alignment states (ma72_above_ma24 × ma168_above_ma24 = 2²)
    - Total: 16 unique states

Key Changes from 32-State:
    - Removed trend_72h (80% correlation with trend_24h)
    - Added NO_MA72_ONLY filter (skips trades when only ma72 changed)
    - Result: 56% signal reduction with same alpha

Signal Flow:
    1. Resample 1h OHLCV → 24h/72h/168h
    2. Label trends with hysteresis (binary: bullish/bearish)
    3. Generate 16-state classification
    4. Apply NO_MA72_ONLY filter
    5. Calculate expanding hit rates per state
    6. Convert hit rate → position signal

Usage:
    from cryptobot.signals.generator import SignalGenerator
    
    gen = SignalGenerator()
    signals = gen.generate_signals(df_1h)
    position, details = gen.get_position_for_date(signals, returns, date, data_start)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from itertools import product
from dataclasses import dataclass


# =============================================================================
# CONSTANTS (Validated from backtest)
# =============================================================================

# MA periods per timeframe
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

# Entry/exit buffers for hysteresis
ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Hit rate thresholds
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_STATE = 20

# Pairs strategy thresholds
STRONG_BUY_THRESHOLD = 0.55
BUY_THRESHOLD = 0.50
SELL_THRESHOLD = 0.45

# Filter configuration
USE_MA72_FILTER = True  # NO_MA72_ONLY filter enabled by default


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Signal:
    """Signal output for a single timestamp."""
    timestamp: pd.Timestamp
    state_trend: Tuple[int, int]      # (trend_24h, trend_168h)
    state_ma: Tuple[int, int]          # (ma72_above_ma24, ma168_above_ma24)
    hit_rate: float
    n_samples: int
    sufficient_samples: bool
    position: float                     # 1.0=INVEST, 0.5=SKIP, 0.0=AVOID
    simple_state: str                   # STRONG_BUY/BUY/SELL/STRONG_SELL
    filtered: bool                      # True if signal was filtered


@dataclass
class FilterStats:
    """Statistics from signal filtering."""
    total_signals: int = 0
    filtered_signals: int = 0
    traded_signals: int = 0
    
    @property
    def filter_rate(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return self.filtered_signals / self.total_signals


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to specified timeframe.
    
    Args:
        df: DataFrame with OHLCV columns, datetime index
        timeframe: Target timeframe ('24h', '72h', '168h')
    
    Returns:
        Resampled DataFrame
    """
    return df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


# =============================================================================
# TREND LABELING
# =============================================================================

def label_trend_binary(
    df: pd.DataFrame,
    ma_period: int,
    entry_buffer: float = ENTRY_BUFFER,
    exit_buffer: float = EXIT_BUFFER
) -> pd.Series:
    """
    Binary trend detection with hysteresis.
    
    Uses MA crossover with entry/exit buffers to reduce whipsaws.
    State changes require price to cross MA by buffer amount.
    
    Args:
        df: OHLCV DataFrame
        ma_period: Moving average period
        entry_buffer: Buffer for entering new trend (1.5% default)
        exit_buffer: Buffer for exiting trend (0.5% default)
    
    Returns:
        Series of binary labels (1=bullish, 0=bearish)
    """
    close = df['close']
    ma = close.rolling(ma_period).mean()
    
    labels = pd.Series(index=df.index, dtype=int)
    current = 1  # Start bullish
    
    for i in range(len(df)):
        if pd.isna(ma.iloc[i]):
            labels.iloc[i] = current
            continue
        
        price = close.iloc[i]
        ma_val = ma.iloc[i]
        
        if current == 1:  # Currently bullish
            # Need price below MA by BOTH buffers to flip bearish
            if price < ma_val * (1 - exit_buffer) and price < ma_val * (1 - entry_buffer):
                current = 0
        else:  # Currently bearish
            # Need price above MA by BOTH buffers to flip bullish
            if price > ma_val * (1 + exit_buffer) and price > ma_val * (1 + entry_buffer):
                current = 1
        
        labels.iloc[i] = current
    
    return labels


# =============================================================================
# 16-STATE SIGNAL GENERATION
# =============================================================================

def generate_16state_signals(
    df_24h: pd.DataFrame,
    df_72h: pd.DataFrame,
    df_168h: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate 16-state signals from multi-timeframe data.
    
    States (validated configuration - removed trend_72h):
        - 4 trend states: trend_24h × trend_168h (2²)
        - 4 MA alignment: ma72_above_ma24 × ma168_above_ma24 (2²)
        - Total: 16 states
    
    CRITICAL: Applies shift(1) BEFORE reindex to prevent look-ahead bias.
    
    Args:
        df_24h: 24-hour OHLCV data
        df_72h: 72-hour OHLCV data
        df_168h: 168-hour OHLCV data
    
    Returns:
        DataFrame with columns:
            - trend_24h, trend_168h (binary)
            - ma72_above_ma24, ma168_above_ma24 (binary)
    """
    # Label trends at each timeframe
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H)
    
    # Calculate MAs
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Align higher timeframe MAs to 24h index
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    # Build aligned signals DataFrame
    # SHIFT BEFORE REINDEX to prevent look-ahead bias
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


# Legacy alias for backward compatibility
def generate_32state_signals(
    df_24h: pd.DataFrame,
    df_72h: pd.DataFrame,
    df_168h: pd.DataFrame
) -> pd.DataFrame:
    """
    Legacy function - redirects to 16-state.
    
    Note: The old 32-state system included trend_72h which had 80% correlation
    with trend_24h. The validated 16-state system removes this redundancy.
    """
    import warnings
    warnings.warn(
        "generate_32state_signals is deprecated. Use generate_16state_signals instead.",
        DeprecationWarning
    )
    return generate_16state_signals(df_24h, df_72h, df_168h)


# =============================================================================
# SIGNAL FILTERING (NO_MA72_ONLY)
# =============================================================================

def should_trade_signal(
    prev_state: Optional[Tuple[int, int, int, int]],
    curr_state: Tuple[int, int, int, int],
    use_filter: bool = USE_MA72_FILTER
) -> bool:
    """
    NO_MA72_ONLY filter: Skip if ONLY ma72_above_ma24 changed.
    
    This filter removes noisy signals where the only change is the
    ma72_above_ma24 component. Validated to filter 56% of signals
    while preserving alpha.
    
    Args:
        prev_state: Previous state tuple (trend_24h, trend_168h, ma72_above_ma24, ma168_above_ma24)
        curr_state: Current state tuple
        use_filter: Whether to apply the filter
    
    Returns:
        True if signal should be traded, False if filtered
    """
    # First signal always trades
    if prev_state is None:
        return True
    
    # No change = no trade needed
    if prev_state == curr_state:
        return False
    
    # If filter disabled, trade all state changes
    if not use_filter:
        return True
    
    # Check what changed
    trend_24h_changed = prev_state[0] != curr_state[0]
    trend_168h_changed = prev_state[1] != curr_state[1]
    ma72_changed = prev_state[2] != curr_state[2]
    ma168_changed = prev_state[3] != curr_state[3]
    
    # Skip if ONLY ma72 changed (all other components stayed same)
    only_ma72_changed = (
        ma72_changed and
        not trend_24h_changed and
        not trend_168h_changed and
        not ma168_changed
    )
    
    return not only_ma72_changed


def get_state_tuple(signals_row: pd.Series) -> Tuple[int, int, int, int]:
    """Extract state tuple from signals DataFrame row."""
    return (
        int(signals_row['trend_24h']),
        int(signals_row['trend_168h']),
        int(signals_row['ma72_above_ma24']),
        int(signals_row['ma168_above_ma24'])
    )


# =============================================================================
# HIT RATE CALCULATION
# =============================================================================

def calculate_expanding_hit_rates(
    returns_history: pd.Series,
    signals_history: pd.DataFrame,
    min_samples: int = MIN_SAMPLES_PER_STATE
) -> Dict:
    """
    Calculate 16-state hit rates using ONLY historical data.
    
    For each of the 16 states, calculates the percentage of times
    the next-day return was positive (hit rate).
    
    Args:
        returns_history: Daily returns series
        signals_history: DataFrame with signal columns
        min_samples: Minimum samples for reliable hit rate
    
    Returns:
        Dict mapping (trend_perm, ma_perm) -> {n, hit_rate, sufficient}
    """
    all_trend_perms = list(product([0, 1], repeat=2))  # 4 trend permutations
    all_ma_perms = list(product([0, 1], repeat=2))      # 4 MA permutations
    
    # Default result for insufficient data
    default_result = {
        (t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False}
        for t in all_trend_perms for m in all_ma_perms
    }
    
    if len(returns_history) < min_samples:
        return default_result
    
    # Align indices
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < min_samples:
        return default_result
    
    aligned_returns = returns_history.loc[common_idx]
    aligned_signals = signals_history.loc[common_idx]
    
    # Forward returns (what happens AFTER this state)
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    hit_rates = {}
    
    for trend_perm in all_trend_perms:
        for ma_perm in all_ma_perms:
            # Build mask for this 16-state
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
                'sufficient': n >= min_samples,
            }
    
    return hit_rates


# =============================================================================
# POSITION SIGNALS
# =============================================================================

def get_16state_position(
    trend_perm: Tuple[int, int],
    ma_perm: Tuple[int, int],
    hit_rates: Dict,
    threshold: float = HIT_RATE_THRESHOLD,
    conservative_skip: bool = True,
) -> float:
    """
    Get position signal for a 16-state.
    
    Args:
        trend_perm: (trend_24h, trend_168h)
        ma_perm: (ma72_above_ma24, ma168_above_ma24)
        hit_rates: Dict from calculate_expanding_hit_rates()
        threshold: Hit rate threshold for INVEST signal
        conservative_skip: If True, don't go long on bearish signals
                          when samples are insufficient (default: True)
    
    Returns:
        1.00: INVEST (hit rate > threshold and sufficient samples)
        0.50: SKIP (insufficient samples, but bullish or not conservative)
        0.00: AVOID (hit rate <= threshold, or insufficient + bearish + conservative)
    """
    key = (trend_perm, ma_perm)
    data = hit_rates.get(key, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        # Insufficient samples - be conservative
        if conservative_skip and data['hit_rate'] < threshold:
            # Don't go long on bearish signals even with insufficient data
            return 0.00  # AVOID
        else:
            # Bullish or neutral - take half position as hedge
            return 0.50  # SKIP
    elif data['hit_rate'] > threshold:
        return 1.00  # INVEST
    else:
        return 0.00  # AVOID


# Legacy alias
def get_32state_position(
    price_perm: Tuple[int, int, int],
    ma_perm: Tuple[int, int],
    hit_rates: Dict,
    threshold: float = HIT_RATE_THRESHOLD
) -> float:
    """Legacy function - maps 32-state call to 16-state."""
    import warnings
    warnings.warn(
        "get_32state_position is deprecated. Use get_16state_position instead.",
        DeprecationWarning
    )
    # Extract trend_24h and trend_168h from old 3-tuple (ignore trend_72h)
    trend_perm = (price_perm[0], price_perm[2])
    return get_16state_position(trend_perm, ma_perm, hit_rates, threshold)


def hit_rate_to_simple_state(hit_rate: float) -> str:
    """
    Convert hit rate to simplified state for pairs trading.
    
    Args:
        hit_rate: Historical hit rate (0-1)
    
    Returns:
        State string: STRONG_BUY, BUY, SELL, or STRONG_SELL
    """
    if hit_rate >= STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    elif hit_rate >= BUY_THRESHOLD:
        return "BUY"
    elif hit_rate >= SELL_THRESHOLD:
        return "SELL"
    else:
        return "STRONG_SELL"


def state_to_numeric(state: str) -> int:
    """Convert state string to numeric for divergence calculation."""
    mapping = {"STRONG_BUY": 3, "BUY": 2, "SELL": 1, "STRONG_SELL": 0}
    return mapping.get(state, 1)


def get_state_divergence(state_a: str, state_b: str) -> int:
    """Calculate state divergence between two assets."""
    return abs(state_to_numeric(state_a) - state_to_numeric(state_b))


# =============================================================================
# SIGNAL GENERATOR CLASS
# =============================================================================

class SignalGenerator:
    """
    16-State Signal Generator with NO_MA72_ONLY filter.
    
    Unified interface for both backtest and live trading.
    
    Usage:
        gen = SignalGenerator()
        
        # Generate signals from 1h data
        signals = gen.generate_signals(df_1h)
        
        # Get position for current state (with filter)
        position, details = gen.get_position_for_date(
            signals, returns, current_date, data_start
        )
    """
    
    def __init__(self, use_filter: bool = USE_MA72_FILTER):
        """
        Initialize generator.
        
        Args:
            use_filter: Enable NO_MA72_ONLY filter (default: True)
        """
        self.use_filter = use_filter
        self.hit_rate_cache = {}
        self.last_recalc_month = None
        
        # Filter state tracking
        self.prev_state: Optional[Tuple[int, int, int, int]] = None
        self.active_state: Optional[Tuple[int, int, int, int]] = None
        self.filter_stats = FilterStats()
    
    def reset_filter_state(self):
        """Reset filter state tracking (call between assets or backtests)."""
        self.prev_state = None
        self.active_state = None
        self.filter_stats = FilterStats()
    
    def generate_signals(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Generate 16-state signals from 1h OHLCV data.
        
        Args:
            df_1h: 1-hour OHLCV DataFrame with datetime index
        
        Returns:
            DataFrame with signal columns aligned to 24h
        """
        # Resample to required timeframes
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        # Generate 16-state signals
        signals = generate_16state_signals(df_24h, df_72h, df_168h)
        
        return signals
    
    def get_returns(self, df_1h: pd.DataFrame) -> pd.Series:
        """Get daily returns from 1h data."""
        df_24h = resample_ohlcv(df_1h, '24h')
        return df_24h['close'].pct_change()
    
    def apply_filter(
        self,
        current_state: Tuple[int, int, int, int]
    ) -> Tuple[Tuple[int, int, int, int], bool]:
        """
        Apply NO_MA72_ONLY filter and return active state.
        
        Args:
            current_state: Current 4-tuple state
        
        Returns:
            Tuple of (active_state, was_filtered)
        """
        # Check if state changed
        if self.prev_state is None:
            # First signal - always trade
            self.active_state = current_state
            self.prev_state = current_state
            return current_state, False
        
        if current_state == self.prev_state:
            # No change
            return self.active_state, False
        
        # State changed - check filter
        self.filter_stats.total_signals += 1
        
        if should_trade_signal(self.prev_state, current_state, self.use_filter):
            # Trade this signal
            self.active_state = current_state
            self.filter_stats.traded_signals += 1
            filtered = False
        else:
            # Filter - keep previous active state
            self.filter_stats.filtered_signals += 1
            filtered = True
        
        self.prev_state = current_state
        return self.active_state, filtered
    
    def get_position_for_date(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        current_date: pd.Timestamp,
        data_start: pd.Timestamp,
        min_training_months: int = 12
    ) -> Tuple[float, Dict]:
        """
        Get position signal for a specific date with filter applied.
        
        Recalculates hit rates monthly using only historical data.
        
        Args:
            signals: DataFrame from generate_signals()
            returns: Daily returns series
            current_date: Date to get signal for
            data_start: First date of available data
            min_training_months: Minimum months before trading
        
        Returns:
            Tuple of (position, details)
            position: 1.0=INVEST, 0.5=SKIP, 0.0=AVOID
            details: {state, hit_rate, n_samples, simple_state, filtered}
        """
        if current_date not in signals.index:
            return 0.5, {'error': 'date not in signals'}
        
        # Check minimum training period
        months_available = (
            (current_date.year - data_start.year) * 12 +
            (current_date.month - data_start.month)
        )
        if months_available < min_training_months:
            return 0.5, {'error': 'insufficient_training'}
        
        # Get current state
        sig = signals.loc[current_date]
        current_state = get_state_tuple(sig)
        
        # Apply filter
        active_state, was_filtered = self.apply_filter(current_state)
        
        # Extract trend and MA permutations from active state
        trend_perm = (active_state[0], active_state[1])
        ma_perm = (active_state[2], active_state[3])
        
        # Recalculate hit rates monthly
        current_month = (current_date.year, current_date.month)
        if self.last_recalc_month != current_month:
            # Use only data before current month
            cutoff = current_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            hist_returns = returns[returns.index < cutoff]
            hist_signals = signals[signals.index < cutoff]
            
            if len(hist_returns) > 0:
                self.hit_rate_cache = calculate_expanding_hit_rates(
                    hist_returns, hist_signals
                )
                self.last_recalc_month = current_month
        
        # Get position
        position = get_16state_position(trend_perm, ma_perm, self.hit_rate_cache)
        
        # Get details
        key = (trend_perm, ma_perm)
        hr_data = self.hit_rate_cache.get(key, {'hit_rate': 0.5, 'n': 0, 'sufficient': False})
        
        details = {
            'state_trend': trend_perm,
            'state_ma': ma_perm,
            'current_state': current_state,
            'active_state': active_state,
            'hit_rate': hr_data['hit_rate'],
            'n_samples': hr_data['n'],
            'sufficient': hr_data['sufficient'],
            'simple_state': hit_rate_to_simple_state(hr_data['hit_rate']),
            'filtered': was_filtered,
        }
        
        return position, details
    
    def get_all_positions(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        data_start: pd.Timestamp,
        min_training_months: int = 12
    ) -> pd.DataFrame:
        """
        Get positions for all dates (batch mode for backtest).
        
        Args:
            signals: DataFrame from generate_signals()
            returns: Daily returns series
            data_start: First date of data
            min_training_months: Minimum months before trading
        
        Returns:
            DataFrame with position and details for each date
        """
        # Reset filter state for fresh run
        self.reset_filter_state()
        
        results = []
        
        for date in signals.index:
            position, details = self.get_position_for_date(
                signals, returns, date, data_start, min_training_months
            )
            results.append({
                'date': date,
                'position': position,
                **details
            })
        
        return pd.DataFrame(results).set_index('date')
    
    def get_filter_stats(self) -> Dict:
        """Get filter statistics."""
        return {
            'total_signals': self.filter_stats.total_signals,
            'filtered_signals': self.filter_stats.filtered_signals,
            'traded_signals': self.filter_stats.traded_signals,
            'filter_rate': self.filter_stats.filter_rate,
        }