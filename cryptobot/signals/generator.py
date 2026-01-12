# -*- coding: utf-8 -*-
"""
CryptoBot - 32-State Signal Generator
======================================
Extracted from validated backtest for shared use by backtest and live trading.

32-State System:
    - 8 price states (trend_24h × trend_72h × trend_168h = 2³)
    - 4 MA alignment states (ma72_above_ma24 × ma168_above_ma24 = 2²)
    - Total: 32 unique states

Signal Flow:
    1. Resample 1h OHLCV → 24h/72h/168h
    2. Label trends with hysteresis (binary: bullish/bearish)
    3. Generate 32-state classification
    4. Calculate expanding hit rates per state
    5. Convert hit rate → position signal

Usage:
    from cryptobot.signals.generator import SignalGenerator
    
    gen = SignalGenerator()
    signals = gen.generate_signals(df_1h)
    position = gen.get_position(signals, returns, current_date)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Signal:
    """Signal output for a single timestamp."""
    timestamp: pd.Timestamp
    state_price: Tuple[int, int, int]  # (trend_24h, trend_72h, trend_168h)
    state_ma: Tuple[int, int]          # (ma72_above_ma24, ma168_above_ma24)
    hit_rate: float
    n_samples: int
    sufficient_samples: bool
    position: float                     # 1.0=INVEST, 0.5=SKIP, 0.0=AVOID
    simple_state: str                   # STRONG_BUY/BUY/SELL/STRONG_SELL


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
# 32-STATE SIGNAL GENERATION
# =============================================================================

def generate_32state_signals(
    df_24h: pd.DataFrame,
    df_72h: pd.DataFrame,
    df_168h: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate 32-state signals from multi-timeframe data.
    
    States:
        - 8 price states: trend_24h × trend_72h × trend_168h (2³)
        - 4 MA alignment: ma72_above_ma24 × ma168_above_ma24 (2²)
        - Total: 32 states
    
    CRITICAL: Applies shift(1) BEFORE reindex to prevent look-ahead bias.
    
    Args:
        df_24h: 24-hour OHLCV data
        df_72h: 72-hour OHLCV data
        df_168h: 168-hour OHLCV data
    
    Returns:
        DataFrame with columns:
            - trend_24h, trend_72h, trend_168h (binary)
            - ma72_above_ma24, ma168_above_ma24 (binary)
    """
    # Label trends at each timeframe
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H)
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
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


# =============================================================================
# HIT RATE CALCULATION
# =============================================================================

def calculate_expanding_hit_rates(
    returns_history: pd.Series,
    signals_history: pd.DataFrame,
    min_samples: int = MIN_SAMPLES_PER_STATE
) -> Dict:
    """
    Calculate 32-state hit rates using ONLY historical data.
    
    For each of the 32 states, calculates the percentage of times
    the next-day return was positive (hit rate).
    
    Args:
        returns_history: Daily returns series
        signals_history: DataFrame with signal columns
        min_samples: Minimum samples for reliable hit rate
    
    Returns:
        Dict mapping (price_perm, ma_perm) -> {n, hit_rate, sufficient}
    """
    all_price_perms = list(product([0, 1], repeat=3))
    all_ma_perms = list(product([0, 1], repeat=2))
    
    # Default result for insufficient data
    default_result = {
        (p, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False}
        for p in all_price_perms for m in all_ma_perms
    }
    
    if len(returns_history) < min_samples:
        return default_result
    
    # Align indices
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < min_samples:
        return default_result
    
    aligned_returns = returns_history.loc[common_idx]
    aligned_signals = signals_history.loc[common_idx]
    
    # Forward returns (what we're predicting)
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    hit_rates = {}
    
    for price_perm in all_price_perms:
        for ma_perm in all_ma_perms:
            # Find rows matching this state
            mask = (
                (aligned_signals['trend_24h'] == price_perm[0]) &
                (aligned_signals['trend_72h'] == price_perm[1]) &
                (aligned_signals['trend_168h'] == price_perm[2]) &
                (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
                (aligned_signals['ma168_above_ma24'] == ma_perm[1])
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            
            if n > 0:
                hit_rate = (perm_returns > 0).sum() / n
            else:
                hit_rate = 0.5
            
            hit_rates[(price_perm, ma_perm)] = {
                'n': n,
                'hit_rate': hit_rate,
                'sufficient': n >= min_samples,
            }
    
    return hit_rates


# =============================================================================
# POSITION SIGNALS
# =============================================================================

def get_32state_position(
    price_perm: Tuple[int, int, int],
    ma_perm: Tuple[int, int],
    hit_rates: Dict,
    threshold: float = HIT_RATE_THRESHOLD
) -> float:
    """
    Get position signal for a 32-state.
    
    Args:
        price_perm: (trend_24h, trend_72h, trend_168h)
        ma_perm: (ma72_above_ma24, ma168_above_ma24)
        hit_rates: Dict from calculate_expanding_hit_rates()
        threshold: Hit rate threshold for INVEST signal
    
    Returns:
        1.00: INVEST (hit rate > threshold and sufficient samples)
        0.50: SKIP (insufficient samples)
        0.00: AVOID (hit rate <= threshold)
    """
    key = (price_perm, ma_perm)
    data = hit_rates.get(key, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        return 0.50  # SKIP
    elif data['hit_rate'] > threshold:
        return 1.00  # INVEST
    else:
        return 0.00  # AVOID


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
    32-State Signal Generator.
    
    Unified interface for both backtest and live trading.
    
    Usage:
        gen = SignalGenerator()
        
        # Generate signals from 1h data
        signals = gen.generate_signals(df_1h)
        
        # Get position for current state
        position = gen.get_position_for_date(
            signals, returns, current_date, data_start
        )
    """
    
    def __init__(self):
        self.hit_rate_cache = {}
        self.last_recalc_month = None
    
    def generate_signals(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Generate 32-state signals from 1h OHLCV data.
        
        Args:
            df_1h: 1-hour OHLCV DataFrame with datetime index
        
        Returns:
            DataFrame with signal columns aligned to 24h
        """
        # Resample to required timeframes
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        # Generate 32-state signals
        signals = generate_32state_signals(df_24h, df_72h, df_168h)
        
        return signals
    
    def get_returns(self, df_1h: pd.DataFrame) -> pd.Series:
        """Get daily returns from 1h data."""
        df_24h = resample_ohlcv(df_1h, '24h')
        return df_24h['close'].pct_change()
    
    def get_position_for_date(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        current_date: pd.Timestamp,
        data_start: pd.Timestamp,
        min_training_months: int = 12
    ) -> Tuple[float, Dict]:
        """
        Get position signal for a specific date.
        
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
            details: {state, hit_rate, n_samples, simple_state}
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
        price_perm = (
            int(sig['trend_24h']),
            int(sig['trend_72h']),
            int(sig['trend_168h'])
        )
        ma_perm = (
            int(sig['ma72_above_ma24']),
            int(sig['ma168_above_ma24'])
        )
        
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
        position = get_32state_position(price_perm, ma_perm, self.hit_rate_cache)
        
        # Get details
        key = (price_perm, ma_perm)
        hr_data = self.hit_rate_cache.get(key, {'hit_rate': 0.5, 'n': 0, 'sufficient': False})
        
        details = {
            'state_price': price_perm,
            'state_ma': ma_perm,
            'hit_rate': hr_data['hit_rate'],
            'n_samples': hr_data['n'],
            'sufficient': hr_data['sufficient'],
            'simple_state': hit_rate_to_simple_state(hr_data['hit_rate']),
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
