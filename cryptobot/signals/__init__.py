# -*- coding: utf-8 -*-
"""
CryptoBot - Signals Module
===========================
16-state signal generation with NO_MA72_ONLY filter for directional and pairs trading.

Validated Configuration:
    - 16 states (4 trend × 4 MA alignment)
    - NO_MA72_ONLY filter (56% signal reduction, same alpha)
    - MA periods: 24h=16, 72h=6, 168h=2
    - Buffers: entry=1.5%, exit=0.5%

Usage:
    from cryptobot.signals import SignalGenerator
    
    gen = SignalGenerator()
    signals = gen.generate_signals(df_1h)
    position, details = gen.get_position_for_date(signals, returns, date, data_start)
"""

from cryptobot.signals.generator import (
    # Main class
    SignalGenerator,
    
    # Core functions (16-state)
    resample_ohlcv,
    label_trend_binary,
    generate_16state_signals,
    calculate_expanding_hit_rates,
    get_16state_position,
    
    # Filter function
    should_trade_signal,
    get_state_tuple,
    
    # Pairs trading helpers
    hit_rate_to_simple_state,
    state_to_numeric,
    get_state_divergence,
    
    # Constants
    MA_PERIOD_24H,
    MA_PERIOD_72H,
    MA_PERIOD_168H,
    ENTRY_BUFFER,
    EXIT_BUFFER,
    HIT_RATE_THRESHOLD,
    MIN_SAMPLES_PER_STATE,
    STRONG_BUY_THRESHOLD,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    USE_MA72_FILTER,
    
    # Data structures
    Signal,
    FilterStats,
    
    # Legacy aliases (deprecated)
    generate_32state_signals,
    get_32state_position,
)

__all__ = [
    # Main class
    'SignalGenerator',
    
    # Core functions
    'resample_ohlcv',
    'label_trend_binary',
    'generate_16state_signals',
    'calculate_expanding_hit_rates',
    'get_16state_position',
    
    # Filter
    'should_trade_signal',
    'get_state_tuple',
    
    # Pairs helpers
    'hit_rate_to_simple_state',
    'state_to_numeric',
    'get_state_divergence',
    
    # Data structures
    'Signal',
    'FilterStats',
    
    # Constants
    'MA_PERIOD_24H',
    'MA_PERIOD_72H', 
    'MA_PERIOD_168H',
    'ENTRY_BUFFER',
    'EXIT_BUFFER',
    'USE_MA72_FILTER',
    
    # Legacy (deprecated)
    'generate_32state_signals',
    'get_32state_position',
]