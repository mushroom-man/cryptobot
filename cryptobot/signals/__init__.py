# -*- coding: utf-8 -*-
"""
CryptoBot - Signals Module
===========================
32-state signal generation for directional and pairs trading.

Usage:
    from cryptobot.signals import SignalGenerator
    
    gen = SignalGenerator()
    signals = gen.generate_signals(df_1h)
    position, details = gen.get_position_for_date(signals, returns, date, data_start)
"""

from cryptobot.signals.generator import (
    # Main class
    SignalGenerator,
    
    # Core functions (for direct use if needed)
    resample_ohlcv,
    label_trend_binary,
    generate_32state_signals,
    calculate_expanding_hit_rates,
    get_32state_position,
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
    
    # Data structures
    Signal,
)

__all__ = [
    'SignalGenerator',
    'resample_ohlcv',
    'label_trend_binary',
    'generate_32state_signals',
    'calculate_expanding_hit_rates',
    'get_32state_position',
    'hit_rate_to_simple_state',
    'state_to_numeric',
    'get_state_divergence',
    'Signal',
]
