# -*- coding: utf-8 -*-
"""
CryptoBot - Strategies Module
===============================
Trading strategy implementations.

Primary strategy:
    MomentumStrategy — 16-state regime momentum with dynamic hit-rate
    signals, boost logic, quality filter, and long/short support.

Usage:
    from cryptobot.strategies import MomentumStrategy, MomentumConfig, SignalResult

    strategy = MomentumStrategy()
    strategy.init_hit_rates(pair, signals_24h, returns_24h, current_date)
    result = strategy.predict(features)
    # result.multiplier: signed position scalar
    # result.signal_type: LONG / SHORT / BOOSTED_LONG / FLAT
"""

from cryptobot.strategies.momentum import (
    MomentumStrategy,
    MomentumConfig,
    SignalResult,
)

__all__ = [
    'MomentumStrategy',
    'MomentumConfig',
    'SignalResult',
]