# -*- coding: utf-8 -*-
"""
CryptoBot - Strategies Package
===============================
Trading strategies implementing the Predictor protocol.

Available Strategies:
    - MomentumStrategy: Validated 16-state momentum (default)

Usage:
    from cryptobot.strategies import MomentumStrategy
    
    strategy = MomentumStrategy()
    multiplier = strategy.predict(features)
"""

from .momentum import MomentumStrategy, MomentumConfig, PairState

__all__ = [
    'MomentumStrategy',
    'MomentumConfig',
    'PairState',
]