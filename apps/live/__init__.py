# -*- coding: utf-8 -*-
"""
CryptoBot Live Trading Module
==============================
Paper and live trading runners.
Usage:
    python -m apps.live.runner              # Run paper trading
    python -m apps.live.runner --dry-run    # Dry run mode
"""

def __getattr__(name):
    if name == 'TradingRunner':
        from .trader import TradingRunner
        return TradingRunner
    if name == 'load_config':
        from .trader import load_config
        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['TradingRunner', 'load_config']
