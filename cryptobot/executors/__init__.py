# -*- coding: utf-8 -*-
"""
CryptoBot - Executors Package
==============================
Order executors implementing the Executor protocol.

Available Executors:
    - PaperExecutor: Simulated execution (paper trading)
    - LiveExecutor: Real execution via Kraken API

Usage:
    from cryptobot.executors import PaperExecutor, LiveExecutor
    
    # Paper trading
    executor = PaperExecutor(slippage_bps=10, commission_bps=35)
    
    # Live trading
    executor = LiveExecutor(kraken_api)
    
    # Execute order
    fill = executor.execute(order, bar)
"""

from .paper import PaperExecutor
from .live import LiveExecutor

__all__ = [
    'PaperExecutor',
    'LiveExecutor',
]