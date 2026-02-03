# -*- coding: utf-8 -*-
"""
CryptoBot - Paper Executor
===========================
Simulated order execution for paper trading.

Implements the Executor protocol from engine.py.

Usage:
    from cryptobot.executors import PaperExecutor
    
    executor = PaperExecutor(slippage_bps=10, commission_bps=35)
    fill = executor.execute(order, bar)
"""

import logging

from cryptobot.types.order import Order
from cryptobot.types.fill import Fill
from cryptobot.types.bar import Bar


logger = logging.getLogger(__name__)


class PaperExecutor:
    """
    Simulated order executor for paper trading.
    
    Implements Executor protocol:
        execute(order, bar) -> Fill
    
    Features:
        - Configurable slippage (adverse price movement)
        - Configurable commission
        - Realistic fill simulation
    """
    
    def __init__(
        self,
        slippage_bps: float = 10.0,
        commission_bps: float = 35.0,
    ):
        """
        Initialize paper executor.
        
        Args:
            slippage_bps: Slippage in basis points (10 = 0.1%)
            commission_bps: Commission in basis points (35 = 0.35%)
        """
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        
        # Tracking
        self.fill_count = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0
    
    def execute(self, order: Order, bar: Bar) -> Fill:
        """
        Execute order with simulated slippage and commission.
        
        Implements Executor protocol.
        
        Args:
            order: Order to execute
            bar: Current market bar (for price)
        
        Returns:
            Fill with simulated execution
        """
        # Use bar close as market price
        market_price = bar.close
        
        # Create simulated fill
        fill = Fill.simulated(
            order=order,
            market_price=market_price,
            slippage_bps=self.slippage_bps,
            commission_bps=self.commission_bps,
        )
        
        # Track stats
        self.fill_count += 1
        self.total_slippage += fill.slippage
        self.total_commission += fill.commission
        
        # Log
        logger.info(
            f"📝 PAPER {order.side.value.upper()} {order.pair}: "
            f"{fill.fill_size:.6f} @ ${fill.fill_price:,.2f} "
            f"(slip: ${fill.slippage:.2f}, comm: ${fill.commission:.2f})"
        )
        
        return fill
    
    def get_stats(self) -> dict:
        """Get execution statistics."""
        return {
            'mode': 'paper',
            'fill_count': self.fill_count,
            'total_slippage': self.total_slippage,
            'total_commission': self.total_commission,
            'total_costs': self.total_slippage + self.total_commission,
            'slippage_bps': self.slippage_bps,
            'commission_bps': self.commission_bps,
        }
    
    def reset_stats(self):
        """Reset execution statistics."""
        self.fill_count = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0
    
    def __repr__(self) -> str:
        return f"PaperExecutor(slippage={self.slippage_bps}bps, comm={self.commission_bps}bps)"