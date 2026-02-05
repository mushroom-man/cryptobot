# -*- coding: utf-8 -*-
"""
CryptoBot - Backtest Executor
==============================
Simulates order execution with realistic slippage and transaction costs.

Implements the Executor protocol from shared.core.engine.

Usage:
    from cryptobot.research.backtest import SimulatedExecutor
    
    executor = SimulatedExecutor(
        slippage_bps=10,    # 0.10% slippage
        commission_bps=10,  # 0.10% commission
    )
    
    fill = executor.execute(order, bar)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np

from cryptobot.shared.core.bar import Bar
from cryptobot.shared.core.order import Order, OrderSide, OrderType
from cryptobot.shared.core.fill import Fill


class SlippageModel(Enum):
    """Slippage model types."""
    FIXED = "fixed"              # Fixed basis points
    PROPORTIONAL = "proportional"  # Proportional to order size
    VOLATILITY = "volatility"    # Based on current volatility


@dataclass
class ExecutionConfig:
    """Configuration for simulated execution."""
    
    # Slippage settings
    slippage_bps: float = 10.0           # Base slippage in basis points
    slippage_model: SlippageModel = SlippageModel.FIXED
    
    # Commission/fees
    commission_bps: float = 10.0          # Commission in basis points
    min_commission: float = 0.0           # Minimum commission per trade
    
    # Execution assumptions
    fill_at: str = "close"               # "close", "open", "mid", "worst"
    partial_fills: bool = False          # Allow partial fills
    fill_probability: float = 1.0        # Probability of fill (1.0 = always)
    
    # Market impact (for large orders)
    impact_factor: float = 0.0           # Market impact as % of order size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'slippage_bps': self.slippage_bps,
            'slippage_model': self.slippage_model.value,
            'commission_bps': self.commission_bps,
            'min_commission': self.min_commission,
            'fill_at': self.fill_at,
            'partial_fills': self.partial_fills,
            'fill_probability': self.fill_probability,
            'impact_factor': self.impact_factor,
        }


class SimulatedExecutor:
    """
    Simulates order execution for backtesting.
    
    Implements the Executor protocol:
        execute(order, bar) -> Fill
    
    Features:
        - Configurable slippage models
        - Transaction costs
        - Market impact simulation
        - Execution price options (close, open, mid, worst)
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None, **kwargs):
        """
        Initialize simulated executor.
        
        Args:
            config: Execution configuration
            **kwargs: Override config parameters directly
                slippage_bps: Slippage in basis points
                commission_bps: Commission in basis points
        """
        self.config = config or ExecutionConfig()
        
        # Allow direct parameter overrides
        if 'slippage_bps' in kwargs:
            self.config.slippage_bps = kwargs['slippage_bps']
        if 'commission_bps' in kwargs:
            self.config.commission_bps = kwargs['commission_bps']
        
        # Statistics tracking
        self.total_orders = 0
        self.total_fills = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0
    
    def execute(self, order: Order, bar: Bar) -> Fill:
        """
        Execute an order against a bar.
        
        Implements Executor protocol.
        
        Args:
            order: Order to execute
            bar: Current bar data
        
        Returns:
            Fill with execution details
        """
        self.total_orders += 1
        
        # Check fill probability
        if self.config.fill_probability < 1.0:
            if np.random.random() > self.config.fill_probability:
                # Order not filled - return zero fill
                return Fill(
                    order=order,
                    fill_price=0.0,
                    fill_size=0.0,
                    timestamp=bar.timestamp,
                    commission=0.0,
                    slippage=0.0,
                )
        
        # Get base execution price
        base_price = self._get_base_price(bar)
        
        # Apply slippage
        slippage = self._calculate_slippage(order, bar, base_price)
        
        # Calculate fill price (adverse to trader)
        if order.is_buy:
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage
        
        # Ensure fill price within bar range (realistic)
        fill_price = np.clip(fill_price, bar.low, bar.high)
        
        # Calculate fill size
        if self.config.partial_fills:
            fill_size = self._calculate_partial_fill(order, bar)
        else:
            fill_size = order.size
        
        # Calculate commission
        notional = fill_size * fill_price
        commission = self._calculate_commission(notional)
        
        # Track statistics
        self.total_fills += 1
        self.total_slippage += abs(fill_price - base_price) * fill_size
        self.total_commission += commission
        
        # Create fill
        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_size=fill_size,
            timestamp=bar.timestamp,
            commission=commission,
            slippage=abs(fill_price - base_price) * fill_size,
        )
        
        return fill
    
    def _get_base_price(self, bar: Bar) -> float:
        """Get base execution price from bar."""
        if self.config.fill_at == "open":
            return bar.open
        elif self.config.fill_at == "close":
            return bar.close
        elif self.config.fill_at == "mid":
            return (bar.high + bar.low) / 2
        elif self.config.fill_at == "worst":
            # Worst price for the trader
            # For buys: high, for sells: low
            # But we don't know direction here, so use mid
            return (bar.high + bar.low) / 2
        elif self.config.fill_at == "vwap":
            # Approximate VWAP as typical price
            return (bar.high + bar.low + bar.close) / 3
        else:
            return bar.close
    
    def _calculate_slippage(self, order: Order, bar: Bar, base_price: float) -> float:
        """Calculate slippage amount."""
        if self.config.slippage_model == SlippageModel.FIXED:
            # Fixed basis points
            slippage = base_price * self.config.slippage_bps / 10000
        
        elif self.config.slippage_model == SlippageModel.PROPORTIONAL:
            # Slippage proportional to order size relative to volume
            if bar.volume > 0:
                order_pct = (order.size * base_price) / (bar.volume * base_price)
                slippage = base_price * self.config.slippage_bps / 10000 * (1 + order_pct)
            else:
                slippage = base_price * self.config.slippage_bps / 10000
        
        elif self.config.slippage_model == SlippageModel.VOLATILITY:
            # Slippage based on bar volatility
            bar_vol = (bar.high - bar.low) / base_price
            slippage = base_price * bar_vol * self.config.slippage_bps / 100
        
        else:
            slippage = base_price * self.config.slippage_bps / 10000
        
        # Add market impact for large orders
        if self.config.impact_factor > 0:
            impact = order.size * base_price * self.config.impact_factor
            slippage += impact
        
        return slippage
    
    def _calculate_partial_fill(self, order: Order, bar: Bar) -> float:
        """Calculate partial fill size based on volume."""
        if bar.volume <= 0:
            return order.size
        
        # Assume we can fill up to 10% of bar volume
        max_fill = bar.volume * 0.1
        
        return min(order.size, max_fill)
    
    def _calculate_commission(self, notional: float) -> float:
        """Calculate commission/fees."""
        commission = notional * self.config.commission_bps / 10000
        
        return max(commission, self.config.min_commission)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        fill_rate = self.total_fills / self.total_orders if self.total_orders > 0 else 0
        avg_slippage = self.total_slippage / self.total_fills if self.total_fills > 0 else 0
        avg_commission = self.total_commission / self.total_fills if self.total_fills > 0 else 0
        
        return {
            'total_orders': self.total_orders,
            'total_fills': self.total_fills,
            'fill_rate': fill_rate,
            'total_slippage': self.total_slippage,
            'total_commission': self.total_commission,
            'avg_slippage_per_fill': avg_slippage,
            'avg_commission_per_fill': avg_commission,
            'total_costs': self.total_slippage + self.total_commission,
        }
    
    def reset_statistics(self):
        """Reset execution statistics."""
        self.total_orders = 0
        self.total_fills = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0
    
    def __repr__(self) -> str:
        return (
            f"SimulatedExecutor(slippage={self.config.slippage_bps}bps, "
            f"commission={self.config.commission_bps}bps)"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_executor(
    slippage_bps: float = 10.0,
    commission_bps: float = 10.0,
    fill_at: str = "close",
) -> SimulatedExecutor:
    """Create executor with custom settings."""
    config = ExecutionConfig(
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        fill_at=fill_at,
    )
    return SimulatedExecutor(config=config)


def create_realistic_executor() -> SimulatedExecutor:
    """Create executor with realistic Kraken-like costs."""
    config = ExecutionConfig(
        slippage_bps=5.0,           # Crypto typically has tight spreads
        commission_bps=26.0,         # Kraken taker fee ~0.26%
        fill_at="close",
    )
    return SimulatedExecutor(config=config)


def create_pessimistic_executor() -> SimulatedExecutor:
    """Create executor with pessimistic (high) costs."""
    config = ExecutionConfig(
        slippage_bps=20.0,
        commission_bps=30.0,
        slippage_model=SlippageModel.PROPORTIONAL,
        fill_at="close",
    )
    return SimulatedExecutor(config=config)


def create_zero_cost_executor() -> SimulatedExecutor:
    """Create executor with no costs (for testing)."""
    config = ExecutionConfig(
        slippage_bps=0.0,
        commission_bps=0.0,
    )
    return SimulatedExecutor(config=config)
