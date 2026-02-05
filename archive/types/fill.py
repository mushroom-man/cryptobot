# -*- coding: utf-8 -*-
"""
CryptoBot - Fill Data Structure
================================
Represents an order execution (fill).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

from cryptobot.types.order import Order, OrderStatus


@dataclass
class Fill:
    """
    Order fill/execution representation.
    
    Attributes:
        order: Original order that was filled
        fill_price: Actual execution price
        fill_size: Size filled (may be partial)
        timestamp: Fill timestamp
        commission: Trading commission/fee
        slippage: Price slippage from reference
    """
    order: Order
    fill_price: float
    fill_size: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    
    # Auto-generated
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    @classmethod
    def from_order(cls, order: Order, fill_price: float, 
                   commission: float = 0.0,
                   timestamp: datetime = None) -> 'Fill':
        """
        Create fill from order (full fill at given price).
        
        Args:
            order: Order being filled
            fill_price: Execution price
            commission: Trading commission
            timestamp: Fill timestamp (defaults to order timestamp)
        """
        # Calculate slippage if reference price available
        slippage = 0.0
        if order.reference_price:
            if order.is_buy:
                slippage = fill_price - order.reference_price
            else:
                slippage = order.reference_price - fill_price
        
        return cls(
            order=order,
            fill_price=fill_price,
            fill_size=order.size,
            timestamp=timestamp or order.timestamp,
            commission=commission,
            slippage=slippage,
        )
    
    @classmethod
    def simulated(cls, order: Order, market_price: float,
                  slippage_bps: float = 10.0,
                  commission_bps: float = 10.0) -> 'Fill':
        """
        Create simulated fill with slippage and commission.
        
        Args:
            order: Order being filled
            market_price: Current market price
            slippage_bps: Slippage in basis points (10 = 0.1%)
            commission_bps: Commission in basis points
        """
        # Apply slippage (adverse for trader)
        slippage_pct = slippage_bps / 10000
        if order.is_buy:
            fill_price = market_price * (1 + slippage_pct)
        else:
            fill_price = market_price * (1 - slippage_pct)
        
        # Calculate commission
        notional = order.size * fill_price
        commission = notional * commission_bps / 10000
        
        # Calculate total slippage
        slippage = abs(fill_price - market_price) * order.size
        
        return cls(
            order=order,
            fill_price=fill_price,
            fill_size=order.size,
            timestamp=order.timestamp,
            commission=commission,
            slippage=slippage,
        )
    
    @property
    def pair(self) -> str:
        """Trading pair."""
        return self.order.pair
    
    @property
    def side(self):
        """Order side."""
        return self.order.side
    
    @property
    def is_buy(self) -> bool:
        """True if buy fill."""
        return self.order.is_buy
    
    @property
    def is_sell(self) -> bool:
        """True if sell fill."""
        return self.order.is_sell
    
    @property
    def signed_size(self) -> float:
        """Size with sign (positive for buy, negative for sell)."""
        return self.fill_size if self.is_buy else -self.fill_size
    
    @property
    def notional_value(self) -> float:
        """Total notional value of fill."""
        return self.fill_size * self.fill_price
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        return self.commission + self.slippage
    
    @property
    def is_complete(self) -> bool:
        """True if order fully filled."""
        return abs(self.fill_size - self.order.size) < 1e-10
    
    def update_order_status(self):
        """Update the order's status based on fill."""
        if self.is_complete:
            self.order.status = OrderStatus.FILLED
        else:
            self.order.status = OrderStatus.PARTIALLY_FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order.order_id,
            'pair': self.pair,
            'side': self.side.value,
            'fill_price': self.fill_price,
            'fill_size': self.fill_size,
            'notional_value': self.notional_value,
            'commission': self.commission,
            'slippage': self.slippage,
            'total_cost': self.total_cost,
            'timestamp': self.timestamp,
        }
    
    def __repr__(self) -> str:
        return (
            f"Fill({self.fill_id}, {self.pair}, {self.side.value}, "
            f"size={self.fill_size:.6f} @ {self.fill_price:.2f}, "
            f"cost={self.total_cost:.2f})"
        )
