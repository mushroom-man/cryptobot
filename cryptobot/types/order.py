# -*- coding: utf-8 -*-
"""
CryptoBot - Order Data Structure
=================================
Represents a trade order.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import uuid


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Trade order representation.
    
    Attributes:
        pair: Trading pair (e.g., "XBTUSD")
        side: Buy or sell
        size: Order size (positive for buy, negative for sell, or use side)
        order_type: Market or limit
        timestamp: Order creation time
        limit_price: Limit price (for limit orders)
        reason: Why order was created (signal, stop_loss, etc.)
    """
    pair: str
    side: OrderSide
    size: float  # Absolute size (always positive)
    timestamp: datetime
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    reason: str = "signal"
    
    # Auto-generated
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    
    # Reference price at time of order (for slippage calculation)
    reference_price: Optional[float] = None
    
    @classmethod
    def market_buy(cls, pair: str, size: float, timestamp: datetime, 
                   reason: str = "signal", reference_price: float = None) -> 'Order':
        """Create market buy order."""
        return cls(
            pair=pair,
            side=OrderSide.BUY,
            size=abs(size),
            timestamp=timestamp,
            order_type=OrderType.MARKET,
            reason=reason,
            reference_price=reference_price,
        )
    
    @classmethod
    def market_sell(cls, pair: str, size: float, timestamp: datetime,
                    reason: str = "signal", reference_price: float = None) -> 'Order':
        """Create market sell order."""
        return cls(
            pair=pair,
            side=OrderSide.SELL,
            size=abs(size),
            timestamp=timestamp,
            order_type=OrderType.MARKET,
            reason=reason,
            reference_price=reference_price,
        )
    
    @classmethod
    def from_target_position(cls, pair: str, current_position: float, 
                              target_position: float, timestamp: datetime,
                              reference_price: float = None,
                              reason: str = "signal") -> Optional['Order']:
        """
        Create order to move from current to target position.
        
        Args:
            pair: Trading pair
            current_position: Current position size (signed)
            target_position: Target position size (signed)
            timestamp: Order timestamp
            reference_price: Current price for reference
            reason: Order reason
        
        Returns:
            Order if position change needed, None otherwise
        """
        delta = target_position - current_position
        
        if abs(delta) < 1e-10:  # No change needed
            return None
        
        if delta > 0:
            return cls.market_buy(pair, delta, timestamp, reason, reference_price)
        else:
            return cls.market_sell(pair, abs(delta), timestamp, reason, reference_price)
    
    @property
    def signed_size(self) -> float:
        """Size with sign (positive for buy, negative for sell)."""
        return self.size if self.side == OrderSide.BUY else -self.size
    
    @property
    def is_buy(self) -> bool:
        """True if buy order."""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """True if sell order."""
        return self.side == OrderSide.SELL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'pair': self.pair,
            'side': self.side.value,
            'size': self.size,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'reason': self.reason,
            'reference_price': self.reference_price,
        }
    
    def __repr__(self) -> str:
        return (
            f"Order({self.order_id}, {self.pair}, {self.side.value}, "
            f"size={self.size:.6f}, {self.status.value})"
        )
