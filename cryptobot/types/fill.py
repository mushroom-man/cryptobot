# -*- coding: utf-8 -*-
"""CryptoBot - Fill type."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from cryptobot.types.order import Order, OrderSide


@dataclass
class Fill:
    """Represents a filled order."""
    order: Order
    fill_price: float
    fill_size: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def notional_value(self) -> float:
        return abs(self.fill_price * self.fill_size)

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage

    @classmethod
    def simulated(cls, order: Order, market_price: float,
                  slippage_bps: float = 10.0, commission_bps: float = 35.0):
        """Create a simulated fill with modelled slippage and commission."""
        slip_frac = slippage_bps / 10_000
        comm_frac = commission_bps / 10_000

        if order.side == OrderSide.BUY:
            fill_price = market_price * (1 + slip_frac)
        else:
            fill_price = market_price * (1 - slip_frac)

        notional = abs(fill_price * order.size)
        commission = notional * comm_frac
        slippage = abs(fill_price - market_price) * order.size

        return cls(
            order=order,
            fill_price=fill_price,
            fill_size=order.size,
            commission=commission,
            slippage=slippage,
        )
