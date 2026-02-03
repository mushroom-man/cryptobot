# -*- coding: utf-8 -*-
"""
CryptoBot - Live Executor
==========================
Real order execution via Kraken API.

Implements the Executor protocol from engine.py.

Usage:
    from cryptobot.executors import LiveExecutor
    from cryptobot.data.kraken import KrakenAPI
    
    kraken = KrakenAPI()
    executor = LiveExecutor(kraken)
    fill = executor.execute(order, bar)
"""

from datetime import datetime, timezone
import logging
import time

from cryptobot.types.order import Order, OrderSide, OrderStatus
from cryptobot.types.fill import Fill
from cryptobot.types.bar import Bar


logger = logging.getLogger(__name__)


class LiveExecutor:
    """
    Live order executor via Kraken API.
    
    Implements Executor protocol:
        execute(order, bar) -> Fill
    
    Features:
        - Real order placement
        - Order confirmation
        - Retry logic
        - Safety checks
    """
    
    def __init__(
        self,
        kraken_api,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        require_confirmation: bool = True,
    ):
        """
        Initialize live executor.
        
        Args:
            kraken_api: KrakenAPI instance (authenticated)
            max_retries: Max order retries on failure
            retry_delay: Seconds between retries
            require_confirmation: Require user confirmation before first trade
        """
        self.kraken = kraken_api
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.require_confirmation = require_confirmation
        
        # Safety flag
        self._confirmed = False
        
        # Tracking
        self.fill_count = 0
        self.total_volume = 0.0
        self.total_commission = 0.0
    
    def execute(self, order: Order, bar: Bar) -> Fill:
        """
        Execute order via Kraken API.
        
        Implements Executor protocol.
        
        Args:
            order: Order to execute
            bar: Current market bar (for reference price)
        
        Returns:
            Fill from actual execution
        
        Raises:
            RuntimeError: If order fails after retries
        """
        # Safety confirmation
        if self.require_confirmation and not self._confirmed:
            self._request_confirmation()
        
        logger.warning(
            f"🔴 LIVE {order.side.value.upper()} {order.pair}: "
            f"{order.size:.6f} @ ~${bar.close:,.2f}"
        )
        
        # Attempt order with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = self._place_order(order)
                fill = self._create_fill(order, result)
                
                # Track stats
                self.fill_count += 1
                self.total_volume += fill.notional_value
                self.total_commission += fill.commission
                
                logger.info(
                    f"✅ FILLED {order.pair}: {fill.fill_size:.6f} @ ${fill.fill_price:,.2f} "
                    f"(comm: ${fill.commission:.2f})"
                )
                
                return fill
                
            except Exception as e:
                last_error = e
                logger.error(f"Order attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # All retries failed
        order.status = OrderStatus.REJECTED
        raise RuntimeError(f"Order failed after {self.max_retries} attempts: {last_error}")
    
    def _request_confirmation(self):
        """Request user confirmation before first live trade."""
        print("\n" + "=" * 60)
        print("⚠️  LIVE TRADING CONFIRMATION REQUIRED")
        print("=" * 60)
        print("You are about to execute REAL trades with REAL money.")
        print("This action cannot be undone.")
        print()
        
        response = input("Type 'YES' to confirm live trading: ")
        
        if response.strip() != 'YES':
            raise RuntimeError("Live trading not confirmed. Aborting.")
        
        self._confirmed = True
        logger.warning("Live trading confirmed by user")
    
    def _place_order(self, order: Order) -> dict:
        """
        Place order via Kraken API.
        
        Args:
            order: Order to place
        
        Returns:
            Kraken API response dict
        """
        # Map to Kraken order format
        side = 'buy' if order.side == OrderSide.BUY else 'sell'
        
        # Place market order
        result = self.kraken.add_order(
            pair=order.pair,
            type=side,
            ordertype='market',
            volume=str(order.size),
        )
        
        return result
    
    def _create_fill(self, order: Order, result: dict) -> Fill:
        """
        Create Fill from Kraken API response.
        
        Args:
            order: Original order
            result: Kraken API response
        
        Returns:
            Fill object
        """
        # Extract fill details from Kraken response
        # Response structure varies - handle common cases
        
        if 'result' in result:
            result = result['result']
        
        # Get fill price and fee
        # Kraken returns these in the order info or trade info
        fill_price = float(result.get('price', order.reference_price or 0))
        fee = float(result.get('fee', 0))
        fill_size = float(result.get('vol_exec', order.size))
        
        # Calculate slippage if we have reference price
        slippage = 0.0
        if order.reference_price:
            if order.is_buy:
                slippage = (fill_price - order.reference_price) * fill_size
            else:
                slippage = (order.reference_price - fill_price) * fill_size
            slippage = max(0, slippage)  # Only count adverse slippage
        
        return Fill(
            order=order,
            fill_price=fill_price,
            fill_size=fill_size,
            timestamp=datetime.now(timezone.utc),
            commission=fee,
            slippage=slippage,
        )
    
    def get_stats(self) -> dict:
        """Get execution statistics."""
        return {
            'mode': 'live',
            'fill_count': self.fill_count,
            'total_volume': self.total_volume,
            'total_commission': self.total_commission,
            'confirmed': self._confirmed,
        }
    
    def __repr__(self) -> str:
        status = "CONFIRMED" if self._confirmed else "UNCONFIRMED"
        return f"LiveExecutor({status}, fills={self.fill_count})"