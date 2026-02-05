# -*- coding: utf-8 -*-
"""
CryptoBot - Stop Loss Management
=================================
Multiple stop loss strategies from the strategy document.

Stop Loss Methods:
    - ATR-based: entry - (ATR × multiplier)
    - GARCH-based: entry - (GARCH_vol × multiplier × entry)
    - Signal-based: Exit if P(increase) < threshold
    - Combined: Most conservative of above, capped at max %

Usage:
    from cryptobot.shared.risk import StopLossManager
    
    stops = StopLossManager(
        atr_multiplier=2.0,
        garch_multiplier=2.5,
        signal_threshold=0.35,
        max_loss_pct=0.03,
    )
    
    # Check stops
    triggered = stops.check_stops(portfolio, bar, features, prediction)
    for pair in triggered:
        # Close position
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class StopType(Enum):
    """Type of stop loss triggered."""
    ATR = "atr"
    GARCH = "garch"
    SIGNAL = "signal"
    MAX_LOSS = "max_loss"
    TRAILING = "trailing"


@dataclass
class StopLevel:
    """Stop loss level for a position."""
    pair: str
    stop_price: float
    stop_type: StopType
    entry_price: float
    current_price: float
    loss_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pair': self.pair,
            'stop_price': self.stop_price,
            'stop_type': self.stop_type.value,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'loss_pct': self.loss_pct,
        }


@dataclass
class StopTriggered:
    """Record of a triggered stop."""
    pair: str
    stop_type: StopType
    trigger_price: float
    stop_price: float
    loss_pct: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pair': self.pair,
            'stop_type': self.stop_type.value,
            'trigger_price': self.trigger_price,
            'stop_price': self.stop_price,
            'loss_pct': self.loss_pct,
            'timestamp': self.timestamp,
        }


class StopLossManager:
    """
    Manages stop losses for all positions.
    
    Supports multiple stop loss methods:
        - ATR-based: Volatility-adjusted stops
        - GARCH-based: Forecast volatility stops
        - Signal-based: Exit on weak predictions
        - Max loss: Hard cap on position loss
        - Trailing: Trail stop behind price
    
    Uses the most conservative (tightest) stop by default.
    """
    
    def __init__(
        self,
        atr_multiplier: float = 2.0,
        garch_multiplier: float = 2.5,
        signal_threshold: float = 0.35,
        max_loss_pct: float = 0.03,
        use_trailing: bool = False,
        trailing_pct: float = 0.02,
        use_combined: bool = True,
    ):
        """
        Initialize stop loss manager.
        
        Args:
            atr_multiplier: ATR multiplier for stop distance
            garch_multiplier: GARCH vol multiplier for stop distance
            signal_threshold: Exit if prediction below this
            max_loss_pct: Maximum loss per position (0.03 = 3%)
            use_trailing: Enable trailing stops
            trailing_pct: Trailing stop distance as percentage
            use_combined: Use most conservative stop across methods
        """
        self.atr_multiplier = atr_multiplier
        self.garch_multiplier = garch_multiplier
        self.signal_threshold = signal_threshold
        self.max_loss_pct = max_loss_pct
        self.use_trailing = use_trailing
        self.trailing_pct = trailing_pct
        self.use_combined = use_combined
        
        # Track trailing stops (pair -> highest price since entry)
        self.trailing_peaks: Dict[str, float] = {}
        
        # Track entry prices if not available from portfolio
        self.entry_prices: Dict[str, float] = {}
        
        # History of triggered stops
        self.triggered_history: List[StopTriggered] = []
    
    def update_entry(self, pair: str, entry_price: float):
        """Record entry price for a position."""
        self.entry_prices[pair] = entry_price
        self.trailing_peaks[pair] = entry_price
    
    def update_trailing(self, pair: str, current_price: float, position_size: float):
        """Update trailing stop peak price."""
        if pair not in self.trailing_peaks:
            self.trailing_peaks[pair] = current_price
            return
        
        # Update peak based on position direction
        if position_size > 0:  # Long
            if current_price > self.trailing_peaks[pair]:
                self.trailing_peaks[pair] = current_price
        else:  # Short
            if current_price < self.trailing_peaks[pair]:
                self.trailing_peaks[pair] = current_price
    
    def calculate_stop_levels(
        self,
        pair: str,
        entry_price: float,
        current_price: float,
        position_size: float,
        atr: Optional[float] = None,
        garch_vol: Optional[float] = None,
    ) -> Dict[StopType, StopLevel]:
        """
        Calculate all stop levels for a position.
        
        Args:
            pair: Trading pair
            entry_price: Average entry price
            current_price: Current market price
            position_size: Position size (signed)
            atr: Current ATR value
            garch_vol: Current GARCH volatility (annualized)
        
        Returns:
            Dictionary of stop levels by type
        """
        stops = {}
        is_long = position_size > 0
        
        # ATR-based stop
        if atr is not None and atr > 0:
            if is_long:
                stop_price = entry_price - (atr * self.atr_multiplier)
            else:
                stop_price = entry_price + (atr * self.atr_multiplier)
            
            loss_pct = abs(stop_price - entry_price) / entry_price
            stops[StopType.ATR] = StopLevel(
                pair=pair,
                stop_price=stop_price,
                stop_type=StopType.ATR,
                entry_price=entry_price,
                current_price=current_price,
                loss_pct=loss_pct,
            )
        
        # GARCH-based stop
        if garch_vol is not None and garch_vol > 0:
            # Convert annualized vol to hourly
            hourly_vol = garch_vol / (8760 ** 0.5)
            stop_distance = hourly_vol * self.garch_multiplier * entry_price
            
            if is_long:
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
            
            loss_pct = abs(stop_price - entry_price) / entry_price
            stops[StopType.GARCH] = StopLevel(
                pair=pair,
                stop_price=stop_price,
                stop_type=StopType.GARCH,
                entry_price=entry_price,
                current_price=current_price,
                loss_pct=loss_pct,
            )
        
        # Max loss stop
        if is_long:
            stop_price = entry_price * (1 - self.max_loss_pct)
        else:
            stop_price = entry_price * (1 + self.max_loss_pct)
        
        stops[StopType.MAX_LOSS] = StopLevel(
            pair=pair,
            stop_price=stop_price,
            stop_type=StopType.MAX_LOSS,
            entry_price=entry_price,
            current_price=current_price,
            loss_pct=self.max_loss_pct,
        )
        
        # Trailing stop
        if self.use_trailing and pair in self.trailing_peaks:
            peak = self.trailing_peaks[pair]
            
            if is_long:
                stop_price = peak * (1 - self.trailing_pct)
            else:
                stop_price = peak * (1 + self.trailing_pct)
            
            loss_pct = abs(stop_price - peak) / peak
            stops[StopType.TRAILING] = StopLevel(
                pair=pair,
                stop_price=stop_price,
                stop_type=StopType.TRAILING,
                entry_price=entry_price,
                current_price=current_price,
                loss_pct=loss_pct,
            )
        
        return stops
    
    def get_effective_stop(
        self,
        stops: Dict[StopType, StopLevel],
        is_long: bool,
    ) -> Optional[StopLevel]:
        """
        Get the most conservative (tightest) stop.
        
        For longs: highest stop price
        For shorts: lowest stop price
        """
        if not stops:
            return None
        
        if self.use_combined:
            if is_long:
                # Highest stop price for longs
                return max(stops.values(), key=lambda s: s.stop_price)
            else:
                # Lowest stop price for shorts
                return min(stops.values(), key=lambda s: s.stop_price)
        else:
            # Return max loss stop as default
            return stops.get(StopType.MAX_LOSS)
    
    def check_price_stop(
        self,
        pair: str,
        entry_price: float,
        current_price: float,
        position_size: float,
        atr: Optional[float] = None,
        garch_vol: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[StopTriggered]:
        """
        Check if price-based stop is triggered.
        
        Returns:
            StopTriggered if stop hit, None otherwise
        """
        if abs(position_size) < 1e-10:
            return None
        
        is_long = position_size > 0
        
        # Update trailing
        if self.use_trailing:
            self.update_trailing(pair, current_price, position_size)
        
        # Calculate stops
        stops = self.calculate_stop_levels(
            pair=pair,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            atr=atr,
            garch_vol=garch_vol,
        )
        
        # Get effective stop
        effective = self.get_effective_stop(stops, is_long)
        if effective is None:
            return None
        
        # Check if triggered
        triggered = False
        if is_long and current_price <= effective.stop_price:
            triggered = True
        elif not is_long and current_price >= effective.stop_price:
            triggered = True
        
        if triggered:
            loss_pct = abs(current_price - entry_price) / entry_price
            if not is_long:
                loss_pct = -loss_pct if current_price > entry_price else loss_pct
            
            result = StopTriggered(
                pair=pair,
                stop_type=effective.stop_type,
                trigger_price=current_price,
                stop_price=effective.stop_price,
                loss_pct=loss_pct,
                timestamp=timestamp or datetime.now(),
            )
            
            self.triggered_history.append(result)
            return result
        
        return None
    
    def check_signal_stop(
        self,
        pair: str,
        prediction: float,
        position_size: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[StopTriggered]:
        """
        Check if signal-based stop is triggered.
        
        Exit long if P(up) < threshold
        Exit short if P(up) > (1 - threshold)
        
        Returns:
            StopTriggered if stop hit, None otherwise
        """
        if abs(position_size) < 1e-10:
            return None
        
        is_long = position_size > 0
        triggered = False
        
        if is_long and prediction < self.signal_threshold:
            triggered = True
        elif not is_long and prediction > (1 - self.signal_threshold):
            triggered = True
        
        if triggered:
            result = StopTriggered(
                pair=pair,
                stop_type=StopType.SIGNAL,
                trigger_price=0.0,  # Not price-based
                stop_price=0.0,
                loss_pct=0.0,  # Unknown at this point
                timestamp=timestamp or datetime.now(),
            )
            
            self.triggered_history.append(result)
            return result
        
        return None
    
    def check_stops(
        self,
        pair: str,
        entry_price: float,
        current_price: float,
        position_size: float,
        prediction: Optional[float] = None,
        atr: Optional[float] = None,
        garch_vol: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[StopTriggered]:
        """
        Check all stops for a position.
        
        Returns first triggered stop, or None.
        """
        # Check signal stop first (often most responsive)
        if prediction is not None:
            signal_stop = self.check_signal_stop(
                pair, prediction, position_size, timestamp
            )
            if signal_stop:
                return signal_stop
        
        # Check price-based stops
        price_stop = self.check_price_stop(
            pair=pair,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            atr=atr,
            garch_vol=garch_vol,
            timestamp=timestamp,
        )
        
        return price_stop
    
    def clear_position(self, pair: str):
        """Clear tracking for a closed position."""
        self.trailing_peaks.pop(pair, None)
        self.entry_prices.pop(pair, None)
    
    def get_triggered_history(self) -> List[Dict[str, Any]]:
        """Get history of triggered stops."""
        return [t.to_dict() for t in self.triggered_history]
    
    def __repr__(self) -> str:
        return (
            f"StopLossManager(atr={self.atr_multiplier}x, "
            f"garch={self.garch_multiplier}x, "
            f"signal<{self.signal_threshold}, "
            f"max={self.max_loss_pct:.1%})"
        )
