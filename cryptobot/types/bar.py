# -*- coding: utf-8 -*-
"""
CryptoBot - Bar Data Structure
===============================
Represents a single OHLCV bar (candlestick).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class Bar:
    """
    Single OHLCV bar representation.
    
    Attributes:
        timestamp: Bar timestamp (UTC)
        pair: Trading pair (e.g., "XBTUSD")
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
    """
    timestamp: datetime
    pair: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Optional: pre-computed features for this bar
    features: Dict[str, float] = field(default_factory=dict)
    
    # Optional: model prediction for this bar
    prediction: Optional[float] = None
    
    @classmethod
    def from_row(cls, row: pd.Series, pair: str = None) -> 'Bar':
        """
        Create Bar from DataFrame row.
        
        Args:
            row: pandas Series with OHLCV data
            pair: Trading pair (if not in row)
        """
        # Handle timestamp - could be index or column
        if isinstance(row.name, (datetime, pd.Timestamp)):
            timestamp = row.name
        elif 'timestamp' in row.index:
            timestamp = row['timestamp']
        else:
            timestamp = datetime.now()
        
        # Ensure timestamp is datetime
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        
        return cls(
            timestamp=timestamp,
            pair=pair or row.get('pair', 'UNKNOWN'),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bar':
        """Create Bar from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            pair=data['pair'],
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data['volume']),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'pair': self.pair,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }
    
    @property
    def mid_price(self) -> float:
        """Mid price (average of high and low)."""
        return (self.high + self.low) / 2
    
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range(self) -> float:
        """Bar range (high - low)."""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Candle body (close - open)."""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open
    
    def __repr__(self) -> str:
        return (
            f"Bar({self.pair}, {self.timestamp}, "
            f"O={self.open:.2f}, H={self.high:.2f}, "
            f"L={self.low:.2f}, C={self.close:.2f})"
        )
