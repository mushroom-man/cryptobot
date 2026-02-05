# -*- coding: utf-8 -*-
"""
CryptoBot - Base Sizer Protocol
================================
Abstract base class defining the interface for position sizers.

All sizers must implement the `calculate()` method which returns
a target position as a fraction of equity.

Sizer Types:
    - Single-asset sizers: KellySizer, AvoidDangerSizer, FixedSizer
    - Portfolio sizers: RiskParitySizer (allocates across multiple assets)

Usage:
    from cryptobot.risk.base_sizer import BaseSizer
    
    class MySizer(BaseSizer):
        def calculate(self, prediction, features, portfolio, pair) -> float:
            # Your sizing logic
            return 0.25  # 25% of equity
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class PositionSizerProtocol(Protocol):
    """
    Protocol defining the position sizer interface.
    
    Any class implementing this protocol can be used as a position sizer
    in the trading engine.
    """
    
    def calculate(
        self,
        prediction: float,
        features: Dict[str, Any],
        portfolio: Any,
        pair: str,
    ) -> float:
        """
        Calculate target position size.
        
        Args:
            prediction: Model prediction (e.g., probability of price increase)
            features: Current features dict
            portfolio: Current portfolio state
            pair: Trading pair
        
        Returns:
            Target position as fraction of equity (signed: + long, - short)
        """
        ...


class BaseSizer(ABC):
    """
    Abstract base class for position sizers.
    
    Subclasses must implement:
        - calculate(): Returns target position as fraction of equity
    
    Optional methods:
        - reset(): Reset internal state
        - get_stats(): Return sizer statistics
    """
    
    @abstractmethod
    def calculate(
        self,
        prediction: float,
        features: Dict[str, Any],
        portfolio: Any,
        pair: str,
    ) -> float:
        """
        Calculate target position size.
        
        Args:
            prediction: Model prediction (probability of price increase)
            features: Current features dict
            portfolio: Current portfolio state
            pair: Trading pair
        
        Returns:
            Target position as fraction of equity (signed: + long, - short)
        """
        pass
    
    def reset(self) -> None:
        """Reset internal state. Override if sizer has state."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return sizer statistics. Override to provide stats."""
        return {}


class BasePortfolioSizer(ABC):
    """
    Abstract base class for portfolio-level sizers.
    
    Unlike single-asset sizers, portfolio sizers allocate capital
    across multiple assets simultaneously.
    
    Subclasses must implement:
        - calculate_weights(): Returns allocation weights per asset
        - calculate_position(): Returns position for a single asset
    """
    
    @abstractmethod
    def calculate_weights(
        self,
        returns_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Calculate allocation weights for all assets.
        
        Args:
            returns_df: DataFrame of asset returns (columns = pairs)
            current_date: Current date for lookback calculations
        
        Returns:
            Dict mapping pair -> weight (weights sum to 1.0)
        """
        pass
    
    @abstractmethod
    def calculate_position(
        self,
        pair: str,
        signal: float,
        weight: float,
        equity: float,
        peak_equity: float,
        returns_df: pd.DataFrame,
        current_date: pd.Timestamp,
        price: float,
    ) -> float:
        """
        Calculate target position for a single asset.
        
        Args:
            pair: Trading pair
            signal: Signal strength (0.0 to 1.0)
            weight: Portfolio weight for this pair
            equity: Current portfolio equity
            peak_equity: Peak portfolio equity (for drawdown calc)
            returns_df: DataFrame of asset returns
            current_date: Current date
            price: Current price of the asset
        
        Returns:
            Target position in units (not fraction)
        """
        pass
    
    def reset(self) -> None:
        """Reset internal state."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return sizer statistics."""
        return {}