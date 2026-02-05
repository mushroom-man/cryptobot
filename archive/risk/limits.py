# -*- coding: utf-8 -*-
"""
CryptoBot - Position Limits
============================
Position sizing limits and exposure controls.

Limits:
    - Max position per pair (% of equity)
    - Max total exposure (sum of all positions)
    - Max leverage
    - Correlation-based limits (reduce if pairs too correlated)

Usage:
    from cryptobot.shared.risk import PositionLimits
    
    limits = PositionLimits(
        max_position_pct=0.30,
        max_total_exposure=3.0,
        max_correlated_exposure=0.50,
    )
    
    # Apply limits
    adjusted = limits.apply(target_position, pair, portfolio)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class LimitResult:
    """Result of applying position limits."""
    original: float
    adjusted: float
    was_limited: bool
    limit_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original': self.original,
            'adjusted': self.adjusted,
            'was_limited': self.was_limited,
            'limit_reason': self.limit_reason,
        }


class PositionLimits:
    """
    Position limits and exposure controls.
    
    Enforces:
        - Maximum position size per pair
        - Maximum total portfolio exposure
        - Maximum leverage
        - Correlation-based exposure limits
    
    Attributes:
        max_position_pct: Max single position as % of equity (0.30 = 30%)
        max_total_exposure: Max total exposure as multiple of equity (3.0 = 300%)
        max_leverage: Max leverage allowed
        max_correlated_exposure: Max exposure to correlated pairs
        correlation_threshold: Correlation above which pairs are "correlated"
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.30,
        max_total_exposure: float = 3.0,
        max_leverage: float = 3.0,
        max_correlated_exposure: float = 0.50,
        correlation_threshold: float = 0.80,
        min_position_pct: float = 0.01,
    ):
        """
        Initialize position limits.
        
        Args:
            max_position_pct: Max single position (0.30 = 30% of equity)
            max_total_exposure: Max total exposure (3.0 = 3x equity)
            max_leverage: Max leverage allowed
            max_correlated_exposure: Max exposure to correlated pairs
            correlation_threshold: Pairs with correlation > this are "correlated"
            min_position_pct: Minimum position size (below this = 0)
        """
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.max_leverage = max_leverage
        self.max_correlated_exposure = max_correlated_exposure
        self.correlation_threshold = correlation_threshold
        self.min_position_pct = min_position_pct
        
        # Correlation matrix (pair -> pair -> correlation)
        self.correlations: Dict[str, Dict[str, float]] = {}
        
        # Default correlations for crypto (high correlation typical)
        self._set_default_correlations()
    
    def _set_default_correlations(self):
        """Set default correlations for common crypto pairs."""
        # Crypto pairs are highly correlated
        default_pairs = ["XBTUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AVAXUSD", "LTCUSD"]
        
        for pair1 in default_pairs:
            self.correlations[pair1] = {}
            for pair2 in default_pairs:
                if pair1 == pair2:
                    self.correlations[pair1][pair2] = 1.0
                else:
                    # Default high correlation for crypto
                    self.correlations[pair1][pair2] = 0.85
    
    def set_correlation(self, pair1: str, pair2: str, correlation: float):
        """Set correlation between two pairs."""
        if pair1 not in self.correlations:
            self.correlations[pair1] = {}
        if pair2 not in self.correlations:
            self.correlations[pair2] = {}
        
        self.correlations[pair1][pair2] = correlation
        self.correlations[pair2][pair1] = correlation
    
    def set_correlation_matrix(self, matrix: Dict[str, Dict[str, float]]):
        """Set full correlation matrix."""
        self.correlations = matrix
    
    def get_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two pairs."""
        if pair1 == pair2:
            return 1.0
        
        if pair1 in self.correlations:
            if pair2 in self.correlations[pair1]:
                return self.correlations[pair1][pair2]
        
        # Default high correlation for crypto
        return 0.85
    
    def apply_position_limit(
        self,
        target_position: float,
        pair: str,
        current_price: float,
        equity: float,
    ) -> LimitResult:
        """
        Apply maximum position size limit.
        
        Args:
            target_position: Target position in units
            pair: Trading pair
            current_price: Current price
            equity: Portfolio equity
        
        Returns:
            LimitResult with adjusted position
        """
        if equity <= 0:
            return LimitResult(target_position, 0, True, "Zero equity")
        
        # Calculate position value as % of equity
        position_value = abs(target_position * current_price)
        position_pct = position_value / equity
        
        # Check if exceeds limit
        if position_pct > self.max_position_pct:
            # Scale down to limit
            max_value = equity * self.max_position_pct
            max_units = max_value / current_price
            
            if target_position >= 0:
                adjusted = max_units
            else:
                adjusted = -max_units
            
            return LimitResult(
                original=target_position,
                adjusted=adjusted,
                was_limited=True,
                limit_reason=f"Position {position_pct:.1%} exceeds limit {self.max_position_pct:.1%}"
            )
        
        # Check minimum position
        if position_pct < self.min_position_pct and position_pct > 0:
            return LimitResult(
                original=target_position,
                adjusted=0,
                was_limited=True,
                limit_reason=f"Position {position_pct:.1%} below minimum {self.min_position_pct:.1%}"
            )
        
        return LimitResult(target_position, target_position, False)
    
    def apply_exposure_limit(
        self,
        target_position: float,
        pair: str,
        current_price: float,
        equity: float,
        current_positions: Dict[str, float],
        position_prices: Dict[str, float],
    ) -> LimitResult:
        """
        Apply total exposure limit.
        
        Args:
            target_position: Target position for this pair
            pair: Trading pair
            current_price: Current price for this pair
            equity: Portfolio equity
            current_positions: Current positions (pair -> size)
            position_prices: Current prices for all pairs
        
        Returns:
            LimitResult with adjusted position
        """
        if equity <= 0:
            return LimitResult(target_position, 0, True, "Zero equity")
        
        # Calculate current total exposure (excluding this pair)
        current_exposure = 0
        for p, size in current_positions.items():
            if p != pair:
                price = position_prices.get(p, 0)
                current_exposure += abs(size * price)
        
        # Add proposed position
        proposed_exposure = current_exposure + abs(target_position * current_price)
        exposure_ratio = proposed_exposure / equity
        
        # Check if exceeds limit
        if exposure_ratio > self.max_total_exposure:
            # Calculate how much room we have
            max_exposure = equity * self.max_total_exposure
            remaining = max_exposure - current_exposure
            
            if remaining <= 0:
                return LimitResult(
                    original=target_position,
                    adjusted=0,
                    was_limited=True,
                    limit_reason=f"Total exposure {exposure_ratio:.1%} exceeds limit {self.max_total_exposure:.0%}"
                )
            
            # Scale down to fit remaining room
            max_units = remaining / current_price
            if target_position >= 0:
                adjusted = min(target_position, max_units)
            else:
                adjusted = max(target_position, -max_units)
            
            return LimitResult(
                original=target_position,
                adjusted=adjusted,
                was_limited=True,
                limit_reason=f"Reduced to fit exposure limit {self.max_total_exposure:.0%}"
            )
        
        return LimitResult(target_position, target_position, False)
    
    def apply_correlation_limit(
        self,
        target_position: float,
        pair: str,
        current_price: float,
        equity: float,
        current_positions: Dict[str, float],
        position_prices: Dict[str, float],
    ) -> LimitResult:
        """
        Apply correlation-based exposure limit.
        
        Reduces position if total correlated exposure is too high.
        
        Args:
            target_position: Target position for this pair
            pair: Trading pair
            current_price: Current price for this pair
            equity: Portfolio equity
            current_positions: Current positions (pair -> size)
            position_prices: Current prices for all pairs
        
        Returns:
            LimitResult with adjusted position
        """
        if equity <= 0:
            return LimitResult(target_position, 0, True, "Zero equity")
        
        # Find highly correlated pairs
        correlated_pairs = []
        for p in current_positions.keys():
            if p != pair:
                corr = self.get_correlation(pair, p)
                if corr >= self.correlation_threshold:
                    correlated_pairs.append((p, corr))
        
        if not correlated_pairs:
            return LimitResult(target_position, target_position, False)
        
        # Calculate exposure to correlated pairs
        correlated_exposure = 0
        for p, corr in correlated_pairs:
            size = current_positions.get(p, 0)
            price = position_prices.get(p, 0)
            # Weight by correlation
            correlated_exposure += abs(size * price) * corr
        
        # Add proposed position
        proposed_correlated = correlated_exposure + abs(target_position * current_price)
        correlated_ratio = proposed_correlated / equity
        
        # Check if exceeds limit
        if correlated_ratio > self.max_correlated_exposure:
            # Calculate how much room we have
            max_correlated = equity * self.max_correlated_exposure
            remaining = max_correlated - correlated_exposure
            
            if remaining <= 0:
                return LimitResult(
                    original=target_position,
                    adjusted=0,
                    was_limited=True,
                    limit_reason=f"Correlated exposure {correlated_ratio:.1%} exceeds limit"
                )
            
            # Scale down to fit
            max_units = remaining / current_price
            if target_position >= 0:
                adjusted = min(target_position, max_units)
            else:
                adjusted = max(target_position, -max_units)
            
            return LimitResult(
                original=target_position,
                adjusted=adjusted,
                was_limited=True,
                limit_reason=f"Reduced due to correlation with {[p for p, _ in correlated_pairs]}"
            )
        
        return LimitResult(target_position, target_position, False)
    
    def apply_all_limits(
        self,
        target_position: float,
        pair: str,
        current_price: float,
        equity: float,
        current_positions: Optional[Dict[str, float]] = None,
        position_prices: Optional[Dict[str, float]] = None,
    ) -> LimitResult:
        """
        Apply all position limits.
        
        Limits are applied in order:
            1. Position size limit
            2. Total exposure limit
            3. Correlation limit
        
        Returns most restrictive result.
        """
        current_positions = current_positions or {}
        position_prices = position_prices or {}
        
        # Start with target
        adjusted = target_position
        reasons = []
        
        # Apply position limit
        result = self.apply_position_limit(adjusted, pair, current_price, equity)
        if result.was_limited:
            adjusted = result.adjusted
            reasons.append(result.limit_reason)
        
        # Apply exposure limit
        if abs(adjusted) > 0:
            result = self.apply_exposure_limit(
                adjusted, pair, current_price, equity,
                current_positions, position_prices
            )
            if result.was_limited:
                adjusted = result.adjusted
                reasons.append(result.limit_reason)
        
        # Apply correlation limit
        if abs(adjusted) > 0:
            result = self.apply_correlation_limit(
                adjusted, pair, current_price, equity,
                current_positions, position_prices
            )
            if result.was_limited:
                adjusted = result.adjusted
                reasons.append(result.limit_reason)
        
        return LimitResult(
            original=target_position,
            adjusted=adjusted,
            was_limited=len(reasons) > 0,
            limit_reason="; ".join(reasons) if reasons else None,
        )
    
    def __repr__(self) -> str:
        return (
            f"PositionLimits(max_pos={self.max_position_pct:.0%}, "
            f"max_exp={self.max_total_exposure:.0%}, "
            f"max_corr={self.max_correlated_exposure:.0%})"
        )
