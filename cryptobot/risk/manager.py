# -*- coding: utf-8 -*-
"""
CryptoBot - Risk Manager
=========================
Unified risk management combining circuit breakers, stops, and limits.

Implements the RiskManager protocol from shared.core.engine.

Usage:
    from cryptobot.shared.risk import RiskManager
    
    risk = RiskManager(
        circuit_breaker_config={...},
        stop_loss_config={...},
        position_limits_config={...},
    )
    
    # Use with TradingEngine
    engine = TradingEngine(
        portfolio=portfolio,
        risk_manager=risk,
        ...
    )
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from cryptobot.shared.core.bar import Bar
from cryptobot.shared.core.portfolio import Portfolio
from cryptobot.shared.risk.circuit_breaker import (
    CircuitBreaker, 
    CircuitBreakerAction,
    CircuitBreakerState,
)
from cryptobot.shared.risk.stops import (
    StopLossManager, 
    StopTriggered,
    StopType,
)
from cryptobot.shared.risk.limits import (
    PositionLimits, 
    LimitResult,
)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    # Circuit breaker settings
    daily_loss_limit: float = -0.05
    weekly_loss_limit: float = -0.10
    monthly_loss_limit: float = -0.15
    max_drawdown: float = -0.25
    
    # Stop loss settings
    atr_multiplier: float = 2.0
    garch_multiplier: float = 2.5
    signal_threshold: float = 0.35
    max_loss_per_trade: float = 0.03
    use_trailing_stops: bool = False
    trailing_stop_pct: float = 0.02
    
    # Position limits
    max_position_pct: float = 0.30
    max_total_exposure: float = 3.0
    max_leverage: float = 3.0
    max_correlated_exposure: float = 0.50
    correlation_threshold: float = 0.80
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'daily_loss_limit': self.daily_loss_limit,
            'weekly_loss_limit': self.weekly_loss_limit,
            'monthly_loss_limit': self.monthly_loss_limit,
            'max_drawdown': self.max_drawdown,
            'atr_multiplier': self.atr_multiplier,
            'garch_multiplier': self.garch_multiplier,
            'signal_threshold': self.signal_threshold,
            'max_loss_per_trade': self.max_loss_per_trade,
            'use_trailing_stops': self.use_trailing_stops,
            'trailing_stop_pct': self.trailing_stop_pct,
            'max_position_pct': self.max_position_pct,
            'max_total_exposure': self.max_total_exposure,
            'max_leverage': self.max_leverage,
            'max_correlated_exposure': self.max_correlated_exposure,
            'correlation_threshold': self.correlation_threshold,
        }


class RiskManager:
    """
    Unified risk manager combining all risk components.
    
    Implements the RiskManager protocol from shared.core.engine:
        - check_circuit_breaker(portfolio) -> bool
        - check_stops(portfolio, bar) -> List[str]
        - apply_limits(target, pair, portfolio) -> float
    
    Components:
        - CircuitBreaker: P&L and drawdown limits
        - StopLossManager: ATR, GARCH, signal stops
        - PositionLimits: Position and exposure limits
    """
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        enabled: bool = True,
    ):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration (uses defaults if None)
            enabled: Whether risk management is enabled
        """
        self.config = config or RiskConfig()
        self.enabled = enabled
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            daily_limit=self.config.daily_loss_limit,
            weekly_limit=self.config.weekly_loss_limit,
            monthly_limit=self.config.monthly_loss_limit,
            max_drawdown=self.config.max_drawdown,
        )
        
        self.stop_manager = StopLossManager(
            atr_multiplier=self.config.atr_multiplier,
            garch_multiplier=self.config.garch_multiplier,
            signal_threshold=self.config.signal_threshold,
            max_loss_pct=self.config.max_loss_per_trade,
            use_trailing=self.config.use_trailing_stops,
            trailing_pct=self.config.trailing_stop_pct,
        )
        
        self.position_limits = PositionLimits(
            max_position_pct=self.config.max_position_pct,
            max_total_exposure=self.config.max_total_exposure,
            max_leverage=self.config.max_leverage,
            max_correlated_exposure=self.config.max_correlated_exposure,
            correlation_threshold=self.config.correlation_threshold,
        )
        
        # State tracking
        self.is_paused = False
        self.pause_reason: Optional[str] = None
        self.last_circuit_breaker_action = CircuitBreakerAction.NONE
    
    # =========================================================================
    # Protocol Implementation
    # =========================================================================
    
    def check_circuit_breaker(self, portfolio: Portfolio) -> bool:
        """
        Check if circuit breaker is triggered.
        
        Implements RiskManager protocol.
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            True if circuit breaker triggered (should flatten/pause)
        """
        if not self.enabled:
            return False
        
        # Get current timestamp from latest snapshot
        timestamp = datetime.now()
        if portfolio.snapshots:
            timestamp = portfolio.snapshots[-1].timestamp
        
        # Check circuit breaker
        action = self.circuit_breaker.check(portfolio.equity, timestamp)
        self.last_circuit_breaker_action = action
        
        if action in [
            CircuitBreakerAction.FLATTEN,
            CircuitBreakerAction.FULL_STOP,
        ]:
            self.is_paused = True
            self.pause_reason = f"Circuit breaker: {action.value}"
            return True
        
        return False
    
    def check_stops(
        self, 
        portfolio: Portfolio, 
        bar: Bar,
        features: Optional[Dict[str, float]] = None,
        prediction: Optional[float] = None,
    ) -> List[str]:
        """
        Check stop losses for all positions.
        
        Implements RiskManager protocol.
        
        Args:
            portfolio: Current portfolio state
            bar: Current bar data
            features: Current features (for GARCH vol, ATR)
            prediction: Current prediction (for signal stops)
        
        Returns:
            List of pairs that hit stop loss
        """
        if not self.enabled:
            return []
        
        features = features or {}
        stopped_pairs = []
        
        # Get feature values for stops
        atr = features.get('atr_14')
        garch_vol = features.get('garch_vol_simple')
        
        # Check each position
        for pair, position in portfolio.positions.items():
            if position.is_flat:
                continue
            
            # Only check stop if this bar is for this pair
            # (In multi-pair scenario, we'd have separate bars)
            if bar.pair != pair:
                continue
            
            # Check all stops
            triggered = self.stop_manager.check_stops(
                pair=pair,
                entry_price=position.avg_entry_price,
                current_price=bar.close,
                position_size=position.size,
                prediction=prediction,
                atr=atr,
                garch_vol=garch_vol,
                timestamp=bar.timestamp,
            )
            
            if triggered:
                stopped_pairs.append(pair)
                # Clear position tracking
                self.stop_manager.clear_position(pair)
        
        return stopped_pairs
    
    def apply_limits(
        self,
        target: float,
        pair: str,
        portfolio: Portfolio,
        current_price: Optional[float] = None,
    ) -> float:
        """
        Apply position limits to target position.
        
        Implements RiskManager protocol.
        
        Args:
            target: Target position size (in units)
            pair: Trading pair
            portfolio: Current portfolio state
            current_price: Current price (if not in portfolio)
        
        Returns:
            Adjusted position size after limits
        """
        if not self.enabled:
            return target
        
        # Get current price
        if current_price is None:
            if pair in portfolio.positions:
                current_price = portfolio.positions[pair].current_price
            else:
                return target  # Can't apply limits without price
        
        # Build current positions dict
        current_positions = {}
        position_prices = {}
        for p, pos in portfolio.positions.items():
            current_positions[p] = pos.size
            position_prices[p] = pos.current_price
        
        # Apply all limits
        result = self.position_limits.apply_all_limits(
            target_position=target,
            pair=pair,
            current_price=current_price,
            equity=portfolio.equity,
            current_positions=current_positions,
            position_prices=position_prices,
        )
        
        return result.adjusted
    
    # =========================================================================
    # Additional Methods
    # =========================================================================
    
    def update_entry(self, pair: str, entry_price: float):
        """Record entry price for trailing stops."""
        self.stop_manager.update_entry(pair, entry_price)
    
    def clear_position(self, pair: str):
        """Clear tracking for closed position."""
        self.stop_manager.clear_position(pair)
    
    def resume(self):
        """Resume trading after pause."""
        self.is_paused = False
        self.pause_reason = None
        self.circuit_breaker.reset()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current risk manager state."""
        return {
            'enabled': self.enabled,
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'circuit_breaker': self.circuit_breaker.get_state().to_dict(),
            'last_cb_action': self.last_circuit_breaker_action.value,
        }
    
    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Get circuit breaker state."""
        return self.circuit_breaker.get_state()
    
    def set_correlation_matrix(self, matrix: Dict[str, Dict[str, float]]):
        """Set correlation matrix for position limits."""
        self.position_limits.set_correlation_matrix(matrix)
    
    def __repr__(self) -> str:
        status = "PAUSED" if self.is_paused else ("ENABLED" if self.enabled else "DISABLED")
        return f"RiskManager(status={status})"


# =============================================================================
# Factory Functions
# =============================================================================

def create_risk_manager(
    # Circuit breaker
    daily_limit: float = -0.05,
    weekly_limit: float = -0.10,
    monthly_limit: float = -0.15,
    max_drawdown: float = -0.25,
    # Stops
    atr_multiplier: float = 2.0,
    garch_multiplier: float = 2.5,
    signal_threshold: float = 0.35,
    max_loss_per_trade: float = 0.03,
    # Limits
    max_position_pct: float = 0.30,
    max_total_exposure: float = 3.0,
    enabled: bool = True,
) -> RiskManager:
    """
    Create a risk manager with custom settings.
    
    Convenience function for quick setup.
    """
    config = RiskConfig(
        daily_loss_limit=daily_limit,
        weekly_loss_limit=weekly_limit,
        monthly_loss_limit=monthly_limit,
        max_drawdown=max_drawdown,
        atr_multiplier=atr_multiplier,
        garch_multiplier=garch_multiplier,
        signal_threshold=signal_threshold,
        max_loss_per_trade=max_loss_per_trade,
        max_position_pct=max_position_pct,
        max_total_exposure=max_total_exposure,
    )
    
    return RiskManager(config=config, enabled=enabled)


def create_conservative_risk_manager() -> RiskManager:
    """Create a conservative risk manager with tight limits."""
    config = RiskConfig(
        daily_loss_limit=-0.03,
        weekly_loss_limit=-0.07,
        monthly_loss_limit=-0.10,
        max_drawdown=-0.15,
        max_loss_per_trade=0.02,
        max_position_pct=0.20,
        max_total_exposure=2.0,
    )
    return RiskManager(config=config)


def create_aggressive_risk_manager() -> RiskManager:
    """Create an aggressive risk manager with wider limits."""
    config = RiskConfig(
        daily_loss_limit=-0.10,
        weekly_loss_limit=-0.20,
        monthly_loss_limit=-0.30,
        max_drawdown=-0.40,
        max_loss_per_trade=0.05,
        max_position_pct=0.50,
        max_total_exposure=5.0,
    )
    return RiskManager(config=config)
