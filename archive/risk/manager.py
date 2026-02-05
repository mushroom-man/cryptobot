# -*- coding: utf-8 -*-
"""
CryptoBot - Risk Manager
=========================
Unified risk management combining circuit breakers, stops, and limits.

Usage:
    from cryptobot.risk import RiskManager
    
    risk = RiskManager(
        circuit_breaker_config={...},
        stop_loss_config={...},
        position_limits_config={...},
    )
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable

# Use relative imports within the risk module
from cryptobot.risk.circuit_breaker import (
    CircuitBreaker, 
    CircuitBreakerAction,
    CircuitBreakerState,
)
from cryptobot.risk.stops import (
    StopLossManager, 
    StopTriggered,
    StopType,
)
from cryptobot.risk.limits import (
    PositionLimits, 
    LimitResult,
)


# =============================================================================
# Protocols for type hints (avoid importing non-existent modules)
# =============================================================================

@runtime_checkable
class BarProtocol(Protocol):
    """Protocol for Bar-like objects."""
    timestamp: datetime
    pair: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@runtime_checkable
class PortfolioProtocol(Protocol):
    """Protocol for Portfolio-like objects."""
    equity: float
    cash: float
    
    def get_position(self, pair: str) -> Optional[Any]: ...


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
        }


@dataclass
class RiskCheckResult:
    """Result of risk check."""
    
    allowed: bool
    adjusted_position: Optional[float] = None
    reason: Optional[str] = None
    circuit_breaker_status: Optional[CircuitBreakerState] = None
    stop_triggered: Optional[StopTriggered] = None
    limit_result: Optional[LimitResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'allowed': self.allowed,
            'adjusted_position': self.adjusted_position,
            'reason': self.reason,
        }
        if self.circuit_breaker_status:
            result['circuit_breaker'] = self.circuit_breaker_status.value
        if self.stop_triggered:
            result['stop_triggered'] = {
                'type': self.stop_triggered.stop_type.value,
                'price': self.stop_triggered.trigger_price,
            }
        if self.limit_result:
            result['limit'] = self.limit_result.to_dict()
        return result


class RiskManager:
    """
    Unified risk manager.
    
    Combines:
        - Circuit breakers (portfolio-level protection)
        - Stop losses (position-level protection)
        - Position limits (exposure control)
    
    Process:
        1. Check circuit breakers (can halt all trading)
        2. Check stop losses for existing positions
        3. Apply position limits to new positions
    """
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        stop_manager: Optional[StopLossManager] = None,
        position_limits: Optional[PositionLimits] = None,
    ):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration
            circuit_breaker: Custom circuit breaker (creates default if None)
            stop_manager: Custom stop manager (creates default if None)
            position_limits: Custom position limits (creates default if None)
        """
        self.config = config or RiskConfig()
        
        # Initialize components
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            daily_loss_limit=self.config.daily_loss_limit,
            weekly_loss_limit=self.config.weekly_loss_limit,
            monthly_loss_limit=self.config.monthly_loss_limit,
            max_drawdown=self.config.max_drawdown,
        )
        
        self.stop_manager = stop_manager or StopLossManager(
            atr_multiplier=self.config.atr_multiplier,
            garch_multiplier=self.config.garch_multiplier,
            max_loss_per_trade=self.config.max_loss_per_trade,
            use_trailing=self.config.use_trailing_stops,
            trailing_pct=self.config.trailing_stop_pct,
        )
        
        self.position_limits = position_limits or PositionLimits(
            max_position_pct=self.config.max_position_pct,
            max_total_exposure=self.config.max_total_exposure,
            max_leverage=self.config.max_leverage,
            max_correlated_exposure=self.config.max_correlated_exposure,
        )
        
        # Track risk events
        self.risk_events: List[Dict[str, Any]] = []
    
    def check_risk(
        self,
        bar: Any,  # BarProtocol
        portfolio: Any,  # PortfolioProtocol
        target_position: float,
        features: Optional[Dict[str, float]] = None,
    ) -> RiskCheckResult:
        """
        Run full risk check.
        
        Args:
            bar: Current market bar
            portfolio: Current portfolio state
            target_position: Proposed target position
            features: Current features (for dynamic stops)
        
        Returns:
            RiskCheckResult with decision and details
        """
        features = features or {}
        
        # 1. Check circuit breakers
        cb_action = self.circuit_breaker.check(portfolio)
        
        if cb_action.action != "CONTINUE":
            self._log_risk_event("circuit_breaker", bar, {
                'action': cb_action.action,
                'reason': cb_action.reason,
            })
            
            return RiskCheckResult(
                allowed=False,
                adjusted_position=0.0 if cb_action.action == "CLOSE_ALL" else None,
                reason=f"Circuit breaker: {cb_action.reason}",
                circuit_breaker_status=self.circuit_breaker.state,
            )
        
        # 2. Check stop losses for existing position
        pair = bar.pair
        position = portfolio.get_position(pair)
        
        if position and position.size != 0:
            stop_triggered = self.stop_manager.check(
                pair=pair,
                current_price=bar.close,
                position_size=position.size,
                entry_price=position.avg_entry_price,
                features=features,
            )
            
            if stop_triggered:
                self._log_risk_event("stop_loss", bar, {
                    'type': stop_triggered.stop_type.value,
                    'trigger_price': stop_triggered.trigger_price,
                })
                
                return RiskCheckResult(
                    allowed=True,
                    adjusted_position=0.0,  # Close position
                    reason=f"Stop triggered: {stop_triggered.stop_type.value}",
                    stop_triggered=stop_triggered,
                )
        
        # 3. Apply position limits
        if target_position != 0:
            limit_result = self.position_limits.apply(
                target_position=target_position,
                pair=pair,
                portfolio=portfolio,
                price=bar.close,
            )
            
            if limit_result.was_limited:
                self._log_risk_event("position_limit", bar, {
                    'original': limit_result.original,
                    'adjusted': limit_result.adjusted,
                    'reason': limit_result.limit_reason,
                })
                
                return RiskCheckResult(
                    allowed=True,
                    adjusted_position=limit_result.adjusted,
                    reason=f"Position limited: {limit_result.limit_reason}",
                    limit_result=limit_result,
                )
        
        # All checks passed
        return RiskCheckResult(
            allowed=True,
            adjusted_position=target_position,
            circuit_breaker_status=self.circuit_breaker.state,
        )
    
    def update(
        self,
        bar: Any,  # BarProtocol
        portfolio: Any,  # PortfolioProtocol
    ):
        """
        Update risk components with new data.
        
        Call this after each bar to update trailing stops, etc.
        """
        # Update trailing stops
        pair = bar.pair
        position = portfolio.get_position(pair)
        
        if position and position.size != 0:
            self.stop_manager.update_trailing(
                pair=pair,
                current_price=bar.close,
                position_size=position.size,
            )
        
        # Update circuit breaker P&L tracking
        self.circuit_breaker.update_pnl(portfolio)
    
    def _log_risk_event(
        self,
        event_type: str,
        bar: Any,
        details: Dict[str, Any],
    ):
        """Log a risk event."""
        self.risk_events.append({
            'timestamp': bar.timestamp,
            'pair': bar.pair,
            'type': event_type,
            'price': bar.close,
            **details,
        })
    
    def get_risk_events(self) -> List[Dict[str, Any]]:
        """Get all logged risk events."""
        return self.risk_events.copy()
    
    def reset(self):
        """Reset risk manager state."""
        self.circuit_breaker.reset()
        self.stop_manager.reset()
        self.risk_events.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'active_stops': self.stop_manager.get_active_stops(),
            'risk_events_count': len(self.risk_events),
        }
    
    def __repr__(self) -> str:
        return (
            f"RiskManager(max_dd={self.config.max_drawdown:.0%}, "
            f"max_pos={self.config.max_position_pct:.0%})"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_risk_manager(
    max_drawdown: float = -0.25,
    max_position: float = 0.30,
    daily_loss_limit: float = -0.05,
) -> RiskManager:
    """Create risk manager with common settings."""
    config = RiskConfig(
        max_drawdown=max_drawdown,
        max_position_pct=max_position,
        daily_loss_limit=daily_loss_limit,
    )
    return RiskManager(config=config)


def create_conservative_risk_manager() -> RiskManager:
    """Create conservative risk manager."""
    config = RiskConfig(
        daily_loss_limit=-0.03,
        weekly_loss_limit=-0.07,
        monthly_loss_limit=-0.10,
        max_drawdown=-0.15,
        max_position_pct=0.20,
        max_total_exposure=2.0,
    )
    return RiskManager(config=config)


def create_aggressive_risk_manager() -> RiskManager:
    """Create aggressive risk manager."""
    config = RiskConfig(
        daily_loss_limit=-0.10,
        weekly_loss_limit=-0.20,
        monthly_loss_limit=-0.30,
        max_drawdown=-0.40,
        max_position_pct=0.50,
        max_total_exposure=5.0,
    )
    return RiskManager(config=config)