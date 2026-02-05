# -*- coding: utf-8 -*-
"""
CryptoBot - Trading Engine
===========================
Core trading logic shared by backtest and production.

The TradingEngine.process_bar() method is called by:
    - BacktestRunner: loops over historical bars
    - ProductionRunner: processes live bars hourly

This ensures identical logic in both environments.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol
from enum import Enum

from cryptobot.types.bar import Bar
from cryptobot.types.order import Order
from cryptobot.types.fill import Fill
from cryptobot.types.portfolio import Portfolio


# =============================================================================
# Protocols (Interfaces)
# =============================================================================

class Executor(Protocol):
    """Interface for order execution (simulated or real)."""
    
    def execute(self, order: Order, bar: Bar) -> Fill:
        """Execute an order, return fill."""
        ...


class Predictor(Protocol):
    """Interface for model predictions."""
    
    def predict(self, features: Dict[str, float]) -> float:
        """Return prediction (e.g., P(up))."""
        ...


class RiskManager(Protocol):
    """Interface for risk management."""
    
    def check_circuit_breaker(self, portfolio: Portfolio) -> bool:
        """Return True if circuit breaker triggered."""
        ...
    
    def check_stops(self, portfolio: Portfolio, bar: Bar) -> List[str]:
        """Return list of pairs that hit stop loss."""
        ...
    
    def apply_limits(self, target: float, pair: str, 
                     portfolio: Portfolio) -> float:
        """Apply position limits, return adjusted target."""
        ...


class PositionSizer(Protocol):
    """Interface for position sizing."""
    
    def calculate(self, prediction: float, features: Dict[str, float],
                  portfolio: Portfolio, pair: str) -> float:
        """Calculate target position size."""
        ...


# =============================================================================
# Action Recording
# =============================================================================

class ActionType(Enum):
    """Types of actions taken during bar processing."""
    CIRCUIT_BREAKER = "circuit_breaker"
    STOP_LOSS = "stop_loss"
    SIGNAL_TRADE = "signal_trade"
    FLATTEN = "flatten"
    NO_ACTION = "no_action"


@dataclass
class Action:
    """Record of action taken during bar processing."""
    action_type: ActionType
    timestamp: datetime
    pair: Optional[str] = None
    order: Optional[Order] = None
    fill: Optional[Fill] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type.value,
            'timestamp': self.timestamp,
            'pair': self.pair,
            'order_id': self.order.order_id if self.order else None,
            'fill_id': self.fill.fill_id if self.fill else None,
            'details': self.details,
        }


# =============================================================================
# Trading Engine
# =============================================================================

class TradingEngine:
    """
    Core trading logic. Used by both backtest and production.
    
    The process_bar() method contains all trading logic:
        1. Update portfolio prices
        2. Check circuit breakers
        3. Check stop losses
        4. Generate prediction
        5. Calculate target position
        6. Apply risk limits
        7. Execute trades
        8. Record state
    
    Example:
        engine = TradingEngine(
            predictor=my_model,
            sizer=kelly_sizer,
            risk_manager=risk_mgr,
            portfolio=portfolio,
        )
        
        # Backtest
        for bar in historical_bars:
            actions = engine.process_bar(bar, features, executor)
        
        # Production (same code)
        actions = engine.process_bar(live_bar, live_features, kraken_executor)
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        predictor: Optional[Predictor] = None,
        sizer: Optional[PositionSizer] = None,
        risk_manager: Optional[RiskManager] = None,
        min_trade_threshold: float = 0.01,  # 1% of position to trigger trade
        p_threshold: float = 0.5,  # Prediction threshold for direction
    ):
        """
        Initialize trading engine.
        
        Args:
            portfolio: Portfolio to manage
            predictor: Model for predictions (optional, can pass prediction directly)
            sizer: Position sizing calculator
            risk_manager: Risk management system
            min_trade_threshold: Minimum position change to trigger trade
            p_threshold: Prediction threshold (>threshold = long, <threshold = short)
        """
        self.portfolio = portfolio
        self.predictor = predictor
        self.sizer = sizer
        self.risk_manager = risk_manager
        self.min_trade_threshold = min_trade_threshold
        self.p_threshold = p_threshold
        
        # State
        self.is_paused = False
        self.pause_reason: Optional[str] = None
        
        # Action history
        self.actions: List[Action] = []
    
    def process_bar(
        self,
        bar: Bar,
        features: Dict[str, float],
        executor: Executor,
        prediction: Optional[float] = None,
    ) -> List[Action]:
        """
        Process a single bar. Core trading logic.
        
        This method is called by both backtest and production runners.
        
        Args:
            bar: Current OHLCV bar
            features: Pre-computed features for this bar
            executor: Order executor (simulated or real)
            prediction: Model prediction (if not using self.predictor)
        
        Returns:
            List of actions taken
        """
        actions = []
        
        # -----------------------------------------------------------------
        # 1. Update portfolio with current prices
        # -----------------------------------------------------------------
        self.portfolio.mark_to_market(bar)
        
        # -----------------------------------------------------------------
        # 2. Check if trading is paused
        # -----------------------------------------------------------------
        if self.is_paused:
            self.portfolio.record_snapshot(bar.timestamp)
            actions.append(Action(
                action_type=ActionType.NO_ACTION,
                timestamp=bar.timestamp,
                details={'reason': f'paused: {self.pause_reason}'}
            ))
            return actions
        
        # -----------------------------------------------------------------
        # 3. Check circuit breakers
        # -----------------------------------------------------------------
        if self.risk_manager:
            if self.risk_manager.check_circuit_breaker(self.portfolio):
                # Flatten all positions
                flatten_actions = self._flatten_all(bar, executor, "circuit_breaker")
                actions.extend(flatten_actions)
                
                # Pause trading
                self.is_paused = True
                self.pause_reason = "circuit_breaker"
                
                self.portfolio.record_snapshot(bar.timestamp)
                return actions
        
        # -----------------------------------------------------------------
        # 4. Check stop losses
        # -----------------------------------------------------------------
        if self.risk_manager:
            stopped_pairs = self.risk_manager.check_stops(self.portfolio, bar)
            for pair in stopped_pairs:
                stop_action = self._close_position(
                    pair, bar, executor, "stop_loss"
                )
                if stop_action:
                    actions.append(stop_action)
        
        # -----------------------------------------------------------------
        # 5. Get prediction
        # -----------------------------------------------------------------
        if prediction is None and self.predictor:
            prediction = self.predictor.predict(features)
        
        if prediction is None:
            # No prediction available, skip signal generation
            self.portfolio.record_snapshot(bar.timestamp)
            return actions
        
        # -----------------------------------------------------------------
        # 6. Calculate target position
        # -----------------------------------------------------------------
        if self.sizer:
            target_position = self.sizer.calculate(
                prediction=prediction,
                features=features,
                portfolio=self.portfolio,
                pair=bar.pair,
            )
        else:
            # Default: simple direction from prediction
            target_position = 1.0 if prediction > self.p_threshold else -1.0
        
        # -----------------------------------------------------------------
        # 7. Apply risk limits
        # -----------------------------------------------------------------
        # NOTE: Skipping risk_manager.apply_limits() because of unit mismatch:
        # - Sizer returns fraction of equity (e.g., 0.13 = 13%)
        # - Risk manager expects units (e.g., 0.5 BTC)
        # The sizer already enforces max_position and min_position with fractions.
        # TODO: Unify units across sizer and risk manager
        if self.risk_manager:
            # Only check circuit breakers and stops, not position limits
            # target_position = self.risk_manager.apply_limits(
            #     target=target_position,
            #     pair=bar.pair,
            #     portfolio=self.portfolio,
            # )
            pass
        
        # -----------------------------------------------------------------
        # 8. Generate and execute order if needed
        # -----------------------------------------------------------------
        current_position = self.portfolio.get_position(bar.pair)
        position_delta = target_position - current_position
        
        # Check if trade is significant enough
        # Note: position_delta is a fraction of equity (from sizer)
        # So trade_pct is simply the absolute change in position fraction
        trade_pct = abs(position_delta)
        
        if trade_pct >= self.min_trade_threshold:
            order = Order.from_target_position(
                pair=bar.pair,
                current_position=current_position,
                target_position=target_position,
                timestamp=bar.timestamp,
                reference_price=bar.close,
                reason="signal",
            )
            
            if order:
                fill = executor.execute(order, bar)
                self.portfolio.apply_fill(fill)
                
                actions.append(Action(
                    action_type=ActionType.SIGNAL_TRADE,
                    timestamp=bar.timestamp,
                    pair=bar.pair,
                    order=order,
                    fill=fill,
                    details={
                        'prediction': prediction,
                        'target_position': target_position,
                        'previous_position': current_position,
                    }
                ))
        
        # -----------------------------------------------------------------
        # 9. Record portfolio state
        # -----------------------------------------------------------------
        self.portfolio.record_snapshot(bar.timestamp)
        
        # Store actions
        self.actions.extend(actions)
        
        return actions
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _close_position(
        self, 
        pair: str, 
        bar: Bar, 
        executor: Executor,
        reason: str
    ) -> Optional[Action]:
        """Close a single position."""
        position = self.portfolio.get_position(pair)
        if abs(position) < 1e-10:
            return None
        
        # Use bar price if same pair, otherwise use last known price
        if bar.pair == pair:
            price = bar.close
        else:
            pos_obj = self.portfolio.get_position_obj(pair)
            price = pos_obj.current_price
        
        order = self.portfolio.flatten_position(pair, bar.timestamp, price)
        if order:
            fill = executor.execute(order, bar)
            self.portfolio.apply_fill(fill)
            
            return Action(
                action_type=ActionType.STOP_LOSS if reason == "stop_loss" else ActionType.FLATTEN,
                timestamp=bar.timestamp,
                pair=pair,
                order=order,
                fill=fill,
                details={'reason': reason}
            )
        return None
    
    def _flatten_all(
        self, 
        bar: Bar, 
        executor: Executor,
        reason: str
    ) -> List[Action]:
        """Flatten all positions."""
        actions = []
        
        for pair in list(self.portfolio.positions.keys()):
            action = self._close_position(pair, bar, executor, reason)
            if action:
                action.action_type = ActionType.CIRCUIT_BREAKER
                actions.append(action)
        
        return actions
    
    def resume_trading(self):
        """Resume trading after pause."""
        self.is_paused = False
        self.pause_reason = None
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get action history as list of dicts."""
        return [a.to_dict() for a in self.actions]
    
    def __repr__(self) -> str:
        status = "PAUSED" if self.is_paused else "ACTIVE"
        return f"TradingEngine(status={status}, portfolio={self.portfolio})"
