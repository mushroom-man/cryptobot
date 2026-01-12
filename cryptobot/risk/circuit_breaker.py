# -*- coding: utf-8 -*-
"""
CryptoBot - Circuit Breakers
=============================
P&L-based circuit breakers to prevent catastrophic losses.

From strategy document:
    - -5% daily P&L → Reduce positions 50%
    - -10% weekly P&L → Pause new trades
    - -15% monthly P&L → Flatten all positions
    - -25% peak drawdown → Full stop, manual review

Usage:
    from cryptobot.shared.risk import CircuitBreaker
    
    cb = CircuitBreaker(
        daily_limit=-0.05,
        weekly_limit=-0.10,
        monthly_limit=-0.15,
        max_drawdown=-0.25,
    )
    
    # Check in process_bar
    action = cb.check(portfolio, current_timestamp)
    if action == CircuitBreakerAction.FLATTEN:
        # Flatten all positions
    elif action == CircuitBreakerAction.REDUCE:
        # Reduce positions by 50%
    elif action == CircuitBreakerAction.PAUSE:
        # Stop opening new positions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import pandas as pd


class CircuitBreakerAction(Enum):
    """Actions triggered by circuit breakers."""
    NONE = "none"              # No action needed
    REDUCE = "reduce"          # Reduce positions (e.g., 50%)
    PAUSE = "pause"            # Pause new trades
    FLATTEN = "flatten"        # Flatten all positions
    FULL_STOP = "full_stop"    # Full stop, manual review required


@dataclass
class CircuitBreakerState:
    """Current state of circuit breakers."""
    is_triggered: bool = False
    action: CircuitBreakerAction = CircuitBreakerAction.NONE
    trigger_reason: Optional[str] = None
    trigger_time: Optional[datetime] = None
    
    # P&L tracking
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    monthly_pnl_pct: float = 0.0
    current_drawdown: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_triggered': self.is_triggered,
            'action': self.action.value,
            'trigger_reason': self.trigger_reason,
            'trigger_time': self.trigger_time,
            'daily_pnl_pct': self.daily_pnl_pct,
            'weekly_pnl_pct': self.weekly_pnl_pct,
            'monthly_pnl_pct': self.monthly_pnl_pct,
            'current_drawdown': self.current_drawdown,
        }


class CircuitBreaker:
    """
    Circuit breaker system for risk management.
    
    Monitors P&L at various time horizons and drawdown from peak.
    Triggers protective actions when limits are breached.
    
    Attributes:
        daily_limit: Max daily loss before action (e.g., -0.05 = -5%)
        weekly_limit: Max weekly loss before action
        monthly_limit: Max monthly loss before action
        max_drawdown: Max drawdown from peak before full stop
        reduce_factor: Position reduction factor (e.g., 0.5 = reduce 50%)
    """
    
    def __init__(
        self,
        daily_limit: float = -0.05,
        weekly_limit: float = -0.10,
        monthly_limit: float = -0.15,
        max_drawdown: float = -0.25,
        reduce_factor: float = 0.5,
        cooldown_hours: int = 24,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            daily_limit: Daily loss limit (negative, e.g., -0.05)
            weekly_limit: Weekly loss limit (negative)
            monthly_limit: Monthly loss limit (negative)
            max_drawdown: Maximum drawdown limit (negative)
            reduce_factor: Factor to reduce positions by (0.5 = 50%)
            cooldown_hours: Hours before resetting after trigger
        """
        # Ensure limits are negative
        self.daily_limit = -abs(daily_limit)
        self.weekly_limit = -abs(weekly_limit)
        self.monthly_limit = -abs(monthly_limit)
        self.max_drawdown = -abs(max_drawdown)
        self.reduce_factor = reduce_factor
        self.cooldown_hours = cooldown_hours
        
        # State tracking
        self.state = CircuitBreakerState()
        
        # Equity history for P&L calculation
        self.equity_history: List[Dict[str, Any]] = []
        
        # Peak equity for drawdown calculation
        self.peak_equity: float = 0.0
        self.initial_equity: float = 0.0
        
        # Tracking timestamps for period starts
        self.day_start_equity: float = 0.0
        self.week_start_equity: float = 0.0
        self.month_start_equity: float = 0.0
        self.day_start_time: Optional[datetime] = None
        self.week_start_time: Optional[datetime] = None
        self.month_start_time: Optional[datetime] = None
    
    def initialize(self, equity: float, timestamp: datetime):
        """Initialize with starting equity."""
        self.initial_equity = equity
        self.peak_equity = equity
        self.day_start_equity = equity
        self.week_start_equity = equity
        self.month_start_equity = equity
        self.day_start_time = timestamp
        self.week_start_time = timestamp
        self.month_start_time = timestamp
    
    def update(self, equity: float, timestamp: datetime):
        """
        Update equity tracking and check for period rollovers.
        
        Call this before check() on each bar.
        """
        # Initialize if first update
        if self.initial_equity == 0:
            self.initialize(equity, timestamp)
            return
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Check for period rollovers
        if self.day_start_time:
            if timestamp.date() > self.day_start_time.date():
                self.day_start_equity = self.equity_history[-1]['equity'] if self.equity_history else equity
                self.day_start_time = timestamp
        
        if self.week_start_time:
            # New week (Monday)
            if timestamp.isocalendar()[1] > self.week_start_time.isocalendar()[1]:
                self.week_start_equity = self.equity_history[-1]['equity'] if self.equity_history else equity
                self.week_start_time = timestamp
        
        if self.month_start_time:
            if timestamp.month > self.month_start_time.month or timestamp.year > self.month_start_time.year:
                self.month_start_equity = self.equity_history[-1]['equity'] if self.equity_history else equity
                self.month_start_time = timestamp
        
        # Record equity
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': equity,
        })
        
        # Keep only last 30 days of history
        cutoff = timestamp - timedelta(days=30)
        self.equity_history = [
            h for h in self.equity_history 
            if h['timestamp'] > cutoff
        ]
    
    def check(self, equity: float, timestamp: datetime) -> CircuitBreakerAction:
        """
        Check if any circuit breaker is triggered.
        
        Args:
            equity: Current portfolio equity
            timestamp: Current timestamp
        
        Returns:
            CircuitBreakerAction indicating what action to take
        """
        # Update tracking
        self.update(equity, timestamp)
        
        # Check if in cooldown
        if self.state.is_triggered and self.state.trigger_time:
            cooldown_end = self.state.trigger_time + timedelta(hours=self.cooldown_hours)
            if timestamp < cooldown_end:
                return self.state.action
        
        # Calculate P&L percentages
        daily_pnl = self._calculate_pnl(equity, self.day_start_equity)
        weekly_pnl = self._calculate_pnl(equity, self.week_start_equity)
        monthly_pnl = self._calculate_pnl(equity, self.month_start_equity)
        drawdown = self._calculate_drawdown(equity)
        
        # Update state
        self.state.daily_pnl_pct = daily_pnl
        self.state.weekly_pnl_pct = weekly_pnl
        self.state.monthly_pnl_pct = monthly_pnl
        self.state.current_drawdown = drawdown
        
        # Check limits (most severe first)
        action = CircuitBreakerAction.NONE
        reason = None
        
        if drawdown <= self.max_drawdown:
            action = CircuitBreakerAction.FULL_STOP
            reason = f"Drawdown {drawdown:.1%} exceeded limit {self.max_drawdown:.1%}"
        
        elif monthly_pnl <= self.monthly_limit:
            action = CircuitBreakerAction.FLATTEN
            reason = f"Monthly P&L {monthly_pnl:.1%} exceeded limit {self.monthly_limit:.1%}"
        
        elif weekly_pnl <= self.weekly_limit:
            action = CircuitBreakerAction.PAUSE
            reason = f"Weekly P&L {weekly_pnl:.1%} exceeded limit {self.weekly_limit:.1%}"
        
        elif daily_pnl <= self.daily_limit:
            action = CircuitBreakerAction.REDUCE
            reason = f"Daily P&L {daily_pnl:.1%} exceeded limit {self.daily_limit:.1%}"
        
        # Update state if triggered
        if action != CircuitBreakerAction.NONE:
            self.state.is_triggered = True
            self.state.action = action
            self.state.trigger_reason = reason
            self.state.trigger_time = timestamp
        else:
            self.state.is_triggered = False
            self.state.action = CircuitBreakerAction.NONE
            self.state.trigger_reason = None
        
        return action
    
    def reset(self):
        """Manually reset circuit breaker state."""
        self.state = CircuitBreakerState()
    
    def _calculate_pnl(self, current: float, start: float) -> float:
        """Calculate P&L percentage."""
        if start == 0:
            return 0.0
        return (current - start) / start
    
    def _calculate_drawdown(self, equity: float) -> float:
        """Calculate drawdown from peak (as negative percentage)."""
        if self.peak_equity == 0:
            return 0.0
        return (equity - self.peak_equity) / self.peak_equity
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state
    
    def __repr__(self) -> str:
        status = "TRIGGERED" if self.state.is_triggered else "OK"
        return (
            f"CircuitBreaker(status={status}, "
            f"daily={self.state.daily_pnl_pct:.1%}, "
            f"weekly={self.state.weekly_pnl_pct:.1%}, "
            f"dd={self.state.current_drawdown:.1%})"
        )
