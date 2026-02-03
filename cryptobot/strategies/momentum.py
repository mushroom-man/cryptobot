# -*- coding: utf-8 -*-
"""
CryptoBot - Validated Momentum Strategy
========================================
16-state momentum strategy with validated parameters.

Implements the Predictor protocol from engine.py.

Validated Parameters (DO NOT MODIFY):
    - Entry Buffer: 2.5%
    - Exit Buffer: 0.5%
    - Confirmation: 3 hours
    - Excluded States: 12, 14
    - Boost: State 11 @ ≥12h → 150%

Backtest Performance:
    - Sharpe: 3.04
    - Annual Return: +58.6%
    - Max Drawdown: -12.2%

Usage:
    from cryptobot.strategies import MomentumStrategy
    
    strategy = MomentumStrategy()
    
    # In trading loop:
    multiplier = strategy.predict(features)
    # Returns: 0.0 (flat), 1.0 (long), or 1.5 (boosted long)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATED PARAMETERS - DO NOT MODIFY
# =============================================================================

@dataclass(frozen=True)
class MomentumConfig:
    """Immutable validated configuration."""
    
    # MA Periods
    MA_PERIOD_24H: int = 16
    MA_PERIOD_72H: int = 6
    MA_PERIOD_168H: int = 2
    
    # Hysteresis Buffers
    ENTRY_BUFFER: float = 0.025  # 2.5%
    EXIT_BUFFER: float = 0.005   # 0.5%
    
    # Confirmation
    CONFIRMATION_HOURS: int = 3
    
    # State Rules
    EXCLUDED_STATES: frozenset = frozenset({12, 14})
    BULLISH_STATES: frozenset = frozenset({8, 9, 10, 11, 13, 15})  # Excludes 12, 14
    
    # Boost
    BOOST_STATE: int = 11
    BOOST_THRESHOLD_HOURS: int = 12
    BOOST_MULTIPLIER: float = 1.5


CONFIG = MomentumConfig()


# =============================================================================
# STATE TRACKER (per pair)
# =============================================================================

@dataclass
class PairState:
    """Tracks state and duration for a single pair."""
    pair: str
    
    # Trend states (with hysteresis)
    trend_24h: int = 1  # 1=bullish, 0=bearish
    trend_168h: int = 1
    
    # Confirmed state
    confirmed_state: int = 0
    duration_hours: int = 0
    
    # Pending confirmation
    pending_state: Optional[int] = None
    pending_hours: int = 0
    
    def get_position_multiplier(self) -> float:
        """Get position multiplier based on current state."""
        # Bearish states (0-7): flat
        if self.confirmed_state < 8:
            return 0.0
        
        # Excluded exhaustion states: flat
        if self.confirmed_state in CONFIG.EXCLUDED_STATES:
            return 0.0
        
        # State 11 with duration >= threshold: boost
        if (self.confirmed_state == CONFIG.BOOST_STATE and 
            self.duration_hours >= CONFIG.BOOST_THRESHOLD_HOURS):
            return CONFIG.BOOST_MULTIPLIER
        
        # Standard bullish: long
        return 1.0


# =============================================================================
# MOMENTUM STRATEGY
# =============================================================================

class MomentumStrategy:
    """
    Validated 16-state momentum strategy.
    
    Implements Predictor protocol:
        predict(features) -> float
    
    Features dict expected:
        - price: Current price
        - ma_24h: 24h MA value
        - ma_72h: 72h MA value  
        - ma_168h: 168h MA value
        - pair: Trading pair (optional, defaults to 'DEFAULT')
    
    Returns:
        Position multiplier: 0.0, 1.0, or 1.5
    """
    
    def __init__(self):
        """Initialize strategy."""
        self.config = CONFIG
        self.pair_states: Dict[str, PairState] = {}
        
    def load_state(self, db) -> int:
        """
        Load persisted state from database for all known pairs.
        
        Args:
            db: Database instance
        
        Returns:
            Number of pairs loaded
        """
        loaded = 0
        for pair in list(self.pair_states.keys()):
            state_data = db.get_strategy_state(pair, strategy='momentum')
            if state_data:
                self.pair_states[pair] = PairState(
                    pair=pair,
                    trend_24h=state_data['trend_24h'],
                    trend_168h=state_data['trend_168h'],
                    confirmed_state=state_data['confirmed_state'],
                    duration_hours=state_data['duration_hours'],
                    pending_state=state_data['pending_state'],
                    pending_hours=state_data['pending_hours'],
                )
                loaded += 1
                logger.debug(f"Loaded state for {pair}: S{state_data['confirmed_state']} ({state_data['duration_hours']}h)")
        return loaded
    
    def load_state_for_pairs(self, db, pairs: list) -> int:
        """
        Load persisted state from database for specific pairs.
        
        Args:
            db: Database instance
            pairs: List of trading pairs
        
        Returns:
            Number of pairs loaded
        """
        loaded = 0
        for pair in pairs:
            state_data = db.get_strategy_state(pair, strategy='momentum')
            if state_data:
                self.pair_states[pair] = PairState(
                    pair=pair,
                    trend_24h=state_data['trend_24h'],
                    trend_168h=state_data['trend_168h'],
                    confirmed_state=state_data['confirmed_state'],
                    duration_hours=state_data['duration_hours'],
                    pending_state=state_data['pending_state'],
                    pending_hours=state_data['pending_hours'],
                )
                loaded += 1
                logger.info(f"Loaded state for {pair}: S{state_data['confirmed_state']} ({state_data['duration_hours']}h)")
            else:
                # Initialize fresh state
                self.pair_states[pair] = PairState(pair=pair)
                logger.info(f"Initialized fresh state for {pair}")
        return loaded
    
    def save_state(self, db) -> int:
        """
        Save current state to database for all pairs.
        
        Args:
            db: Database instance
        
        Returns:
            Number of pairs saved
        """
        saved = 0
        for pair, state in self.pair_states.items():
            db.save_strategy_state(
                pair=pair,
                confirmed_state=state.confirmed_state,
                duration_hours=state.duration_hours,
                pending_state=state.pending_state,
                pending_hours=state.pending_hours,
                trend_24h=state.trend_24h,
                trend_168h=state.trend_168h,
                strategy='momentum'
            )
            saved += 1
            logger.debug(f"Saved state for {pair}: S{state.confirmed_state} ({state.duration_hours}h)")
        return saved
        
    def predict(self, features: Dict[str, float]) -> float:
        """
        Generate position signal from features.
        
        Implements Predictor protocol.
        
        Args:
            features: Dict with price, ma_24h, ma_72h, ma_168h, pair
        
        Returns:
            Position multiplier (0.0, 1.0, or 1.5)
        """
        pair = features.get('pair', 'DEFAULT')
        
        # Get or create pair state
        if pair not in self.pair_states:
            self.pair_states[pair] = PairState(pair=pair)
        
        state = self.pair_states[pair]
        
        # Extract features
        price = features.get('price') or features.get('close')
        ma_24h = features.get('ma_24h')
        ma_72h = features.get('ma_72h')
        ma_168h = features.get('ma_168h')
        
        if price is None or ma_24h is None or ma_168h is None:
            logger.warning(f"{pair}: Missing required features")
            return 0.0
        
        # Update trend states with hysteresis
        state.trend_24h = self._update_trend(
            price, ma_24h, state.trend_24h
        )
        state.trend_168h = self._update_trend(
            price, ma_168h, state.trend_168h
        )
        
        # Compute MA alignment
        ma72_above_ma24 = 1 if (ma_72h and ma_72h > ma_24h) else 0
        ma168_above_ma24 = 1 if (ma_168h and ma_168h > ma_24h) else 0
        
        # Calculate raw 16-state
        raw_state = (
            state.trend_24h * 8 +
            state.trend_168h * 4 +
            ma72_above_ma24 * 2 +
            ma168_above_ma24 * 1
        )
        
        # Apply 3h confirmation
        self._apply_confirmation(state, raw_state)
        
        # Return position multiplier
        multiplier = state.get_position_multiplier()
        
        # Log state changes
        self._log_state(pair, state, raw_state, multiplier)
        
        return multiplier
    
    def _update_trend(self, price: float, ma: float, current: int) -> int:
        """Update trend state with hysteresis."""
        if current == 1:  # Currently bullish
            # Exit bullish when price drops below MA - exit_buffer
            if price < ma * (1 - self.config.EXIT_BUFFER):
                return 0
        else:  # Currently bearish
            # Enter bullish when price rises above MA + entry_buffer
            if price > ma * (1 + self.config.ENTRY_BUFFER):
                return 1
        return current
    
    def _apply_confirmation(self, state: PairState, raw_state: int):
        """Apply 3-hour confirmation filter."""
        if raw_state == state.confirmed_state:
            # Same state - increment duration, clear pending
            state.duration_hours += 1
            state.pending_state = None
            state.pending_hours = 0
        
        elif raw_state == state.pending_state:
            # Same as pending - increment pending hours
            state.pending_hours += 1
            
            # Check if confirmed
            if state.pending_hours >= self.config.CONFIRMATION_HOURS:
                old_state = state.confirmed_state
                state.confirmed_state = raw_state
                state.duration_hours = state.pending_hours
                state.pending_state = None
                state.pending_hours = 0
                
                logger.info(f"{state.pair}: State confirmed {old_state} → {raw_state}")
        
        else:
            # New pending state
            state.pending_state = raw_state
            state.pending_hours = 1
    
    def _log_state(self, pair: str, state: PairState, raw: int, mult: float):
        """Log state for debugging."""
        boost_str = " 🚀BOOST" if mult == 1.5 else ""
        excl_str = " ⛔EXCLUDED" if state.confirmed_state in self.config.EXCLUDED_STATES else ""
        pend_str = f" (pending S{state.pending_state}:{state.pending_hours}h)" if state.pending_state else ""
        
        logger.debug(
            f"{pair}: S{state.confirmed_state}({state.duration_hours}h) "
            f"raw={raw} → {mult:.0%}{boost_str}{excl_str}{pend_str}"
        )
    
    # =========================================================================
    # State Access (for reporting/debugging)
    # =========================================================================
    
    def get_pair_state(self, pair: str) -> Optional[PairState]:
        """Get current state for a pair."""
        return self.pair_states.get(pair)
    
    def get_all_states(self) -> Dict[str, PairState]:
        """Get all pair states."""
        return self.pair_states.copy()
    
    def reset(self, pair: str = None):
        """Reset state for a pair or all pairs."""
        if pair:
            if pair in self.pair_states:
                del self.pair_states[pair]
        else:
            self.pair_states.clear()
    
    def get_active_boosts(self) -> list:
        """Get list of pairs with active boost."""
        return [
            pair for pair, state in self.pair_states.items()
            if (state.confirmed_state == self.config.BOOST_STATE and
                state.duration_hours >= self.config.BOOST_THRESHOLD_HOURS)
        ]
    
    def get_excluded_pairs(self) -> list:
        """Get list of pairs in excluded states."""
        return [
            pair for pair, state in self.pair_states.items()
            if state.confirmed_state in self.config.EXCLUDED_STATES
        ]
    
    def get_pending_boosts(self) -> Dict[str, int]:
        """Get pairs approaching boost with hours remaining."""
        result = {}
        for pair, state in self.pair_states.items():
            if (state.confirmed_state == self.config.BOOST_STATE and
                state.duration_hours < self.config.BOOST_THRESHOLD_HOURS):
                hours_to_boost = self.config.BOOST_THRESHOLD_HOURS - state.duration_hours
                result[pair] = hours_to_boost
        return result
    
    def describe_state(self, pair: str) -> str:
        """Get human-readable state description."""
        state = self.pair_states.get(pair)
        if not state:
            return f"{pair}: No state"
        
        mult = state.get_position_multiplier()
        
        if state.confirmed_state < 8:
            regime = "BEARISH"
        elif state.confirmed_state in self.config.EXCLUDED_STATES:
            regime = "EXHAUSTION"
        elif mult == 1.5:
            regime = "BOOSTED"
        else:
            regime = "BULLISH"
        
        return (
            f"{pair}: State {state.confirmed_state} ({state.duration_hours}h) "
            f"→ {regime} @ {mult:.0%}"
        )
    
    def __repr__(self) -> str:
        active = len([s for s in self.pair_states.values() if s.confirmed_state >= 8])
        return f"MomentumStrategy(pairs={len(self.pair_states)}, active={active})"