# -*- coding: utf-8 -*-
"""
CryptoBot - Avoid Danger Position Sizing (V16 Strategy)
========================================================
Binary position sizing: 100% long unless danger signal.

Based on empirical testing (3-year backtest):
    - Avoid Regime 0 (Calm/Calm) when prediction < 0.5
    - Result: +295.9% vs +197.1% B&H, Sharpe 1.07

Strategy Logic:
    IF regime_hybrid == 0 AND prediction < threshold:
        position = 0  (exit/stay out)
    ELSE:
        position = 1  (100% long)

Economic Reasoning:
    Regime 0 (Calm/Calm) = low structural vol + low momentum vol
    "Calm before the storm" - quiet periods with bearish undertones
    often precede breakdowns. Volatility is mean-reverting, so
    extremely calm periods WILL see volatility expand. Bearish
    signals during compression suggest the expansion will be negative.

Usage:
    from cryptobot.risk.drawdown import AvoidDangerSizer
    
    sizer = AvoidDangerSizer(
        danger_regime=0,
        prediction_threshold=0.5,
    )
    
    target = sizer.calculate(
        prediction=0.45,
        features={'regime_hybrid_simple': 0},
        portfolio=portfolio,
        pair="XBTUSD",
    )
    # Returns 0.0 (danger signal) because regime=0 and prediction<0.5
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import numpy as np


@dataclass
class AvoidDangerConfig:
    """Configuration for Avoid Danger strategy."""
    
    # Danger conditions
    danger_regime: int = 0                    # Regime to avoid (0 = Calm/Calm)
    prediction_threshold: float = 0.5         # Exit if prediction below this
    
    # Position sizing
    full_position: float = 1.0                # Position when not in danger
    danger_position: float = 0.0              # Position when in danger
    
    # Optional: Multiple danger regimes
    danger_regimes: List[int] = field(default_factory=lambda: [0])
    
    # Optional: Use prediction as position size (scaled)
    scale_by_prediction: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'danger_regime': self.danger_regime,
            'prediction_threshold': self.prediction_threshold,
            'full_position': self.full_position,
            'danger_position': self.danger_position,
            'danger_regimes': self.danger_regimes,
            'scale_by_prediction': self.scale_by_prediction,
        }


class AvoidDangerSizer:
    """
    Binary position sizer: Full position unless danger signal.
    
    Simple but effective strategy based on empirical testing.
    
    The "danger" condition is:
        regime == danger_regime AND prediction < threshold
    
    When danger detected: position = 0 (exit)
    Otherwise: position = 1 (full long)
    """
    
    def __init__(self, config: Optional[AvoidDangerConfig] = None):
        """
        Initialize Avoid Danger sizer.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or AvoidDangerConfig()
    
    def calculate(
        self,
        prediction: float,
        features: Dict[str, float],
        portfolio: Any = None,
        pair: str = None,
    ) -> float:
        """
        Calculate target position size.
        
        Args:
            prediction: Model prediction (probability of price increase)
            features: Current features dict (must include regime)
            portfolio: Current portfolio state (optional)
            pair: Trading pair (optional)
        
        Returns:
            Target position (1.0 = full long, 0.0 = flat)
        """
        # Get current regime
        regime = features.get('regime_hybrid_simple', features.get('regime_hybrid', 1))
        regime = int(regime)
        
        # Check danger condition
        in_danger = self._is_danger(regime, prediction)
        
        if in_danger:
            return self.config.danger_position
        
        # Not in danger - return position
        if self.config.scale_by_prediction:
            # Scale position by prediction strength
            return self.config.full_position * prediction
        else:
            return self.config.full_position
    
    def _is_danger(self, regime: int, prediction: float) -> bool:
        """
        Check if current conditions indicate danger.
        
        Danger = in a danger regime AND prediction is bearish
        """
        # Check if in any danger regime
        in_danger_regime = (
            regime == self.config.danger_regime or 
            regime in self.config.danger_regimes
        )
        
        # Check if prediction is below threshold
        bearish_prediction = prediction < self.config.prediction_threshold
        
        return in_danger_regime and bearish_prediction
    
    def get_signal_details(
        self,
        prediction: float,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of signal.
        
        Useful for debugging and logging.
        """
        regime = features.get('regime_hybrid_simple', features.get('regime_hybrid', 1))
        regime = int(regime)
        
        in_danger_regime = (
            regime == self.config.danger_regime or 
            regime in self.config.danger_regimes
        )
        bearish_prediction = prediction < self.config.prediction_threshold
        in_danger = in_danger_regime and bearish_prediction
        
        position = self.calculate(prediction, features)
        
        return {
            'prediction': prediction,
            'regime': regime,
            'in_danger_regime': in_danger_regime,
            'bearish_prediction': bearish_prediction,
            'danger_signal': in_danger,
            'position': position,
            'action': 'EXIT' if in_danger else 'HOLD/ENTER',
        }
    
    def __repr__(self) -> str:
        return (
            f"AvoidDangerSizer(danger_regime={self.config.danger_regime}, "
            f"threshold={self.config.prediction_threshold})"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_v16_sizer() -> AvoidDangerSizer:
    """
    Create V16 strategy sizer.
    
    V16 = Avoid Regime 0 (Calm/Calm) when bearish.
    """
    config = AvoidDangerConfig(
        danger_regime=0,
        prediction_threshold=0.5,
        full_position=1.0,
        danger_position=0.0,
    )
    return AvoidDangerSizer(config=config)


def create_conservative_avoid_danger() -> AvoidDangerSizer:
    """
    Conservative version: Avoid more regimes.
    
    Avoids both Regime 0 (Calm/Calm) and Regime 3 (Volatile/Volatile).
    """
    config = AvoidDangerConfig(
        danger_regime=0,
        danger_regimes=[0, 3],  # Avoid calm consolidation AND crisis
        prediction_threshold=0.5,
        full_position=1.0,
        danger_position=0.0,
    )
    return AvoidDangerSizer(config=config)


def create_scaled_sizer() -> AvoidDangerSizer:
    """
    Scaled version: Position size varies with prediction.
    
    Instead of binary (0 or 1), uses prediction as position size.
    """
    config = AvoidDangerConfig(
        danger_regime=0,
        prediction_threshold=0.5,
        full_position=1.0,
        danger_position=0.0,
        scale_by_prediction=True,
    )
    return AvoidDangerSizer(config=config)