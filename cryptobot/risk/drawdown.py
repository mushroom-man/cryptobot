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
    from cryptobot.shared.sizing import AvoidDangerSizer
    
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

from cryptobot.shared.core.portfolio import Portfolio


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
    
    # Regime feature name
    regime_feature: str = 'regime_hybrid_simple'
    
    # Alternative: Use correct formula (binseg * 2 + msm)
    use_regime_components: bool = False       # If True, compute from binseg + msm


class AvoidDangerSizer:
    """
    Binary position sizer: 100% long unless danger signal.
    
    Implements V16 "Avoid Danger" strategy that outperformed buy-and-hold
    by +98.8% over a 3-year test period.
    
    Danger Signal:
        regime in danger_regimes AND prediction < threshold
    
    Attributes:
        config: AvoidDangerConfig with strategy parameters
        danger_count: Number of danger signals triggered
        total_bars: Total bars processed
    """
    
    def __init__(self, config: Optional[AvoidDangerConfig] = None, **kwargs):
        """
        Initialize sizer.
        
        Args:
            config: AvoidDangerConfig object
            **kwargs: Override config parameters directly
        """
        if config is None:
            config = AvoidDangerConfig()
        
        # Allow kwargs to override config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        self.danger_count = 0
        self.total_bars = 0
    
    def calculate(
        self,
        prediction: float,
        features: Dict[str, Any],
        portfolio: Portfolio,
        pair: str,
    ) -> float:
        """
        Calculate target position.
        
        Args:
            prediction: Model prediction (probability of price increase)
            features: Feature dict containing regime info
            portfolio: Current portfolio state
            pair: Trading pair
        
        Returns:
            Target position as fraction of equity (0.0 or 1.0)
        """
        self.total_bars += 1
        
        # Get regime
        if self.config.use_regime_components:
            # Compute from components
            binseg = features.get('regime_binseg', 0)
            msm = features.get('regime_msm', 0)
            regime = int(binseg) * 2 + int(msm)
        else:
            # Use pre-computed hybrid
            regime = int(features.get(self.config.regime_feature, -1))
        
        # Check danger condition
        is_danger = (
            regime in self.config.danger_regimes and
            prediction < self.config.prediction_threshold
        )
        
        if is_danger:
            self.danger_count += 1
            return self.config.danger_position
        else:
            return self.config.full_position
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sizer statistics."""
        return {
            'total_bars': self.total_bars,
            'danger_count': self.danger_count,
            'danger_pct': self.danger_count / self.total_bars if self.total_bars > 0 else 0,
            'time_in_market': 1 - (self.danger_count / self.total_bars) if self.total_bars > 0 else 1,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.danger_count = 0
        self.total_bars = 0


# Convenience functions
def create_v16_sizer() -> AvoidDangerSizer:
    """Create sizer with optimal V16 parameters."""
    return AvoidDangerSizer(config=AvoidDangerConfig(
        danger_regime=0,
        danger_regimes=[0],
        prediction_threshold=0.5,
        full_position=1.0,
        danger_position=0.0,
    ))


def create_conservative_sizer() -> AvoidDangerSizer:
    """Create conservative sizer that avoids multiple regimes."""
    return AvoidDangerSizer(config=AvoidDangerConfig(
        danger_regimes=[0, 2],  # Avoid Calm/Calm and Volatile/Calm
        prediction_threshold=0.5,
        full_position=1.0,
        danger_position=0.0,
    ))
