# -*- coding: utf-8 -*-
"""
CryptoBot - Kelly Criterion Position Sizing
============================================
Position sizing based on Kelly criterion with regime and MA adjustments.

From strategy document:
    - Base Kelly from LSTM probability
    - Regime multiplier: State 0=0.8, State 1=1.0, State 2=0.5, State 3=0.3
    - MA score multiplier: Score 0=0.3, Score 1=0.5, Score 2=0.7, Score 3=1.0
    - Volatility scaling disabled for crypto (high native volatility)
    - Fractional Kelly (0.25-0.5) for smoother equity curve

Usage:
    from cryptobot.risk.position import KellySizer
    
    sizer = KellySizer(
        kelly_fraction=0.25,
        max_position=0.30,
    )
    
    target = sizer.calculate(
        prediction=0.65,
        features={'regime_hybrid_simple': 1, 'ma_score': 3},
        portfolio=portfolio,
        pair="XBTUSD",
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np


@dataclass
class SizingConfig:
    """Configuration for position sizing."""
    
    # Kelly settings
    kelly_fraction: float = 0.25          # Fractional Kelly (0.25 = quarter Kelly)
    min_edge: float = 0.02                # Minimum edge to take position
    
    # Position limits
    max_position: float = 0.30            # Max position as fraction of equity
    min_position: float = 0.005           # Min position (below = 0) - lowered for crypto
    
    # Regime multipliers (from strategy doc)
    regime_multipliers: Dict[int, float] = field(default_factory=lambda: {
        0: 0.8,   # Calm/Calm - quiet consolidation
        1: 1.0,   # Calm/Volatile - breakout (best)
        2: 0.5,   # Volatile/Calm - recovery
        3: 0.3,   # Volatile/Volatile - crisis
    })
    
    # MA score multipliers - raised floor to prevent position crushing
    # Note: For shorts, score is inverted (0 becomes 3, etc.)
    ma_multipliers: Dict[int, float] = field(default_factory=lambda: {
        0: 0.3,   # All MAs bearish (raised from 0.1)
        1: 0.5,   # 1 MA bullish (raised from 0.4)
        2: 0.7,   # 2 MAs bullish
        3: 1.0,   # All MAs bullish
    })
    
    # Volatility scaling - disabled by default for crypto
    # Crypto vol (~80-100% annualized) vs stocks (~20%)
    use_vol_scaling: bool = False
    target_vol: float = 0.80              # Crypto-appropriate target (if enabled)
    vol_scale_min: float = 0.5            # Minimum vol scale factor
    vol_scale_max: float = 1.5            # Maximum vol scale factor
    
    # Direction
    p_threshold: float = 0.50             # Above = long, below = short
    allow_short: bool = True              # Allow short positions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'kelly_fraction': self.kelly_fraction,
            'min_edge': self.min_edge,
            'max_position': self.max_position,
            'min_position': self.min_position,
            'regime_multipliers': self.regime_multipliers,
            'ma_multipliers': self.ma_multipliers,
            'use_vol_scaling': self.use_vol_scaling,
            'target_vol': self.target_vol,
            'p_threshold': self.p_threshold,
            'allow_short': self.allow_short,
        }


class KellySizer:
    """
    Kelly criterion position sizer.
    
    Calculates optimal position size based on:
        1. Base Kelly from prediction probability
        2. Regime adjustment
        3. MA score adjustment
        4. Volatility scaling
        5. Fractional Kelly for risk reduction
    
    Formula:
        kelly = (p * b - q) / b
        
        where:
            p = probability of win
            q = 1 - p = probability of loss
            b = win/loss ratio (assumed 1:1 for simplicity)
        
        Simplified for binary outcomes:
            kelly = 2p - 1  (when win/loss ratio = 1)
        
        Final position:
            position = kelly * kelly_fraction * regime_mult * ma_mult * vol_scale
    """
    
    def __init__(self, config: Optional[SizingConfig] = None):
        """
        Initialize Kelly sizer.
        
        Args:
            config: Sizing configuration (uses defaults if None)
        """
        self.config = config or SizingConfig()
    
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
            features: Current features dict
            portfolio: Current portfolio state (optional, for interface compatibility)
            pair: Trading pair (optional, for interface compatibility)
        
        Returns:
            Target position as fraction of equity (signed: + long, - short)
        """
        # 1. Determine direction and edge
        if prediction > self.config.p_threshold:
            direction = 1.0
            p_win = prediction
        elif prediction < self.config.p_threshold and self.config.allow_short:
            direction = -1.0
            p_win = 1.0 - prediction
        else:
            return 0.0
        
        # 2. Calculate base Kelly
        edge = p_win - 0.5
        
        # Check minimum edge
        if edge < self.config.min_edge:
            return 0.0
        
        # Kelly for binary outcome with 1:1 payoff
        base_kelly = 2 * p_win - 1  # Equivalent to 2 * edge
        
        # 3. Apply fractional Kelly
        position = base_kelly * self.config.kelly_fraction
        
        # 4. Apply regime multiplier
        regime = features.get('regime_hybrid_simple', features.get('regime_hybrid', 1))
        regime_mult = self.config.regime_multipliers.get(int(regime), 0.5)
        position *= regime_mult
        
        # 5. Apply MA score multiplier
        ma_score = features.get('ma_score', 2)
        
        # For short positions, invert MA score (bearish = good)
        if direction < 0:
            ma_score = 3 - ma_score
        
        ma_mult = self.config.ma_multipliers.get(int(ma_score), 0.5)
        position *= ma_mult
        
        # 6. Apply volatility scaling
        if self.config.use_vol_scaling:
            vol = features.get('garch_vol_simple', features.get('rolling_vol_168h'))
            if vol and vol > 0:
                vol_scale = self.config.target_vol / vol
                vol_scale = np.clip(vol_scale, self.config.vol_scale_min, self.config.vol_scale_max)
                position *= vol_scale
        
        # 7. Apply direction
        position *= direction
        
        # 8. Apply position limits
        position = np.clip(position, -self.config.max_position, self.config.max_position)
        
        # 9. Check minimum position
        if abs(position) < self.config.min_position:
            position = 0.0
        
        return position
    
    def calculate_kelly(self, p_win: float, win_loss_ratio: float = 1.0) -> float:
        """
        Calculate raw Kelly criterion.
        
        Args:
            p_win: Probability of winning
            win_loss_ratio: Ratio of win amount to loss amount
        
        Returns:
            Optimal fraction to bet
        """
        if p_win <= 0 or p_win >= 1:
            return 0.0
        
        q = 1 - p_win
        kelly = (p_win * win_loss_ratio - q) / win_loss_ratio
        
        return max(0, kelly)
    
    def size_in_units(
        self,
        position_fraction: float,
        equity: float,
        price: float,
    ) -> float:
        """
        Convert position fraction to units.
        
        Args:
            position_fraction: Position as fraction of equity (from calculate())
            equity: Portfolio equity
            price: Current price
        
        Returns:
            Position size in units (e.g., BTC)
        """
        if price <= 0:
            return 0.0
        
        notional = equity * abs(position_fraction)
        units = notional / price
        
        if position_fraction < 0:
            units = -units
        
        return units
    
    def get_sizing_breakdown(
        self,
        prediction: float,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of sizing calculation.
        
        Useful for debugging and understanding position sizes.
        """
        # Direction
        if prediction > self.config.p_threshold:
            direction = "LONG"
            p_win = prediction
        elif prediction < self.config.p_threshold and self.config.allow_short:
            direction = "SHORT"
            p_win = 1.0 - prediction
        else:
            direction = "FLAT"
            p_win = 0.5
        
        edge = p_win - 0.5
        base_kelly = 2 * p_win - 1
        fractional_kelly = base_kelly * self.config.kelly_fraction
        
        regime = features.get('regime_hybrid_simple', 1)
        regime_mult = self.config.regime_multipliers.get(int(regime), 0.5)
        
        ma_score = features.get('ma_score', 2)
        if direction == "SHORT":
            ma_score = 3 - ma_score
        ma_mult = self.config.ma_multipliers.get(int(ma_score), 0.5)
        
        vol = features.get('garch_vol_simple', features.get('rolling_vol_168h', 0.5))
        if vol and vol > 0:
            vol_scale = np.clip(
                self.config.target_vol / vol,
                self.config.vol_scale_min,
                self.config.vol_scale_max
            )
        else:
            vol_scale = 1.0
        
        final_position = fractional_kelly * regime_mult * ma_mult * vol_scale
        final_position = np.clip(final_position, -self.config.max_position, self.config.max_position)
        
        if direction == "SHORT":
            final_position = -abs(final_position)
        elif direction == "FLAT":
            final_position = 0.0
        
        return {
            'prediction': prediction,
            'direction': direction,
            'p_win': p_win,
            'edge': edge,
            'base_kelly': base_kelly,
            'fractional_kelly': fractional_kelly,
            'regime': int(regime),
            'regime_mult': regime_mult,
            'ma_score': int(ma_score),
            'ma_mult': ma_mult,
            'volatility': vol,
            'vol_scale': vol_scale,
            'final_position': final_position,
            'final_position_pct': f"{final_position:.1%}",
        }
    
    def __repr__(self) -> str:
        return (
            f"KellySizer(fraction={self.config.kelly_fraction}, "
            f"max={self.config.max_position:.0%})"
        )


# =============================================================================
# Alternative Sizers
# =============================================================================

class FixedSizer:
    """
    Fixed position size regardless of prediction.
    
    Useful as baseline comparison.
    """
    
    def __init__(
        self, 
        position_size: float = 0.10,
        p_threshold: float = 0.50,
        allow_short: bool = True,
    ):
        self.position_size = position_size
        self.p_threshold = p_threshold
        self.allow_short = allow_short
    
    def calculate(
        self,
        prediction: float,
        features: Dict[str, float],
        portfolio: Any = None,
        pair: str = None,
    ) -> float:
        if prediction > self.p_threshold:
            return self.position_size
        elif prediction < self.p_threshold and self.allow_short:
            return -self.position_size
        return 0.0


class VolatilityTargetSizer:
    """
    Size positions to target specific volatility contribution.
    """
    
    def __init__(
        self,
        target_vol: float = 0.15,
        max_position: float = 0.50,
        p_threshold: float = 0.50,
    ):
        self.target_vol = target_vol
        self.max_position = max_position
        self.p_threshold = p_threshold
    
    def calculate(
        self,
        prediction: float,
        features: Dict[str, float],
        portfolio: Any = None,
        pair: str = None,
    ) -> float:
        # Get current volatility
        vol = features.get('rolling_vol_168h', features.get('garch_vol_simple', 0.5))
        
        if vol <= 0:
            return 0.0
        
        # Size to target volatility
        position = self.target_vol / vol
        position = min(position, self.max_position)
        
        # Apply direction
        if prediction > self.p_threshold:
            return position
        elif prediction < self.p_threshold:
            return -position
        
        return 0.0


# =============================================================================
# Factory Functions
# =============================================================================

def create_kelly_sizer(
    kelly_fraction: float = 0.25,
    max_position: float = 0.30,
    use_regime: bool = True,
    use_ma: bool = True,
    use_vol_scaling: bool = True,
) -> KellySizer:
    """Create Kelly sizer with custom settings."""
    config = SizingConfig(
        kelly_fraction=kelly_fraction,
        max_position=max_position,
        use_vol_scaling=use_vol_scaling,
    )
    
    if not use_regime:
        config.regime_multipliers = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    
    if not use_ma:
        config.ma_multipliers = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    
    return KellySizer(config=config)


def create_conservative_sizer() -> KellySizer:
    """Create conservative Kelly sizer."""
    config = SizingConfig(
        kelly_fraction=0.15,
        max_position=0.20,
        min_edge=0.05,
    )
    return KellySizer(config=config)


def create_aggressive_sizer() -> KellySizer:
    """Create aggressive Kelly sizer."""
    config = SizingConfig(
        kelly_fraction=0.50,
        max_position=0.50,
        min_edge=0.01,
    )
    return KellySizer(config=config)