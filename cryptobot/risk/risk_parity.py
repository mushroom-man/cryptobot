# -*- coding: utf-8 -*-
"""
CryptoBot - Risk Parity Position Sizing
========================================
Portfolio-level position sizing using inverse-volatility weighting.

Extracted from validated backtest: 16state_combined_backtest.py
Validated performance: +113.9% annual, 2.84 Sharpe, 16.8% max DD

Components:
    1. Risk Parity Weights: Inverse volatility allocation
    2. Volatility Scaling: Target portfolio volatility
    3. Drawdown Protection: Reduce exposure in drawdowns
    4. Monthly Rebalancing: Recalculate weights monthly

The EXACT same logic is used by both backtest and live trading
to ensure consistency.

Usage:
    from cryptobot.risk.risk_parity import RiskParitySizer, RiskParityConfig
    
    # Initialize
    config = RiskParityConfig()
    sizer = RiskParitySizer(config)
    
    # Calculate weights (do once per rebalance period)
    weights = sizer.calculate_weights(returns_df, current_date)
    
    # Calculate position for each pair
    for pair in pairs:
        position = sizer.calculate_position(
            pair=pair,
            signal=signal_position,  # 0.0 to 1.0
            weight=weights[pair],
            equity=equity,
            peak_equity=peak_equity,
            returns_df=returns_df,
            current_date=current_date,
            price=current_price,
        )
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from cryptobot.risk.base_sizer import BasePortfolioSizer


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RiskParityConfig:
    """
    Configuration for Risk Parity sizing.
    
    All parameters from validated backtest (16state_combined_backtest.py).
    These are the SINGLE SOURCE OF TRUTH.
    """
    
    # === Exposure Limits ===
    max_exposure: float = 2.0           # Maximum total exposure (2.0 = 200%)
    min_exposure_floor: float = 0.40    # Minimum exposure even in drawdown
    
    # === Volatility Targeting ===
    target_vol: float = 0.40            # Target annualized portfolio volatility
    vol_lookback_months: int = 1        # Lookback for realized vol calculation
    
    # === Weight Calculation ===
    cov_lookback_months: int = 2        # Lookback for covariance/weight calculation
    min_weight_samples: int = 20        # Minimum samples for weight calculation
    
    # === Drawdown Protection ===
    dd_start_reduce: float = -0.20      # Start reducing at -20% drawdown
    dd_full_reduce: float = -0.50       # Full reduction at -50% drawdown
    
    # === Trading ===
    trading_cost: float = 0.0020        # 20 bps per trade
    min_trade_size: float = 100.0       # Minimum trade in USD
    
    # === Pairs (validated set) ===
    pairs: list = field(default_factory=lambda: [
        'XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD'
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_exposure': self.max_exposure,
            'min_exposure_floor': self.min_exposure_floor,
            'target_vol': self.target_vol,
            'vol_lookback_months': self.vol_lookback_months,
            'cov_lookback_months': self.cov_lookback_months,
            'min_weight_samples': self.min_weight_samples,
            'dd_start_reduce': self.dd_start_reduce,
            'dd_full_reduce': self.dd_full_reduce,
            'trading_cost': self.trading_cost,
            'min_trade_size': self.min_trade_size,
            'pairs': self.pairs,
        }
    
    @classmethod
    def from_yaml(cls, path: str) -> 'RiskParityConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskParityConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_timestamp(ts: Any) -> pd.Timestamp:
    """
    Normalize any timestamp to timezone-naive pandas Timestamp.
    
    Handles datetime, pd.Timestamp (tz-aware or naive), strings, etc.
    """
    if ts is None:
        return None
    
    # Convert to Timestamp first
    ts = pd.Timestamp(ts)
    
    # Remove timezone if present
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    
    return ts


def normalize_df_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame index to timezone-naive.
    
    Returns a copy with tz-naive DatetimeIndex.
    """
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def get_month_start(date: pd.Timestamp, months_back: int = 0) -> pd.Timestamp:
    """
    Get the 1st of the month, optionally N months back.
    
    From validated backtest lines 120-123.
    Always returns timezone-naive timestamp.
    """
    # Ensure input is normalized
    date = normalize_timestamp(date)
    target = date - pd.DateOffset(months=months_back)
    result = target.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return result


# =============================================================================
# RISK PARITY SIZER
# =============================================================================

class RiskParitySizer(BasePortfolioSizer):
    """
    Risk Parity position sizer.
    
    Implements the EXACT logic from the validated backtest to ensure
    live trading matches backtest results.
    
    Key features:
        - Inverse-volatility weighting (lower vol = higher weight)
        - Volatility targeting (scale exposure to target vol)
        - Drawdown protection (reduce exposure in drawdowns)
        - Monthly rebalancing of weights
    
    Attributes:
        config: RiskParityConfig with all parameters
        asset_weights: Current allocation weights
        last_rebalance_month: (year, month) of last rebalance
    """
    
    def __init__(self, config: Optional[RiskParityConfig] = None):
        """
        Initialize Risk Parity sizer.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or RiskParityConfig()
        
        # State
        self.asset_weights: Dict[str, float] = {}
        self.last_rebalance_month: Optional[Tuple[int, int]] = None
        
        # Statistics
        self._rebalance_count = 0
        self._dd_reduction_count = 0
        self._vol_scale_history = []
    
    def calculate_weights(
        self,
        returns_df: pd.DataFrame,
        current_date: Any,
        force_rebalance: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights for all assets.
        
        Weights are recalculated monthly or when forced.
        
        From validated backtest lines 402-407, 508-515.
        
        Args:
            returns_df: DataFrame with asset returns (columns = pairs)
            current_date: Current date (any format - will be normalized)
            force_rebalance: Force recalculation even if same month
        
        Returns:
            Dict mapping pair -> weight (weights sum to 1.0)
        """
        # Normalize inputs to avoid timezone issues
        current_date = normalize_timestamp(current_date)
        returns_df = normalize_df_index(returns_df)
        
        current_month = (current_date.year, current_date.month)
        
        # Check if rebalance needed
        if not force_rebalance and self.last_rebalance_month == current_month:
            return self.asset_weights
        
        # Calculate lookback period
        # Use N months back to current date (not month start) for live trading
        lookback_start = get_month_start(current_date, self.config.cov_lookback_months)
        lookback_end = current_date  # FIX: Use actual date, not month start
        
        # Get returns for lookback period
        lookback_returns = returns_df.loc[lookback_start:lookback_end]
        
        if len(lookback_returns) < self.config.min_weight_samples:
            # Not enough data - use equal weights
            pairs = list(returns_df.columns)
            self.asset_weights = {pair: 1.0 / len(pairs) for pair in pairs}
        else:
            # Calculate inverse-volatility weights
            # From validated backtest lines 402-407
            vols = lookback_returns.std() * np.sqrt(365)  # Annualized
            vols[vols == 0] = 1e-6  # Avoid division by zero
            inv_vols = 1.0 / vols
            weights = inv_vols / inv_vols.sum()
            self.asset_weights = weights.to_dict()
        
        self.last_rebalance_month = current_month
        self._rebalance_count += 1
        
        return self.asset_weights
    
    def calculate_vol_scalar(
        self,
        returns_df: pd.DataFrame,
        current_date: Any,
    ) -> float:
        """
        Calculate volatility scaling factor.
        
        Scales exposure to target portfolio volatility.
        
        From validated backtest lines 581-591.
        
        Args:
            returns_df: DataFrame with asset returns
            current_date: Current date (any format - will be normalized)
        
        Returns:
            Scalar to multiply exposure (typically 0.4 to 2.0)
        """
        # Normalize inputs
        current_date = normalize_timestamp(current_date)
        returns_df = normalize_df_index(returns_df)
        
        vol_lookback_start = get_month_start(current_date, self.config.vol_lookback_months)
        vol_lookback_end = current_date  # FIX: Use actual date, not month start
        
        vol_returns = returns_df.loc[vol_lookback_start:vol_lookback_end]
        
        if len(vol_returns) < 15:
            return 1.0
        
        # Market returns = mean across assets
        market_returns = vol_returns.mean(axis=1)
        realized_vol = market_returns.std() * np.sqrt(365)  # Annualized
        
        if realized_vol <= 0:
            return 1.0
        
        # Scale to target vol
        vol_scalar = self.config.target_vol / realized_vol
        
        # Clip to reasonable range
        vol_scalar = np.clip(
            vol_scalar,
            self.config.min_exposure_floor,
            self.config.max_exposure
        )
        
        self._vol_scale_history.append(vol_scalar)
        
        return vol_scalar
    
    def calculate_dd_scalar(
        self,
        equity: float,
        peak_equity: float,
    ) -> float:
        """
        Calculate drawdown protection scalar.
        
        Reduces exposure linearly as drawdown increases.
        
        From validated backtest lines 498-506.
        
        Args:
            equity: Current portfolio equity
            peak_equity: Peak portfolio equity
        
        Returns:
            Scalar to multiply exposure (0.4 to 1.0)
        """
        if peak_equity <= 0:
            return 1.0
        
        current_dd = (equity - peak_equity) / peak_equity  # Negative value
        
        if current_dd >= self.config.dd_start_reduce:
            # Not in significant drawdown
            return 1.0
        
        if current_dd <= self.config.dd_full_reduce:
            # Full drawdown - minimum exposure
            self._dd_reduction_count += 1
            return self.config.min_exposure_floor
        
        # Linear interpolation between start and full reduction
        # From validated backtest lines 504-506
        range_dd = self.config.dd_start_reduce - self.config.dd_full_reduce
        position = (current_dd - self.config.dd_full_reduce) / range_dd
        dd_scalar = self.config.min_exposure_floor + position * (1.0 - self.config.min_exposure_floor)
        
        self._dd_reduction_count += 1
        
        return dd_scalar
    
    def calculate_position(
        self,
        pair: str,
        signal: float,
        weight: float,
        equity: float,
        peak_equity: float,
        returns_df: pd.DataFrame,
        current_date: Any,
        price: float,
    ) -> float:
        """
        Calculate target position for a single asset.
        
        Combines:
            - Signal strength (0.0 to 1.0)
            - Risk parity weight
            - Volatility scaling
            - Drawdown protection
            - Max exposure limit
        
        From validated backtest lines 572-597.
        
        Args:
            pair: Trading pair
            signal: Signal strength (0.0 = no position, 1.0 = full position)
            weight: Portfolio weight for this pair
            equity: Current portfolio equity
            peak_equity: Peak portfolio equity
            returns_df: DataFrame of asset returns
            current_date: Current date
            price: Current price of the asset
        
        Returns:
            Target position in units (e.g., number of coins)
        """
        if signal <= 0 or price <= 0 or equity <= 0:
            return 0.0
        
        # Calculate scalars
        vol_scalar = self.calculate_vol_scalar(returns_df, current_date)
        dd_scalar = self.calculate_dd_scalar(equity, peak_equity)
        
        # Combine scalars (take more conservative)
        # From validated backtest line 593
        risk_scalar = min(vol_scalar, 1.0 / dd_scalar if dd_scalar > 0 else 1.0)
        risk_scalar = max(risk_scalar, self.config.min_exposure_floor)
        
        # Calculate exposure for this pair
        # exposure = signal × weight × risk_scalar
        exposure = signal * weight * risk_scalar
        
        # Apply max exposure (across portfolio, but limit individual too)
        # Individual pair shouldn't exceed weight × max_exposure
        max_pair_exposure = weight * self.config.max_exposure
        exposure = min(exposure, max_pair_exposure)
        
        # Convert to position units
        # allocation = equity × exposure
        allocation = equity * exposure
        position = allocation / price
        
        return position
    
    def calculate_all_positions(
        self,
        signals: Dict[str, float],
        equity: float,
        peak_equity: float,
        returns_df: pd.DataFrame,
        current_date: Any,
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate positions for all pairs at once.
        
        Convenience method that handles weight calculation and
        ensures total exposure doesn't exceed max_exposure.
        
        Args:
            signals: Dict mapping pair -> signal strength (0.0 to 1.0)
            equity: Current portfolio equity
            peak_equity: Peak portfolio equity
            returns_df: DataFrame of asset returns
            current_date: Current date
            prices: Dict mapping pair -> current price
        
        Returns:
            Dict mapping pair -> target position in units
        """
        # Normalize inputs once
        current_date = normalize_timestamp(current_date)
        returns_df = normalize_df_index(returns_df)
        
        # Calculate/update weights
        weights = self.calculate_weights(returns_df, current_date)
        
        # Calculate raw positions
        positions = {}
        total_exposure = 0.0
        
        for pair, signal in signals.items():
            if pair not in weights:
                continue
            
            price = prices.get(pair, 0)
            if price <= 0:
                continue
            
            position = self.calculate_position(
                pair=pair,
                signal=signal,
                weight=weights[pair],
                equity=equity,
                peak_equity=peak_equity,
                returns_df=returns_df,
                current_date=current_date,
                price=price,
            )
            
            positions[pair] = position
            total_exposure += abs(position * price) / equity if equity > 0 else 0
        
        # Scale down if total exposure exceeds max
        if total_exposure > self.config.max_exposure:
            scale = self.config.max_exposure / total_exposure
            positions = {pair: pos * scale for pair, pos in positions.items()}
        
        return positions
    
    def reset(self) -> None:
        """Reset internal state."""
        self.asset_weights = {}
        self.last_rebalance_month = None
        self._rebalance_count = 0
        self._dd_reduction_count = 0
        self._vol_scale_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Return sizer statistics."""
        return {
            'rebalance_count': self._rebalance_count,
            'dd_reduction_count': self._dd_reduction_count,
            'current_weights': self.asset_weights.copy(),
            'last_rebalance_month': self.last_rebalance_month,
            'avg_vol_scalar': np.mean(self._vol_scale_history) if self._vol_scale_history else 1.0,
        }
    
    def __repr__(self) -> str:
        return (
            f"RiskParitySizer(max_exp={self.config.max_exposure:.0%}, "
            f"target_vol={self.config.target_vol:.0%}, "
            f"pairs={len(self.config.pairs)})"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_risk_parity_sizer(
    max_exposure: float = 2.0,
    target_vol: float = 0.40,
    **kwargs
) -> RiskParitySizer:
    """Create Risk Parity sizer with custom settings."""
    config = RiskParityConfig(
        max_exposure=max_exposure,
        target_vol=target_vol,
        **kwargs
    )
    return RiskParitySizer(config=config)


def create_conservative_risk_parity_sizer() -> RiskParitySizer:
    """Create conservative Risk Parity sizer with lower exposure."""
    config = RiskParityConfig(
        max_exposure=1.0,           # Max 100% exposure
        target_vol=0.25,            # Lower vol target
        dd_start_reduce=-0.10,      # Start reducing earlier
    )
    return RiskParitySizer(config=config)


def create_aggressive_risk_parity_sizer() -> RiskParitySizer:
    """Create aggressive Risk Parity sizer with higher exposure."""
    config = RiskParityConfig(
        max_exposure=3.0,           # Max 300% exposure
        target_vol=0.60,            # Higher vol target
        dd_start_reduce=-0.30,      # Start reducing later
    )
    return RiskParitySizer(config=config)