# -*- coding: utf-8 -*-
"""
CryptoBot - Feature Engineering Base
=====================================
Base classes and registry for extensible feature engineering.

To create a new feature:
    
    from cryptobot.features.base import Feature, register_feature
    
    @register_feature
    class MyFeature(Feature):
        name = "my_feature"
        lookback = 24  # hours needed
        
        def compute(self, df: pd.DataFrame) -> pd.Series:
            return df['close'].rolling(24).mean()

Usage:
    from cryptobot.features import FeatureEngine
    
    engine = FeatureEngine()
    df_with_features = engine.compute(df, ['ma_score', 'rolling_vol_168h'])
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Optional, Any
import warnings

# Global feature registry
_FEATURE_REGISTRY: Dict[str, Type['Feature']] = {}


def register_feature(cls: Type['Feature']) -> Type['Feature']:
    """
    Decorator to register a feature class.
    
    Usage:
        @register_feature
        class MyFeature(Feature):
            name = "my_feature"
            ...
    """
    if not hasattr(cls, 'name') or cls.name is None:
        raise ValueError(f"Feature class {cls.__name__} must define 'name' attribute")
    
    _FEATURE_REGISTRY[cls.name] = cls
    return cls


def get_feature(name: str) -> Type['Feature']:
    """Get a feature class by name."""
    if name not in _FEATURE_REGISTRY:
        available = list(_FEATURE_REGISTRY.keys())
        raise KeyError(f"Feature '{name}' not found. Available: {available}")
    return _FEATURE_REGISTRY[name]


def list_features() -> List[str]:
    """List all registered feature names."""
    return sorted(_FEATURE_REGISTRY.keys())


def get_feature_info() -> pd.DataFrame:
    """Get info about all registered features."""
    info = []
    for name, cls in _FEATURE_REGISTRY.items():
        info.append({
            'name': name,
            'lookback': cls.lookback,
            'output_type': cls.output_type,
            'description': cls.__doc__.strip().split('\n')[0] if cls.__doc__ else '',
            'class': cls.__name__,
        })
    return pd.DataFrame(info).set_index('name')


class Feature(ABC):
    """
    Base class for all features.
    
    Subclasses must define:
        - name: str - unique identifier
        - lookback: int - minimum rows needed before first valid output
        - compute(df) -> pd.Series or pd.DataFrame
    
    Optional:
        - output_type: str - 'continuous', 'discrete', 'binary'
        - dependencies: List[str] - other features this one requires
        - params: dict - configurable parameters
    """
    
    name: str = None
    lookback: int = 0
    output_type: str = 'continuous'  # 'continuous', 'discrete', 'binary'
    dependencies: List[str] = []
    
    def __init__(self, **params):
        """Initialize with optional parameters."""
        self.params = params
        # Override defaults with any passed params
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the feature from OHLCV data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be timestamp (sorted ascending)
        
        Returns:
            pd.Series with same index as input
        """
        pass
    
    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns."""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if not df.index.is_monotonic_increasing:
            warnings.warn("DataFrame index is not sorted. Sorting automatically.")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', lookback={self.lookback})"


class FeatureGroup:
    """
    A named group of related features.
    
    Usage:
        trend_features = FeatureGroup(
            name="trend",
            features=['ma_score', 'price_vs_sma_6', 'price_vs_sma_24']
        )
    """
    
    def __init__(self, name: str, features: List[str], description: str = ""):
        self.name = name
        self.features = features
        self.description = description
    
    def __repr__(self):
        return f"FeatureGroup('{self.name}', {len(self.features)} features)"


# Pre-defined feature groups (populated by feature modules)
FEATURE_GROUPS: Dict[str, FeatureGroup] = {}


def register_group(group: FeatureGroup) -> FeatureGroup:
    """Register a feature group."""
    FEATURE_GROUPS[group.name] = group
    return group


# =============================================================================
# Utility Functions
# =============================================================================

def compute_returns(df: pd.DataFrame, column: str = 'close', periods: int = 1) -> pd.Series:
    """Compute simple returns."""
    return df[column].pct_change(periods)


def compute_log_returns(df: pd.DataFrame, column: str = 'close', periods: int = 1) -> pd.Series:
    """Compute log returns."""
    return np.log(df[column] / df[column].shift(periods))


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=window, min_periods=window).std()


def compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()
