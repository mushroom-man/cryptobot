# -*- coding: utf-8 -*-
"""
CryptoBot - Feature Engine
===========================
Main interface for computing features consistently across backtest and production.

Usage:
    from cryptobot.features import FeatureEngine
    
    # Initialize
    engine = FeatureEngine()
    
    # Compute specific features
    df = engine.compute(ohlcv_df, ['ma_score', 'rolling_vol_168h', 'regime_hybrid'])
    
    # Compute feature group
    df = engine.compute_group(ohlcv_df, 'trend')
    
    # Compute all strategy features
    df = engine.compute_strategy_features(ohlcv_df)
    
    # Get list of computed feature columns (not OHLCV)
    feature_cols = engine.get_computed_feature_names()
    
    # List available features
    print(engine.list_features())
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime
import warnings
import os

# Load environment for database
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class FeatureEngine:
    """
    Main feature computation engine.
    
    Handles:
        - Consistent feature computation for backtest and production
        - Dependency resolution between features
        - Caching computed features
        - Database integration (optional)
    """
    
    # Core OHLCV columns (only these are copied from input)
    OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    # Strategy features - updated for new regime architecture
    STRATEGY_FEATURES = [
        # Trend features
        'ma_score',
        'price_vs_sma_6',
        'price_vs_sma_24', 
        'price_vs_sma_72',
        # Volatility features
        'rolling_vol_168h',
        'garch_vol_simple',
        # Regime features (new architecture)
        'regime_structural',  # LASSO-based structural regime
        'regime_msm',         # MSM tactical regime
        'regime_hybrid',      # Combined 4-state regime
        'regime_multiplier',  # Position sizing multiplier
    ]
    
    def __init__(self, cache: bool = True):
        """
        Initialize Feature Engine.
        
        Args:
            cache: Whether to cache computed features
        """
        self.cache = cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._feature_classes = {}
        self._last_computed_features: List[str] = []
        
        # Import feature modules to register them
        self._load_features()
    
    def _load_features(self):
        """Import all feature modules to register features."""
        try:
            from cryptobot.features import base
            from cryptobot.features import technical
            from cryptobot.features import volatility
            from cryptobot.features import regime
            
            from cryptobot.features.base import _FEATURE_REGISTRY
            self._feature_classes = _FEATURE_REGISTRY
        except ImportError as e:
            warnings.warn(f"Could not import feature modules: {e}")
            self._feature_classes = {}
    
    def list_features(self) -> List[str]:
        """List all available feature names."""
        from cryptobot.features.base import list_features
        return list_features()
    
    def get_feature_info(self) -> pd.DataFrame:
        """Get detailed info about all registered features."""
        from cryptobot.features.base import get_feature_info
        return get_feature_info()
    
    def list_groups(self) -> List[str]:
        """List all available feature groups."""
        from cryptobot.features.base import FEATURE_GROUPS
        return list(FEATURE_GROUPS.keys())
    
    def get_group_features(self, group_name: str) -> List[str]:
        """Get feature names in a group."""
        from cryptobot.features.base import FEATURE_GROUPS
        if group_name not in FEATURE_GROUPS:
            raise KeyError(f"Group '{group_name}' not found. Available: {list(FEATURE_GROUPS.keys())}")
        return FEATURE_GROUPS[group_name].features
    
    def get_computed_feature_names(self) -> List[str]:
        """
        Get the list of feature names from the last compute() call.
        
        Returns only computed features, NOT OHLCV columns.
        
        Returns:
            List of feature column names
        """
        return self._last_computed_features.copy()
    
    def compute(
        self, 
        df: pd.DataFrame, 
        features: List[str],
        include_ohlcv: bool = True
    ) -> pd.DataFrame:
        """
        Compute specified features.
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
                Index should be timestamp
            features: List of feature names to compute
            include_ohlcv: Whether to include original OHLCV columns in output
        
        Returns:
            DataFrame with computed features (and optionally OHLCV)
        """
        from cryptobot.features.base import get_feature
        
        # Validate input
        self._validate_input(df)
        
        # Sort by index if not already
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        # Start with ONLY core OHLCV columns if requested (not all input columns!)
        if include_ohlcv:
            # Only copy the standard OHLCV columns, not any extra columns from database
            ohlcv_cols = [c for c in self.OHLCV_COLUMNS if c in df.columns]
            result = df[ohlcv_cols].copy()
        else:
            result = pd.DataFrame(index=df.index)
        
        # Track which features we're computing
        computed_features = []
        
        # Compute each feature
        for feature_name in features:
            try:
                feature_cls = get_feature(feature_name)
                feature = feature_cls()
                
                print(f"  Computing {feature_name}...", end=' ')
                feature_values = feature.compute(df)
                result[feature_name] = feature_values
                computed_features.append(feature_name)
                print("✓")
                
            except Exception as e:
                warnings.warn(f"Failed to compute '{feature_name}': {e}")
                result[feature_name] = np.nan
                computed_features.append(feature_name)
        
        # Store the list of computed features (not OHLCV)
        self._last_computed_features = computed_features
        
        return result
    
    def compute_group(
        self, 
        df: pd.DataFrame, 
        group_name: str,
        include_ohlcv: bool = True
    ) -> pd.DataFrame:
        """
        Compute all features in a feature group.
        
        Args:
            df: OHLCV DataFrame
            group_name: Name of feature group ('trend', 'volatility', 'regime', etc.)
            include_ohlcv: Whether to include original OHLCV columns
        
        Returns:
            DataFrame with group features
        """
        features = self.get_group_features(group_name)
        return self.compute(df, features, include_ohlcv=include_ohlcv)
    
    def compute_strategy_features(
        self, 
        df: pd.DataFrame,
        include_ohlcv: bool = True
    ) -> pd.DataFrame:
        """
        Compute all features needed for the trading strategy.
        
        Features:
            - Trend: ma_score, price_vs_sma_*
            - Volatility: rolling_vol_168h, garch_vol_simple
            - Regime: regime_structural, regime_msm, regime_hybrid, regime_multiplier
        
        Args:
            df: OHLCV DataFrame
            include_ohlcv: Whether to include original OHLCV columns
        
        Returns:
            DataFrame with strategy features
        """
        print(f"\nComputing strategy features ({len(self.STRATEGY_FEATURES)} features)...")
        return self.compute(df, self.STRATEGY_FEATURES, include_ohlcv=include_ohlcv)
    
    def compute_all(
        self, 
        df: pd.DataFrame,
        include_ohlcv: bool = True
    ) -> pd.DataFrame:
        """
        Compute ALL registered features.
        
        Warning: This may be slow for large datasets.
        """
        all_features = self.list_features()
        print(f"\nComputing all features ({len(all_features)} features)...")
        return self.compute(df, all_features, include_ohlcv=include_ohlcv)
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
    
    # =========================================================================
    # Database Integration
    # =========================================================================
    
    def compute_and_save(
        self,
        pair: str,
        features: List[str] = None,
        start: str = None,
        end: str = None
    ) -> pd.DataFrame:
        """
        Compute features for a pair and save to database.
        
        Args:
            pair: Trading pair (e.g., "XBTUSD")
            features: Features to compute (default: strategy features)
            start: Start date
            end: End date
        
        Returns:
            DataFrame with computed features
        """
        from cryptobot.datasources import Database
        
        db = Database()
        
        # Load OHLCV data
        print(f"Loading {pair} data from database...")
        df = db.get_ohlcv(pair, start=start, end=end)
        
        if len(df) == 0:
            raise ValueError(f"No data found for {pair}")
        
        print(f"Loaded {len(df):,} rows ({df.index.min()} to {df.index.max()})")
        
        # Compute features
        features = features or self.STRATEGY_FEATURES
        result = self.compute(df, features, include_ohlcv=False)
        
        # Save to database
        print(f"\nSaving features to database...")
        rows_saved = db.save_features(result, pair)
        print(f"Saved {rows_saved:,} rows")
        
        return result
    
    def load_or_compute(
        self,
        pair: str,
        features: List[str] = None,
        start: str = None,
        end: str = None,
        force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Load features from database if available, otherwise compute.
        
        Args:
            pair: Trading pair
            features: Features to load/compute
            start: Start date
            end: End date
            force_recompute: Force recomputation even if cached
        
        Returns:
            DataFrame with OHLCV + features
        """
        from cryptobot.datasources import Database
        
        db = Database()
        features = features or self.STRATEGY_FEATURES
        
        # Load OHLCV
        df = db.get_ohlcv(pair, start=start, end=end)
        
        if len(df) == 0:
            raise ValueError(f"No data found for {pair}")
        
        if not force_recompute:
            # Try to load cached features
            try:
                cached = db.get_features(pair, start=start, end=end)
                if len(cached) > 0:
                    # Check if all features are present
                    missing_features = [f for f in features if f not in cached.columns]
                    if not missing_features:
                        print(f"Loaded {len(cached):,} cached feature rows for {pair}")
                        return df.join(cached[features], how='left')
            except Exception:
                pass
        
        # Compute features
        print(f"Computing features for {pair}...")
        result = self.compute(df, features, include_ohlcv=True)
        
        return result
    
    # =========================================================================
    # Production Interface
    # =========================================================================
    
    def compute_latest(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        lookback: int = 504
    ) -> pd.Series:
        """
        Compute features for the latest row only.
        
        Optimized for production use - only keeps necessary lookback.
        
        Args:
            df: OHLCV DataFrame (should have at least `lookback` rows)
            features: Features to compute
            lookback: Number of historical rows needed
        
        Returns:
            Series with feature values for latest timestamp
        """
        features = features or self.STRATEGY_FEATURES
        
        # Only keep necessary lookback for efficiency
        if len(df) > lookback:
            df = df.iloc[-lookback:]
        
        # Compute features
        result = self.compute(df, features, include_ohlcv=False)
        
        # Return only latest row
        return result.iloc[-1]
    
    def get_max_lookback(self, features: List[str] = None) -> int:
        """
        Get maximum lookback period needed for given features.
        
        Useful for production to know how much history to load.
        """
        from cryptobot.features.base import get_feature
        
        features = features or self.STRATEGY_FEATURES
        max_lookback = 0
        
        for feature_name in features:
            try:
                feature_cls = get_feature(feature_name)
                max_lookback = max(max_lookback, feature_cls.lookback)
            except Exception:
                pass
        
        return max_lookback


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_features(
    df: pd.DataFrame, 
    features: List[str] = None
) -> pd.DataFrame:
    """
    Quick function to compute features.
    
    Usage:
        from cryptobot.features import compute_features
        df = compute_features(ohlcv_df, ['ma_score', 'rolling_vol_168h'])
    """
    engine = FeatureEngine()
    features = features or engine.STRATEGY_FEATURES
    return engine.compute(df, features)


def compute_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all strategy features.
    
    Usage:
        from cryptobot.features import compute_strategy_features
        df = compute_strategy_features(ohlcv_df)
    """
    engine = FeatureEngine()
    return engine.compute_strategy_features(df)