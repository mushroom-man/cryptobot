# -*- coding: utf-8 -*-
"""
CryptoBot - Multi-Timeframe Feature Engine
============================================
Main orchestrator for computing features across multiple timeframes.

Architecture:
    MultiTimeframeData (from DataLoader)
           │
           ▼
    ┌─────────────────────────────────────┐
    │   MultiTimeframeFeatureEngine       │
    ├─────────────────────────────────────┤
    │ 1. Per-TF features (mt_features)    │
    │ 2. Cross-TF features                │
    │ 3. Regime features (mt_regime)      │
    └─────────────────────────────────────┘
           │
           ▼
      DataFrame (~230+ features)

Usage:
    from cryptobot.features.mt_engine import MultiTimeframeFeatureEngine
    from cryptobot.datasources import DataLoader
    
    # Load data
    data = DataLoader.load(pairs=['XBTUSD'], timeframes=['1h', '24h', '168h'])
    
    # Compute features
    engine = MultiTimeframeFeatureEngine()
    df_features = engine.compute(data.get_aligned('XBTUSD'))
    
    # Or compute for all pairs
    all_features = engine.compute_all_pairs(data)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import time


# =============================================================================
# Constants
# =============================================================================

TIMEFRAMES = ['1h', '4h', '12h', '24h', '72h', '168h']

# Feature categories
CATEGORIES = ['ret', 'vol', 'rng', 'ma', 'mom', 'vlm']


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MTFeatureConfig:
    """Configuration for Multi-Timeframe Feature Engine."""
    
    # Timeframes to use
    timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '12h', '24h', '72h', '168h'])
    
    # Feature categories to compute
    categories: List[str] = field(default_factory=lambda: ['ret', 'vol', 'rng', 'ma', 'mom', 'vlm'])
    
    # Include regime features
    include_regime: bool = True
    
    # Include legacy regime (BinSeg, MSM, Hybrid)
    include_legacy_regime: bool = True
    
    # Include cross-timeframe features
    include_cross_tf: bool = True
    
    # Drop rows with NaN (warmup period)
    drop_na: bool = False
    
    # Verbose output
    verbose: bool = True


# Default configuration
DEFAULT_CONFIG = MTFeatureConfig()


# =============================================================================
# Cross-Timeframe Features
# =============================================================================

def compute_cross_tf_features(
    df: pd.DataFrame,
    timeframes: List[str],
) -> pd.DataFrame:
    """
    Compute cross-timeframe features.
    
    Features:
        x_vol_1h_v_24h      - Vol ratio short/medium
        x_vol_1h_v_168h     - Vol ratio short/long
        x_vol_24h_v_168h    - Vol ratio medium/long
        x_ret_agree         - Return direction agreement
        x_ma_agree_count    - # of TFs with bullish MA
        x_trend_strength    - Multi-TF trend strength
        x_ma_1h_vs_24h      - Short vs medium trend alignment
        x_ma_1h_vs_168h     - Short vs long trend alignment
        x_ma_24h_vs_168h    - Medium vs long trend alignment
        x_price_position    - Price position relative to all TF MAs
    
    Args:
        df: DataFrame with per-TF features
        timeframes: List of timeframes
    
    Returns:
        DataFrame with cross-TF features
    """
    result = pd.DataFrame(index=df.index)
    
    # === Volatility ratios ===
    vol_pairs = [
        ('1h', '24h'),
        ('1h', '168h'),
        ('24h', '168h'),
        ('4h', '72h'),
    ]
    
    for tf_short, tf_long in vol_pairs:
        if tf_short not in timeframes or tf_long not in timeframes:
            continue
        
        vol_short_col = f'vol_ann_{tf_short}'
        vol_long_col = f'vol_ann_{tf_long}'
        
        if vol_short_col in df.columns and vol_long_col in df.columns:
            ratio = df[vol_short_col] / df[vol_long_col].replace(0, np.nan)
            result[f'x_vol_{tf_short}_v_{tf_long}'] = ratio.fillna(1.0)
    
    # === Return agreement ===
    ret_cols = [f'ret_sign_{tf}' for tf in timeframes if f'ret_sign_{tf}' in df.columns]
    if len(ret_cols) >= 2:
        ret_df = df[ret_cols]
        
        # Count agreements (all same sign)
        ret_mode = ret_df.mode(axis=1)[0]
        ret_agree = (ret_df.eq(ret_mode, axis=0)).sum(axis=1)
        result['x_ret_agree'] = ret_agree
        
        # Average return sign
        result['x_ret_avg_sign'] = ret_df.mean(axis=1)
    
    # === MA agreement ===
    ma_cross_cols = [f'ma_cross_{tf}' for tf in timeframes if f'ma_cross_{tf}' in df.columns]
    if len(ma_cross_cols) >= 2:
        ma_df = df[ma_cross_cols]
        
        # Count bullish (ma_cross = 1)
        result['x_ma_agree_bullish'] = (ma_df == 1).sum(axis=1)
        
        # Count bearish (ma_cross = -1)
        result['x_ma_agree_bearish'] = (ma_df == -1).sum(axis=1)
        
        # Net score (-n to +n)
        result['x_ma_agree_score'] = ma_df.sum(axis=1)
    
    # === MA trend alignment (short vs long) ===
    ma_pairs = [
        ('1h', '24h'),
        ('1h', '168h'),
        ('24h', '168h'),
    ]
    
    for tf_short, tf_long in ma_pairs:
        if tf_short not in timeframes or tf_long not in timeframes:
            continue
        
        short_col = f'ma_cross_{tf_short}'
        long_col = f'ma_cross_{tf_long}'
        
        if short_col in df.columns and long_col in df.columns:
            # +1 = aligned, -1 = conflicting, 0 = one neutral
            alignment = df[short_col] * df[long_col]
            result[f'x_ma_{tf_short}_vs_{tf_long}'] = alignment
    
    # === Trend strength (weighted by timeframe) ===
    tf_weights = {'1h': 0.1, '4h': 0.12, '12h': 0.15, '24h': 0.18, '72h': 0.2, '168h': 0.25}
    
    # Find price_vs_ma columns (use middle MA period for each TF)
    price_vs_ma_cols = []
    for tf in timeframes:
        # Try different MA periods to find one that exists
        for period in [12, 6, 4, 3, 2, 1]:
            col = f'price_vs_ma_{period}_{tf}'
            if col in df.columns:
                price_vs_ma_cols.append((tf, col))
                break
    
    if price_vs_ma_cols:
        weighted_trend = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for tf, col in price_vs_ma_cols:
            weight = tf_weights.get(tf, 0.1)
            weighted_trend += (df[col] - 1.0) * weight  # Centered at 0
            total_weight += weight
        
        if total_weight > 0:
            result['x_trend_strength'] = weighted_trend / total_weight
    
    # === Price position (average distance from all TF MAs) ===
    if price_vs_ma_cols:
        price_positions = []
        for tf, col in price_vs_ma_cols:
            price_positions.append(df[col])
        
        # Average price vs MA across all TFs
        price_pos_df = pd.concat(price_positions, axis=1)
        result['x_price_position'] = price_pos_df.mean(axis=1)
        
        # Also compute std (dispersion - are TFs agreeing on price position?)
        result['x_price_position_std'] = price_pos_df.std(axis=1)
    
    # === Momentum alignment ===
    rsi_cols = [f'mom_rsi_{tf}' for tf in timeframes if f'mom_rsi_{tf}' in df.columns]
    if len(rsi_cols) >= 2:
        rsi_df = df[rsi_cols]
        
        # RSI convergence (std across TFs)
        result['x_rsi_convergence'] = rsi_df.std(axis=1)
        
        # Average RSI
        result['x_rsi_avg'] = rsi_df.mean(axis=1)
    
    return result


# =============================================================================
# Multi-Timeframe Feature Engine
# =============================================================================

class MultiTimeframeFeatureEngine:
    """
    Main engine for computing multi-timeframe features.
    
    Orchestrates:
        1. Per-timeframe feature computation
        2. Cross-timeframe features
        3. Regime detection and transition probability
    """
    
    def __init__(self, config: MTFeatureConfig = None):
        """
        Initialize the engine.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG
        self._last_computed_features: List[str] = []
    
    def compute(
        self,
        df: pd.DataFrame,
        timeframes: List[str] = None,
        categories: List[str] = None,
        include_ohlcv: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all multi-timeframe features.
        
        Args:
            df: DataFrame with aligned OHLCV data.
                Columns should be: open_1h, close_1h, ..., open_24h, close_24h, ...
            timeframes: Timeframes to compute (default: from config)
            categories: Feature categories (default: from config)
            include_ohlcv: Whether to include original OHLCV columns
        
        Returns:
            DataFrame with all features
        
        Example:
            >>> engine = MultiTimeframeFeatureEngine()
            >>> df_features = engine.compute(data.get_aligned('XBTUSD'))
            >>> print(f"Computed {len(df_features.columns)} features")
        """
        start_time = time.time()
        
        timeframes = timeframes or self.config.timeframes
        categories = categories or self.config.categories
        
        # Auto-detect available timeframes
        available_tfs = self._detect_timeframes(df)
        timeframes = [tf for tf in timeframes if tf in available_tfs]
        
        if not timeframes:
            raise ValueError(
                f"No valid timeframes found. Columns: {list(df.columns)[:10]}..."
            )
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Multi-Timeframe Feature Engine")
            print(f"{'='*60}")
            print(f"Timeframes: {timeframes}")
            print(f"Categories: {categories}")
            print(f"Input rows: {len(df):,}")
        
        # Start result DataFrame
        if include_ohlcv:
            ohlcv_cols = [c for c in df.columns if any(
                c.startswith(x) for x in ['open_', 'high_', 'low_', 'close_', 'volume_']
            )]
            result = df[ohlcv_cols].copy()
        else:
            result = pd.DataFrame(index=df.index)
        
        # === 1. Per-Timeframe Features ===
        if self.config.verbose:
            print(f"\n[1/3] Computing per-timeframe features...")
        
        from cryptobot.features.mt_features import compute_all_tf_features
        
        tf_features = compute_all_tf_features(
            df,
            timeframes=timeframes,
            categories=categories,
        )
        result = pd.concat([result, tf_features], axis=1)
        
        if self.config.verbose:
            print(f"      Added {len(tf_features.columns)} per-TF features")
        
        # === 2. Cross-Timeframe Features ===
        if self.config.include_cross_tf:
            if self.config.verbose:
                print(f"\n[2/3] Computing cross-timeframe features...")
            
            cross_features = compute_cross_tf_features(result, timeframes)
            result = pd.concat([result, cross_features], axis=1)
            
            if self.config.verbose:
                print(f"      Added {len(cross_features.columns)} cross-TF features")
        
        # === 3. Regime Features ===
        if self.config.include_regime:
            if self.config.verbose:
                print(f"\n[3/3] Computing regime features...")
            
            from cryptobot.features.mt_regime import compute_regime_features
            
            regime_features = compute_regime_features(
                result,
                timeframes=timeframes,
                include_legacy=self.config.include_legacy_regime,
            )
            result = pd.concat([result, regime_features], axis=1)
            
            if self.config.verbose:
                print(f"      Added {len(regime_features.columns)} regime features")
        
        # === Finalize ===
        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]
        
        # Track computed features (excluding OHLCV)
        ohlcv_prefixes = ('open_', 'high_', 'low_', 'close_', 'volume_', 'volume_quote_')
        self._last_computed_features = [
            c for c in result.columns if not any(c.startswith(p) for p in ohlcv_prefixes)
        ]
        
        # Drop NaN if requested
        if self.config.drop_na:
            result = result.dropna()
        
        elapsed = time.time() - start_time
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"COMPLETE")
            print(f"{'='*60}")
            print(f"Total features: {len(self._last_computed_features)}")
            print(f"Output rows: {len(result):,}")
            print(f"Time: {elapsed:.2f}s")
            print(f"{'='*60}")
        
        return result
    
    def compute_for_pair(
        self,
        data,  # MultiTimeframeData container
        pair: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute features for a specific pair from container.
        
        Args:
            data: MultiTimeframeData container from DataLoader
            pair: Trading pair symbol
            **kwargs: Additional arguments for compute()
        
        Returns:
            DataFrame with features for this pair
        """
        df_aligned = data.get_aligned(pair)
        return self.compute(df_aligned, **kwargs)
    
    def compute_all_pairs(
        self,
        data,  # MultiTimeframeData container
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute features for all pairs in container.
        
        Args:
            data: MultiTimeframeData container from DataLoader
            **kwargs: Additional arguments for compute()
        
        Returns:
            Dict mapping pair -> DataFrame with features
        """
        results = {}
        
        pairs = data.pairs
        
        for i, pair in enumerate(pairs):
            if self.config.verbose:
                print(f"\n[{i+1}/{len(pairs)}] Processing {pair}...")
            
            results[pair] = self.compute_for_pair(data, pair, **kwargs)
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names from last compute() call.
        
        Returns only computed features, NOT OHLCV columns.
        """
        return self._last_computed_features.copy()
    
    def get_feature_count(self) -> int:
        """Get number of features from last compute()."""
        return len(self._last_computed_features)
    
    def get_features_by_category(self) -> Dict[str, List[str]]:
        """
        Get features grouped by category.
        
        Returns:
            Dict mapping category -> list of feature names
        """
        features = self._last_computed_features
        
        categories = {
            'ret': [f for f in features if f.startswith('ret_')],
            'vol': [f for f in features if f.startswith('vol_')],
            'rng': [f for f in features if f.startswith('rng_')],
            'ma': [f for f in features if f.startswith('ma_') or f.startswith('price_vs_ma_')],
            'mom': [f for f in features if f.startswith('mom_')],
            'vlm': [f for f in features if f.startswith('vlm_')],
            'regime': [f for f in features if f.startswith('regime_')],
            'cross_tf': [f for f in features if f.startswith('x_')],
        }
        
        # Add "other" for uncategorized
        all_categorized = set(f for flist in categories.values() for f in flist)
        categories['other'] = [f for f in features if f not in all_categorized]
        
        return {k: v for k, v in categories.items() if v}
    
    def get_features_by_timeframe(self) -> Dict[str, List[str]]:
        """
        Get features grouped by timeframe.
        
        Returns:
            Dict mapping timeframe -> list of feature names
        """
        features = self._last_computed_features
        
        by_tf = {tf: [] for tf in TIMEFRAMES}
        by_tf['cross_tf'] = []
        by_tf['global'] = []
        
        for f in features:
            matched = False
            for tf in TIMEFRAMES:
                if f.endswith(f'_{tf}'):
                    by_tf[tf].append(f)
                    matched = True
                    break
            
            if not matched:
                if f.startswith('x_'):
                    by_tf['cross_tf'].append(f)
                else:
                    by_tf['global'].append(f)
        
        return {k: v for k, v in by_tf.items() if v}
    
    def info(self) -> None:
        """Print summary of last computed features."""
        print(f"\n{'='*60}")
        print("Multi-Timeframe Feature Engine - Summary")
        print(f"{'='*60}")
        print(f"Total features: {self.get_feature_count()}")
        
        print(f"\nBy category:")
        for cat, feats in self.get_features_by_category().items():
            print(f"  {cat}: {len(feats)}")
        
        print(f"\nBy timeframe:")
        for tf, feats in self.get_features_by_timeframe().items():
            print(f"  {tf}: {len(feats)}")
        
        print(f"{'='*60}")
    
    def _detect_timeframes(self, df: pd.DataFrame) -> List[str]:
        """Detect available timeframes from column names."""
        available = []
        for tf in TIMEFRAMES:
            if f'close_{tf}' in df.columns:
                available.append(tf)
        return available


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_mt_features(
    df: pd.DataFrame,
    timeframes: List[str] = None,
    categories: List[str] = None,
) -> pd.DataFrame:
    """
    Quick function to compute multi-timeframe features.
    
    Args:
        df: Aligned OHLCV DataFrame
        timeframes: Timeframes to use
        categories: Categories to compute
    
    Returns:
        DataFrame with features
    """
    engine = MultiTimeframeFeatureEngine()
    return engine.compute(df, timeframes=timeframes, categories=categories)


def get_feature_engine(config: MTFeatureConfig = None) -> MultiTimeframeFeatureEngine:
    """Get a configured feature engine instance."""
    return MultiTimeframeFeatureEngine(config)