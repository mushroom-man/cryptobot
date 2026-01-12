# -*- coding: utf-8 -*-
"""
CryptoBot - Feature Engineering Module
=======================================
Extensible feature computation for backtesting and production.

Two Engines Available:

1. FeatureEngine (Original)
   - Single-timeframe features
   - Rolling window calculations on 1h data
   - Good for: Simple backtests, production signals

2. MultiTimeframeFeatureEngine (New)
   - Multi-timeframe features across 1h, 4h, 12h, 24h, 72h, 168h
   - Cross-timeframe analysis
   - Regime detection with transition probability
   - Good for: Comprehensive backtests, feature selection

Quick Start - Multi-Timeframe (Recommended):
    from cryptobot.datasources import DataLoader
    from cryptobot.features import MultiTimeframeFeatureEngine
    
    # Load multi-TF data
    data = DataLoader.load(
        pairs=['XBTUSD'],
        timeframes=['1h', '24h', '72h', '168h'],
        start='2020-01-01',
        end='2024-12-31',
    )
    
    # Compute ~230 features
    engine = MultiTimeframeFeatureEngine()
    df_features = engine.compute(data.get_aligned('XBTUSD'))
    
    # Check what we computed
    engine.info()

Quick Start - Single Timeframe:
    from cryptobot.features import FeatureEngine
    
    engine = FeatureEngine()
    df = engine.compute(ohlcv_df, ['ma_score', 'rolling_vol_168h'])

Feature Categories:
    ret_   - Returns (log, absolute, sign, cumulative)
    vol_   - Volatility (std, annualized, Parkinson, Garman-Klass)
    rng_   - Range (high-low, ATR, candle body)
    ma_    - Moving Averages (6/12/24 period, ratios, slopes)
    mom_   - Momentum (RSI, ROC, acceleration, streaks)
    vlm_   - Volume (MA, ratio, trend, price correlation)
    x_     - Cross-timeframe (vol ratios, trend agreement)
    regime_- Regime (per-TF, composite, transition probability)

Adding Custom Features:
    from cryptobot.features.base import Feature, register_feature
    
    @register_feature
    class MyFeature(Feature):
        name = "my_custom_feature"
        lookback = 24
        
        def compute(self, df):
            return df['close'].rolling(24).mean()
"""

# =============================================================================
# Base Classes and Utilities
# =============================================================================

from cryptobot.features.base import (
    Feature,
    FeatureGroup,
    register_feature,
    register_group,
    get_feature,
    list_features,
    get_feature_info,
    FEATURE_GROUPS,
    # Utility functions
    compute_returns,
    compute_log_returns,
    compute_sma,
    compute_ema,
    compute_rolling_std,
    compute_rolling_zscore,
    compute_atr,
)

# =============================================================================
# Import Feature Modules (registers features automatically)
# =============================================================================

from cryptobot.features import technical
from cryptobot.features import volatility
from cryptobot.features import regime

# =============================================================================
# Original Single-TF Engine
# =============================================================================

from cryptobot.features.engine import (
    FeatureEngine,
    compute_features,
    compute_strategy_features,
)

# =============================================================================
# Multi-Timeframe Engine (New)
# =============================================================================

from cryptobot.features.mt_features import (
    compute_returns as compute_mt_returns,
    compute_volatility as compute_mt_volatility,
    compute_range as compute_mt_range,
    compute_moving_averages as compute_mt_moving_averages,
    compute_momentum as compute_mt_momentum,
    compute_volume as compute_mt_volume,
    compute_tf_features,
    compute_all_tf_features,
    TIMEFRAMES,
    MA_PERIODS,
)

from cryptobot.features.mt_regime import (
    compute_vol_regime,
    compute_trend_regime,
    compute_tf_regime,
    compute_regime_agreement,
    compute_regime_composite,
    compute_transition_probability,
    compute_binseg_regime,
    compute_msm_regime,
    compute_hybrid_regime,
    compute_regime_features,
)

from cryptobot.features.mt_engine import (
    MultiTimeframeFeatureEngine,
    MTFeatureConfig,
    compute_mt_features,
    compute_cross_tf_features,
    get_feature_engine,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # === Original Engine ===
    "FeatureEngine",
    "compute_features",
    "compute_strategy_features",
    
    # === Multi-Timeframe Engine ===
    "MultiTimeframeFeatureEngine",
    "MTFeatureConfig",
    "compute_mt_features",
    "get_feature_engine",
    
    # === Per-TF Feature Functions ===
    "compute_tf_features",
    "compute_all_tf_features",
    "compute_cross_tf_features",
    
    # === Regime Functions ===
    "compute_regime_features",
    "compute_vol_regime",
    "compute_trend_regime",
    "compute_regime_agreement",
    "compute_transition_probability",
    "compute_binseg_regime",
    "compute_msm_regime",
    "compute_hybrid_regime",
    
    # === Base Classes ===
    "Feature",
    "FeatureGroup",
    "register_feature",
    "register_group",
    "get_feature",
    "list_features",
    "get_feature_info",
    "FEATURE_GROUPS",
    
    # === Utilities ===
    "compute_returns",
    "compute_log_returns",
    "compute_sma",
    "compute_ema",
    "compute_rolling_std",
    "compute_rolling_zscore",
    "compute_atr",
    
    # === Constants ===
    "TIMEFRAMES",
    "MA_PERIODS",
]