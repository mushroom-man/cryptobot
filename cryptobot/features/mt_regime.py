# -*- coding: utf-8 -*-
"""
CryptoBot - Multi-Timeframe Regime Detection
==============================================
Regime detection across multiple timeframes with transition probability.

Key Insight:
    - 1h regime: Fast/noisy, catches early transitions
    - 24h regime: Medium-term, more stable signal
    - 168h regime: Structural, slow-moving
    
    Agreement across timeframes = High confidence
    Disagreement = Transition zone (regime change brewing)

Features:
    Per-timeframe:
        regime_vol_{tf}      - Volatility regime (0=low, 1=mid, 2=high)
        regime_trend_{tf}    - Trend regime (0=down, 1=neutral, 2=up)
    
    Cross-timeframe:
        regime_agree_vol     - # of TFs agreeing on vol regime
        regime_agree_trend   - # of TFs agreeing on trend regime
        regime_conflict      - Disagreement score (higher = transition likely)
        regime_composite     - Weighted multi-TF regime score
    
    Transition probability:
        regime_chg_prob      - Composite transition probability
        regime_duration      - Bars since last regime change
        regime_stability     - Regime stability score

Usage:
    from cryptobot.features.mt_regime import compute_regime_features
    
    features = compute_regime_features(df_aligned, timeframes=['1h', '24h', '168h'])
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Constants
# =============================================================================

TIMEFRAMES = ['1h', '4h', '12h', '24h', '72h', '168h']

# Hours per timeframe bar
TF_HOURS = {
    '1h': 1,
    '4h': 4,
    '12h': 12,
    '24h': 24,
    '72h': 72,
    '168h': 168,
}

# Window scaling: target ~24 hours of lookback
WINDOW_SCALE = {
    '1h': 24,    # 24 bars = 24 hours
    '4h': 6,     # 6 bars = 24 hours
    '12h': 2,    # 2 bars = 24 hours
    '24h': 1,    # 1 bar = 24 hours
    '72h': 1,    # 1 bar = 72 hours (min)
    '168h': 1,   # 1 bar = 168 hours (min)
}

# Extended window scaling: target ~168 hours (for regime lookback)
WINDOW_SCALE_EXTENDED = {
    '1h': 168,   # 168 bars = 168 hours
    '4h': 42,    # 42 bars = 168 hours
    '12h': 14,   # 14 bars = 168 hours
    '24h': 7,    # 7 bars = 168 hours
    '72h': 3,    # 3 bars = 216 hours
    '168h': 2,   # 2 bars = 336 hours (min)
}

# Timeframe weights for composite scores (longer TF = more weight for structural)
TF_WEIGHTS = {
    '1h': 0.10,
    '4h': 0.12,
    '12h': 0.15,
    '24h': 0.18,
    '72h': 0.20,
    '168h': 0.25,
}

# Vol regime thresholds (percentiles)
VOL_REGIME_LOW = 0.33
VOL_REGIME_HIGH = 0.67

# Trend regime thresholds
TREND_NEUTRAL_BAND = 0.02  # +/- 2% from MA considered neutral


def get_scaled_window(tf: str, base_window: int = 14, extended: bool = False) -> int:
    """Get scaled window for a timeframe."""
    if extended:
        scale = WINDOW_SCALE_EXTENDED.get(tf, 1)
    else:
        scale = WINDOW_SCALE.get(tf, 1)
    
    tf_hours = TF_HOURS.get(tf, 1)
    scaled = max(1, round(base_window * scale / 24))
    
    return max(2, scaled)  # Minimum 2 for most calculations


# =============================================================================
# Helper Functions
# =============================================================================

def _get_col(df: pd.DataFrame, col: str, tf: str) -> pd.Series:
    """Get column for timeframe."""
    col_name = f"{col}_{tf}"
    if col_name in df.columns:
        return df[col_name]
    if col in df.columns:
        return df[col]
    raise KeyError(f"Column {col_name} or {col} not found")


def _safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Safe division."""
    result = a / b
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(fill)


# =============================================================================
# Per-Timeframe Regime Detection
# =============================================================================

def compute_vol_regime(
    df: pd.DataFrame,
    tf: str,
    lookback: int = None,
    window: int = None,
) -> pd.DataFrame:
    """
    Compute volatility regime for a timeframe.
    
    Regime values:
        0 = Low volatility (bottom tercile)
        1 = Normal volatility (middle tercile)
        2 = High volatility (top tercile)
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string
        lookback: Lookback for percentile calculation (auto-scaled if None)
        window: Window for volatility calculation (auto-scaled if None)
    
    Returns:
        DataFrame with regime_vol_{tf}
    """
    # Auto-scale windows
    if window is None:
        window = get_scaled_window(tf, base_window=14)
    if lookback is None:
        lookback = get_scaled_window(tf, base_window=14, extended=True)
    
    window = max(2, window)
    lookback = max(3, lookback)
    close = _get_col(df, 'close', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # Compute volatility
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(window=window, min_periods=window).std()
    
    # Rolling percentile rank
    vol_pct = vol.rolling(window=lookback, min_periods=lookback//2).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5,
        raw=True
    )
    
    # Assign regime
    regime = pd.Series(1, index=df.index)  # Default: normal
    regime[vol_pct < VOL_REGIME_LOW] = 0    # Low vol
    regime[vol_pct > VOL_REGIME_HIGH] = 2   # High vol
    
    result[f'regime_vol_{tf}'] = regime
    result[f'regime_vol_pct_{tf}'] = vol_pct  # Raw percentile for transition calc
    
    return result


def compute_trend_regime(
    df: pd.DataFrame,
    tf: str,
    ma_period: int = None,
) -> pd.DataFrame:
    """
    Compute trend regime for a timeframe.
    
    Regime values:
        0 = Downtrend (price below MA by > threshold)
        1 = Neutral (price near MA)
        2 = Uptrend (price above MA by > threshold)
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string
        ma_period: Moving average period (auto-scaled if None)
    
    Returns:
        DataFrame with regime_trend_{tf}
    """
    # Auto-scale MA period
    if ma_period is None:
        ma_period = get_scaled_window(tf, base_window=24)
    ma_period = max(1, ma_period)
    
    close = _get_col(df, 'close', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # Compute MA and price ratio
    ma = close.rolling(window=ma_period, min_periods=ma_period).mean()
    price_vs_ma = _safe_divide(close, ma, fill=1.0) - 1.0  # Centered at 0
    
    # Assign regime
    regime = pd.Series(1, index=df.index)  # Default: neutral
    regime[price_vs_ma < -TREND_NEUTRAL_BAND] = 0  # Downtrend
    regime[price_vs_ma > TREND_NEUTRAL_BAND] = 2   # Uptrend
    
    result[f'regime_trend_{tf}'] = regime
    result[f'regime_trend_strength_{tf}'] = price_vs_ma  # Raw strength for analysis
    
    return result


def compute_tf_regime(
    df: pd.DataFrame,
    tf: str,
    vol_lookback: int = 168,
    ma_period: int = 24,
) -> pd.DataFrame:
    """
    Compute both vol and trend regime for a timeframe.
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string
    
    Returns:
        DataFrame with regime features for this TF
    """
    vol_regime = compute_vol_regime(df, tf, vol_lookback)
    trend_regime = compute_trend_regime(df, tf, ma_period)
    
    return pd.concat([vol_regime, trend_regime], axis=1)


# =============================================================================
# Cross-Timeframe Regime Analysis
# =============================================================================

def compute_regime_agreement(
    df: pd.DataFrame,
    timeframes: List[str],
) -> pd.DataFrame:
    """
    Compute cross-timeframe regime agreement features.
    
    Features:
        regime_agree_vol     - Count of TFs agreeing on vol regime
        regime_agree_trend   - Count of TFs agreeing on trend regime
        regime_conflict_vol  - Max disagreement in vol regime across TFs
        regime_conflict_trend - Max disagreement in trend regime across TFs
        regime_conflict      - Combined conflict score
    
    Args:
        df: DataFrame with regime_vol_{tf} and regime_trend_{tf} columns
        timeframes: List of timeframes to analyze
    
    Returns:
        DataFrame with agreement features
    """
    result = pd.DataFrame(index=df.index)
    
    # Collect regime values
    vol_regimes = []
    trend_regimes = []
    
    for tf in timeframes:
        vol_col = f'regime_vol_{tf}'
        trend_col = f'regime_trend_{tf}'
        
        if vol_col in df.columns:
            vol_regimes.append(df[vol_col])
        if trend_col in df.columns:
            trend_regimes.append(df[trend_col])
    
    if not vol_regimes and not trend_regimes:
        return result
    
    # Stack into DataFrames
    if vol_regimes:
        vol_df = pd.concat(vol_regimes, axis=1)
        
        # Mode (most common regime)
        vol_mode = vol_df.mode(axis=1)[0]
        
        # Agreement: count matching mode
        vol_agree = (vol_df.eq(vol_mode, axis=0)).sum(axis=1)
        result['regime_agree_vol'] = vol_agree
        
        # Conflict: max - min regime value
        vol_conflict = vol_df.max(axis=1) - vol_df.min(axis=1)
        result['regime_conflict_vol'] = vol_conflict
    
    if trend_regimes:
        trend_df = pd.concat(trend_regimes, axis=1)
        
        # Mode
        trend_mode = trend_df.mode(axis=1)[0]
        
        # Agreement
        trend_agree = (trend_df.eq(trend_mode, axis=0)).sum(axis=1)
        result['regime_agree_trend'] = trend_agree
        
        # Conflict
        trend_conflict = trend_df.max(axis=1) - trend_df.min(axis=1)
        result['regime_conflict_trend'] = trend_conflict
    
    # Combined conflict score
    if 'regime_conflict_vol' in result.columns and 'regime_conflict_trend' in result.columns:
        result['regime_conflict'] = (
            result['regime_conflict_vol'] + result['regime_conflict_trend']
        ) / 2
    
    return result


def compute_regime_composite(
    df: pd.DataFrame,
    timeframes: List[str],
) -> pd.DataFrame:
    """
    Compute weighted composite regime scores.
    
    Features:
        regime_composite_vol   - Weighted vol regime (0-2 scale)
        regime_composite_trend - Weighted trend regime (0-2 scale)
        regime_composite       - Combined regime score
    
    Args:
        df: DataFrame with regime features
        timeframes: Timeframes to include
    
    Returns:
        DataFrame with composite scores
    """
    result = pd.DataFrame(index=df.index)
    
    # Normalize weights for available timeframes
    available_weights = {tf: TF_WEIGHTS[tf] for tf in timeframes if tf in TF_WEIGHTS}
    weight_sum = sum(available_weights.values())
    norm_weights = {tf: w/weight_sum for tf, w in available_weights.items()}
    
    # Weighted vol regime
    vol_composite = pd.Series(0.0, index=df.index)
    for tf, weight in norm_weights.items():
        col = f'regime_vol_{tf}'
        if col in df.columns:
            vol_composite += df[col] * weight
    result['regime_composite_vol'] = vol_composite
    
    # Weighted trend regime
    trend_composite = pd.Series(0.0, index=df.index)
    for tf, weight in norm_weights.items():
        col = f'regime_trend_{tf}'
        if col in df.columns:
            trend_composite += df[col] * weight
    result['regime_composite_trend'] = trend_composite
    
    # Combined (vol slightly more weighted for risk)
    result['regime_composite'] = 0.6 * vol_composite + 0.4 * trend_composite
    
    return result


# =============================================================================
# Regime Transition Probability
# =============================================================================

def compute_transition_probability(
    df: pd.DataFrame,
    timeframes: List[str],
    threshold_distance: float = 0.1,
) -> pd.DataFrame:
    """
    Compute regime transition probability features.
    
    Logic:
        - Distance to regime threshold: closer = higher prob
        - Cross-TF disagreement: higher = transition brewing
        - Vol of vol: higher = unstable, transition likely
        - Duration in current regime: longer = mean reversion pressure
    
    Features:
        regime_chg_prob_vol     - Vol regime change probability
        regime_chg_prob_trend   - Trend regime change probability
        regime_chg_prob         - Combined transition probability
        regime_duration_vol     - Bars since last vol regime change
        regime_duration_trend   - Bars since last trend regime change
        regime_stability        - Regime stability score (inverse of chg_prob)
    
    Args:
        df: DataFrame with regime features
        timeframes: Timeframes used
        threshold_distance: Distance from threshold considered "close"
    
    Returns:
        DataFrame with transition features
    """
    result = pd.DataFrame(index=df.index)
    
    # Find the base/reference timeframe (prefer 24h or fallback)
    ref_tf = '24h' if '24h' in timeframes else timeframes[0]
    
    # === Vol transition probability ===
    vol_pct_col = f'regime_vol_pct_{ref_tf}'
    if vol_pct_col in df.columns:
        vol_pct = df[vol_pct_col]
        
        # Distance to nearest threshold
        dist_to_low = (vol_pct - VOL_REGIME_LOW).abs()
        dist_to_high = (vol_pct - VOL_REGIME_HIGH).abs()
        dist_to_threshold = pd.concat([dist_to_low, dist_to_high], axis=1).min(axis=1)
        
        # Closer to threshold = higher probability (normalized 0-1)
        vol_chg_prob = 1 - (dist_to_threshold / 0.5).clip(0, 1)
        result['regime_chg_prob_vol'] = vol_chg_prob
    
    # === Trend transition probability ===
    trend_str_col = f'regime_trend_strength_{ref_tf}'
    if trend_str_col in df.columns:
        trend_str = df[trend_str_col]
        
        # Distance to neutral band
        dist_to_neutral = trend_str.abs() - TREND_NEUTRAL_BAND
        dist_to_neutral = dist_to_neutral.clip(lower=0)
        
        # Closer to neutral = higher prob of direction change
        trend_chg_prob = 1 - (dist_to_neutral / 0.1).clip(0, 1)
        result['regime_chg_prob_trend'] = trend_chg_prob
    
    # === Duration since last change ===
    vol_col = f'regime_vol_{ref_tf}'
    if vol_col in df.columns:
        regime_changed = df[vol_col].diff().abs() > 0
        # Cumulative count since last change
        duration = (~regime_changed).cumsum()
        duration = duration - duration.where(regime_changed).ffill().fillna(0)
        result['regime_duration_vol'] = duration
    
    trend_col = f'regime_trend_{ref_tf}'
    if trend_col in df.columns:
        regime_changed = df[trend_col].diff().abs() > 0
        duration = (~regime_changed).cumsum()
        duration = duration - duration.where(regime_changed).ffill().fillna(0)
        result['regime_duration_trend'] = duration
    
    # === Cross-TF disagreement contribution ===
    if 'regime_conflict' in df.columns:
        conflict_prob = df['regime_conflict'] / 2.0  # Normalize (max conflict = 2)
        result['regime_chg_prob_conflict'] = conflict_prob.clip(0, 1)
    
    # === Combined probability ===
    prob_components = []
    if 'regime_chg_prob_vol' in result.columns:
        prob_components.append(result['regime_chg_prob_vol'] * 0.35)
    if 'regime_chg_prob_trend' in result.columns:
        prob_components.append(result['regime_chg_prob_trend'] * 0.25)
    if 'regime_chg_prob_conflict' in result.columns:
        prob_components.append(result['regime_chg_prob_conflict'] * 0.40)
    
    if prob_components:
        result['regime_chg_prob'] = sum(prob_components).clip(0, 1)
        result['regime_stability'] = 1 - result['regime_chg_prob']
    
    return result


# =============================================================================
# Legacy Regime Features (BinSeg, MSM, Hybrid)
# =============================================================================

def compute_binseg_regime(
    df: pd.DataFrame,
    tf: str = '1h',
    vol_window: int = None,
    min_segment: int = None,
) -> pd.DataFrame:
    """
    Compute Binary Segmentation regime (structural changes).
    
    Uses threshold-based approach (faster than ruptures library).
    
    Features:
        regime_binseg      - Binary (0=calm, 1=volatile)
        regime_binseg_raw  - Raw volatility percentile
    
    Args:
        df: DataFrame with OHLCV data
        tf: Timeframe to use
        vol_window: Window for volatility (auto-scaled if None)
        min_segment: Minimum bars between regime changes (auto-scaled if None)
    
    Returns:
        DataFrame with BinSeg regime
    """
    # Auto-scale windows
    if vol_window is None:
        vol_window = get_scaled_window(tf, base_window=14, extended=True)
    if min_segment is None:
        min_segment = get_scaled_window(tf, base_window=48)
    
    vol_window = max(2, vol_window)
    min_segment = max(1, min_segment)
    
    close = _get_col(df, 'close', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # Compute rolling volatility
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(window=vol_window, min_periods=vol_window).std()
    
    # Rolling median as threshold
    vol_median = vol.rolling(window=vol_window*3, min_periods=vol_window).median()
    
    # Binary regime
    regime = (vol > vol_median).astype(int)
    
    # Smooth to avoid flickering (enforce minimum segment)
    regime_smooth = regime.copy()
    last_change = 0
    last_regime = regime.iloc[vol_window] if len(regime) > vol_window else 0
    
    for i in range(vol_window, len(regime)):
        if regime.iloc[i] != last_regime:
            if i - last_change >= min_segment:
                last_regime = regime.iloc[i]
                last_change = i
            else:
                regime_smooth.iloc[i] = last_regime
    
    result['regime_binseg'] = regime_smooth
    result['regime_binseg_raw'] = vol / vol_median
    
    return result


def compute_msm_regime(
    df: pd.DataFrame,
    tf: str = '1h',
    fast_window: int = None,
    slow_window: int = None,
) -> pd.DataFrame:
    """
    Compute Markov Switching Model-like regime (momentum).
    
    Simplified version using volatility ratio as proxy.
    
    Features:
        regime_msm     - Binary (0=calm, 1=volatile)
        regime_msm_raw - Raw regime indicator
    
    Args:
        df: DataFrame with OHLCV data
        tf: Timeframe to use
        fast_window: Fast volatility window (auto-scaled if None)
        slow_window: Slow volatility window (auto-scaled if None)
    
    Returns:
        DataFrame with MSM regime
    """
    # Auto-scale windows
    if fast_window is None:
        fast_window = get_scaled_window(tf, base_window=24)
    if slow_window is None:
        slow_window = get_scaled_window(tf, base_window=24, extended=True)
    
    fast_window = max(2, fast_window)
    slow_window = max(fast_window + 1, slow_window)  # Ensure slow > fast
    
    close = _get_col(df, 'close', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # Compute fast and slow volatility
    ret = np.log(close / close.shift(1))
    vol_fast = ret.rolling(window=fast_window, min_periods=fast_window).std()
    vol_slow = ret.rolling(window=slow_window, min_periods=slow_window).std()
    
    # Ratio indicates regime
    vol_ratio = _safe_divide(vol_fast, vol_slow, fill=1.0)
    
    # Binary regime (fast vol > slow vol = volatile regime)
    regime = (vol_ratio > 1.0).astype(int)
    
    result['regime_msm'] = regime
    result['regime_msm_raw'] = vol_ratio
    
    return result


def compute_hybrid_regime(
    df: pd.DataFrame,
    tf: str = '1h',
) -> pd.DataFrame:
    """
    Compute hybrid 4-state regime (BinSeg x MSM).
    
    States:
        0 = Calm/Calm (quiet consolidation)
        1 = Calm/Volatile (breakout - historically best returns)
        2 = Volatile/Calm (recovery)
        3 = Volatile/Volatile (crisis)
    
    Features:
        regime_hybrid       - 4-state regime (0-3)
        regime_hybrid_name  - Human-readable state name
    
    Args:
        df: DataFrame with regime_binseg and regime_msm
        tf: Timeframe (for column lookup)
    
    Returns:
        DataFrame with hybrid regime
    """
    result = pd.DataFrame(index=df.index)
    
    # Check if we have the components
    if 'regime_binseg' in df.columns and 'regime_msm' in df.columns:
        binseg = df['regime_binseg']
        msm = df['regime_msm']
    else:
        # Compute them
        binseg_df = compute_binseg_regime(df, tf)
        msm_df = compute_msm_regime(df, tf)
        binseg = binseg_df['regime_binseg']
        msm = msm_df['regime_msm']
    
    # Combine into 4-state
    hybrid = binseg * 2 + msm
    result['regime_hybrid'] = hybrid
    
    # Multiplier from strategy document
    # State 1 (calm/volatile) = best, State 3 (volatile/volatile) = worst
    multiplier_map = {0: 1.0, 1: 1.2, 2: 0.8, 3: 0.5}
    result['regime_multiplier'] = hybrid.map(multiplier_map)
    
    return result


# =============================================================================
# Main Function
# =============================================================================

def compute_regime_features(
    df: pd.DataFrame,
    timeframes: List[str] = None,
    include_legacy: bool = True,
) -> pd.DataFrame:
    """
    Compute all regime features.
    
    Args:
        df: DataFrame with aligned OHLCV data
        timeframes: Timeframes to analyze (default: auto-detect)
        include_legacy: Include BinSeg/MSM/Hybrid features
    
    Returns:
        DataFrame with all regime features
    """
    if timeframes is None:
        # Auto-detect from columns
        timeframes = []
        for tf in TIMEFRAMES:
            if f'close_{tf}' in df.columns:
                timeframes.append(tf)
        
        if not timeframes:
            # Fall back to single-TF data
            timeframes = ['1h']
    
    print(f"  Computing regime features for {timeframes}...")
    
    result = pd.DataFrame(index=df.index)
    
    # Per-timeframe regimes
    for tf in timeframes:
        try:
            tf_regime = compute_tf_regime(df, tf)
            result = pd.concat([result, tf_regime], axis=1)
        except KeyError:
            pass
    
    # Cross-timeframe analysis
    agreement = compute_regime_agreement(result, timeframes)
    result = pd.concat([result, agreement], axis=1)
    
    composite = compute_regime_composite(result, timeframes)
    result = pd.concat([result, composite], axis=1)
    
    transition = compute_transition_probability(result, timeframes)
    result = pd.concat([result, transition], axis=1)
    
    # Legacy features (BinSeg, MSM, Hybrid)
    if include_legacy:
        ref_tf = '1h' if '1h' in timeframes else timeframes[0]
        
        try:
            binseg = compute_binseg_regime(df, ref_tf)
            result = pd.concat([result, binseg], axis=1)
            
            msm = compute_msm_regime(df, ref_tf)
            result = pd.concat([result, msm], axis=1)
            
            hybrid = compute_hybrid_regime(result, ref_tf)
            result = pd.concat([result, hybrid], axis=1)
        except Exception as e:
            print(f"  Warning: Legacy regime failed: {e}")
    
    print(f"  Computed {len(result.columns)} regime features")
    
    return result