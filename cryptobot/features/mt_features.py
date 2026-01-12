# -*- coding: utf-8 -*-
"""
CryptoBot - Multi-Timeframe Features
=====================================
Per-timeframe feature computations for the multi-timeframe backtesting platform.

Naming Convention:
    {category}_{metric}_{timeframe}
    
    Categories: ret, vol, rng, ma, mom, vlm
    Examples: ret_log_24h, vol_ann_168h, ma_ratio_72h

Window Scaling:
    Windows are scaled by timeframe to cover similar calendar time.
    Target: ~24-72 hours of lookback for most features.
    
    1h:   window=24 → 24 hours
    24h:  window=1  → 24 hours
    168h: window=1  → 168 hours (minimum 1 bar)

Usage:
    from cryptobot.features.mt_features import compute_tf_features
    
    # Compute all features for one timeframe
    features = compute_tf_features(df_aligned, timeframe='24h')
    
    # Compute specific category
    features = compute_returns(df_aligned, timeframe='24h')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Constants
# =============================================================================

TIMEFRAMES = ['1h', '4h', '12h', '24h', '72h', '168h']

# Hours per year for annualization
HOURS_PER_YEAR = 8760

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
# Formula: max(1, round(24 / tf_hours))
WINDOW_SCALE = {
    '1h': 24,    # 24 bars = 24 hours
    '4h': 6,     # 6 bars = 24 hours
    '12h': 2,    # 2 bars = 24 hours
    '24h': 1,    # 1 bar = 24 hours
    '72h': 1,    # 1 bar = 72 hours (min)
    '168h': 1,   # 1 bar = 168 hours (min)
}

# Extended window scaling: target ~72 hours (for volatility, etc.)
WINDOW_SCALE_EXTENDED = {
    '1h': 72,    # 72 bars = 72 hours
    '4h': 18,    # 18 bars = 72 hours
    '12h': 6,    # 6 bars = 72 hours
    '24h': 3,    # 3 bars = 72 hours
    '72h': 1,    # 1 bar = 72 hours
    '168h': 1,   # 1 bar = 168 hours (min)
}

# Annualization factors per timeframe
ANNUALIZATION_FACTOR = {
    '1h': np.sqrt(HOURS_PER_YEAR),
    '4h': np.sqrt(HOURS_PER_YEAR / 4),
    '12h': np.sqrt(HOURS_PER_YEAR / 12),
    '24h': np.sqrt(365),
    '72h': np.sqrt(365 / 3),
    '168h': np.sqrt(52),
}

# Standard MA periods (will be scaled per TF)
MA_PERIODS = [6, 12, 24]


def get_scaled_window(tf: str, base_window: int = 14, extended: bool = False) -> int:
    """
    Get scaled window for a timeframe.
    
    Args:
        tf: Timeframe string
        base_window: Base window for 1h (default 14)
        extended: Use extended (72h target) scaling
    
    Returns:
        Scaled window size (minimum 1)
    """
    if extended:
        scale = WINDOW_SCALE_EXTENDED.get(tf, 1)
    else:
        scale = WINDOW_SCALE.get(tf, 1)
    
    # Scale proportionally but ensure minimum of 1
    tf_hours = TF_HOURS.get(tf, 1)
    scaled = max(1, round(base_window * scale / 24))
    
    return scaled


def get_scaled_ma_periods(tf: str) -> List[int]:
    """
    Get scaled MA periods for a timeframe.
    
    For 1h: [6, 12, 24]
    For 24h: [1, 2, 3] (covering ~24h, 48h, 72h)
    For 168h: [1, 2, 3] (minimum)
    """
    tf_hours = TF_HOURS.get(tf, 1)
    
    if tf_hours <= 4:
        return [6, 12, 24]
    elif tf_hours <= 12:
        return [3, 6, 12]
    elif tf_hours <= 24:
        return [2, 4, 6]
    else:
        return [1, 2, 3]


# =============================================================================
# Helper Functions
# =============================================================================

def _get_col(df: pd.DataFrame, col: str, tf: str) -> pd.Series:
    """Get column for timeframe, handling naming conventions."""
    # Try suffixed name first (aligned data)
    col_name = f"{col}_{tf}"
    if col_name in df.columns:
        return df[col_name]
    # Try plain name (single-tf data)
    if col in df.columns:
        return df[col]
    raise KeyError(f"Column {col_name} or {col} not found in DataFrame")


def _safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Safe division handling zeros and infinities."""
    result = a / b
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(fill)


# =============================================================================
# Returns Features (ret_)
# =============================================================================

def compute_returns(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Compute return features for a timeframe.
    
    Features:
        ret_log_{tf}      - Log return
        ret_abs_{tf}      - Absolute log return
        ret_sign_{tf}     - Direction (+1/0/-1)
        ret_cum_2_{tf}    - 2-period cumulative return
        ret_cum_3_{tf}    - 3-period cumulative return
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string ('1h', '24h', etc.)
    
    Returns:
        DataFrame with return features
    """
    close = _get_col(df, 'close', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # Log return
    ret_log = np.log(close / close.shift(1))
    result[f'ret_log_{tf}'] = ret_log
    
    # Absolute return
    result[f'ret_abs_{tf}'] = ret_log.abs()
    
    # Sign
    result[f'ret_sign_{tf}'] = np.sign(ret_log)
    
    # Cumulative returns
    result[f'ret_cum_2_{tf}'] = np.log(close / close.shift(2))
    result[f'ret_cum_3_{tf}'] = np.log(close / close.shift(3))
    
    return result


# =============================================================================
# Volatility Features (vol_)
# =============================================================================

def compute_volatility(df: pd.DataFrame, tf: str, window: int = None) -> pd.DataFrame:
    """
    Compute volatility features for a timeframe.
    
    Features:
        vol_std_{tf}        - Rolling std of returns (raw)
        vol_ann_{tf}        - Annualized volatility
        vol_parkinson_{tf}  - Parkinson (high-low) estimator
        vol_garman_{tf}     - Garman-Klass (OHLC) estimator
        vol_zscore_{tf}     - Vol z-score vs rolling mean
        vol_regime_{tf}     - Vol tercile (0=low, 1=mid, 2=high)
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string
        window: Rolling window (auto-scaled if None)
    
    Returns:
        DataFrame with volatility features
    """
    # Auto-scale window based on timeframe
    if window is None:
        window = get_scaled_window(tf, base_window=14, extended=True)
    
    # Ensure minimum window of 2 for std calculation
    window = max(2, window)
    
    close = _get_col(df, 'close', tf)
    high = _get_col(df, 'high', tf)
    low = _get_col(df, 'low', tf)
    open_ = _get_col(df, 'open', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # Log returns
    ret_log = np.log(close / close.shift(1))
    
    # Rolling std (raw)
    vol_std = ret_log.rolling(window=window, min_periods=window).std()
    result[f'vol_std_{tf}'] = vol_std
    
    # Annualized volatility
    ann_factor = ANNUALIZATION_FACTOR.get(tf, np.sqrt(HOURS_PER_YEAR))
    result[f'vol_ann_{tf}'] = vol_std * ann_factor
    
    # Parkinson volatility (high-low based)
    hl_ratio = np.log(high / low)
    parkinson = np.sqrt(
        hl_ratio.pow(2).rolling(window=window, min_periods=window).mean() / (4 * np.log(2))
    )
    result[f'vol_parkinson_{tf}'] = parkinson * ann_factor
    
    # Garman-Klass volatility (OHLC based)
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = (
        0.5 * log_hl.pow(2) - 
        (2 * np.log(2) - 1) * log_co.pow(2)
    )
    garman = np.sqrt(gk_var.rolling(window=window, min_periods=window).mean())
    result[f'vol_garman_{tf}'] = garman * ann_factor
    
    # Volatility z-score (scaled lookback for comparison)
    zscore_window = max(3, window * 2)
    vol_mean = vol_std.rolling(window=zscore_window, min_periods=window).mean()
    vol_std_of_vol = vol_std.rolling(window=zscore_window, min_periods=window).std()
    result[f'vol_zscore_{tf}'] = _safe_divide(vol_std - vol_mean, vol_std_of_vol)
    
    # Volatility regime (terciles - simpler than quintiles for sparse data)
    # Use rolling percentile rank instead of qcut for stability
    regime_window = max(5, window * 3)
    vol_pct = vol_std.rolling(window=regime_window, min_periods=max(3, window)).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5,
        raw=True
    )
    # Convert percentile to tercile (0, 1, 2)
    vol_regime = pd.Series(1, index=df.index)  # Default: mid
    vol_regime[vol_pct < 0.33] = 0  # Low vol
    vol_regime[vol_pct > 0.67] = 2  # High vol
    result[f'vol_regime_{tf}'] = vol_regime
    
    return result


# =============================================================================
# Range Features (rng_)
# =============================================================================

def compute_range(df: pd.DataFrame, tf: str, window: int = None) -> pd.DataFrame:
    """
    Compute range/candle features for a timeframe.
    
    Features:
        rng_hl_{tf}       - High-Low range
        rng_hl_pct_{tf}   - Range as % of close
        rng_atr_{tf}      - Average True Range
        rng_atr_pct_{tf}  - ATR as % of close
        rng_body_{tf}     - Candle body (close-open)
        rng_body_pct_{tf} - Body as % of range
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string
        window: Rolling window for ATR (auto-scaled if None)
    
    Returns:
        DataFrame with range features
    """
    # Auto-scale window
    if window is None:
        window = get_scaled_window(tf, base_window=14)
    window = max(2, window)
    
    close = _get_col(df, 'close', tf)
    high = _get_col(df, 'high', tf)
    low = _get_col(df, 'low', tf)
    open_ = _get_col(df, 'open', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # High-Low range
    hl_range = high - low
    result[f'rng_hl_{tf}'] = hl_range
    result[f'rng_hl_pct_{tf}'] = _safe_divide(hl_range, close) * 100
    
    # ATR (Average True Range)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    result[f'rng_atr_{tf}'] = atr
    result[f'rng_atr_pct_{tf}'] = _safe_divide(atr, close) * 100
    
    # Candle body
    body = close - open_
    result[f'rng_body_{tf}'] = body
    result[f'rng_body_pct_{tf}'] = _safe_divide(body.abs(), hl_range.replace(0, np.nan)) * 100
    
    return result


# =============================================================================
# Moving Average Features (ma_)
# =============================================================================

def compute_moving_averages(df: pd.DataFrame, tf: str, periods: List[int] = None) -> pd.DataFrame:
    """
    Compute moving average features for a timeframe.
    
    Features (for each period p):
        ma_{p}_{tf}          - Simple moving average
        price_vs_ma_{p}_{tf} - Price / MA ratio
        ma_dist_{p}_{tf}     - Distance from MA (z-score)
    
    Additional:
        ma_ratio_fast_slow_{tf} - Fast/slow MA ratio
        ma_slope_{tf}           - MA slope (momentum)
        ma_cross_{tf}           - Cross signal (+1/-1/0)
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string
        periods: MA periods (auto-scaled if None)
    
    Returns:
        DataFrame with MA features
    """
    # Auto-scale MA periods based on timeframe
    if periods is None:
        periods = get_scaled_ma_periods(tf)
    
    close = _get_col(df, 'close', tf)
    
    result = pd.DataFrame(index=df.index)
    mas = {}
    
    # Compute MAs for each period
    for p in periods:
        p = max(1, p)  # Ensure minimum of 1
        ma = close.rolling(window=p, min_periods=p).mean()
        mas[p] = ma
        
        result[f'ma_{p}_{tf}'] = ma
        result[f'price_vs_ma_{p}_{tf}'] = _safe_divide(close, ma, fill=1.0)
        
        # Distance from MA (z-score)
        if p >= 2:
            ma_std = close.rolling(window=p, min_periods=p).std()
            result[f'ma_dist_{p}_{tf}'] = _safe_divide(close - ma, ma_std)
        else:
            # For p=1, use a simple percentage difference
            result[f'ma_dist_{p}_{tf}'] = (close - ma) / ma * 100
    
    # MA ratios (fast/slow) - use first and last periods
    if len(periods) >= 2:
        fast_p = min(periods)
        slow_p = max(periods)
        if fast_p in mas and slow_p in mas:
            result[f'ma_ratio_{fast_p}_{slow_p}_{tf}'] = _safe_divide(mas[fast_p], mas[slow_p], fill=1.0)
    
    # MA slope (using middle period if available)
    mid_p = periods[len(periods)//2] if periods else 2
    if mid_p in mas:
        ma_mid = mas[mid_p]
        slope_window = max(1, mid_p // 2)
        result[f'ma_slope_{tf}'] = ma_mid.diff(slope_window) / ma_mid.shift(slope_window) * 100
    
    # MA cross signal (fast vs slow)
    if len(periods) >= 2:
        fast_p = min(periods)
        slow_p = max(periods)
        if fast_p in mas and slow_p in mas:
            fast = mas[fast_p]
            slow = mas[slow_p]
            cross = pd.Series(0, index=df.index)
            cross[fast > slow] = 1
            cross[fast < slow] = -1
            result[f'ma_cross_{tf}'] = cross
    
    return result


# =============================================================================
# Momentum Features (mom_)
# =============================================================================

def compute_momentum(df: pd.DataFrame, tf: str, rsi_period: int = None) -> pd.DataFrame:
    """
    Compute momentum features for a timeframe.
    
    Features:
        mom_rsi_{tf}      - RSI
        mom_roc_{tf}      - Rate of change
        mom_accel_{tf}    - Return acceleration (2nd derivative)
        mom_streak_{tf}   - Consecutive up/down bars
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string
        rsi_period: Period for RSI calculation (auto-scaled if None)
    
    Returns:
        DataFrame with momentum features
    """
    # Auto-scale RSI period
    if rsi_period is None:
        rsi_period = get_scaled_window(tf, base_window=14)
    rsi_period = max(2, rsi_period)
    
    close = _get_col(df, 'close', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    
    rs = _safe_divide(avg_gain, avg_loss, fill=1.0)
    rsi = 100 - (100 / (1 + rs))
    result[f'mom_rsi_{tf}'] = rsi
    
    # Rate of change (scaled lookback)
    roc_period = max(1, rsi_period)
    roc = (close - close.shift(roc_period)) / close.shift(roc_period) * 100
    result[f'mom_roc_{tf}'] = roc
    
    # Acceleration (2nd derivative of price)
    ret = np.log(close / close.shift(1))
    accel = ret.diff()
    result[f'mom_accel_{tf}'] = accel
    
    # Streak (consecutive up/down bars)
    direction = np.sign(close.diff())
    streak = pd.Series(0, index=df.index, dtype=float)
    
    current_streak = 0
    prev_dir = 0
    
    for i in range(len(direction)):
        if pd.isna(direction.iloc[i]):
            streak.iloc[i] = 0
            continue
        
        curr_dir = direction.iloc[i]
        if curr_dir == prev_dir:
            current_streak += curr_dir
        else:
            current_streak = curr_dir
        
        streak.iloc[i] = current_streak
        prev_dir = curr_dir
    
    result[f'mom_streak_{tf}'] = streak
    
    return result


# =============================================================================
# Volume Features (vlm_)
# =============================================================================

def compute_volume(df: pd.DataFrame, tf: str, window: int = None) -> pd.DataFrame:
    """
    Compute volume features for a timeframe.
    
    Features:
        vlm_ma_{tf}         - Volume moving average
        vlm_ratio_{tf}      - Current / MA ratio
        vlm_trend_{tf}      - Volume trend (slope)
        vlm_price_corr_{tf} - Volume-price correlation (only for TFs <= 24h)
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string
        window: Rolling window (auto-scaled if None)
    
    Returns:
        DataFrame with volume features
    """
    # Auto-scale window
    if window is None:
        window = get_scaled_window(tf, base_window=14)
    window = max(2, window)
    
    close = _get_col(df, 'close', tf)
    volume = _get_col(df, 'volume', tf)
    
    result = pd.DataFrame(index=df.index)
    
    # Volume MA
    vlm_ma = volume.rolling(window=window, min_periods=window).mean()
    result[f'vlm_ma_{tf}'] = vlm_ma
    
    # Volume ratio
    result[f'vlm_ratio_{tf}'] = _safe_divide(volume, vlm_ma, fill=1.0)
    
    # Volume trend (slope of MA)
    slope_window = max(1, window // 2)
    result[f'vlm_trend_{tf}'] = vlm_ma.diff(slope_window) / vlm_ma.shift(slope_window) * 100
    
    # Volume-price correlation (only for lower TFs where we have enough bars)
    # Skip for 72h and 168h as they don't have enough bars for meaningful correlation
    tf_hours = TF_HOURS.get(tf, 1)
    if tf_hours <= 24:
        ret = np.log(close / close.shift(1))
        vlm_price_corr = ret.rolling(window=window, min_periods=window).corr(volume)
        result[f'vlm_price_corr_{tf}'] = vlm_price_corr
    
    return result


# =============================================================================
# Combined Per-Timeframe Features
# =============================================================================

def compute_tf_features(
    df: pd.DataFrame,
    tf: str,
    categories: List[str] = None,
) -> pd.DataFrame:
    """
    Compute all features for a single timeframe.
    
    Windows are auto-scaled based on timeframe to cover similar
    calendar time across all timeframes.
    
    Args:
        df: DataFrame with aligned OHLCV data
        tf: Timeframe string ('1h', '24h', etc.)
        categories: Feature categories to compute (default: all)
                   Options: 'ret', 'vol', 'rng', 'ma', 'mom', 'vlm'
    
    Returns:
        DataFrame with all features for this timeframe
    """
    if categories is None:
        categories = ['ret', 'vol', 'rng', 'ma', 'mom', 'vlm']
    
    result = pd.DataFrame(index=df.index)
    
    # All functions now auto-scale windows based on tf
    category_functions = {
        'ret': lambda: compute_returns(df, tf),
        'vol': lambda: compute_volatility(df, tf),  # Auto-scaled
        'rng': lambda: compute_range(df, tf),       # Auto-scaled
        'ma': lambda: compute_moving_averages(df, tf),  # Auto-scaled
        'mom': lambda: compute_momentum(df, tf),    # Auto-scaled
        'vlm': lambda: compute_volume(df, tf),      # Auto-scaled
    }
    
    for cat in categories:
        if cat in category_functions:
            try:
                cat_features = category_functions[cat]()
                result = pd.concat([result, cat_features], axis=1)
            except KeyError as e:
                # Missing columns for this timeframe - skip
                pass
            except Exception as e:
                print(f"Warning: Failed to compute {cat} for {tf}: {e}")
    
    return result


def compute_all_tf_features(
    df: pd.DataFrame,
    timeframes: List[str] = None,
    categories: List[str] = None,
) -> pd.DataFrame:
    """
    Compute features for all timeframes.
    
    Windows are auto-scaled per timeframe to ensure similar
    calendar coverage and reasonable warmup periods.
    
    Args:
        df: DataFrame with aligned OHLCV data (columns like close_1h, close_24h)
        timeframes: Timeframes to compute (default: all available)
        categories: Feature categories (default: all)
    
    Returns:
        DataFrame with all features for all timeframes
    """
    if timeframes is None:
        # Auto-detect from columns
        timeframes = []
        for tf in TIMEFRAMES:
            if f'close_{tf}' in df.columns:
                timeframes.append(tf)
    
    result = pd.DataFrame(index=df.index)
    
    for tf in timeframes:
        print(f"  Computing features for {tf}...", end=' ')
        tf_features = compute_tf_features(df, tf, categories)
        result = pd.concat([result, tf_features], axis=1)
        print(f"({len(tf_features.columns)} features)")
    
    return result