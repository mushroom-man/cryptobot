# -*- coding: utf-8 -*-
"""
CryptoBot - Technical Features
===============================
Moving averages, price ratios, and standard technical indicators.

Features:
    - sma_6, sma_24, sma_72: Simple moving averages
    - ma_score: Trend strength (0-3)
    - price_vs_sma_*: Price relative to moving averages
    - atr_14: Average True Range
    - rsi_14: Relative Strength Index
"""

import pandas as pd
import numpy as np
from cryptobot.features.base import (
    Feature, register_feature, register_group, FeatureGroup,
    compute_sma, compute_atr, compute_log_returns
)


# =============================================================================
# Moving Averages
# =============================================================================

@register_feature
class SMA6(Feature):
    """6-hour Simple Moving Average (intraday trend)."""
    name = "sma_6"
    lookback = 6
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return compute_sma(df['close'], window=6)


@register_feature
class SMA24(Feature):
    """24-hour Simple Moving Average (daily trend)."""
    name = "sma_24"
    lookback = 24
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return compute_sma(df['close'], window=24)


@register_feature
class SMA72(Feature):
    """72-hour Simple Moving Average (3-day trend)."""
    name = "sma_72"
    lookback = 72
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return compute_sma(df['close'], window=72)


@register_feature
class SMA168(Feature):
    """168-hour Simple Moving Average (weekly trend)."""
    name = "sma_168"
    lookback = 168
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return compute_sma(df['close'], window=168)


# =============================================================================
# MA Score (Trend Strength)
# =============================================================================

@register_feature
class MAScore(Feature):
    """
    Moving Average Score: Count of MAs price is above.
    
    Score = (close > SMA_6) + (close > SMA_24) + (close > SMA_72)
    
    Values: 0 (bearish), 1, 2, 3 (bullish)
    """
    name = "ma_score"
    lookback = 72
    output_type = "discrete"
    dependencies = ["sma_6", "sma_24", "sma_72"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        sma_6 = compute_sma(close, 6)
        sma_24 = compute_sma(close, 24)
        sma_72 = compute_sma(close, 72)
        
        score = (
            (close > sma_6).astype(int) + 
            (close > sma_24).astype(int) + 
            (close > sma_72).astype(int)
        )
        return score


# =============================================================================
# Price vs MA Ratios
# =============================================================================

@register_feature
class PriceVsSMA6(Feature):
    """Price relative to 6-hour SMA (close / sma_6)."""
    name = "price_vs_sma_6"
    lookback = 6
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        sma = compute_sma(df['close'], 6)
        return df['close'] / sma


@register_feature
class PriceVsSMA24(Feature):
    """Price relative to 24-hour SMA (close / sma_24)."""
    name = "price_vs_sma_24"
    lookback = 24
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        sma = compute_sma(df['close'], 24)
        return df['close'] / sma


@register_feature
class PriceVsSMA72(Feature):
    """Price relative to 72-hour SMA (close / sma_72)."""
    name = "price_vs_sma_72"
    lookback = 72
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        sma = compute_sma(df['close'], 72)
        return df['close'] / sma


@register_feature
class PriceVsSMA168(Feature):
    """Price relative to 168-hour SMA (close / sma_168)."""
    name = "price_vs_sma_168"
    lookback = 168
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        sma = compute_sma(df['close'], 168)
        return df['close'] / sma


# =============================================================================
# Other Technical Indicators
# =============================================================================

@register_feature
class ATR14(Feature):
    """14-period Average True Range."""
    name = "atr_14"
    lookback = 15
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return compute_atr(df, window=14)


@register_feature
class ATRPercent(Feature):
    """ATR as percentage of price (ATR / close * 100)."""
    name = "atr_percent"
    lookback = 15
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        atr = compute_atr(df, window=14)
        return (atr / df['close']) * 100


@register_feature
class RSI14(Feature):
    """14-period Relative Strength Index."""
    name = "rsi_14"
    lookback = 15
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


@register_feature
class Returns1H(Feature):
    """1-hour log returns."""
    name = "returns_1h"
    lookback = 2
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return compute_log_returns(df, 'close', periods=1)


@register_feature
class Returns24H(Feature):
    """24-hour log returns."""
    name = "returns_24h"
    lookback = 25
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return compute_log_returns(df, 'close', periods=24)


@register_feature
class Returns72H(Feature):
    """72-hour (3-day) log returns."""
    name = "returns_72h"
    lookback = 73
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return compute_log_returns(df, 'close', periods=72)


@register_feature
class VolumeRatio(Feature):
    """Current volume vs 24-hour average volume."""
    name = "volume_ratio"
    lookback = 24
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        avg_vol = df['volume'].rolling(window=24).mean()
        return df['volume'] / avg_vol


@register_feature  
class HighLowRange(Feature):
    """High-Low range as percentage of close."""
    name = "hl_range_pct"
    lookback = 1
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return (df['high'] - df['low']) / df['close'] * 100


# =============================================================================
# Feature Groups
# =============================================================================

register_group(FeatureGroup(
    name="trend",
    features=['ma_score', 'price_vs_sma_6', 'price_vs_sma_24', 'price_vs_sma_72'],
    description="Trend-following features based on moving averages"
))

register_group(FeatureGroup(
    name="technical",
    features=['rsi_14', 'atr_14', 'atr_percent', 'volume_ratio', 'hl_range_pct'],
    description="Standard technical indicators"
))

register_group(FeatureGroup(
    name="returns",
    features=['returns_1h', 'returns_24h', 'returns_72h'],
    description="Return features at different horizons"
))
