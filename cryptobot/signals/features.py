# -*- coding: utf-8 -*-
"""
CryptoBot — Historical Signal & Return Builder
===============================================
Batch generation of 24h-resolution signals and returns for hit rate
training.  Both trader.py and backtest import this to build the
DataFrames that HitRateCalculator consumes.

This is intentionally a SEPARATE hysteresis implementation from
regime.py's bar-by-bar classifier.  The numbers come from the same
single source of truth (RegimeConfig), but the batch loop replicates
the validated backtest methodology (generate_signals_24h, lines 416-459
of backtest_long_short_full_analystics_03.py):

  * trend_168h is computed on native 168h bars, then forward-filled
    to 24h — it can only change every 168 hours.
  * trend_24h is computed on native 24h bars.
  * All components are shift(1) to prevent look-ahead bias.

regime.py's classify_bar() operates at hourly resolution with forward-
filled MAs and detects crossovers at any hour.  That difference is
correct: the 1h path is for live trading, the 24h path is for training
hit rate estimates whose statistical validity was established on 24h
bars.

Usage:
    from cryptobot.signals.features import build_24h_signals
    from cryptobot.signals.regime import RegimeConfig

    signals_24h, returns_24h = build_24h_signals(df_1h)

    # Feed to HitRateCalculator
    calculator.update(pair, signals_24h, returns_24h, current_date)
"""

from typing import Tuple

import numpy as np
import pandas as pd

from cryptobot.signals.regime import RegimeConfig


# =============================================================================
# OHLCV RESAMPLING
# =============================================================================

def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 1h OHLCV to a coarser timeframe.

    Matches backtest v03 resample_ohlcv (line 315-318).
    """
    return df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()


# =============================================================================
# HYSTERESIS LABELLING (batch, series-level)
# =============================================================================

def _label_trend_binary(
    close: pd.Series,
    ma: pd.Series,
    entry_buffer: float,
    exit_buffer: float,
) -> pd.Series:
    """
    Label each bar as bullish (1) or bearish (0) with AND-logic
    hysteresis buffers.

    Matches backtest v03 label_trend_binary_rolling (lines 321-347).
    Uses the SAME buffer values as regime.py's classify_bar — both
    read from RegimeConfig.

    AND-logic (locked):
        Bull → Bear: price < MA * (1 - exit_buffer) AND price < MA * (1 - entry_buffer)
        Bear → Bull: price > MA * (1 + exit_buffer) AND price > MA * (1 + entry_buffer)

    Since entry_buffer (1.5%) > exit_buffer (0.5%), the AND collapses
    to ±entry_buffer symmetric thresholds.
    """
    labels = pd.Series(index=close.index, dtype=float)
    labels[:] = np.nan
    current = 1  # default bullish

    for i in range(len(close)):
        idx = close.index[i]
        price = close.iloc[i]
        ma_val = ma.iloc[i] if idx in ma.index else np.nan

        if pd.isna(ma_val):
            labels.iloc[i] = current
            continue

        if current == 1:
            # Bull → Bear: price must be below BOTH thresholds
            if (price < ma_val * (1 - exit_buffer)
                    and price < ma_val * (1 - entry_buffer)):
                current = 0
        else:
            # Bear → Bull: price must be above BOTH thresholds
            if (price > ma_val * (1 + exit_buffer)
                    and price > ma_val * (1 + entry_buffer)):
                current = 1

        labels.iloc[i] = current

    return labels.astype(int)


# =============================================================================
# PUBLIC API
# =============================================================================

def build_24h_signals(
    df_1h: pd.DataFrame,
    config: RegimeConfig | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate 16-state signals and forward returns at 24h resolution.

    Replicates backtest v03 generate_signals_24h (lines 416-459)
    exactly, using RegimeConfig as single source of truth for buffer
    values and MA periods.

    Args:
        df_1h:   Hourly OHLCV DataFrame with DatetimeIndex.
                 Must contain columns: open, high, low, close, volume.
        config:  RegimeConfig (uses defaults if None).

    Returns:
        signals_24h:  DataFrame indexed on 24h bars with columns:
                      trend_24h, trend_168h, ma72_above_ma24,
                      ma168_above_ma24, state_int
        returns_24h:  Series of 24h close-to-close percentage returns,
                      aligned to signals_24h index.
    """
    cfg = config or RegimeConfig()

    # ------------------------------------------------------------------
    # 1. Resample to native timeframes
    # ------------------------------------------------------------------
    df_24h = _resample_ohlcv(df_1h, '24h')
    df_72h = _resample_ohlcv(df_1h, '72h')
    df_168h = _resample_ohlcv(df_1h, '168h')

    # ------------------------------------------------------------------
    # 2. Moving averages on native bars
    # ------------------------------------------------------------------
    ma_24h = df_24h['close'].rolling(cfg.ma_period_24h).mean()
    ma_72h = df_72h['close'].rolling(cfg.ma_period_72h).mean()
    ma_168h = df_168h['close'].rolling(cfg.ma_period_168h).mean()

    # Forward-fill slower MAs onto 24h index
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')

    # ------------------------------------------------------------------
    # 3. Trend components with hysteresis
    # ------------------------------------------------------------------
    # trend_24h: 24h close vs 24h MA — computed on native 24h bars
    trend_24h = _label_trend_binary(
        df_24h['close'], ma_24h,
        cfg.entry_buffer, cfg.exit_buffer,
    )

    # trend_168h: 168h close vs 168h MA — computed on native 168h bars,
    # then forward-filled to 24h.  This means trend_168h can only change
    # every 168 hours, matching the validated backtest methodology.
    trend_168h_raw = _label_trend_binary(
        df_168h['close'],
        df_168h['close'].rolling(cfg.ma_period_168h).mean(),
        cfg.entry_buffer, cfg.exit_buffer,
    )

    # ------------------------------------------------------------------
    # 4. Assemble on 24h index with shift(1) look-ahead prevention
    # ------------------------------------------------------------------
    signals = pd.DataFrame(index=df_24h.index)
    signals['trend_24h'] = trend_24h.shift(1)
    signals['trend_168h'] = (
        trend_168h_raw
        .shift(1)
        .reindex(df_24h.index, method='ffill')
    )
    signals['ma72_above_ma24'] = (
        (ma_72h_aligned > ma_24h).astype(int).shift(1)
    )
    signals['ma168_above_ma24'] = (
        (ma_168h_aligned > ma_24h).astype(int).shift(1)
    )

    # Drop warm-up NaNs, cast to int
    signals = signals.dropna().astype(int)

    # State integer: trend_24h(8) + trend_168h(4) + ma72>ma24(2) + ma168>ma24(1)
    signals['state_int'] = (
        signals['trend_24h'] * 8
        + signals['trend_168h'] * 4
        + signals['ma72_above_ma24'] * 2
        + signals['ma168_above_ma24'] * 1
    )

    # ------------------------------------------------------------------
    # 5. Returns (24h close-to-close)
    # ------------------------------------------------------------------
    returns_24h = df_24h['close'].pct_change()

    return signals, returns_24h