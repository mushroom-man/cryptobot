# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 09:12:34 2026

@author: John

CryptoBot - Timeframe Aligner
==============================
Align higher timeframe bars to 1h index with proper lag to avoid look-ahead bias.

The Look-Ahead Problem:
    A 24h bar timestamped at 00:00 contains data from 00:00-23:59.
    At 1h bar 06:00 on the same day, that 24h bar is NOT complete.
    We can only use the PREVIOUS completed 24h bar.

Solution:
    Shift higher TF timestamps forward by their period length,
    then forward-fill to 1h index.

Usage:
    from cryptobot.datasources.aligner import Aligner
    
    # Single timeframe
    df_24h_aligned = Aligner.align(
        df_higher=df_24h,
        target_index=df_1h.index,
        source_tf='24h'
    )
    
    # All timeframes combined
    df_combined = Aligner.align_all(
        data={'1h': df_1h, '12h': df_12h, '24h': df_24h},
        base_tf='1h'
    )
    # Columns: open_1h, close_1h, ..., open_24h, close_24h, ...
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


# =============================================================================
# Constants
# =============================================================================

TIMEFRAME_MINUTES = {
    '1h': 60,
    '4h': 240,
    '12h': 720,
    '24h': 1440,
    '72h': 4320,
    '168h': 10080,
}

# Standard OHLCV columns
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'volume_quote']


# =============================================================================
# Alignment Result
# =============================================================================

@dataclass
class AlignmentInfo:
    """Information about alignment operation."""
    source_tf: str
    target_tf: str
    source_rows: int
    target_rows: int
    lag_hours: int
    columns_aligned: List[str]
    first_valid: pd.Timestamp      # First target row with valid data
    coverage: float                # Fraction of target rows with data
    
    def __str__(self) -> str:
        return (
            f"Aligned {self.source_tf} → {self.target_tf}: "
            f"{self.source_rows} → {self.target_rows} rows, "
            f"lag={self.lag_hours}h, coverage={self.coverage:.1%}"
        )


# =============================================================================
# Aligner Class
# =============================================================================

class Aligner:
    """
    Align higher timeframe data to base timeframe index.
    
    All methods are static — no state needed.
    """
    
    @staticmethod
    def align(
        df_higher: pd.DataFrame,
        target_index: pd.DatetimeIndex,
        source_tf: str,
        target_tf: str = '1h',
        columns: List[str] = None,
        add_suffix: bool = True,
    ) -> pd.DataFrame:
        """
        Align higher timeframe data to target index with proper lag.
        
        Args:
            df_higher: DataFrame with higher timeframe OHLCV data
            target_index: DatetimeIndex to align to (typically 1h)
            source_tf: Source timeframe ('12h', '24h', '72h', '168h')
            target_tf: Target timeframe ('1h')
            columns: Columns to align (default: all OHLCV columns present)
            add_suffix: If True, add timeframe suffix to columns
        
        Returns:
            DataFrame aligned to target_index with lagged values.
            Each row contains the last COMPLETED higher TF bar.
        
        Example:
            >>> df_24h_aligned = Aligner.align(
            ...     df_24h, 
            ...     target_index=df_1h.index,
            ...     source_tf='24h'
            ... )
            >>> # Row at 2024-01-02 06:00 contains 2024-01-01 24h bar
        """
        # Validate inputs
        if source_tf not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unknown source timeframe: {source_tf}")
        if target_tf not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unknown target timeframe: {target_tf}")
        
        source_minutes = TIMEFRAME_MINUTES[source_tf]
        target_minutes = TIMEFRAME_MINUTES[target_tf]
        
        if source_minutes < target_minutes:
            raise ValueError(
                f"Source TF ({source_tf}) must be >= target TF ({target_tf})"
            )
        
        # Ensure datetime index
        df_higher = Aligner._ensure_datetime_index(df_higher)
        
        # Select columns
        if columns is None:
            columns = [c for c in OHLCV_COLUMNS if c in df_higher.columns]
        else:
            columns = [c for c in columns if c in df_higher.columns]
        
        if not columns:
            raise ValueError("No valid columns to align")
        
        df_source = df_higher[columns].copy()
        
        # Calculate lag: a bar starting at T is available at T + period_length
        lag = pd.Timedelta(minutes=source_minutes)
        
        # Shift index forward by the lag
        # This means: bar timestamped 00:00 becomes available at 00:00 + 24h = next day 00:00
        df_shifted = df_source.copy()
        df_shifted.index = df_shifted.index + lag
        
        # Reindex to target and forward-fill
        # This propagates the last completed bar to all following target rows
        df_aligned = df_shifted.reindex(target_index, method='ffill')
        
        # Add suffix if requested
        if add_suffix and source_tf != target_tf:
            df_aligned.columns = [f"{col}_{source_tf}" for col in df_aligned.columns]
        
        return df_aligned
    
    @staticmethod
    def align_all(
        data: Dict[str, pd.DataFrame],
        base_tf: str = '1h',
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Align all timeframes to base and merge into single DataFrame.
        
        Args:
            data: Dict mapping timeframe → DataFrame
                  {'1h': df_1h, '12h': df_12h, '24h': df_24h, ...}
            base_tf: Base timeframe to align to (must be in data)
            columns: Columns to include (default: all OHLCV)
        
        Returns:
            Single DataFrame with all timeframes as columns.
            Columns: open_1h, high_1h, ..., open_12h, ..., close_168h
        
        Example:
            >>> df_combined = Aligner.align_all({
            ...     '1h': df_1h,
            ...     '24h': df_24h,
            ...     '168h': df_168h,
            ... })
            >>> print(df_combined.columns.tolist())
            ['open_1h', 'high_1h', ..., 'open_24h', ..., 'close_168h']
        """
        if base_tf not in data:
            raise ValueError(f"Base timeframe '{base_tf}' not in data")
        
        # Get base data and index
        df_base = Aligner._ensure_datetime_index(data[base_tf])
        target_index = df_base.index
        
        # Select columns
        if columns is None:
            columns = [c for c in OHLCV_COLUMNS if c in df_base.columns]
        
        # Start with base timeframe
        df_base_selected = df_base[columns].copy()
        df_base_selected.columns = [f"{col}_{base_tf}" for col in columns]
        
        result = df_base_selected.copy()
        
        # Align and merge each higher timeframe
        timeframes = sorted(
            data.keys(),
            key=lambda tf: TIMEFRAME_MINUTES.get(tf, 0)
        )
        
        for tf in timeframes:
            if tf == base_tf:
                continue
            
            df_aligned = Aligner.align(
                df_higher=data[tf],
                target_index=target_index,
                source_tf=tf,
                target_tf=base_tf,
                columns=columns,
                add_suffix=True,
            )
            
            result = pd.concat([result, df_aligned], axis=1)
        
        return result
    
    @staticmethod
    def get_alignment_info(
        df_higher: pd.DataFrame,
        target_index: pd.DatetimeIndex,
        source_tf: str,
        target_tf: str = '1h',
    ) -> AlignmentInfo:
        """
        Get information about alignment without performing it.
        
        Useful for validation and debugging.
        
        Args:
            df_higher: Higher timeframe DataFrame
            target_index: Target index to align to
            source_tf: Source timeframe
            target_tf: Target timeframe
        
        Returns:
            AlignmentInfo with details
        """
        df_higher = Aligner._ensure_datetime_index(df_higher)
        
        source_minutes = TIMEFRAME_MINUTES[source_tf]
        lag_hours = source_minutes // 60
        
        columns = [c for c in OHLCV_COLUMNS if c in df_higher.columns]
        
        # Calculate first valid target row
        # First higher TF bar becomes available after lag
        first_source = df_higher.index.min()
        first_valid = first_source + pd.Timedelta(minutes=source_minutes)
        
        # Calculate coverage
        valid_target_rows = target_index[target_index >= first_valid]
        coverage = len(valid_target_rows) / len(target_index) if len(target_index) > 0 else 0
        
        return AlignmentInfo(
            source_tf=source_tf,
            target_tf=target_tf,
            source_rows=len(df_higher),
            target_rows=len(target_index),
            lag_hours=lag_hours,
            columns_aligned=columns,
            first_valid=first_valid,
            coverage=coverage,
        )
    
    @staticmethod
    def verify_no_lookahead(
        df_aligned: pd.DataFrame,
        df_source: pd.DataFrame,
        source_tf: str,
    ) -> bool:
        """
        Verify that aligned data has no look-ahead bias.
        
        Checks that values at each target timestamp come from
        a source bar that was complete BEFORE that timestamp.
        
        Args:
            df_aligned: Aligned DataFrame (output of align())
            df_source: Original higher TF DataFrame
            source_tf: Source timeframe
        
        Returns:
            True if no look-ahead bias detected
        
        Raises:
            ValueError: If look-ahead bias detected (with details)
        """
        df_source = Aligner._ensure_datetime_index(df_source)
        source_minutes = TIMEFRAME_MINUTES[source_tf]
        
        # Sample some rows to check
        sample_size = min(100, len(df_aligned))
        sample_indices = np.random.choice(len(df_aligned), sample_size, replace=False)
        
        for idx in sample_indices:
            target_ts = df_aligned.index[idx]
            
            # Find which source bar this value came from
            # The aligned value should be from a bar that COMPLETED before target_ts
            aligned_close = df_aligned.iloc[idx].get('close') or df_aligned.iloc[idx].get(f'close_{source_tf}')
            
            if pd.isna(aligned_close):
                continue
            
            # Find matching source bar
            matching = df_source[df_source['close'] == aligned_close]
            
            if len(matching) == 0:
                continue
            
            source_ts = matching.index[0]
            source_completion = source_ts + pd.Timedelta(minutes=source_minutes)
            
            # Source bar must have completed BEFORE target timestamp
            if source_completion > target_ts:
                raise ValueError(
                    f"Look-ahead detected at {target_ts}: "
                    f"using {source_tf} bar from {source_ts} "
                    f"which completes at {source_completion}"
                )
        
        return True
    
    @staticmethod
    def get_lag_hours(source_tf: str) -> int:
        """Get lag in hours for a timeframe."""
        if source_tf not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unknown timeframe: {source_tf}")
        return TIMEFRAME_MINUTES[source_tf] // 60
    
    @staticmethod
    def explain_alignment(source_tf: str, target_tf: str = '1h') -> str:
        """
        Generate human-readable explanation of alignment logic.
        
        Useful for documentation and debugging.
        """
        source_min = TIMEFRAME_MINUTES.get(source_tf, 0)
        source_hours = source_min // 60
        
        return f"""
Alignment: {source_tf} → {target_tf}
{'=' * 40}

Lag: {source_hours} hours

Logic:
  - A {source_tf} bar timestamped at T covers period [T, T+{source_hours}h)
  - This bar is COMPLETE at T+{source_hours}h
  - First available at next {target_tf} bar: T+{source_hours}h
  
Example:
  - {source_tf} bar at 2024-01-01 00:00 covers 2024-01-01 00:00 to 2024-01-01 {source_hours:02d}:00
  - Available starting at 2024-01-01 {source_hours:02d}:00 (or 2024-01-02 00:00 if {source_hours}>=24)
  - All {target_tf} bars from then until next {source_tf} bar see this value
  
No Look-Ahead:
  - At any {target_tf} timestamp T, we only see {source_tf} bars that
    completed BEFORE T
  - Never see current incomplete {source_tf} bar
"""
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has DatetimeIndex."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        
        if 'timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            return df
        
        raise ValueError(
            "DataFrame must have DatetimeIndex or 'timestamp' column"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def align(
    df_higher: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    source_tf: str,
) -> pd.DataFrame:
    """Convenience function for single alignment."""
    return Aligner.align(df_higher, target_index, source_tf)


def align_all(
    data: Dict[str, pd.DataFrame],
    base_tf: str = '1h',
) -> pd.DataFrame:
    """Convenience function for multi-timeframe alignment."""
    return Aligner.align_all(data, base_tf)