# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 09:06:02 2026

@author: John

CryptoBot - OHLCV Resampler
============================
Resample 1h OHLCV data to higher timeframes (12h, 24h, 72h, 168h).

Uses true aggregation (not rolling windows):
    - Open: First value in period
    - High: Max value in period
    - Low: Min value in period
    - Close: Last value in period
    - Volume: Sum over period

Usage:
    from cryptobot.datasources.resampler import Resampler
    
    # Single timeframe
    df_24h = Resampler.resample(df_1h, target_tf='24h')
    
    # Multiple timeframes
    all_tfs = Resampler.resample_all(df_1h, timeframes=['12h', '24h', '72h', '168h'])
    # Returns: {'1h': df_1h, '12h': df_12h, '24h': df_24h, ...}
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
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

# Standard OHLCV resampling rules
OHLCV_RULES = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'volume_quote': 'sum',
}


# =============================================================================
# Validation Result
# =============================================================================

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    row_count: int
    date_range: Tuple[pd.Timestamp, pd.Timestamp]
    frequency: str
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp, int]]  # start, end, missing_bars
    warnings: List[str]
    errors: List[str]
    
    def __str__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines = [
            f"Validation: {status}",
            f"  Rows: {self.row_count:,}",
            f"  Range: {self.date_range[0]} → {self.date_range[1]}",
            f"  Frequency: {self.frequency}",
            f"  Gaps: {len(self.gaps)}",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings[:3]:
                lines.append(f"    - {w}")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"    - {e}")
        return "\n".join(lines)


# =============================================================================
# Resampler Class
# =============================================================================

class Resampler:
    """
    Resample 1h OHLCV data to higher timeframes.
    
    All methods are static — no state needed.
    """
    
    @staticmethod
    def resample(
        df: pd.DataFrame,
        target_tf: str,
        drop_incomplete: bool = True,
    ) -> pd.DataFrame:
        """
        Resample 1h OHLCV data to target timeframe.
        
        Args:
            df: DataFrame with 1h OHLCV data.
                Must have datetime index and columns: open, high, low, close, volume
                Optional column: volume_quote
            target_tf: Target timeframe ('12h', '24h', '72h', '168h')
            drop_incomplete: If True, drop incomplete periods at end
        
        Returns:
            Resampled DataFrame with same columns, fewer rows.
            Timestamp is period START.
        
        Raises:
            ValueError: If target_tf not recognized or df invalid
        
        Example:
            >>> df_1h = db.get_ohlcv("XBTUSD", start="2020-01-01")
            >>> df_24h = Resampler.resample(df_1h, '24h')
            >>> print(f"1h rows: {len(df_1h)}, 24h rows: {len(df_24h)}")
        """
        # Validate target timeframe
        if target_tf not in TIMEFRAME_MINUTES:
            valid = list(TIMEFRAME_MINUTES.keys())
            raise ValueError(f"Unknown timeframe '{target_tf}'. Valid: {valid}")
        
        if target_tf == '1h':
            return df.copy()
        
        # Validate input
        Resampler._validate_input(df)
        
        # Ensure datetime index
        df = Resampler._ensure_datetime_index(df)
        
        # Build resampling rules for columns present
        rules = {}
        for col, rule in OHLCV_RULES.items():
            if col in df.columns:
                rules[col] = rule
        
        # Calculate resample frequency
        minutes = TIMEFRAME_MINUTES[target_tf]
        freq = f'{minutes}min'
        
        # Resample
        # label='left' → timestamp is period start
        # closed='left' → period includes start, excludes end
        resampled = df.resample(freq, label='left', closed='left').agg(rules)
        
        # Drop rows with NaN (incomplete periods or gaps)
        resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Drop incomplete final period if requested
        if drop_incomplete:
            resampled = Resampler._drop_incomplete_final(
                resampled, df, target_tf
            )
        
        return resampled
    
    @staticmethod
    def resample_all(
        df: pd.DataFrame,
        timeframes: List[str] = None,
        include_source: bool = True,
        drop_incomplete: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample to multiple timeframes at once.
        
        Args:
            df: DataFrame with 1h OHLCV data
            timeframes: List of target timeframes. 
                       Default: ['12h', '24h', '72h', '168h']
            include_source: If True, include '1h' in output
            drop_incomplete: If True, drop incomplete periods
        
        Returns:
            Dict mapping timeframe → DataFrame
            {'1h': df_1h, '12h': df_12h, '24h': df_24h, ...}
        
        Example:
            >>> all_tfs = Resampler.resample_all(df_1h)
            >>> for tf, data in all_tfs.items():
            ...     print(f"{tf}: {len(data)} rows")
        """
        if timeframes is None:
            timeframes = ['12h', '24h', '72h', '168h']
        
        result = {}
        
        # Include source 1h data
        if include_source:
            result['1h'] = df.copy()
        
        # Resample each timeframe
        for tf in timeframes:
            if tf == '1h':
                continue
            result[tf] = Resampler.resample(df, tf, drop_incomplete)
        
        return result
    
    @staticmethod
    def validate(df: pd.DataFrame) -> ValidationResult:
        """
        Validate 1h OHLCV data for resampling.
        
        Checks:
            - Required columns present
            - Datetime index
            - Hourly frequency
            - Gaps in data
            - Data quality (no negative prices, etc.)
        
        Args:
            df: DataFrame to validate
        
        Returns:
            ValidationResult with details
        
        Example:
            >>> result = Resampler.validate(df_1h)
            >>> print(result)
            >>> if not result.is_valid:
            ...     print("Fix errors before resampling")
        """
        errors = []
        warnings = []
        gaps = []
        
        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
        
        # Check for volume_quote (optional but expected)
        if 'volume_quote' not in df.columns:
            warnings.append("Column 'volume_quote' not present")
        
        # Check index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find timestamp column
            if 'timestamp' in df.columns:
                warnings.append("Index is not DatetimeIndex. Use 'timestamp' column.")
            else:
                errors.append("Index must be DatetimeIndex or 'timestamp' column must exist")
        
        # Get date range
        if len(df) > 0:
            if isinstance(df.index, pd.DatetimeIndex):
                start = df.index.min()
                end = df.index.max()
            elif 'timestamp' in df.columns:
                start = pd.to_datetime(df['timestamp'].min())
                end = pd.to_datetime(df['timestamp'].max())
            else:
                start = end = pd.NaT
        else:
            errors.append("DataFrame is empty")
            start = end = pd.NaT
        
        # Check frequency and gaps
        freq = "unknown"
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            # Infer frequency
            diffs = df.index.to_series().diff().dropna()
            median_diff = diffs.median()
            
            if pd.Timedelta(hours=0.9) <= median_diff <= pd.Timedelta(hours=1.1):
                freq = "1h"
            else:
                freq = str(median_diff)
                warnings.append(f"Expected 1h frequency, found {freq}")
            
            # Find gaps (missing hours)
            expected_diff = pd.Timedelta(hours=1)
            gap_threshold = pd.Timedelta(hours=1.5)
            
            gap_mask = diffs > gap_threshold
            if gap_mask.any():
                gap_indices = diffs[gap_mask].index
                for gap_end in gap_indices:
                    gap_start = df.index[df.index.get_loc(gap_end) - 1]
                    missing_bars = int((gap_end - gap_start) / expected_diff) - 1
                    gaps.append((gap_start, gap_end, missing_bars))
                
                total_missing = sum(g[2] for g in gaps)
                warnings.append(f"Found {len(gaps)} gaps, {total_missing} missing bars total")
        
        # Data quality checks
        if not errors and len(df) > 0:
            # Negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns and (df[col] <= 0).any():
                    errors.append(f"Non-positive values in '{col}'")
            
            # High < Low
            if 'high' in df.columns and 'low' in df.columns:
                invalid_hl = (df['high'] < df['low']).sum()
                if invalid_hl > 0:
                    errors.append(f"High < Low in {invalid_hl} rows")
            
            # Negative volume
            if 'volume' in df.columns and (df['volume'] < 0).any():
                errors.append("Negative values in 'volume'")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            row_count=len(df),
            date_range=(start, end),
            frequency=freq,
            gaps=gaps,
            warnings=warnings,
            errors=errors,
        )
    
    @staticmethod
    def get_timeframe_minutes(tf: str) -> int:
        """Get minutes for a timeframe string."""
        if tf not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unknown timeframe: {tf}")
        return TIMEFRAME_MINUTES[tf]
    
    @staticmethod
    def get_resample_factor(source_tf: str, target_tf: str) -> int:
        """
        Get number of source bars per target bar.
        
        Example:
            >>> Resampler.get_resample_factor('1h', '24h')
            24
        """
        source_min = TIMEFRAME_MINUTES.get(source_tf)
        target_min = TIMEFRAME_MINUTES.get(target_tf)
        
        if source_min is None:
            raise ValueError(f"Unknown source timeframe: {source_tf}")
        if target_min is None:
            raise ValueError(f"Unknown target timeframe: {target_tf}")
        
        if target_min % source_min != 0:
            raise ValueError(
                f"Target {target_tf} not evenly divisible by source {source_tf}"
            )
        
        return target_min // source_min
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """Raise ValueError if input is invalid."""
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
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
    
    @staticmethod
    def _drop_incomplete_final(
        resampled: pd.DataFrame,
        source: pd.DataFrame,
        target_tf: str,
    ) -> pd.DataFrame:
        """
        Drop the final bar if it's incomplete.
        
        A bar is incomplete if the source data doesn't cover
        the full period.
        """
        if len(resampled) == 0:
            return resampled
        
        # Get the last resampled timestamp
        last_bar_start = resampled.index[-1]
        
        # Calculate when this bar should end
        minutes = TIMEFRAME_MINUTES[target_tf]
        last_bar_end = last_bar_start + pd.Timedelta(minutes=minutes)
        
        # Get the actual last source timestamp
        source_end = source.index.max()
        
        # If source doesn't cover the full period, drop last bar
        # Allow 1 hour tolerance (the final hour of the period)
        expected_final_hour = last_bar_end - pd.Timedelta(hours=1)
        
        if source_end < expected_final_hour:
            resampled = resampled.iloc[:-1]
        
        return resampled


# =============================================================================
# Convenience Functions
# =============================================================================

def resample(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Convenience function for single resample."""
    return Resampler.resample(df, target_tf)


def resample_all(
    df: pd.DataFrame,
    timeframes: List[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience function for multi-resample."""
    return Resampler.resample_all(df, timeframes)