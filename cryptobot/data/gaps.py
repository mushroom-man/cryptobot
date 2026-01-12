# -*- coding: utf-8 -*-

"""
Created on Wed Jan  7 09:08:35 2026

@author: John

CryptoBot - Gap Detection and Handling
=======================================
Detect and handle missing bars in 1h OHLCV data.

Gap Severity Levels:
    - Small (≤5 bars): Impute silently
    - Medium (6-24 bars): Impute + flag
    - Large (25-168 bars): Flag, warn user
    - Fatal (>168 bars): Split into segments

Usage:
    from cryptobot.datasources.gaps import GapHandler, GapInfo
    
    # Detect gaps
    gaps = GapHandler.detect(df_1h)
    for gap in gaps:
        print(f"{gap.severity}: {gap.bars_missing} bars at {gap.start}")
    
    # Impute small/medium gaps
    df_clean, quality = GapHandler.impute(df_1h)
    
    # Full pipeline (detect, impute, segment)
    result = GapHandler.process(df_1h)
    print(result.report())
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

class GapSeverity(Enum):
    """Gap severity levels."""
    SMALL = "small"      # ≤5 bars - impute silently
    MEDIUM = "medium"    # 6-24 bars - impute + flag
    LARGE = "large"      # 25-168 bars - flag, warn
    FATAL = "fatal"      # >168 bars - split segments


# Thresholds (in bars/hours for 1h data)
GAP_THRESHOLDS = {
    'small': 5,      # ≤5 bars
    'medium': 24,    # 6-24 bars
    'large': 168,    # 25-168 bars
    'fatal': 168,    # >168 bars triggers segment split
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GapInfo:
    """Information about a single gap in the data."""
    start: pd.Timestamp          # Last timestamp before gap
    end: pd.Timestamp            # First timestamp after gap
    bars_missing: int            # Number of missing bars
    severity: GapSeverity        # Severity classification
    
    @property
    def duration(self) -> pd.Timedelta:
        """Duration of the gap."""
        return self.end - self.start
    
    @property
    def hours_missing(self) -> float:
        """Hours of missing data."""
        return self.bars_missing  # For 1h data, bars = hours
    
    def __str__(self) -> str:
        return (
            f"Gap({self.severity.value}): {self.bars_missing} bars "
            f"from {self.start} to {self.end}"
        )


@dataclass
class DataSegment:
    """A continuous segment of data (no fatal gaps)."""
    start: pd.Timestamp
    end: pd.Timestamp
    df: pd.DataFrame
    row_count: int
    gaps_imputed: int            # Number of gaps that were imputed
    bars_imputed: int            # Total bars that were imputed
    
    @property
    def duration_days(self) -> float:
        """Segment duration in days."""
        return (self.end - self.start).total_seconds() / 86400
    
    def __str__(self) -> str:
        return (
            f"Segment: {self.start.date()} → {self.end.date()} "
            f"({self.row_count:,} rows, {self.duration_days:.1f} days)"
        )


@dataclass
class GapResult:
    """Result of gap processing."""
    # Original data info
    original_rows: int
    date_range: Tuple[pd.Timestamp, pd.Timestamp]
    
    # Gap analysis
    gaps: List[GapInfo]
    total_bars_missing: int
    
    # Counts by severity
    gaps_small: int
    gaps_medium: int
    gaps_large: int
    gaps_fatal: int
    
    # Output
    segments: List[DataSegment]
    
    # Quality info
    bars_imputed: int
    impute_rate: float           # Fraction of output that was imputed
    
    @property
    def has_fatal_gaps(self) -> bool:
        """True if data has fatal gaps (split into segments)."""
        return self.gaps_fatal > 0
    
    @property
    def is_continuous(self) -> bool:
        """True if data is one continuous segment."""
        return len(self.segments) == 1
    
    @property
    def total_output_rows(self) -> int:
        """Total rows across all segments."""
        return sum(s.row_count for s in self.segments)
    
    def get_largest_segment(self) -> Optional[DataSegment]:
        """Return the segment with most rows."""
        if not self.segments:
            return None
        return max(self.segments, key=lambda s: s.row_count)
    
    def get_combined_df(self) -> pd.DataFrame:
        """
        Combine all segments into one DataFrame.
        
        Warning: This includes gaps between segments!
        Only use if you understand the implications.
        """
        if not self.segments:
            return pd.DataFrame()
        if len(self.segments) == 1:
            return self.segments[0].df
        return pd.concat([s.df for s in self.segments])
    
    def report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 50,
            "Gap Analysis Report",
            "=" * 50,
            f"Original rows: {self.original_rows:,}",
            f"Date range: {self.date_range[0]} → {self.date_range[1]}",
            "",
            "Gaps Found:",
            f"  Small (≤5 bars):    {self.gaps_small}",
            f"  Medium (6-24 bars): {self.gaps_medium}",
            f"  Large (25-168):     {self.gaps_large}",
            f"  Fatal (>168):       {self.gaps_fatal}",
            f"  Total missing bars: {self.total_bars_missing:,}",
            "",
            f"Imputation:",
            f"  Bars imputed: {self.bars_imputed:,}",
            f"  Impute rate:  {self.impute_rate:.2%}",
            "",
            f"Output Segments: {len(self.segments)}",
        ]
        
        for i, seg in enumerate(self.segments, 1):
            lines.append(f"  {i}. {seg}")
        
        if self.has_fatal_gaps:
            lines.append("")
            lines.append("⚠ WARNING: Fatal gaps found. Data split into segments.")
            lines.append("  Consider using only the largest segment or handling separately.")
        
        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# Gap Handler Class
# =============================================================================

class GapHandler:
    """
    Detect and handle gaps in OHLCV data.
    
    All methods are static — no state needed.
    """
    
    @staticmethod
    def detect(
        df: pd.DataFrame,
        expected_freq: str = '1h',
    ) -> List[GapInfo]:
        """
        Detect gaps in time series data.
        
        Args:
            df: DataFrame with DatetimeIndex
            expected_freq: Expected frequency ('1h')
        
        Returns:
            List of GapInfo objects, sorted by start time
        
        Example:
            >>> gaps = GapHandler.detect(df_1h)
            >>> print(f"Found {len(gaps)} gaps")
            >>> for gap in gaps:
            ...     print(gap)
        """
        df = GapHandler._ensure_datetime_index(df)
        
        if len(df) < 2:
            return []
        
        # Calculate time differences
        expected_delta = pd.Timedelta(expected_freq)
        diffs = df.index.to_series().diff()
        
        # Find gaps (where diff > expected)
        gap_threshold = expected_delta * 1.5  # Allow small tolerance
        gap_mask = diffs > gap_threshold
        
        gaps = []
        gap_indices = diffs[gap_mask].index
        
        for gap_end in gap_indices:
            # Get the timestamp before the gap
            loc = df.index.get_loc(gap_end)
            gap_start = df.index[loc - 1]
            
            # Calculate missing bars
            actual_diff = gap_end - gap_start
            bars_missing = int(actual_diff / expected_delta) - 1
            
            # Classify severity
            severity = GapHandler._classify_severity(bars_missing)
            
            gaps.append(GapInfo(
                start=gap_start,
                end=gap_end,
                bars_missing=bars_missing,
                severity=severity,
            ))
        
        return sorted(gaps, key=lambda g: g.start)
    
    @staticmethod
    def impute(
        df: pd.DataFrame,
        gaps: List[GapInfo] = None,
        max_gap: int = 24,
        method: str = 'ffill',
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Impute (fill) gaps up to max_gap size.
        
        Args:
            df: DataFrame with OHLCV data
            gaps: Pre-detected gaps (if None, will detect)
            max_gap: Maximum gap size to impute (default 24 = medium)
            method: Imputation method ('ffill' only currently)
        
        Returns:
            Tuple of:
                - DataFrame with gaps filled
                - Series with data quality flags
        
        Example:
            >>> df_clean, quality = GapHandler.impute(df_1h)
            >>> print(f"Imputed {(quality != 'original').sum()} bars")
        """
        df = GapHandler._ensure_datetime_index(df)
        
        if gaps is None:
            gaps = GapHandler.detect(df)
        
        # Filter to imputable gaps
        imputable = [g for g in gaps if g.bars_missing <= max_gap]
        
        if not imputable:
            # No gaps to fill
            quality = pd.Series('original', index=df.index)
            return df.copy(), quality
        
        # Create complete datetime index
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='1h',
        )
        
        # Reindex to full range (creates NaN for missing)
        df_full = df.reindex(full_index)
        
        # Track quality
        quality = pd.Series('original', index=full_index)
        
        # Mark imputed rows by severity
        for gap in imputable:
            # Get timestamps that fall within this gap
            gap_timestamps = full_index[
                (full_index > gap.start) & (full_index < gap.end)
            ]
            
            if gap.severity == GapSeverity.SMALL:
                quality.loc[gap_timestamps] = 'imputed_small'
            else:
                quality.loc[gap_timestamps] = 'imputed_medium'
        
        # Forward fill to impute
        df_filled = df_full.ffill()
        
        # Mark large/fatal gaps as unfilled (leave NaN)
        large_gaps = [g for g in gaps if g.bars_missing > max_gap]
        for gap in large_gaps:
            gap_timestamps = full_index[
                (full_index > gap.start) & (full_index < gap.end)
            ]
            # Restore NaN for large gaps
            df_filled.loc[gap_timestamps] = np.nan
            quality.loc[gap_timestamps] = 'gap_unfilled'
        
        return df_filled, quality
    
    @staticmethod
    def process(
        df: pd.DataFrame,
        impute_max: int = 24,
        segment_threshold: int = 168,
    ) -> GapResult:
        """
        Full gap processing pipeline.
        
        1. Detect all gaps
        2. Impute small/medium gaps
        3. Split on fatal gaps
        4. Return segments with quality info
        
        Args:
            df: DataFrame with OHLCV data
            impute_max: Maximum gap to impute (default 24)
            segment_threshold: Gap size that triggers segment split (default 168)
        
        Returns:
            GapResult with segments and full report
        
        Example:
            >>> result = GapHandler.process(df_1h)
            >>> print(result.report())
            >>> 
            >>> if result.is_continuous:
            ...     df_clean = result.segments[0].df
            >>> else:
            ...     df_clean = result.get_largest_segment().df
        """
        df = GapHandler._ensure_datetime_index(df)
        original_rows = len(df)
        date_range = (df.index.min(), df.index.max())
        
        # Detect gaps
        gaps = GapHandler.detect(df)
        
        # Count by severity
        gaps_small = sum(1 for g in gaps if g.severity == GapSeverity.SMALL)
        gaps_medium = sum(1 for g in gaps if g.severity == GapSeverity.MEDIUM)
        gaps_large = sum(1 for g in gaps if g.severity == GapSeverity.LARGE)
        gaps_fatal = sum(1 for g in gaps if g.severity == GapSeverity.FATAL)
        total_missing = sum(g.bars_missing for g in gaps)
        
        # Find fatal gaps (segment boundaries)
        fatal_gaps = [g for g in gaps if g.bars_missing > segment_threshold]
        
        # Split into segments
        segments = []
        
        if not fatal_gaps:
            # One continuous segment
            df_imputed, quality = GapHandler.impute(df, gaps, max_gap=impute_max)
            
            # Drop any remaining NaN rows
            df_clean = df_imputed.dropna(subset=['close'])
            quality_clean = quality.loc[df_clean.index]
            
            bars_imputed = (quality_clean != 'original').sum()
            
            segments.append(DataSegment(
                start=df_clean.index.min(),
                end=df_clean.index.max(),
                df=df_clean.copy(),
                row_count=len(df_clean),
                gaps_imputed=gaps_small + gaps_medium,
                bars_imputed=bars_imputed,
            ))
            
            total_bars_imputed = bars_imputed
        
        else:
            # Multiple segments
            total_bars_imputed = 0
            
            # Create segment boundaries
            boundaries = [df.index.min()]
            for gap in fatal_gaps:
                boundaries.append(gap.start)
                boundaries.append(gap.end)
            boundaries.append(df.index.max())
            
            # Process each segment
            for i in range(0, len(boundaries) - 1, 2):
                seg_start = boundaries[i]
                seg_end = boundaries[i + 1] if i + 1 < len(boundaries) else df.index.max()
                
                # Extract segment
                mask = (df.index >= seg_start) & (df.index <= seg_end)
                df_segment = df.loc[mask].copy()
                
                if len(df_segment) == 0:
                    continue
                
                # Find gaps within this segment
                segment_gaps = [
                    g for g in gaps 
                    if g.start >= seg_start and g.end <= seg_end
                    and g.bars_missing <= segment_threshold
                ]
                
                # Impute within segment
                df_imputed, quality = GapHandler.impute(
                    df_segment, segment_gaps, max_gap=impute_max
                )
                
                # Drop NaN
                df_clean = df_imputed.dropna(subset=['close'])
                
                if len(df_clean) == 0:
                    continue
                
                quality_clean = quality.loc[df_clean.index]
                bars_imputed = (quality_clean != 'original').sum()
                total_bars_imputed += bars_imputed
                
                # Count imputed gaps in this segment
                imputed_count = sum(
                    1 for g in segment_gaps 
                    if g.bars_missing <= impute_max
                )
                
                segments.append(DataSegment(
                    start=df_clean.index.min(),
                    end=df_clean.index.max(),
                    df=df_clean.copy(),
                    row_count=len(df_clean),
                    gaps_imputed=imputed_count,
                    bars_imputed=bars_imputed,
                ))
        
        # Calculate impute rate
        total_output = sum(s.row_count for s in segments)
        impute_rate = total_bars_imputed / total_output if total_output > 0 else 0
        
        return GapResult(
            original_rows=original_rows,
            date_range=date_range,
            gaps=gaps,
            total_bars_missing=total_missing,
            gaps_small=gaps_small,
            gaps_medium=gaps_medium,
            gaps_large=gaps_large,
            gaps_fatal=gaps_fatal,
            segments=segments,
            bars_imputed=total_bars_imputed,
            impute_rate=impute_rate,
        )
    
    @staticmethod
    def summary(df: pd.DataFrame) -> str:
        """
        Quick summary of data quality.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Short summary string
        """
        gaps = GapHandler.detect(df)
        
        if not gaps:
            return f"✓ No gaps found in {len(df):,} rows"
        
        total_missing = sum(g.bars_missing for g in gaps)
        fatal = sum(1 for g in gaps if g.severity == GapSeverity.FATAL)
        
        if fatal > 0:
            return (
                f"⚠ {len(gaps)} gaps ({total_missing:,} bars missing), "
                f"{fatal} fatal - will split into segments"
            )
        else:
            return f"! {len(gaps)} gaps ({total_missing:,} bars missing) - will impute"
    
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
    
    @staticmethod
    def _classify_severity(bars_missing: int) -> GapSeverity:
        """Classify gap severity based on missing bars."""
        if bars_missing <= GAP_THRESHOLDS['small']:
            return GapSeverity.SMALL
        elif bars_missing <= GAP_THRESHOLDS['medium']:
            return GapSeverity.MEDIUM
        elif bars_missing <= GAP_THRESHOLDS['large']:
            return GapSeverity.LARGE
        else:
            return GapSeverity.FATAL


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_gaps(df: pd.DataFrame) -> List[GapInfo]:
    """Convenience function to detect gaps."""
    return GapHandler.detect(df)


def process_gaps(df: pd.DataFrame) -> GapResult:
    """Convenience function for full gap processing."""
    return GapHandler.process(df)
