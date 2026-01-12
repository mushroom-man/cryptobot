# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 09:18:23 2026

@author: John

CryptoBot - Multi-Timeframe Data Container
============================================
Container class for holding multi-asset, multi-timeframe OHLCV data.

Provides convenient access methods for:
    - Raw data by pair/timeframe
    - Aligned data (all TFs merged to base)
    - Close prices only
    - Coverage and quality reports

Usage:
    from cryptobot.datasources.container import MultiTimeframeData
    
    # Created by DataLoader (not directly)
    data = DataLoader.load(pairs=['XBTUSD', 'ETHUSD'], ...)
    
    # Access raw data
    df_btc_1h = data.get('XBTUSD', '1h')
    
    # Access aligned data (for feature computation)
    df_btc_aligned = data.get_aligned('XBTUSD')
    
    # Quick summary
    data.info()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# Data Container
# =============================================================================

@dataclass
class MultiTimeframeData:
    """
    Container for multi-asset, multi-timeframe OHLCV data.
    
    Holds both raw and aligned data, plus quality information.
    
    Attributes:
        pairs: List of trading pairs loaded
        timeframes: List of timeframes available
        base_tf: Base timeframe for alignment (typically '1h')
        start: Start of date range
        end: End of date range
    """
    
    # Metadata
    pairs: List[str]
    timeframes: List[str]
    base_tf: str
    start: datetime
    end: datetime
    
    # Internal storage (initialized via factory or post_init)
    _raw: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict, repr=False)
    _aligned: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _quality: Dict[str, pd.Series] = field(default_factory=dict, repr=False)
    _coverage: pd.DataFrame = field(default=None, repr=False)
    
    # =========================================================================
    # Access Methods
    # =========================================================================
    
    def get(self, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Get raw OHLCV data for one pair and timeframe.
        
        Args:
            pair: Trading pair (e.g., 'XBTUSD')
            timeframe: Timeframe (e.g., '1h', '24h')
        
        Returns:
            DataFrame with columns: open, high, low, close, volume, volume_quote
            Index: timestamp (DatetimeIndex)
        
        Raises:
            KeyError: If pair or timeframe not found
        
        Example:
            >>> df = data.get('XBTUSD', '1h')
            >>> print(df.head())
        """
        if pair not in self._raw:
            available = list(self._raw.keys())
            raise KeyError(f"Pair '{pair}' not found. Available: {available}")
        
        if timeframe not in self._raw[pair]:
            available = list(self._raw[pair].keys())
            raise KeyError(f"Timeframe '{timeframe}' not found for {pair}. Available: {available}")
        
        return self._raw[pair][timeframe].copy()
    
    def get_aligned(self, pair: str) -> pd.DataFrame:
        """
        Get all timeframes aligned to base for one pair.
        
        Higher timeframes are lagged appropriately to avoid look-ahead.
        
        Args:
            pair: Trading pair
        
        Returns:
            DataFrame with columns:
                open_1h, high_1h, low_1h, close_1h, volume_1h,
                open_12h, high_12h, ..., close_168h
            Index: base timeframe timestamps
        
        Example:
            >>> df_aligned = data.get_aligned('XBTUSD')
            >>> # Use for feature computation across timeframes
        """
        if pair not in self._aligned:
            available = list(self._aligned.keys())
            raise KeyError(f"Pair '{pair}' not found. Available: {available}")
        
        return self._aligned[pair].copy()
    
    def get_close_prices(self, pair: str) -> pd.DataFrame:
        """
        Get just close prices across all timeframes for one pair.
        
        Convenience method for quick analysis.
        
        Args:
            pair: Trading pair
        
        Returns:
            DataFrame with columns: close_1h, close_12h, close_24h, ...
        """
        df = self.get_aligned(pair)
        close_cols = [c for c in df.columns if c.startswith('close_')]
        return df[close_cols].copy()
    
    def get_ohlc(self, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Get OHLC (no volume) for one pair/timeframe.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe
        
        Returns:
            DataFrame with columns: open, high, low, close
        """
        df = self.get(pair, timeframe)
        return df[['open', 'high', 'low', 'close']].copy()
    
    def get_all_pairs_raw(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Get raw data for all pairs at one timeframe.
        
        Args:
            timeframe: Timeframe to get
        
        Returns:
            Dict mapping pair → DataFrame
        """
        result = {}
        for pair in self.pairs:
            if timeframe in self._raw.get(pair, {}):
                result[pair] = self._raw[pair][timeframe].copy()
        return result
    
    def get_all_pairs_aligned(self) -> Dict[str, pd.DataFrame]:
        """
        Get aligned data for all pairs.
        
        Returns:
            Dict mapping pair → aligned DataFrame
        
        Example:
            >>> all_data = data.get_all_pairs_aligned()
            >>> for pair, df in all_data.items():
            ...     print(f"{pair}: {len(df)} rows")
        """
        return {pair: df.copy() for pair, df in self._aligned.items()}
    
    def get_quality(self, pair: str) -> pd.Series:
        """
        Get data quality flags for a pair.
        
        Args:
            pair: Trading pair
        
        Returns:
            Series with values: 'original', 'imputed_small', 'imputed_medium'
        """
        if pair not in self._quality:
            raise KeyError(f"Quality data not found for '{pair}'")
        return self._quality[pair].copy()
    
    # =========================================================================
    # Reporting Methods
    # =========================================================================
    
    def coverage(self) -> pd.DataFrame:
        """
        Report data coverage for each pair.
        
        Returns:
            DataFrame with columns:
                pair, start, end, days, bars_1h, gaps, imputed_pct, complete
        
        Example:
            >>> print(data.coverage())
        """
        if self._coverage is not None:
            return self._coverage.copy()
        
        rows = []
        for pair in self.pairs:
            if pair not in self._raw or self.base_tf not in self._raw[pair]:
                continue
            
            df = self._raw[pair][self.base_tf]
            quality = self._quality.get(pair)
            
            start = df.index.min()
            end = df.index.max()
            days = (end - start).total_seconds() / 86400
            bars = len(df)
            
            # Count imputed
            imputed_pct = 0.0
            if quality is not None:
                imputed = (quality != 'original').sum()
                imputed_pct = imputed / len(quality) if len(quality) > 0 else 0.0
            
            # Check completeness (expected bars vs actual)
            expected_bars = int(days * 24) + 1
            complete = bars >= expected_bars * 0.99  # Allow 1% tolerance
            
            rows.append({
                'pair': pair,
                'start': start,
                'end': end,
                'days': round(days, 1),
                'bars_1h': bars,
                'imputed_pct': round(imputed_pct * 100, 2),
                'complete': complete,
            })
        
        self._coverage = pd.DataFrame(rows)
        return self._coverage.copy()
    
    def quality_report(self) -> pd.DataFrame:
        """
        Detailed quality report for all pairs.
        
        Returns:
            DataFrame with quality breakdown per pair
        """
        rows = []
        for pair in self.pairs:
            quality = self._quality.get(pair)
            if quality is None:
                continue
            
            total = len(quality)
            original = (quality == 'original').sum()
            imputed_small = (quality == 'imputed_small').sum()
            imputed_medium = (quality == 'imputed_medium').sum()
            unfilled = (quality == 'gap_unfilled').sum()
            
            rows.append({
                'pair': pair,
                'total_bars': total,
                'original': original,
                'original_pct': round(original / total * 100, 2) if total > 0 else 0,
                'imputed_small': imputed_small,
                'imputed_medium': imputed_medium,
                'gap_unfilled': unfilled,
            })
        
        return pd.DataFrame(rows)
    
    def info(self) -> None:
        """
        Print summary of loaded data.
        
        Example:
            >>> data.info()
            MultiTimeframeData Summary
            ==========================
            Pairs: 12
            Timeframes: ['1h', '12h', '24h', '72h', '168h']
            Date range: 2019-01-01 → 2024-12-31 (2191 days)
            ...
        """
        print("=" * 50)
        print("MultiTimeframeData Summary")
        print("=" * 50)
        print(f"Pairs: {len(self.pairs)}")
        print(f"  {', '.join(self.pairs[:5])}" + ("..." if len(self.pairs) > 5 else ""))
        print(f"Timeframes: {self.timeframes}")
        print(f"Base TF: {self.base_tf}")
        
        if self.start and self.end:
            days = (self.end - self.start).days
            print(f"Date range: {self.start.date()} → {self.end.date()} ({days} days)")
        
        # Row counts
        print("\nRow counts (base TF):")
        for pair in self.pairs[:5]:
            if pair in self._raw and self.base_tf in self._raw[pair]:
                rows = len(self._raw[pair][self.base_tf])
                print(f"  {pair}: {rows:,}")
        if len(self.pairs) > 5:
            print(f"  ... and {len(self.pairs) - 5} more")
        
        # Memory estimate
        mem_mb = self._estimate_memory_mb()
        print(f"\nMemory: ~{mem_mb:.1f} MB")
        
        print("=" * 50)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self.pairs)
    
    def __contains__(self, pair: str) -> bool:
        """Check if pair is loaded."""
        return pair in self.pairs
    
    def __iter__(self):
        """Iterate over pairs."""
        return iter(self.pairs)
    
    def __getitem__(self, pair: str) -> pd.DataFrame:
        """
        Shorthand for get_aligned(pair).
        
        Example:
            >>> df = data['XBTUSD']  # Same as data.get_aligned('XBTUSD')
        """
        return self.get_aligned(pair)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Return (n_pairs, n_timeframes, n_bars).
        
        n_bars is for base timeframe of first pair.
        """
        n_bars = 0
        if self.pairs and self._raw.get(self.pairs[0], {}).get(self.base_tf) is not None:
            n_bars = len(self._raw[self.pairs[0]][self.base_tf])
        return (len(self.pairs), len(self.timeframes), n_bars)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary (for serialization).
        
        Note: Does not include DataFrames.
        """
        return {
            'pairs': self.pairs,
            'timeframes': self.timeframes,
            'base_tf': self.base_tf,
            'start': self.start.isoformat() if self.start else None,
            'end': self.end.isoformat() if self.end else None,
            'shape': self.shape,
        }
    
    def validate(self) -> List[str]:
        """
        Validate internal consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check all pairs have base_tf
        for pair in self.pairs:
            if pair not in self._raw:
                errors.append(f"Pair '{pair}' has no raw data")
            elif self.base_tf not in self._raw[pair]:
                errors.append(f"Pair '{pair}' missing base TF '{self.base_tf}'")
        
        # Check aligned data exists
        for pair in self.pairs:
            if pair not in self._aligned:
                errors.append(f"Pair '{pair}' has no aligned data")
        
        # Check date alignment
        if len(self.pairs) >= 2:
            indices = []
            for pair in self.pairs[:2]:
                if pair in self._aligned:
                    indices.append(self._aligned[pair].index)
            
            if len(indices) == 2 and not indices[0].equals(indices[1]):
                errors.append("Aligned data has inconsistent indices across pairs")
        
        return errors
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        total_bytes = 0
        
        for pair in self._raw:
            for tf in self._raw[pair]:
                total_bytes += self._raw[pair][tf].memory_usage(deep=True).sum()
        
        for pair in self._aligned:
            total_bytes += self._aligned[pair].memory_usage(deep=True).sum()
        
        return total_bytes / (1024 * 1024)
    
    def _set_raw(self, pair: str, timeframe: str, df: pd.DataFrame) -> None:
        """Set raw data (used by DataLoader)."""
        if pair not in self._raw:
            self._raw[pair] = {}
        self._raw[pair][timeframe] = df
    
    def _set_aligned(self, pair: str, df: pd.DataFrame) -> None:
        """Set aligned data (used by DataLoader)."""
        self._aligned[pair] = df
    
    def _set_quality(self, pair: str, quality: pd.Series) -> None:
        """Set quality flags (used by DataLoader)."""
        self._quality[pair] = quality


# =============================================================================
# Factory Function
# =============================================================================

def create_container(
    pairs: List[str],
    timeframes: List[str],
    base_tf: str = '1h',
    start: datetime = None,
    end: datetime = None,
) -> MultiTimeframeData:
    """
    Create an empty MultiTimeframeData container.
    
    Used by DataLoader to build up the container incrementally.
    
    Args:
        pairs: List of trading pairs
        timeframes: List of timeframes
        base_tf: Base timeframe
        start: Start date
        end: End date
    
    Returns:
        Empty MultiTimeframeData ready to be populated
    """
    return MultiTimeframeData(
        pairs=pairs,
        timeframes=timeframes,
        base_tf=base_tf,
        start=start,
        end=end,
        _raw={},
        _aligned={},
        _quality={},
        _coverage=None,
    )