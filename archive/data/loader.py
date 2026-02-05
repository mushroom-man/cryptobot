# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 09:21:57 2026

@author: John

CryptoBot - Multi-Timeframe Data Loader
========================================
Main entry point for loading multi-asset, multi-timeframe OHLCV data.

Orchestrates:
    - Database reading (1h data)
    - Gap detection and imputation
    - Resampling to higher timeframes
    - Alignment with proper lag
    - Packaging into MultiTimeframeData container

Usage:
    from cryptobot.datasources.data_loader import DataLoader
    
    # Load multiple pairs with all timeframes
    data = DataLoader.load(
        pairs=['XBTUSD', 'ETHUSD', 'LTCUSD'],
        timeframes=['1h', '12h', '24h', '72h', '168h'],
        start='2020-01-01',
        end='2024-12-31',
    )
    
    # Check what's available
    print(DataLoader.available_pairs())
    print(DataLoader.date_range('XBTUSD'))
    
    # Access loaded data
    data.info()
    df_btc = data.get_aligned('XBTUSD')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# Internal imports
from cryptobot.datasources.database import Database
from cryptobot.datasources.resampler import Resampler
from cryptobot.datasources.gaps import GapHandler, GapResult
from cryptobot.datasources.aligner import Aligner
from cryptobot.datasources.container import MultiTimeframeData, create_container


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader."""
    
    # Database
    connection_url: Optional[str] = None  # Uses DATABASE_URL env var if None
    
    # Timeframes
    default_timeframes: List[str] = None
    base_timeframe: str = '1h'
    
    # Gap handling
    gap_impute_max: int = 24          # Impute gaps up to 24 bars
    gap_segment_threshold: int = 168  # Split on gaps > 168 bars
    
    # Data requirements
    require_complete: bool = False    # Only include pairs with full coverage
    min_bars: int = 1000              # Minimum bars required per pair
    
    def __post_init__(self):
        if self.default_timeframes is None:
            self.default_timeframes = ['1h', '12h', '24h', '72h', '168h']


# Default configuration
DEFAULT_CONFIG = DataLoaderConfig()


# =============================================================================
# Load Result
# =============================================================================

@dataclass
class LoadResult:
    """Detailed result from loading operation."""
    
    success: bool
    data: Optional[MultiTimeframeData]
    
    # Stats
    pairs_requested: int
    pairs_loaded: int
    pairs_excluded: List[str]
    exclusion_reasons: Dict[str, str]
    
    # Timing
    load_time_seconds: float
    
    # Warnings
    warnings: List[str]
    
    def summary(self) -> str:
        """Generate summary string."""
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        lines = [
            f"Load Result: {status}",
            f"  Pairs: {self.pairs_loaded}/{self.pairs_requested} loaded",
            f"  Time: {self.load_time_seconds:.2f}s",
        ]
        
        if self.pairs_excluded:
            lines.append(f"  Excluded: {', '.join(self.pairs_excluded)}")
        
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings[:3]:
                lines.append(f"    - {w}")
        
        return "\n".join(lines)


# =============================================================================
# Data Loader Class
# =============================================================================

class DataLoader:
    """
    Main entry point for loading multi-asset, multi-timeframe data.
    
    All primary methods are static for convenience.
    Instance methods available for custom configuration.
    """
    
    def __init__(self, config: DataLoaderConfig = None):
        """
        Initialize with custom configuration.
        
        Args:
            config: DataLoaderConfig (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG
        self._db = None
    
    @property
    def db(self) -> Database:
        """Lazy-load database connection."""
        if self._db is None:
            self._db = Database(self.config.connection_url)
        return self._db
    
    # =========================================================================
    # Static Methods (Main API)
    # =========================================================================
    
    @staticmethod
    def load(
        pairs: Union[str, List[str]],
        timeframes: List[str] = None,
        start: str = None,
        end: str = None,
        align: bool = True,
        handle_gaps: bool = True,
        config: DataLoaderConfig = None,
    ) -> MultiTimeframeData:
        """
        Load OHLCV data for multiple pairs and timeframes.
        
        Args:
            pairs: Single pair or list of pairs ['XBTUSD', 'ETHUSD', ...]
            timeframes: List of timeframes (default: ['1h', '12h', '24h', '72h', '168h'])
            start: Start date (ISO format, e.g., '2020-01-01')
            end: End date (ISO format)
            align: If True, align all timeframes to base (default: True)
            handle_gaps: If True, detect and impute gaps (default: True)
            config: Custom configuration (optional)
        
        Returns:
            MultiTimeframeData container with all loaded data
        
        Raises:
            ValueError: If no valid data found
        
        Example:
            >>> data = DataLoader.load(
            ...     pairs=['XBTUSD', 'ETHUSD'],
            ...     start='2020-01-01',
            ...     end='2024-12-31',
            ... )
            >>> data.info()
        """
        loader = DataLoader(config)
        return loader._load(
            pairs=pairs,
            timeframes=timeframes,
            start=start,
            end=end,
            align=align,
            handle_gaps=handle_gaps,
        )
    
    @staticmethod
    def load_detailed(
        pairs: Union[str, List[str]],
        timeframes: List[str] = None,
        start: str = None,
        end: str = None,
        config: DataLoaderConfig = None,
    ) -> LoadResult:
        """
        Load data with detailed result information.
        
        Same as load() but returns LoadResult with diagnostics.
        
        Example:
            >>> result = DataLoader.load_detailed(['XBTUSD', 'ETHUSD'])
            >>> print(result.summary())
            >>> if result.success:
            ...     data = result.data
        """
        loader = DataLoader(config)
        return loader._load_detailed(
            pairs=pairs,
            timeframes=timeframes,
            start=start,
            end=end,
        )
    
    @staticmethod
    def available_pairs(connection_url: str = None) -> List[str]:
        """
        List trading pairs available in database.
        
        Args:
            connection_url: Database connection (uses env var if None)
        
        Returns:
            List of pair symbols
        
        Example:
            >>> pairs = DataLoader.available_pairs()
            >>> print(pairs)
            ['XBTUSD', 'ETHUSD', 'LTCUSD', ...]
        """
        db = Database(connection_url)
        return db.get_available_pairs()
    
    @staticmethod
    def date_range(
        pair: str,
        connection_url: str = None,
    ) -> Tuple[datetime, datetime]:
        """
        Get available date range for a pair.
        
        Args:
            pair: Trading pair symbol
            connection_url: Database connection
        
        Returns:
            Tuple of (start_date, end_date)
        
        Example:
            >>> start, end = DataLoader.date_range('XBTUSD')
            >>> print(f"Data from {start} to {end}")
        """
        db = Database(connection_url)
        range_info = db.get_data_range(pair)
        return (range_info['start'], range_info['end'])
    
    @staticmethod
    def find_common_range(
        pairs: List[str],
        connection_url: str = None,
    ) -> Tuple[datetime, datetime]:
        """
        Find overlapping date range across multiple pairs.
        
        Args:
            pairs: List of trading pairs
            connection_url: Database connection
        
        Returns:
            Tuple of (common_start, common_end)
        
        Raises:
            ValueError: If no overlapping range exists
        
        Example:
            >>> start, end = DataLoader.find_common_range(['XBTUSD', 'ETHUSD'])
        """
        db = Database(connection_url)
        
        ranges = []
        for pair in pairs:
            try:
                range_info = db.get_data_range(pair)
                if range_info['start'] and range_info['end']:
                    ranges.append((range_info['start'], range_info['end']))
            except Exception as e:
                logger.warning(f"Could not get range for {pair}: {e}")
        
        if not ranges:
            raise ValueError("No valid date ranges found")
        
        # Find intersection
        common_start = max(r[0] for r in ranges)
        common_end = min(r[1] for r in ranges)
        
        if common_start >= common_end:
            raise ValueError(
                f"No overlapping date range. "
                f"Ranges end before they start: {common_start} >= {common_end}"
            )
        
        return (common_start, common_end)
    
    @staticmethod
    def info(connection_url: str = None) -> None:
        """
        Print information about available data.
        
        Example:
            >>> DataLoader.info()
        """
        db = Database(connection_url)
        pairs = db.get_available_pairs()
        
        print("=" * 50)
        print("DataLoader - Available Data")
        print("=" * 50)
        print(f"Pairs: {len(pairs)}")
        
        for pair in pairs[:10]:
            try:
                range_info = db.get_data_range(pair)
                start, end = range_info['start'], range_info['end']
                days = (end - start).days
                print(f"  {pair}: {start.date()} → {end.date()} ({days} days)")
            except Exception:
                print(f"  {pair}: (error getting range)")
        
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more")
        
        print("=" * 50)
    
    # =========================================================================
    # Instance Methods (Internal)
    # =========================================================================
    
    def _load(
        self,
        pairs: Union[str, List[str]],
        timeframes: List[str] = None,
        start: str = None,
        end: str = None,
        align: bool = True,
        handle_gaps: bool = True,
    ) -> MultiTimeframeData:
        """Internal load implementation."""
        
        result = self._load_detailed(
            pairs=pairs,
            timeframes=timeframes,
            start=start,
            end=end,
            align=align,
            handle_gaps=handle_gaps,
        )
        
        if not result.success or result.data is None:
            errors = list(result.exclusion_reasons.values())
            raise ValueError(f"Failed to load data: {errors}")
        
        return result.data
    
    def _load_detailed(
        self,
        pairs: Union[str, List[str]],
        timeframes: List[str] = None,
        start: str = None,
        end: str = None,
        align: bool = True,
        handle_gaps: bool = True,
    ) -> LoadResult:
        """Internal detailed load implementation."""
        
        import time
        start_time = time.time()
        
        # Normalize inputs
        if isinstance(pairs, str):
            pairs = [pairs]
        
        if timeframes is None:
            timeframes = self.config.default_timeframes
        
        base_tf = self.config.base_timeframe
        if base_tf not in timeframes:
            timeframes = [base_tf] + list(timeframes)
        
        # Track results
        pairs_requested = len(pairs)
        pairs_excluded = []
        exclusion_reasons = {}
        warnings = []
        
        # Storage for loaded data
        raw_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        aligned_data: Dict[str, pd.DataFrame] = {}
        quality_data: Dict[str, pd.Series] = {}
        
        # Determine date range
        data_start = None
        data_end = None
        
        # Load each pair
        for pair in pairs:
            try:
                result = self._load_single_pair(
                    pair=pair,
                    timeframes=timeframes,
                    start=start,
                    end=end,
                    handle_gaps=handle_gaps,
                    align=align,
                )
                
                if result is None:
                    pairs_excluded.append(pair)
                    exclusion_reasons[pair] = "No data returned"
                    continue
                
                pair_raw, pair_aligned, pair_quality, pair_warnings = result
                
                # Check minimum bars
                if base_tf in pair_raw:
                    n_bars = len(pair_raw[base_tf])
                    if n_bars < self.config.min_bars:
                        pairs_excluded.append(pair)
                        exclusion_reasons[pair] = f"Insufficient bars: {n_bars} < {self.config.min_bars}"
                        continue
                
                raw_data[pair] = pair_raw
                if pair_aligned is not None:
                    aligned_data[pair] = pair_aligned
                if pair_quality is not None:
                    quality_data[pair] = pair_quality
                
                warnings.extend(pair_warnings)
                
                # Track date range
                if base_tf in pair_raw:
                    df = pair_raw[base_tf]
                    if data_start is None or df.index.min() > data_start:
                        data_start = df.index.min()
                    if data_end is None or df.index.max() < data_end:
                        data_end = df.index.max()
                
            except Exception as e:
                logger.error(f"Error loading {pair}: {e}")
                pairs_excluded.append(pair)
                exclusion_reasons[pair] = str(e)
        
        # Check if any data loaded
        pairs_loaded = list(raw_data.keys())
        
        if not pairs_loaded:
            return LoadResult(
                success=False,
                data=None,
                pairs_requested=pairs_requested,
                pairs_loaded=0,
                pairs_excluded=pairs_excluded,
                exclusion_reasons=exclusion_reasons,
                load_time_seconds=time.time() - start_time,
                warnings=warnings,
            )
        
        # Create container
        container = create_container(
            pairs=pairs_loaded,
            timeframes=timeframes,
            base_tf=base_tf,
            start=data_start,
            end=data_end,
        )
        
        # Populate container
        for pair in pairs_loaded:
            for tf, df in raw_data[pair].items():
                container._set_raw(pair, tf, df)
            
            if pair in aligned_data:
                container._set_aligned(pair, aligned_data[pair])
            
            if pair in quality_data:
                container._set_quality(pair, quality_data[pair])
        
        return LoadResult(
            success=True,
            data=container,
            pairs_requested=pairs_requested,
            pairs_loaded=len(pairs_loaded),
            pairs_excluded=pairs_excluded,
            exclusion_reasons=exclusion_reasons,
            load_time_seconds=time.time() - start_time,
            warnings=warnings,
        )
    
    def _load_single_pair(
        self,
        pair: str,
        timeframes: List[str],
        start: str = None,
        end: str = None,
        handle_gaps: bool = True,
        align: bool = True,
    ) -> Optional[Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.Series, List[str]]]:
        """
        Load data for a single pair.
        
        Returns:
            Tuple of (raw_data, aligned_data, quality, warnings)
            or None if loading failed
        """
        warnings = []
        base_tf = self.config.base_timeframe
        
        # Step 1: Load 1h data from database
        logger.info(f"Loading {pair} from database...")
        df_1h = self.db.get_ohlcv(pair, start=start, end=end)
        
        if df_1h is None or len(df_1h) == 0:
            logger.warning(f"No data found for {pair}")
            return None
        
        # Ensure datetime index
        if 'timestamp' in df_1h.columns:
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
            df_1h = df_1h.set_index('timestamp')
        
        # Sort by index
        df_1h = df_1h.sort_index()
        
        logger.info(f"  Loaded {len(df_1h):,} rows for {pair}")
        
        # Step 2: Handle gaps
        quality = None
        if handle_gaps:
            gap_result = GapHandler.process(
                df_1h,
                impute_max=self.config.gap_impute_max,
                segment_threshold=self.config.gap_segment_threshold,
            )
            
            if gap_result.has_fatal_gaps:
                warnings.append(
                    f"{pair}: {gap_result.gaps_fatal} fatal gaps, "
                    f"using largest segment ({gap_result.get_largest_segment().row_count} rows)"
                )
                # Use largest segment
                segment = gap_result.get_largest_segment()
                df_1h = segment.df
            elif gap_result.bars_imputed > 0:
                # Use the imputed data from first segment
                df_1h = gap_result.segments[0].df
                warnings.append(
                    f"{pair}: Imputed {gap_result.bars_imputed} bars "
                    f"({gap_result.impute_rate:.1%})"
                )
            
            # Create quality series
            quality = pd.Series('original', index=df_1h.index)
            # Note: detailed quality tracking would require keeping it through gap processing
        
        # Step 3: Resample to higher timeframes
        logger.info(f"  Resampling {pair} to {timeframes}...")
        
        higher_tfs = [tf for tf in timeframes if tf != base_tf]
        resampled = Resampler.resample_all(
            df_1h,
            timeframes=higher_tfs,
            include_source=True,
        )
        
        # Step 4: Align if requested
        aligned = None
        if align:
            logger.info(f"  Aligning {pair} timeframes...")
            aligned = Aligner.align_all(
                data=resampled,
                base_tf=base_tf,
            )
        
        return (resampled, aligned, quality, warnings)


# =============================================================================
# Convenience Functions
# =============================================================================

def load(
    pairs: Union[str, List[str]],
    timeframes: List[str] = None,
    start: str = None,
    end: str = None,
) -> MultiTimeframeData:
    """Convenience function for loading data."""
    return DataLoader.load(pairs, timeframes, start, end)


def available_pairs() -> List[str]:
    """Convenience function to list available pairs."""
    return DataLoader.available_pairs()