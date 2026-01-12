# DataLoader Design Document

**Version:** 1.0  
**Date:** January 2025  
**Status:** Draft for Review

---

## 1. Purpose

Load multi-asset, multi-timeframe OHLCV data from TimescaleDB for backtesting. All higher timeframes (12h, 24h, 72h, 168h) are resampled from 1h data to ensure consistency.

---

## 2. Requirements

| Requirement | Description |
|-------------|-------------|
| Multi-pair | Load 12+ pairs simultaneously |
| Multi-timeframe | 1h, 12h, 24h, 72h, 168h |
| No look-ahead | Higher TF aligned with proper lag |
| Gap handling | Impute small gaps, flag large ones |
| Date alignment | Common date range across pairs |
| Validation | Report data quality issues |

---

## 3. Timeframe Specification

| Timeframe | Minutes | Source | Resample Rule |
|-----------|---------|--------|---------------|
| 1h | 60 | Database | Native |
| 12h | 720 | Resample | 12 × 1h bars |
| 24h | 1440 | Resample | 24 × 1h bars |
| 72h | 4320 | Resample | 72 × 1h bars |
| 168h | 10080 | Resample | 168 × 1h bars |

---

## 4. Class Design

### 4.1 DataLoader (Main Entry Point)

```python
class DataLoader:
    """
    Main entry point for loading multi-asset multi-timeframe data.
    """
    
    @staticmethod
    def load(
        pairs: List[str],
        timeframes: List[str] = ['1h', '12h', '24h', '72h', '168h'],
        start: str = None,
        end: str = None,
        align_to: str = '1h',
        gap_policy: str = 'impute_small',
        require_complete: bool = True,
    ) -> 'MultiTimeframeData':
        """
        Load data for multiple pairs and timeframes.
        
        Args:
            pairs: List of trading pairs ['XBTUSD', 'ETHUSD', ...]
            timeframes: List of timeframes to load
            start: Start date (ISO format)
            end: End date (ISO format)
            align_to: Base timeframe for alignment
            gap_policy: How to handle gaps ('impute_small', 'strict', 'flag_only')
            require_complete: Only include pairs with full date coverage
        
        Returns:
            MultiTimeframeData container
        """
        pass
    
    @staticmethod
    def available_pairs() -> List[str]:
        """List pairs available in database."""
        pass
    
    @staticmethod
    def date_range(pair: str) -> Tuple[datetime, datetime]:
        """Get available date range for a pair."""
        pass
    
    @staticmethod
    def find_common_range(pairs: List[str]) -> Tuple[datetime, datetime]:
        """Find overlapping date range across pairs."""
        pass
```

---

### 4.2 MultiTimeframeData (Container)

```python
@dataclass
class MultiTimeframeData:
    """
    Container for multi-asset multi-timeframe data.
    
    Holds loaded data and provides access methods.
    """
    
    pairs: List[str]
    timeframes: List[str]
    base_tf: str
    start: datetime
    end: datetime
    
    # Internal storage
    _raw: Dict[str, Dict[str, pd.DataFrame]]  # pair -> tf -> DataFrame
    _aligned: Dict[str, pd.DataFrame]          # pair -> aligned DataFrame
    _quality: Dict[str, pd.DataFrame]          # pair -> quality flags
    
    def get(self, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Get raw OHLCV data for one pair/timeframe.
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: timestamp
        """
        pass
    
    def get_aligned(self, pair: str) -> pd.DataFrame:
        """
        Get all timeframes aligned to base for one pair.
        
        Returns:
            DataFrame with columns:
                open_1h, high_1h, low_1h, close_1h, volume_1h,
                open_12h, high_12h, low_12h, close_12h, volume_12h,
                ... (for each timeframe)
            Index: base timeframe timestamps
        """
        pass
    
    def get_close_prices(self, pair: str) -> pd.DataFrame:
        """
        Get just close prices across all timeframes.
        
        Returns:
            DataFrame with columns: close_1h, close_12h, close_24h, ...
        """
        pass
    
    def get_all_pairs_aligned(self) -> Dict[str, pd.DataFrame]:
        """Get aligned data for all pairs."""
        pass
    
    def coverage(self) -> pd.DataFrame:
        """
        Report data coverage for each pair.
        
        Returns:
            DataFrame with columns: pair, start, end, bars, gaps, complete
        """
        pass
    
    def quality_report(self) -> pd.DataFrame:
        """
        Detailed quality report.
        
        Returns:
            DataFrame with gap locations, imputed regions, flags
        """
        pass
    
    def info(self) -> None:
        """Print summary of loaded data."""
        pass
```

---

### 4.3 DatabaseReader (Internal)

```python
class DatabaseReader:
    """
    Read OHLCV data from TimescaleDB.
    """
    
    def __init__(self, connection_url: str = None):
        pass
    
    def read(
        self,
        pair: str,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Read 1h OHLCV for a pair.
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: timestamp (UTC)
        """
        pass
    
    def read_multiple(
        self,
        pairs: List[str],
        start: str = None,
        end: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """Read 1h data for multiple pairs."""
        pass
```

---

### 4.4 Resampler (Internal)

```python
class Resampler:
    """
    Resample 1h data to higher timeframes.
    
    OHLCV resampling rules:
        - Open: first value
        - High: max value
        - Low: min value
        - Close: last value
        - Volume: sum
    """
    
    TIMEFRAME_MINUTES = {
        '1h': 60,
        '12h': 720,
        '24h': 1440,
        '72h': 4320,
        '168h': 10080,
    }
    
    @staticmethod
    def resample(
        df: pd.DataFrame,
        target_tf: str,
    ) -> pd.DataFrame:
        """
        Resample 1h data to target timeframe.
        
        Args:
            df: 1h OHLCV data
            target_tf: Target timeframe ('12h', '24h', '72h', '168h')
        
        Returns:
            Resampled DataFrame
        """
        pass
    
    @staticmethod
    def resample_all(
        df_1h: pd.DataFrame,
        timeframes: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Resample to multiple timeframes."""
        pass
```

---

### 4.5 GapHandler (Internal)

```python
@dataclass
class GapInfo:
    """Information about a gap in data."""
    start: datetime
    end: datetime
    bars_missing: int
    severity: str  # 'small', 'medium', 'large', 'fatal'

class GapHandler:
    """
    Detect and handle gaps in OHLCV data.
    """
    
    # Gap thresholds (in bars)
    SMALL = 5       # Impute silently
    MEDIUM = 24     # Impute but flag
    LARGE = 168     # Split into segments
    FATAL = 720     # Exclude period
    
    @staticmethod
    def detect_gaps(
        df: pd.DataFrame,
        expected_freq: str = '1h',
    ) -> List[GapInfo]:
        """
        Detect gaps in timeseries.
        
        Returns:
            List of GapInfo objects
        """
        pass
    
    @staticmethod
    def impute(
        df: pd.DataFrame,
        gaps: List[GapInfo],
        max_gap: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Impute small gaps.
        
        Returns:
            (imputed_df, quality_flags)
            quality_flags: 1.0 = original, 0.5 = imputed
        """
        pass
    
    @staticmethod
    def create_quality_mask(
        df: pd.DataFrame,
        gaps: List[GapInfo],
    ) -> pd.Series:
        """Create quality flag series for each bar."""
        pass
```

---

### 4.6 Aligner (Internal)

```python
class Aligner:
    """
    Align higher timeframes to base timeframe.
    
    Critical for avoiding look-ahead bias:
    - At time T on base timeframe
    - Higher TF value must be from LAST COMPLETED bar
    - Not current (incomplete) bar
    
    Example (24h aligned to 1h):
        At 2024-01-15 14:00 (1h bar):
        - 24h value should be from 2024-01-14 00:00 bar (completed)
        - NOT from 2024-01-15 00:00 bar (still in progress)
    """
    
    @staticmethod
    def align(
        df_higher: pd.DataFrame,
        target_index: pd.DatetimeIndex,
        source_tf: str,
        target_tf: str,
    ) -> pd.DataFrame:
        """
        Align higher TF to target index with proper lag.
        
        Args:
            df_higher: Higher timeframe data
            target_index: Base timeframe index to align to
            source_tf: Timeframe of df_higher
            target_tf: Base timeframe
        
        Returns:
            DataFrame aligned to target_index (forward-filled)
        """
        pass
    
    @staticmethod
    def align_all(
        data: Dict[str, pd.DataFrame],
        base_tf: str,
    ) -> pd.DataFrame:
        """
        Align all timeframes to base.
        
        Returns:
            Single DataFrame with all TFs as columns
        """
        pass
```

---

## 5. Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DataLoader.load()                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. DatabaseReader.read_multiple()                            │
│    - Load 1h data for all pairs from TimescaleDB            │
│    - Output: Dict[pair, DataFrame_1h]                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GapHandler.detect_gaps()                                  │
│    - Find missing bars in each pair                          │
│    - Classify by severity                                    │
│    - Output: Dict[pair, List[GapInfo]]                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. GapHandler.impute()                                       │
│    - Fill small gaps (≤5 bars)                               │
│    - Create quality flags                                    │
│    - Output: Dict[pair, (DataFrame_1h, quality_flags)]       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Filter by date range                                      │
│    - Find common intersection if require_complete=True       │
│    - Or use specified start/end                              │
│    - Exclude pairs with fatal gaps                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Resampler.resample_all()                                  │
│    - Create 12h, 24h, 72h, 168h from 1h                      │
│    - For each pair                                           │
│    - Output: Dict[pair, Dict[tf, DataFrame]]                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Aligner.align_all()                                       │
│    - Align all TFs to 1h base                                │
│    - Apply proper lag (no look-ahead)                        │
│    - Output: Dict[pair, DataFrame_aligned]                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Create MultiTimeframeData                                 │
│    - Package all data                                        │
│    - Store raw and aligned versions                          │
│    - Include quality reports                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Alignment Logic (Critical)

### The Look-Ahead Problem

```
Time:     00:00  06:00  12:00  18:00  00:00  06:00  12:00
          |------|------|------|------|------|------|
1h bars:  [  1  ][  2  ][  3  ][  4  ][  5  ][  6  ][  7  ]
24h bar:  [          Day 1           ][     Day 2    ...

At 1h bar 5 (Day 2, 00:00):
  - Day 1 bar is COMPLETE → can use
  - Day 2 bar is INCOMPLETE → cannot use

At 1h bar 6 (Day 2, 06:00):
  - Still only Day 1 is complete
  - Day 2 won't complete until bar 9
```

### Implementation

```python
def align(df_higher, target_index, source_tf_minutes, base_tf_minutes):
    """
    Align by shifting higher TF to its completion time.
    
    A 24h bar timestamped at 00:00 completes at 23:59.
    At 1h bar 06:00, we can use 24h bar from previous day.
    """
    # Shift higher TF index to completion time
    completion_offset = pd.Timedelta(minutes=source_tf_minutes)
    df_higher_completed = df_higher.copy()
    df_higher_completed.index = df_higher_completed.index + completion_offset
    
    # Reindex to target and forward-fill
    aligned = df_higher_completed.reindex(target_index, method='ffill')
    
    return aligned
```

---

## 7. Column Naming Convention

### Raw DataFrames
```
open, high, low, close, volume
```

### Aligned DataFrame
```
open_1h, high_1h, low_1h, close_1h, volume_1h,
open_12h, high_12h, low_12h, close_12h, volume_12h,
open_24h, high_24h, low_24h, close_24h, volume_24h,
open_72h, high_72h, low_72h, close_72h, volume_72h,
open_168h, high_168h, low_168h, close_168h, volume_168h,
data_quality
```

---

## 8. Usage Examples

### Basic Loading
```python
from cryptobot.data import DataLoader

# Load 12 pairs with all timeframes
data = DataLoader.load(
    pairs=['XBTUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'XLMUSD', 
           'XMRUSD', 'ADAUSD', 'ZECUSD', 'DASHUSD', 'BCHUSD',
           'EOSUSD', 'ETCUSD'],
    timeframes=['1h', '12h', '24h', '72h', '168h'],
    start='2019-01-01',
    end='2024-12-31',
)

# Check what loaded
data.info()
```

### Accessing Data
```python
# Raw 1h data for BTC
df_btc_1h = data.get('XBTUSD', '1h')

# All timeframes aligned for BTC (for feature computation)
df_btc_aligned = data.get_aligned('XBTUSD')

# Just close prices
df_btc_closes = data.get_close_prices('XBTUSD')

# All pairs aligned (for portfolio backtest)
all_aligned = data.get_all_pairs_aligned()
```

### Quality Checks
```python
# Coverage summary
print(data.coverage())

# Detailed quality report
print(data.quality_report())
```

---

## 9. Error Handling

| Error | Handling |
|-------|----------|
| Pair not in database | Raise `ValueError` with available pairs |
| Date range not available | Raise `ValueError` with actual range |
| No common date range | Raise `ValueError`, suggest reducing pairs |
| Fatal gaps (>720 bars) | Exclude pair, warn user |
| Database connection failed | Raise `ConnectionError` |

---

## 10. Configuration

```python
@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader."""
    
    # Database
    connection_url: str = "postgresql://..."
    
    # Gap handling
    gap_small: int = 5       # Impute silently
    gap_medium: int = 24     # Impute + flag  
    gap_large: int = 168     # Split segments
    gap_fatal: int = 720     # Exclude
    
    # Imputation
    impute_method: str = 'interpolate'  # 'ffill', 'interpolate'
    
    # Timeframes
    default_timeframes: List[str] = ['1h', '12h', '24h', '72h', '168h']
    base_timeframe: str = '1h'
```

---

## 11. File Structure

```
cryptobot/
└── data/
    ├── __init__.py
    ├── loader.py          # DataLoader class
    ├── container.py       # MultiTimeframeData class
    ├── reader.py          # DatabaseReader class
    ├── resampler.py       # Resampler class
    ├── gaps.py            # GapHandler, GapInfo
    ├── aligner.py         # Aligner class
    └── config.py          # DataLoaderConfig
```

---

## 12. Testing Requirements

| Test | Description |
|------|-------------|
| Alignment correctness | Verify no look-ahead at TF boundaries |
| Resampling accuracy | Compare to known good resampled data |
| Gap detection | Test with synthetic gaps |
| Imputation quality | Verify interpolation is reasonable |
| Coverage reporting | Test with ragged date ranges |
| Memory usage | Profile with 12 pairs × 6 years |

---

## 13. Open Questions

1. **Volume handling in resampling:** Sum is standard, but should we also compute VWAP?

2. **Trade count:** Database has trade count column. Include in output?

3. **Timezone:** Assume all UTC. Confirm this matches your database.

4. **Caching:** Should we cache loaded data to disk for faster subsequent loads?

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2025 | Initial design |

---

*End of Design Document*
