#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Strategy Backtest v3
=============================
Combines Directional 32-State and Pairs Regime Trading strategies.

Allocation Methods:
- Primary: Risk Parity (inverse volatility weighting)
- Option: Regime Adaptive (BTC 32-State regime - CONSISTENT with trading signals)

Key Fix in v3:
- Regime allocation now uses BTC's 32-state hit rate (same system as trading)
- Replaces inconsistent MA200 check with unified 32-state framework

Features:
- 32-state regime sensitivity testing
- Yearly performance breakdown
- Full attribution tracking

Usage:
    python 32state_combined_adaptive_pairs_backtest_v3.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked from validation)
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Hit rate parameters
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_STATE = 20

# Calendar alignment
MIN_TRAINING_MONTHS = 12
HIT_RATE_RECALC_MONTHS = 1
VOL_LOOKBACK_MONTHS = 1
COV_LOOKBACK_MONTHS = 2
REBALANCE_MONTHS = 1

# =============================================================================
# ALLOCATION METHOD SELECTION
# =============================================================================

# Primary: Risk Parity
# Set USE_REGIME_ADAPTIVE = True to use Regime Adaptive instead (lower drawdown)
USE_REGIME_ADAPTIVE = False

# Allocation parameters
ALLOCATION_MIN_FLOOR = 0.10
ALLOCATION_MAX_CEILING = 0.90
ALLOCATION_LOOKBACK_MONTHS = 3

# =============================================================================
# 32-STATE REGIME ALLOCATION THRESHOLDS
# =============================================================================
# Uses BTC's 32-state hit rate (same system as trading signals)
#
# BTC Hit Rate      State          Default Allocation (Dir/Pairs)
# ─────────────────────────────────────────────────────────────────
# ≥ 0.55            STRONG_BUY     70% / 30%
# ≥ 0.50            BUY            60% / 40%
# ≥ 0.45            SELL           50% / 50%
# < 0.45            STRONG_SELL    40% / 60%
# =============================================================================

STRONG_BUY_THRESHOLD = 0.55
BUY_THRESHOLD = 0.50
SELL_THRESHOLD = 0.45

# Default allocation per state (directional weight)
DEFAULT_ALLOC_STRONG_BUY = 0.70
DEFAULT_ALLOC_BUY = 0.60
DEFAULT_ALLOC_SELL = 0.50
DEFAULT_ALLOC_STRONG_SELL = 0.40

# =============================================================================
# REGIME SENSITIVITY TEST GRID (32-State Based)
# =============================================================================
# Each tuple: (STRONG_BUY, BUY, SELL, STRONG_SELL, Name)
# Values are directional weights; pairs weight = 1 - directional

REGIME_SENSITIVITY_GRID = [
    (0.80, 0.70, 0.50, 0.30, "Aggressive"),
    (0.70, 0.60, 0.50, 0.40, "Default"),
    (0.65, 0.55, 0.50, 0.45, "Moderate"),
    (0.60, 0.55, 0.50, 0.45, "Conservative"),
    (0.55, 0.52, 0.50, 0.48, "Minimal"),
]

# =============================================================================
# DIRECTIONAL STRATEGY EXPOSURE LIMITS
# =============================================================================

DIR_MAX_EXPOSURE = 2.0
DIR_TARGET_VOL = 0.40
DIR_DD_START_REDUCE = -0.20
DIR_DD_MIN_EXPOSURE = -0.50
DIR_MIN_EXPOSURE_FLOOR = 0.40

# =============================================================================
# PAIRS STRATEGY CONFIGURATION
# =============================================================================

PAIRS_MAX_EXPOSURE = 3.0
PAIRS_DD_START_REDUCE = -0.20
PAIRS_DD_MIN_EXPOSURE = -0.50
PAIRS_MIN_EXPOSURE_FLOOR = 0.40

# Divergence thresholds
ENTRY_DIVERGENCE = 2
EXIT_DIVERGENCE = 2
MAX_HOLD_DAYS = 10

# Position sizing
BASE_POSITION_SIZE = 0.10
MAX_POSITION_SIZE = 0.25
MIN_POSITION_SIZE = 0.02

# =============================================================================
# SHARED PARAMETERS
# =============================================================================

INITIAL_CAPITAL = 100000.0
TRADING_COST = 0.0015
MIN_TRADE_SIZE = 100.0

# Shared drawdown protection
SHARED_DD_START_REDUCE = -0.20
SHARED_DD_MIN_EXPOSURE = -0.50
SHARED_MIN_EXPOSURE_FLOOR = 0.40


# =============================================================================
# CALENDAR MONTH HELPER FUNCTIONS
# =============================================================================

def get_month_start(date: pd.Timestamp, months_back: int = 0) -> pd.Timestamp:
    """Get the 1st of the month, optionally N months back."""
    target = date - pd.DateOffset(months=months_back)
    return target.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def is_first_of_month(date: pd.Timestamp) -> bool:
    """Check if date is the first day of a month."""
    return date.day == 1


def get_months_of_data(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Calculate full calendar months between two dates."""
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def has_minimum_training(current_date: pd.Timestamp, data_start: pd.Timestamp, 
                          min_months: int) -> bool:
    """Check if we have minimum training months of data."""
    months_available = get_months_of_data(data_start, current_date)
    return months_available >= min_months


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SpreadTrade:
    """A pairs spread trade."""
    entry_date: pd.Timestamp
    long_asset: str
    short_asset: str
    entry_state_long: str
    entry_state_short: str
    entry_hit_rate_long: float
    entry_hit_rate_short: float
    position_size: float
    entry_capital: float
    exit_date: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    spread_return: Optional[float] = None
    holding_days: Optional[int] = None
    pnl: Optional[float] = None
    cum_long_return: float = 0.0
    cum_short_return: float = 0.0
    
    @property
    def pair_key(self) -> Tuple[str, str]:
        return tuple(sorted([self.long_asset, self.short_asset]))
    
    @property
    def current_spread_return(self) -> float:
        return self.cum_long_return - self.cum_short_return
    
    def get_current_value(self) -> float:
        return self.entry_capital * self.position_size * (1 + self.current_spread_return)


@dataclass
class YearlyStats:
    """Yearly performance statistics."""
    year: int
    return_pct: float
    dir_pnl: float
    pairs_pnl: float
    dir_contribution: float
    pairs_contribution: float
    max_drawdown: float
    start_equity: float
    end_equity: float


@dataclass
class CombinedBacktestResult:
    """Results from combined backtest."""
    equity_curve: pd.Series
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    realized_vol: float
    
    # Attribution
    directional_pnl: float
    pairs_pnl: float
    directional_contribution: float
    pairs_contribution: float
    
    # Allocation tracking
    avg_directional_weight: float
    avg_pairs_weight: float
    allocation_method: str
    
    # Yearly breakdown
    yearly_stats: List[YearlyStats]
    
    # BTC regime tracking
    btc_state_counts: Optional[Dict[str, int]] = None
    
    # Pairs stats
    n_pairs_trades: int = 0
    pairs_win_rate: float = 0.0
    pairs_avg_return: float = 0.0
    pairs_avg_hold_days: float = 0.0
    pairs_trades: List[SpreadTrade] = None
    pairs_entries_blocked: int = 0
    
    # Costs
    total_trading_costs: float = 0.0
    n_days: int = 0


# =============================================================================
# SIGNAL GENERATION (SHARED)
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def label_trend_binary(df: pd.DataFrame, ma_period: int,
                       entry_buffer: float, exit_buffer: float) -> pd.Series:
    """Binary trend detection with hysteresis."""
    close = df['close']
    ma = close.rolling(ma_period).mean()
    
    labels = pd.Series(index=df.index, dtype=int)
    current = 1
    
    for i in range(len(df)):
        if pd.isna(ma.iloc[i]):
            labels.iloc[i] = current
            continue
        
        price = close.iloc[i]
        
        if current == 1:
            if price < ma.iloc[i] * (1 - exit_buffer) and price < ma.iloc[i] * (1 - entry_buffer):
                current = 0
        else:
            if price > ma.iloc[i] * (1 + exit_buffer) and price > ma.iloc[i] * (1 + entry_buffer):
                current = 1
        
        labels.iloc[i] = current
    
    return labels


def generate_32state_signals(df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                              df_168h: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 32-state signals.
    8 price states × 4 MA alignment states = 32 total states
    SHIFT BEFORE REINDEX to prevent look-ahead bias.
    """
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


def calculate_expanding_hit_rates(returns_history: pd.Series, 
                                   signals_history: pd.DataFrame) -> Dict:
    """Calculate 32-state hit rates using ONLY historical data."""
    all_price_perms = list(product([0, 1], repeat=3))
    all_ma_perms = list(product([0, 1], repeat=2))
    
    if len(returns_history) < MIN_SAMPLES_PER_STATE:
        return {(p, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for p in all_price_perms for m in all_ma_perms}
    
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < MIN_SAMPLES_PER_STATE:
        return {(p, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for p in all_price_perms for m in all_ma_perms}
    
    aligned_returns = returns_history.loc[common_idx]
    aligned_signals = signals_history.loc[common_idx]
    
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    hit_rates = {}
    
    for price_perm in all_price_perms:
        for ma_perm in all_ma_perms:
            mask = (
                (aligned_signals['trend_24h'] == price_perm[0]) &
                (aligned_signals['trend_72h'] == price_perm[1]) &
                (aligned_signals['trend_168h'] == price_perm[2]) &
                (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
                (aligned_signals['ma168_above_ma24'] == ma_perm[1])
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            
            if n > 0:
                hit_rate = (perm_returns > 0).sum() / n
            else:
                hit_rate = 0.5
            
            hit_rates[(price_perm, ma_perm)] = {
                'n': n,
                'hit_rate': hit_rate,
                'sufficient': n >= MIN_SAMPLES_PER_STATE,
            }
    
    return hit_rates


# =============================================================================
# DIRECTIONAL STRATEGY FUNCTIONS
# =============================================================================

def get_32state_position(price_perm: Tuple[int, int, int], 
                         ma_perm: Tuple[int, int],
                         hit_rates: Dict) -> float:
    """
    Get position signal for 32-state.
    Returns:
        1.00: INVEST (hit rate > threshold and sufficient samples)
        0.50: SKIP (insufficient samples)
        0.00: AVOID (hit rate <= threshold)
    """
    key = (price_perm, ma_perm)
    data = hit_rates.get(key, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.00


def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Calculate risk parity weights for assets (inverse volatility)."""
    vols = returns_df.std() * np.sqrt(365)
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return weights


# =============================================================================
# PAIRS STRATEGY FUNCTIONS
# =============================================================================

def hit_rate_to_simple_state(hit_rate: float) -> str:
    """Convert hit rate to simplified state."""
    if hit_rate >= STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    elif hit_rate >= BUY_THRESHOLD:
        return "BUY"
    elif hit_rate >= SELL_THRESHOLD:
        return "SELL"
    else:
        return "STRONG_SELL"


def state_to_numeric(state: str) -> int:
    """Convert state to numeric for divergence calculation."""
    mapping = {"STRONG_BUY": 3, "BUY": 2, "SELL": 1, "STRONG_SELL": 0}
    return mapping.get(state, 1)


def get_state_divergence(state_a: str, state_b: str) -> int:
    """Calculate state divergence between two assets."""
    return abs(state_to_numeric(state_a) - state_to_numeric(state_b))


class SpreadStatsTracker:
    """Track spread statistics for inverse volatility sizing."""
    
    def __init__(self, data: Dict):
        self.data = data
        self.pairs = list(combinations(DEPLOY_PAIRS, 2))
        
        self.spread_returns = {}
        for a, b in self.pairs:
            key = tuple(sorted([a, b]))
            ret_a = data[a]['returns']
            ret_b = data[b]['returns']
            common_idx = ret_a.index.intersection(ret_b.index)
            self.spread_returns[key] = (ret_a.loc[common_idx] - ret_b.loc[common_idx]).dropna()
        
        self._vol_cache = {}
    
    def get_spread_volatility(self, pair_key: Tuple[str, str], as_of_date: pd.Timestamp) -> float:
        """Get annualized spread volatility using expanding window."""
        cache_key = (pair_key, as_of_date)
        if cache_key in self._vol_cache:
            return self._vol_cache[cache_key]
        
        spread_ret = self.spread_returns.get(pair_key)
        if spread_ret is None:
            return 0.20
        
        hist = spread_ret[spread_ret.index < as_of_date]
        
        if len(hist) < 30:
            return 0.20
        
        daily_vol = hist.std()
        annual_vol = daily_vol * np.sqrt(365)
        
        self._vol_cache[cache_key] = annual_vol
        return annual_vol
    
    def get_average_spread_volatility(self, as_of_date: pd.Timestamp) -> float:
        """Get average volatility across all pairs."""
        vols = [self.get_spread_volatility(key, as_of_date) for key in self.spread_returns.keys()]
        return np.mean(vols) if vols else 0.20
    
    def get_inverse_vol_size(self, pair_key: Tuple[str, str], as_of_date: pd.Timestamp) -> float:
        """Get position size using inverse volatility."""
        pair_vol = self.get_spread_volatility(pair_key, as_of_date)
        avg_vol = self.get_average_spread_volatility(as_of_date)
        
        if pair_vol <= 0:
            return BASE_POSITION_SIZE
        
        scale = avg_vol / pair_vol
        size = BASE_POSITION_SIZE * scale
        return max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, size))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database) -> Dict:
    """Load and prepare all data for trading pairs."""
    data = {}
    
    print("\n  Loading data...", flush=True)
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        signals = generate_32state_signals(df_24h, df_72h, df_168h)
        returns = df_24h['close'].pct_change()
        
        data[pair] = {
            'prices': df_24h,
            'returns': returns,
            'signals': signals,
        }
        
        print(f"{len(df_24h)} days", flush=True)
    
    print("  Data loading complete.", flush=True)
    return data


def load_btc_data(db: Database) -> Dict:
    """Load BTC data with 32-state signals for regime detection."""
    print("  Loading BTC data for 32-state regime detection...", flush=True)
    
    btc_1h = db.get_ohlcv('XBTUSD')
    btc_24h = resample_ohlcv(btc_1h, '24h')
    btc_72h = resample_ohlcv(btc_1h, '72h')
    btc_168h = resample_ohlcv(btc_1h, '168h')
    
    signals = generate_32state_signals(btc_24h, btc_72h, btc_168h)
    returns = btc_24h['close'].pct_change()
    
    print(f"    XBTUSD... {len(btc_24h)} days", flush=True)
    
    return {
        'prices': btc_24h,
        'returns': returns,
        'signals': signals,
    }


# =============================================================================
# POSITION HISTORY CALCULATION
# =============================================================================

def calculate_directional_positions(data: Dict, dates: List, data_start: pd.Timestamp) -> Dict:
    """Calculate directional position signals for all assets over time."""
    positions = {pair: {} for pair in DEPLOY_PAIRS}
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}
    
    for date in dates:
        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            
            if date not in signals.index:
                continue
            
            sig = signals.loc[date]
            price_perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
            ma_perm = (int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24']))
            
            if not has_minimum_training(date, data_start, MIN_TRAINING_MONTHS):
                positions[pair][date] = 0.50
                continue
            
            current_month = (date.year, date.month)
            if last_recalc_month[pair] != current_month:
                cutoff_date = get_month_start(date, 0)
                hist_returns = returns[returns.index < cutoff_date]
                hist_signals = signals[signals.index < cutoff_date]
                
                if len(hist_returns) > 0:
                    hit_rate_cache[pair] = calculate_expanding_hit_rates(hist_returns, hist_signals)
                    last_recalc_month[pair] = current_month
            
            pos = get_32state_position(price_perm, ma_perm, hit_rate_cache[pair])
            positions[pair][date] = pos
    
    return positions


def calculate_pairs_states(data: Dict, dates: List, data_start: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate state and hit rate history for pairs strategy."""
    state_history = {pair: {} for pair in DEPLOY_PAIRS}
    hitrate_history = {pair: {} for pair in DEPLOY_PAIRS}
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}
    
    for date in dates:
        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            
            if date not in signals.index:
                continue
            
            sig = signals.loc[date]
            price_perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
            ma_perm = (int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24']))
            
            if not has_minimum_training(date, data_start, MIN_TRAINING_MONTHS):
                state_history[pair][date] = "NEUTRAL"
                hitrate_history[pair][date] = 0.50
                continue
            
            current_month = (date.year, date.month)
            if last_recalc_month[pair] != current_month:
                cutoff_date = get_month_start(date, 0)
                hist_returns = returns[returns.index < cutoff_date]
                hist_signals = signals[signals.index < cutoff_date]
                
                if len(hist_returns) > 0:
                    hit_rate_cache[pair] = calculate_expanding_hit_rates(hist_returns, hist_signals)
                    last_recalc_month[pair] = current_month
            
            key = (price_perm, ma_perm)
            hr_data = hit_rate_cache[pair].get(key, {'hit_rate': 0.5})
            hit_rate = hr_data['hit_rate']
            
            state_history[pair][date] = hit_rate_to_simple_state(hit_rate)
            hitrate_history[pair][date] = hit_rate
    
    state_df = pd.DataFrame(state_history)
    state_df.index = pd.to_datetime(state_df.index)
    
    hitrate_df = pd.DataFrame(hitrate_history)
    hitrate_df.index = pd.to_datetime(hitrate_df.index)
    
    return state_df.dropna(), hitrate_df.dropna()


def calculate_btc_hit_rate_history(btc_data: Dict, dates: List, data_start: pd.Timestamp) -> pd.Series:
    """Calculate BTC's 32-state hit rate history for regime allocation."""
    hitrate_history = {}
    hit_rate_cache = {}
    last_recalc_month = None
    
    signals = btc_data['signals']
    returns = btc_data['returns']
    
    for date in dates:
        if date not in signals.index:
            continue
        
        sig = signals.loc[date]
        price_perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
        ma_perm = (int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24']))
        
        if not has_minimum_training(date, data_start, MIN_TRAINING_MONTHS):
            hitrate_history[date] = 0.50
            continue
        
        current_month = (date.year, date.month)
        if last_recalc_month != current_month:
            cutoff_date = get_month_start(date, 0)
            hist_returns = returns[returns.index < cutoff_date]
            hist_signals = signals[signals.index < cutoff_date]
            
            if len(hist_returns) > 0:
                hit_rate_cache = calculate_expanding_hit_rates(hist_returns, hist_signals)
                last_recalc_month = current_month
        
        key = (price_perm, ma_perm)
        hr_data = hit_rate_cache.get(key, {'hit_rate': 0.5})
        hitrate_history[date] = hr_data['hit_rate']
    
    return pd.Series(hitrate_history)


# =============================================================================
# ALLOCATION METHODS
# =============================================================================

def calculate_allocation_risk_parity(dir_returns: pd.Series, pairs_returns: pd.Series,
                                      lookback_start: pd.Timestamp, lookback_end: pd.Timestamp) -> Tuple[float, float]:
    """Risk parity allocation - inverse volatility weighting."""
    if len(dir_returns) == 0 or len(pairs_returns) == 0:
        return 0.50, 0.50
    
    # Ensure datetime index
    if not isinstance(dir_returns.index, pd.DatetimeIndex):
        dir_returns.index = pd.to_datetime(dir_returns.index)
    if not isinstance(pairs_returns.index, pd.DatetimeIndex):
        pairs_returns.index = pd.to_datetime(pairs_returns.index)
    
    dir_hist = dir_returns[(dir_returns.index >= lookback_start) & (dir_returns.index < lookback_end)]
    pairs_hist = pairs_returns[(pairs_returns.index >= lookback_start) & (pairs_returns.index < lookback_end)]
    
    if len(dir_hist) < 20 or len(pairs_hist) < 20:
        return 0.50, 0.50
    
    dir_vol = dir_hist.std() * np.sqrt(365)
    pairs_vol = pairs_hist.std() * np.sqrt(365)
    
    if dir_vol <= 0 or pairs_vol <= 0:
        return 0.50, 0.50
    
    inv_dir = 1.0 / dir_vol
    inv_pairs = 1.0 / pairs_vol
    
    w_dir = inv_dir / (inv_dir + inv_pairs)
    w_pairs = inv_pairs / (inv_dir + inv_pairs)
    
    w_dir = max(ALLOCATION_MIN_FLOOR, min(ALLOCATION_MAX_CEILING, w_dir))
    w_pairs = 1.0 - w_dir
    
    return w_dir, w_pairs


def calculate_allocation_regime_32state(btc_hit_rate: float,
                                         alloc_strong_buy: float,
                                         alloc_buy: float,
                                         alloc_sell: float,
                                         alloc_strong_sell: float) -> Tuple[float, float, str]:
    """
    Regime-adaptive allocation based on BTC's 32-state hit rate.
    Uses the SAME 32-state system as trading signals for consistency.
    
    Returns: (directional_weight, pairs_weight, btc_state)
    """
    if btc_hit_rate >= STRONG_BUY_THRESHOLD:
        state = "STRONG_BUY"
        w_dir = alloc_strong_buy
    elif btc_hit_rate >= BUY_THRESHOLD:
        state = "BUY"
        w_dir = alloc_buy
    elif btc_hit_rate >= SELL_THRESHOLD:
        state = "SELL"
        w_dir = alloc_sell
    else:
        state = "STRONG_SELL"
        w_dir = alloc_strong_sell
    
    w_pairs = 1.0 - w_dir
    return w_dir, w_pairs, state


# =============================================================================
# COMBINED BACKTEST ENGINE
# =============================================================================

def run_combined_backtest(data: Dict, btc_data: Dict, btc_hitrate: pd.Series,
                          dir_positions: Dict, state_df: pd.DataFrame, 
                          hitrate_df: pd.DataFrame, spread_tracker: SpreadStatsTracker,
                          dates: List, use_regime: bool = False,
                          alloc_strong_buy: float = 0.70, alloc_buy: float = 0.60,
                          alloc_sell: float = 0.50, alloc_strong_sell: float = 0.40,
                          silent: bool = False) -> CombinedBacktestResult:
    """Run the combined backtest."""
    
    if use_regime:
        method_name = f"regime_{int(alloc_strong_buy*100)}_{int(alloc_buy*100)}_{int(alloc_sell*100)}_{int(alloc_strong_sell*100)}"
    else:
        method_name = "risk_parity"
    
    if not silent:
        print(f"\n  Running combined backtest ({method_name})...", flush=True)
    
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    equity_curve = {}
    
    # Attribution tracking
    cumulative_dir_pnl = 0.0
    cumulative_pairs_pnl = 0.0
    
    # Yearly tracking
    yearly_dir_pnl = {}
    yearly_pairs_pnl = {}
    yearly_start_equity = {}
    yearly_peak = {}
    yearly_max_dd = {}
    
    # Allocation tracking
    allocation_weights = []
    btc_state_counts = {"STRONG_BUY": 0, "BUY": 0, "SELL": 0, "STRONG_SELL": 0}
    
    # Directional state
    returns_dict = {pair: data[pair]['returns'] for pair in DEPLOY_PAIRS}
    returns_df = pd.DataFrame(returns_dict).dropna()
    asset_weights = pd.Series({pair: 1.0/len(DEPLOY_PAIRS) for pair in DEPLOY_PAIRS})
    dir_prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    dir_last_rebalance_month = None
    
    # Pairs state
    open_trades: List[SpreadTrade] = []
    all_trades: List[SpreadTrade] = []
    pairs_entries_blocked = 0
    prev_unrealized_pnl = 0.0
    
    # Strategy return tracking for allocation calculation
    dir_daily_returns = {}
    pairs_daily_returns = {}
    
    # Costs
    total_costs = 0.0
    
    # Current allocation (starts 50/50)
    w_dir, w_pairs = 0.50, 0.50
    last_alloc_month = None
    
    # Record initial
    equity_curve[dates[0]] = equity
    current_year = dates[0].year
    yearly_start_equity[current_year] = equity
    yearly_dir_pnl[current_year] = 0.0
    yearly_pairs_pnl[current_year] = 0.0
    yearly_peak[current_year] = equity
    yearly_max_dd[current_year] = 0.0
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        current_month = (date.year, date.month)
        
        # Year tracking
        if next_date.year != current_year:
            current_year = next_date.year
            yearly_start_equity[current_year] = equity
            yearly_dir_pnl[current_year] = 0.0
            yearly_pairs_pnl[current_year] = 0.0
            yearly_peak[current_year] = equity
            yearly_max_dd[current_year] = 0.0
        
        # ─────────────────────────────────────────────────────────────
        # MONTHLY ALLOCATION RECALCULATION
        # ─────────────────────────────────────────────────────────────
        if last_alloc_month != current_month:
            if use_regime:
                # Use BTC's 32-state hit rate for regime detection
                btc_hr = btc_hitrate.get(date, 0.50)
                w_dir, w_pairs, btc_state = calculate_allocation_regime_32state(
                    btc_hr, alloc_strong_buy, alloc_buy, alloc_sell, alloc_strong_sell
                )
                btc_state_counts[btc_state] = btc_state_counts.get(btc_state, 0) + 1
            else:
                lookback_start = get_month_start(date, ALLOCATION_LOOKBACK_MONTHS)
                lookback_end = get_month_start(date, 0)
                dir_ret_series = pd.Series(dir_daily_returns)
                pairs_ret_series = pd.Series(pairs_daily_returns)
                w_dir, w_pairs = calculate_allocation_risk_parity(
                    dir_ret_series, pairs_ret_series, lookback_start, lookback_end)
            
            last_alloc_month = current_month
        
        allocation_weights.append((w_dir, w_pairs))
        
        # ─────────────────────────────────────────────────────────────
        # SHARED DRAWDOWN PROTECTION
        # ─────────────────────────────────────────────────────────────
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        
        if current_dd >= SHARED_DD_START_REDUCE:
            dd_scalar = 1.0
        elif current_dd <= SHARED_DD_MIN_EXPOSURE:
            dd_scalar = SHARED_MIN_EXPOSURE_FLOOR
        else:
            range_dd = SHARED_DD_START_REDUCE - SHARED_DD_MIN_EXPOSURE
            position = (current_dd - SHARED_DD_MIN_EXPOSURE) / range_dd
            dd_scalar = SHARED_MIN_EXPOSURE_FLOOR + position * (1.0 - SHARED_MIN_EXPOSURE_FLOOR)
        
        # Allocated capital for each strategy (after DD protection)
        dir_allocated_capital = equity * w_dir * dd_scalar
        pairs_allocated_capital = equity * w_pairs * dd_scalar
        
        # ═══════════════════════════════════════════════════════════════
        # PAIRS STRATEGY (FIRST PRIORITY)
        # ═══════════════════════════════════════════════════════════════
        
        pairs_daily_pnl = 0.0
        
        # Update cumulative returns for open trades (MTM)
        for trade in open_trades:
            if next_date in data[trade.long_asset]['returns'].index:
                long_ret_today = data[trade.long_asset]['returns'].loc[next_date]
                if not pd.isna(long_ret_today):
                    trade.cum_long_return = (1 + trade.cum_long_return) * (1 + long_ret_today) - 1
            
            if next_date in data[trade.short_asset]['returns'].index:
                short_ret_today = data[trade.short_asset]['returns'].loc[next_date]
                if not pd.isna(short_ret_today):
                    trade.cum_short_return = (1 + trade.cum_short_return) * (1 + short_ret_today) - 1
        
        # Check for exits
        realized_pairs_pnl = 0.0
        trades_to_close = []
        
        for trade in open_trades:
            holding_days = (next_date - trade.entry_date).days
            
            if next_date in state_df.index:
                curr_long_state = state_df.loc[next_date, trade.long_asset]
                curr_short_state = state_df.loc[next_date, trade.short_asset]
                curr_div = get_state_divergence(curr_long_state, curr_short_state)
            else:
                curr_div = ENTRY_DIVERGENCE
            
            should_exit = False
            exit_reason = None
            
            if curr_div < EXIT_DIVERGENCE:
                should_exit = True
                exit_reason = "CONVERGED"
            elif holding_days >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = "MAX_HOLD"
            
            if should_exit:
                spread_ret_gross = trade.cum_long_return - trade.cum_short_return
                
                entry_notional = trade.entry_capital * trade.position_size * 2
                exit_notional_long = trade.entry_capital * trade.position_size * (1 + trade.cum_long_return)
                exit_notional_short = trade.entry_capital * trade.position_size * (1 + trade.cum_short_return)
                exit_notional = exit_notional_long + exit_notional_short
                total_trading_cost = (entry_notional + exit_notional) * TRADING_COST
                
                cost_as_return = total_trading_cost / (trade.entry_capital * trade.position_size)
                spread_ret_net = spread_ret_gross - cost_as_return
                
                trade.exit_date = next_date
                trade.exit_reason = exit_reason
                trade.spread_return = spread_ret_net
                trade.holding_days = holding_days
                
                trade_pnl = trade.entry_capital * trade.position_size * spread_ret_net
                trade.pnl = trade_pnl
                realized_pairs_pnl += trade_pnl
                total_costs += total_trading_cost
                
                trades_to_close.append(trade)
        
        for trade in trades_to_close:
            open_trades.remove(trade)
            all_trades.append(trade)
        
        # Calculate unrealized change
        current_unrealized_pnl = 0.0
        for trade in open_trades:
            current_unrealized_pnl += trade.entry_capital * trade.position_size * trade.current_spread_return
        
        unrealized_change = current_unrealized_pnl - prev_unrealized_pnl
        prev_unrealized_pnl = current_unrealized_pnl
        
        pairs_daily_pnl = realized_pairs_pnl + unrealized_change
        
        # Enter new pairs trades
        if next_date in state_df.index:
            active_pairs = set()
            current_pairs_exposure = 0.0
            
            for trade in open_trades:
                active_pairs.add((trade.long_asset, trade.short_asset))
                active_pairs.add((trade.short_asset, trade.long_asset))
                current_pairs_exposure += (trade.get_current_value() / pairs_allocated_capital) * 2 if pairs_allocated_capital > 0 else 0
            
            for asset_a, asset_b in combinations(DEPLOY_PAIRS, 2):
                if (asset_a, asset_b) in active_pairs:
                    continue
                
                a_state = state_df.loc[next_date, asset_a]
                b_state = state_df.loc[next_date, asset_b]
                divergence = get_state_divergence(a_state, b_state)
                
                if divergence >= ENTRY_DIVERGENCE:
                    pair_key = tuple(sorted([asset_a, asset_b]))
                    proposed_size = spread_tracker.get_inverse_vol_size(pair_key, next_date)
                    proposed_exposure = proposed_size * 2
                    
                    trade_value_per_leg = proposed_size * pairs_allocated_capital
                    if trade_value_per_leg < MIN_TRADE_SIZE:
                        continue
                    
                    if current_pairs_exposure + proposed_exposure > PAIRS_MAX_EXPOSURE:
                        pairs_entries_blocked += 1
                        continue
                    
                    a_numeric = state_to_numeric(a_state)
                    b_numeric = state_to_numeric(b_state)
                    
                    if a_numeric > b_numeric:
                        long_asset, short_asset = asset_a, asset_b
                    else:
                        long_asset, short_asset = asset_b, asset_a
                    
                    trade = SpreadTrade(
                        entry_date=next_date,
                        long_asset=long_asset,
                        short_asset=short_asset,
                        entry_state_long=state_df.loc[next_date, long_asset],
                        entry_state_short=state_df.loc[next_date, short_asset],
                        entry_hit_rate_long=hitrate_df.loc[next_date, long_asset],
                        entry_hit_rate_short=hitrate_df.loc[next_date, short_asset],
                        position_size=proposed_size,
                        entry_capital=pairs_allocated_capital,
                        cum_long_return=0.0,
                        cum_short_return=0.0,
                    )
                    
                    open_trades.append(trade)
                    current_pairs_exposure += proposed_exposure
        
        # ═══════════════════════════════════════════════════════════════
        # DIRECTIONAL STRATEGY (SECOND PRIORITY)
        # ═══════════════════════════════════════════════════════════════
        
        dir_daily_pnl = 0.0
        dir_daily_cost = 0.0
        
        # Monthly rebalance of asset weights
        if dir_last_rebalance_month != current_month:
            lookback_start = get_month_start(date, COV_LOOKBACK_MONTHS)
            lookback_end = get_month_start(date, 0)
            lookback_returns = returns_df.loc[lookback_start:lookback_end]
            
            if len(lookback_returns) >= 20:
                asset_weights = calculate_risk_parity_weights(lookback_returns)
            dir_last_rebalance_month = current_month
        
        # Calculate base exposure from signals
        base_exposure = 0.0
        asset_exposures = {}
        
        for pair in DEPLOY_PAIRS:
            if date in dir_positions[pair]:
                pos = dir_positions[pair][date]
                asset_exposures[pair] = pos * asset_weights[pair]
                base_exposure += asset_exposures[pair]
        
        if base_exposure > 1.0:
            for pair in asset_exposures:
                asset_exposures[pair] /= base_exposure
            base_exposure = 1.0
        
        # Volatility scaling (using MARKET returns)
        vol_scalar = 1.0
        if date in returns_df.index:
            vol_lookback_start = get_month_start(date, VOL_LOOKBACK_MONTHS)
            vol_lookback_end = get_month_start(date, 0)
            vol_returns = returns_df.loc[vol_lookback_start:vol_lookback_end]
            
            if len(vol_returns) >= 15:
                market_returns = vol_returns.mean(axis=1)
                realized_vol = market_returns.std() * np.sqrt(365)
                if realized_vol > 0:
                    vol_scalar = DIR_TARGET_VOL / realized_vol
                    vol_scalar = np.clip(vol_scalar, DIR_MIN_EXPOSURE_FLOOR, DIR_MAX_EXPOSURE)
        
        # Apply vol scaling (DD protection already applied at portfolio level)
        risk_scalar = min(vol_scalar, 1.0 / dd_scalar if dd_scalar > 0 else 1.0)
        risk_scalar = max(risk_scalar, DIR_MIN_EXPOSURE_FLOOR)
        
        for pair in asset_exposures:
            asset_exposures[pair] *= risk_scalar
        
        # Calculate returns with costs
        for pair in DEPLOY_PAIRS:
            curr_exp = asset_exposures.get(pair, 0.0)
            prev_exp = dir_prev_exposures.get(pair, 0.0)
            
            trade_value = abs(curr_exp - prev_exp) * dir_allocated_capital
            
            if trade_value >= MIN_TRADE_SIZE:
                dir_daily_cost += trade_value * TRADING_COST
                dir_prev_exposures[pair] = curr_exp
            else:
                curr_exp = prev_exp
                asset_exposures[pair] = prev_exp
            
            if curr_exp > 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    dir_daily_pnl += dir_allocated_capital * curr_exp * ret
        
        dir_daily_pnl -= dir_daily_cost
        total_costs += dir_daily_cost
        
        # ═══════════════════════════════════════════════════════════════
        # UPDATE EQUITY
        # ═══════════════════════════════════════════════════════════════
        
        daily_pnl = dir_daily_pnl + pairs_daily_pnl
        equity += daily_pnl
        
        # Track attribution
        cumulative_dir_pnl += dir_daily_pnl
        cumulative_pairs_pnl += pairs_daily_pnl
        
        # Track yearly
        yearly_dir_pnl[next_date.year] = yearly_dir_pnl.get(next_date.year, 0) + dir_daily_pnl
        yearly_pairs_pnl[next_date.year] = yearly_pairs_pnl.get(next_date.year, 0) + pairs_daily_pnl
        
        # Track yearly drawdown
        if equity > yearly_peak.get(next_date.year, equity):
            yearly_peak[next_date.year] = equity
        year_dd = (yearly_peak[next_date.year] - equity) / yearly_peak[next_date.year]
        if year_dd > yearly_max_dd.get(next_date.year, 0):
            yearly_max_dd[next_date.year] = year_dd
        
        # Track daily returns for allocation
        if equity > 0:
            dir_daily_returns[next_date] = dir_daily_pnl / (equity - daily_pnl) if (equity - daily_pnl) > 0 else 0
            pairs_daily_returns[next_date] = pairs_daily_pnl / (equity - daily_pnl) if (equity - daily_pnl) > 0 else 0
        
        equity_curve[next_date] = equity
        
        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
    
    # ─────────────────────────────────────────────────────────────
    # CLOSE REMAINING PAIRS TRADES
    # ─────────────────────────────────────────────────────────────
    final_date = dates[-1]
    for trade in open_trades:
        spread_ret_gross = trade.cum_long_return - trade.cum_short_return
        
        entry_notional = trade.entry_capital * trade.position_size * 2
        exit_notional_long = trade.entry_capital * trade.position_size * (1 + trade.cum_long_return)
        exit_notional_short = trade.entry_capital * trade.position_size * (1 + trade.cum_short_return)
        exit_notional = exit_notional_long + exit_notional_short
        total_trading_cost = (entry_notional + exit_notional) * TRADING_COST
        cost_as_return = total_trading_cost / (trade.entry_capital * trade.position_size)
        spread_ret_net = spread_ret_gross - cost_as_return
        
        trade.exit_date = final_date
        trade.exit_reason = "END"
        trade.spread_return = spread_ret_net
        trade.holding_days = (final_date - trade.entry_date).days
        trade.pnl = trade.entry_capital * trade.position_size * spread_ret_net
        all_trades.append(trade)
    
    if not silent:
        print("    Complete.", flush=True)
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    returns_series = equity_series.pct_change().dropna()
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(365) if returns_series.std() > 0 else 0
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
    realized_vol = returns_series.std() * np.sqrt(365)
    
    # Attribution
    total_pnl = cumulative_dir_pnl + cumulative_pairs_pnl
    dir_contribution = cumulative_dir_pnl / total_pnl if total_pnl != 0 else 0.5
    pairs_contribution = cumulative_pairs_pnl / total_pnl if total_pnl != 0 else 0.5
    
    # Average allocation
    if allocation_weights:
        avg_dir_weight = np.mean([w[0] for w in allocation_weights])
        avg_pairs_weight = np.mean([w[1] for w in allocation_weights])
    else:
        avg_dir_weight = 0.5
        avg_pairs_weight = 0.5
    
    # Build yearly stats
    yearly_stats = []
    for year in sorted(yearly_start_equity.keys()):
        if year not in yearly_dir_pnl:
            continue
        
        # Find end equity for year
        year_dates = [d for d in equity_series.index if d.year == year]
        if not year_dates:
            continue
        end_eq = equity_series.loc[year_dates[-1]]
        start_eq = yearly_start_equity[year]
        
        year_return = (end_eq - start_eq) / start_eq if start_eq > 0 else 0
        year_total_pnl = yearly_dir_pnl[year] + yearly_pairs_pnl[year]
        
        yearly_stats.append(YearlyStats(
            year=year,
            return_pct=year_return,
            dir_pnl=yearly_dir_pnl[year],
            pairs_pnl=yearly_pairs_pnl[year],
            dir_contribution=yearly_dir_pnl[year] / year_total_pnl if year_total_pnl != 0 else 0.5,
            pairs_contribution=yearly_pairs_pnl[year] / year_total_pnl if year_total_pnl != 0 else 0.5,
            max_drawdown=yearly_max_dd.get(year, 0),
            start_equity=start_eq,
            end_equity=end_eq,
        ))
    
    # Pairs stats
    n_trades = len(all_trades)
    winning = [t for t in all_trades if t.spread_return and t.spread_return > 0]
    win_rate = len(winning) / n_trades if n_trades > 0 else 0
    
    returns_list = [t.spread_return for t in all_trades if t.spread_return is not None]
    avg_return = np.mean(returns_list) if returns_list else 0
    
    hold_days = [t.holding_days for t in all_trades if t.holding_days is not None]
    avg_hold = np.mean(hold_days) if hold_days else 0
    
    return CombinedBacktestResult(
        equity_curve=equity_series,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        realized_vol=realized_vol,
        directional_pnl=cumulative_dir_pnl,
        pairs_pnl=cumulative_pairs_pnl,
        directional_contribution=dir_contribution,
        pairs_contribution=pairs_contribution,
        avg_directional_weight=avg_dir_weight,
        avg_pairs_weight=avg_pairs_weight,
        allocation_method=method_name,
        yearly_stats=yearly_stats,
        btc_state_counts=btc_state_counts if use_regime else None,
        n_pairs_trades=n_trades,
        pairs_win_rate=win_rate,
        pairs_avg_return=avg_return,
        pairs_avg_hold_days=avg_hold,
        pairs_trades=all_trades,
        pairs_entries_blocked=pairs_entries_blocked,
        total_trading_costs=total_costs,
        n_days=len(dates),
    )


# =============================================================================
# BUY & HOLD BENCHMARK
# =============================================================================

def calculate_buy_and_hold(data: Dict, dates: List) -> pd.Series:
    """Calculate equal-weight buy & hold benchmark."""
    equity = INITIAL_CAPITAL
    weight = 1.0 / len(DEPLOY_PAIRS)
    equity_curve = {}
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        
        daily_return = 0.0
        for pair in DEPLOY_PAIRS:
            if next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    daily_return += weight * ret
        
        equity *= (1 + daily_return)
        equity_curve[date] = equity
    
    return pd.Series(equity_curve)


def calculate_bh_yearly(bh_curve: pd.Series) -> Dict[int, float]:
    """Calculate yearly returns for buy & hold."""
    yearly = {}
    years = sorted(set(d.year for d in bh_curve.index))
    
    for year in years:
        year_data = bh_curve[bh_curve.index.year == year]
        if len(year_data) < 2:
            continue
        
        # Get start (first of year or first available)
        prev_year_data = bh_curve[bh_curve.index.year == year - 1]
        if len(prev_year_data) > 0:
            start_val = prev_year_data.iloc[-1]
        else:
            start_val = year_data.iloc[0]
        
        end_val = year_data.iloc[-1]
        yearly[year] = (end_val - start_val) / start_val if start_val > 0 else 0
    
    return yearly


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def display_results(rp_result: CombinedBacktestResult, 
                    regime_results: Dict[str, CombinedBacktestResult],
                    bh_curve: pd.Series, dates: List):
    """Display comprehensive results."""
    
    # B&H metrics
    bh_total_return = (bh_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    bh_annual = (1 + bh_total_return) ** (1 / years) - 1
    bh_returns = bh_curve.pct_change().dropna()
    bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(365) if bh_returns.std() > 0 else 0
    bh_peak = bh_curve.expanding().max()
    bh_dd = ((bh_peak - bh_curve) / bh_peak).max()
    bh_yearly = calculate_bh_yearly(bh_curve)
    
    print("\n" + "=" * 100)
    print("COMBINED STRATEGY BACKTEST RESULTS")
    print("=" * 100)
    
    # ─────────────────────────────────────────────────────────────
    # RISK PARITY RESULTS
    # ─────────────────────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════════════════════════════════╗")
    print("  ║  RISK PARITY (PRIMARY METHOD)                                        ║")
    print("  ╚══════════════════════════════════════════════════════════════════════╝")
    
    print(f"""
    Total Return:        {rp_result.total_return*100:>+,.0f}%
    Annual Return:       {rp_result.annual_return*100:>+.1f}%
    Sharpe Ratio:        {rp_result.sharpe_ratio:.2f}
    Max Drawdown:        {rp_result.max_drawdown*100:.1f}%
    Calmar Ratio:        {rp_result.calmar_ratio:.2f}
    Realized Vol:        {rp_result.realized_vol*100:.1f}%
    
    Directional P&L:     ${rp_result.directional_pnl:>,.0f} ({rp_result.directional_contribution*100:.1f}%)
    Pairs P&L:           ${rp_result.pairs_pnl:>,.0f} ({rp_result.pairs_contribution*100:.1f}%)
    Avg Dir Weight:      {rp_result.avg_directional_weight*100:.1f}%
    Avg Pairs Weight:    {rp_result.avg_pairs_weight*100:.1f}%
    
    Final Equity:        ${rp_result.equity_curve.iloc[-1]:>,.0f}
    """)
    
    # Yearly breakdown for Risk Parity
    print("  YEARLY BREAKDOWN - RISK PARITY")
    print("  " + "-" * 80)
    print(f"  {'Year':<6} {'Return':>10} {'Dir P&L':>14} {'Pairs P&L':>14} {'Dir %':>8} {'Pairs %':>8} {'Max DD':>8}")
    print("  " + "-" * 80)
    
    for ys in rp_result.yearly_stats:
        print(f"  {ys.year:<6} {ys.return_pct*100:>+9.1f}% ${ys.dir_pnl:>12,.0f} ${ys.pairs_pnl:>12,.0f} "
              f"{ys.dir_contribution*100:>7.1f}% {ys.pairs_contribution*100:>7.1f}% {ys.max_drawdown*100:>7.1f}%")
    
    # ─────────────────────────────────────────────────────────────
    # 32-STATE REGIME ADAPTIVE SENSITIVITY
    # ─────────────────────────────────────────────────────────────
    print("\n\n  ╔══════════════════════════════════════════════════════════════════════╗")
    print("  ║  32-STATE REGIME ADAPTIVE SENSITIVITY ANALYSIS                       ║")
    print("  ║  (Uses BTC 32-State Hit Rate - Same System as Trading Signals)       ║")
    print("  ╚══════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n  {'Scheme':<15} {'Annual':>10} {'Sharpe':>8} {'Max DD':>8} {'Calmar':>8} {'Vol':>8} {'Dir %':>8}")
    print("  " + "-" * 80)
    
    best_sharpe = None
    best_calmar = None
    
    for name, result in regime_results.items():
        print(f"  {name:<15} {result.annual_return*100:>+9.1f}% {result.sharpe_ratio:>8.2f} "
              f"{result.max_drawdown*100:>7.1f}% {result.calmar_ratio:>8.2f} "
              f"{result.realized_vol*100:>7.1f}% {result.directional_contribution*100:>7.1f}%")
        
        if best_sharpe is None or result.sharpe_ratio > best_sharpe[1].sharpe_ratio:
            best_sharpe = (name, result)
        if best_calmar is None or result.calmar_ratio > best_calmar[1].calmar_ratio:
            best_calmar = (name, result)
    
    print("  " + "-" * 80)
    print(f"  {'Risk Parity':<15} {rp_result.annual_return*100:>+9.1f}% {rp_result.sharpe_ratio:>8.2f} "
          f"{rp_result.max_drawdown*100:>7.1f}% {rp_result.calmar_ratio:>8.2f} "
          f"{rp_result.realized_vol*100:>7.1f}% {rp_result.directional_contribution*100:>7.1f}%")
    print(f"  {'Buy & Hold':<15} {bh_annual*100:>+9.1f}% {bh_sharpe:>8.2f} {bh_dd*100:>7.1f}%")
    
    print(f"\n  BEST REGIME BY SHARPE: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")
    print(f"  BEST REGIME BY CALMAR: {best_calmar[0]} ({best_calmar[1].calmar_ratio:.2f})")
    
    # Show BTC state distribution for best regime
    if best_calmar[1].btc_state_counts:
        counts = best_calmar[1].btc_state_counts
        total = sum(counts.values())
        print(f"\n  BTC STATE DISTRIBUTION ({best_calmar[0]}):")
        for state in ["STRONG_BUY", "BUY", "SELL", "STRONG_SELL"]:
            pct = counts.get(state, 0) / total * 100 if total > 0 else 0
            print(f"    {state:<12}: {counts.get(state, 0):>4} months ({pct:>5.1f}%)")
    
    # Yearly breakdown for best regime
    best_regime = best_calmar[1]
    print(f"\n\n  YEARLY BREAKDOWN - {best_calmar[0].upper()} (Best Calmar)")
    print("  " + "-" * 80)
    print(f"  {'Year':<6} {'Return':>10} {'Dir P&L':>14} {'Pairs P&L':>14} {'Dir %':>8} {'Pairs %':>8} {'Max DD':>8}")
    print("  " + "-" * 80)
    
    for ys in best_regime.yearly_stats:
        print(f"  {ys.year:<6} {ys.return_pct*100:>+9.1f}% ${ys.dir_pnl:>12,.0f} ${ys.pairs_pnl:>12,.0f} "
              f"{ys.dir_contribution*100:>7.1f}% {ys.pairs_contribution*100:>7.1f}% {ys.max_drawdown*100:>7.1f}%")
    
    # ─────────────────────────────────────────────────────────────
    # COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────
    print("\n\n  ╔══════════════════════════════════════════════════════════════════════╗")
    print("  ║  YEARLY COMPARISON: RISK PARITY vs BEST REGIME vs BUY & HOLD         ║")
    print("  ╚══════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n  {'Year':<6} {'Risk Parity':>12} {'Best Regime':>12} {'Buy & Hold':>12} {'RP vs B&H':>12}")
    print("  " + "-" * 60)
    
    for ys in rp_result.yearly_stats:
        year = ys.year
        rp_ret = ys.return_pct
        
        # Find matching regime year
        regime_ret = 0
        for rs in best_regime.yearly_stats:
            if rs.year == year:
                regime_ret = rs.return_pct
                break
        
        bh_ret = bh_yearly.get(year, 0)
        diff = rp_ret - bh_ret
        
        print(f"  {year:<6} {rp_ret*100:>+11.1f}% {regime_ret*100:>+11.1f}% {bh_ret*100:>+11.1f}% {diff*100:>+11.1f}%")
    
    # ─────────────────────────────────────────────────────────────
    # RECOMMENDATION
    # ─────────────────────────────────────────────────────────────
    print("\n\n  ╔══════════════════════════════════════════════════════════════════════╗")
    print("  ║  RECOMMENDATION                                                       ║")
    print("  ╚══════════════════════════════════════════════════════════════════════╝")
    
    if rp_result.sharpe_ratio >= best_sharpe[1].sharpe_ratio:
        print(f"""
    PRIMARY: Risk Parity (Sharpe: {rp_result.sharpe_ratio:.2f})
    - Highest Sharpe ratio
    - Adaptive allocation based on realised volatility
    - Better capital efficiency
        """)
    else:
        print(f"""
    PRIMARY: Risk Parity (Sharpe: {rp_result.sharpe_ratio:.2f})
    ALTERNATIVE: {best_calmar[0]} (Calmar: {best_calmar[1].calmar_ratio:.2f})
    - Use Risk Parity for maximum risk-adjusted returns
    - Use {best_calmar[0]} for lower drawdowns ({best_calmar[1].max_drawdown*100:.1f}% vs {rp_result.max_drawdown*100:.1f}%)
        """)


def plot_results(rp_result: CombinedBacktestResult, 
                 best_regime: CombinedBacktestResult,
                 bh_curve: pd.Series, btc_data: Dict,
                 filename: str = 'combined_backtest_32state_regime.png'):
    """Generate comparison chart."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        btc_price = btc_data['prices']['close']
        btc_aligned = btc_price.reindex(rp_result.equity_curve.index, method='ffill')
        
        # Panel 1: Equity curves
        ax1 = axes[0]
        ax1.plot(rp_result.equity_curve.index, rp_result.equity_curve.values, 
                 label='Risk Parity', color='#2C5282', linewidth=2)
        ax1.plot(best_regime.equity_curve.index, best_regime.equity_curve.values, 
                 label=f'32-State Regime ({best_regime.allocation_method})', color='#B7791F', linewidth=2)
        ax1.plot(bh_curve.index, bh_curve.values, 
                 label='Buy & Hold', color='#C53030', linewidth=2, linestyle='--')
        
        ax1.set_title('Combined Strategy - Risk Parity vs 32-State Regime vs Buy & Hold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Panel 2: BTC with state indication (simplified - just price)
        ax2 = axes[1]
        ax2.plot(btc_aligned.index, btc_aligned.values, color='#F6AD55', linewidth=2)
        ax2.set_title('Bitcoin Price (BTC/USD) - Regime from 32-State Hit Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Price ($)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Drawdowns comparison
        ax3 = axes[2]
        
        rp_peak = rp_result.equity_curve.expanding().max()
        rp_dd = (rp_peak - rp_result.equity_curve) / rp_peak * 100
        
        regime_peak = best_regime.equity_curve.expanding().max()
        regime_dd = (regime_peak - best_regime.equity_curve) / regime_peak * 100
        
        ax3.fill_between(rp_dd.index, rp_dd.values, 0, color='#2C5282', alpha=0.3, label='Risk Parity')
        ax3.fill_between(regime_dd.index, regime_dd.values, 0, color='#B7791F', alpha=0.3, label='32-State Regime')
        ax3.plot(rp_dd.index, rp_dd.values, color='#2C5282', linewidth=1)
        ax3.plot(regime_dd.index, regime_dd.values, color='#B7791F', linewidth=1)
        
        ax3.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.set_ylim(top=0)
        ax3.legend(loc='lower left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\n  Chart saved to: {filename}")
        plt.close()
        
    except ImportError:
        print("\n  (matplotlib not available - skipping chart)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("COMBINED STRATEGY BACKTEST v3")
    print("Risk Parity + 32-State Regime Adaptive (Consistent Framework)")
    print("=" * 80)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  CONFIGURATION                                                     ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Primary: Risk Parity (inverse volatility weighting)               ║
    ║  Alternative: 32-State Regime Adaptive (BTC hit rate)              ║
    ║                                                                    ║
    ║  KEY FIX: Regime now uses BTC's 32-state hit rate                 ║
    ║           (Same system as trading signals - CONSISTENT)            ║
    ║                                                                    ║
    ║  32-State Allocation Grid:                                         ║
    ║    State        Aggressive  Default  Moderate  Conservative  Min   ║
    ║    STRONG_BUY   80/20       70/30    65/35     60/40        55/45  ║
    ║    BUY          70/30       60/40    55/45     55/45        52/48  ║
    ║    SELL         50/50       50/50    50/50     50/50        50/50  ║
    ║    STRONG_SELL  30/70       40/60    45/55     45/55        48/52  ║
    ║                                                                    ║
    ║  • Shared drawdown protection                                      ║
    ║  • Pairs-first execution priority                                  ║
    ║  • Hard monthly rebalancing                                        ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print("  Connecting to database...", flush=True)
    db = Database()
    print("  Connected.", flush=True)
    
    # Load data
    data = load_all_data(db)
    btc_data = load_btc_data(db)
    
    # Find common dates
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))
    
    # Also intersect with BTC dates
    all_dates = all_dates.intersection(set(btc_data['signals'].index))
    
    dates = sorted(list(all_dates))
    data_start = dates[0]
    
    print(f"\n  Common dates: {len(dates)} ({dates[0].date()} to {dates[-1].date()})")
    
    # Calculate position/state histories
    print("\n  Calculating directional positions...", flush=True)
    dir_positions = calculate_directional_positions(data, dates, data_start)
    
    print("  Calculating pairs states...", flush=True)
    state_df, hitrate_df = calculate_pairs_states(data, dates, data_start)
    
    print("  Calculating BTC 32-state hit rates...", flush=True)
    btc_hitrate = calculate_btc_hit_rate_history(btc_data, dates, data_start)
    print(f"    BTC hit rates calculated for {len(btc_hitrate)} dates", flush=True)
    
    # Create spread tracker
    spread_tracker = SpreadStatsTracker(data)
    
    # ─────────────────────────────────────────────────────────────
    # RUN RISK PARITY
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RUNNING RISK PARITY")
    print("=" * 60)
    
    rp_result = run_combined_backtest(
        data, btc_data, btc_hitrate, dir_positions, state_df, hitrate_df,
        spread_tracker, dates, use_regime=False
    )
    
    # ─────────────────────────────────────────────────────────────
    # RUN 32-STATE REGIME SENSITIVITY
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RUNNING 32-STATE REGIME ADAPTIVE SENSITIVITY")
    print("=" * 60)
    
    regime_results = {}
    for sb, b, s, ss, name in REGIME_SENSITIVITY_GRID:
        result = run_combined_backtest(
            data, btc_data, btc_hitrate, dir_positions, state_df, hitrate_df,
            spread_tracker, dates, use_regime=True,
            alloc_strong_buy=sb, alloc_buy=b, alloc_sell=s, alloc_strong_sell=ss,
            silent=True
        )
        regime_results[name] = result
        print(f"    {name}: Sharpe={result.sharpe_ratio:.2f}, Calmar={result.calmar_ratio:.2f}")
    
    # ─────────────────────────────────────────────────────────────
    # BENCHMARK
    # ─────────────────────────────────────────────────────────────
    print("\n  Calculating buy & hold benchmark...", flush=True)
    bh_curve = calculate_buy_and_hold(data, dates)
    
    # ─────────────────────────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────────────────────────
    display_results(rp_result, regime_results, bh_curve, dates)
    
    # Find best regime for plotting
    best_calmar = max(regime_results.items(), key=lambda x: x[1].calmar_ratio)
    
    # Plot
    plot_results(rp_result, best_calmar[1], bh_curve, btc_data)
    
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    
    return rp_result, regime_results, bh_curve


if __name__ == "__main__":
    rp_result, regime_results, bh_curve = main()