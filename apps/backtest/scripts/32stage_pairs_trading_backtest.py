#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairs Regime Trading Strategy Backtest
======================================
Standalone backtest for the pairs regime divergence trading strategy.

Features:
- 32-state regime classification shared with directional strategy
- State divergence entry (≥2 states apart)
- Convergence or max-hold exit (10 days)
- Inverse volatility position sizing
- Configurable exposure limits
- Transaction costs

Documented Performance: +112% annual, 1.39 Sharpe (with leverage)
Expected with 100% max exposure: ~20-40% annual

Usage:
    python pairs_backtest.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker/cryptobot/data')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations

from database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked from validation)
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005

# Hit rate parameters
MIN_SAMPLES_PER_STATE = 20

# =============================================================================
# CALENDAR MONTH ALIGNMENT
# =============================================================================
# All lookbacks and recalculations align to calendar month boundaries
# (e.g., "2 months back" from Nov 15 = Sep 1, not Oct 16)
#
# This aligns with financial market rhythms:
#   - Earnings cycles
#   - Fund flows and rebalancing
#   - Options expiry
#   - Economic data releases
# =============================================================================

MIN_TRAINING_MONTHS = 12          # Require 12 full calendar months before trading
HIT_RATE_RECALC_MONTHS = 1        # Recalculate hit rates on 1st of each month

# =============================================================================
# EXPOSURE & RISK CONTROLS - ADJUST THESE TO YOUR COMFORT LEVEL
# =============================================================================
#
# MAX_EXPOSURE: Maximum total exposure as multiple of account equity
#
# In pairs trading, each trade has TWO legs (long + short), so:
#   - 10% position size = 20% exposure per trade (10% long + 10% short)
#   - With 15 possible pairs, maximum theoretical exposure = 300%
#
# This parameter LIMITS how much total exposure you can have at once:
#
#   1.0 = No leverage (max 100% exposure, ~3-5 concurrent trades)
#   1.5 = Moderate leverage (max 150% exposure, ~5-7 concurrent trades)
#   2.0 = 2x leverage (max 200% exposure, ~7-10 concurrent trades)
#   3.0 = 3x leverage (max 300% exposure, all pairs possible)
#
# Example with $100,000 account and 10% position size:
#   MAX_EXPOSURE = 1.0  →  Max 5 trades × 20% = 100% exposure
#   MAX_EXPOSURE = 2.0  →  Max 10 trades × 20% = 200% exposure
#   MAX_EXPOSURE = 3.0  →  Max 15 trades × 20% = 300% exposure
#
# RISK vs RETURN TRADEOFF (approximate):
#   MAX_EXPOSURE = 1.0  →  ~15-20% annual return, ~30% max drawdown
#   MAX_EXPOSURE = 1.5  →  ~30-40% annual return, ~40% max drawdown
#   MAX_EXPOSURE = 2.0  →  ~50-60% annual return, ~45% max drawdown
#   MAX_EXPOSURE = 3.0  →  ~100%+ annual return, ~55% max drawdown (original)
#
# =============================================================================

MAX_EXPOSURE = 2.0  # ← ADJUST THIS (1.0 = no leverage, conservative)

# =============================================================================
# DRAWDOWN PROTECTION - Automatically reduces exposure during losses
# =============================================================================
#
# When the pairs strategy is losing money, exposure is automatically reduced
# to protect capital and allow for recovery.
#
# How it works:
#   - Above DD_START_REDUCE: Full exposure (100% of MAX_EXPOSURE)
#   - Between DD_START_REDUCE and DD_MIN_EXPOSURE: Linear reduction
#   - Below DD_MIN_EXPOSURE: Minimum exposure (MIN_EXPOSURE_FLOOR)
#
# Example with MAX_EXPOSURE = 2.0 (200%):
#   Drawdown =  -10%  →  200% exposure (full)
#   Drawdown =  -20%  →  200% exposure (full, at threshold)
#   Drawdown =  -35%  →  140% exposure (reduced)
#   Drawdown =  -50%  →   80% exposure (minimum = 40% of 200%)
#
# =============================================================================

DD_START_REDUCE = -0.20       # Start reducing exposure at -20% drawdown
DD_MIN_EXPOSURE = -0.50       # Maximum drawdown before minimum exposure
MIN_EXPOSURE_FLOOR = 0.40     # Never go below 40% of MAX_EXPOSURE

# =============================================================================
# STATE & DIVERGENCE CONFIGURATION
# =============================================================================
#
# Each asset's 32-state is converted to a simplified state based on hit rate:
#
#   Hit Rate >= 0.55  →  STRONG_BUY  (numeric: 3)  "Bullish, expect up"
#   Hit Rate >= 0.50  →  BUY         (numeric: 2)  "Slightly bullish"
#   Hit Rate >= 0.45  →  SELL        (numeric: 1)  "Slightly bearish"
#   Hit Rate <  0.45  →  STRONG_SELL (numeric: 0)  "Bearish, expect down"
#
# DIVERGENCE = abs(state_A_numeric - state_B_numeric)
#
# Possible divergence values: 0, 1, 2, 3
#
#   Divergence 3:  STRONG_BUY vs STRONG_SELL  (maximum divergence)
#   Divergence 2:  STRONG_BUY vs SELL, BUY vs STRONG_SELL
#   Divergence 1:  STRONG_BUY vs BUY, BUY vs SELL, SELL vs STRONG_SELL
#   Divergence 0:  Same state (no divergence)
#
# TRADE LOGIC:
#   Entry: When divergence >= ENTRY_DIVERGENCE
#   Exit:  When divergence < EXIT_DIVERGENCE OR holding_days >= MAX_HOLD_DAYS
#
# =============================================================================

# State thresholds (for converting hit rate to simplified state)
STRONG_BUY_THRESHOLD = 0.55   # Hit rate >= this → STRONG_BUY
BUY_THRESHOLD = 0.50          # Hit rate >= this → BUY
SELL_THRESHOLD = 0.45         # Hit rate >= this → SELL (else STRONG_SELL)

# Divergence thresholds
ENTRY_DIVERGENCE = 2          # Minimum divergence to enter a trade
EXIT_DIVERGENCE = 2           # Exit when divergence drops below this

# Trade management
MAX_HOLD_DAYS = 10            # Force exit after this many days

# Position sizing (inverse volatility)
BASE_POSITION_SIZE = 0.10     # Base position size per leg (10%)
MAX_POSITION_SIZE = 0.25      # Maximum position size per leg (25%)
MIN_POSITION_SIZE = 0.02      # Minimum position size per leg (2%)

# Backtest parameters
INITIAL_CAPITAL = 100000.0
TRADING_COST = 0.0015         # Per trade per leg (0.15%)
MIN_TRADE_SIZE = 100.0        # Minimum trade value in dollars per leg


# =============================================================================
# =============================================================================
# CALENDAR MONTH HELPER FUNCTIONS
# =============================================================================

def get_month_start(date: pd.Timestamp, months_back: int = 0) -> pd.Timestamp:
    """Get the 1st of the month, optionally N months back.
    
    Examples:
        get_month_start(Nov 15, 0) → Nov 1
        get_month_start(Nov 15, 1) → Oct 1
        get_month_start(Nov 15, 2) → Sep 1
    """
    # Go back N months
    target = date - pd.DateOffset(months=months_back)
    # Return 1st of that month
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
    position_size: float  # Per leg (total trade exposure = 2 × position_size)
    entry_capital: float
    exit_date: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    spread_return: Optional[float] = None
    holding_days: Optional[int] = None
    pnl: Optional[float] = None
    # For mark-to-market tracking
    cum_long_return: float = 0.0  # Cumulative compounded return on long leg
    cum_short_return: float = 0.0  # Cumulative compounded return on short leg
    
    @property
    def pair_key(self) -> Tuple[str, str]:
        return tuple(sorted([self.long_asset, self.short_asset]))
    
    @property
    def current_spread_return(self) -> float:
        """Current spread return (long - short) before costs."""
        return self.cum_long_return - self.cum_short_return
    
    def get_current_value(self) -> float:
        """Get current trade value (reflects MTM gains/losses)."""
        # Position value = entry_capital × position_size × (1 + spread_return)
        return self.entry_capital * self.position_size * (1 + self.current_spread_return)


@dataclass
class BacktestResult:
    """Results from backtest."""
    equity_curve: pd.Series
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    realized_vol: float
    n_trades: int
    win_rate: float
    avg_return_per_trade: float
    avg_holding_days: float
    trades: List[SpreadTrade]
    entries_blocked: int


# =============================================================================
# SIGNAL GENERATION
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
    """Generate 32-state signals. SHIFT BEFORE REINDEX to prevent look-ahead."""
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
        return {(p, m): {'hit_rate': 0.5} for p in all_price_perms for m in all_ma_perms}
    
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < MIN_SAMPLES_PER_STATE:
        return {(p, m): {'hit_rate': 0.5} for p in all_price_perms for m in all_ma_perms}
    
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
            
            if n >= MIN_SAMPLES_PER_STATE:
                hit_rate = (perm_returns > 0).sum() / n
            else:
                hit_rate = 0.5
            
            hit_rates[(price_perm, ma_perm)] = {'hit_rate': hit_rate, 'n': n}
    
    return hit_rates


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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database) -> Dict:
    """Load and prepare all data."""
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


# =============================================================================
# STATE & HIT RATE HISTORY
# =============================================================================

def calculate_state_history(data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """Calculate state and hit rate history for all assets."""
    
    # Find common dates
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))
    
    all_dates = sorted(list(all_dates))
    
    # Get data start date for training check
    data_start = all_dates[0]
    
    print(f"\n  Calculating state history for {len(all_dates)} common dates...", flush=True)
    print(f"  (Recalculating hit rates on 1st of each month)", flush=True)
    
    state_history = {pair: {} for pair in DEPLOY_PAIRS}
    hitrate_history = {pair: {} for pair in DEPLOY_PAIRS}
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}
    
    last_progress = 0
    recalc_count = 0
    
    for i, date in enumerate(all_dates):
        progress = int((i / len(all_dates)) * 100)
        if progress >= last_progress + 10:
            print(f"    Progress: {progress}%", flush=True)
            last_progress = progress
        
        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            
            if date not in signals.index:
                continue
            
            sig = signals.loc[date]
            price_perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
            ma_perm = (int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24']))
            
            # Check if we have minimum training months
            if not has_minimum_training(date, data_start, MIN_TRAINING_MONTHS):
                state_history[pair][date] = "NEUTRAL"
                hitrate_history[pair][date] = 0.50
                continue
            
            # Recalculate hit rates on 1st of each month (or if never calculated)
            current_month = (date.year, date.month)
            if last_recalc_month[pair] != current_month:
                # Use data up to start of current month for calculation
                cutoff_date = get_month_start(date, 0)
                hist_returns = returns[returns.index < cutoff_date]
                hist_signals = signals[signals.index < cutoff_date]
                
                if len(hist_returns) > 0:
                    hit_rate_cache[pair] = calculate_expanding_hit_rates(hist_returns, hist_signals)
                    last_recalc_month[pair] = current_month
                    recalc_count += 1
            
            key = (price_perm, ma_perm)
            hr_data = hit_rate_cache[pair].get(key, {'hit_rate': 0.5})
            hit_rate = hr_data['hit_rate']
            
            state_history[pair][date] = hit_rate_to_simple_state(hit_rate)
            hitrate_history[pair][date] = hit_rate
    
    print(f"    Progress: 100%", flush=True)
    print(f"    Total recalculations: {recalc_count}", flush=True)
    
    state_df = pd.DataFrame(state_history)
    state_df.index = pd.to_datetime(state_df.index)
    
    hitrate_df = pd.DataFrame(hitrate_history)
    hitrate_df.index = pd.to_datetime(hitrate_df.index)
    
    return state_df.dropna(), hitrate_df.dropna(), all_dates


# =============================================================================
# SPREAD STATISTICS (for inverse volatility sizing)
# =============================================================================

class SpreadStatsTracker:
    """Track spread statistics for inverse volatility sizing."""
    
    def __init__(self, data: Dict):
        self.data = data
        self.pairs = list(combinations(DEPLOY_PAIRS, 2))
        
        # Pre-calculate spread returns
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
# BACKTEST ENGINE
# =============================================================================

def run_backtest(data: Dict, state_df: pd.DataFrame, hitrate_df: pd.DataFrame,
                 spread_tracker: SpreadStatsTracker, dates: List) -> BacktestResult:
    """
    Run the pairs trading backtest.
    
    FIXES APPLIED:
    1. Proper compounding for multi-day returns (not summing)
    2. Daily mark-to-market for open positions
    3. Trading costs based on notional value
    4. Equity curve aligned to correct dates
    """
    
    print("\n  Running backtest...", flush=True)
    print(f"    Max total exposure: {MAX_EXPOSURE*100:.0f}% (with drawdown protection)", flush=True)
    
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    equity_curve = {}
    
    # Track previous day's unrealized P&L for proper accounting
    prev_unrealized_pnl = 0.0
    
    open_trades: List[SpreadTrade] = []
    all_trades: List[SpreadTrade] = []
    entries_blocked = 0
    
    # Record initial equity
    equity_curve[dates[0]] = equity
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        
        # ─────────────────────────────────────────────────────────────
        # STEP 1: UPDATE CUMULATIVE RETURNS FOR ALL OPEN TRADES (MTM)
        # This must happen BEFORE checking exit conditions
        # ─────────────────────────────────────────────────────────────
        for trade in open_trades:
            # Get today's return for each leg
            if next_date in data[trade.long_asset]['returns'].index:
                long_ret_today = data[trade.long_asset]['returns'].loc[next_date]
                if not pd.isna(long_ret_today):
                    # Compound: (1 + cum) × (1 + today) - 1
                    trade.cum_long_return = (1 + trade.cum_long_return) * (1 + long_ret_today) - 1
            
            if next_date in data[trade.short_asset]['returns'].index:
                short_ret_today = data[trade.short_asset]['returns'].loc[next_date]
                if not pd.isna(short_ret_today):
                    trade.cum_short_return = (1 + trade.cum_short_return) * (1 + short_ret_today) - 1
        
        # ─────────────────────────────────────────────────────────────
        # STEP 2: CHECK FOR EXIT CONDITIONS
        # ─────────────────────────────────────────────────────────────
        realized_pnl = 0.0
        trades_to_close = []
        
        for trade in open_trades:
            holding_days = (next_date - trade.entry_date).days
            
            # Check exit conditions based on next_date's state
            if next_date in state_df.index:
                curr_long_state = state_df.loc[next_date, trade.long_asset]
                curr_short_state = state_df.loc[next_date, trade.short_asset]
                curr_div = get_state_divergence(curr_long_state, curr_short_state)
            else:
                curr_div = ENTRY_DIVERGENCE  # Don't exit if no data
            
            should_exit = False
            exit_reason = None
            
            if curr_div < EXIT_DIVERGENCE:
                should_exit = True
                exit_reason = "CONVERGED"
            elif holding_days >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = "MAX_HOLD"
            
            if should_exit:
                # Spread return = long return - short return (already compounded)
                spread_ret_gross = trade.cum_long_return - trade.cum_short_return
                
                # Trading costs: 4 legs total (entry long, entry short, exit long, exit short)
                # Cost is based on NOTIONAL VALUE, not return
                # Entry notional = entry_capital × position_size per leg
                # Exit notional ≈ entry_capital × position_size × (1 + leg_return) per leg
                entry_notional = trade.entry_capital * trade.position_size * 2  # Both legs
                exit_notional_long = trade.entry_capital * trade.position_size * (1 + trade.cum_long_return)
                exit_notional_short = trade.entry_capital * trade.position_size * (1 + trade.cum_short_return)
                exit_notional = exit_notional_long + exit_notional_short
                total_trading_cost = (entry_notional + exit_notional) * TRADING_COST
                
                # Net spread return (after costs as percentage of entry capital × position_size)
                cost_as_return = total_trading_cost / (trade.entry_capital * trade.position_size)
                spread_ret_net = spread_ret_gross - cost_as_return
                
                trade.exit_date = next_date
                trade.exit_reason = exit_reason
                trade.spread_return = spread_ret_net
                trade.holding_days = holding_days
                
                # PnL = entry_capital × position_size × net_spread_return
                trade_pnl = trade.entry_capital * trade.position_size * spread_ret_net
                trade.pnl = trade_pnl
                realized_pnl += trade_pnl
                
                trades_to_close.append(trade)
        
        for trade in trades_to_close:
            open_trades.remove(trade)
            all_trades.append(trade)
        
        # ─────────────────────────────────────────────────────────────
        # STEP 3: CALCULATE UNREALIZED P&L FOR REMAINING OPEN TRADES
        # ─────────────────────────────────────────────────────────────
        current_unrealized_pnl = 0.0
        for trade in open_trades:
            # Unrealized P&L = entry_capital × position_size × current_spread_return
            current_unrealized_pnl += trade.entry_capital * trade.position_size * trade.current_spread_return
        
        # Daily unrealized change = current unrealized - previous unrealized
        unrealized_change = current_unrealized_pnl - prev_unrealized_pnl
        prev_unrealized_pnl = current_unrealized_pnl
        
        # ─────────────────────────────────────────────────────────────
        # STEP 4: UPDATE EQUITY (realized + unrealized change)
        # ─────────────────────────────────────────────────────────────
        # Note: When a trade closes, its unrealized becomes realized
        # The unrealized_change for closed trades is negative (removed from unrealized)
        # The realized_pnl captures the final P&L
        # This naturally handles the transition correctly
        daily_pnl = realized_pnl + unrealized_change
        equity += daily_pnl
        
        # Store equity on next_date (when returns were realized)
        equity_curve[next_date] = equity
        
        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
        
        # ─────────────────────────────────────────────────────────────
        # STEP 5: CALCULATE DYNAMIC EXPOSURE LIMIT (drawdown protection)
        # ─────────────────────────────────────────────────────────────
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        
        if current_dd >= DD_START_REDUCE:
            dd_scalar = 1.0
        elif current_dd <= DD_MIN_EXPOSURE:
            dd_scalar = MIN_EXPOSURE_FLOOR
        else:
            range_dd = DD_START_REDUCE - DD_MIN_EXPOSURE
            position = (current_dd - DD_MIN_EXPOSURE) / range_dd
            dd_scalar = MIN_EXPOSURE_FLOOR + position * (1.0 - MIN_EXPOSURE_FLOOR)
        
        dynamic_max_exposure = MAX_EXPOSURE * dd_scalar
        
        # ─────────────────────────────────────────────────────────────
        # STEP 6: ENTER NEW TRADES (with dynamic exposure limit)
        # Use next_date for state checking (signal available at end of next_date)
        # ─────────────────────────────────────────────────────────────
        if next_date not in state_df.index:
            continue
            
        active_pairs = set()
        current_exposure = 0.0
        
        for trade in open_trades:
            active_pairs.add((trade.long_asset, trade.short_asset))
            active_pairs.add((trade.short_asset, trade.long_asset))
            # Use current value for exposure calculation (reflects MTM)
            current_exposure += (trade.get_current_value() / equity) * 2 if equity > 0 else 0
        
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
                
                # Check minimum trade size per leg
                trade_value_per_leg = proposed_size * equity
                if trade_value_per_leg < MIN_TRADE_SIZE:
                    continue
                
                # Check dynamic exposure limit
                if current_exposure + proposed_exposure > dynamic_max_exposure:
                    entries_blocked += 1
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
                    entry_capital=equity,
                    cum_long_return=0.0,
                    cum_short_return=0.0,
                )
                
                open_trades.append(trade)
                current_exposure += proposed_exposure
    
    # ─────────────────────────────────────────────────────────────
    # CLOSE REMAINING TRADES AT END
    # ─────────────────────────────────────────────────────────────
    final_date = dates[-1]
    for trade in open_trades:
        spread_ret_gross = trade.cum_long_return - trade.cum_short_return
        
        # Calculate trading costs
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
    
    # Trade stats
    n_trades = len(all_trades)
    winning = [t for t in all_trades if t.spread_return and t.spread_return > 0]
    win_rate = len(winning) / n_trades if n_trades > 0 else 0
    
    returns_list = [t.spread_return for t in all_trades if t.spread_return is not None]
    avg_return = np.mean(returns_list) if returns_list else 0
    
    hold_days = [t.holding_days for t in all_trades if t.holding_days is not None]
    avg_hold = np.mean(hold_days) if hold_days else 0
    
    return BacktestResult(
        equity_curve=equity_series,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        realized_vol=realized_vol,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_return_per_trade=avg_return,
        avg_holding_days=avg_hold,
        trades=all_trades,
        entries_blocked=entries_blocked,
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


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def calculate_yearly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """Calculate yearly returns from equity curve."""
    # Resample to yearly, taking last value of each year
    yearly_equity = equity_curve.resample('YE').last()
    
    # Calculate returns
    yearly_returns = yearly_equity.pct_change()
    
    # First year: calculate from initial capital
    if len(yearly_equity) > 0:
        first_year = yearly_equity.index[0].year
        first_return = (yearly_equity.iloc[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL
        yearly_returns.iloc[0] = first_return
    
    return yearly_returns


def display_results(result: BacktestResult, bh_curve: pd.Series, dates: List):
    """Display backtest results."""
    
    # B&H metrics
    bh_total_return = (bh_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    bh_annual = (1 + bh_total_return) ** (1 / years) - 1
    
    bh_returns = bh_curve.pct_change().dropna()
    bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(365) if bh_returns.std() > 0 else 0
    
    bh_peak = bh_curve.expanding().max()
    bh_dd = ((bh_peak - bh_curve) / bh_peak).max()
    
    print("\n" + "=" * 80)
    print("PAIRS REGIME TRADING BACKTEST RESULTS")
    print("=" * 80)
    
    print(f"""
    ┌────────────────────────┬─────────────────┬─────────────────┐
    │  METRIC                │  STRATEGY       │  BUY & HOLD     │
    ├────────────────────────┼─────────────────┼─────────────────┤
    │  Total Return          │  {result.total_return*100:>+12.0f}%  │  {bh_total_return*100:>+12.0f}%  │
    │  Annual Return         │  {result.annual_return*100:>+12.1f}%  │  {bh_annual*100:>+12.1f}%  │
    │  Sharpe Ratio          │  {result.sharpe_ratio:>14.2f}  │  {bh_sharpe:>14.2f}  │
    │  Max Drawdown          │  {result.max_drawdown*100:>13.1f}%  │  {bh_dd*100:>13.1f}%  │
    │  Calmar Ratio          │  {result.calmar_ratio:>14.2f}  │  {bh_annual/bh_dd if bh_dd > 0 else 0:>14.2f}  │
    │  Realized Volatility   │  {result.realized_vol*100:>13.1f}%  │                 │
    ├────────────────────────┼─────────────────┼─────────────────┤
    │  Final Equity          │  ${result.equity_curve.iloc[-1]:>12,.0f}  │  ${bh_curve.iloc[-1]:>12,.0f}  │
    └────────────────────────┴─────────────────┴─────────────────┘
    """)
    
    # Yearly breakdown
    print("  YEARLY RETURNS")
    print("  " + "-" * 60)
    
    strategy_yearly = calculate_yearly_returns(result.equity_curve)
    bh_yearly = calculate_yearly_returns(bh_curve)
    
    print(f"    {'Year':<8} {'Strategy':>12} {'Buy & Hold':>12} {'Difference':>12}")
    print("    " + "-" * 48)
    
    for year_date in strategy_yearly.index:
        year = year_date.year
        strat_ret = strategy_yearly.loc[year_date]
        bh_ret = bh_yearly.loc[year_date] if year_date in bh_yearly.index else 0
        diff = strat_ret - bh_ret
        
        if not pd.isna(strat_ret):
            print(f"    {year:<8} {strat_ret*100:>+11.1f}% {bh_ret*100:>+11.1f}% {diff*100:>+11.1f}%")
    
    # Trade statistics
    print("\n  TRADE STATISTICS")
    print("  " + "-" * 60)
    print(f"    Total Trades:           {result.n_trades:>6}")
    print(f"    Win Rate:               {result.win_rate*100:>6.1f}%")
    print(f"    Avg Return per Trade:   {result.avg_return_per_trade*100:>6.2f}%")
    print(f"    Avg Holding Days:       {result.avg_holding_days:>6.1f}")
    print(f"    Entries Blocked:        {result.entries_blocked:>6}")
    
    # By exit reason
    converged = [t for t in result.trades if t.exit_reason == "CONVERGED"]
    max_hold = [t for t in result.trades if t.exit_reason == "MAX_HOLD"]
    
    print("\n  BY EXIT REASON")
    print("  " + "-" * 60)
    
    if converged:
        conv_returns = [t.spread_return for t in converged if t.spread_return is not None]
        conv_win = sum(1 for r in conv_returns if r > 0) / len(conv_returns) if conv_returns else 0
        print(f"    CONVERGED: {len(converged):>5} trades, win rate: {conv_win*100:.1f}%, avg return: {np.mean(conv_returns)*100:.2f}%")
    
    if max_hold:
        mh_returns = [t.spread_return for t in max_hold if t.spread_return is not None]
        mh_win = sum(1 for r in mh_returns if r > 0) / len(mh_returns) if mh_returns else 0
        print(f"    MAX_HOLD:  {len(max_hold):>5} trades, win rate: {mh_win*100:.1f}%, avg return: {np.mean(mh_returns)*100:.2f}%")
    
    # Position size distribution
    sizes = [t.position_size for t in result.trades]
    if sizes:
        print("\n  POSITION SIZE DISTRIBUTION")
        print("  " + "-" * 60)
        print(f"    Min:  {min(sizes)*100:>5.1f}%")
        print(f"    Max:  {max(sizes)*100:>5.1f}%")
        print(f"    Avg:  {np.mean(sizes)*100:>5.1f}%")
    
    # Configuration
    print("\n  EXPOSURE & RISK SETTINGS")
    print("  " + "-" * 60)
    print(f"    Max Exposure:        {MAX_EXPOSURE*100:.0f}% ({MAX_EXPOSURE:.1f}x leverage)")
    print(f"    Drawdown Protection: Starts at {abs(DD_START_REDUCE)*100:.0f}%, min floor at {abs(DD_MIN_EXPOSURE)*100:.0f}%")
    print(f"    Min Exposure Floor:  {MIN_EXPOSURE_FLOOR*100:.0f}% of max")
    print(f"    Base Position Size:  {BASE_POSITION_SIZE*100:.0f}% per leg")
    print(f"    Max Hold Days:       {MAX_HOLD_DAYS}")
    print(f"    Entry Divergence:    {ENTRY_DIVERGENCE} (enter when divergence >= this)")
    print(f"    Exit Divergence:     {EXIT_DIVERGENCE} (exit when divergence < this)")
    print(f"    Min Trade Size:      ${MIN_TRADE_SIZE:.0f}")


def plot_results(result: BacktestResult, bh_curve: pd.Series, db: Database, 
                 filename: str = 'pairs_backtest.png'):
    """Generate charts with Bitcoin price overlay."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        # Load Bitcoin price for reference
        try:
            btc_data = db.get_ohlcv('XBTUSD')
            btc_24h = btc_data.resample('24h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 
                'close': 'last', 'volume': 'sum'
            }).dropna()
            btc_price = btc_24h['close']
            # Align to strategy dates
            btc_price = btc_price.reindex(result.equity_curve.index, method='ffill')
            has_btc = True
        except:
            has_btc = False
        
        # Panel 1: Equity curves (log scale)
        ax1 = axes[0]
        ax1.plot(result.equity_curve.index, result.equity_curve.values, 
                 label='Pairs Strategy', color='#B7791F', linewidth=2)
        ax1.plot(bh_curve.index, bh_curve.values, 
                 label='Buy & Hold (6 Alts)', color='#C53030', linewidth=2, linestyle='--')
        ax1.set_title('Pairs Regime Trading Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Panel 2: Bitcoin price (log scale)
        ax2 = axes[1]
        if has_btc:
            ax2.plot(btc_price.index, btc_price.values, color='#F6AD55', linewidth=2)
            ax2.set_title('Bitcoin Price (BTC/USD)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Price ($)')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            # Add market regime shading
            btc_ma200 = btc_price.rolling(200).mean()
            bull_mask = btc_price > btc_ma200
            ax2.fill_between(btc_price.index, btc_price.min(), btc_price.max(),
                           where=bull_mask, alpha=0.1, color='green', label='Bull (BTC > MA200)')
            ax2.fill_between(btc_price.index, btc_price.min(), btc_price.max(),
                           where=~bull_mask, alpha=0.1, color='red', label='Bear (BTC < MA200)')
            ax2.legend(loc='upper left')
        else:
            ax2.text(0.5, 0.5, 'Bitcoin data not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
        
        # Panel 3: Drawdown
        ax3 = axes[2]
        equity = result.equity_curve
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak * 100
        ax3.fill_between(drawdown.index, drawdown.values, 0, color='#C53030', alpha=0.3)
        ax3.plot(drawdown.index, drawdown.values, color='#C53030', linewidth=1)
        ax3.set_title('Strategy Drawdown', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_ylim(top=0)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Trade returns histogram
        ax4 = axes[3]
        returns = [t.spread_return*100 for t in result.trades if t.spread_return is not None]
        ax4.hist(returns, bins=50, color='#2C5282', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.axvline(x=np.mean(returns), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        ax4.set_title('Trade Returns Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
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
    print("=" * 80, flush=True)
    print("PAIRS REGIME TRADING STRATEGY BACKTEST", flush=True)
    print("=" * 80, flush=True)
    
    print(f"""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  STRATEGY CONFIGURATION                                            ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  • 32-state regime classification per asset                        ║
    ║  • Entry: State divergence ≥ {ENTRY_DIVERGENCE}                                      ║
    ║  • Exit: Divergence < {EXIT_DIVERGENCE} OR {MAX_HOLD_DAYS}-day max hold                        ║
    ║  • Position sizing: Inverse volatility ({BASE_POSITION_SIZE*100:.0f}% base per leg)       ║
    ║  • Max exposure: {MAX_EXPOSURE*100:.0f}% ({MAX_EXPOSURE:.1f}x leverage)                               ║
    ║  • Drawdown protection: Reduces exposure from {abs(DD_START_REDUCE)*100:.0f}% to {abs(DD_MIN_EXPOSURE)*100:.0f}% DD   ║
    ║  • Transaction costs: {TRADING_COST*100:.2f}% per trade per leg                   ║
    ╚════════════════════════════════════════════════════════════════════╝
    """, flush=True)
    
    print("  Connecting to database...", flush=True)
    db = Database()
    print("  Connected.", flush=True)
    
    # Load data
    data = load_all_data(db)
    
    # Calculate state history
    state_df, hitrate_df, dates = calculate_state_history(data)
    
    # Create spread tracker
    spread_tracker = SpreadStatsTracker(data)
    
    print(f"\n  Trading period: {dates[0].date()} to {dates[-1].date()}", flush=True)
    
    # Run backtest
    result = run_backtest(data, state_df, hitrate_df, spread_tracker, dates)
    
    # Calculate benchmark
    print("\n  Calculating buy & hold benchmark...", flush=True)
    bh_curve = calculate_buy_and_hold(data, dates)
    
    # Display results
    display_results(result, bh_curve, dates)
    
    # Plot
    plot_results(result, bh_curve, db)
    
    print("\n" + "=" * 80, flush=True)
    print("BACKTEST COMPLETE", flush=True)
    print("=" * 80, flush=True)
    
    return result, bh_curve


if __name__ == "__main__":
    result, bh_curve = main()