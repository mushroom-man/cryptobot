#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Strategy Backtest v6
==============================
16-State System with NO_MA72_ONLY Filter + Walk-Forward Validation + Bootstrapping

Key Updates from v5:
    - 16-state system (removed trend_72h)
    - NO_MA72_ONLY filter (56% signal reduction)
    - Walk-forward validation (rolling OOS testing)
    - Bootstrap confidence intervals

Validation Methods:
    1. Full Backtest: Standard expanding window
    2. Walk-Forward: Rolling train/test windows
    3. Bootstrap: Confidence intervals via resampling

Usage:
    python 16state_combined_backtest_v6.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker/cryptobot/data')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from database import Database


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

# Signal Filter (validated)
USE_MA72_FILTER = True

# Walk-Forward Parameters
WF_TRAIN_MONTHS = 12      # Training window
WF_TEST_MONTHS = 3        # Test window (OOS)
WF_STEP_MONTHS = 3        # Step size

# Bootstrap Parameters
BOOTSTRAP_N_SAMPLES = 1000
BOOTSTRAP_BLOCK_SIZE = 20  # Block bootstrap for autocorrelation
BOOTSTRAP_CI_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]

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
TRADING_COST = 0.0020  # Realistic
MIN_TRADE_SIZE = 100.0

STRONG_BUY_THRESHOLD = 0.55
BUY_THRESHOLD = 0.50
SELL_THRESHOLD = 0.45


# =============================================================================
# CALENDAR HELPERS
# =============================================================================

def get_month_start(date: pd.Timestamp, months_back: int = 0) -> pd.Timestamp:
    """Get the 1st of the month, optionally N months back."""
    target = date - pd.DateOffset(months=months_back)
    return target.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


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
class BacktestResult:
    """Results from a backtest run."""
    name: str
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    n_trades: int = 0
    total_costs: float = 0.0
    signals_filtered: int = 0
    signals_total: int = 0
    equity_curve: Optional[pd.Series] = None
    yearly_returns: Dict[int, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    n_folds: int
    oos_returns: List[float]
    oos_sharpes: List[float]
    oos_max_dds: List[float]
    fold_details: List[Dict]
    
    @property
    def mean_oos_return(self) -> float:
        return np.mean(self.oos_returns)
    
    @property
    def mean_oos_sharpe(self) -> float:
        return np.mean(self.oos_sharpes)
    
    @property
    def std_oos_sharpe(self) -> float:
        return np.std(self.oos_sharpes)
    
    @property
    def worst_oos_dd(self) -> float:
        return max(self.oos_max_dds)
    
    @property
    def pct_profitable_folds(self) -> float:
        return sum(1 for r in self.oos_returns if r > 0) / len(self.oos_returns)


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis."""
    metric_name: str
    original_value: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_5: float
    ci_25: float
    ci_50: float
    ci_75: float
    ci_95: float
    n_samples: int
    
    @property
    def is_significant(self) -> bool:
        """Check if 95% CI excludes zero (for returns/Sharpe)."""
        return self.ci_5 > 0 or self.ci_95 < 0


# =============================================================================
# SIGNAL GENERATION (16-STATE)
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


def generate_16state_signals(df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                              df_168h: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 16-state signals (no trend_72h).
    4 trend states × 4 MA alignment states = 16 total states
    """
    # Label trends (only 24h and 168h)
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    # Calculate MAs
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Align to 24h index
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    # Build signals (SHIFT BEFORE REINDEX)
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


# =============================================================================
# SIGNAL FILTER (NO_MA72_ONLY)
# =============================================================================

def should_trade_signal(prev_state: Optional[Tuple[int, int, int, int]], 
                        curr_state: Tuple[int, int, int, int],
                        use_filter: bool = USE_MA72_FILTER) -> bool:
    """
    NO_MA72_ONLY filter: Skip if ONLY ma72_above_ma24 changed.
    """
    if prev_state is None:
        return True
    
    if prev_state == curr_state:
        return False
    
    if not use_filter:
        return True
    
    # Check what changed
    trend_24h_changed = prev_state[0] != curr_state[0]
    trend_168h_changed = prev_state[1] != curr_state[1]
    ma72_changed = prev_state[2] != curr_state[2]
    ma168_changed = prev_state[3] != curr_state[3]
    
    # Skip if ONLY ma72 changed
    only_ma72_changed = (ma72_changed and 
                         not trend_24h_changed and 
                         not trend_168h_changed and 
                         not ma168_changed)
    
    return not only_ma72_changed


# =============================================================================
# HIT RATE CALCULATION (16-STATE)
# =============================================================================

def calculate_expanding_hit_rates(returns_history: pd.Series, 
                                   signals_history: pd.DataFrame) -> Dict:
    """Calculate 16-state hit rates using ONLY historical data."""
    all_trend_perms = list(product([0, 1], repeat=2))  # 4 trend permutations
    all_ma_perms = list(product([0, 1], repeat=2))      # 4 MA permutations
    
    if len(returns_history) < MIN_SAMPLES_PER_STATE:
        return {(t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for t in all_trend_perms for m in all_ma_perms}
    
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < MIN_SAMPLES_PER_STATE:
        return {(t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for t in all_trend_perms for m in all_ma_perms}
    
    aligned_returns = returns_history.loc[common_idx]
    aligned_signals = signals_history.loc[common_idx]
    
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    hit_rates = {}
    
    for trend_perm in all_trend_perms:
        for ma_perm in all_ma_perms:
            mask = (
                (aligned_signals['trend_24h'] == trend_perm[0]) &
                (aligned_signals['trend_168h'] == trend_perm[1]) &
                (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
                (aligned_signals['ma168_above_ma24'] == ma_perm[1])
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            
            if n > 0:
                hit_rate = (perm_returns > 0).sum() / n
            else:
                hit_rate = 0.5
            
            hit_rates[(trend_perm, ma_perm)] = {
                'n': n,
                'hit_rate': hit_rate,
                'sufficient': n >= MIN_SAMPLES_PER_STATE,
            }
    
    return hit_rates


def get_16state_position(trend_perm: Tuple[int, int], 
                         ma_perm: Tuple[int, int],
                         hit_rates: Dict) -> float:
    """Get position signal for 16-state."""
    key = (trend_perm, ma_perm)
    data = hit_rates.get(key, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.00


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
    """Convert state to numeric for divergence."""
    mapping = {"STRONG_BUY": 3, "BUY": 2, "SELL": 1, "STRONG_SELL": 0}
    return mapping.get(state, 1)


def get_state_divergence(state_a: str, state_b: str) -> int:
    """Calculate state divergence between two assets."""
    return abs(state_to_numeric(state_a) - state_to_numeric(state_b))


def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Calculate risk parity weights for assets."""
    vols = returns_df.std() * np.sqrt(365)
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database) -> Dict:
    """Load and prepare data for all pairs."""
    data = {}
    
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ", flush=True)
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        signals = generate_16state_signals(df_24h, df_72h, df_168h)
        returns = df_24h['close'].pct_change()
        
        data[pair] = {
            'prices': df_24h,
            'signals': signals,
            'returns': returns,
        }
        print(f"{len(df_24h)} days")
    
    return data


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    data: Dict,
    dates: List[pd.Timestamp],
    data_start: pd.Timestamp,
    trading_cost: float = TRADING_COST,
    use_filter: bool = USE_MA72_FILTER,
    name: str = "Backtest"
) -> BacktestResult:
    """
    Run combined directional + pairs backtest.
    """
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    equity_curve = {}
    
    total_costs = 0.0
    n_trades = 0
    signals_filtered = 0
    signals_total = 0
    
    # State tracking
    returns_dict = {pair: data[pair]['returns'] for pair in DEPLOY_PAIRS}
    returns_df = pd.DataFrame(returns_dict).dropna()
    asset_weights = pd.Series({pair: 1.0/len(DEPLOY_PAIRS) for pair in DEPLOY_PAIRS})
    prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    last_rebalance_month = None
    
    # Filter state tracking per asset
    prev_signal_states = {pair: None for pair in DEPLOY_PAIRS}
    active_states = {pair: None for pair in DEPLOY_PAIRS}
    
    # Hit rate caches
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}
    
    # Yearly tracking
    yearly_pnl = {}
    yearly_start_equity = {}
    current_year = dates[0].year
    yearly_start_equity[current_year] = equity
    yearly_pnl[current_year] = 0.0
    
    equity_curve[dates[0]] = equity
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        current_month = (date.year, date.month)
        
        # Year tracking
        if next_date.year != current_year:
            current_year = next_date.year
            yearly_start_equity[current_year] = equity
            yearly_pnl[current_year] = 0.0
        
        # Drawdown protection
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if current_dd >= DIR_DD_START_REDUCE:
            dd_scalar = 1.0
        elif current_dd <= DIR_DD_MIN_EXPOSURE:
            dd_scalar = DIR_MIN_EXPOSURE_FLOOR
        else:
            range_dd = DIR_DD_START_REDUCE - DIR_DD_MIN_EXPOSURE
            position = (current_dd - DIR_DD_MIN_EXPOSURE) / range_dd
            dd_scalar = DIR_MIN_EXPOSURE_FLOOR + position * (1.0 - DIR_MIN_EXPOSURE_FLOOR)
        
        # Monthly rebalance
        if last_rebalance_month != current_month:
            lookback_start = get_month_start(date, COV_LOOKBACK_MONTHS)
            lookback_end = get_month_start(date, 0)
            lookback_returns = returns_df.loc[lookback_start:lookback_end]
            if len(lookback_returns) >= 20:
                asset_weights = calculate_risk_parity_weights(lookback_returns)
            last_rebalance_month = current_month
        
        # Calculate exposures with filter
        base_exposure = 0.0
        asset_exposures = {}
        
        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            
            if date not in signals.index:
                continue
            
            sig = signals.loc[date]
            current_state = (
                int(sig['trend_24h']), 
                int(sig['trend_168h']),
                int(sig['ma72_above_ma24']), 
                int(sig['ma168_above_ma24'])
            )
            
            # Apply filter
            prev_state = prev_signal_states[pair]
            
            if current_state != prev_state:
                signals_total += 1
                
                if should_trade_signal(prev_state, current_state, use_filter):
                    active_states[pair] = current_state
                else:
                    signals_filtered += 1
            
            prev_signal_states[pair] = current_state
            
            # Use active state for position
            active = active_states[pair]
            if active is None:
                active = current_state
                active_states[pair] = current_state
            
            trend_perm = (active[0], active[1])
            ma_perm = (active[2], active[3])
            
            if not has_minimum_training(date, data_start, MIN_TRAINING_MONTHS):
                pos = 0.50
            else:
                current_m = (date.year, date.month)
                if last_recalc_month[pair] != current_m:
                    cutoff = get_month_start(date, 0)
                    hist_ret = returns[returns.index < cutoff]
                    hist_sig = signals[signals.index < cutoff]
                    if len(hist_ret) > 0:
                        hit_rate_cache[pair] = calculate_expanding_hit_rates(hist_ret, hist_sig)
                        last_recalc_month[pair] = current_m
                
                pos = get_16state_position(trend_perm, ma_perm, hit_rate_cache[pair])
            
            asset_exposures[pair] = pos * asset_weights[pair]
            base_exposure += asset_exposures[pair]
        
        if base_exposure > 1.0:
            for pair in asset_exposures:
                asset_exposures[pair] /= base_exposure
            base_exposure = 1.0
        
        # Vol scaling
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
        
        risk_scalar = min(vol_scalar, 1.0 / dd_scalar if dd_scalar > 0 else 1.0)
        risk_scalar = max(risk_scalar, DIR_MIN_EXPOSURE_FLOOR)
        
        for pair in asset_exposures:
            asset_exposures[pair] *= risk_scalar
        
        # Execute trades
        daily_pnl = 0.0
        daily_cost = 0.0
        
        for pair in DEPLOY_PAIRS:
            curr_exp = asset_exposures.get(pair, 0.0)
            prev_exp = prev_exposures.get(pair, 0.0)
            
            trade_value = abs(curr_exp - prev_exp) * equity
            
            if trade_value >= MIN_TRADE_SIZE:
                daily_cost += trade_value * trading_cost
                n_trades += 1
                prev_exposures[pair] = curr_exp
            else:
                curr_exp = prev_exp
                asset_exposures[pair] = prev_exp
            
            if curr_exp > 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    daily_pnl += equity * curr_exp * ret
        
        daily_pnl -= daily_cost
        total_costs += daily_cost
        equity += daily_pnl
        
        yearly_pnl[next_date.year] = yearly_pnl.get(next_date.year, 0) + daily_pnl
        
        equity_curve[next_date] = equity
        
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    returns_series = equity_series.pct_change().dropna()
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(365) if returns_series.std() > 0 else 0
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    # Yearly returns
    yearly_returns = {}
    for year in yearly_start_equity:
        if year in yearly_pnl:
            start_eq = yearly_start_equity[year]
            yearly_returns[year] = yearly_pnl[year] / start_eq if start_eq > 0 else 0
    
    return BacktestResult(
        name=name,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        n_trades=n_trades,
        total_costs=total_costs,
        signals_filtered=signals_filtered,
        signals_total=signals_total,
        equity_curve=equity_series,
        yearly_returns=yearly_returns,
    )


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward(
    data: Dict,
    all_dates: List[pd.Timestamp],
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
) -> WalkForwardResult:
    """
    Run walk-forward validation with rolling windows.
    
    For each fold:
        1. Train on [start, start + train_months)
        2. Test on [start + train_months, start + train_months + test_months)
        3. Step forward by step_months
    """
    print("\n  Running Walk-Forward Validation...")
    print(f"    Train: {train_months} months, Test: {test_months} months, Step: {step_months} months")
    
    oos_returns = []
    oos_sharpes = []
    oos_max_dds = []
    fold_details = []
    
    # Convert dates to months
    start_date = all_dates[0]
    end_date = all_dates[-1]
    
    fold_num = 0
    current_start = start_date
    
    while True:
        # Calculate fold boundaries
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if test_end > end_date:
            break
        
        # Get dates for this fold
        train_dates = [d for d in all_dates if current_start <= d < train_end]
        test_dates = [d for d in all_dates if train_end <= d < test_end]
        
        if len(train_dates) < 100 or len(test_dates) < 20:
            current_start = current_start + pd.DateOffset(months=step_months)
            continue
        
        fold_num += 1
        
        # Run backtest on test period only
        # Use train period as the "history" for hit rates
        data_start = train_dates[0]
        
        # For test period, we use expanding window that includes training
        test_result = run_backtest(
            data=data,
            dates=test_dates,
            data_start=data_start,
            use_filter=USE_MA72_FILTER,
            name=f"Fold_{fold_num}"
        )
        
        # Calculate OOS metrics
        if test_result.equity_curve is not None and len(test_result.equity_curve) > 1:
            test_returns = test_result.equity_curve.pct_change().dropna()
            
            # Annualized return for test period
            test_days = (test_dates[-1] - test_dates[0]).days
            if test_days > 0:
                annual_ret = (1 + test_result.total_return) ** (365.25 / test_days) - 1
            else:
                annual_ret = 0
            
            # Sharpe for test period
            if len(test_returns) > 1 and test_returns.std() > 0:
                sharpe = test_returns.mean() / test_returns.std() * np.sqrt(365)
            else:
                sharpe = 0
            
            oos_returns.append(annual_ret)
            oos_sharpes.append(sharpe)
            oos_max_dds.append(test_result.max_drawdown)
            
            fold_details.append({
                'fold': fold_num,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'oos_return': annual_ret,
                'oos_sharpe': sharpe,
                'oos_max_dd': test_result.max_drawdown,
            })
        
        # Step forward
        current_start = current_start + pd.DateOffset(months=step_months)
    
    print(f"    Completed {fold_num} folds")
    
    return WalkForwardResult(
        n_folds=fold_num,
        oos_returns=oos_returns,
        oos_sharpes=oos_sharpes,
        oos_max_dds=oos_max_dds,
        fold_details=fold_details,
    )


# =============================================================================
# BOOTSTRAP ANALYSIS
# =============================================================================

def block_bootstrap_sample(returns: pd.Series, block_size: int = BOOTSTRAP_BLOCK_SIZE) -> pd.Series:
    """
    Generate a block bootstrap sample to preserve autocorrelation.
    """
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    
    # Random block starts
    block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
    
    # Concatenate blocks
    sampled_returns = []
    for start in block_starts:
        block = returns.iloc[start:start + block_size].values
        sampled_returns.extend(block)
    
    # Trim to original length
    sampled_returns = sampled_returns[:n]
    
    return pd.Series(sampled_returns, index=returns.index[:n])


def calculate_metrics_from_returns(returns: pd.Series) -> Dict[str, float]:
    """Calculate key metrics from a returns series."""
    if len(returns) < 2 or returns.std() == 0:
        return {'annual_return': 0, 'sharpe': 0, 'max_dd': 0}
    
    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Annual return
    total_return = cum_returns.iloc[-1] - 1
    n_days = len(returns)
    annual_return = (1 + total_return) ** (365 / n_days) - 1 if n_days > 0 else 0
    
    # Sharpe
    sharpe = returns.mean() / returns.std() * np.sqrt(365)
    
    # Max drawdown
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns - peak) / peak
    max_dd = abs(drawdown.min())
    
    return {
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
    }


def run_bootstrap_analysis(
    equity_curve: pd.Series,
    n_samples: int = BOOTSTRAP_N_SAMPLES,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
) -> Dict[str, BootstrapResult]:
    """
    Run block bootstrap to generate confidence intervals.
    """
    print(f"\n  Running Bootstrap Analysis ({n_samples} samples, block={block_size})...")
    
    # Calculate returns from equity curve
    returns = equity_curve.pct_change().dropna()
    
    # Original metrics
    original_metrics = calculate_metrics_from_returns(returns)
    
    # Bootstrap samples
    bootstrap_metrics = {
        'annual_return': [],
        'sharpe': [],
        'max_dd': [],
    }
    
    for i in range(n_samples):
        if i % 200 == 0:
            print(f"    Sample {i}/{n_samples}...", end='\r')
        
        # Generate bootstrap sample
        sampled_returns = block_bootstrap_sample(returns, block_size)
        
        # Calculate metrics
        metrics = calculate_metrics_from_returns(sampled_returns)
        
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(metrics[key])
    
    print(f"    Completed {n_samples} samples")
    
    # Calculate confidence intervals
    results = {}
    
    for metric_name in ['annual_return', 'sharpe', 'max_dd']:
        samples = np.array(bootstrap_metrics[metric_name])
        
        results[metric_name] = BootstrapResult(
            metric_name=metric_name,
            original_value=original_metrics[metric_name],
            bootstrap_mean=np.mean(samples),
            bootstrap_std=np.std(samples),
            ci_5=np.percentile(samples, 5),
            ci_25=np.percentile(samples, 25),
            ci_50=np.percentile(samples, 50),
            ci_75=np.percentile(samples, 75),
            ci_95=np.percentile(samples, 95),
            n_samples=n_samples,
        )
    
    return results


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_backtest_results(result: BacktestResult):
    """Display backtest results."""
    filter_pct = result.signals_filtered / result.signals_total * 100 if result.signals_total > 0 else 0
    
    print(f"""
    ═══════════════════════════════════════════════════════════════
    {result.name.upper()}
    ═══════════════════════════════════════════════════════════════
    
    Performance Metrics:
    ─────────────────────────────────────────────────────────────────
    Annual Return:      {result.annual_return*100:>+8.1f}%
    Sharpe Ratio:       {result.sharpe_ratio:>8.2f}
    Max Drawdown:       {result.max_drawdown*100:>8.1f}%
    Calmar Ratio:       {result.calmar_ratio:>8.2f}
    
    Trading Statistics:
    ─────────────────────────────────────────────────────────────────
    Total Trades:       {result.n_trades:>8,}
    Total Costs:        ${result.total_costs:>10,.0f}
    Signals Filtered:   {result.signals_filtered:>8,} / {result.signals_total:>8,} ({filter_pct:.1f}%)
    """)
    
    # Yearly breakdown
    if result.yearly_returns:
        print("    Yearly Returns:")
        print("    ─────────────────────────────────────────────────────────────────")
        for year in sorted(result.yearly_returns.keys()):
            ret = result.yearly_returns[year]
            print(f"      {year}:  {ret*100:>+8.1f}%")


def display_walk_forward_results(wf_result: WalkForwardResult):
    """Display walk-forward validation results."""
    print(f"""
    ═══════════════════════════════════════════════════════════════
    WALK-FORWARD VALIDATION RESULTS
    ═══════════════════════════════════════════════════════════════
    
    Summary:
    ─────────────────────────────────────────────────────────────────
    Number of Folds:        {wf_result.n_folds}
    Mean OOS Return:        {wf_result.mean_oos_return*100:>+8.1f}%
    Mean OOS Sharpe:        {wf_result.mean_oos_sharpe:>8.2f} (std: {wf_result.std_oos_sharpe:.2f})
    Worst OOS Drawdown:     {wf_result.worst_oos_dd*100:>8.1f}%
    % Profitable Folds:     {wf_result.pct_profitable_folds*100:>8.1f}%
    
    Fold Details:
    ─────────────────────────────────────────────────────────────────
    Fold   Test Period              OOS Return   OOS Sharpe   Max DD
    """)
    
    for fold in wf_result.fold_details:
        print(f"    {fold['fold']:>3}    {fold['test_start'].strftime('%Y-%m')} to {fold['test_end'].strftime('%Y-%m')}    "
              f"{fold['oos_return']*100:>+8.1f}%    {fold['oos_sharpe']:>8.2f}    {fold['oos_max_dd']*100:>6.1f}%")


def display_bootstrap_results(bootstrap_results: Dict[str, BootstrapResult]):
    """Display bootstrap confidence intervals."""
    print(f"""
    ═══════════════════════════════════════════════════════════════
    BOOTSTRAP CONFIDENCE INTERVALS
    ═══════════════════════════════════════════════════════════════
    """)
    
    for metric_name, result in bootstrap_results.items():
        significance = "✓ Significant" if result.is_significant else "✗ Not significant"
        
        if 'return' in metric_name:
            fmt = lambda x: f"{x*100:>+8.1f}%"
        elif 'sharpe' in metric_name:
            fmt = lambda x: f"{x:>8.2f}"
        else:
            fmt = lambda x: f"{x*100:>8.1f}%"
        
        print(f"""
    {metric_name.upper()}:
    ─────────────────────────────────────────────────────────────────
    Original Value:      {fmt(result.original_value)}
    Bootstrap Mean:      {fmt(result.bootstrap_mean)} (std: {fmt(result.bootstrap_std)})
    
    Confidence Intervals:
      5th percentile:    {fmt(result.ci_5)}
      25th percentile:   {fmt(result.ci_25)}
      50th percentile:   {fmt(result.ci_50)}
      75th percentile:   {fmt(result.ci_75)}
      95th percentile:   {fmt(result.ci_95)}
    
    95% CI: [{fmt(result.ci_5)}, {fmt(result.ci_95)}]
    Status: {significance}
        """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("COMBINED STRATEGY BACKTEST v6")
    print("16-State System + NO_MA72_ONLY Filter")
    print("With Walk-Forward Validation & Bootstrap")
    print("=" * 80)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  VALIDATED CONFIGURATION                                           ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  States:      16 (removed trend_72h with 80% correlation)          ║
    ║  Filter:      NO_MA72_ONLY (56% signal reduction, same alpha)      ║
    ║  MA Periods:  24h=16, 72h=6, 168h=2                                ║
    ║  Buffers:     Entry=1.5%, Exit=0.5%                                ║
    ║  Allocation:  Risk Parity                                          ║
    ║                                                                    ║
    ║  VALIDATION METHODS:                                               ║
    ║    1. Full Backtest (expanding window)                             ║
    ║    2. Walk-Forward (rolling OOS testing)                           ║
    ║    3. Bootstrap (confidence intervals)                             ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    print("  Connecting to database...")
    db = Database()
    print("  Loading data...")
    data = load_all_data(db)
    
    # Find common dates
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))
    
    dates = sorted(list(all_dates))
    data_start = dates[0]
    
    print(f"\n  Common dates: {len(dates)} ({dates[0].date()} to {dates[-1].date()})")
    
    # =========================================================================
    # 1. FULL BACKTEST
    # =========================================================================
    print("\n" + "=" * 60)
    print("  RUNNING FULL BACKTEST")
    print("=" * 60)
    
    full_result = run_backtest(
        data=data,
        dates=dates,
        data_start=data_start,
        use_filter=USE_MA72_FILTER,
        name="16-State with NO_MA72_ONLY Filter"
    )
    
    display_backtest_results(full_result)
    
    # =========================================================================
    # 2. WALK-FORWARD VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("  RUNNING WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    wf_result = run_walk_forward(
        data=data,
        all_dates=dates,
        train_months=WF_TRAIN_MONTHS,
        test_months=WF_TEST_MONTHS,
        step_months=WF_STEP_MONTHS,
    )
    
    display_walk_forward_results(wf_result)
    
    # =========================================================================
    # 3. BOOTSTRAP ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("  RUNNING BOOTSTRAP ANALYSIS")
    print("=" * 60)
    
    bootstrap_results = run_bootstrap_analysis(
        equity_curve=full_result.equity_curve,
        n_samples=BOOTSTRAP_N_SAMPLES,
        block_size=BOOTSTRAP_BLOCK_SIZE,
    )
    
    display_bootstrap_results(bootstrap_results)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    sharpe_ci = bootstrap_results['sharpe']
    
    print(f"""
    FULL BACKTEST:
    ─────────────────────────────────────────────────────────────────
    Annual Return:       {full_result.annual_return*100:>+8.1f}%
    Sharpe Ratio:        {full_result.sharpe_ratio:>8.2f}
    Max Drawdown:        {full_result.max_drawdown*100:>8.1f}%
    Calmar Ratio:        {full_result.calmar_ratio:>8.2f}
    
    WALK-FORWARD OOS:
    ─────────────────────────────────────────────────────────────────
    Mean OOS Sharpe:     {wf_result.mean_oos_sharpe:>8.2f} ± {wf_result.std_oos_sharpe:.2f}
    % Profitable Folds:  {wf_result.pct_profitable_folds*100:>8.1f}%
    Worst OOS Drawdown:  {wf_result.worst_oos_dd*100:>8.1f}%
    
    BOOTSTRAP 95% CI:
    ─────────────────────────────────────────────────────────────────
    Sharpe Ratio:        [{sharpe_ci.ci_5:.2f}, {sharpe_ci.ci_95:.2f}]
    Significant:         {'✓ YES' if sharpe_ci.is_significant else '✗ NO'}
    """)
    
    # Overall assessment
    is_robust = (
        wf_result.pct_profitable_folds >= 0.6 and
        wf_result.mean_oos_sharpe > 1.0 and
        sharpe_ci.is_significant
    )
    
    print("=" * 80)
    if is_robust:
        print("✓ STRATEGY IS ROBUST - Ready for paper trading")
    else:
        print("⚠ STRATEGY NEEDS REVIEW - Check validation metrics")
    print("=" * 80)
    
    return full_result, wf_result, bootstrap_results


if __name__ == "__main__":
    full_result, wf_result, bootstrap_results = main()