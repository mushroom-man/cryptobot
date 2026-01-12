#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward Validation: 24-State vs 32-State
==============================================
The GOLD STANDARD for strategy validation:

1. TRAIN on 2017-2020: Calculate hit rates
2. TEST on 2021-2025: Apply those FIXED hit rates (no updating)

If the strategy works out-of-sample, the signal is real.

Usage:
    python walk_forward_validation.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import product
import warnings
import time
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# Walk-Forward Periods
TRAIN_START = '2017-01-01'
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2025-12-31'

# Timezone-aware versions (for comparison with UTC data)
TRAIN_END_TZ = pd.Timestamp(TRAIN_END, tz='UTC')
TEST_START_TZ = pd.Timestamp(TEST_START, tz='UTC')

# MA Parameters
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_PERM = 20

# Risk Management
TARGET_VOL = 0.40
VOL_LOOKBACK = 30
DD_START_REDUCE = -0.20
DD_MIN_EXPOSURE = -0.50
MIN_EXPOSURE_FLOOR = 0.40
MAX_LEVERAGE = 1.0

# Portfolio
REBALANCE_DAYS = 30
COV_LOOKBACK = 60
TRADING_COST = 0.0015


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WalkForwardResult:
    """Walk-forward validation results."""
    name: str
    
    # Training period
    train_return: float
    train_sharpe: float
    train_max_dd: float
    
    # Test period (OUT OF SAMPLE)
    test_return: float
    test_sharpe: float
    test_max_dd: float
    test_calmar: float
    
    # Comparison
    degradation: float  # How much worse is test vs train
    
    # Details
    train_trades: int
    test_trades: int
    n_sufficient_states: int
    equity_curve: pd.Series


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
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
        ma_val = ma.iloc[i]
        
        if current == 1:
            if price < ma_val * (1 - exit_buffer) and price < ma_val * (1 - entry_buffer):
                current = 0
        else:
            if price > ma_val * (1 + exit_buffer) and price > ma_val * (1 + entry_buffer):
                current = 1
        
        labels.iloc[i] = current
    
    return labels


def calculate_ma_values(df_24h, df_72h, df_168h):
    """Calculate MA values aligned to 24h timeframe."""
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['ma_24h'] = ma_24h
    aligned['ma_72h'] = ma_72h.reindex(df_24h.index, method='ffill')
    aligned['ma_168h'] = ma_168h.reindex(df_24h.index, method='ffill')
    
    return aligned


def generate_8state_signals(df_24h, df_72h, df_168h):
    """Generate current 8-state signals (price vs MA)."""
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    
    return aligned.dropna().astype(int)


def generate_ma_order_signals(df_24h, df_72h, df_168h):
    """Generate MA ordering signals."""
    ma_vals = calculate_ma_values(df_24h, df_72h, df_168h)
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['ma72_above_ma24'] = (ma_vals['ma_72h'] > ma_vals['ma_24h']).astype(int)
    aligned['ma168_above_ma24'] = (ma_vals['ma_168h'] > ma_vals['ma_24h']).astype(int)
    
    aligned = aligned.shift(1)
    
    return aligned.dropna().astype(int)


def get_ma_order_category(ma72_above_ma24, ma168_above_ma24):
    """Classify MA order into 3 categories."""
    if ma72_above_ma24 == 0 and ma168_above_ma24 == 0:
        return 'GOLDEN'
    elif ma72_above_ma24 == 1 and ma168_above_ma24 == 1:
        return 'DEATH'
    else:
        return 'MIXED'


def calculate_vol_scalar(returns):
    """Calculate volatility scalar."""
    if len(returns) < VOL_LOOKBACK:
        return 1.0
    realized_vol = returns.iloc[-VOL_LOOKBACK:].std() * np.sqrt(365)
    if realized_vol <= 0:
        return 1.0
    return np.clip(TARGET_VOL / realized_vol, MIN_EXPOSURE_FLOOR, MAX_LEVERAGE)


def calculate_dd_scalar(current_dd):
    """Calculate drawdown scalar."""
    if current_dd >= DD_START_REDUCE:
        return 1.0
    elif current_dd <= DD_MIN_EXPOSURE:
        return MIN_EXPOSURE_FLOOR
    else:
        range_dd = DD_START_REDUCE - DD_MIN_EXPOSURE
        position = (current_dd - DD_MIN_EXPOSURE) / range_dd
        return MIN_EXPOSURE_FLOOR + position * (1.0 - MIN_EXPOSURE_FLOOR)


def calculate_risk_parity_weights(returns_df):
    """Calculate risk parity weights."""
    cov = returns_df.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return pd.Series(weights, index=returns_df.columns)


# =============================================================================
# HIT RATE CALCULATIONS (TRAINING PERIOD ONLY)
# =============================================================================

def calculate_24state_hit_rates_training(returns, signals_8state, signals_ma_order, train_end):
    """Calculate 24-state hit rates using ONLY training data."""
    all_price_perms = list(product([0, 1], repeat=3))
    ma_categories = ['GOLDEN', 'DEATH', 'MIXED']
    
    # Filter to training period only (handle timezone)
    if returns.index.tz is not None:
        train_end_cmp = pd.Timestamp(train_end).tz_localize('UTC') if train_end.tzinfo is None else train_end
    else:
        train_end_cmp = pd.Timestamp(train_end).tz_localize(None) if hasattr(train_end, 'tzinfo') and train_end.tzinfo else train_end
    
    train_mask = returns.index <= train_end_cmp
    returns_train = returns[train_mask]
    signals_8_train = signals_8state[signals_8state.index <= train_end]
    signals_ma_train = signals_ma_order[signals_ma_order.index <= train_end]
    
    common_idx = returns_train.index.intersection(signals_8_train.index).intersection(signals_ma_train.index)
    aligned_returns = returns_train.loc[common_idx]
    aligned_8state = signals_8_train.loc[common_idx]
    aligned_ma_order = signals_ma_train.loc[common_idx]
    forward_returns = aligned_returns.shift(-1)
    
    # Add MA category column
    ma_category = aligned_ma_order.apply(
        lambda row: get_ma_order_category(
            row['ma72_above_ma24'], 
            row['ma168_above_ma24']
        ), axis=1
    )
    
    hit_rates = {}
    
    for price_perm in all_price_perms:
        for ma_cat in ma_categories:
            mask = (
                (aligned_8state['trend_24h'] == price_perm[0]) &
                (aligned_8state['trend_72h'] == price_perm[1]) &
                (aligned_8state['trend_168h'] == price_perm[2]) &
                (ma_category == ma_cat)
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            hit_rate = (perm_returns > 0).sum() / n if n > 0 else 0.5
            
            hit_rates[(price_perm, ma_cat)] = {
                'n': n,
                'hit_rate': hit_rate,
                'sufficient': n >= MIN_SAMPLES_PER_PERM,
            }
    
    return hit_rates


def calculate_32state_hit_rates_training(returns, signals_8state, signals_ma_order, train_end):
    """Calculate 32-state hit rates using ONLY training data."""
    all_price_perms = list(product([0, 1], repeat=3))
    all_ma_perms = list(product([0, 1], repeat=2))
    
    # Filter to training period only (handle timezone)
    if returns.index.tz is not None:
        train_end_cmp = pd.Timestamp(train_end).tz_localize('UTC') if not hasattr(train_end, 'tzinfo') or train_end.tzinfo is None else train_end
    else:
        train_end_cmp = train_end
    
    train_mask = returns.index <= train_end_cmp
    returns_train = returns[train_mask]
    signals_8_train = signals_8state[signals_8state.index <= train_end]
    signals_ma_train = signals_ma_order[signals_ma_order.index <= train_end]
    
    common_idx = returns_train.index.intersection(signals_8_train.index).intersection(signals_ma_train.index)
    aligned_returns = returns_train.loc[common_idx]
    aligned_8state = signals_8_train.loc[common_idx]
    aligned_ma_order = signals_ma_train.loc[common_idx]
    forward_returns = aligned_returns.shift(-1)
    
    hit_rates = {}
    
    for price_perm in all_price_perms:
        for ma_perm in all_ma_perms:
            mask = (
                (aligned_8state['trend_24h'] == price_perm[0]) &
                (aligned_8state['trend_72h'] == price_perm[1]) &
                (aligned_8state['trend_168h'] == price_perm[2]) &
                (aligned_ma_order['ma72_above_ma24'] == ma_perm[0]) &
                (aligned_ma_order['ma168_above_ma24'] == ma_perm[1])
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            hit_rate = (perm_returns > 0).sum() / n if n > 0 else 0.5
            
            hit_rates[(price_perm, ma_perm)] = {
                'n': n,
                'hit_rate': hit_rate,
                'sufficient': n >= MIN_SAMPLES_PER_PERM,
            }
    
    return hit_rates


# =============================================================================
# WALK-FORWARD BACKTESTER
# =============================================================================

def run_walk_forward_backtest(db, system_type='24state') -> WalkForwardResult:
    """
    Run walk-forward validation:
    1. Calculate hit rates on TRAINING period (2017-2020)
    2. Apply FIXED hit rates to TEST period (2021-2025)
    """
    
    initial_capital = 100000.0
    train_end = TRAIN_END_TZ
    test_start = TEST_START_TZ
    
    # Load data
    prices = {}
    returns = {}
    signals_8state = {}
    signals_ma_order = {}
    hit_rates = {}
    
    print(f"\n    Loading data and calculating TRAINING hit rates...")
    
    for pair in DEPLOY_PAIRS:
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        prices[pair] = df_24h
        returns[pair] = df_24h['close'].pct_change()
        signals_8state[pair] = generate_8state_signals(df_24h, df_72h, df_168h)
        signals_ma_order[pair] = generate_ma_order_signals(df_24h, df_72h, df_168h)
        
        # Calculate hit rates on TRAINING period only
        if system_type == '24state':
            hit_rates[pair] = calculate_24state_hit_rates_training(
                returns[pair], signals_8state[pair], signals_ma_order[pair], train_end
            )
        elif system_type == '32state':
            hit_rates[pair] = calculate_32state_hit_rates_training(
                returns[pair], signals_8state[pair], signals_ma_order[pair], train_end
            )
    
    # Count sufficient states
    if system_type == '24state':
        n_sufficient = sum(1 for hr in hit_rates[DEPLOY_PAIRS[0]].values() if hr['sufficient'])
    else:
        n_sufficient = sum(1 for hr in hit_rates[DEPLOY_PAIRS[0]].values() if hr['sufficient'])
    
    returns_df = pd.DataFrame(returns).dropna()
    
    # Split into train and test periods
    train_dates = [d for d in returns_df.index if d <= train_end]
    test_dates = [d for d in returns_df.index if d >= test_start]
    
    print(f"    Training period: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"    Testing period:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
    print(f"    Sufficient states (from training): {n_sufficient}")
    
    # Run backtest on BOTH periods
    def run_period(dates, period_name):
        cash = initial_capital
        positions = {pair: 0.0 for pair in DEPLOY_PAIRS}
        current_weights = {pair: 0.0 for pair in DEPLOY_PAIRS}
        equity = initial_capital
        peak_equity = initial_capital
        
        total_trades = 0
        rp_weights = pd.Series(1/len(DEPLOY_PAIRS), index=DEPLOY_PAIRS)
        last_rebalance = None
        
        equity_history = []
        
        # Pre-calculate rolling vol
        port_returns_series = returns_df.mean(axis=1)
        rolling_vol = port_returns_series.rolling(VOL_LOOKBACK).std() * np.sqrt(365)
        rolling_vol = rolling_vol.fillna(TARGET_VOL)
        
        for i, date in enumerate(dates):
            # Update positions with returns
            for pair in DEPLOY_PAIRS:
                if positions[pair] > 0:
                    ret = returns_df.loc[date, pair]
                    positions[pair] *= (1 + ret)
            
            equity = cash + sum(positions.values())
            peak_equity = max(peak_equity, equity)
            current_dd = (equity - peak_equity) / peak_equity
            
            # Rebalance weights
            should_rebalance = (last_rebalance is None or 
                               (date - last_rebalance).days >= REBALANCE_DAYS)
            
            if should_rebalance:
                # Get trailing returns for cov calculation
                all_dates = returns_df.index.tolist()
                date_idx = all_dates.index(date)
                if date_idx >= COV_LOOKBACK:
                    trailing_returns = returns_df.iloc[date_idx-COV_LOOKBACK:date_idx]
                    rp_weights = calculate_risk_parity_weights(trailing_returns)
                last_rebalance = date
            
            # Get signals using FIXED hit rates from training
            target_signal = {}
            
            for pair in DEPLOY_PAIRS:
                sig_8 = signals_8state[pair]
                sig_ma = signals_ma_order[pair]
                
                # Get current signal
                prior_8 = sig_8[sig_8.index <= date]
                prior_ma = sig_ma[sig_ma.index <= date]
                
                if len(prior_8) > 0 and len(prior_ma) > 0:
                    s8 = prior_8.iloc[-1]
                    sma = prior_ma.iloc[-1]
                    price_perm = (int(s8['trend_24h']), int(s8['trend_72h']), int(s8['trend_168h']))
                    ma_perm = (int(sma['ma72_above_ma24']), int(sma['ma168_above_ma24']))
                else:
                    price_perm = (1, 1, 1)
                    ma_perm = (1, 1)
                
                # Look up FIXED hit rate from training
                if system_type == '24state':
                    ma_cat = get_ma_order_category(ma_perm[0], ma_perm[1])
                    hr_data = hit_rates[pair].get((price_perm, ma_cat), {'sufficient': False, 'hit_rate': 0.5})
                elif system_type == '32state':
                    hr_data = hit_rates[pair].get((price_perm, ma_perm), {'sufficient': False, 'hit_rate': 0.5})
                
                if not hr_data['sufficient']:
                    target_signal[pair] = 0.50
                elif hr_data['hit_rate'] > HIT_RATE_THRESHOLD:
                    target_signal[pair] = 1.00
                else:
                    target_signal[pair] = 0.00
            
            # Risk management
            vol_idx = returns_df.index.get_loc(date)
            realized_vol = rolling_vol.iloc[vol_idx]
            vol_scalar = TARGET_VOL / realized_vol if realized_vol > 0 else 1.0
            vol_scalar = np.clip(vol_scalar, MIN_EXPOSURE_FLOOR, MAX_LEVERAGE)
            
            dd_scalar = calculate_dd_scalar(current_dd)
            exposure = max(min(vol_scalar, dd_scalar), MIN_EXPOSURE_FLOOR)
            
            # Target weights
            target_weights = {
                pair: rp_weights[pair] * target_signal[pair] * exposure
                for pair in DEPLOY_PAIRS
            }
            
            # Execute trades
            for pair in DEPLOY_PAIRS:
                if abs(target_weights[pair] - current_weights[pair]) >= 0.01:
                    current_value = positions[pair]
                    target_value = equity * target_weights[pair]
                    trade_value = abs(target_value - current_value)
                    
                    if trade_value >= 100:
                        cost = trade_value * TRADING_COST
                        positions[pair] = target_value
                        cash -= (target_value - current_value) + cost
                        total_trades += 1
                        current_weights[pair] = target_weights[pair]
            
            equity = cash + sum(positions.values())
            equity_history.append(equity)
        
        # Calculate metrics
        equity_series = pd.Series(equity_history, index=dates)
        
        total_return = (equity - initial_capital) / initial_capital
        years = len(dates) / 365
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        daily_returns = equity_series.pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
        
        rolling_max = equity_series.expanding().max()
        max_dd = ((equity_series - rolling_max) / rolling_max).min()
        
        return {
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_trades': total_trades,
            'equity_curve': equity_series
        }
    
    # Run training period
    print(f"\n    Running TRAINING period backtest...")
    train_results = run_period(train_dates, "TRAIN")
    
    # Run test period
    print(f"    Running TEST period backtest (out-of-sample)...")
    test_results = run_period(test_dates, "TEST")
    
    # Calculate degradation
    degradation = (train_results['annual_return'] - test_results['annual_return']) / train_results['annual_return'] * 100
    
    # Combine equity curves
    full_equity = pd.concat([train_results['equity_curve'], test_results['equity_curve']])
    
    test_calmar = test_results['annual_return'] / abs(test_results['max_dd']) if test_results['max_dd'] != 0 else 0
    
    return WalkForwardResult(
        name=system_type.upper(),
        train_return=train_results['annual_return'],
        train_sharpe=train_results['sharpe'],
        train_max_dd=train_results['max_dd'],
        test_return=test_results['annual_return'],
        test_sharpe=test_results['sharpe'],
        test_max_dd=test_results['max_dd'],
        test_calmar=test_calmar,
        degradation=degradation,
        train_trades=train_results['total_trades'],
        test_trades=test_results['total_trades'],
        n_sufficient_states=n_sufficient,
        equity_curve=full_equity
    )


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_training_hit_rates(db, system_type='24state'):
    """Display hit rates calculated from training period."""
    train_end = TRAIN_END_TZ
    
    print(f"\n  {system_type.upper()} HIT RATES (Training Period Only: {TRAIN_START} to {TRAIN_END})")
    print("  " + "-" * 70)
    
    # Aggregate across pairs
    all_hit_rates = {}
    
    for pair in DEPLOY_PAIRS:
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        returns = df_24h['close'].pct_change()
        sig_8 = generate_8state_signals(df_24h, df_72h, df_168h)
        sig_ma = generate_ma_order_signals(df_24h, df_72h, df_168h)
        
        if system_type == '24state':
            hr = calculate_24state_hit_rates_training(returns, sig_8, sig_ma, train_end)
        else:
            hr = calculate_32state_hit_rates_training(returns, sig_8, sig_ma, train_end)
        
        for k, v in hr.items():
            if k not in all_hit_rates:
                all_hit_rates[k] = {'n': 0, 'hits': 0}
            all_hit_rates[k]['n'] += v['n']
            all_hit_rates[k]['hits'] += int(v['hit_rate'] * v['n'])
    
    # Sort and display top states
    sorted_hr = sorted(all_hit_rates.items(), 
                       key=lambda x: x[1]['hits']/x[1]['n'] if x[1]['n'] > 0 else 0, 
                       reverse=True)
    
    if system_type == '24state':
        print(f"\n  {'Price':<10} {'MA Order':<10} {'Hit Rate':>10} {'n':>8} {'Action':>10}")
        print("  " + "-" * 60)
        
        for (price_perm, ma_cat), data in sorted_hr[:12]:
            hr = data['hits'] / data['n'] if data['n'] > 0 else 0
            sufficient = data['n'] >= MIN_SAMPLES_PER_PERM * len(DEPLOY_PAIRS)
            
            p = '/'.join(['U' if x == 1 else 'D' for x in price_perm])
            action = "INVEST" if hr > 0.50 and sufficient else "AVOID" if sufficient else "SKIP"
            
            print(f"  {p:<10} {ma_cat:<10} {hr*100:>9.1f}% {data['n']:>8,} {action:>10}")
    else:
        print(f"\n  {'Price':<10} {'MA72>24':<8} {'MA168>24':<9} {'Hit Rate':>10} {'n':>8} {'Action':>10}")
        print("  " + "-" * 70)
        
        for (price_perm, ma_perm), data in sorted_hr[:12]:
            hr = data['hits'] / data['n'] if data['n'] > 0 else 0
            sufficient = data['n'] >= MIN_SAMPLES_PER_PERM * len(DEPLOY_PAIRS)
            
            p = '/'.join(['U' if x == 1 else 'D' for x in price_perm])
            m72 = 'Yes' if ma_perm[0] == 1 else 'No'
            m168 = 'Yes' if ma_perm[1] == 1 else 'No'
            action = "INVEST" if hr > 0.50 and sufficient else "AVOID" if sufficient else "SKIP"
            
            print(f"  {p:<10} {m72:<8} {m168:<9} {hr*100:>9.1f}% {data['n']:>8,} {action:>10}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("WALK-FORWARD VALIDATION: 24-STATE vs 32-STATE")
    print("=" * 100)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║  WALK-FORWARD VALIDATION - THE GOLD STANDARD                                  ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  TRAINING PERIOD:  {TRAIN_START} to {TRAIN_END}                               ║
    ║  TESTING PERIOD:   {TEST_START} to {TEST_END}                                 ║
    ║                                                                               ║
    ║  METHOD:                                                                      ║
    ║  1. Calculate hit rates using ONLY training data                              ║
    ║  2. Apply those FIXED hit rates to test period                                ║
    ║  3. NO updating of hit rates during test (true out-of-sample)                 ║
    ║                                                                               ║
    ║  If test performance is similar to training, the signal is REAL.              ║
    ║  If test performance degrades significantly, it's OVERFITTING.                ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    db = Database()
    
    # Display training hit rates
    print("\n" + "=" * 100)
    print("TRAINING PERIOD HIT RATES")
    print("=" * 100)
    
    display_training_hit_rates(db, '24state')
    display_training_hit_rates(db, '32state')
    
    # Run walk-forward validation
    print("\n" + "=" * 100)
    print("WALK-FORWARD BACKTEST")
    print("=" * 100)
    
    results = []
    
    for system in ['24state', '32state']:
        print(f"\n  Running {system.upper()} walk-forward validation...")
        start_time = time.time()
        result = run_walk_forward_backtest(db, system)
        elapsed = time.time() - start_time
        print(f"    Completed in {elapsed:.1f}s")
        results.append(result)
    
    # Display results
    print(f"""
    
    ═══════════════════════════════════════════════════════════════════════════════════════════════════
                                    WALK-FORWARD RESULTS
    ═══════════════════════════════════════════════════════════════════════════════════════════════════
    
                           TRAINING ({TRAIN_START} - {TRAIN_END})        TEST ({TEST_START} - {TEST_END})
    ┌────────────┬────────────┬──────────┬──────────┬────────────┬──────────┬──────────┬──────────────┐
    │  System    │  Annual %  │  Sharpe  │  Max DD  │  Annual %  │  Sharpe  │  Max DD  │  Degradation │
    ├────────────┼────────────┼──────────┼──────────┼────────────┼──────────┼──────────┼──────────────┤""")
    
    for r in results:
        deg_str = f"{r.degradation:+.0f}%" if r.degradation >= 0 else f"{r.degradation:.0f}%"
        print(f"    │  {r.name:<8} │  {r.train_return*100:>+8.1f}% │  {r.train_sharpe:>7.2f} │  {r.train_max_dd*100:>7.1f}% │  {r.test_return*100:>+8.1f}% │  {r.test_sharpe:>7.2f} │  {r.test_max_dd*100:>7.1f}% │  {deg_str:>12} │")
    
    print("    └────────────┴────────────┴──────────┴──────────┴────────────┴──────────┴──────────┴──────────────┘")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    r24 = results[0]
    r32 = results[1]
    
    print(f"""
    24-STATE ANALYSIS:
    ─────────────────────────────────────────
    Training:  {r24.train_return*100:+.1f}% annual, {r24.train_sharpe:.2f} Sharpe
    Test:      {r24.test_return*100:+.1f}% annual, {r24.test_sharpe:.2f} Sharpe
    
    Degradation: {r24.degradation:+.1f}%
    {"✓ SIGNAL IS REAL - Test performance holds up" if r24.degradation < 30 else "⚠ POSSIBLE OVERFITTING - Significant degradation"}
    
    32-STATE ANALYSIS:
    ─────────────────────────────────────────
    Training:  {r32.train_return*100:+.1f}% annual, {r32.train_sharpe:.2f} Sharpe
    Test:      {r32.test_return*100:+.1f}% annual, {r32.test_sharpe:.2f} Sharpe
    
    Degradation: {r32.degradation:+.1f}%
    {"✓ SIGNAL IS REAL - Test performance holds up" if r32.degradation < 30 else "⚠ POSSIBLE OVERFITTING - Significant degradation"}
    """)
    
    # Determine winner
    if r24.test_calmar > r32.test_calmar:
        winner = "24-STATE"
        winner_result = r24
    else:
        winner = "32-STATE"
        winner_result = r32
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║  WINNER (OUT-OF-SAMPLE): {winner:<15}                                       ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  Test Period Performance:                                                     ║
    ║  ────────────────────────                                                     ║
    ║  Annual Return:     {winner_result.test_return*100:>+8.1f}%                                             ║
    ║  Sharpe Ratio:      {winner_result.test_sharpe:>8.2f}                                               ║
    ║  Max Drawdown:      {winner_result.test_max_dd*100:>8.1f}%                                             ║
    ║  Calmar Ratio:      {winner_result.test_calmar:>8.2f}                                               ║
    ║                                                                               ║
    ║  Degradation from Training: {winner_result.degradation:>+6.1f}%                                       ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Final recommendation
    if r24.degradation < 30 and r32.degradation < 30:
        if r24.test_return > r32.test_return:
            rec = "24-STATE"
            reason = "Better out-of-sample returns with simpler model"
        else:
            rec = "32-STATE"
            reason = "Better out-of-sample returns"
    elif r24.degradation < r32.degradation:
        rec = "24-STATE"
        reason = "More robust out-of-sample (less degradation)"
    else:
        rec = "32-STATE"
        reason = "More robust out-of-sample (less degradation)"
    
    print(f"""
    FINAL RECOMMENDATION: {rec}
    
    Reason: {reason}
    
    INTERPRETATION:
    - Degradation < 30%: Signal is likely real and robust
    - Degradation 30-50%: Some overfitting, use with caution
    - Degradation > 50%: Significant overfitting, do not trust
    """)
    
    print("\n" + "=" * 100)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    results = main()