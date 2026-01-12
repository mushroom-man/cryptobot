#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directional 32-State Strategy Backtest
======================================
Standalone backtest for the directional 32-state trading strategy.

Features:
- 32-state regime classification (8 price states × 4 MA alignment)
- Expanding window hit rate calculation (no look-ahead bias)
- Risk parity asset weighting
- Volatility scaling (target 40% annual vol)
- Drawdown protection
- Transaction costs

Documented Performance: +66% annual, 1.99 Sharpe, -19% max DD

Usage:
    python directional_backtest.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


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
HIT_RATE_THRESHOLD = 0.50
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
VOL_LOOKBACK_MONTHS = 1           # Volatility estimation window (1 month)
COV_LOOKBACK_MONTHS = 2           # Covariance estimation window (2 months)
REBALANCE_MONTHS = 1              # Rebalance portfolio on 1st of each month

# =============================================================================
# EXPOSURE & RISK CONTROLS - ADJUST THESE TO YOUR COMFORT LEVEL
# =============================================================================
#
# MAX_EXPOSURE: Maximum total portfolio exposure as multiple of account equity
#
#   1.0 = No leverage (max 100% invested, recommended for most users)
#   1.5 = 50% leverage (max 150% exposure)
#   2.0 = 2x leverage (max 200% exposure, higher risk)
#
# Example with $100,000 account:
#   MAX_EXPOSURE = 1.0  →  Maximum $100,000 in positions
#   MAX_EXPOSURE = 1.5  →  Maximum $150,000 in positions
#   MAX_EXPOSURE = 2.0  →  Maximum $200,000 in positions
#
# RISK vs RETURN TRADEOFF (approximate):
#   MAX_EXPOSURE = 1.0  →  ~65% annual return, ~20% max drawdown
#   MAX_EXPOSURE = 1.5  →  ~95% annual return, ~30% max drawdown
#   MAX_EXPOSURE = 2.0  →  ~125% annual return, ~40% max drawdown
#
# =============================================================================

MAX_EXPOSURE = 2.0  # ← ADJUST THIS (1.0 = no leverage, conservative)

# Volatility targeting (scales exposure to maintain consistent risk)
TARGET_VOL = 0.40                 # Target 40% annualized portfolio volatility

# Drawdown protection (reduces exposure during losses)
DD_START_REDUCE = -0.20           # Start reducing at -20% drawdown
DD_MIN_EXPOSURE = -0.50           # Maximum drawdown before minimum exposure
MIN_EXPOSURE_FLOOR = 0.40         # Never go below 40% of target exposure

# Backtest parameters
INITIAL_CAPITAL = 100000.0
TRADING_COST = 0.0015             # Per trade (0.15% = fee + slippage)
MIN_TRADE_SIZE = 100.0            # Minimum trade value in dollars


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
class BacktestResult:
    """Results from backtest."""
    equity_curve: pd.Series
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    realized_vol: float
    trading_costs: float
    n_days: int


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
    """
    Generate 32-state signals.
    
    8 price states × 4 MA alignment states = 32 total states
    SHIFT BEFORE REINDEX to prevent look-ahead bias.
    """
    # Price vs MA signals
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    # MA values for alignment
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Align to 24h index
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    # Build aligned DataFrame - SHIFT BEFORE REINDEX (critical)
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


def calculate_expanding_hit_rates(returns_history: pd.Series, 
                                   signals_history: pd.DataFrame) -> Dict:
    """
    Calculate 32-state hit rates using ONLY historical data.
    No look-ahead bias.
    """
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
        return 0.50  # Skip
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00  # Invest
    else:
        return 0.00  # Avoid


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
# HIT RATE HISTORY CALCULATION
# =============================================================================

def calculate_position_history(data: Dict) -> Tuple[List, Dict]:
    """
    Calculate position signals for all assets over time.
    Returns: dates list, positions dict
    """
    # Find common dates
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))
    
    all_dates = sorted(list(all_dates))
    
    print(f"\n  Calculating position history for {len(all_dates)} common dates...", flush=True)
    print(f"  (Recalculating hit rates on 1st of each month)", flush=True)
    
    positions = {pair: {} for pair in DEPLOY_PAIRS}
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}
    
    # Get data start date for training check
    data_start = all_dates[0]
    
    last_progress = 0
    recalc_count = 0
    
    # Track state distribution
    invest_count = 0
    skip_count = 0
    avoid_count = 0
    
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
                positions[pair][date] = 0.50  # Skip during warmup
                skip_count += 1
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
            
            pos = get_32state_position(price_perm, ma_perm, hit_rate_cache[pair])
            positions[pair][date] = pos
            
            if pos == 1.0:
                invest_count += 1
            elif pos == 0.5:
                skip_count += 1
            else:
                avoid_count += 1
    
    print(f"    Progress: 100%", flush=True)
    print(f"    Total recalculations: {recalc_count}", flush=True)
    
    total_signals = invest_count + skip_count + avoid_count
    print(f"\n  Signal Distribution:", flush=True)
    print(f"    INVEST: {invest_count:>6} ({invest_count/total_signals*100:.1f}%)", flush=True)
    print(f"    SKIP:   {skip_count:>6} ({skip_count/total_signals*100:.1f}%)", flush=True)
    print(f"    AVOID:  {avoid_count:>6} ({avoid_count/total_signals*100:.1f}%)", flush=True)
    
    return all_dates, positions


# =============================================================================
# RISK MANAGEMENT
# =============================================================================

def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Calculate risk parity weights for assets (inverse volatility)."""
    vols = returns_df.std() * np.sqrt(365)
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return weights


# Note: Vol scaling and DD protection are now inlined in run_backtest()
# to match the original implementation exactly (using MARKET returns for vol scaling)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(data: Dict, positions: Dict, dates: List) -> BacktestResult:
    """Run the directional backtest."""
    
    print("\n  Running backtest...", flush=True)
    
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    equity_curve = {}
    
    # Risk parity weights for assets
    returns_dict = {pair: data[pair]['returns'] for pair in DEPLOY_PAIRS}
    returns_df = pd.DataFrame(returns_dict).dropna()
    asset_weights = pd.Series({pair: 1.0/len(DEPLOY_PAIRS) for pair in DEPLOY_PAIRS})
    
    prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    total_costs = 0.0
    last_rebalance_month = None
    
    # Record initial equity at start
    equity_curve[dates[0]] = equity
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        
        # Monthly rebalance of asset weights (on 1st of month or month change)
        current_month = (date.year, date.month)
        if last_rebalance_month != current_month:
            # Look back N calendar months for covariance estimation
            lookback_start = get_month_start(date, COV_LOOKBACK_MONTHS)
            lookback_end = get_month_start(date, 0)  # Start of current month
            lookback_returns = returns_df.loc[lookback_start:lookback_end]
            
            if len(lookback_returns) >= 20:  # Minimum data points
                asset_weights = calculate_risk_parity_weights(lookback_returns)
            last_rebalance_month = current_month
        
        # Calculate base exposure from signals
        base_exposure = 0.0
        asset_exposures = {}
        
        for pair in DEPLOY_PAIRS:
            if date in positions[pair]:
                pos = positions[pair][date]
                asset_exposures[pair] = pos * asset_weights[pair]
                base_exposure += asset_exposures[pair]
        
        # Normalize base exposure
        if base_exposure > 1.0:
            for pair in asset_exposures:
                asset_exposures[pair] /= base_exposure
            base_exposure = 1.0
        
        # ─────────────────────────────────────────────────────────────
        # VOLATILITY SCALING - Use MARKET returns, not strategy returns
        # This matches the original implementation
        # Lookback uses calendar month alignment
        # ─────────────────────────────────────────────────────────────
        vol_scalar = 1.0
        if date in returns_df.index:
            # Look back N calendar months for volatility estimation
            vol_lookback_start = get_month_start(date, VOL_LOOKBACK_MONTHS)
            vol_lookback_end = get_month_start(date, 0)  # Start of current month
            vol_returns = returns_df.loc[vol_lookback_start:vol_lookback_end]
            
            if len(vol_returns) >= 15:  # Minimum data points
                # Use equal-weight MARKET returns (not strategy returns!)
                market_returns = vol_returns.mean(axis=1)
                realized_vol = market_returns.std() * np.sqrt(365)
                if realized_vol > 0:
                    vol_scalar = TARGET_VOL / realized_vol
                    vol_scalar = np.clip(vol_scalar, MIN_EXPOSURE_FLOOR, MAX_EXPOSURE)
        
        # ─────────────────────────────────────────────────────────────
        # DRAWDOWN PROTECTION
        # Note: current_dd is NEGATIVE in original (equity - peak)/peak
        # ─────────────────────────────────────────────────────────────
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0  # NEGATIVE
        
        if current_dd >= DD_START_REDUCE:  # e.g., -0.10 >= -0.20
            dd_scalar = 1.0
        elif current_dd <= DD_MIN_EXPOSURE:  # e.g., -0.60 <= -0.50
            dd_scalar = MIN_EXPOSURE_FLOOR
        else:
            range_dd = DD_START_REDUCE - DD_MIN_EXPOSURE  # -0.20 - (-0.50) = 0.30
            position = (current_dd - DD_MIN_EXPOSURE) / range_dd
            dd_scalar = MIN_EXPOSURE_FLOOR + position * (1.0 - MIN_EXPOSURE_FLOOR)
        
        # Apply risk scalars to exposure
        risk_scalar = min(vol_scalar, dd_scalar)
        risk_scalar = max(risk_scalar, MIN_EXPOSURE_FLOOR)
        
        for pair in asset_exposures:
            asset_exposures[pair] *= risk_scalar
        
        # Calculate returns with costs (enforcing minimum trade size)
        daily_return = 0.0
        daily_cost = 0.0
        
        for pair in DEPLOY_PAIRS:
            curr_exp = asset_exposures.get(pair, 0.0)
            prev_exp = prev_exposures.get(pair, 0.0)
            
            # Calculate trade value
            trade_value = abs(curr_exp - prev_exp) * equity
            
            # Only execute if trade exceeds minimum size
            if trade_value >= MIN_TRADE_SIZE:
                daily_cost += trade_value * TRADING_COST
                prev_exposures[pair] = curr_exp  # Update exposure
            else:
                # Keep previous exposure (don't trade)
                curr_exp = prev_exp
                asset_exposures[pair] = prev_exp
            
            # Return based on actual exposure
            if curr_exp > 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    daily_return += curr_exp * ret
        
        pnl = equity * daily_return - daily_cost
        equity += pnl
        total_costs += daily_cost
        
        # FIX: Store equity on next_date (when the return was realized)
        equity_curve[next_date] = equity
        
        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
    
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
    
    return BacktestResult(
        equity_curve=equity_series,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        realized_vol=realized_vol,
        trading_costs=total_costs,
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
    print("DIRECTIONAL 32-STATE BACKTEST RESULTS")
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
    │  Trading Costs         │  ${result.trading_costs:>12,.0f}  │           $0    │
    │  Trading Days          │  {result.n_days:>14,}  │                 │
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
    
    print()
    print("  EXPOSURE & RISK SETTINGS:")
    print(f"    Max Exposure:        {MAX_EXPOSURE:.1f}x ({MAX_EXPOSURE*100:.0f}% max exposure)")
    print(f"    Target Volatility:   {TARGET_VOL*100:.0f}%")
    print(f"    DD Protection Start: {abs(DD_START_REDUCE)*100:.0f}%")
    print(f"    Min Exposure Floor:  {MIN_EXPOSURE_FLOOR*100:.0f}%")
    print(f"    Min Trade Size:      ${MIN_TRADE_SIZE:.0f}")


def plot_results(result: BacktestResult, bh_curve: pd.Series, db: Database, 
                 filename: str = 'directional_backtest.png'):
    """Generate equity curve chart with Bitcoin price overlay."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
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
                 label='Directional 32-State', color='#2C5282', linewidth=2)
        ax1.plot(bh_curve.index, bh_curve.values, 
                 label='Buy & Hold (6 Alts)', color='#C53030', linewidth=2, linestyle='--')
        ax1.set_title('Directional 32-State Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
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
            # Bull: BTC > 200-day MA, Bear: BTC < 200-day MA
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
        ax3.set_xlabel('Date')
        ax3.set_ylim(top=0)
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
    print("=" * 80, flush=True)
    print("DIRECTIONAL 32-STATE STRATEGY BACKTEST", flush=True)
    print("=" * 80, flush=True)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  STRATEGY CONFIGURATION                                            ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  • 32-state regime classification                                  ║
    ║  • Expanding window hit rate calculation (no look-ahead)           ║
    ║  • Risk parity asset weighting                                     ║
    ║  • Volatility scaling (target {TARGET_VOL*100:.0f}% annual vol)                      ║
    ║  • Drawdown protection ({DD_START_REDUCE*100:.0f}% to {DD_MIN_EXPOSURE*100:.0f}%)                              ║
    ║  • Max exposure: {MAX_EXPOSURE:.1f}x ({MAX_EXPOSURE*100:.0f}% max exposure)                          ║
    ║  • Transaction costs: 0.15% per trade                              ║
    ╚════════════════════════════════════════════════════════════════════╝
    """, flush=True)
    
    print("  Connecting to database...", flush=True)
    db = Database()
    print("  Connected.", flush=True)
    
    # Load data
    data = load_all_data(db)
    
    # Calculate position history
    dates, positions = calculate_position_history(data)
    
    print(f"\n  Trading period: {dates[0].date()} to {dates[-1].date()}", flush=True)
    
    # Run backtest
    result = run_backtest(data, positions, dates)
    
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