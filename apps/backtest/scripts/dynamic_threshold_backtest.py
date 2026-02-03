#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Threshold Backtest
==========================
Backtest various threshold levels for P(bearish flip) based dynamic strategies.

Tests:
1. BASELINE - 100% all bullish states
2. EXCLUDE_12_14 - Static exclusion (current best)
3. DYNAMIC_EXIT - Exit when P(flip 3h) > threshold
4. DYNAMIC_SIZING - Position = f(P(flip 3h))
5. HYBRID - Static exclusion + dynamic monitoring
6. STATE11_BOOST - Increase position in state 11 after 48h

Uses 3h horizon for all dynamic decisions.

Usage:
    python dynamic_threshold_backtest.py
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from cryptobot.data.database import Database
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    print("Warning: Database not available.")

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked)
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

# Buffers (validated)
ENTRY_BUFFER = 0.025
EXIT_BUFFER = 0.005

# Confirmation
CONFIRMATION_HOURS = 3

# Position sizing
KELLY_FRACTION = 0.25
INITIAL_CAPITAL = 100000

# State classification
BULLISH_STATES = set([8, 9, 10, 11, 12, 13, 14, 15])
BEARISH_STATES = set([0, 1, 2, 3, 4, 5, 6, 7])

# Duration buckets (must match hazard analysis)
DURATION_BUCKETS = [
    (0, 6),
    (6, 12),
    (12, 24),
    (24, 48),
    (48, 72),
    (72, 120),
    (120, 999),
]

# Horizon for decisions
DECISION_HORIZON = 3  # hours

# Thresholds to test
EXIT_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]  # P(flip) triggers exit
SIZING_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]  # For tiered sizing


# =============================================================================
# HAZARD MODEL PARAMETERS (from empirical_hazard_analysis.py)
# =============================================================================

# Hourly exit rate by state and duration bucket
# Format: {state: {bucket_idx: (exit_rate_per_hour, p_to_bearish)}}
HAZARD_TABLE = {
    8: {0: (0.05, 0.50), 1: (0.05, 0.50), 2: (0.05, 0.50), 3: (0.05, 0.50), 4: (0.05, 0.50), 5: (0.05, 0.50), 6: (0.05, 0.50)},
    9: {
        0: (0.0197, 0.315),   # 0-6h
        1: (0.0285, 0.263),   # 6-12h
        2: (0.0341, 0.213),   # 12-24h
        3: (0.0807, 0.120),   # 24-48h
        4: (0.0527, 0.183),   # 48-72h
        5: (0.0380, 0.275),   # 72-120h
        6: (0.0200, 0.333),   # 120h+
    },
    10: {0: (0.05, 0.50), 1: (0.05, 0.50), 2: (0.05, 0.50), 3: (0.05, 0.50), 4: (0.05, 0.50), 5: (0.05, 0.50), 6: (0.05, 0.50)},
    11: {
        0: (0.0143, 0.200),   # 0-6h
        1: (0.0252, 0.212),   # 6-12h
        2: (0.0230, 0.138),   # 12-24h
        3: (0.0278, 0.083),   # 24-48h
        4: (0.0332, 0.042),   # 48-72h
        5: (0.0233, 0.063),   # 72-120h
        6: (0.0303, 0.048),   # 120h+
    },
    12: {
        0: (0.0080, 0.708),   # 0-6h
        1: (0.0163, 0.735),   # 6-12h
        2: (0.0250, 0.686),   # 12-24h
        3: (0.0316, 0.617),   # 24-48h
        4: (0.0392, 0.748),   # 48-72h
        5: (0.0337, 0.828),   # 72-120h
        6: (0.0209, 0.444),   # 120h+
    },
    13: {
        0: (0.0075, 0.063),   # 0-6h
        1: (0.0211, 0.030),   # 6-12h
        2: (0.0200, 0.070),   # 12-24h
        3: (0.0801, 0.048),   # 24-48h
        4: (0.0495, 0.183),   # 48-72h
        5: (0.0412, 0.250),   # 72-120h
        6: (0.0491, 0.091),   # 120h+
    },
    14: {
        0: (0.0084, 0.521),   # 0-6h
        1: (0.0188, 0.588),   # 6-12h
        2: (0.0231, 0.519),   # 12-24h
        3: (0.0342, 0.241),   # 24-48h
        4: (0.0235, 0.370),   # 48-72h
        5: (0.0264, 0.241),   # 72-120h
        6: (0.0348, 0.444),   # 120h+
    },
    15: {
        0: (0.0085, 0.000),   # 0-6h
        1: (0.0195, 0.025),   # 6-12h
        2: (0.0187, 0.035),   # 12-24h
        3: (0.0216, 0.033),   # 24-48h
        4: (0.0235, 0.032),   # 48-72h
        5: (0.0169, 0.042),   # 72-120h
        6: (0.0276, 0.057),   # 120h+
    },
}


# =============================================================================
# HAZARD MODEL FUNCTIONS
# =============================================================================

def get_bucket_idx(duration_hours: int) -> int:
    """Get bucket index for a duration."""
    for i, (start, end) in enumerate(DURATION_BUCKETS):
        if start <= duration_hours < end:
            return i
    return len(DURATION_BUCKETS) - 1


def get_p_bearish_flip(state: int, duration_hours: int, horizon: int = 3) -> float:
    """
    Calculate P(bearish flip within horizon hours | state, duration).
    
    P(flip) = P(exit within horizon) × P(→bearish | exit)
    """
    if state not in BULLISH_STATES:
        return 0.0
    
    bucket_idx = get_bucket_idx(duration_hours)
    
    if state not in HAZARD_TABLE or bucket_idx not in HAZARD_TABLE[state]:
        return 0.5  # Default uncertain
    
    exit_rate_per_hour, p_to_bearish = HAZARD_TABLE[state][bucket_idx]
    
    # P(exit within horizon) = 1 - (1 - hourly_rate)^horizon
    p_exit = 1 - (1 - exit_rate_per_hour) ** horizon
    
    # P(bearish flip) = P(exit) × P(→bearish | exit)
    p_flip = p_exit * p_to_bearish
    
    return p_flip


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a single backtest."""
    pair: str
    config_name: str
    
    # Performance metrics
    total_return: float = 0
    annual_return: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    max_drawdown: float = 0
    calmar_ratio: float = 0
    
    # Trade stats
    n_trades: int = 0
    win_rate: float = 0
    avg_holding_hours: float = 0
    
    # Position stats
    avg_position_size: float = 0
    exposure_pct: float = 0
    
    # Data
    equity_curve: pd.Series = None
    yearly_returns: Dict[int, float] = field(default_factory=dict)


# =============================================================================
# REGIME COMPUTATION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_trend_with_hysteresis(close: pd.Series, ma: pd.Series,
                                   entry_buf: float, exit_buf: float) -> List[int]:
    """Compute trend signal with hysteresis buffers."""
    trend = []
    current = 1
    for i in range(len(close)):
        if pd.isna(ma.iloc[i]):
            trend.append(current)
            continue
        price = close.iloc[i]
        ma_val = ma.iloc[i]
        if current == 1:
            if price < ma_val * (1 - exit_buf) and price < ma_val * (1 - entry_buf):
                current = 0
        else:
            if price > ma_val * (1 + exit_buf) and price > ma_val * (1 + entry_buf):
                current = 1
        trend.append(current)
    return trend


def compute_states_hourly(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Compute 16-state regime at hourly frequency with confirmed states and durations."""
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_24h_hourly = ma_24h.reindex(df_1h.index, method='ffill')
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
    trend_24h = compute_trend_with_hysteresis(df_1h['close'], ma_24h_hourly, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = compute_trend_with_hysteresis(df_1h['close'], ma_168h_hourly, ENTRY_BUFFER, EXIT_BUFFER)
    
    states = pd.DataFrame(index=df_1h.index)
    states['close'] = df_1h['close']
    
    states['raw_state'] = (
        pd.Series(trend_24h, index=df_1h.index) * 8 +
        pd.Series(trend_168h, index=df_1h.index) * 4 +
        (ma_72h_hourly > ma_24h_hourly).astype(int) * 2 +
        (ma_168h_hourly > ma_24h_hourly).astype(int) * 1
    )
    
    # Apply confirmation filter
    raw_states = states['raw_state'].values
    n = len(raw_states)
    
    confirmed_states = []
    current_confirmed = int(raw_states[0])
    pending_state = None
    pending_count = 0
    
    for i in range(n):
        raw = int(raw_states[i])
        
        if raw == current_confirmed:
            pending_state = None
            pending_count = 0
            confirmed_states.append(current_confirmed)
        elif raw == pending_state:
            pending_count += 1
            if pending_count >= CONFIRMATION_HOURS:
                current_confirmed = pending_state
                pending_state = None
                pending_count = 0
            confirmed_states.append(current_confirmed)
        else:
            pending_state = raw
            pending_count = 1
            confirmed_states.append(current_confirmed)
    
    states['confirmed_state'] = confirmed_states
    
    # Track duration in confirmed state
    durations = []
    current_duration = 1
    for i in range(n):
        if i > 0 and confirmed_states[i] == confirmed_states[i-1]:
            current_duration += 1
        else:
            current_duration = 1
        durations.append(current_duration)
    
    states['duration_hours'] = durations
    
    # Pre-compute P(bearish flip) for efficiency
    p_flips = []
    for i in range(n):
        state = int(confirmed_states[i])
        duration = int(durations[i])
        p_flip = get_p_bearish_flip(state, duration, DECISION_HORIZON)
        p_flips.append(p_flip)
    
    states['p_bearish_flip'] = p_flips
    
    return states.dropna()


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_baseline(state: int, duration: int, p_flip: float, **kwargs) -> float:
    """Baseline: 100% for all bullish states."""
    if state in BULLISH_STATES:
        return 1.0
    return 0.0


def strategy_exclude_12_14(state: int, duration: int, p_flip: float, **kwargs) -> float:
    """Static exclusion of states 12 and 14."""
    if state in [12, 14]:
        return 0.0
    if state in BULLISH_STATES:
        return 1.0
    return 0.0


def make_dynamic_exit_strategy(threshold: float):
    """Factory for dynamic exit strategies with different thresholds."""
    def strategy(state: int, duration: int, p_flip: float, **kwargs) -> float:
        if state not in BULLISH_STATES:
            return 0.0
        if p_flip > threshold:
            return 0.0  # Exit
        return 1.0
    return strategy


def make_dynamic_sizing_strategy(thresholds: List[float]):
    """
    Factory for dynamic sizing strategies.
    thresholds = [t1, t2, t3] creates:
    - p_flip < t1: 100%
    - t1 <= p_flip < t2: 75%
    - t2 <= p_flip < t3: 50%
    - p_flip >= t3: 0% (exit)
    """
    def strategy(state: int, duration: int, p_flip: float, **kwargs) -> float:
        if state not in BULLISH_STATES:
            return 0.0
        
        if p_flip < thresholds[0]:
            return 1.0
        elif p_flip < thresholds[1]:
            return 0.75
        elif p_flip < thresholds[2]:
            return 0.50
        else:
            return 0.0  # Exit
    return strategy


def make_hybrid_strategy(exit_threshold: float):
    """
    Hybrid: Static exclusion of 12/14 + dynamic exit for other states.
    """
    def strategy(state: int, duration: int, p_flip: float, **kwargs) -> float:
        if state not in BULLISH_STATES:
            return 0.0
        
        # Static exclusion
        if state in [12, 14]:
            return 0.0
        
        # Dynamic exit for remaining states
        if p_flip > exit_threshold:
            return 0.0
        
        return 1.0
    return strategy


def make_state11_boost_strategy(boost_after_hours: int = 48, boost_amount: float = 1.25):
    """
    State 11 boost: Increase position in state 11 after duration threshold.
    Based on finding that state 11 is safer the longer you stay.
    """
    def strategy(state: int, duration: int, p_flip: float, **kwargs) -> float:
        if state not in BULLISH_STATES:
            return 0.0
        
        # Static exclusion
        if state in [12, 14]:
            return 0.0
        
        # Boost state 11 after threshold
        if state == 11 and duration >= boost_after_hours:
            return min(boost_amount, 1.5)  # Cap at 150%
        
        return 1.0
    return strategy


def make_combined_boost_hybrid(exit_threshold: float, boost_after: int = 48, boost_amount: float = 1.25):
    """
    Combined: Exclusion + dynamic exit + state 11 boost.
    """
    def strategy(state: int, duration: int, p_flip: float, **kwargs) -> float:
        if state not in BULLISH_STATES:
            return 0.0
        
        # Static exclusion
        if state in [12, 14]:
            return 0.0
        
        # Dynamic exit
        if p_flip > exit_threshold:
            return 0.0
        
        # State 11 boost
        if state == 11 and duration >= boost_after:
            return boost_amount
        
        return 1.0
    return strategy


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(states: pd.DataFrame, config_name: str, pair: str,
                 strategy_func: Callable) -> BacktestResult:
    """Run backtest with specified strategy function."""
    
    prices = states['close'].values
    confirmed_states = states['confirmed_state'].values
    durations = states['duration_hours'].values
    p_flips = states['p_bearish_flip'].values
    dates = states.index
    n = len(prices)
    
    # Track state
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    hourly_returns = []
    position_sizes = []
    
    position = 0
    position_size = 0
    entry_price = 0
    entry_date = None
    
    n_trades = 0
    trade_returns = []
    trade_hours = []
    hours_in_market = 0
    
    for i in range(1, n):
        current_state = int(confirmed_states[i])
        prev_state = int(confirmed_states[i-1])
        duration = int(durations[i])
        p_flip = float(p_flips[i])
        
        # Calculate target position size
        target_size = strategy_func(
            state=current_state,
            duration=duration,
            p_flip=p_flip,
        )
        
        # Calculate hourly return
        price_return = (prices[i] - prices[i-1]) / prices[i-1]
        
        if position_size > 0:
            position_return = position_size * KELLY_FRACTION * price_return
            equity *= (1 + position_return)
            hourly_returns.append(position_return)
            hours_in_market += 1
        else:
            hourly_returns.append(0)
        
        position_sizes.append(target_size)
        
        # Detect position changes
        was_in_position = position_size > 0
        should_be_in_position = target_size > 0
        
        if should_be_in_position and not was_in_position:
            # Enter position
            entry_price = prices[i]
            entry_date = dates[i]
            n_trades += 1
        elif not should_be_in_position and was_in_position:
            # Exit position
            if entry_price > 0:
                trade_return = (prices[i] - entry_price) / entry_price
                trade_returns.append(trade_return)
                if entry_date is not None:
                    trade_hours.append((dates[i] - entry_date).total_seconds() / 3600)
        
        position_size = target_size
        
        equity_curve.append(equity)
    
    # Close final position
    if position_size > 0 and entry_price > 0:
        trade_return = (prices[-1] - entry_price) / entry_price
        trade_returns.append(trade_return)
        if entry_date is not None:
            trade_hours.append((dates[-1] - entry_date).total_seconds() / 3600)
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=dates)
    returns_series = pd.Series(hourly_returns, index=dates[1:])
    
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Sharpe & Sortino (annualized from hourly)
    if returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(24 * 365)
        downside_returns = returns_series[returns_series < 0]
        sortino = returns_series.mean() / downside_returns.std() * np.sqrt(24 * 365) if len(downside_returns) > 0 else 0
    else:
        sharpe = sortino = 0
    
    # Max drawdown
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # Calmar ratio
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    # Trade stats
    if trade_returns:
        win_rate = len([t for t in trade_returns if t > 0]) / len(trade_returns)
        avg_holding = np.mean(trade_hours) if trade_hours else 0
    else:
        win_rate = avg_holding = 0
    
    avg_pos_size = np.mean([p for p in position_sizes if p > 0]) if any(p > 0 for p in position_sizes) else 0
    exposure_pct = hours_in_market / n if n > 0 else 0
    
    # Yearly returns
    yearly_returns = {}
    for year in range(dates[0].year, dates[-1].year + 1):
        year_mask = equity_series.index.year == year
        if year_mask.sum() > 0:
            year_data = equity_series[year_mask]
            if len(year_data) > 1:
                yearly_returns[year] = (year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]
    
    return BacktestResult(
        pair=pair,
        config_name=config_name,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_holding_hours=avg_holding,
        avg_position_size=avg_pos_size,
        exposure_pct=exposure_pct,
        equity_curve=equity_series,
        yearly_returns=yearly_returns,
    )


# =============================================================================
# PORTFOLIO AGGREGATION
# =============================================================================

def create_portfolio(pair_results: Dict[str, BacktestResult]) -> BacktestResult:
    """Aggregate pair results into portfolio result."""
    
    if not pair_results:
        return BacktestResult(pair='PORTFOLIO', config_name='')
    
    config_name = list(pair_results.values())[0].config_name
    
    # Combine equity curves (equal weight)
    equity_curves = []
    for pair, result in pair_results.items():
        if result.equity_curve is not None:
            normalized = result.equity_curve / result.equity_curve.iloc[0]
            equity_curves.append(normalized)
    
    if not equity_curves:
        return BacktestResult(pair='PORTFOLIO', config_name=config_name)
    
    combined = pd.concat(equity_curves, axis=1)
    combined = combined.fillna(method='ffill').fillna(method='bfill')
    portfolio_curve = combined.mean(axis=1) * INITIAL_CAPITAL
    
    total_return = (portfolio_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (portfolio_curve.index[-1] - portfolio_curve.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    daily_curve = portfolio_curve.resample('24h').last().dropna()
    daily_returns = daily_curve.pct_change().dropna()
    
    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        downside = daily_returns[daily_returns < 0]
        sortino = daily_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 else 0
    else:
        sharpe = sortino = 0
    
    rolling_max = portfolio_curve.expanding().max()
    drawdowns = (portfolio_curve - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    total_trades = sum(r.n_trades for r in pair_results.values())
    avg_win_rate = np.mean([r.win_rate for r in pair_results.values()])
    avg_holding = np.mean([r.avg_holding_hours for r in pair_results.values() if r.avg_holding_hours > 0])
    avg_pos_size = np.mean([r.avg_position_size for r in pair_results.values() if r.avg_position_size > 0])
    avg_exposure = np.mean([r.exposure_pct for r in pair_results.values()])
    
    yearly_returns = {}
    for year in range(daily_curve.index[0].year, daily_curve.index[-1].year + 1):
        year_mask = portfolio_curve.index.year == year
        if year_mask.sum() > 0:
            year_data = portfolio_curve[year_mask]
            if len(year_data) > 1:
                yearly_returns[year] = (year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]
    
    return BacktestResult(
        pair='PORTFOLIO',
        config_name=config_name,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        n_trades=total_trades,
        win_rate=avg_win_rate,
        avg_holding_hours=avg_holding,
        avg_position_size=avg_pos_size,
        exposure_pct=avg_exposure,
        equity_curve=portfolio_curve,
        yearly_returns=yearly_returns,
    )


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(portfolios: Dict[str, BacktestResult]):
    """Display comparison of all configurations."""
    
    print(f"\n{'=' * 140}")
    print("PORTFOLIO PERFORMANCE COMPARISON")
    print(f"{'=' * 140}")
    
    # Sort by Sharpe
    sorted_configs = sorted(portfolios.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
    
    print(f"\n  {'Config':<30} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10} {'Calmar':>8} "
          f"{'WinRate':>8} {'Trades':>8} {'Exposure':>10}")
    print(f"  {'─' * 110}")
    
    baseline_sharpe = portfolios.get('BASELINE', BacktestResult(pair='', config_name='')).sharpe_ratio
    
    for config, result in sorted_configs:
        improvement = ""
        if config != 'BASELINE' and baseline_sharpe > 0:
            pct = (result.sharpe_ratio - baseline_sharpe) / baseline_sharpe * 100
            improvement = f" ({pct:+.0f}%)"
        
        print(f"  {config:<30} {result.sharpe_ratio:>8.2f} {result.annual_return:>+9.1%} "
              f"{result.max_drawdown:>9.1%} {result.calmar_ratio:>8.2f} {result.win_rate:>7.1%} "
              f"{result.n_trades:>8} {result.exposure_pct:>9.1%}{improvement}")


def display_threshold_analysis(portfolios: Dict[str, BacktestResult]):
    """Display analysis of threshold effects."""
    
    print(f"\n{'=' * 140}")
    print("THRESHOLD ANALYSIS")
    print(f"{'=' * 140}")
    
    # Group by strategy type
    dynamic_exit = {k: v for k, v in portfolios.items() if k.startswith('DYN_EXIT')}
    hybrid = {k: v for k, v in portfolios.items() if k.startswith('HYBRID')}
    state11 = {k: v for k, v in portfolios.items() if 'STATE11' in k}
    
    if dynamic_exit:
        print(f"\n  DYNAMIC EXIT (exit when P(flip) > threshold)")
        print(f"  {'Threshold':>12} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Trades':>10}")
        print(f"  {'─' * 60}")
        for config in sorted(dynamic_exit.keys()):
            r = dynamic_exit[config]
            thresh = config.split('_')[-1]
            print(f"  {thresh:>12} {r.sharpe_ratio:>10.2f} {r.annual_return:>+11.1%} "
                  f"{r.max_drawdown:>9.1%} {r.n_trades:>10}")
    
    if hybrid:
        print(f"\n  HYBRID (exclude 12/14 + dynamic exit)")
        print(f"  {'Threshold':>12} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Trades':>10}")
        print(f"  {'─' * 60}")
        for config in sorted(hybrid.keys()):
            r = hybrid[config]
            thresh = config.split('_')[-1]
            print(f"  {thresh:>12} {r.sharpe_ratio:>10.2f} {r.annual_return:>+11.1%} "
                  f"{r.max_drawdown:>9.1%} {r.n_trades:>10}")
    
    if state11:
        print(f"\n  STATE 11 BOOST VARIANTS")
        print(f"  {'Config':<25} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10}")
        print(f"  {'─' * 60}")
        for config in sorted(state11.keys()):
            r = state11[config]
            print(f"  {config:<25} {r.sharpe_ratio:>10.2f} {r.annual_return:>+11.1%} "
                  f"{r.max_drawdown:>9.1%}")


def display_year_comparison(portfolios: Dict[str, BacktestResult], configs_to_show: List[str]):
    """Display year-by-year comparison for selected configs."""
    
    print(f"\n{'=' * 140}")
    print("YEAR-BY-YEAR RETURNS")
    print(f"{'=' * 140}")
    
    # Get all years
    all_years = set()
    for p in portfolios.values():
        all_years.update(p.yearly_returns.keys())
    all_years = sorted(all_years)
    
    # Header
    header = f"  {'Year':<8}"
    for config in configs_to_show:
        header += f" {config[:14]:>14}"
    print(header)
    print(f"  {'─' * (8 + 15 * len(configs_to_show))}")
    
    for year in all_years:
        row = f"  {year:<8}"
        for config in configs_to_show:
            if config in portfolios:
                ret = portfolios[config].yearly_returns.get(year, 0)
                row += f" {ret:>+13.1%}"
            else:
                row += f" {'N/A':>14}"
        print(row)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dynamic Threshold Backtest')
    parser.add_argument('--pair', type=str, default=None, help='Test specific pair only')
    args = parser.parse_args()
    
    print("=" * 140)
    print("DYNAMIC THRESHOLD BACKTEST")
    print("=" * 140)
    print(f"""
    Testing strategies with {DECISION_HORIZON}h decision horizon:
    
    1. BASELINE         - 100% all bullish states
    2. EXCLUDE_12_14    - Static exclusion (current best)
    3. DYN_EXIT_X%      - Dynamic exit when P(flip) > X%
    4. HYBRID_X%        - Exclusion + dynamic exit at X%
    5. STATE11_BOOST    - Boost position in state 11 after 48h
    6. COMBINED         - All of the above
    
    Exit thresholds tested: {EXIT_THRESHOLDS}
    """)
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    
    pairs = [args.pair] if args.pair else DEPLOY_PAIRS
    
    # Define all strategies to test
    strategies = {
        'BASELINE': strategy_baseline,
        'EXCLUDE_12_14': strategy_exclude_12_14,
    }
    
    # Add dynamic exit strategies
    for thresh in EXIT_THRESHOLDS:
        name = f'DYN_EXIT_{int(thresh*100)}%'
        strategies[name] = make_dynamic_exit_strategy(thresh)
    
    # Add hybrid strategies
    for thresh in [0.08, 0.10, 0.12, 0.15, 0.20]:
        name = f'HYBRID_{int(thresh*100)}%'
        strategies[name] = make_hybrid_strategy(thresh)
    
    # Add state 11 boost variants
    strategies['STATE11_BOOST_48h_125%'] = make_state11_boost_strategy(48, 1.25)
    strategies['STATE11_BOOST_48h_150%'] = make_state11_boost_strategy(48, 1.50)
    strategies['STATE11_BOOST_72h_125%'] = make_state11_boost_strategy(72, 1.25)
    
    # Add combined strategies
    for thresh in [0.10, 0.15]:
        name = f'COMBINED_{int(thresh*100)}%_BOOST'
        strategies[name] = make_combined_boost_hybrid(thresh, 48, 1.25)
    
    # Add dynamic sizing strategy
    strategies['DYN_SIZING'] = make_dynamic_sizing_strategy([0.05, 0.10, 0.20])
    
    all_results = {name: {} for name in strategies.keys()}
    
    # Process each pair
    for pair in pairs:
        print(f"\nProcessing {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        if df_1h is None or len(df_1h) == 0:
            print(f"WARNING: No data for {pair}")
            continue
        
        print(f"loaded {len(df_1h):,} hours...", end=" ", flush=True)
        
        # Compute states once
        states = compute_states_hourly(df_1h)
        print(f"states computed...", end=" ", flush=True)
        
        # Run all strategies
        for name, strategy_func in strategies.items():
            result = run_backtest(states, name, pair, strategy_func)
            all_results[name][pair] = result
        
        print("done")
    
    # Create portfolios
    print("\nAggregating portfolios...")
    portfolios = {}
    for name in strategies.keys():
        portfolios[name] = create_portfolio(all_results[name])
    
    # Display results
    display_results(portfolios)
    display_threshold_analysis(portfolios)
    
    # Year comparison for top performers
    top_configs = ['BASELINE', 'EXCLUDE_12_14']
    sorted_by_sharpe = sorted(portfolios.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
    for config, _ in sorted_by_sharpe[:3]:
        if config not in top_configs:
            top_configs.append(config)
    
    display_year_comparison(portfolios, top_configs[:5])
    
    print(f"\n{'=' * 140}")
    print("BACKTEST COMPLETE")
    print("=" * 140)
    
    return all_results, portfolios


if __name__ == "__main__":
    results, portfolios = main()