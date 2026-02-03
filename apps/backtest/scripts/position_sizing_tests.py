#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Sizing Strategy Tests
==============================
Compare multiple position sizing approaches for the 16-state regime system.

Configurations tested:
1. BASELINE       - 100% all bullish states
2. TIER_STATIC    - 100%/75%/50%/25% by risk tier
3. DECAY_1H_RAW   - Survival decay at 1h frequency, raw duration start
4. DECAY_1H_CONF  - Survival decay at 1h frequency, confirmed duration start
5. DECAY_3H_RAW   - Survival decay at 3h frequency, raw duration start  
6. DECAY_3H_CONF  - Survival decay at 3h frequency, confirmed duration start
7. TRANS_RISK     - Scale by P(stay bullish)
8. COMBINED_1H    - Tier × survival × trans_risk (1h confirmed)
9. COMBINED_3H    - Tier × survival × trans_risk (3h confirmed)

Usage:
    python position_sizing_tests.py
    python position_sizing_tests.py --pair ETHUSD
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

# MA Parameters (locked from validation)
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

# Entry/exit buffers (validated E2.5%)
ENTRY_BUFFER = 0.025
EXIT_BUFFER = 0.005

# Confirmation hours
CONFIRMATION_HOURS = 3

# Position sizing
KELLY_FRACTION = 0.25
INITIAL_CAPITAL = 100000

# State classification
BULLISH_STATES = set([8, 9, 10, 11, 12, 13, 14, 15])
BEARISH_STATES = set([0, 1, 2, 3, 4, 5, 6, 7])


# =============================================================================
# STATE PARAMETERS (from hourly_state_duration_analysis.py)
# =============================================================================

# Risk-based tier classification
STATE_TIERS = {
    'safe':     [15],           # 3.4% bearish, 1.95%/h exit
    'standard': [11, 13],       # <11% bearish
    'caution':  [9, 14],        # 20-38% bearish risk  
    'danger':   [12],           # 69.6% bearish
    'rare':     [8, 10],        # Insufficient data - treat as caution
}

TIER_POSITIONS = {
    'safe': 1.00,
    'standard': 0.75,
    'caution': 0.50,
    'danger': 0.25,
    'rare': 0.50,
}

def get_state_tier(state: int) -> str:
    """Get tier for a given state."""
    for tier, states in STATE_TIERS.items():
        if state in states:
            return tier
    return 'caution'  # Default

# Exit rates per hour (confirmed duration start)
STATE_EXIT_RATE_1H_CONF = {
    8: 0.05,    # rare - estimate
    9: 0.0396,
    10: 0.05,   # rare - estimate
    11: 0.0249,
    12: 0.0260,
    13: 0.0336,
    14: 0.0245,
    15: 0.0195,
}

# Exit rates per 3h block (confirmed duration start)
STATE_EXIT_RATE_3H_CONF = {
    8: 0.12,    # rare - estimate
    9: 0.1152,
    10: 0.12,   # rare - estimate
    11: 0.0732,
    12: 0.0763,
    13: 0.0988,
    14: 0.0725,
    15: 0.0577,
}

# Exit rates per hour (raw duration start)
STATE_EXIT_RATE_1H_RAW = {
    8: 0.06,    # rare - estimate
    9: 0.0444,
    10: 0.06,   # rare - estimate
    11: 0.0272,
    12: 0.0273,
    13: 0.0360,
    14: 0.0252,
    15: 0.0204,
}

# Exit rates per 3h block (raw duration start)
STATE_EXIT_RATE_3H_RAW = {
    8: 0.14,    # rare - estimate
    9: 0.1294,
    10: 0.14,   # rare - estimate
    11: 0.0797,
    12: 0.0797,
    13: 0.1052,
    14: 0.0748,
    15: 0.0608,
}

# P(→bearish | exit from state) - same across frequencies
STATE_P_TO_BEARISH = {
    8: 0.30,    # rare - estimate
    9: 0.202,
    10: 0.25,   # rare - estimate
    11: 0.108,
    12: 0.696,  # DANGER
    13: 0.079,
    14: 0.383,  # CAUTION
    15: 0.034,
}

# P(stay bullish) = 1 - P(→bearish)
STATE_P_STAY_BULLISH = {k: 1 - v for k, v in STATE_P_TO_BEARISH.items()}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeRecord:
    """Record of a single trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    holding_hours: int
    avg_position_size: float
    entry_state: int
    exit_state: int


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
    avg_trade_return: float = 0
    profit_factor: float = 0
    avg_holding_hours: float = 0
    
    # Position stats
    avg_position_size: float = 0
    exposure_pct: float = 0
    
    # Data
    equity_curve: pd.Series = None
    trades: List[TradeRecord] = field(default_factory=list)
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
    """Compute 16-state regime at hourly frequency."""
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
    
    return states.dropna()


def apply_confirmation_filter(states: pd.DataFrame) -> pd.DataFrame:
    """Apply 3h confirmation filter and track durations."""
    states = states.copy()
    raw_states = states['raw_state'].values
    n = len(raw_states)
    
    confirmed_states = []
    raw_durations = []  # Hours in current raw state
    confirmed_durations = []  # Hours in current confirmed state
    
    current_confirmed = int(raw_states[0])
    current_raw = int(raw_states[0])
    pending_state = None
    pending_count = 0
    
    raw_duration = 1
    confirmed_duration = 1
    
    for i in range(n):
        raw = int(raw_states[i])
        
        # Track raw duration
        if raw == current_raw:
            raw_duration += 1
        else:
            current_raw = raw
            raw_duration = 1
        
        # Confirmation logic
        if raw == current_confirmed:
            pending_state = None
            pending_count = 0
            confirmed_states.append(current_confirmed)
        elif raw == pending_state:
            pending_count += 1
            if pending_count >= CONFIRMATION_HOURS:
                current_confirmed = pending_state
                confirmed_duration = pending_count  # Start from confirmation
                pending_state = None
                pending_count = 0
            confirmed_states.append(current_confirmed)
        else:
            pending_state = raw
            pending_count = 1
            confirmed_states.append(current_confirmed)
        
        # Track confirmed duration
        if i > 0 and confirmed_states[-1] == confirmed_states[-2] if len(confirmed_states) > 1 else True:
            confirmed_duration += 1
        else:
            confirmed_duration = 1
        
        raw_durations.append(raw_duration)
        confirmed_durations.append(confirmed_duration)
    
    states['confirmed_state'] = confirmed_states
    states['raw_duration_hours'] = raw_durations
    states['confirmed_duration_hours'] = confirmed_durations
    
    # Also track 3h block durations
    states['raw_duration_3h'] = [d // 3 for d in raw_durations]
    states['confirmed_duration_3h'] = [d // 3 for d in confirmed_durations]
    
    return states


# =============================================================================
# POSITION SIZING MODELS
# =============================================================================

def sizing_baseline(state: int, duration_hours: int, duration_3h: int, **kwargs) -> float:
    """Baseline: 100% for all bullish states."""
    if state in BULLISH_STATES:
        return 1.0
    return 0.0


def sizing_tier_static(state: int, duration_hours: int, duration_3h: int, **kwargs) -> float:
    """Static tier-based sizing."""
    if state not in BULLISH_STATES:
        return 0.0
    tier = get_state_tier(state)
    return TIER_POSITIONS[tier]


def sizing_decay_1h_raw(state: int, duration_hours: int, duration_3h: int, **kwargs) -> float:
    """Survival decay at 1h frequency, raw duration."""
    if state not in BULLISH_STATES:
        return 0.0
    exit_rate = STATE_EXIT_RATE_1H_RAW.get(state, 0.03)
    survival = (1 - exit_rate) ** duration_hours
    return max(survival, 0.0)  # Floor at 0 (can create exit)


def sizing_decay_1h_conf(state: int, duration_hours: int, duration_3h: int, 
                          confirmed_duration_hours: int = None, **kwargs) -> float:
    """Survival decay at 1h frequency, confirmed duration."""
    if state not in BULLISH_STATES:
        return 0.0
    dur = confirmed_duration_hours if confirmed_duration_hours is not None else duration_hours
    exit_rate = STATE_EXIT_RATE_1H_CONF.get(state, 0.03)
    survival = (1 - exit_rate) ** dur
    return max(survival, 0.0)


def sizing_decay_3h_raw(state: int, duration_hours: int, duration_3h: int, **kwargs) -> float:
    """Survival decay at 3h frequency, raw duration."""
    if state not in BULLISH_STATES:
        return 0.0
    exit_rate = STATE_EXIT_RATE_3H_RAW.get(state, 0.08)
    survival = (1 - exit_rate) ** duration_3h
    return max(survival, 0.0)


def sizing_decay_3h_conf(state: int, duration_hours: int, duration_3h: int,
                          confirmed_duration_3h: int = None, **kwargs) -> float:
    """Survival decay at 3h frequency, confirmed duration."""
    if state not in BULLISH_STATES:
        return 0.0
    dur = confirmed_duration_3h if confirmed_duration_3h is not None else duration_3h
    exit_rate = STATE_EXIT_RATE_3H_CONF.get(state, 0.08)
    survival = (1 - exit_rate) ** dur
    return max(survival, 0.0)


def sizing_trans_risk(state: int, duration_hours: int, duration_3h: int, **kwargs) -> float:
    """Scale by P(stay bullish) - transition risk."""
    if state not in BULLISH_STATES:
        return 0.0
    return STATE_P_STAY_BULLISH.get(state, 0.5)


def sizing_combined_1h(state: int, duration_hours: int, duration_3h: int,
                        confirmed_duration_hours: int = None, **kwargs) -> float:
    """Combined: tier × survival × trans_risk (1h confirmed)."""
    if state not in BULLISH_STATES:
        return 0.0
    
    # Tier base
    tier = get_state_tier(state)
    tier_mult = TIER_POSITIONS[tier]
    
    # Survival (1h confirmed)
    dur = confirmed_duration_hours if confirmed_duration_hours is not None else duration_hours
    exit_rate = STATE_EXIT_RATE_1H_CONF.get(state, 0.03)
    survival = (1 - exit_rate) ** dur
    
    # Transition risk
    trans_mult = STATE_P_STAY_BULLISH.get(state, 0.5)
    
    return max(tier_mult * survival * trans_mult, 0.0)


def sizing_combined_3h(state: int, duration_hours: int, duration_3h: int,
                        confirmed_duration_3h: int = None, **kwargs) -> float:
    """Combined: tier × survival × trans_risk (3h confirmed)."""
    if state not in BULLISH_STATES:
        return 0.0
    
    # Tier base
    tier = get_state_tier(state)
    tier_mult = TIER_POSITIONS[tier]
    
    # Survival (3h confirmed)
    dur = confirmed_duration_3h if confirmed_duration_3h is not None else duration_3h
    exit_rate = STATE_EXIT_RATE_3H_CONF.get(state, 0.08)
    survival = (1 - exit_rate) ** dur
    
    # Transition risk
    trans_mult = STATE_P_STAY_BULLISH.get(state, 0.5)
    
    return max(tier_mult * survival * trans_mult, 0.0)


def sizing_exclude_12_14(state: int, duration_hours: int, duration_3h: int, **kwargs) -> float:
    """Exclude states 12 and 14 entirely. Full position for other bullish states."""
    if state in [12, 14]:
        return 0.0  # Don't enter these states
    if state in BULLISH_STATES:
        return 1.0
    return 0.0


def sizing_exclude_12_only(state: int, duration_hours: int, duration_3h: int, **kwargs) -> float:
    """Exclude only state 12 (worst offender). Full position for other bullish states."""
    if state == 12:
        return 0.0  # Don't enter state 12
    if state in BULLISH_STATES:
        return 1.0
    return 0.0


def sizing_exclude_12_14_trans(state: int, duration_hours: int, duration_3h: int, **kwargs) -> float:
    """Exclude 12/14 + apply TRANS_RISK to remaining states."""
    if state in [12, 14]:
        return 0.0  # Don't enter these states
    if state not in BULLISH_STATES:
        return 0.0
    return STATE_P_STAY_BULLISH.get(state, 0.5)


# Configuration registry
SIZING_CONFIGS = {
    'BASELINE': sizing_baseline,
    'TIER_STATIC': sizing_tier_static,
    'DECAY_1H_RAW': sizing_decay_1h_raw,
    'DECAY_1H_CONF': sizing_decay_1h_conf,
    'DECAY_3H_RAW': sizing_decay_3h_raw,
    'DECAY_3H_CONF': sizing_decay_3h_conf,
    'TRANS_RISK': sizing_trans_risk,
    'COMBINED_1H': sizing_combined_1h,
    'COMBINED_3H': sizing_combined_3h,
    'EXCLUDE_12_14': sizing_exclude_12_14,
    'EXCLUDE_12': sizing_exclude_12_only,
    'EXCL_12_14_TR': sizing_exclude_12_14_trans,
}


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(states: pd.DataFrame, config_name: str, pair: str,
                 sizing_func: Callable) -> BacktestResult:
    """Run backtest with specified position sizing function."""
    
    prices = states['close'].values
    confirmed_states = states['confirmed_state'].values
    raw_durations = states['raw_duration_hours'].values
    confirmed_durations = states['confirmed_duration_hours'].values
    raw_durations_3h = states['raw_duration_3h'].values
    confirmed_durations_3h = states['confirmed_duration_3h'].values
    dates = states.index
    n = len(prices)
    
    # Track state
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    hourly_returns = []
    position_sizes = []
    
    position = 0  # Current position (0 = flat, 1 = long)
    position_size = 0  # Position size multiplier
    entry_price = 0
    entry_date = None
    entry_state = None
    trade_position_sizes = []  # Track position sizes during trade
    
    trades = []
    hours_in_market = 0
    
    for i in range(1, n):
        current_state = int(confirmed_states[i])
        prev_state = int(confirmed_states[i-1])
        
        is_bullish = current_state in BULLISH_STATES
        was_bullish = prev_state in BULLISH_STATES
        
        # Calculate position size for current state
        new_position_size = sizing_func(
            state=current_state,
            duration_hours=int(raw_durations[i]),
            duration_3h=int(raw_durations_3h[i]),
            confirmed_duration_hours=int(confirmed_durations[i]),
            confirmed_duration_3h=int(confirmed_durations_3h[i]),
        )
        
        # Calculate hourly return
        price_return = (prices[i] - prices[i-1]) / prices[i-1]
        
        if position > 0:
            # We're in a position - use current position size
            position_return = position_size * KELLY_FRACTION * price_return
            equity *= (1 + position_return)
            hourly_returns.append(position_return)
            hours_in_market += 1
            trade_position_sizes.append(position_size)
        else:
            hourly_returns.append(0)
        
        position_sizes.append(new_position_size)
        
        # Update position size for next period
        position_size = new_position_size
        
        # Handle regime changes
        if is_bullish and not was_bullish:
            # Enter long
            position = 1
            entry_price = prices[i]
            entry_date = dates[i]
            entry_state = current_state
            trade_position_sizes = [position_size]
            
        elif not is_bullish and was_bullish:
            # Exit long
            if position > 0:
                trade_return = (prices[i] - entry_price) / entry_price
                holding_hours = (dates[i] - entry_date).total_seconds() / 3600
                avg_pos_size = np.mean(trade_position_sizes) if trade_position_sizes else 0
                
                trades.append(TradeRecord(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    entry_price=entry_price,
                    exit_price=prices[i],
                    return_pct=trade_return,
                    holding_hours=int(holding_hours),
                    avg_position_size=avg_pos_size,
                    entry_state=entry_state,
                    exit_state=current_state,
                ))
            
            position = 0
            position_size = 0
        
        # Handle position size decay to zero (creates exit signal)
        elif position > 0 and new_position_size <= 0.01:
            # Position decayed to effectively zero - exit
            trade_return = (prices[i] - entry_price) / entry_price
            holding_hours = (dates[i] - entry_date).total_seconds() / 3600
            avg_pos_size = np.mean(trade_position_sizes) if trade_position_sizes else 0
            
            trades.append(TradeRecord(
                entry_date=entry_date,
                exit_date=dates[i],
                entry_price=entry_price,
                exit_price=prices[i],
                return_pct=trade_return,
                holding_hours=int(holding_hours),
                avg_position_size=avg_pos_size,
                entry_state=entry_state,
                exit_state=current_state,
            ))
            
            position = 0
            position_size = 0
        
        equity_curve.append(equity)
    
    # Close final position
    if position > 0:
        trade_return = (prices[-1] - entry_price) / entry_price
        holding_hours = (dates[-1] - entry_date).total_seconds() / 3600
        avg_pos_size = np.mean(trade_position_sizes) if trade_position_sizes else 0
        
        trades.append(TradeRecord(
            entry_date=entry_date,
            exit_date=dates[-1],
            entry_price=entry_price,
            exit_price=prices[-1],
            return_pct=trade_return,
            holding_hours=int(holding_hours),
            avg_position_size=avg_pos_size,
            entry_state=entry_state,
            exit_state=int(confirmed_states[-1]),
        ))
    
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
    n_trades = len(trades)
    if n_trades > 0:
        trade_returns = [t.return_pct for t in trades]
        winners = [t for t in trade_returns if t > 0]
        losers = [t for t in trade_returns if t <= 0]
        
        win_rate = len(winners) / n_trades
        avg_trade = np.mean(trade_returns)
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0.001
        profit_factor = gross_profit / gross_loss
        avg_holding = np.mean([t.holding_hours for t in trades])
        avg_pos_size = np.mean([t.avg_position_size for t in trades])
    else:
        win_rate = avg_trade = profit_factor = avg_holding = avg_pos_size = 0
    
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
        avg_trade_return=avg_trade,
        profit_factor=profit_factor,
        avg_holding_hours=avg_holding,
        avg_position_size=avg_pos_size,
        exposure_pct=exposure_pct,
        equity_curve=equity_series,
        trades=trades,
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
            # Normalize to returns
            normalized = result.equity_curve / result.equity_curve.iloc[0]
            equity_curves.append(normalized)
    
    if not equity_curves:
        return BacktestResult(pair='PORTFOLIO', config_name=config_name)
    
    # Align and average
    combined = pd.concat(equity_curves, axis=1)
    combined = combined.fillna(method='ffill').fillna(method='bfill')
    portfolio_curve = combined.mean(axis=1) * INITIAL_CAPITAL
    
    # Calculate portfolio metrics
    total_return = (portfolio_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (portfolio_curve.index[-1] - portfolio_curve.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Daily returns for Sharpe calculation
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
    
    # Aggregate trade stats
    total_trades = sum(r.n_trades for r in pair_results.values())
    avg_win_rate = np.mean([r.win_rate for r in pair_results.values()])
    avg_holding = np.mean([r.avg_holding_hours for r in pair_results.values() if r.avg_holding_hours > 0])
    avg_pos_size = np.mean([r.avg_position_size for r in pair_results.values() if r.avg_position_size > 0])
    avg_exposure = np.mean([r.exposure_pct for r in pair_results.values()])
    
    # Yearly returns
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

def display_results(all_results: Dict[str, Dict[str, BacktestResult]], 
                    portfolios: Dict[str, BacktestResult]):
    """Display comparison of all configurations."""
    
    print(f"\n{'=' * 140}")
    print("POSITION SIZING STRATEGY COMPARISON")
    print(f"{'=' * 140}")
    
    # Portfolio summary table
    print(f"\n{'─' * 140}")
    print("PORTFOLIO PERFORMANCE")
    print(f"{'─' * 140}")
    
    configs = list(portfolios.keys())
    
    # Header
    header = f"{'Metric':<25}"
    for config in configs:
        header += f" {config:<14}"
    print(header)
    print("─" * 140)
    
    # Metrics
    metrics = [
        ('Annual Return', 'annual_return', lambda x: f"{x:+.1%}"),
        ('Sharpe Ratio', 'sharpe_ratio', lambda x: f"{x:.2f}"),
        ('Sortino Ratio', 'sortino_ratio', lambda x: f"{x:.2f}"),
        ('Max Drawdown', 'max_drawdown', lambda x: f"{x:.1%}"),
        ('Calmar Ratio', 'calmar_ratio', lambda x: f"{x:.2f}"),
        ('Total Trades', 'n_trades', lambda x: f"{x:,}"),
        ('Win Rate', 'win_rate', lambda x: f"{x:.1%}"),
        ('Avg Position Size', 'avg_position_size', lambda x: f"{x:.1%}"),
        ('Market Exposure', 'exposure_pct', lambda x: f"{x:.1%}"),
    ]
    
    for label, attr, fmt in metrics:
        row = f"{label:<25}"
        for config in configs:
            val = getattr(portfolios[config], attr)
            row += f" {fmt(val):<14}"
        print(row)
    
    # Final equity
    print("─" * 140)
    row = f"{'Final Equity':<25}"
    for config in configs:
        equity = portfolios[config].equity_curve.iloc[-1] if portfolios[config].equity_curve is not None else INITIAL_CAPITAL
        row += f" ${equity:>12,.0f}"
    print(row)
    
    # Ranking by Sharpe
    print(f"\n{'─' * 140}")
    print("RANKING BY SHARPE RATIO")
    print(f"{'─' * 140}")
    
    ranked = sorted(portfolios.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
    for i, (config, result) in enumerate(ranked, 1):
        improvement = ""
        if i > 1:
            baseline_sharpe = portfolios['BASELINE'].sharpe_ratio
            if baseline_sharpe > 0:
                pct_diff = (result.sharpe_ratio - baseline_sharpe) / baseline_sharpe * 100
                improvement = f" ({pct_diff:+.1f}% vs baseline)"
        print(f"  {i}. {config:<20} Sharpe: {result.sharpe_ratio:.3f}  "
              f"Return: {result.annual_return:+.1%}  DD: {result.max_drawdown:.1%}{improvement}")
    
    # Year-by-year comparison (top 3 vs baseline)
    print(f"\n{'─' * 140}")
    print("YEAR-BY-YEAR RETURNS: TOP 3 vs BASELINE")
    print(f"{'─' * 140}")
    
    top_configs = [c for c, _ in ranked[:3]]
    if 'BASELINE' not in top_configs:
        top_configs.append('BASELINE')
    
    years = sorted(portfolios['BASELINE'].yearly_returns.keys())
    
    header = f"{'Year':<8}"
    for config in top_configs:
        header += f" {config:<14}"
    print(header)
    print("─" * 80)
    
    for year in years:
        row = f"{year:<8}"
        for config in top_configs:
            ret = portfolios[config].yearly_returns.get(year, 0)
            row += f" {ret:>+13.1%}"
        print(row)
    
    # Per-pair breakdown for top config
    print(f"\n{'─' * 140}")
    top_config = ranked[0][0]
    print(f"PER-PAIR BREAKDOWN: {top_config}")
    print(f"{'─' * 140}")
    
    print(f"  {'Pair':<12} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Trades':>8} {'Avg Pos':>10}")
    print(f"  {'─' * 70}")
    
    for pair in DEPLOY_PAIRS:
        if pair in all_results[top_config]:
            r = all_results[top_config][pair]
            print(f"  {pair:<12} {r.sharpe_ratio:>10.2f} {r.annual_return:>+11.1%} "
                  f"{r.max_drawdown:>10.1%} {r.n_trades:>8} {r.avg_position_size:>9.1%}")


def display_state_analysis(all_results: Dict[str, Dict[str, BacktestResult]]):
    """Analyze performance by entry state."""
    
    print(f"\n{'=' * 140}")
    print("TRADE ANALYSIS BY ENTRY STATE")
    print(f"{'=' * 140}")
    
    # Aggregate trades by entry state for top configs
    for config in ['BASELINE', 'TRANS_RISK', 'EXCLUDE_12_14', 'EXCL_12_14_TR']:
        if config not in all_results:
            continue
            
        state_trades = defaultdict(list)
        
        for pair, result in all_results[config].items():
            for trade in result.trades:
                state_trades[trade.entry_state].append(trade.return_pct)
        
        print(f"\n{config}:")
        print(f"  {'State':<8} {'N Trades':>10} {'Win Rate':>10} {'Avg Return':>12} {'Total Return':>14}")
        print(f"  {'─' * 60}")
        
        for state in sorted(BULLISH_STATES):
            trades = state_trades.get(state, [])
            if len(trades) > 0:
                n = len(trades)
                win_rate = len([t for t in trades if t > 0]) / n
                avg_ret = np.mean(trades)
                total_ret = sum(trades)
                print(f"  {state:<8} {n:>10} {win_rate:>9.1%} {avg_ret:>+11.2%} {total_ret:>+13.1%}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Position Sizing Strategy Tests')
    parser.add_argument('--pair', type=str, default=None, help='Test specific pair only')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Specific configs to test (default: all)')
    args = parser.parse_args()
    
    print("=" * 140)
    print("POSITION SIZING STRATEGY TESTS")
    print("=" * 140)
    print(f"""
    Testing {len(SIZING_CONFIGS)} position sizing configurations:
    
    POSITION SIZING:
    1. BASELINE      - 100% for all bullish states
    2. TIER_STATIC   - 100%/75%/50%/25% by risk tier
    3. TRANS_RISK    - Scale by P(stay bullish)
    
    DECAY MODELS:
    4. DECAY_1H_RAW  - Survival decay at 1h, raw duration
    5. DECAY_1H_CONF - Survival decay at 1h, confirmed duration
    6. DECAY_3H_RAW  - Survival decay at 3h, raw duration
    7. DECAY_3H_CONF - Survival decay at 3h, confirmed duration
    
    COMBINED:
    8. COMBINED_1H   - Tier × survival × trans_risk (1h)
    9. COMBINED_3H   - Tier × survival × trans_risk (3h)
    
    EXCLUSION:
    10. EXCLUDE_12_14  - Skip states 12 & 14, full position otherwise
    11. EXCLUDE_12     - Skip state 12 only, full position otherwise
    12. EXCL_12_14_TR  - Skip 12 & 14 + TRANS_RISK on remaining
    
    Parameters: E{ENTRY_BUFFER:.1%} buffer, {CONFIRMATION_HOURS}h confirmation, {KELLY_FRACTION:.0%} Kelly
    """)
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    
    pairs = [args.pair] if args.pair else DEPLOY_PAIRS
    configs = args.configs if args.configs else list(SIZING_CONFIGS.keys())
    
    all_results = {config: {} for config in configs}
    
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
        states = apply_confirmation_filter(states)
        print(f"states computed...", end=" ", flush=True)
        
        # Run all configs
        for config in configs:
            sizing_func = SIZING_CONFIGS[config]
            result = run_backtest(states, config, pair, sizing_func)
            all_results[config][pair] = result
        
        print("done")
    
    # Create portfolios
    print("\nAggregating portfolios...")
    portfolios = {}
    for config in configs:
        portfolios[config] = create_portfolio(all_results[config])
    
    # Display results
    display_results(all_results, portfolios)
    display_state_analysis(all_results)
    
    print(f"\n{'=' * 140}")
    print("TESTS COMPLETE")
    print("=" * 140)
    
    return all_results, portfolios


if __name__ == "__main__":
    results, portfolios = main()