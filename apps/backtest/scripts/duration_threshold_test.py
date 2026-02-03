#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duration Threshold Optimization
================================
Test finer granularity on boost activation timing.

Current best: 24h
Test range: 6h, 12h, 18h, 24h, 30h, 36h, 42h, 48h

Includes realistic transaction costs (0.35% per side).

Usage:
    python duration_threshold_test.py
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
from typing import Dict, List
from dataclasses import dataclass, field
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

# Strategy parameters (locked)
ENTRY_BUFFER = 0.025
EXIT_BUFFER = 0.005
CONFIRMATION_HOURS = 3
EXCLUDED_STATES = {12, 14}
BOOST_STATE = 11

# Position sizing
KELLY_FRACTION = 0.25
INITIAL_CAPITAL = 100000
N_PAIRS = len(DEPLOY_PAIRS)

# State classification
BULLISH_STATES = {8, 9, 10, 11, 12, 13, 14, 15}
BEARISH_STATES = {0, 1, 2, 3, 4, 5, 6, 7}

# Realistic transaction costs
COST_PER_SIDE = 0.0035  # 0.35%
BOOST_EXTRA_COST = 0.001  # Extra slippage for larger positions

# Duration thresholds to test (hours)
DURATION_THRESHOLDS = [6, 12, 18, 24, 30, 36, 42, 48, 60, 72]

# Boost levels to test
BOOST_LEVELS = [1.5, 2.0, 2.5]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestResult:
    config_name: str
    duration_threshold: int
    boost_level: float
    
    sharpe_ratio: float = 0
    annual_return: float = 0
    max_drawdown: float = 0
    calmar_ratio: float = 0
    n_trades: int = 0
    win_rate: float = 0
    n_boosts_activated: int = 0
    avg_boost_duration: float = 0
    total_costs: float = 0


# =============================================================================
# REGIME COMPUTATION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_trend_with_hysteresis(close: pd.Series, ma: pd.Series,
                                   entry_buf: float, exit_buf: float) -> List[int]:
    trend = []
    current = 1
    for i in range(len(close)):
        if pd.isna(ma.iloc[i]):
            trend.append(current)
            continue
        price = close.iloc[i]
        ma_val = ma.iloc[i]
        if current == 1:
            if price < ma_val * (1 - exit_buf):
                current = 0
        else:
            if price > ma_val * (1 + entry_buf):
                current = 1
        trend.append(current)
    return trend


def compute_states_hourly(df_1h: pd.DataFrame) -> pd.DataFrame:
    if len(df_1h) == 0:
        return pd.DataFrame()
    
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    if len(df_24h) < MA_PERIOD_24H or len(df_72h) < MA_PERIOD_72H or len(df_168h) < MA_PERIOD_168H:
        return pd.DataFrame()
    
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
    states = states.copy()
    raw_states = states['raw_state'].values
    n = len(raw_states)
    
    confirmed_states = []
    durations = []
    
    current_confirmed = int(raw_states[0])
    current_duration = 1
    pending_state = None
    pending_count = 0
    
    for i in range(n):
        raw = int(raw_states[i])
        
        if raw == current_confirmed:
            pending_state = None
            pending_count = 0
            current_duration += 1 if i > 0 else 1
        elif raw == pending_state:
            pending_count += 1
            if pending_count >= CONFIRMATION_HOURS:
                current_confirmed = pending_state
                current_duration = pending_count + 1
                pending_state = None
                pending_count = 0
        else:
            pending_state = raw
            pending_count = 1
        
        confirmed_states.append(current_confirmed)
        durations.append(current_duration)
    
    states['confirmed_state'] = confirmed_states
    states['duration_hours'] = durations
    
    return states


# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(states: pd.DataFrame, duration_threshold: int, 
                 boost_level: float) -> BacktestResult:
    
    prices = states['close'].values
    confirmed_states = states['confirmed_state'].values
    durations = states['duration_hours'].values
    dates = states.index
    n = len(prices)
    
    capital = INITIAL_CAPITAL / N_PAIRS
    equity = capital
    equity_curve = [equity]
    hourly_returns = []
    
    position_multiplier = 0.0
    is_boosted = False
    entry_price = 0.0
    
    n_trades = 0
    wins = 0
    total_costs = 0.0
    
    # Track boost activations
    n_boosts = 0
    boost_durations = []
    current_boost_start = None
    
    for i in range(1, n):
        state = int(confirmed_states[i])
        duration = int(durations[i])
        price = prices[i]
        prev_price = prices[i-1]
        
        # Determine target position
        if state in BEARISH_STATES or state in EXCLUDED_STATES:
            target_mult = 0.0
            target_boosted = False
        elif state == BOOST_STATE and duration >= duration_threshold:
            target_mult = boost_level
            target_boosted = True
        else:
            target_mult = 1.0
            target_boosted = False
        
        # Calculate P&L
        price_return = (price - prev_price) / prev_price
        if position_multiplier > 0:
            position_return = position_multiplier * KELLY_FRACTION * price_return
            equity *= (1 + position_return)
            hourly_returns.append(position_return)
        else:
            hourly_returns.append(0.0)
        
        # Track boost activations
        if target_boosted and not is_boosted:
            n_boosts += 1
            current_boost_start = i
        elif not target_boosted and is_boosted:
            if current_boost_start is not None:
                boost_durations.append(i - current_boost_start)
                current_boost_start = None
        
        # Handle position changes with costs
        if target_mult != position_multiplier:
            position_change = abs(target_mult - position_multiplier)
            cost = COST_PER_SIDE * position_change
            if is_boosted or target_boosted:
                cost += BOOST_EXTRA_COST * position_change
            
            cost_amount = equity * cost * KELLY_FRACTION
            equity -= cost_amount
            total_costs += cost_amount
            
            # Track trades
            if target_mult > 0 and position_multiplier == 0:
                entry_price = price
                n_trades += 1
            elif target_mult == 0 and position_multiplier > 0:
                if entry_price > 0 and price > entry_price:
                    wins += 1
            
            position_multiplier = target_mult
            is_boosted = target_boosted
        
        equity_curve.append(equity)
    
    # Close final boost tracking
    if is_boosted and current_boost_start is not None:
        boost_durations.append(n - current_boost_start)
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=dates)
    returns_series = pd.Series(hourly_returns, index=dates[1:])
    
    total_return = (equity - capital) / capital
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    if returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(24 * 365)
    else:
        sharpe = 0
    
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    win_rate = wins / n_trades if n_trades > 0 else 0
    avg_boost_dur = np.mean(boost_durations) if boost_durations else 0
    
    config_name = f"D{duration_threshold}h_B{int(boost_level*100)}"
    
    return BacktestResult(
        config_name=config_name,
        duration_threshold=duration_threshold,
        boost_level=boost_level,
        sharpe_ratio=sharpe,
        annual_return=annual_return,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        n_trades=n_trades,
        win_rate=win_rate,
        n_boosts_activated=n_boosts,
        avg_boost_duration=avg_boost_dur,
        total_costs=total_costs
    )


def aggregate_results(pair_results: List[BacktestResult]) -> BacktestResult:
    if not pair_results:
        return BacktestResult("", 0, 0)
    
    # Simple average for demonstration
    return BacktestResult(
        config_name=pair_results[0].config_name,
        duration_threshold=pair_results[0].duration_threshold,
        boost_level=pair_results[0].boost_level,
        sharpe_ratio=np.mean([r.sharpe_ratio for r in pair_results]),
        annual_return=np.mean([r.annual_return for r in pair_results]),
        max_drawdown=np.mean([r.max_drawdown for r in pair_results]),
        calmar_ratio=np.mean([r.calmar_ratio for r in pair_results]),
        n_trades=sum(r.n_trades for r in pair_results),
        win_rate=np.mean([r.win_rate for r in pair_results]),
        n_boosts_activated=sum(r.n_boosts_activated for r in pair_results),
        avg_boost_duration=np.mean([r.avg_boost_duration for r in pair_results if r.avg_boost_duration > 0]),
        total_costs=sum(r.total_costs for r in pair_results)
    )


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(all_results: Dict[str, BacktestResult]):
    print(f"\n{'=' * 120}")
    print("  DURATION THRESHOLD OPTIMIZATION RESULTS")
    print(f"{'=' * 120}")
    
    print(f"""
  Testing boost activation timing with realistic costs (0.35% + 0.10% boost extra)
  
  Hazard data reference for State 11:
    • 0-6h:   P(→bearish) = 20.0%  (higher risk)
    • 6-12h:  P(→bearish) = 21.2%  (higher risk)
    • 12-24h: P(→bearish) = 13.8%  (transitioning)
    • 24-48h: P(→bearish) = 8.3%   (lower risk)
    • 48-72h: P(→bearish) = 4.2%   (lowest risk)
    """)
    
    # By boost level
    for boost in BOOST_LEVELS:
        print(f"\n{'─' * 100}")
        print(f"  BOOST LEVEL: {int(boost*100)}%")
        print(f"{'─' * 100}")
        
        print(f"\n  {'Threshold':>12} {'Sharpe':>10} {'Annual':>12} {'MaxDD':>10} {'Calmar':>10} "
              f"{'Boosts':>10} {'Avg Dur':>10}")
        print(f"  {'─' * 85}")
        
        for dur in DURATION_THRESHOLDS:
            key = f"D{dur}h_B{int(boost*100)}"
            if key in all_results:
                r = all_results[key]
                print(f"  {dur:>10}h {r.sharpe_ratio:>10.2f} {r.annual_return:>+11.1%} "
                      f"{r.max_drawdown:>9.1%} {r.calmar_ratio:>10.2f} "
                      f"{r.n_boosts_activated:>10} {r.avg_boost_duration:>9.1f}h")
    
    # Find optimal per boost level
    print(f"\n{'=' * 120}")
    print("  OPTIMAL DURATION BY BOOST LEVEL")
    print(f"{'=' * 120}")
    
    for boost in BOOST_LEVELS:
        best_sharpe = 0
        best_dur = 0
        for dur in DURATION_THRESHOLDS:
            key = f"D{dur}h_B{int(boost*100)}"
            if key in all_results and all_results[key].sharpe_ratio > best_sharpe:
                best_sharpe = all_results[key].sharpe_ratio
                best_dur = dur
        
        r = all_results.get(f"D{best_dur}h_B{int(boost*100)}")
        if r:
            print(f"\n  Boost {int(boost*100)}%: Optimal = {best_dur}h")
            print(f"    Sharpe: {r.sharpe_ratio:.2f}, Annual: {r.annual_return:+.1%}, MaxDD: {r.max_drawdown:.1%}")
    
    # Compare to no boost
    print(f"\n{'=' * 120}")
    print("  COMPARISON: OPTIMAL BOOST vs NO BOOST")
    print(f"{'=' * 120}")
    
    # Find overall best
    best_key = max(all_results.keys(), key=lambda k: all_results[k].sharpe_ratio)
    best = all_results[best_key]
    
    print(f"""
  OVERALL OPTIMAL:
    Configuration:  {best.config_name}
    Sharpe Ratio:   {best.sharpe_ratio:.2f}
    Annual Return:  {best.annual_return:+.1%}
    Max Drawdown:   {best.max_drawdown:.1%}
    Boosts/Year:    ~{best.n_boosts_activated / 10:.0f}
    Avg Boost Dur:  {best.avg_boost_duration:.1f}h
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 120)
    print("  DURATION THRESHOLD OPTIMIZATION")
    print("  Finding optimal boost activation timing")
    print("=" * 120)
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return None
    
    db = Database()
    
    # Load and prepare data
    pair_states = {}
    for pair in DEPLOY_PAIRS:
        print(f"  Loading {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        if df_1h is None or len(df_1h) == 0:
            print("SKIP")
            continue
        
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h.index = pd.to_datetime(df_1h.index)
        df_1h = df_1h.sort_index()
        
        states = compute_states_hourly(df_1h)
        if len(states) == 0:
            print("SKIP")
            continue
        
        states = apply_confirmation_filter(states)
        pair_states[pair] = states
        print(f"{len(states):,} hours")
    
    print(f"\n  Testing {len(DURATION_THRESHOLDS)} durations × {len(BOOST_LEVELS)} boost levels...")
    
    # Run all combinations
    all_results = {}
    
    for dur in DURATION_THRESHOLDS:
        for boost in BOOST_LEVELS:
            pair_results = []
            for pair, states in pair_states.items():
                result = run_backtest(states, dur, boost)
                pair_results.append(result)
            
            agg = aggregate_results(pair_results)
            all_results[agg.config_name] = agg
    
    print("  Done.\n")
    
    display_results(all_results)
    
    print(f"\n{'=' * 120}")
    print("  DURATION THRESHOLD TEST COMPLETE")
    print("=" * 120)
    
    return all_results


if __name__ == "__main__":
    results = main()