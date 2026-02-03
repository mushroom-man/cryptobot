#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Quality Optimization: Hysteresis & Confirmation Barriers
==============================================================
Test different approaches to improve trade quality (fewer trades, higher win rate).

Tests:
1. Hysteresis parameters: Entry buffer × Exit buffer combinations
2. Confirmation barriers: State must persist N HOURS before acting (1h, 6h, 12h, 18h, 24h, 48h)
3. Combined: Hysteresis + confirmation

Current baseline: Entry 1.5%, Exit 0.5%, No confirmation

Usage:
    python trade_quality_optimization.py --pair ETHUSD
    python trade_quality_optimization.py --all-pairs
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
from typing import Dict, List, Tuple
from dataclasses import dataclass
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

# Hysteresis combinations to test
# Format: (entry_buffer, exit_buffer)
HYSTERESIS_CONFIGS = [
    (0.010, 0.005),  # Tighter entry
    (0.015, 0.005),  # Current baseline
    (0.020, 0.005),  # Wider entry
    (0.025, 0.005),  # Much wider entry
    (0.030, 0.005),  # Very wide entry
    (0.015, 0.010),  # Wider exit
    (0.020, 0.010),  # Both wider
    (0.025, 0.010),  # Wide entry, wider exit
    (0.030, 0.015),  # Very wide both
    (0.020, 0.000),  # No exit buffer
    (0.025, 0.000),  # Wide entry, no exit
]

# Confirmation periods to test (in HOURS - state must persist N hours before acting)
CONFIRMATION_HOURS = [0, 1, 3, 6, 12, 18, 24, 36, 48]

# Trading parameters
INITIAL_CAPITAL = 100000
KELLY_FRACTION = 0.25

BULLISH_STATES = [8, 9, 10, 11, 12, 13, 14, 15]


# =============================================================================
# REGIME COMPUTATION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_trend_with_hysteresis(close: pd.Series, ma: pd.Series,
                                   entry_buf: float, exit_buf: float) -> List[int]:
    """Compute trend with configurable hysteresis buffers."""
    trend = []
    current = 1
    for i in range(len(close)):
        if pd.isna(ma.iloc[i]):
            trend.append(current)
            continue
        price = close.iloc[i]
        ma_val = ma.iloc[i]
        if current == 1:  # Currently bullish
            # Need to break below MA by exit_buf AND entry_buf to go bearish
            if price < ma_val * (1 - exit_buf) and price < ma_val * (1 - entry_buf):
                current = 0
        else:  # Currently bearish
            # Need to break above MA by exit_buf AND entry_buf to go bullish
            if price > ma_val * (1 + exit_buf) and price > ma_val * (1 + entry_buf):
                current = 1
        trend.append(current)
    return trend


def compute_states_with_config(df_1h: pd.DataFrame, entry_buf: float, 
                                exit_buf: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute 16-state regime with specified hysteresis.
    Returns both daily and hourly states.
    """
    # Daily states (for trading)
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_daily = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_daily = ma_168h.reindex(df_24h.index, method='ffill')
    
    trend_24h_daily = compute_trend_with_hysteresis(df_24h['close'], ma_24h, entry_buf, exit_buf)
    trend_168h_daily = compute_trend_with_hysteresis(df_24h['close'], ma_168h_daily, entry_buf, exit_buf)
    
    states_daily = pd.DataFrame(index=df_24h.index)
    states_daily['close'] = df_24h['close']
    states_daily['state'] = (
        pd.Series(trend_24h_daily, index=df_24h.index) * 8 +
        pd.Series(trend_168h_daily, index=df_24h.index) * 4 +
        (ma_72h_daily > ma_24h).astype(int) * 2 +
        (ma_168h_daily > ma_24h).astype(int) * 1
    )
    
    # Hourly states (for confirmation filtering)
    ma_24h_hourly = ma_24h.reindex(df_1h.index, method='ffill')
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
    trend_24h_hourly = compute_trend_with_hysteresis(df_1h['close'], ma_24h_hourly, entry_buf, exit_buf)
    trend_168h_hourly = compute_trend_with_hysteresis(df_1h['close'], ma_168h_hourly, entry_buf, exit_buf)
    
    states_hourly = pd.DataFrame(index=df_1h.index)
    states_hourly['close'] = df_1h['close']
    states_hourly['state'] = (
        pd.Series(trend_24h_hourly, index=df_1h.index) * 8 +
        pd.Series(trend_168h_hourly, index=df_1h.index) * 4 +
        (ma_72h_hourly > ma_24h_hourly).astype(int) * 2 +
        (ma_168h_hourly > ma_24h_hourly).astype(int) * 1
    )
    
    return states_daily.dropna(), states_hourly.dropna()


def apply_confirmation_filter(states_daily: pd.DataFrame, states_hourly: pd.DataFrame, 
                               confirmation_hours: int) -> pd.DataFrame:
    """
    Apply confirmation barrier using hourly data: only act when state persists for N hours.
    
    This looks back N hours to check if the state has been consistent.
    If state changed within the last N hours, we use the previous confirmed state.
    """
    if confirmation_hours == 0:
        return states_daily.copy()
    
    states_daily = states_daily.copy()
    hourly_states = states_hourly['state'].values
    hourly_index = states_hourly.index
    
    confirmed_states = []
    
    for daily_time in states_daily.index:
        # Find the current hourly state
        hourly_mask = hourly_index <= daily_time
        if not hourly_mask.any():
            confirmed_states.append(states_daily.loc[daily_time, 'state'])
            continue
        
        # Get last N hours of states
        recent_hours = hourly_index[hourly_mask][-confirmation_hours:]
        if len(recent_hours) < confirmation_hours:
            # Not enough history, use current state
            confirmed_states.append(states_daily.loc[daily_time, 'state'])
            continue
        
        recent_states = states_hourly.loc[recent_hours, 'state'].values
        current_state = recent_states[-1]
        
        # Check if state has been consistent for N hours
        if np.all(recent_states == current_state):
            # Confirmed - state has persisted
            confirmed_states.append(current_state)
        else:
            # Not confirmed - find the most recent stable state
            # Use the state that was held before the most recent change
            for i in range(len(recent_states) - 2, -1, -1):
                if recent_states[i] != current_state:
                    confirmed_states.append(recent_states[i])
                    break
            else:
                confirmed_states.append(current_state)
    
    states_daily['confirmed_state'] = confirmed_states
    return states_daily


# =============================================================================
# BACKTESTING
# =============================================================================

@dataclass
class TradeResult:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    entry_state: int
    exit_state: int


@dataclass
class BacktestMetrics:
    config_name: str
    entry_buf: float
    exit_buf: float
    confirm_periods: int
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    avg_trade_return: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    avg_holding_days: float
    trades_per_year: float


def run_backtest(states: pd.DataFrame, config_name: str,
                 entry_buf: float, exit_buf: float, 
                 confirm_periods: int) -> BacktestMetrics:
    """Run backtest and compute comprehensive trade quality metrics."""
    
    # Use confirmed state if confirmation is applied
    if 'confirmed_state' in states.columns:
        state_col = 'confirmed_state'
    else:
        state_col = 'state'
    
    prices = states['close'].values
    state_values = states[state_col].values
    dates = states.index
    
    # Track trades
    trades = []
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    
    position = 0
    entry_price = 0
    entry_date = None
    entry_state = None
    
    for i in range(1, len(prices)):
        current_state = state_values[i]
        prev_state = state_values[i-1]
        is_bullish = current_state in BULLISH_STATES
        was_bullish = prev_state in BULLISH_STATES
        
        # Calculate returns if positioned
        if position > 0:
            period_return = (prices[i] - prices[i-1]) / prices[i-1]
            equity += equity * KELLY_FRACTION * period_return
        
        # State change = trade
        if is_bullish != was_bullish:
            if position > 0:
                # Close position
                trade_return = (prices[i] - entry_price) / entry_price
                holding_days = (dates[i] - entry_date).days
                trades.append(TradeResult(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    entry_price=entry_price,
                    exit_price=prices[i],
                    return_pct=trade_return,
                    holding_days=holding_days,
                    entry_state=entry_state,
                    exit_state=current_state
                ))
            
            if is_bullish:
                # Open position
                position = 1
                entry_price = prices[i]
                entry_date = dates[i]
                entry_state = current_state
            else:
                position = 0
        
        equity_curve.append(equity)
    
    # Close final position
    if position > 0:
        trade_return = (prices[-1] - entry_price) / entry_price
        holding_days = (dates[-1] - entry_date).days
        trades.append(TradeResult(
            entry_date=entry_date,
            exit_date=dates[-1],
            entry_price=entry_price,
            exit_price=prices[-1],
            return_pct=trade_return,
            holding_days=holding_days,
            entry_state=entry_state,
            exit_state=state_values[-1]
        ))
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=dates)
    returns = equity_series.pct_change().dropna()
    
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        trade_returns = [t.return_pct for t in trades]
        winners = [r for r in trade_returns if r > 0]
        losers = [r for r in trade_returns if r <= 0]
        
        win_rate = len(winners) / n_trades
        avg_trade = np.mean(trade_returns)
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = np.mean(losers) if losers else 0
        
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_holding = np.mean([t.holding_days for t in trades])
        trades_per_year = n_trades / years if years > 0 else 0
    else:
        win_rate = 0
        avg_trade = 0
        avg_winner = 0
        avg_loser = 0
        profit_factor = 0
        avg_holding = 0
        trades_per_year = 0
    
    return BacktestMetrics(
        config_name=config_name,
        entry_buf=entry_buf,
        exit_buf=exit_buf,
        confirm_periods=confirm_periods,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_trade_return=avg_trade,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        profit_factor=profit_factor,
        avg_holding_days=avg_holding,
        trades_per_year=trades_per_year
    )


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_pair(df_1h: pd.DataFrame, pair: str) -> List[BacktestMetrics]:
    """Test all configurations on a single pair."""
    results = []
    
    # Test hysteresis configurations (no confirmation)
    for entry_buf, exit_buf in HYSTERESIS_CONFIGS:
        config_name = f"E{entry_buf:.1%}_X{exit_buf:.1%}"
        states_daily, states_hourly = compute_states_with_config(df_1h, entry_buf, exit_buf)
        metrics = run_backtest(states_daily, config_name, entry_buf, exit_buf, 0)
        results.append(metrics)
    
    # Test confirmation periods (in hours) with baseline hysteresis
    baseline_entry = 0.015
    baseline_exit = 0.005
    states_daily_base, states_hourly_base = compute_states_with_config(df_1h, baseline_entry, baseline_exit)
    
    for confirm_hours in CONFIRMATION_HOURS:
        if confirm_hours == 0:
            continue  # Already tested above
        config_name = f"E{baseline_entry:.1%}_X{baseline_exit:.1%}_C{confirm_hours}h"
        states = apply_confirmation_filter(states_daily_base, states_hourly_base, confirm_hours)
        metrics = run_backtest(states, config_name, baseline_entry, baseline_exit, confirm_hours)
        results.append(metrics)
    
    # Test wider hysteresis + confirmation
    wider_entry = 0.025
    wider_exit = 0.005
    states_daily_wide, states_hourly_wide = compute_states_with_config(df_1h, wider_entry, wider_exit)
    
    for confirm_hours in [6, 12, 18, 24]:
        config_name = f"E{wider_entry:.1%}_X{wider_exit:.1%}_C{confirm_hours}h"
        states = apply_confirmation_filter(states_daily_wide, states_hourly_wide, confirm_hours)
        metrics = run_backtest(states, config_name, wider_entry, wider_exit, confirm_hours)
        results.append(metrics)
    
    return results


def display_results(all_results: Dict[str, List[BacktestMetrics]]):
    """Display comprehensive results."""
    
    print(f"""
{'='*140}
TRADE QUALITY OPTIMIZATION RESULTS
{'='*140}

Current baseline: Entry 1.5%, Exit 0.5%, No confirmation
Goal: Fewer trades, higher win rate, better Sharpe
""")
    
    # Per-pair results - Hysteresis only
    for pair, results in all_results.items():
        hysteresis_results = [r for r in results if r.confirm_periods == 0]
        
        print(f"""
{'─'*140}
{pair}: HYSTERESIS COMPARISON (No Confirmation)
{'─'*140}
{'Config':<20} {'Trades':<8} {'Win%':<8} {'AvgTrade':<10} {'ProfitF':<10} {'Sharpe':<10} {'Annual':<10} {'MaxDD':<10}
{'─'*140}""")
        
        # Sort by Sharpe
        hysteresis_results = sorted(hysteresis_results, key=lambda x: -x.sharpe_ratio)
        
        for r in hysteresis_results[:10]:
            marker = "***" if r.entry_buf == 0.015 and r.exit_buf == 0.005 else ""
            print(f"{r.config_name:<20} {r.n_trades:<8} {r.win_rate:<8.1%} "
                  f"{r.avg_trade_return:<+10.2%} {r.profit_factor:<10.2f} "
                  f"{r.sharpe_ratio:<10.2f} {r.annual_return:<+10.1%} {r.max_drawdown:<10.1%} {marker}")
    
    # Confirmation results
    for pair, results in all_results.items():
        confirm_results = [r for r in results if r.confirm_periods > 0]
        
        if not confirm_results:
            continue
        
        print(f"""
{'─'*140}
{pair}: CONFIRMATION BARRIER COMPARISON
{'─'*140}
{'Config':<25} {'Trades':<8} {'Win%':<8} {'AvgTrade':<10} {'ProfitF':<10} {'Sharpe':<10} {'Annual':<10} {'AvgHold':<10}
{'─'*140}""")
        
        confirm_results = sorted(confirm_results, key=lambda x: -x.sharpe_ratio)
        
        for r in confirm_results:
            print(f"{r.config_name:<25} {r.n_trades:<8} {r.win_rate:<8.1%} "
                  f"{r.avg_trade_return:<+10.2%} {r.profit_factor:<10.2f} "
                  f"{r.sharpe_ratio:<10.2f} {r.annual_return:<+10.1%} {r.avg_holding_days:<10.1f}")
    
    # Cross-pair summary
    print(f"""

{'='*140}
CROSS-PAIR SUMMARY: BEST CONFIGURATIONS
{'='*140}
""")
    
    # Aggregate results by config
    config_stats = {}
    for pair, results in all_results.items():
        for r in results:
            key = r.config_name
            if key not in config_stats:
                config_stats[key] = {
                    'sharpe': [], 'annual': [], 'win_rate': [], 
                    'trades': [], 'profit_factor': [], 'max_dd': []
                }
            config_stats[key]['sharpe'].append(r.sharpe_ratio)
            config_stats[key]['annual'].append(r.annual_return)
            config_stats[key]['win_rate'].append(r.win_rate)
            config_stats[key]['trades'].append(r.n_trades)
            config_stats[key]['profit_factor'].append(r.profit_factor)
            config_stats[key]['max_dd'].append(r.max_drawdown)
    
    # Calculate averages and rank by Sharpe
    config_avgs = []
    for config, stats in config_stats.items():
        config_avgs.append({
            'config': config,
            'avg_sharpe': np.mean(stats['sharpe']),
            'avg_annual': np.mean(stats['annual']),
            'avg_win_rate': np.mean(stats['win_rate']),
            'avg_trades': np.mean(stats['trades']),
            'avg_pf': np.mean(stats['profit_factor']),
            'avg_dd': np.mean(stats['max_dd']),
        })
    
    config_avgs = sorted(config_avgs, key=lambda x: -x['avg_sharpe'])
    
    print(f"{'Config':<25} {'AvgSharpe':<12} {'AvgAnnual':<12} {'AvgWin%':<10} {'AvgTrades':<12} {'AvgPF':<10} {'AvgDD':<10}")
    print(f"{'─'*100}")
    
    baseline_config = "E1.5%_X0.5%"
    
    for c in config_avgs[:15]:
        marker = "← BASELINE" if c['config'] == baseline_config else ""
        print(f"{c['config']:<25} {c['avg_sharpe']:<12.2f} {c['avg_annual']:<+12.1%} "
              f"{c['avg_win_rate']:<10.1%} {c['avg_trades']:<12.0f} "
              f"{c['avg_pf']:<10.2f} {c['avg_dd']:<10.1%} {marker}")
    
    # Find best config
    best = config_avgs[0]
    baseline = next((c for c in config_avgs if c['config'] == baseline_config), None)
    
    print(f"""

{'='*140}
RECOMMENDATION
{'='*140}

Best configuration: {best['config']}
    Avg Sharpe:     {best['avg_sharpe']:.2f}
    Avg Annual:     {best['avg_annual']:+.1%}
    Avg Win Rate:   {best['avg_win_rate']:.1%}
    Avg Trades:     {best['avg_trades']:.0f}
""")
    
    if baseline:
        sharpe_improvement = (best['avg_sharpe'] - baseline['avg_sharpe']) / baseline['avg_sharpe'] * 100
        trade_reduction = (baseline['avg_trades'] - best['avg_trades']) / baseline['avg_trades'] * 100
        
        print(f"""
vs Baseline ({baseline_config}):
    Sharpe improvement: {sharpe_improvement:+.1f}%
    Trade reduction:    {trade_reduction:+.1f}%
    Win rate change:    {best['avg_win_rate'] - baseline['avg_win_rate']:+.1%}
""")


def main():
    parser = argparse.ArgumentParser(description='Trade Quality Optimization')
    parser.add_argument('--pair', type=str, default=None)
    parser.add_argument('--all-pairs', action='store_true')
    args = parser.parse_args()
    
    print("=" * 140)
    print("TRADE QUALITY OPTIMIZATION: HYSTERESIS & CONFIRMATION BARRIERS")
    print("=" * 140)
    
    print("""
Testing parameters to improve trade quality:

1. HYSTERESIS (Entry/Exit buffers):
   - Wider entry buffer = fewer entries, wait for stronger signals
   - Wider exit buffer = hold positions longer, avoid whipsaw exits

2. CONFIRMATION BARRIERS (in HOURS):
   - State must persist for 1h, 6h, 12h, 18h, 24h, or 48h before acting
   - Filters out brief state changes that reverse quickly

Goal: Fewer trades + higher win rate = better risk-adjusted returns
""")
    
    if args.pair:
        pairs = [args.pair]
    elif args.all_pairs:
        pairs = DEPLOY_PAIRS
    else:
        pairs = ['ETHUSD']
    
    print(f"Pairs: {pairs}\n")
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    all_results = {}
    
    for pair in pairs:
        print(f"Processing {pair}...", end=" ", flush=True)
        df_1h = db.get_ohlcv(pair)
        results = analyze_pair(df_1h, pair)
        all_results[pair] = results
        print(f"done ({len(results)} configs tested)")
    
    display_results(all_results)
    
    print(f"\n{'='*140}")
    print("ANALYSIS COMPLETE")
    print("=" * 140)
    
    return all_results


if __name__ == "__main__":
    results = main()