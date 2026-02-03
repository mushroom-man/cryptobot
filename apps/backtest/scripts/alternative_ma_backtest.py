#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alternative MA Configuration Backtest
=====================================
Compare trading performance of:
    - Baseline: 24h MA updates, 24h sampling (current system)
    - Option 1: Consensus filter (only act when baseline AND faster agree)
    - Option 2: 1h MA updates, 12h sampling

Uses same Kelly criterion position sizing and risk parity allocation as main system.

Usage:
    python alternative_ma_backtest.py --pair ETHUSD
    python alternative_ma_backtest.py --all-pairs
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
from typing import Dict, List, Tuple, Optional
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

ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Trading parameters
INITIAL_CAPITAL = 100000
KELLY_FRACTION = 0.25  # Quarter Kelly for safety

# Bullish states (used for position sizing)
BULLISH_STATES = [8, 9, 10, 11, 12, 13, 14, 15]


# =============================================================================
# REGIME COMPUTATION FUNCTIONS
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_trend_with_hysteresis(close: pd.Series, ma: pd.Series,
                                   entry_buf: float = ENTRY_BUFFER,
                                   exit_buf: float = EXIT_BUFFER) -> List[int]:
    """Compute trend with hysteresis buffers."""
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


def compute_ma_at_frequency(df_1h: pd.DataFrame, ma_period: int, update_freq: str) -> pd.Series:
    """Compute MA with specified update frequency."""
    df_freq = resample_ohlcv(df_1h, update_freq)
    
    if update_freq == '24h':
        ma = df_freq['close'].rolling(ma_period).mean()
    else:
        hours_per_period = {'24h': 24, '12h': 12, '6h': 6, '1h': 1}[update_freq]
        equivalent_periods = (ma_period * 24) // hours_per_period
        ma = df_freq['close'].rolling(equivalent_periods).mean()
    
    ma_hourly = ma.reindex(df_1h.index, method='ffill')
    return ma_hourly


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def compute_baseline_states(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline: 24h MA updates at 24h frequency, sampled daily.
    This is the current production system.
    """
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_daily = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_daily = ma_168h.reindex(df_24h.index, method='ffill')
    
    trend_24h = compute_trend_with_hysteresis(df_24h['close'], ma_24h)
    trend_168h = compute_trend_with_hysteresis(df_24h['close'], ma_168h_daily)
    
    states = pd.DataFrame(index=df_24h.index)
    states['close'] = df_24h['close']
    states['state'] = (
        pd.Series(trend_24h, index=df_24h.index) * 8 +
        pd.Series(trend_168h, index=df_24h.index) * 4 +
        (ma_72h_daily > ma_24h).astype(int) * 2 +
        (ma_168h_daily > ma_24h).astype(int) * 1
    )
    
    return states.dropna()


def compute_option2_states(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Option 2: 1h MA updates, 12h sampling.
    MA is computed hourly for smoothness, but we only check/act every 12 hours.
    """
    # MA updated every hour
    ma_24h = compute_ma_at_frequency(df_1h, MA_PERIOD_24H, '1h')
    
    # 72h and 168h stay at natural frequencies
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
    # Compute hourly states
    trend_24h = compute_trend_with_hysteresis(df_1h['close'], ma_24h)
    trend_168h = compute_trend_with_hysteresis(df_1h['close'], ma_168h_hourly)
    
    states_hourly = pd.DataFrame(index=df_1h.index)
    states_hourly['close'] = df_1h['close']
    states_hourly['state'] = (
        pd.Series(trend_24h, index=df_1h.index) * 8 +
        pd.Series(trend_168h, index=df_1h.index) * 4 +
        (ma_72h_hourly > ma_24h).astype(int) * 2 +
        (ma_168h_hourly > ma_24h).astype(int) * 1
    )
    
    # Sample every 12 hours
    states_12h = states_hourly.resample('12h').first().dropna()
    
    return states_12h


def compute_fast_states(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Fast config: 1h MA updates, 24h sampling (for consensus comparison).
    """
    ma_24h = compute_ma_at_frequency(df_1h, MA_PERIOD_24H, '1h')
    
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_hourly = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_hourly = ma_168h.reindex(df_1h.index, method='ffill')
    
    trend_24h = compute_trend_with_hysteresis(df_1h['close'], ma_24h)
    trend_168h = compute_trend_with_hysteresis(df_1h['close'], ma_168h_hourly)
    
    states_hourly = pd.DataFrame(index=df_1h.index)
    states_hourly['close'] = df_1h['close']
    states_hourly['state'] = (
        pd.Series(trend_24h, index=df_1h.index) * 8 +
        pd.Series(trend_168h, index=df_1h.index) * 4 +
        (ma_72h_hourly > ma_24h).astype(int) * 2 +
        (ma_168h_hourly > ma_24h).astype(int) * 1
    )
    
    # Sample daily
    states_daily = states_hourly.resample('24h').first().dropna()
    
    return states_daily


def compute_consensus_states(baseline_states: pd.DataFrame, fast_states: pd.DataFrame) -> pd.DataFrame:
    """
    Option 1: Consensus filter.
    
    Position sizing based on agreement:
    - Both bullish: Full position
    - Both bearish: No position (or short)
    - Disagree: Half position (or wait)
    """
    # Align indices
    common_idx = baseline_states.index.intersection(fast_states.index)
    
    baseline = baseline_states.loc[common_idx].copy()
    fast = fast_states.loc[common_idx].copy()
    
    baseline['baseline_state'] = baseline['state']
    baseline['fast_state'] = fast['state']
    baseline['baseline_bullish'] = baseline['baseline_state'].isin(BULLISH_STATES)
    baseline['fast_bullish'] = baseline['fast_state'].isin(BULLISH_STATES)
    
    # Consensus levels
    # 2 = both bullish (full position)
    # 1 = disagree (half position)
    # 0 = both bearish (no position)
    def get_consensus(row):
        if row['baseline_bullish'] and row['fast_bullish']:
            return 2  # Both bullish
        elif not row['baseline_bullish'] and not row['fast_bullish']:
            return 0  # Both bearish
        else:
            return 1  # Disagree
    
    baseline['consensus'] = baseline.apply(get_consensus, axis=1)
    
    # Use baseline state for direction, consensus for sizing
    baseline['state'] = baseline['baseline_state']
    
    return baseline


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

@dataclass
class BacktestResult:
    name: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    avg_trade_return: float
    equity_curve: pd.Series


def run_backtest(states: pd.DataFrame, name: str, 
                 use_consensus: bool = False) -> BacktestResult:
    """
    Run backtest on a state series.
    
    Args:
        states: DataFrame with 'close', 'state' columns
        name: Strategy name
        use_consensus: If True, use 'consensus' column for position sizing
    """
    prices = states['close'].values
    state_values = states['state'].values
    dates = states.index
    
    if use_consensus:
        consensus_values = states['consensus'].values
    else:
        consensus_values = None
    
    # Initialize
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    position = 0  # 0 = flat, 1 = long
    position_size = 0
    entry_price = 0
    trades = []
    
    for i in range(1, len(prices)):
        current_state = state_values[i]
        is_bullish = current_state in BULLISH_STATES
        
        # Determine target position
        if use_consensus:
            consensus = consensus_values[i]
            if consensus == 2:  # Both bullish
                target_position = 1.0
            elif consensus == 1:  # Disagree
                target_position = 0.5
            else:  # Both bearish
                target_position = 0.0
        else:
            target_position = 1.0 if is_bullish else 0.0
        
        # Calculate returns if positioned
        if position > 0:
            period_return = (prices[i] - prices[i-1]) / prices[i-1]
            equity += equity * position_size * period_return
        
        # State change = trade
        prev_bullish = state_values[i-1] in BULLISH_STATES
        if is_bullish != prev_bullish:
            if position > 0:
                # Close position
                trade_return = (prices[i] - entry_price) / entry_price
                trades.append(trade_return)
            
            if is_bullish:
                # Open position
                position = 1
                position_size = KELLY_FRACTION * target_position
                entry_price = prices[i]
            else:
                position = 0
                position_size = 0
        elif use_consensus and position > 0:
            # Adjust position size based on consensus (without trade)
            position_size = KELLY_FRACTION * target_position
        
        equity_curve.append(equity)
    
    # Close final position
    if position > 0:
        trade_return = (prices[-1] - entry_price) / entry_price
        trades.append(trade_return)
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=dates)
    returns = equity_series.pct_change().dropna()
    
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # For 12h sampling, adjust Sharpe calculation
    if len(states) > len(returns) * 1.5:  # Rough check for sub-daily
        periods_per_year = len(returns) / years if years > 0 else 252
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
    
    # Max drawdown
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # Trade stats
    n_trades = len(trades)
    win_rate = np.mean([t > 0 for t in trades]) if trades else 0
    avg_trade = np.mean(trades) if trades else 0
    
    return BacktestResult(
        name=name,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_trade_return=avg_trade,
        equity_curve=equity_series
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_pair(df_1h: pd.DataFrame, pair: str) -> Dict[str, BacktestResult]:
    """Run all strategies on a single pair."""
    
    # Compute states for each strategy
    baseline_states = compute_baseline_states(df_1h)
    option2_states = compute_option2_states(df_1h)
    fast_states = compute_fast_states(df_1h)
    consensus_states = compute_consensus_states(baseline_states, fast_states)
    
    results = {}
    
    # Baseline
    results['baseline'] = run_backtest(baseline_states, 'Baseline (24h@24h)')
    
    # Option 2: 1h MA, 12h sampling
    results['option2'] = run_backtest(option2_states, 'Option2 (1h MA, 12h sample)')
    
    # Option 1: Consensus filter
    results['consensus'] = run_backtest(consensus_states, 'Consensus Filter', use_consensus=True)
    
    # Also test: 1h MA, 24h sampling (for comparison)
    results['fast_daily'] = run_backtest(fast_states, 'Fast (1h MA, 24h sample)')
    
    return results


def display_results(all_results: Dict[str, Dict[str, BacktestResult]]):
    """Display comparison of all strategies."""
    
    print(f"""
{'='*120}
STRATEGY COMPARISON: BASELINE vs OPTION 1 (CONSENSUS) vs OPTION 2 (1h MA, 12h SAMPLE)
{'='*120}
""")
    
    # Per-pair results
    for pair, results in all_results.items():
        print(f"""
{'─'*120}
{pair}
{'─'*120}
{'Strategy':<35} {'Total Ret':<12} {'Annual Ret':<12} {'Sharpe':<10} {'Max DD':<12} {'Trades':<10} {'Win Rate':<10}
{'─'*120}""")
        
        for name, result in results.items():
            print(f"{result.name:<35} {result.total_return:>+10.1%}  {result.annual_return:>+10.1%}  "
                  f"{result.sharpe_ratio:>8.2f}  {result.max_drawdown:>10.1%}  "
                  f"{result.n_trades:>8}  {result.win_rate:>8.1%}")
    
    # Cross-pair summary
    print(f"""

{'='*120}
CROSS-PAIR SUMMARY
{'='*120}

SHARPE RATIO COMPARISON
{'─'*80}
{'Pair':<12} {'Baseline':<15} {'Consensus':<15} {'Option2 (12h)':<15} {'Fast (24h)':<15}
{'─'*80}""")
    
    sharpe_by_strategy = {s: [] for s in ['baseline', 'consensus', 'option2', 'fast_daily']}
    
    for pair, results in all_results.items():
        row = f"{pair:<12}"
        for strategy in ['baseline', 'consensus', 'option2', 'fast_daily']:
            sharpe = results[strategy].sharpe_ratio
            sharpe_by_strategy[strategy].append(sharpe)
            row += f" {sharpe:>13.2f}"
        print(row)
    
    # Averages
    print(f"{'─'*80}")
    row = f"{'AVERAGE':<12}"
    for strategy in ['baseline', 'consensus', 'option2', 'fast_daily']:
        avg = np.mean(sharpe_by_strategy[strategy])
        row += f" {avg:>13.2f}"
    print(row)
    
    # Annual return comparison
    print(f"""

ANNUAL RETURN COMPARISON
{'─'*80}
{'Pair':<12} {'Baseline':<15} {'Consensus':<15} {'Option2 (12h)':<15} {'Fast (24h)':<15}
{'─'*80}""")
    
    returns_by_strategy = {s: [] for s in ['baseline', 'consensus', 'option2', 'fast_daily']}
    
    for pair, results in all_results.items():
        row = f"{pair:<12}"
        for strategy in ['baseline', 'consensus', 'option2', 'fast_daily']:
            ret = results[strategy].annual_return
            returns_by_strategy[strategy].append(ret)
            row += f" {ret:>+13.1%}"
        print(row)
    
    print(f"{'─'*80}")
    row = f"{'AVERAGE':<12}"
    for strategy in ['baseline', 'consensus', 'option2', 'fast_daily']:
        avg = np.mean(returns_by_strategy[strategy])
        row += f" {avg:>+13.1%}"
    print(row)
    
    # Max drawdown comparison
    print(f"""

MAX DRAWDOWN COMPARISON
{'─'*80}
{'Pair':<12} {'Baseline':<15} {'Consensus':<15} {'Option2 (12h)':<15} {'Fast (24h)':<15}
{'─'*80}""")
    
    for pair, results in all_results.items():
        row = f"{pair:<12}"
        for strategy in ['baseline', 'consensus', 'option2', 'fast_daily']:
            dd = results[strategy].max_drawdown
            row += f" {dd:>13.1%}"
        print(row)
    
    # Winner summary
    print(f"""

{'='*120}
WINNER BY METRIC
{'='*120}
""")
    
    wins = {s: {'sharpe': 0, 'return': 0, 'drawdown': 0} for s in ['baseline', 'consensus', 'option2', 'fast_daily']}
    
    for pair, results in all_results.items():
        # Sharpe winner
        best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        wins[best_sharpe[0]]['sharpe'] += 1
        
        # Return winner
        best_return = max(results.items(), key=lambda x: x[1].annual_return)
        wins[best_return[0]]['return'] += 1
        
        # Drawdown winner (least negative)
        best_dd = max(results.items(), key=lambda x: x[1].max_drawdown)
        wins[best_dd[0]]['drawdown'] += 1
    
    print(f"{'Strategy':<25} {'Best Sharpe':<15} {'Best Return':<15} {'Best Drawdown':<15}")
    print(f"{'─'*70}")
    for strategy in ['baseline', 'consensus', 'option2', 'fast_daily']:
        name = results[strategy].name[:25]
        print(f"{name:<25} {wins[strategy]['sharpe']:>13}  {wins[strategy]['return']:>13}  {wins[strategy]['drawdown']:>13}")
    
    print(f"""

INTERPRETATION
{'─'*80}
- Baseline (24h@24h): Current production system
- Consensus: Only full position when baseline AND fast config both bullish
- Option2 (12h): MA updates hourly, check/act every 12 hours
- Fast (24h): MA updates hourly, check/act daily

If Consensus wins → Filtering noise is valuable
If Option2 wins → Faster MA + less frequent trading is optimal
If Fast wins → Faster MA alone helps
If Baseline wins → Current system is already optimal
""")


def main():
    parser = argparse.ArgumentParser(description='Alternative MA Configuration Backtest')
    parser.add_argument('--pair', type=str, default=None)
    parser.add_argument('--all-pairs', action='store_true')
    args = parser.parse_args()
    
    print("=" * 120)
    print("ALTERNATIVE MA CONFIGURATION BACKTEST")
    print("=" * 120)
    
    print("""
Testing three approaches:
1. Baseline: 24h MA updates, 24h sampling (current system)
2. Option 1 (Consensus): Full position only when baseline AND fast agree on bullish
3. Option 2: 1h MA updates, 12h sampling

All use same Kelly fraction (0.25) and state-based position sizing.
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
        print("done")
    
    display_results(all_results)
    
    print(f"\n{'='*120}")
    print("ANALYSIS COMPLETE")
    print("=" * 120)
    
    return all_results


if __name__ == "__main__":
    results = main()