#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Confirmation Testing
=============================
Test different confirmation thresholds based on transition type:
- Regime changes (BULL↔BEAR): Use confirmation (filter noisy reversals)
- Within-regime (BULL→BULL, BEAR→BEAR): Act immediately (don't filter good signals)

Based on findings:
- BULL_TO_BEAR filtered: -8.13%, kept: -1.14% → confirmation helps +6.99%
- BEAR_TO_BULL filtered: -2.11%, kept: +0.73% → confirmation helps +2.84%
- BULL_TO_BULL filtered: +2.74%, kept: +0.82% → confirmation hurts -1.92%
- BEAR_TO_BEAR filtered: +0.59%, kept: -0.04% → confirmation hurts -0.63%

Usage:
    python adaptive_confirmation_test.py --all-pairs
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
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

BULLISH_STATES = set([8, 9, 10, 11, 12, 13, 14, 15])
BEARISH_STATES = set([0, 1, 2, 3, 4, 5, 6, 7])

INITIAL_CAPITAL = 100000
KELLY_FRACTION = 0.25

# Configurations to test
# Format: (entry_buf, exit_buf, name, confirmation_config)
# confirmation_config: either int (uniform) or dict with 'regime_change' and 'within_regime'
CONFIGS_TO_TEST = [
    # Baselines for comparison
    (0.015, 0.005, "Baseline (E1.5%)", {'regime_change': 0, 'within_regime': 0}),
    (0.025, 0.005, "E2.5%_C3h (uniform)", {'regime_change': 3, 'within_regime': 3}),
    (0.020, 0.005, "E2.0%_C3h (uniform)", {'regime_change': 3, 'within_regime': 3}),
    
    # Adaptive: Confirmation only for regime changes
    (0.015, 0.005, "E1.5%_Adaptive_3h/0h", {'regime_change': 3, 'within_regime': 0}),
    (0.020, 0.005, "E2.0%_Adaptive_3h/0h", {'regime_change': 3, 'within_regime': 0}),
    (0.025, 0.005, "E2.5%_Adaptive_3h/0h", {'regime_change': 3, 'within_regime': 0}),
    
    # Adaptive with 6h for regime changes
    (0.020, 0.005, "E2.0%_Adaptive_6h/0h", {'regime_change': 6, 'within_regime': 0}),
    (0.025, 0.005, "E2.5%_Adaptive_6h/0h", {'regime_change': 6, 'within_regime': 0}),
    
    # Adaptive with 1h for within-regime (slight filter)
    (0.020, 0.005, "E2.0%_Adaptive_3h/1h", {'regime_change': 3, 'within_regime': 1}),
    (0.025, 0.005, "E2.5%_Adaptive_3h/1h", {'regime_change': 3, 'within_regime': 1}),
    
    # Aggressive: No confirmation for within-regime, 6h for changes
    (0.025, 0.005, "E2.5%_Adaptive_6h/1h", {'regime_change': 6, 'within_regime': 1}),
]


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
            if price < ma_val * (1 - exit_buf) and price < ma_val * (1 - entry_buf):
                current = 0
        else:
            if price > ma_val * (1 + exit_buf) and price > ma_val * (1 + entry_buf):
                current = 1
        trend.append(current)
    return trend


def compute_states_hourly(df_1h: pd.DataFrame, entry_buf: float, 
                          exit_buf: float) -> pd.DataFrame:
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
    
    trend_24h = compute_trend_with_hysteresis(df_1h['close'], ma_24h_hourly, entry_buf, exit_buf)
    trend_168h = compute_trend_with_hysteresis(df_1h['close'], ma_168h_hourly, entry_buf, exit_buf)
    
    states = pd.DataFrame(index=df_1h.index)
    states['close'] = df_1h['close']
    states['state'] = (
        pd.Series(trend_24h, index=df_1h.index) * 8 +
        pd.Series(trend_168h, index=df_1h.index) * 4 +
        (ma_72h_hourly > ma_24h_hourly).astype(int) * 2 +
        (ma_168h_hourly > ma_24h_hourly).astype(int) * 1
    )
    
    return states.dropna()


def is_regime_change(from_state: int, to_state: int) -> bool:
    """Check if transition crosses between BULL and BEAR regimes."""
    from_bull = from_state in BULLISH_STATES
    to_bull = to_state in BULLISH_STATES
    return from_bull != to_bull


def apply_adaptive_confirmation(states_hourly: pd.DataFrame, 
                                 confirm_config: Dict) -> pd.DataFrame:
    """
    Apply adaptive confirmation based on transition type.
    
    confirm_config: {
        'regime_change': hours to confirm for BULL↔BEAR transitions,
        'within_regime': hours to confirm for BULL→BULL or BEAR→BEAR
    }
    """
    regime_change_hours = confirm_config['regime_change']
    within_regime_hours = confirm_config['within_regime']
    
    # If both are 0, just resample to daily
    if regime_change_hours == 0 and within_regime_hours == 0:
        return states_hourly.resample('24h').first().dropna()
    
    states = states_hourly.copy()
    state_values = states['state'].values
    
    confirmed_states = []
    current_confirmed = state_values[0]
    pending_state = None
    pending_count = 0
    required_confirmation = 0
    
    for i, state in enumerate(state_values):
        if state == current_confirmed:
            # Still in confirmed state, reset pending
            pending_state = None
            pending_count = 0
            confirmed_states.append(current_confirmed)
        elif state == pending_state:
            # Continuing in pending state
            pending_count += 1
            if pending_count >= required_confirmation:
                # Confirmed! Switch to new state
                current_confirmed = state
                pending_state = None
                pending_count = 0
            confirmed_states.append(current_confirmed)
        else:
            # New pending state - determine required confirmation
            pending_state = state
            pending_count = 1
            
            # Determine confirmation requirement based on transition type
            if is_regime_change(current_confirmed, state):
                required_confirmation = regime_change_hours
            else:
                required_confirmation = within_regime_hours
            
            # If no confirmation required, switch immediately
            if required_confirmation == 0:
                current_confirmed = state
                pending_state = None
                pending_count = 0
            
            confirmed_states.append(current_confirmed)
    
    states['confirmed_state'] = confirmed_states
    
    # Resample to daily for trading
    daily = states.resample('24h').first().dropna()
    return daily


# =============================================================================
# BACKTESTING
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
    profit_factor: float
    n_regime_changes: int
    n_within_regime: int


def run_backtest(states: pd.DataFrame, name: str) -> BacktestResult:
    """Run backtest on states with confirmed_state column."""
    state_col = 'confirmed_state' if 'confirmed_state' in states.columns else 'state'
    
    prices = states['close'].values
    state_values = states[state_col].values
    dates = states.index
    
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    trades = []
    
    position = 0
    entry_price = 0
    entry_state = None
    
    n_regime_changes = 0
    n_within_regime = 0
    
    for i in range(1, len(prices)):
        current_state = int(state_values[i])
        prev_state = int(state_values[i-1])
        is_bullish = current_state in BULLISH_STATES
        was_bullish = prev_state in BULLISH_STATES
        
        if position > 0:
            period_return = (prices[i] - prices[i-1]) / prices[i-1]
            equity += equity * KELLY_FRACTION * period_return
        
        # Check for state change
        if current_state != prev_state:
            # Track transition type
            if is_regime_change(prev_state, current_state):
                n_regime_changes += 1
            else:
                n_within_regime += 1
            
            # Trade if crossing bull/bear boundary
            if is_bullish != was_bullish:
                if position > 0:
                    trade_return = (prices[i] - entry_price) / entry_price
                    trades.append(trade_return)
                
                if is_bullish:
                    position = 1
                    entry_price = prices[i]
                    entry_state = current_state
                else:
                    position = 0
        
        equity_curve.append(equity)
    
    if position > 0:
        trade_return = (prices[-1] - entry_price) / entry_price
        trades.append(trade_return)
    
    # Metrics
    equity_series = pd.Series(equity_curve, index=dates)
    returns = equity_series.pct_change().dropna()
    
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    n_trades = len(trades)
    if n_trades > 0:
        winners = [t for t in trades if t > 0]
        losers = [t for t in trades if t <= 0]
        win_rate = len(winners) / n_trades
        avg_trade = np.mean(trades)
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0.001
        profit_factor = gross_profit / gross_loss
    else:
        win_rate = avg_trade = profit_factor = 0
    
    return BacktestResult(
        name=name,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_trade_return=avg_trade,
        profit_factor=profit_factor,
        n_regime_changes=n_regime_changes,
        n_within_regime=n_within_regime
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_pair(df_1h: pd.DataFrame, pair: str) -> Dict[str, BacktestResult]:
    """Run all configurations on a single pair."""
    results = {}
    
    for entry_buf, exit_buf, name, confirm_config in CONFIGS_TO_TEST:
        states_hourly = compute_states_hourly(df_1h, entry_buf, exit_buf)
        states_daily = apply_adaptive_confirmation(states_hourly, confirm_config)
        backtest = run_backtest(states_daily, name)
        results[name] = backtest
    
    return results


def display_results(all_results: Dict[str, Dict[str, BacktestResult]]):
    """Display comprehensive results."""
    
    print(f"""
{'='*140}
ADAPTIVE CONFIRMATION TEST RESULTS
{'='*140}

Strategy: Use DIFFERENT confirmation thresholds for:
- Regime changes (BULL↔BEAR): Wait for confirmation (filter noise)
- Within-regime (BULL→BULL, BEAR→BEAR): Act immediately (capture moves)
""")
    
    # Collect all metrics
    all_configs = [c[2] for c in CONFIGS_TO_TEST]
    
    # Sharpe ratios
    print(f"\nSHARPE RATIO:")
    print(f"{'Config':<30} ", end="")
    for pair in all_results.keys():
        print(f"{pair:<10} ", end="")
    print(f"{'AVERAGE':<10}")
    print("─" * 120)
    
    config_sharpes = defaultdict(list)
    for config_name in all_configs:
        print(f"{config_name:<30} ", end="")
        for pair, results in all_results.items():
            sharpe = results[config_name].sharpe_ratio
            config_sharpes[config_name].append(sharpe)
            print(f"{sharpe:<10.2f} ", end="")
        print(f"{np.mean(config_sharpes[config_name]):<10.2f}")
    
    # Annual returns
    print(f"\nANNUAL RETURN:")
    print(f"{'Config':<30} ", end="")
    for pair in all_results.keys():
        print(f"{pair:<10} ", end="")
    print(f"{'AVERAGE':<10}")
    print("─" * 120)
    
    config_returns = defaultdict(list)
    for config_name in all_configs:
        print(f"{config_name:<30} ", end="")
        for pair, results in all_results.items():
            ret = results[config_name].annual_return
            config_returns[config_name].append(ret)
            print(f"{ret:<+10.1%}", end="")
        print(f"{np.mean(config_returns[config_name]):<+10.1%}")
    
    # Win rates
    print(f"\nWIN RATE:")
    print(f"{'Config':<30} ", end="")
    for pair in all_results.keys():
        print(f"{pair:<10} ", end="")
    print(f"{'AVERAGE':<10}")
    print("─" * 120)
    
    config_winrates = defaultdict(list)
    for config_name in all_configs:
        print(f"{config_name:<30} ", end="")
        for pair, results in all_results.items():
            wr = results[config_name].win_rate
            config_winrates[config_name].append(wr)
            print(f"{wr:<10.1%} ", end="")
        print(f"{np.mean(config_winrates[config_name]):<10.1%}")
    
    # Trade counts
    print(f"\nTRADE COUNT:")
    print(f"{'Config':<30} ", end="")
    for pair in all_results.keys():
        print(f"{pair:<10} ", end="")
    print(f"{'AVERAGE':<10}")
    print("─" * 120)
    
    config_trades = defaultdict(list)
    for config_name in all_configs:
        print(f"{config_name:<30} ", end="")
        for pair, results in all_results.items():
            n = results[config_name].n_trades
            config_trades[config_name].append(n)
            print(f"{n:<10} ", end="")
        print(f"{np.mean(config_trades[config_name]):<10.0f}")
    
    # Max Drawdown
    print(f"\nMAX DRAWDOWN:")
    print(f"{'Config':<30} ", end="")
    for pair in all_results.keys():
        print(f"{pair:<10} ", end="")
    print(f"{'AVERAGE':<10}")
    print("─" * 120)
    
    config_dd = defaultdict(list)
    for config_name in all_configs:
        print(f"{config_name:<30} ", end="")
        for pair, results in all_results.items():
            dd = results[config_name].max_drawdown
            config_dd[config_name].append(dd)
            print(f"{dd:<10.1%} ", end="")
        print(f"{np.mean(config_dd[config_name]):<10.1%}")
    
    # Summary ranking
    print(f"""

{'='*140}
RANKING BY SHARPE RATIO
{'='*140}
""")
    
    ranked = sorted(config_sharpes.items(), key=lambda x: np.mean(x[1]), reverse=True)
    
    print(f"{'Rank':<6} {'Config':<30} {'Avg Sharpe':<12} {'Avg Return':<12} {'Avg Win%':<12} {'Avg DD':<12}")
    print("─" * 90)
    
    baseline_sharpe = np.mean(config_sharpes["Baseline (E1.5%)"])
    uniform_sharpe = np.mean(config_sharpes["E2.5%_C3h (uniform)"])
    
    for i, (config_name, sharpes) in enumerate(ranked, 1):
        avg_sharpe = np.mean(sharpes)
        avg_return = np.mean(config_returns[config_name])
        avg_wr = np.mean(config_winrates[config_name])
        avg_dd = np.mean(config_dd[config_name])
        
        if config_name == "Baseline (E1.5%)":
            marker = " ← BASELINE"
        elif config_name == "E2.5%_C3h (uniform)":
            marker = " ← BEST UNIFORM"
        elif i == 1:
            marker = " ← BEST OVERALL"
        else:
            marker = ""
        
        print(f"{i:<6} {config_name:<30} {avg_sharpe:<12.3f} {avg_return:<+12.1%} "
              f"{avg_wr:<12.1%} {avg_dd:<12.1%}{marker}")
    
    # Improvement analysis
    best_config = ranked[0][0]
    best_sharpe = np.mean(ranked[0][1])
    
    print(f"""

{'='*140}
IMPROVEMENT ANALYSIS
{'='*140}

Best configuration: {best_config}
    Avg Sharpe: {best_sharpe:.3f}

vs Baseline (E1.5%, no confirmation):
    Sharpe improvement: {(best_sharpe - baseline_sharpe) / baseline_sharpe * 100:+.1f}%
    
vs Best Uniform (E2.5%_C3h):
    Sharpe improvement: {(best_sharpe - uniform_sharpe) / uniform_sharpe * 100:+.1f}%
""")
    
    # Check if adaptive beats uniform
    best_adaptive = None
    for config_name, sharpes in ranked:
        if "Adaptive" in config_name:
            best_adaptive = (config_name, np.mean(sharpes))
            break
    
    if best_adaptive:
        print(f"""
Best Adaptive: {best_adaptive[0]}
    Avg Sharpe: {best_adaptive[1]:.3f}
    
vs Best Uniform: {(best_adaptive[1] - uniform_sharpe) / uniform_sharpe * 100:+.1f}%
""")
    
    print(f"""

{'='*140}
INTERPRETATION
{'='*140}

ADAPTIVE CONFIRMATION:
- Uses different thresholds: X hours for regime changes, Y hours for within-regime
- "3h/0h" means: 3h confirmation for BULL↔BEAR, 0h (immediate) for BULL→BULL

IF ADAPTIVE BEATS UNIFORM:
    → The insight is correct: regime changes need filtering, within-regime doesn't
    → Recommendation: Implement adaptive confirmation

IF UNIFORM BEATS ADAPTIVE:
    → Simpler approach works better
    → Within-regime filtering might still add value
    → Recommendation: Use uniform E2.5%_C3h
""")


def main():
    parser = argparse.ArgumentParser(description='Adaptive Confirmation Test')
    parser.add_argument('--pair', type=str, default=None)
    parser.add_argument('--all-pairs', action='store_true')
    args = parser.parse_args()
    
    print("=" * 140)
    print("ADAPTIVE CONFIRMATION TEST")
    print("=" * 140)
    
    print("""
Testing adaptive confirmation thresholds:
- Regime changes (BULL↔BEAR): Use longer confirmation (filter false reversals)
- Within-regime (BULL→BULL, BEAR→BEAR): Use shorter/no confirmation (capture moves)

Hypothesis: This should beat uniform confirmation because:
- Regime changes had +6.99% improvement from confirmation
- Within-regime had -1.92% cost from confirmation
""")
    
    if args.pair:
        pairs = [args.pair]
    elif args.all_pairs:
        pairs = DEPLOY_PAIRS
    else:
        pairs = ['ETHUSD']
    
    print(f"\nPairs: {pairs}\n")
    
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
    
    print(f"\n{'='*140}")
    print("ANALYSIS COMPLETE")
    print("=" * 140)
    
    return all_results


if __name__ == "__main__":
    results = main()