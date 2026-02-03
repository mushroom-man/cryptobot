#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Optimization & Transition Analysis
============================================
1. Test 2.0% entry + 3h confirmation combined
2. Analyze which specific transitions benefit most from confirmation

Usage:
    python combined_optimization_analysis.py --all-pairs
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

BULLISH_STATES = [8, 9, 10, 11, 12, 13, 14, 15]
BEARISH_STATES = [0, 1, 2, 3, 4, 5, 6, 7]

# Configurations to test
CONFIGS_TO_TEST = [
    # (entry_buf, exit_buf, confirm_hours, name)
    (0.015, 0.005, 0, "Baseline"),
    (0.015, 0.005, 1, "E1.5%_C1h"),
    (0.015, 0.005, 3, "E1.5%_C3h"),
    (0.020, 0.005, 0, "E2.0%_NoC"),
    (0.020, 0.005, 1, "E2.0%_C1h"),
    (0.020, 0.005, 3, "E2.0%_C3h"),
    (0.020, 0.005, 6, "E2.0%_C6h"),
    (0.025, 0.005, 3, "E2.5%_C3h"),
]

INITIAL_CAPITAL = 100000
KELLY_FRACTION = 0.25


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
    
    # Resample to hourly
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


def apply_confirmation_filter(states_hourly: pd.DataFrame, 
                               confirmation_hours: int) -> pd.DataFrame:
    """
    Apply confirmation: state must persist for N hours before acting.
    Returns daily states with confirmed signals.
    """
    if confirmation_hours == 0:
        # No confirmation - just resample to daily
        return states_hourly.resample('24h').first().dropna()
    
    states = states_hourly.copy()
    state_values = states['state'].values
    
    # Track confirmed states
    confirmed_states = []
    current_confirmed = state_values[0]
    pending_state = None
    pending_count = 0
    
    for state in state_values:
        if state == current_confirmed:
            pending_state = None
            pending_count = 0
            confirmed_states.append(current_confirmed)
        elif state == pending_state:
            pending_count += 1
            if pending_count >= confirmation_hours:
                current_confirmed = state
                pending_state = None
                pending_count = 0
            confirmed_states.append(current_confirmed)
        else:
            pending_state = state
            pending_count = 1
            confirmed_states.append(current_confirmed)
    
    states['confirmed_state'] = confirmed_states
    
    # Resample to daily for trading
    daily = states.resample('24h').first().dropna()
    return daily


# =============================================================================
# TRANSITION TRACKING
# =============================================================================

@dataclass
class TransitionRecord:
    date: pd.Timestamp
    from_state: int
    to_state: int
    from_bullish: bool
    to_bullish: bool
    direction: str  # 'BULL_TO_BEAR', 'BEAR_TO_BULL', etc.
    entry_price: float
    hours_to_confirm: int  # 0 if no confirmation filter
    was_filtered: bool  # True if confirmation prevented this trade
    return_1d: float
    return_3d: float
    return_7d: float


def extract_transitions_with_confirmation(states_hourly: pd.DataFrame,
                                           confirmation_hours: int) -> List[TransitionRecord]:
    """
    Extract all transitions and track which ones would be filtered by confirmation.
    """
    state_values = states_hourly['state'].values
    close_values = states_hourly['close'].values
    dates = states_hourly.index
    
    transitions = []
    
    # Track raw transitions (no filter)
    for i in range(1, len(state_values)):
        if state_values[i] != state_values[i-1]:
            from_state = int(state_values[i-1])
            to_state = int(state_values[i])
            from_bull = from_state in BULLISH_STATES
            to_bull = to_state in BULLISH_STATES
            
            if from_bull and not to_bull:
                direction = 'BULL_TO_BEAR'
            elif not from_bull and to_bull:
                direction = 'BEAR_TO_BULL'
            elif from_bull and to_bull:
                direction = 'BULL_TO_BULL'
            else:
                direction = 'BEAR_TO_BEAR'
            
            # Check if state persists for confirmation_hours
            hours_persisted = 0
            was_filtered = False
            
            if confirmation_hours > 0:
                for j in range(i, min(i + confirmation_hours + 1, len(state_values))):
                    if state_values[j] == to_state:
                        hours_persisted += 1
                    else:
                        break
                was_filtered = hours_persisted < confirmation_hours
            
            # Calculate returns
            return_1d = return_3d = return_7d = np.nan
            hours_1d, hours_3d, hours_7d = 24, 72, 168
            
            if i + hours_1d < len(close_values):
                return_1d = (close_values[i + hours_1d] - close_values[i]) / close_values[i]
            if i + hours_3d < len(close_values):
                return_3d = (close_values[i + hours_3d] - close_values[i]) / close_values[i]
            if i + hours_7d < len(close_values):
                return_7d = (close_values[i + hours_7d] - close_values[i]) / close_values[i]
            
            transitions.append(TransitionRecord(
                date=dates[i],
                from_state=from_state,
                to_state=to_state,
                from_bullish=from_bull,
                to_bullish=to_bull,
                direction=direction,
                entry_price=close_values[i],
                hours_to_confirm=hours_persisted,
                was_filtered=was_filtered,
                return_1d=return_1d,
                return_3d=return_3d,
                return_7d=return_7d
            ))
    
    return transitions


def analyze_transitions_by_type(transitions: List[TransitionRecord], 
                                 confirmation_hours: int) -> Dict:
    """Analyze which transition types benefit most from confirmation."""
    
    # Group by transition type
    by_type = defaultdict(list)
    for t in transitions:
        key = (t.from_state, t.to_state)
        by_type[key].append(t)
    
    results = {}
    
    for (from_state, to_state), trans_list in by_type.items():
        if len(trans_list) < 10:
            continue
        
        # Split into filtered vs kept
        filtered = [t for t in trans_list if t.was_filtered]
        kept = [t for t in trans_list if not t.was_filtered]
        
        # Calculate signal returns (positive = good signal)
        from_bull = from_state in BULLISH_STATES
        to_bull = to_state in BULLISH_STATES
        
        def signal_return(raw_return, to_bullish):
            if pd.isna(raw_return):
                return np.nan
            return raw_return if to_bullish else -raw_return
        
        # Stats for all transitions
        all_returns_3d = [signal_return(t.return_3d, t.to_bullish) for t in trans_list]
        all_returns_3d = [r for r in all_returns_3d if not pd.isna(r)]
        
        # Stats for filtered transitions (would have been bad trades?)
        filtered_returns_3d = [signal_return(t.return_3d, t.to_bullish) for t in filtered]
        filtered_returns_3d = [r for r in filtered_returns_3d if not pd.isna(r)]
        
        # Stats for kept transitions
        kept_returns_3d = [signal_return(t.return_3d, t.to_bullish) for t in kept]
        kept_returns_3d = [r for r in kept_returns_3d if not pd.isna(r)]
        
        results[(from_state, to_state)] = {
            'from_state': from_state,
            'to_state': to_state,
            'direction': trans_list[0].direction,
            'n_total': len(trans_list),
            'n_filtered': len(filtered),
            'n_kept': len(kept),
            'filter_rate': len(filtered) / len(trans_list) if trans_list else 0,
            'all_mean_return': np.mean(all_returns_3d) if all_returns_3d else np.nan,
            'all_win_rate': np.mean([r > 0 for r in all_returns_3d]) if all_returns_3d else np.nan,
            'filtered_mean_return': np.mean(filtered_returns_3d) if filtered_returns_3d else np.nan,
            'filtered_win_rate': np.mean([r > 0 for r in filtered_returns_3d]) if filtered_returns_3d else np.nan,
            'kept_mean_return': np.mean(kept_returns_3d) if kept_returns_3d else np.nan,
            'kept_win_rate': np.mean([r > 0 for r in kept_returns_3d]) if kept_returns_3d else np.nan,
        }
    
    return results


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


def run_backtest(states: pd.DataFrame, name: str, use_confirmed: bool = False) -> BacktestResult:
    """Run backtest on states."""
    state_col = 'confirmed_state' if use_confirmed and 'confirmed_state' in states.columns else 'state'
    
    prices = states['close'].values
    state_values = states[state_col].values
    dates = states.index
    
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    trades = []
    
    position = 0
    entry_price = 0
    
    for i in range(1, len(prices)):
        current_state = state_values[i]
        prev_state = state_values[i-1]
        is_bullish = current_state in BULLISH_STATES
        was_bullish = prev_state in BULLISH_STATES
        
        if position > 0:
            period_return = (prices[i] - prices[i-1]) / prices[i-1]
            equity += equity * KELLY_FRACTION * period_return
        
        if is_bullish != was_bullish:
            if position > 0:
                trade_return = (prices[i] - entry_price) / entry_price
                trades.append(trade_return)
            
            if is_bullish:
                position = 1
                entry_price = prices[i]
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
        profit_factor=profit_factor
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_pair(df_1h: pd.DataFrame, pair: str) -> Dict:
    """Full analysis for a single pair."""
    
    results = {
        'backtests': {},
        'transition_analysis': {}
    }
    
    # Run backtests for all configs
    for entry_buf, exit_buf, confirm_hours, name in CONFIGS_TO_TEST:
        states_hourly = compute_states_hourly(df_1h, entry_buf, exit_buf)
        
        if confirm_hours > 0:
            states_daily = apply_confirmation_filter(states_hourly, confirm_hours)
            backtest = run_backtest(states_daily, name, use_confirmed=True)
        else:
            states_daily = states_hourly.resample('24h').first().dropna()
            backtest = run_backtest(states_daily, name)
        
        results['backtests'][name] = backtest
    
    # Transition analysis with 3h confirmation
    states_hourly = compute_states_hourly(df_1h, 0.020, 0.005)  # Use 2.0% entry
    transitions = extract_transitions_with_confirmation(states_hourly, confirmation_hours=3)
    transition_analysis = analyze_transitions_by_type(transitions, confirmation_hours=3)
    results['transition_analysis'] = transition_analysis
    
    return results


def display_results(all_results: Dict[str, Dict]):
    """Display comprehensive results."""
    
    print(f"""
{'='*140}
PART 1: COMBINED OPTIMIZATION (2.0% Entry + Confirmation)
{'='*140}
""")
    
    # Backtest comparison
    print(f"{'Config':<20} ", end="")
    for pair in all_results.keys():
        print(f"{pair:<12} ", end="")
    print(f"{'AVERAGE':<12}")
    print("─" * 120)
    
    # Sharpe ratios
    print("\nSHARPE RATIO:")
    for config_name in [c[3] for c in CONFIGS_TO_TEST]:
        print(f"{config_name:<20} ", end="")
        sharpes = []
        for pair, results in all_results.items():
            sharpe = results['backtests'][config_name].sharpe_ratio
            sharpes.append(sharpe)
            print(f"{sharpe:<12.2f} ", end="")
        print(f"{np.mean(sharpes):<12.2f}")
    
    # Annual returns
    print("\nANNUAL RETURN:")
    for config_name in [c[3] for c in CONFIGS_TO_TEST]:
        print(f"{config_name:<20} ", end="")
        returns = []
        for pair, results in all_results.items():
            ret = results['backtests'][config_name].annual_return
            returns.append(ret)
            print(f"{ret:<+12.1%} ", end="")
        print(f"{np.mean(returns):<+12.1%}")
    
    # Win rates
    print("\nWIN RATE:")
    for config_name in [c[3] for c in CONFIGS_TO_TEST]:
        print(f"{config_name:<20} ", end="")
        win_rates = []
        for pair, results in all_results.items():
            wr = results['backtests'][config_name].win_rate
            win_rates.append(wr)
            print(f"{wr:<12.1%} ", end="")
        print(f"{np.mean(win_rates):<12.1%}")
    
    # Trade counts
    print("\nTRADE COUNT:")
    for config_name in [c[3] for c in CONFIGS_TO_TEST]:
        print(f"{config_name:<20} ", end="")
        trades = []
        for pair, results in all_results.items():
            n = results['backtests'][config_name].n_trades
            trades.append(n)
            print(f"{n:<12} ", end="")
        print(f"{np.mean(trades):<12.0f}")
    
    # Best config summary
    print(f"""

{'='*140}
BEST CONFIGURATION SUMMARY
{'='*140}
""")
    
    config_scores = defaultdict(lambda: {'sharpe': [], 'return': [], 'win_rate': []})
    for pair, results in all_results.items():
        for config_name, bt in results['backtests'].items():
            config_scores[config_name]['sharpe'].append(bt.sharpe_ratio)
            config_scores[config_name]['return'].append(bt.annual_return)
            config_scores[config_name]['win_rate'].append(bt.win_rate)
    
    print(f"{'Config':<20} {'Avg Sharpe':<12} {'Avg Return':<12} {'Avg Win%':<12}")
    print("─" * 60)
    
    sorted_configs = sorted(config_scores.items(), 
                           key=lambda x: np.mean(x[1]['sharpe']), reverse=True)
    
    for config_name, scores in sorted_configs:
        marker = " ← BEST" if config_name == sorted_configs[0][0] else ""
        marker = " ← BASELINE" if config_name == "Baseline" else marker
        print(f"{config_name:<20} {np.mean(scores['sharpe']):<12.2f} "
              f"{np.mean(scores['return']):<+12.1%} {np.mean(scores['win_rate']):<12.1%}{marker}")
    
    # Part 2: Transition Analysis
    print(f"""

{'='*140}
PART 2: WHICH TRANSITIONS BENEFIT MOST FROM 3h CONFIRMATION?
{'='*140}

Analysis: With 3h confirmation filter on 2.0% entry buffer
- "Filtered" = transitions that didn't persist 3h (would have been skipped)
- "Kept" = transitions that persisted 3h (would have been traded)
- If filtered_return < kept_return → confirmation is helping!
""")
    
    # Aggregate transition analysis across pairs
    aggregated = defaultdict(lambda: {
        'n_total': 0, 'n_filtered': 0, 'n_kept': 0,
        'filtered_returns': [], 'kept_returns': [],
        'filtered_wins': [], 'kept_wins': []
    })
    
    for pair, results in all_results.items():
        for key, data in results['transition_analysis'].items():
            agg = aggregated[key]
            agg['n_total'] += data['n_total']
            agg['n_filtered'] += data['n_filtered']
            agg['n_kept'] += data['n_kept']
            agg['direction'] = data['direction']
            if not pd.isna(data['filtered_mean_return']):
                agg['filtered_returns'].append(data['filtered_mean_return'])
            if not pd.isna(data['kept_mean_return']):
                agg['kept_returns'].append(data['kept_mean_return'])
            if not pd.isna(data['filtered_win_rate']):
                agg['filtered_wins'].append(data['filtered_win_rate'])
            if not pd.isna(data['kept_win_rate']):
                agg['kept_wins'].append(data['kept_win_rate'])
    
    # Sort by improvement (kept - filtered return)
    transition_improvements = []
    for key, agg in aggregated.items():
        if agg['filtered_returns'] and agg['kept_returns']:
            filtered_avg = np.mean(agg['filtered_returns'])
            kept_avg = np.mean(agg['kept_returns'])
            improvement = kept_avg - filtered_avg
            transition_improvements.append({
                'from': key[0],
                'to': key[1],
                'direction': agg['direction'],
                'n_total': agg['n_total'],
                'n_filtered': agg['n_filtered'],
                'filter_rate': agg['n_filtered'] / agg['n_total'] if agg['n_total'] > 0 else 0,
                'filtered_return': filtered_avg,
                'kept_return': kept_avg,
                'improvement': improvement,
                'filtered_win': np.mean(agg['filtered_wins']) if agg['filtered_wins'] else np.nan,
                'kept_win': np.mean(agg['kept_wins']) if agg['kept_wins'] else np.nan,
            })
    
    transition_improvements = sorted(transition_improvements, key=lambda x: -x['improvement'])
    
    print(f"\nTRANSITIONS THAT BENEFIT MOST FROM CONFIRMATION (3d signal return):")
    print(f"{'From→To':<12} {'Direction':<15} {'N':<8} {'Filter%':<10} {'Filtered':<12} {'Kept':<12} {'Improve':<12}")
    print("─" * 90)
    
    for t in transition_improvements[:15]:
        if t['n_total'] >= 50:  # Only show significant transitions
            print(f"{t['from']:>2}→{t['to']:<7} {t['direction']:<15} {t['n_total']:<8} "
                  f"{t['filter_rate']:<10.1%} {t['filtered_return']:<+12.2%} "
                  f"{t['kept_return']:<+12.2%} {t['improvement']:<+12.2%}")
    
    print(f"\n\nTRANSITIONS WHERE CONFIRMATION HURTS (filters good signals):")
    print(f"{'From→To':<12} {'Direction':<15} {'N':<8} {'Filter%':<10} {'Filtered':<12} {'Kept':<12} {'Hurt':<12}")
    print("─" * 90)
    
    for t in reversed(transition_improvements[-10:]):
        if t['n_total'] >= 50 and t['improvement'] < 0:
            print(f"{t['from']:>2}→{t['to']:<7} {t['direction']:<15} {t['n_total']:<8} "
                  f"{t['filter_rate']:<10.1%} {t['filtered_return']:<+12.2%} "
                  f"{t['kept_return']:<+12.2%} {t['improvement']:<+12.2%}")
    
    # Summary by direction
    print(f"""

SUMMARY BY TRANSITION DIRECTION:
{'─'*80}""")
    
    direction_stats = defaultdict(lambda: {'filtered': [], 'kept': [], 'n': 0})
    for t in transition_improvements:
        direction_stats[t['direction']]['filtered'].append(t['filtered_return'])
        direction_stats[t['direction']]['kept'].append(t['kept_return'])
        direction_stats[t['direction']]['n'] += t['n_total']
    
    print(f"{'Direction':<20} {'N Trans':<12} {'Filtered Ret':<15} {'Kept Ret':<15} {'Improvement':<15}")
    print("─" * 80)
    
    for direction in ['BEAR_TO_BULL', 'BULL_TO_BEAR', 'BULL_TO_BULL', 'BEAR_TO_BEAR']:
        stats = direction_stats[direction]
        if stats['filtered']:
            filtered = np.mean(stats['filtered'])
            kept = np.mean(stats['kept'])
            improvement = kept - filtered
            print(f"{direction:<20} {stats['n']:<12} {filtered:<+15.2%} {kept:<+15.2%} {improvement:<+15.2%}")
    
    print(f"""

{'='*140}
INTERPRETATION
{'='*140}

1. BEST CONFIG: Look at Sharpe ratio comparison above.
   - If E2.0%_C3h beats Baseline → Combined improvement works!

2. TRANSITIONS THAT BENEFIT:
   - High "Improvement" = confirmation filters bad signals for this transition
   - These are likely noisy transitions where waiting 3h helps

3. TRANSITIONS THAT HURT:
   - Negative "Improvement" = confirmation filters good signals
   - These are likely sharp moves that don't persist but are still profitable

4. STRATEGIC INSIGHT:
   - Could use DIFFERENT confirmation thresholds for different transitions
   - Some transitions need filtering, others should trade immediately
""")


def main():
    parser = argparse.ArgumentParser(description='Combined Optimization Analysis')
    parser.add_argument('--pair', type=str, default=None)
    parser.add_argument('--all-pairs', action='store_true')
    args = parser.parse_args()
    
    print("=" * 140)
    print("COMBINED OPTIMIZATION & TRANSITION ANALYSIS")
    print("=" * 140)
    
    print("""
PART 1: Testing 2.0% entry + confirmation combinations
PART 2: Analyzing which transitions benefit from 3h confirmation
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
    
    print(f"\n{'='*140}")
    print("ANALYSIS COMPLETE")
    print("=" * 140)
    
    return all_results


if __name__ == "__main__":
    results = main()