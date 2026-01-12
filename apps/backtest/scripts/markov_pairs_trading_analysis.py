#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markov Pairs Trading Analysis
==============================
Explores pairs trading using 32-state divergence signals.

Core Idea:
    When Asset A is in RISK_OFF (oversold, high hit rate)
    and Asset B is in RISK_ON (extended, low hit rate)
    → Long A / Short B, expecting convergence

Analysis:
    1. Pair correlations
    2. State divergence frequency
    3. Markov transition probabilities
    4. Convergence statistics
    5. Simple backtest

Usage:
    python markov_pairs_trading.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (from 32-state system)
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005

# Hit rate calculation
MIN_SAMPLES_PER_STATE = 20
MIN_TRAINING_DAYS = 365

# Simplified state thresholds
STRONG_BUY_THRESHOLD = 0.55    # Hit rate >= 55% = oversold, bounce likely
BUY_THRESHOLD = 0.50           # Hit rate >= 50% = mild bullish
SELL_THRESHOLD = 0.45          # Hit rate >= 45% = mild bearish
# Below 45% = STRONG_SELL (extended, pullback likely)

# Pairs trading parameters
MAX_HOLD_DAYS = 10             # Maximum days to hold a spread trade
DIVERGENCE_REQUIRED = 2        # State difference required (STRONG_BUY vs STRONG_SELL = 3)

# Backtest
INITIAL_CAPITAL = 100000.0
POSITION_SIZE = 0.10           # 10% of capital per leg
TRADING_COST = 0.0015          # 0.15% per leg


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PairStats:
    """Statistics for a trading pair."""
    pair_name: str
    asset_a: str
    asset_b: str
    correlation: float
    divergence_count: int
    convergence_rate: float
    avg_convergence_days: float
    avg_spread_return: float


@dataclass 
class SpreadTrade:
    """A spread trade."""
    entry_date: pd.Timestamp
    long_asset: str
    short_asset: str
    entry_state_long: str
    entry_state_short: str
    exit_date: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    spread_return: Optional[float] = None
    holding_days: Optional[int] = None


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
    """Generate 32-state signals."""
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


def calculate_expanding_hit_rate(returns: pd.Series, signals: pd.DataFrame, 
                                  current_idx: int, price_perm: Tuple, 
                                  ma_perm: Tuple) -> float:
    """Calculate hit rate using only data up to current_idx."""
    if current_idx < MIN_TRAINING_DAYS:
        return 0.50
    
    # Get historical data
    hist_returns = returns.iloc[:current_idx]
    hist_signals = signals[signals.index <= returns.index[current_idx-1]]
    
    common_idx = hist_returns.index.intersection(hist_signals.index)
    if len(common_idx) < MIN_SAMPLES_PER_STATE:
        return 0.50
    
    aligned_returns = hist_returns.loc[common_idx]
    aligned_signals = hist_signals.loc[common_idx]
    
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    mask = (
        (aligned_signals['trend_24h'] == price_perm[0]) &
        (aligned_signals['trend_72h'] == price_perm[1]) &
        (aligned_signals['trend_168h'] == price_perm[2]) &
        (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
        (aligned_signals['ma168_above_ma24'] == ma_perm[1])
    )
    
    state_returns = forward_returns[mask].dropna()
    
    if len(state_returns) < MIN_SAMPLES_PER_STATE:
        return 0.50
    
    return (state_returns > 0).sum() / len(state_returns)


def hit_rate_to_simple_state(hit_rate: float) -> str:
    """Convert hit rate to simplified 4-state classification."""
    if hit_rate >= STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    elif hit_rate >= BUY_THRESHOLD:
        return "BUY"
    elif hit_rate >= SELL_THRESHOLD:
        return "SELL"
    else:
        return "STRONG_SELL"


def state_to_numeric(state: str) -> int:
    """Convert state to numeric for divergence calculation."""
    mapping = {
        "STRONG_BUY": 3,
        "BUY": 2,
        "SELL": 1,
        "STRONG_SELL": 0
    }
    return mapping.get(state, 1)


def get_state_divergence(state_a: str, state_b: str) -> int:
    """Calculate divergence between two states."""
    return abs(state_to_numeric(state_a) - state_to_numeric(state_b))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database) -> Dict:
    """Load and prepare data for all pairs."""
    data = {}
    
    print("\n  Loading data...")
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ")
        
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
        
        print(f"{len(df_24h)} days")
    
    return data


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_pair_correlations(data: Dict) -> pd.DataFrame:
    """Calculate return correlations between all pairs."""
    returns_df = pd.DataFrame({
        pair: data[pair]['returns'] for pair in DEPLOY_PAIRS
    }).dropna()
    
    return returns_df.corr()


# =============================================================================
# STATE HISTORY CALCULATION
# =============================================================================

def calculate_state_history(data: Dict) -> pd.DataFrame:
    """Calculate simplified state history for all assets."""
    
    # Get common date range
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))
    
    all_dates = sorted(list(all_dates))
    
    print(f"\n  Calculating state history for {len(all_dates)} common dates...")
    print(f"  (This uses expanding window - may take a few minutes)")
    
    # Pre-calculate all 32 states' hit rates for efficiency
    all_price_perms = list(product([0, 1], repeat=3))
    all_ma_perms = list(product([0, 1], repeat=2))
    
    state_history = {pair: {} for pair in DEPLOY_PAIRS}
    
    # Progress tracking
    last_progress = 0
    
    for i, date in enumerate(all_dates):
        
        # Progress
        progress = int((i / len(all_dates)) * 100)
        if progress >= last_progress + 10:
            print(f"    Progress: {progress}%")
            last_progress = progress
        
        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            
            # Get current 32-state
            if date not in signals.index:
                continue
                
            sig = signals.loc[date]
            price_perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
            ma_perm = (int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24']))
            
            # Get date index for expanding window
            returns_idx = returns.index.get_loc(date) if date in returns.index else None
            if returns_idx is None or returns_idx < MIN_TRAINING_DAYS:
                state_history[pair][date] = "NEUTRAL"
                continue
            
            # Calculate hit rate for this state using expanding window
            hit_rate = calculate_expanding_hit_rate(
                returns, signals, returns_idx, price_perm, ma_perm
            )
            
            # Convert to simple state
            simple_state = hit_rate_to_simple_state(hit_rate)
            state_history[pair][date] = simple_state
    
    print(f"    Progress: 100%")
    
    # Convert to DataFrame
    state_df = pd.DataFrame(state_history)
    state_df.index = pd.to_datetime(state_df.index)
    
    return state_df.dropna()


# =============================================================================
# MARKOV TRANSITION ANALYSIS
# =============================================================================

def analyze_pair_transitions(state_df: pd.DataFrame, asset_a: str, asset_b: str) -> Dict:
    """Analyze Markov transitions for a pair."""
    
    # Create joint state
    joint_states = state_df[[asset_a, asset_b]].copy()
    joint_states['joint'] = joint_states[asset_a] + "_" + joint_states[asset_b]
    joint_states['next_joint'] = joint_states['joint'].shift(-1)
    
    # Count transitions
    transitions = joint_states.groupby(['joint', 'next_joint']).size().unstack(fill_value=0)
    
    # Normalize to probabilities
    transition_probs = transitions.div(transitions.sum(axis=1), axis=0)
    
    # Identify divergence states
    divergence_states = []
    for state in transitions.index:
        a_state, b_state = state.split("_")
        div = get_state_divergence(a_state, b_state)
        if div >= DIVERGENCE_REQUIRED:
            divergence_states.append({
                'joint_state': state,
                'a_state': a_state,
                'b_state': b_state,
                'divergence': div,
                'count': transitions.loc[state].sum(),
            })
    
    return {
        'transitions': transitions,
        'transition_probs': transition_probs,
        'divergence_states': divergence_states,
    }


def calculate_convergence_stats(state_df: pd.DataFrame, returns_data: Dict,
                                 asset_a: str, asset_b: str) -> Dict:
    """Calculate convergence statistics for divergent states."""
    
    dates = state_df.index.tolist()
    
    convergence_events = []
    
    for i, date in enumerate(dates[:-MAX_HOLD_DAYS]):
        a_state = state_df.loc[date, asset_a]
        b_state = state_df.loc[date, asset_b]
        
        divergence = get_state_divergence(a_state, b_state)
        
        if divergence >= DIVERGENCE_REQUIRED:
            # Determine long/short direction
            a_numeric = state_to_numeric(a_state)
            b_numeric = state_to_numeric(b_state)
            
            if a_numeric > b_numeric:
                long_asset, short_asset = asset_a, asset_b
            else:
                long_asset, short_asset = asset_b, asset_a
            
            # Track convergence
            for j in range(1, MAX_HOLD_DAYS + 1):
                future_date = dates[i + j]
                
                future_a = state_df.loc[future_date, asset_a]
                future_b = state_df.loc[future_date, asset_b]
                future_div = get_state_divergence(future_a, future_b)
                
                if future_div < divergence:
                    # Converged
                    # Calculate spread return
                    long_ret = returns_data[long_asset]['returns'].loc[date:future_date].sum()
                    short_ret = returns_data[short_asset]['returns'].loc[date:future_date].sum()
                    spread_ret = long_ret - short_ret
                    
                    convergence_events.append({
                        'entry_date': date,
                        'exit_date': future_date,
                        'days': j,
                        'long_asset': long_asset,
                        'short_asset': short_asset,
                        'spread_return': spread_ret,
                        'converged': True,
                    })
                    break
            else:
                # Did not converge within MAX_HOLD_DAYS
                future_date = dates[i + MAX_HOLD_DAYS]
                long_ret = returns_data[long_asset]['returns'].loc[date:future_date].sum()
                short_ret = returns_data[short_asset]['returns'].loc[date:future_date].sum()
                spread_ret = long_ret - short_ret
                
                convergence_events.append({
                    'entry_date': date,
                    'exit_date': future_date,
                    'days': MAX_HOLD_DAYS,
                    'long_asset': long_asset,
                    'short_asset': short_asset,
                    'spread_return': spread_ret,
                    'converged': False,
                })
    
    return convergence_events


# =============================================================================
# BACKTEST
# =============================================================================

def run_pairs_backtest(state_df: pd.DataFrame, returns_data: Dict) -> Dict:
    """Run backtest on pairs trading strategy."""
    
    dates = state_df.index.tolist()
    n_dates = len(dates)
    
    # Track all trades
    all_trades = []
    open_trades = []
    
    # Portfolio tracking
    equity = INITIAL_CAPITAL
    equity_history = [equity]
    
    print(f"\n  Running pairs backtest on {n_dates} days...")
    
    for i, date in enumerate(dates[:-1]):
        
        # Check for exit on open trades
        trades_to_close = []
        for trade in open_trades:
            holding_days = (date - trade.entry_date).days
            
            # Get current states
            curr_long_state = state_df.loc[date, trade.long_asset]
            curr_short_state = state_df.loc[date, trade.short_asset]
            curr_div = get_state_divergence(curr_long_state, curr_short_state)
            
            # Exit conditions
            should_exit = False
            exit_reason = None
            
            if curr_div < DIVERGENCE_REQUIRED:
                should_exit = True
                exit_reason = "CONVERGED"
            elif holding_days >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = "MAX_HOLD"
            
            if should_exit:
                # Calculate spread return
                long_ret = returns_data[trade.long_asset]['returns'].loc[trade.entry_date:date].iloc[1:].sum()
                short_ret = returns_data[trade.short_asset]['returns'].loc[trade.entry_date:date].iloc[1:].sum()
                spread_ret = long_ret - short_ret
                
                # Account for costs
                spread_ret -= 4 * TRADING_COST  # Entry and exit, both legs
                
                trade.exit_date = date
                trade.exit_reason = exit_reason
                trade.spread_return = spread_ret
                trade.holding_days = holding_days
                
                # Update equity
                trade_pnl = equity * POSITION_SIZE * 2 * spread_ret
                equity += trade_pnl
                
                trades_to_close.append(trade)
        
        # Remove closed trades
        for trade in trades_to_close:
            open_trades.remove(trade)
            all_trades.append(trade)
        
        # Check for new entries (only if no open trade on same pair)
        active_pairs = set()
        for trade in open_trades:
            active_pairs.add((trade.long_asset, trade.short_asset))
            active_pairs.add((trade.short_asset, trade.long_asset))
        
        for asset_a, asset_b in combinations(DEPLOY_PAIRS, 2):
            if (asset_a, asset_b) in active_pairs:
                continue
            
            a_state = state_df.loc[date, asset_a]
            b_state = state_df.loc[date, asset_b]
            
            divergence = get_state_divergence(a_state, b_state)
            
            if divergence >= DIVERGENCE_REQUIRED:
                # Determine direction
                a_numeric = state_to_numeric(a_state)
                b_numeric = state_to_numeric(b_state)
                
                if a_numeric > b_numeric:
                    long_asset, short_asset = asset_a, asset_b
                    long_state, short_state = a_state, b_state
                else:
                    long_asset, short_asset = asset_b, asset_a
                    long_state, short_state = b_state, a_state
                
                trade = SpreadTrade(
                    entry_date=date,
                    long_asset=long_asset,
                    short_asset=short_asset,
                    entry_state_long=long_state,
                    entry_state_short=short_state,
                )
                
                open_trades.append(trade)
        
        equity_history.append(equity)
    
    # Close any remaining trades at end
    final_date = dates[-1]
    for trade in open_trades:
        long_ret = returns_data[trade.long_asset]['returns'].loc[trade.entry_date:final_date].iloc[1:].sum()
        short_ret = returns_data[trade.short_asset]['returns'].loc[trade.entry_date:final_date].iloc[1:].sum()
        spread_ret = long_ret - short_ret - 4 * TRADING_COST
        
        trade.exit_date = final_date
        trade.exit_reason = "END"
        trade.spread_return = spread_ret
        trade.holding_days = (final_date - trade.entry_date).days
        all_trades.append(trade)
    
    equity_history.append(equity)
    
    return {
        'trades': all_trades,
        'equity_history': equity_history,
        'final_equity': equity,
    }


# =============================================================================
# ANALYSIS DISPLAY
# =============================================================================

def display_correlation_matrix(corr_matrix: pd.DataFrame):
    """Display correlation matrix."""
    print("\n" + "=" * 80)
    print("PAIR CORRELATIONS")
    print("=" * 80)
    
    # Shorten names
    short_names = {p: p.replace('USD', '') for p in DEPLOY_PAIRS}
    
    print(f"\n  {'':>8}", end="")
    for p in DEPLOY_PAIRS:
        print(f"{short_names[p]:>8}", end="")
    print()
    
    for p1 in DEPLOY_PAIRS:
        print(f"  {short_names[p1]:>8}", end="")
        for p2 in DEPLOY_PAIRS:
            corr = corr_matrix.loc[p1, p2]
            print(f"{corr:>8.2f}", end="")
        print()
    
    # Find most/least correlated pairs
    pairs_corr = []
    for a, b in combinations(DEPLOY_PAIRS, 2):
        pairs_corr.append((a, b, corr_matrix.loc[a, b]))
    
    pairs_corr.sort(key=lambda x: x[2], reverse=True)
    
    print("\n  Highest correlation pairs:")
    for a, b, c in pairs_corr[:3]:
        print(f"    {short_names[a]}-{short_names[b]}: {c:.3f}")
    
    print("\n  Lowest correlation pairs:")
    for a, b, c in pairs_corr[-3:]:
        print(f"    {short_names[a]}-{short_names[b]}: {c:.3f}")


def display_state_distribution(state_df: pd.DataFrame):
    """Display state distribution for each asset."""
    print("\n" + "=" * 80)
    print("STATE DISTRIBUTION BY ASSET")
    print("=" * 80)
    
    short_names = {p: p.replace('USD', '') for p in DEPLOY_PAIRS}
    
    print(f"\n  {'Asset':<10} {'STRONG_BUY':>12} {'BUY':>10} {'SELL':>10} {'STRONG_SELL':>12}")
    print("  " + "-" * 55)
    
    for pair in DEPLOY_PAIRS:
        counts = state_df[pair].value_counts()
        total = len(state_df)
        
        sb = counts.get('STRONG_BUY', 0) / total * 100
        b = counts.get('BUY', 0) / total * 100
        s = counts.get('SELL', 0) / total * 100
        ss = counts.get('STRONG_SELL', 0) / total * 100
        
        print(f"  {short_names[pair]:<10} {sb:>11.1f}% {b:>9.1f}% {s:>9.1f}% {ss:>11.1f}%")


def display_divergence_analysis(state_df: pd.DataFrame, data: Dict):
    """Display divergence analysis for all pairs."""
    print("\n" + "=" * 80)
    print("DIVERGENCE ANALYSIS BY PAIR")
    print("=" * 80)
    
    short_names = {p: p.replace('USD', '') for p in DEPLOY_PAIRS}
    
    pair_stats = []
    
    print(f"\n  {'Pair':<12} {'Divergences':>12} {'Conv Rate':>10} {'Avg Days':>10} {'Avg Return':>12}")
    print("  " + "-" * 60)
    
    for asset_a, asset_b in combinations(DEPLOY_PAIRS, 2):
        events = calculate_convergence_stats(state_df, data, asset_a, asset_b)
        
        if len(events) == 0:
            continue
        
        conv_events = [e for e in events if e['converged']]
        conv_rate = len(conv_events) / len(events) if events else 0
        avg_days = np.mean([e['days'] for e in events]) if events else 0
        avg_ret = np.mean([e['spread_return'] for e in events]) if events else 0
        
        pair_name = f"{short_names[asset_a]}-{short_names[asset_b]}"
        
        print(f"  {pair_name:<12} {len(events):>12} {conv_rate*100:>9.1f}% {avg_days:>10.1f} {avg_ret*100:>+11.2f}%")
        
        pair_stats.append({
            'pair': pair_name,
            'asset_a': asset_a,
            'asset_b': asset_b,
            'divergences': len(events),
            'conv_rate': conv_rate,
            'avg_days': avg_days,
            'avg_return': avg_ret,
        })
    
    # Best pairs
    if pair_stats:
        best_by_return = max(pair_stats, key=lambda x: x['avg_return'])
        best_by_conv = max(pair_stats, key=lambda x: x['conv_rate'])
        most_active = max(pair_stats, key=lambda x: x['divergences'])
        
        print(f"""
    KEY FINDINGS:
    
    Most Active Pair:      {most_active['pair']} ({most_active['divergences']} divergences)
    Highest Conv Rate:     {best_by_conv['pair']} ({best_by_conv['conv_rate']*100:.1f}%)
    Best Avg Return:       {best_by_return['pair']} ({best_by_return['avg_return']*100:+.2f}%)
        """)
    
    return pair_stats


def display_backtest_results(backtest: Dict):
    """Display backtest results."""
    print("\n" + "=" * 80)
    print("PAIRS TRADING BACKTEST RESULTS")
    print("=" * 80)
    
    trades = backtest['trades']
    equity_history = backtest['equity_history']
    
    if not trades:
        print("\n  No trades executed.")
        return
    
    # Trade statistics
    n_trades = len(trades)
    winners = [t for t in trades if t.spread_return and t.spread_return > 0]
    losers = [t for t in trades if t.spread_return and t.spread_return <= 0]
    
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    
    avg_return = np.mean([t.spread_return for t in trades if t.spread_return])
    avg_winner = np.mean([t.spread_return for t in winners]) if winners else 0
    avg_loser = np.mean([t.spread_return for t in losers]) if losers else 0
    
    avg_hold = np.mean([t.holding_days for t in trades if t.holding_days])
    
    # Exit reasons
    converged = len([t for t in trades if t.exit_reason == "CONVERGED"])
    max_hold = len([t for t in trades if t.exit_reason == "MAX_HOLD"])
    
    # Portfolio metrics
    total_return = (backtest['final_equity'] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    equity_series = pd.Series(equity_history)
    daily_returns = equity_series.pct_change().dropna()
    
    years = len(equity_history) / 365
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
    
    rolling_max = equity_series.expanding().max()
    max_dd = ((equity_series - rolling_max) / rolling_max).min()
    
    print(f"""
    TRADE STATISTICS
    ────────────────────────────────────────
    Total Trades:          {n_trades}
    Win Rate:              {win_rate*100:.1f}%
    
    Avg Spread Return:     {avg_return*100:+.2f}%
    Avg Winner:            {avg_winner*100:+.2f}%
    Avg Loser:             {avg_loser*100:+.2f}%
    
    Avg Holding Period:    {avg_hold:.1f} days
    
    Exit Reasons:
      Converged:           {converged} ({converged/n_trades*100:.0f}%)
      Max Hold:            {max_hold} ({max_hold/n_trades*100:.0f}%)
    
    PORTFOLIO METRICS
    ────────────────────────────────────────
    Initial Capital:       ${INITIAL_CAPITAL:,.0f}
    Final Equity:          ${backtest['final_equity']:,.0f}
    
    Total Return:          {total_return*100:+.1f}%
    Annual Return:         {annual_return*100:+.1f}%
    Sharpe Ratio:          {sharpe:.2f}
    Max Drawdown:          {max_dd*100:.1f}%
    """)
    
    # Breakdown by pair
    print("    PERFORMANCE BY PAIR")
    print("    " + "-" * 40)
    
    pair_perf = {}
    for t in trades:
        pair_key = f"{t.long_asset.replace('USD','')}-{t.short_asset.replace('USD','')}"
        if pair_key not in pair_perf:
            pair_perf[pair_key] = []
        pair_perf[pair_key].append(t.spread_return)
    
    print(f"    {'Pair':<15} {'Trades':>8} {'Win Rate':>10} {'Avg Ret':>10}")
    print("    " + "-" * 45)
    
    for pair, returns in sorted(pair_perf.items()):
        n = len(returns)
        wr = sum(1 for r in returns if r > 0) / n if n > 0 else 0
        ar = np.mean(returns) if returns else 0
        print(f"    {pair:<15} {n:>8} {wr*100:>9.1f}% {ar*100:>+9.2f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("MARKOV PAIRS TRADING ANALYSIS")
    print("=" * 100)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║  PAIRS TRADING WITH STATE DIVERGENCE                                          ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  CONCEPT:                                                                     ║
    ║    When Asset A is oversold (STRONG_BUY) and Asset B is extended (STRONG_SELL)║
    ║    → Long A / Short B, expecting mean reversion                               ║
    ║                                                                               ║
    ║  SIMPLIFIED STATES:                                                           ║
    ║    STRONG_BUY:  Hit rate ≥ 55% (oversold, bounce likely)                      ║
    ║    BUY:         Hit rate ≥ 50%                                                ║
    ║    SELL:        Hit rate ≥ 45%                                                ║
    ║    STRONG_SELL: Hit rate < 45% (extended, pullback likely)                    ║
    ║                                                                               ║
    ║  TRADE LOGIC:                                                                 ║
    ║    Entry: State divergence ≥ {DIVERGENCE_REQUIRED} (e.g., STRONG_BUY vs STRONG_SELL)            ║
    ║    Exit:  Convergence OR {MAX_HOLD_DAYS} days max hold                                   ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    db = Database()
    
    # Load data
    data = load_all_data(db)
    
    # Correlation analysis
    corr_matrix = calculate_pair_correlations(data)
    display_correlation_matrix(corr_matrix)
    
    # Calculate state history (this takes time due to expanding window)
    state_df = calculate_state_history(data)
    
    # State distribution
    display_state_distribution(state_df)
    
    # Divergence analysis
    pair_stats = display_divergence_analysis(state_df, data)
    
    # Run backtest
    backtest = run_pairs_backtest(state_df, data)
    display_backtest_results(backtest)
    
    # Final summary
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    return {
        'data': data,
        'corr_matrix': corr_matrix,
        'state_df': state_df,
        'pair_stats': pair_stats,
        'backtest': backtest,
    }


if __name__ == "__main__":
    results = main()

