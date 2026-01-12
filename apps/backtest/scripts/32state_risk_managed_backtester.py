#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairs Trading Performance by Market Regime
============================================
Analyzes whether the pairs trading strategy works in both
bull and bear markets.

Usage:
    python pairs_regime_analysis.py
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
STRONG_BUY_THRESHOLD = 0.55
BUY_THRESHOLD = 0.50
SELL_THRESHOLD = 0.45

# Pairs trading parameters
MAX_HOLD_DAYS = 10
DIVERGENCE_REQUIRED = 2

# Backtest
INITIAL_CAPITAL = 100000.0
POSITION_SIZE = 0.10
TRADING_COST = 0.0015

# Market regimes (approximate periods)
MARKET_REGIMES = [
    ('2018 Bear', '2018-01-01', '2018-12-31', 'BEAR'),
    ('2019 Recovery', '2019-01-01', '2019-12-31', 'BULL'),
    ('2020 COVID Crash + Recovery', '2020-01-01', '2020-12-31', 'MIXED'),
    ('2021 Bull Run', '2021-01-01', '2021-12-31', 'BULL'),
    ('2022 Bear', '2022-01-01', '2022-12-31', 'BEAR'),
    ('2023 Recovery', '2023-01-01', '2023-12-31', 'BULL'),
    ('2024 Bull', '2024-01-01', '2024-12-31', 'BULL'),
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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
# HELPER FUNCTIONS (from main script)
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def label_trend_binary(df: pd.DataFrame, ma_period: int,
                       entry_buffer: float, exit_buffer: float) -> pd.Series:
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
    if current_idx < MIN_TRAINING_DAYS:
        return 0.50
    
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
    if hit_rate >= STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    elif hit_rate >= BUY_THRESHOLD:
        return "BUY"
    elif hit_rate >= SELL_THRESHOLD:
        return "SELL"
    else:
        return "STRONG_SELL"


def state_to_numeric(state: str) -> int:
    mapping = {"STRONG_BUY": 3, "BUY": 2, "SELL": 1, "STRONG_SELL": 0}
    return mapping.get(state, 1)


def get_state_divergence(state_a: str, state_b: str) -> int:
    return abs(state_to_numeric(state_a) - state_to_numeric(state_b))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database) -> Dict:
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
# STATE HISTORY CALCULATION
# =============================================================================

def calculate_state_history(data: Dict) -> pd.DataFrame:
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))
    
    all_dates = sorted(list(all_dates))
    
    print(f"\n  Calculating state history for {len(all_dates)} common dates...")
    
    state_history = {pair: {} for pair in DEPLOY_PAIRS}
    last_progress = 0
    
    for i, date in enumerate(all_dates):
        progress = int((i / len(all_dates)) * 100)
        if progress >= last_progress + 10:
            print(f"    Progress: {progress}%")
            last_progress = progress
        
        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            
            if date not in signals.index:
                continue
                
            sig = signals.loc[date]
            price_perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
            ma_perm = (int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24']))
            
            returns_idx = returns.index.get_loc(date) if date in returns.index else None
            if returns_idx is None or returns_idx < MIN_TRAINING_DAYS:
                state_history[pair][date] = "NEUTRAL"
                continue
            
            hit_rate = calculate_expanding_hit_rate(
                returns, signals, returns_idx, price_perm, ma_perm
            )
            
            simple_state = hit_rate_to_simple_state(hit_rate)
            state_history[pair][date] = simple_state
    
    print(f"    Progress: 100%")
    
    state_df = pd.DataFrame(state_history)
    state_df.index = pd.to_datetime(state_df.index)
    
    return state_df.dropna()


# =============================================================================
# BACKTEST WITH TRADE TRACKING
# =============================================================================

def run_pairs_backtest_with_tracking(state_df: pd.DataFrame, returns_data: Dict) -> Dict:
    """Run backtest and track equity curve by date."""
    
    dates = state_df.index.tolist()
    
    all_trades = []
    open_trades = []
    
    equity = INITIAL_CAPITAL
    
    # Track equity by date
    equity_by_date = {}
    daily_pnl = {}
    
    print(f"\n  Running pairs backtest...")
    
    for i, date in enumerate(dates[:-1]):
        
        day_pnl = 0.0
        
        # Check for exit on open trades
        trades_to_close = []
        for trade in open_trades:
            holding_days = (date - trade.entry_date).days
            
            curr_long_state = state_df.loc[date, trade.long_asset]
            curr_short_state = state_df.loc[date, trade.short_asset]
            curr_div = get_state_divergence(curr_long_state, curr_short_state)
            
            should_exit = False
            exit_reason = None
            
            if curr_div < DIVERGENCE_REQUIRED:
                should_exit = True
                exit_reason = "CONVERGED"
            elif holding_days >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = "MAX_HOLD"
            
            if should_exit:
                long_ret = returns_data[trade.long_asset]['returns'].loc[trade.entry_date:date].iloc[1:].sum()
                short_ret = returns_data[trade.short_asset]['returns'].loc[trade.entry_date:date].iloc[1:].sum()
                spread_ret = long_ret - short_ret
                spread_ret -= 4 * TRADING_COST
                
                trade.exit_date = date
                trade.exit_reason = exit_reason
                trade.spread_return = spread_ret
                trade.holding_days = holding_days
                
                trade_pnl = equity * POSITION_SIZE * 2 * spread_ret
                day_pnl += trade_pnl
                
                trades_to_close.append(trade)
        
        for trade in trades_to_close:
            open_trades.remove(trade)
            all_trades.append(trade)
        
        # Check for new entries
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
        
        equity += day_pnl
        equity_by_date[date] = equity
        daily_pnl[date] = day_pnl
    
    # Close remaining trades
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
    
    return {
        'trades': all_trades,
        'equity_by_date': equity_by_date,
        'daily_pnl': daily_pnl,
        'final_equity': equity,
    }


# =============================================================================
# MARKET RETURN CALCULATION
# =============================================================================

def calculate_market_returns(data: Dict) -> pd.Series:
    """Calculate equal-weighted market return (proxy for crypto market)."""
    
    returns_df = pd.DataFrame({
        pair: data[pair]['returns'] for pair in DEPLOY_PAIRS
    }).dropna()
    
    # Equal weighted average
    market_return = returns_df.mean(axis=1)
    
    return market_return


# =============================================================================
# REGIME ANALYSIS
# =============================================================================

def analyze_by_regime(backtest: Dict, market_returns: pd.Series, data: Dict):
    """Analyze strategy performance by market regime."""
    
    print("\n" + "=" * 100)
    print("PERFORMANCE BY MARKET REGIME")
    print("=" * 100)
    
    equity_series = pd.Series(backtest['equity_by_date'])
    trades = backtest['trades']
    
    # Make index timezone-naive for comparison
    if equity_series.index.tz is not None:
        equity_series.index = equity_series.index.tz_localize(None)
    
    results = []
    
    print(f"\n  {'Period':<30} {'Market':>10} {'Strategy':>12} {'Trades':>8} {'Win Rate':>10} {'Avg Ret':>10}")
    print("  " + "-" * 85)
    
    for regime_name, start_date, end_date, regime_type in MARKET_REGIMES:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Filter equity to period
        period_equity = equity_series[(equity_series.index >= start) & (equity_series.index <= end)]
        
        if len(period_equity) < 10:
            continue
        
        # Strategy return for period
        start_equity = period_equity.iloc[0]
        end_equity = period_equity.iloc[-1]
        strategy_return = (end_equity - start_equity) / start_equity
        
        # Market return for period - handle timezone
        market_idx = market_returns.index
        if market_idx.tz is not None:
            market_idx_naive = market_idx.tz_localize(None)
            period_market = market_returns[(market_idx_naive >= start) & (market_idx_naive <= end)]
        else:
            period_market = market_returns[(market_returns.index >= start) & (market_returns.index <= end)]
        
        market_return = (1 + period_market).prod() - 1
        
        # Get market prices for reference (use ETH as proxy)
        eth_prices = data['ETHUSD']['prices']['close']
        eth_idx = eth_prices.index
        if eth_idx.tz is not None:
            eth_idx_naive = eth_idx.tz_localize(None)
            eth_period = eth_prices[(eth_idx_naive >= start) & (eth_idx_naive <= end)]
        else:
            eth_period = eth_prices[(eth_prices.index >= start) & (eth_prices.index <= end)]
            
        if len(eth_period) > 0:
            eth_return = (eth_period.iloc[-1] - eth_period.iloc[0]) / eth_period.iloc[0]
        else:
            eth_return = 0
        
        # Trades in period - handle timezone on trade dates
        period_trades = []
        for t in trades:
            entry = t.entry_date
            if hasattr(entry, 'tz') and entry.tz is not None:
                entry = entry.tz_localize(None)
            if entry >= start and entry <= end:
                period_trades.append(t)
        
        n_trades = len(period_trades)
        
        if n_trades > 0:
            winners = [t for t in period_trades if t.spread_return and t.spread_return > 0]
            win_rate = len(winners) / n_trades
            avg_ret = np.mean([t.spread_return for t in period_trades if t.spread_return])
        else:
            win_rate = 0
            avg_ret = 0
        
        # Display
        market_str = f"{eth_return*100:+.0f}%"
        strat_str = f"{strategy_return*100:+.1f}%"
        wr_str = f"{win_rate*100:.0f}%"
        ar_str = f"{avg_ret*100:+.2f}%"
        
        print(f"  {regime_name:<30} {market_str:>10} {strat_str:>12} {n_trades:>8} {wr_str:>10} {ar_str:>10}")
        
        results.append({
            'period': regime_name,
            'regime_type': regime_type,
            'market_return': eth_return,
            'strategy_return': strategy_return,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_trade_return': avg_ret,
        })
    
    # Summary by regime type
    print("\n" + "-" * 60)
    print("  SUMMARY BY REGIME TYPE")
    print("-" * 60)
    
    for regime_type in ['BULL', 'BEAR', 'MIXED']:
        type_results = [r for r in results if r['regime_type'] == regime_type]
        
        if not type_results:
            continue
        
        avg_market = np.mean([r['market_return'] for r in type_results])
        avg_strat = np.mean([r['strategy_return'] for r in type_results])
        total_trades = sum([r['n_trades'] for r in type_results])
        avg_wr = np.mean([r['win_rate'] for r in type_results])
        
        print(f"\n  {regime_type} MARKETS:")
        print(f"    Avg Market Return:     {avg_market*100:+.1f}%")
        print(f"    Avg Strategy Return:   {avg_strat*100:+.1f}%")
        print(f"    Total Trades:          {total_trades}")
        print(f"    Avg Win Rate:          {avg_wr*100:.1f}%")
        
        if regime_type == 'BEAR':
            if avg_strat > 0:
                print(f"    ✓ PROFITABLE IN BEAR MARKET")
            else:
                print(f"    ✗ Lost money in bear market")
        elif regime_type == 'BULL':
            if avg_strat > 0:
                print(f"    ✓ PROFITABLE IN BULL MARKET")
    
    return results


def display_monthly_returns(backtest: Dict):
    """Display monthly returns heatmap style."""
    
    print("\n" + "=" * 100)
    print("MONTHLY RETURNS")
    print("=" * 100)
    
    equity_series = pd.Series(backtest['equity_by_date'])
    
    # Make index timezone-naive
    if equity_series.index.tz is not None:
        equity_series.index = equity_series.index.tz_localize(None)
    
    # Calculate monthly returns
    monthly_equity = equity_series.resample('ME').last()
    monthly_returns = monthly_equity.pct_change().dropna()
    
    # Group by year and month
    monthly_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print(f"\n  {'Year':>6}", end="")
    for m in month_names:
        print(f"{m:>8}", end="")
    print(f"{'TOTAL':>10}")
    
    print("  " + "-" * 110)
    
    for year in sorted(pivot.index):
        print(f"  {year:>6}", end="")
        year_total = 0
        
        for month in range(1, 13):
            if month in pivot.columns and not pd.isna(pivot.loc[year, month]):
                ret = pivot.loc[year, month]
                year_total += ret
                
                # Color coding (text-based)
                if ret > 0.05:
                    print(f"{ret*100:>+7.1f}%", end="")
                elif ret > 0:
                    print(f"{ret*100:>+7.1f}%", end="")
                elif ret > -0.05:
                    print(f"{ret*100:>+7.1f}%", end="")
                else:
                    print(f"{ret*100:>+7.1f}%", end="")
            else:
                print(f"{'--':>8}", end="")
        
        print(f"{year_total*100:>+9.1f}%")
    
    # Statistics
    print("\n  Monthly Statistics:")
    print(f"    Positive months: {(monthly_returns > 0).sum()} / {len(monthly_returns)} ({(monthly_returns > 0).mean()*100:.0f}%)")
    print(f"    Average month:   {monthly_returns.mean()*100:+.2f}%")
    print(f"    Best month:      {monthly_returns.max()*100:+.1f}%")
    print(f"    Worst month:     {monthly_returns.min()*100:+.1f}%")


def display_correlation_analysis(backtest: Dict, market_returns: pd.Series):
    """Analyze correlation between strategy and market."""
    
    print("\n" + "=" * 100)
    print("MARKET CORRELATION ANALYSIS")
    print("=" * 100)
    
    equity_series = pd.Series(backtest['equity_by_date'])
    
    # Make index timezone-naive
    if equity_series.index.tz is not None:
        equity_series.index = equity_series.index.tz_localize(None)
    
    strategy_returns = equity_series.pct_change().dropna()
    
    # Make market returns timezone-naive
    market_returns_clean = market_returns.copy()
    if market_returns_clean.index.tz is not None:
        market_returns_clean.index = market_returns_clean.index.tz_localize(None)
    
    # Align dates
    common_dates = strategy_returns.index.intersection(market_returns_clean.index)
    strat_aligned = strategy_returns.loc[common_dates]
    market_aligned = market_returns_clean.loc[common_dates]
    
    # Calculate correlation
    correlation = strat_aligned.corr(market_aligned)
    
    # Calculate beta
    cov = np.cov(strat_aligned, market_aligned)[0, 1]
    var = np.var(market_aligned)
    beta = cov / var if var > 0 else 0
    
    # Performance in up/down days
    up_days = market_aligned > 0
    down_days = market_aligned < 0
    
    strat_on_up_days = strat_aligned[up_days].mean()
    strat_on_down_days = strat_aligned[down_days].mean()
    
    print(f"""
    CORRELATION METRICS
    ────────────────────────────────────────
    Correlation with Market:   {correlation:.3f}
    Beta:                      {beta:.3f}
    
    CONDITIONAL RETURNS
    ────────────────────────────────────────
    Strategy on UP market days:    {strat_on_up_days*100:+.3f}%
    Strategy on DOWN market days:  {strat_on_down_days*100:+.3f}%
    
    INTERPRETATION
    ────────────────────────────────────────""")
    
    if abs(correlation) < 0.3:
        print(f"    ✓ LOW correlation ({correlation:.2f}) — Strategy is market-neutral")
    elif abs(correlation) < 0.5:
        print(f"    ~ MODERATE correlation ({correlation:.2f}) — Some market exposure")
    else:
        print(f"    ✗ HIGH correlation ({correlation:.2f}) — Not truly market-neutral")
    
    if beta < 0.3:
        print(f"    ✓ LOW beta ({beta:.2f}) — Low sensitivity to market moves")
    else:
        print(f"    ~ Beta of {beta:.2f} — Some market sensitivity")
    
    if strat_on_down_days > 0:
        print(f"    ✓ POSITIVE returns on down days — True hedge potential")
    else:
        print(f"    ~ Negative on down days — Not a pure hedge")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("PAIRS TRADING: PERFORMANCE BY MARKET REGIME")
    print("=" * 100)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║  QUESTION: Does pairs trading work in BOTH bull AND bear markets?             ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  If truly market-neutral, strategy should:                                    ║
    ║    ✓ Make money when market rises                                             ║
    ║    ✓ Make money when market falls                                             ║
    ║    ✓ Have low correlation to market returns                                   ║
    ║    ✓ Have low beta                                                            ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    db = Database()
    
    # Load data
    data = load_all_data(db)
    
    # Calculate market returns
    market_returns = calculate_market_returns(data)
    
    # Calculate state history
    state_df = calculate_state_history(data)
    
    # Run backtest with tracking
    backtest = run_pairs_backtest_with_tracking(state_df, data)
    
    # Analyze by regime
    regime_results = analyze_by_regime(backtest, market_returns, data)
    
    # Monthly returns
    display_monthly_returns(backtest)
    
    # Correlation analysis
    display_correlation_analysis(backtest, market_returns)
    
    # Final verdict
    print("\n" + "=" * 100)
    print("VERDICT: IS THIS TRULY MARKET-NEUTRAL?")
    print("=" * 100)
    
    bull_results = [r for r in regime_results if r['regime_type'] == 'BULL']
    bear_results = [r for r in regime_results if r['regime_type'] == 'BEAR']
    
    bull_profitable = all(r['strategy_return'] > 0 for r in bull_results) if bull_results else False
    bear_profitable = all(r['strategy_return'] > 0 for r in bear_results) if bear_results else False
    
    print(f"""
    BULL MARKETS: {"✓ PROFITABLE" if bull_profitable else "✗ NOT ALWAYS PROFITABLE"}
    BEAR MARKETS: {"✓ PROFITABLE" if bear_profitable else "✗ NOT ALWAYS PROFITABLE"}
    
    CONCLUSION:
    """)
    
    if bull_profitable and bear_profitable:
        print("    ✓ Strategy works in BOTH bull and bear markets")
        print("    ✓ This confirms market-neutral characteristics")
        print("    ✓ Returns driven by spread convergence, not market direction")
    elif bear_profitable:
        print("    ✓ Strategy profitable in bear markets (hedge value)")
        print("    ~ May underperform in strong bull markets")
    else:
        print("    ~ Strategy has some market dependency")
        print("    ~ Consider additional hedging or position limits")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    return {
        'backtest': backtest,
        'regime_results': regime_results,
        'market_returns': market_returns,
    }


if __name__ == "__main__":
    results = main()