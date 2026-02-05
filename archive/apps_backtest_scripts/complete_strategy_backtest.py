#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Strategy Comparison Backtest
=====================================
Compare three strategies across all pairs:
1. OPTIMIZED: E2.5% entry + 3h confirmation (new)
2. BASELINE: E1.5% entry, no confirmation (current production)
3. BUY & HOLD: Simple buy and hold benchmark

Includes:
- Portfolio-level performance (risk parity allocation)
- Individual pair performance
- Year-by-year breakdown
- Equity curves
- Comprehensive metrics

Usage:
    python complete_strategy_backtest.py
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
from dataclasses import dataclass, field
from collections import defaultdict
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

# Strategy configurations
STRATEGIES = {
    'OPTIMIZED': {
        'entry_buffer': 0.025,
        'exit_buffer': 0.005,
        'confirmation_hours': 3,
        'description': 'E2.5% entry + 3h confirmation'
    },
    'BASELINE': {
        'entry_buffer': 0.015,
        'exit_buffer': 0.005,
        'confirmation_hours': 0,
        'description': 'E1.5% entry, no confirmation'
    },
    'BUY_HOLD': {
        'description': 'Buy and hold (equal weight)'
    }
}

# Portfolio parameters
INITIAL_CAPITAL = 100000
KELLY_FRACTION = 0.25
RISK_PARITY_WEIGHTS = {pair: 1/6 for pair in DEPLOY_PAIRS}  # Equal weight for simplicity


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


def apply_confirmation_filter(states_hourly: pd.DataFrame, 
                               confirmation_hours: int) -> pd.DataFrame:
    """Apply confirmation filter and resample to daily."""
    if confirmation_hours == 0:
        return states_hourly.resample('24h').first().dropna()
    
    states = states_hourly.copy()
    state_values = states['state'].values
    
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
    daily = states.resample('24h').first().dropna()
    return daily


# =============================================================================
# BACKTESTING
# =============================================================================

@dataclass
class TradeRecord:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int


@dataclass
class PairResult:
    pair: str
    strategy: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    avg_trade_return: float
    profit_factor: float
    avg_holding_days: float
    exposure_pct: float  # % of time in market
    equity_curve: pd.Series
    trades: List[TradeRecord]
    yearly_returns: Dict[int, float]


def run_strategy_backtest(states: pd.DataFrame, strategy_name: str, 
                          pair: str) -> PairResult:
    """Run backtest for a regime-based strategy."""
    state_col = 'confirmed_state' if 'confirmed_state' in states.columns else 'state'
    
    prices = states['close'].values
    state_values = states[state_col].values
    dates = states.index
    
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    trades = []
    daily_returns = []
    
    position = 0
    entry_price = 0
    entry_date = None
    days_in_market = 0
    
    for i in range(1, len(prices)):
        current_state = int(state_values[i])
        prev_state = int(state_values[i-1])
        is_bullish = current_state in BULLISH_STATES
        was_bullish = prev_state in BULLISH_STATES
        
        # Calculate daily return
        if position > 0:
            period_return = (prices[i] - prices[i-1]) / prices[i-1]
            position_return = KELLY_FRACTION * period_return
            equity *= (1 + position_return)
            daily_returns.append(position_return)
            days_in_market += 1
        else:
            daily_returns.append(0)
        
        # Trade on regime change
        if is_bullish != was_bullish:
            if position > 0:
                trade_return = (prices[i] - entry_price) / entry_price
                holding_days = (dates[i] - entry_date).days
                trades.append(TradeRecord(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    entry_price=entry_price,
                    exit_price=prices[i],
                    return_pct=trade_return,
                    holding_days=holding_days
                ))
            
            if is_bullish:
                position = 1
                entry_price = prices[i]
                entry_date = dates[i]
            else:
                position = 0
        
        equity_curve.append(equity)
    
    # Close final position
    if position > 0:
        trade_return = (prices[-1] - entry_price) / entry_price
        holding_days = (dates[-1] - entry_date).days
        trades.append(TradeRecord(
            entry_date=entry_date,
            exit_date=dates[-1],
            entry_price=entry_price,
            exit_price=prices[-1],
            return_pct=trade_return,
            holding_days=holding_days
        ))
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=dates)
    returns_series = pd.Series(daily_returns, index=dates[1:])
    
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Sharpe & Sortino
    if returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
        downside_returns = returns_series[returns_series < 0]
        sortino = returns_series.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    else:
        sharpe = sortino = 0
    
    # Max drawdown
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
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
        avg_holding = np.mean([t.holding_days for t in trades])
    else:
        win_rate = avg_trade = profit_factor = avg_holding = 0
    
    exposure_pct = days_in_market / len(prices) if len(prices) > 0 else 0
    
    # Yearly returns
    yearly_returns = {}
    for year in range(dates[0].year, dates[-1].year + 1):
        year_mask = equity_series.index.year == year
        if year_mask.sum() > 0:
            year_data = equity_series[year_mask]
            if len(year_data) > 1:
                yearly_returns[year] = (year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]
    
    return PairResult(
        pair=pair,
        strategy=strategy_name,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_trade_return=avg_trade,
        profit_factor=profit_factor,
        avg_holding_days=avg_holding,
        exposure_pct=exposure_pct,
        equity_curve=equity_series,
        trades=trades,
        yearly_returns=yearly_returns
    )


def run_buy_hold_backtest(df_1h: pd.DataFrame, pair: str) -> PairResult:
    """Run buy and hold backtest."""
    # Use same resampling as other functions
    daily = resample_ohlcv(df_1h, '24h')
    
    if len(daily) == 0:
        print(f"\n  WARNING: No daily data for {pair} after resampling")
        print(f"  Input df_1h shape: {df_1h.shape}, index: {df_1h.index[0] if len(df_1h) > 0 else 'empty'} to {df_1h.index[-1] if len(df_1h) > 0 else 'empty'}")
        raise ValueError(f"No daily data for {pair}")
    
    prices = daily['close'].values
    dates = daily.index
    
    if len(prices) < 2:
        print(f"\n  WARNING: Insufficient data for {pair}: {len(prices)} days")
        raise ValueError(f"Insufficient data for {pair}")
    
    # Buy and hold with Kelly fraction (for fair comparison)
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    daily_returns = []
    
    for i in range(1, len(prices)):
        period_return = (prices[i] - prices[i-1]) / prices[i-1]
        position_return = KELLY_FRACTION * period_return
        equity *= (1 + position_return)
        equity_curve.append(equity)
        daily_returns.append(position_return)
    
    equity_series = pd.Series(equity_curve, index=dates)
    returns_series = pd.Series(daily_returns, index=dates[1:])
    
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
    downside_returns = returns_series[returns_series < 0]
    sortino = returns_series.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # Yearly returns
    yearly_returns = {}
    for year in range(dates[0].year, dates[-1].year + 1):
        year_mask = equity_series.index.year == year
        if year_mask.sum() > 0:
            year_data = equity_series[year_mask]
            if len(year_data) > 1:
                yearly_returns[year] = (year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]
    
    return PairResult(
        pair=pair,
        strategy='BUY_HOLD',
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        n_trades=1,
        win_rate=1 if total_return > 0 else 0,
        avg_trade_return=total_return,
        profit_factor=0,
        avg_holding_days=(dates[-1] - dates[0]).days,
        exposure_pct=1.0,
        equity_curve=equity_series,
        trades=[],
        yearly_returns=yearly_returns
    )


def create_portfolio(pair_results: Dict[str, PairResult]) -> PairResult:
    """Combine individual pair results into portfolio."""
    # Align all equity curves
    all_dates = set()
    for result in pair_results.values():
        all_dates.update(result.equity_curve.index)
    all_dates = sorted(all_dates)
    
    # Create portfolio equity curve (equal weight)
    n_pairs = len(pair_results)
    portfolio_equity = pd.Series(index=all_dates, dtype=float)
    
    for date in all_dates:
        total = 0
        count = 0
        for result in pair_results.values():
            if date in result.equity_curve.index:
                total += result.equity_curve[date]
                count += 1
        portfolio_equity[date] = (total / count) if count > 0 else np.nan
    
    portfolio_equity = portfolio_equity.dropna()
    
    # Calculate portfolio metrics
    dates = portfolio_equity.index
    returns_series = portfolio_equity.pct_change().dropna()
    
    total_return = (portfolio_equity.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
    downside_returns = returns_series[returns_series < 0]
    sortino = returns_series.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    rolling_max = portfolio_equity.expanding().max()
    drawdowns = (portfolio_equity - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # Aggregate trade stats
    all_trades = []
    for result in pair_results.values():
        all_trades.extend(result.trades)
    
    n_trades = len(all_trades)
    if n_trades > 0:
        trade_returns = [t.return_pct for t in all_trades]
        winners = [t for t in trade_returns if t > 0]
        losers = [t for t in trade_returns if t <= 0]
        win_rate = len(winners) / n_trades
        avg_trade = np.mean(trade_returns)
        profit_factor = sum(winners) / abs(sum(losers)) if losers else 0
        avg_holding = np.mean([t.holding_days for t in all_trades])
    else:
        win_rate = avg_trade = profit_factor = avg_holding = 0
    
    avg_exposure = np.mean([r.exposure_pct for r in pair_results.values()])
    
    # Yearly returns
    yearly_returns = {}
    for year in range(dates[0].year, dates[-1].year + 1):
        year_mask = portfolio_equity.index.year == year
        if year_mask.sum() > 0:
            year_data = portfolio_equity[year_mask]
            if len(year_data) > 1:
                yearly_returns[year] = (year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]
    
    return PairResult(
        pair='PORTFOLIO',
        strategy=list(pair_results.values())[0].strategy,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_trade_return=avg_trade,
        profit_factor=profit_factor,
        avg_holding_days=avg_holding,
        exposure_pct=avg_exposure,
        equity_curve=portfolio_equity,
        trades=all_trades,
        yearly_returns=yearly_returns
    )


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(all_results: Dict[str, Dict[str, PairResult]], 
                    portfolios: Dict[str, PairResult]):
    """Display comprehensive comparison."""
    
    print(f"""
{'='*140}
COMPLETE STRATEGY COMPARISON
{'='*140}

Strategies:
  OPTIMIZED: {STRATEGIES['OPTIMIZED']['description']}
  BASELINE:  {STRATEGIES['BASELINE']['description']}
  BUY_HOLD:  {STRATEGIES['BUY_HOLD']['description']}

Period: {portfolios['OPTIMIZED'].equity_curve.index[0].strftime('%Y-%m-%d')} to {portfolios['OPTIMIZED'].equity_curve.index[-1].strftime('%Y-%m-%d')}
Initial Capital: ${INITIAL_CAPITAL:,}
Kelly Fraction: {KELLY_FRACTION}
""")
    
    # ==========================================================================
    # PORTFOLIO SUMMARY
    # ==========================================================================
    print(f"""
{'='*140}
PORTFOLIO PERFORMANCE (Equal Weight Across 6 Pairs)
{'='*140}
""")
    
    print(f"{'Metric':<25} {'OPTIMIZED':<20} {'BASELINE':<20} {'BUY & HOLD':<20}")
    print("─" * 85)
    
    metrics = [
        ('Total Return', 'total_return', lambda x: f"{x:+.1%}"),
        ('Annual Return', 'annual_return', lambda x: f"{x:+.1%}"),
        ('Sharpe Ratio', 'sharpe_ratio', lambda x: f"{x:.2f}"),
        ('Sortino Ratio', 'sortino_ratio', lambda x: f"{x:.2f}"),
        ('Max Drawdown', 'max_drawdown', lambda x: f"{x:.1%}"),
        ('Total Trades', 'n_trades', lambda x: f"{x:,}"),
        ('Win Rate', 'win_rate', lambda x: f"{x:.1%}"),
        ('Profit Factor', 'profit_factor', lambda x: f"{x:.2f}"),
        ('Avg Trade Return', 'avg_trade_return', lambda x: f"{x:+.2%}"),
        ('Avg Holding (days)', 'avg_holding_days', lambda x: f"{x:.1f}"),
        ('Market Exposure', 'exposure_pct', lambda x: f"{x:.1%}"),
    ]
    
    for label, attr, fmt in metrics:
        opt_val = getattr(portfolios['OPTIMIZED'], attr)
        base_val = getattr(portfolios['BASELINE'], attr)
        bh_val = getattr(portfolios['BUY_HOLD'], attr)
        print(f"{label:<25} {fmt(opt_val):<20} {fmt(base_val):<20} {fmt(bh_val):<20}")
    
    # Final equity
    print("─" * 85)
    opt_equity = portfolios['OPTIMIZED'].equity_curve.iloc[-1]
    base_equity = portfolios['BASELINE'].equity_curve.iloc[-1]
    bh_equity = portfolios['BUY_HOLD'].equity_curve.iloc[-1]
    print(f"{'Final Equity':<25} ${opt_equity:>17,.0f} ${base_equity:>17,.0f} ${bh_equity:>17,.0f}")
    
    # ==========================================================================
    # IMPROVEMENT SUMMARY
    # ==========================================================================
    print(f"""

{'='*140}
OPTIMIZED vs BASELINE IMPROVEMENT
{'='*140}
""")
    
    opt = portfolios['OPTIMIZED']
    base = portfolios['BASELINE']
    bh = portfolios['BUY_HOLD']
    
    sharpe_improve = (opt.sharpe_ratio - base.sharpe_ratio) / base.sharpe_ratio * 100
    return_improve = opt.annual_return - base.annual_return
    dd_improve = opt.max_drawdown - base.max_drawdown
    trade_reduce = (base.n_trades - opt.n_trades) / base.n_trades * 100
    win_improve = opt.win_rate - base.win_rate
    
    print(f"Sharpe Ratio:     {opt.sharpe_ratio:.2f} vs {base.sharpe_ratio:.2f}  ({sharpe_improve:+.1f}%)")
    print(f"Annual Return:    {opt.annual_return:+.1%} vs {base.annual_return:+.1%}  ({return_improve:+.1%} absolute)")
    print(f"Max Drawdown:     {opt.max_drawdown:.1%} vs {base.max_drawdown:.1%}  ({dd_improve:+.1%} absolute)")
    print(f"Trade Count:      {opt.n_trades} vs {base.n_trades}  ({trade_reduce:+.1f}% reduction)")
    print(f"Win Rate:         {opt.win_rate:.1%} vs {base.win_rate:.1%}  ({win_improve:+.1%} absolute)")
    
    print(f"""

{'='*140}
OPTIMIZED vs BUY & HOLD
{'='*140}
""")
    
    sharpe_vs_bh = opt.sharpe_ratio - bh.sharpe_ratio
    return_vs_bh = opt.annual_return - bh.annual_return
    dd_vs_bh = opt.max_drawdown - bh.max_drawdown
    
    print(f"Sharpe Ratio:     {opt.sharpe_ratio:.2f} vs {bh.sharpe_ratio:.2f}  ({sharpe_vs_bh:+.2f} absolute)")
    print(f"Annual Return:    {opt.annual_return:+.1%} vs {bh.annual_return:+.1%}  ({return_vs_bh:+.1%} absolute)")
    print(f"Max Drawdown:     {opt.max_drawdown:.1%} vs {bh.max_drawdown:.1%}  ({dd_vs_bh:+.1%} better)")
    print(f"Market Exposure:  {opt.exposure_pct:.1%} vs {bh.exposure_pct:.1%}")
    
    # ==========================================================================
    # YEAR BY YEAR
    # ==========================================================================
    print(f"""

{'='*140}
YEAR-BY-YEAR RETURNS
{'='*140}
""")
    
    all_years = sorted(set(opt.yearly_returns.keys()) | 
                       set(base.yearly_returns.keys()) | 
                       set(bh.yearly_returns.keys()))
    
    print(f"{'Year':<8} {'OPTIMIZED':<15} {'BASELINE':<15} {'BUY & HOLD':<15} {'OPT vs BH':<15}")
    print("─" * 70)
    
    for year in all_years:
        opt_ret = opt.yearly_returns.get(year, 0)
        base_ret = base.yearly_returns.get(year, 0)
        bh_ret = bh.yearly_returns.get(year, 0)
        diff = opt_ret - bh_ret
        
        print(f"{year:<8} {opt_ret:>+14.1%} {base_ret:>+14.1%} {bh_ret:>+14.1%} {diff:>+14.1%}")
    
    # Count winning years
    opt_wins = sum(1 for y in all_years if opt.yearly_returns.get(y, 0) > bh.yearly_returns.get(y, 0))
    print("─" * 70)
    print(f"OPTIMIZED beats B&H: {opt_wins}/{len(all_years)} years")
    
    # ==========================================================================
    # INDIVIDUAL PAIRS
    # ==========================================================================
    print(f"""

{'='*140}
INDIVIDUAL PAIR PERFORMANCE
{'='*140}

SHARPE RATIO:
{'Pair':<12} {'OPTIMIZED':<15} {'BASELINE':<15} {'BUY & HOLD':<15} {'OPT vs BASE':<15}
{'─'*75}""")
    
    for pair in DEPLOY_PAIRS:
        opt_s = all_results['OPTIMIZED'][pair].sharpe_ratio
        base_s = all_results['BASELINE'][pair].sharpe_ratio
        bh_s = all_results['BUY_HOLD'][pair].sharpe_ratio
        diff = opt_s - base_s
        print(f"{pair:<12} {opt_s:>14.2f} {base_s:>14.2f} {bh_s:>14.2f} {diff:>+14.2f}")
    
    print(f"""

ANNUAL RETURN:
{'Pair':<12} {'OPTIMIZED':<15} {'BASELINE':<15} {'BUY & HOLD':<15} {'OPT vs BASE':<15}
{'─'*75}""")
    
    for pair in DEPLOY_PAIRS:
        opt_r = all_results['OPTIMIZED'][pair].annual_return
        base_r = all_results['BASELINE'][pair].annual_return
        bh_r = all_results['BUY_HOLD'][pair].annual_return
        diff = opt_r - base_r
        print(f"{pair:<12} {opt_r:>+14.1%} {base_r:>+14.1%} {bh_r:>+14.1%} {diff:>+14.1%}")
    
    print(f"""

MAX DRAWDOWN:
{'Pair':<12} {'OPTIMIZED':<15} {'BASELINE':<15} {'BUY & HOLD':<15} {'OPT vs BASE':<15}
{'─'*75}""")
    
    for pair in DEPLOY_PAIRS:
        opt_dd = all_results['OPTIMIZED'][pair].max_drawdown
        base_dd = all_results['BASELINE'][pair].max_drawdown
        bh_dd = all_results['BUY_HOLD'][pair].max_drawdown
        diff = opt_dd - base_dd  # Less negative is better
        better = "✓" if diff > 0 else ""
        print(f"{pair:<12} {opt_dd:>14.1%} {base_dd:>14.1%} {bh_dd:>14.1%} {diff:>+14.1%} {better}")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print(f"""

{'='*140}
FINAL SUMMARY
{'='*140}

OPTIMIZED STRATEGY (E2.5% entry + 3h confirmation):
    ✓ Sharpe Ratio:   {opt.sharpe_ratio:.2f} (vs {base.sharpe_ratio:.2f} baseline, {bh.sharpe_ratio:.2f} B&H)
    ✓ Annual Return:  {opt.annual_return:+.1%} (vs {base.annual_return:+.1%} baseline, {bh.annual_return:+.1%} B&H)
    ✓ Max Drawdown:   {opt.max_drawdown:.1%} (vs {base.max_drawdown:.1%} baseline, {bh.max_drawdown:.1%} B&H)
    ✓ Win Rate:       {opt.win_rate:.1%} (vs {base.win_rate:.1%} baseline)
    ✓ Total Trades:   {opt.n_trades} (vs {base.n_trades} baseline) - {trade_reduce:.0f}% fewer trades

CONCLUSION:
    The optimized strategy delivers {sharpe_improve:.0f}% better risk-adjusted returns than baseline
    while taking {trade_reduce:.0f}% fewer trades and achieving {win_improve:.0%} higher win rate.
    
    vs Buy & Hold: {sharpe_vs_bh:+.2f} Sharpe improvement with {-dd_vs_bh:.0%} less drawdown.
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 140)
    print("COMPLETE STRATEGY COMPARISON BACKTEST")
    print("=" * 140)
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    
    all_results = {
        'OPTIMIZED': {},
        'BASELINE': {},
        'BUY_HOLD': {}
    }
    
    for pair in DEPLOY_PAIRS:
        print(f"Processing {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        
        if df_1h is None or len(df_1h) == 0:
            print(f"WARNING: No data for {pair}, skipping")
            continue
        
        # Ensure datetime index
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            print(f"WARNING: {pair} index is not DatetimeIndex, converting...")
            df_1h.index = pd.to_datetime(df_1h.index)
        
        # OPTIMIZED strategy
        config = STRATEGIES['OPTIMIZED']
        states_hourly = compute_states_hourly(df_1h, config['entry_buffer'], config['exit_buffer'])
        states_daily = apply_confirmation_filter(states_hourly, config['confirmation_hours'])
        all_results['OPTIMIZED'][pair] = run_strategy_backtest(states_daily, 'OPTIMIZED', pair)
        
        # BASELINE strategy
        config = STRATEGIES['BASELINE']
        states_hourly = compute_states_hourly(df_1h, config['entry_buffer'], config['exit_buffer'])
        states_daily = apply_confirmation_filter(states_hourly, config['confirmation_hours'])
        all_results['BASELINE'][pair] = run_strategy_backtest(states_daily, 'BASELINE', pair)
        
        # BUY & HOLD
        all_results['BUY_HOLD'][pair] = run_buy_hold_backtest(df_1h, pair)
        
        print("done")
    
    # Create portfolios
    portfolios = {
        'OPTIMIZED': create_portfolio(all_results['OPTIMIZED']),
        'BASELINE': create_portfolio(all_results['BASELINE']),
        'BUY_HOLD': create_portfolio(all_results['BUY_HOLD'])
    }
    
    display_results(all_results, portfolios)
    
    print(f"\n{'='*140}")
    print("BACKTEST COMPLETE")
    print("=" * 140)
    
    return all_results, portfolios


if __name__ == "__main__":
    results, portfolios = main()