#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Boost Testing
======================
Test higher boost levels and boosts on multiple states.

Based on hazard analysis findings:
- State 11: P(→bearish) DECREASES with duration (17.1% early → 7.0% late) - MOMENTUM
- State 15: P(→bearish) stays very low (0-6%) - SAFE
- State 13: P(→bearish) INCREASES late (0% → 16.7%) - NOT boost candidate

Tests:
1. State 11 boosts: 125%, 150%, 175%, 200%, 250%
2. State 15 boosts: 125%, 150%, 175%, 200%
3. Combined state 11 + 15 boosts
4. Duration thresholds: 24h, 48h, 72h, 96h

Usage:
    python extended_boost_test.py
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
from typing import Dict, List, Callable
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

# Boost levels to test
BOOST_LEVELS = [1.25, 1.50, 1.75, 2.00, 2.50]

# Duration thresholds to test
DURATION_THRESHOLDS = [24, 48, 72, 96]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a single backtest."""
    pair: str
    config_name: str
    total_return: float = 0
    annual_return: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    max_drawdown: float = 0
    calmar_ratio: float = 0
    n_trades: int = 0
    win_rate: float = 0
    avg_position_size: float = 0
    exposure_pct: float = 0
    equity_curve: pd.Series = None
    yearly_returns: Dict[int, float] = field(default_factory=dict)


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


def compute_states_hourly(df_1h: pd.DataFrame) -> pd.DataFrame:
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
    
    # Track duration
    durations = []
    current_duration = 1
    for i in range(n):
        if i > 0 and confirmed_states[i] == confirmed_states[i-1]:
            current_duration += 1
        else:
            current_duration = 1
        durations.append(current_duration)
    
    states['duration_hours'] = durations
    
    return states.dropna()


# =============================================================================
# STRATEGY FACTORIES
# =============================================================================

def strategy_baseline(state: int, duration: int) -> float:
    if state in BULLISH_STATES:
        return 1.0
    return 0.0


def strategy_exclude_12_14(state: int, duration: int) -> float:
    if state in [12, 14]:
        return 0.0
    if state in BULLISH_STATES:
        return 1.0
    return 0.0


def make_state11_boost(duration_threshold: int, boost: float):
    """State 11 only boost."""
    def strategy(state: int, duration: int) -> float:
        if state in [12, 14]:
            return 0.0
        if state == 11 and duration >= duration_threshold:
            return boost
        if state in BULLISH_STATES:
            return 1.0
        return 0.0
    return strategy


def make_state15_boost(duration_threshold: int, boost: float):
    """State 15 only boost."""
    def strategy(state: int, duration: int) -> float:
        if state in [12, 14]:
            return 0.0
        if state == 15 and duration >= duration_threshold:
            return boost
        if state in BULLISH_STATES:
            return 1.0
        return 0.0
    return strategy


def make_dual_boost(duration_threshold: int, boost_11: float, boost_15: float):
    """Both state 11 and 15 boost."""
    def strategy(state: int, duration: int) -> float:
        if state in [12, 14]:
            return 0.0
        if state == 11 and duration >= duration_threshold:
            return boost_11
        if state == 15 and duration >= duration_threshold:
            return boost_15
        if state in BULLISH_STATES:
            return 1.0
        return 0.0
    return strategy


def make_triple_boost(duration_threshold: int, boost_11: float, boost_15: float, boost_9: float):
    """States 9, 11, 15 boost."""
    def strategy(state: int, duration: int) -> float:
        if state in [12, 14]:
            return 0.0
        if state == 11 and duration >= duration_threshold:
            return boost_11
        if state == 15 and duration >= duration_threshold:
            return boost_15
        if state == 9 and duration >= duration_threshold:
            return boost_9
        if state in BULLISH_STATES:
            return 1.0
        return 0.0
    return strategy


def make_graduated_boost(state_target: int):
    """Graduated boost based on duration for specific state."""
    def strategy(state: int, duration: int) -> float:
        if state in [12, 14]:
            return 0.0
        if state == state_target:
            if duration >= 96:
                return 2.0
            elif duration >= 72:
                return 1.75
            elif duration >= 48:
                return 1.5
            elif duration >= 24:
                return 1.25
        if state in BULLISH_STATES:
            return 1.0
        return 0.0
    return strategy


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(states: pd.DataFrame, config_name: str, pair: str,
                 strategy_func: Callable) -> BacktestResult:
    
    prices = states['close'].values
    confirmed_states = states['confirmed_state'].values
    durations = states['duration_hours'].values
    dates = states.index
    n = len(prices)
    
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    hourly_returns = []
    position_sizes = []
    
    position_size = 0
    entry_price = 0
    entry_date = None
    
    n_trades = 0
    trade_returns = []
    hours_in_market = 0
    
    for i in range(1, n):
        current_state = int(confirmed_states[i])
        duration = int(durations[i])
        
        target_size = strategy_func(state=current_state, duration=duration)
        
        price_return = (prices[i] - prices[i-1]) / prices[i-1]
        
        if position_size > 0:
            position_return = position_size * KELLY_FRACTION * price_return
            equity *= (1 + position_return)
            hourly_returns.append(position_return)
            hours_in_market += 1
        else:
            hourly_returns.append(0)
        
        position_sizes.append(target_size)
        
        was_in_position = position_size > 0
        should_be_in_position = target_size > 0
        
        if should_be_in_position and not was_in_position:
            entry_price = prices[i]
            entry_date = dates[i]
            n_trades += 1
        elif not should_be_in_position and was_in_position:
            if entry_price > 0:
                trade_return = (prices[i] - entry_price) / entry_price
                trade_returns.append(trade_return)
        
        position_size = target_size
        equity_curve.append(equity)
    
    # Close final position
    if position_size > 0 and entry_price > 0:
        trade_return = (prices[-1] - entry_price) / entry_price
        trade_returns.append(trade_return)
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=dates)
    returns_series = pd.Series(hourly_returns, index=dates[1:])
    
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    if returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(24 * 365)
        downside_returns = returns_series[returns_series < 0]
        sortino = returns_series.mean() / downside_returns.std() * np.sqrt(24 * 365) if len(downside_returns) > 0 else 0
    else:
        sharpe = sortino = 0
    
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    if trade_returns:
        win_rate = len([t for t in trade_returns if t > 0]) / len(trade_returns)
    else:
        win_rate = 0
    
    avg_pos_size = np.mean([p for p in position_sizes if p > 0]) if any(p > 0 for p in position_sizes) else 0
    exposure_pct = hours_in_market / n if n > 0 else 0
    
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
        avg_position_size=avg_pos_size,
        exposure_pct=exposure_pct,
        equity_curve=equity_series,
        yearly_returns=yearly_returns,
    )


def create_portfolio(pair_results: Dict[str, BacktestResult]) -> BacktestResult:
    if not pair_results:
        return BacktestResult(pair='PORTFOLIO', config_name='')
    
    config_name = list(pair_results.values())[0].config_name
    
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
        avg_position_size=avg_pos_size,
        exposure_pct=avg_exposure,
        equity_curve=portfolio_curve,
        yearly_returns=yearly_returns,
    )


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(portfolios: Dict[str, BacktestResult]):
    print(f"\n{'=' * 140}")
    print("EXTENDED BOOST TEST RESULTS")
    print(f"{'=' * 140}")
    
    sorted_configs = sorted(portfolios.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
    
    print(f"\n  {'Config':<40} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10} {'Calmar':>8} "
          f"{'AvgPos':>8} {'Exposure':>10}")
    print(f"  {'─' * 120}")
    
    baseline_sharpe = portfolios.get('BASELINE', BacktestResult(pair='', config_name='')).sharpe_ratio
    
    for config, result in sorted_configs:
        improvement = ""
        if config != 'BASELINE' and baseline_sharpe > 0:
            pct = (result.sharpe_ratio - baseline_sharpe) / baseline_sharpe * 100
            improvement = f" ({pct:+.0f}%)"
        
        print(f"  {config:<40} {result.sharpe_ratio:>8.2f} {result.annual_return:>+9.1%} "
              f"{result.max_drawdown:>9.1%} {result.calmar_ratio:>8.2f} {result.avg_position_size:>7.0%} "
              f"{result.exposure_pct:>9.1%}{improvement}")


def display_state11_analysis(portfolios: Dict[str, BacktestResult]):
    print(f"\n{'=' * 140}")
    print("STATE 11 BOOST ANALYSIS")
    print(f"{'=' * 140}")
    
    # Group by duration threshold
    for dur in DURATION_THRESHOLDS:
        print(f"\n  Duration Threshold: {dur}h")
        print(f"  {'Boost':>8} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Calmar':>10}")
        print(f"  {'─' * 60}")
        
        for boost in BOOST_LEVELS:
            name = f'S11_{dur}h_{int(boost*100)}%'
            if name in portfolios:
                r = portfolios[name]
                print(f"  {int(boost*100):>7}% {r.sharpe_ratio:>10.2f} {r.annual_return:>+11.1%} "
                      f"{r.max_drawdown:>9.1%} {r.calmar_ratio:>10.2f}")


def display_state15_analysis(portfolios: Dict[str, BacktestResult]):
    print(f"\n{'=' * 140}")
    print("STATE 15 BOOST ANALYSIS")
    print(f"{'=' * 140}")
    
    for dur in DURATION_THRESHOLDS:
        print(f"\n  Duration Threshold: {dur}h")
        print(f"  {'Boost':>8} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Calmar':>10}")
        print(f"  {'─' * 60}")
        
        for boost in BOOST_LEVELS:
            name = f'S15_{dur}h_{int(boost*100)}%'
            if name in portfolios:
                r = portfolios[name]
                print(f"  {int(boost*100):>7}% {r.sharpe_ratio:>10.2f} {r.annual_return:>+11.1%} "
                      f"{r.max_drawdown:>9.1%} {r.calmar_ratio:>10.2f}")


def display_dual_boost_analysis(portfolios: Dict[str, BacktestResult]):
    print(f"\n{'=' * 140}")
    print("DUAL BOOST (STATE 11 + 15) ANALYSIS")
    print(f"{'=' * 140}")
    
    dual_configs = {k: v for k, v in portfolios.items() if k.startswith('DUAL')}
    
    if dual_configs:
        sorted_dual = sorted(dual_configs.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
        
        print(f"\n  {'Config':<35} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Calmar':>10}")
        print(f"  {'─' * 85}")
        
        for config, r in sorted_dual[:15]:  # Top 15
            print(f"  {config:<35} {r.sharpe_ratio:>10.2f} {r.annual_return:>+11.1%} "
                  f"{r.max_drawdown:>9.1%} {r.calmar_ratio:>10.2f}")


def display_top_performers(portfolios: Dict[str, BacktestResult]):
    print(f"\n{'=' * 140}")
    print("TOP 10 PERFORMERS")
    print(f"{'=' * 140}")
    
    sorted_all = sorted(portfolios.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)[:10]
    
    print(f"\n  {'Rank':<6} {'Config':<40} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10} {'Calmar':>8}")
    print(f"  {'─' * 95}")
    
    for i, (config, r) in enumerate(sorted_all, 1):
        print(f"  {i:<6} {config:<40} {r.sharpe_ratio:>8.2f} {r.annual_return:>+9.1%} "
              f"{r.max_drawdown:>9.1%} {r.calmar_ratio:>8.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 140)
    print("EXTENDED BOOST TESTING")
    print("=" * 140)
    print(f"""
    Testing higher boost levels and multiple states:
    
    Boost levels: {[f'{int(b*100)}%' for b in BOOST_LEVELS]}
    Duration thresholds: {DURATION_THRESHOLDS} hours
    
    Strategies:
    1. State 11 only boosts (momentum effect)
    2. State 15 only boosts (very safe state)
    3. Dual boost (states 11 + 15)
    4. Graduated boost (increases with duration)
    """)
    
    if not HAS_DATABASE:
        print("ERROR: Database not available")
        return
    
    db = Database()
    
    # Build all strategies
    strategies = {
        'BASELINE': strategy_baseline,
        'EXCLUDE_12_14': strategy_exclude_12_14,
    }
    
    # State 11 boosts
    for dur in DURATION_THRESHOLDS:
        for boost in BOOST_LEVELS:
            name = f'S11_{dur}h_{int(boost*100)}%'
            strategies[name] = make_state11_boost(dur, boost)
    
    # State 15 boosts
    for dur in DURATION_THRESHOLDS:
        for boost in BOOST_LEVELS:
            name = f'S15_{dur}h_{int(boost*100)}%'
            strategies[name] = make_state15_boost(dur, boost)
    
    # Dual boosts (state 11 + 15)
    for dur in [48, 72]:
        for boost_11 in [1.50, 1.75, 2.00, 2.50]:
            for boost_15 in [1.25, 1.50, 1.75]:
                name = f'DUAL_{dur}h_S11_{int(boost_11*100)}_S15_{int(boost_15*100)}'
                strategies[name] = make_dual_boost(dur, boost_11, boost_15)
    
    # Graduated boosts
    strategies['GRAD_S11'] = make_graduated_boost(11)
    strategies['GRAD_S15'] = make_graduated_boost(15)
    
    # Combined graduated
    def combined_graduated(state: int, duration: int) -> float:
        if state in [12, 14]:
            return 0.0
        if state == 11:
            if duration >= 96: return 2.0
            elif duration >= 72: return 1.75
            elif duration >= 48: return 1.5
            elif duration >= 24: return 1.25
        if state == 15:
            if duration >= 96: return 1.75
            elif duration >= 72: return 1.5
            elif duration >= 48: return 1.25
        if state in BULLISH_STATES:
            return 1.0
        return 0.0
    
    strategies['GRAD_COMBINED'] = combined_graduated
    
    print(f"\nTotal strategies to test: {len(strategies)}")
    
    all_results = {name: {} for name in strategies.keys()}
    
    # Process each pair
    for pair in DEPLOY_PAIRS:
        print(f"\nProcessing {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        if df_1h is None or len(df_1h) == 0:
            print(f"WARNING: No data for {pair}")
            continue
        
        print(f"loaded {len(df_1h):,} hours...", end=" ", flush=True)
        
        states = compute_states_hourly(df_1h)
        print(f"states computed...", end=" ", flush=True)
        
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
    display_top_performers(portfolios)
    display_state11_analysis(portfolios)
    display_state15_analysis(portfolios)
    display_dual_boost_analysis(portfolios)
    
    print(f"\n{'=' * 140}")
    print("TESTING COMPLETE")
    print("=' * 140}")
    
    return all_results, portfolios


if __name__ == "__main__":
    results, portfolios = main()