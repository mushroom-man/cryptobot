#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Grid Search - FULLY CORRECTED
==============================================
Fixes:
1. Higher TF signals shifted by 1 period (no look-ahead bias)
2. Backtest uses position[T-1] * return[T] (correct timing)
3. Hit rates measure forward returns correctly

Usage:
    python multi_tf_grid_search_final.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from enum import IntEnum
from typing import Dict, Tuple, List
from dataclasses import dataclass
from itertools import product
import warnings
import time
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

PAIR = 'XBTUSD'
DATA_START = '2017-01-01'
DATA_END = '2025-11-30'

TF_EXECUTION = '24h'
TF_MEDIUM = '72h'
TF_SLOW = '168h'

# Grid search parameters
MA_PERIODS_24H = [12, 24, 36, 48]
MA_PERIODS_72H = [4, 8, 12, 16]
MA_PERIODS_168H = [2, 4, 6, 8]

ENTRY_BUFFERS = [0.01, 0.02, 0.03, 0.04]
EXIT_BUFFERS = [0.005, 0.01, 0.015, 0.02]

# Strategy settings
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES = 30

# Trading costs
TOTAL_COST = 0.0015


# =============================================================================
# TREND LABELING
# =============================================================================

class Trend(IntEnum):
    DOWN = 0
    UP = 1


def label_trend_binary(df: pd.DataFrame, ma_period: int, entry_buffer: float, exit_buffer: float) -> pd.Series:
    """Binary trend detection with hysteresis."""
    close = df['close']
    ma = close.rolling(ma_period).mean()
    
    enter_up = ma * (1 + entry_buffer)
    exit_up = ma * (1 - exit_buffer)
    enter_down = ma * (1 - entry_buffer)
    exit_down = ma * (1 + exit_buffer)
    
    labels = pd.Series(index=df.index, dtype=int)
    current = Trend.UP
    
    for i in range(len(df)):
        if pd.isna(ma.iloc[i]):
            labels.iloc[i] = current
            continue
        
        price = close.iloc[i]
        
        if current == Trend.UP:
            if price < exit_up.iloc[i] and price < enter_down.iloc[i]:
                current = Trend.DOWN
        elif current == Trend.DOWN:
            if price > exit_down.iloc[i] and price > enter_up.iloc[i]:
                current = Trend.UP
        
        labels.iloc[i] = current
    
    return labels


# =============================================================================
# DATA UTILITIES
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def align_signals_to_24h(df_24h: pd.DataFrame, trend_24h: pd.Series, 
                          trend_72h: pd.Series, trend_168h: pd.Series) -> pd.DataFrame:
    """
    Align signals with proper shifts to avoid look-ahead bias.
    
    Signal at index T represents what we KNOW at time T:
    - 24h: shift(1) = yesterday's signal
    - 72h: shift(1) on 72h bars, then ffill = last completed 72h period
    - 168h: shift(1) on 168h bars, then ffill = last completed 168h period
    """
    aligned = pd.DataFrame(index=df_24h.index)
    
    # 24h: use yesterday's signal for today
    aligned['trend_24h'] = trend_24h.shift(1)
    
    # 72h/168h: shift by 1 period, then forward-fill
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    
    return aligned.dropna().astype(int)


# =============================================================================
# HIT RATE CALCULATION
# =============================================================================

def calculate_hit_rates(df_24h: pd.DataFrame, signals: pd.DataFrame) -> Dict[Tuple[int, int, int], Dict]:
    """
    Calculate hit rate for each permutation.
    
    Hit rate = P(return > 0 | signal state)
    
    For signal at time T, we measure return from T to T+1.
    This is what we'd earn if we enter at close of T and exit at close of T+1.
    """
    all_perms = list(product([0, 1], repeat=3))
    
    # Return from T to T+1 (forward return)
    # pct_change() at T = return from T-1 to T
    # So pct_change().shift(-1) at T = return from T to T+1
    forward_returns = df_24h['close'].pct_change().shift(-1)
    
    hit_rates = {}
    
    for perm in all_perms:
        mask = (
            (signals['trend_24h'] == perm[0]) &
            (signals['trend_72h'] == perm[1]) &
            (signals['trend_168h'] == perm[2])
        )
        
        perm_indices = signals[mask].index
        perm_returns = forward_returns.loc[perm_indices].dropna()
        
        n = len(perm_returns)
        if n > 0:
            n_wins = (perm_returns > 0).sum()
            hit_rate = n_wins / n
            avg_win = perm_returns[perm_returns > 0].mean() if n_wins > 0 else 0
            avg_loss = abs(perm_returns[perm_returns < 0].mean()) if (n - n_wins) > 0 else 0
        else:
            hit_rate, n_wins, avg_win, avg_loss = 0.5, 0, 0, 0
        
        hit_rates[perm] = {
            'n': n, 'n_wins': n_wins, 'hit_rate': hit_rate,
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'sufficient': n >= MIN_SAMPLES,
        }
    
    return hit_rates


# =============================================================================
# POSITION GENERATION
# =============================================================================

def get_positions_8state(signals: pd.DataFrame, hit_rates: Dict) -> pd.Series:
    """8-state binary: 100% if hit_rate > threshold, else 0%."""
    positions = pd.Series(index=signals.index, dtype=float)
    
    for idx in signals.index:
        perm = (int(signals.loc[idx, 'trend_24h']), 
                int(signals.loc[idx, 'trend_72h']), 
                int(signals.loc[idx, 'trend_168h']))
        
        data = hit_rates[perm]
        if not data['sufficient']:
            positions.loc[idx] = 0.50
        elif data['hit_rate'] > HIT_RATE_THRESHOLD:
            positions.loc[idx] = 1.00
        else:
            positions.loc[idx] = 0.00
    
    return positions


def get_positions_168h_only(signals: pd.DataFrame) -> pd.Series:
    """168h-only: 100% if 168h UP, else 0%."""
    return signals['trend_168h'].map({1: 1.0, 0: 0.0}).astype(float)


# =============================================================================
# BACKTESTING
# =============================================================================

@dataclass
class BacktestResult:
    total_return: float
    bh_return: float
    alpha: float
    sharpe: float
    bh_sharpe: float
    max_dd: float
    bh_max_dd: float
    avg_position: float
    n_trades: int
    n_bars: int
    hit_rates: Dict


def backtest(df_24h: pd.DataFrame, signals: pd.DataFrame, 
             positions: pd.Series, hit_rates: Dict) -> BacktestResult:
    """
    Backtest with CORRECT timing alignment.
    
    Position at T determines exposure from T to T+1.
    Return from T to T+1 = pct_change() at T+1.
    
    So: strategy_return[T+1] = position[T] * pct_change[T+1]
    Or equivalently: strategy_return[T] = position[T-1] * pct_change[T]
    """
    # Align to common index
    common_idx = positions.index.intersection(df_24h.index)
    positions = positions.loc[common_idx]
    prices = df_24h.loc[common_idx, 'close']
    
    # Returns: pct_change at T = return from T-1 to T
    returns = prices.pct_change()
    
    # Position at T-1 applied to return at T
    # This means: decision made at close of T-1, held through T
    lagged_positions = positions.shift(1)
    
    # Align
    valid_idx = returns.dropna().index
    returns = returns.loc[valid_idx]
    lagged_positions = lagged_positions.loc[valid_idx].fillna(0)
    
    # Transaction costs (based on position changes)
    position_changes = lagged_positions.diff().abs().fillna(0)
    costs = position_changes * TOTAL_COST
    
    # Strategy returns
    strategy_returns = lagged_positions * returns - costs
    
    # Buy & hold
    bh_returns = returns
    
    if len(strategy_returns) == 0:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, hit_rates)
    
    # Equity curves
    strategy_equity = (1 + strategy_returns).cumprod()
    bh_equity = (1 + bh_returns).cumprod()
    
    # Metrics
    total_return = strategy_equity.iloc[-1] - 1
    bh_return = bh_equity.iloc[-1] - 1
    alpha = total_return - bh_return
    
    bars_per_year = 365
    strategy_sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(bars_per_year)
                       if strategy_returns.std() > 0 else 0)
    bh_sharpe = (bh_returns.mean() / bh_returns.std() * np.sqrt(bars_per_year)
                 if bh_returns.std() > 0 else 0)
    
    # Max drawdown
    rolling_max = strategy_equity.expanding().max()
    max_dd = ((strategy_equity - rolling_max) / rolling_max).min()
    
    bh_rolling_max = bh_equity.expanding().max()
    bh_max_dd = ((bh_equity - bh_rolling_max) / bh_rolling_max).min()
    
    n_trades = (position_changes > 0.05).sum()
    avg_pos = lagged_positions.mean()
    
    return BacktestResult(
        total_return, bh_return, alpha, strategy_sharpe, bh_sharpe,
        max_dd, bh_max_dd, avg_pos, n_trades, len(strategy_returns), hit_rates
    )


# =============================================================================
# GRID SEARCH
# =============================================================================

@dataclass
class GridResult:
    ma_24h: int
    ma_72h: int
    ma_168h: int
    entry_buffer: float
    exit_buffer: float
    bt_8state: BacktestResult
    bt_168h_only: BacktestResult
    
    @property
    def score_8state(self) -> float:
        bt = self.bt_8state
        if bt.bh_return <= 0: return 0
        return (1 + bt.total_return) / (1 + bt.bh_return) * (bt.sharpe / bt.bh_sharpe if bt.bh_sharpe > 0 else 1)
    
    @property
    def score_168h(self) -> float:
        bt = self.bt_168h_only
        if bt.bh_return <= 0: return 0
        return (1 + bt.total_return) / (1 + bt.bh_return) * (bt.sharpe / bt.bh_sharpe if bt.bh_sharpe > 0 else 1)


def run_grid_search(df_24h, df_72h, df_168h) -> List[GridResult]:
    ma_combos = list(product(MA_PERIODS_24H, MA_PERIODS_72H, MA_PERIODS_168H))
    hyst_combos = [(e, x) for e, x in product(ENTRY_BUFFERS, EXIT_BUFFERS) if x < e]
    
    total = len(ma_combos) * len(hyst_combos)
    print(f"\nRunning grid search: {len(ma_combos)} MA × {len(hyst_combos)} hysteresis = {total} combinations")
    print("-" * 80)
    
    results = []
    start = time.time()
    
    for i, ((ma_24, ma_72, ma_168), (entry, exit)) in enumerate(product(ma_combos, hyst_combos)):
        if i % 50 == 0:
            elapsed = time.time() - start
            eta = (total - i) / ((i + 1) / elapsed) if elapsed > 0 else 0
            print(f"  Progress: {i}/{total} ({i/total*100:.1f}%) - ETA: {eta:.0f}s")
        
        trend_24h = label_trend_binary(df_24h, ma_24, entry, exit)
        trend_72h = label_trend_binary(df_72h, ma_72, entry, exit)
        trend_168h = label_trend_binary(df_168h, ma_168, entry, exit)
        
        signals = align_signals_to_24h(df_24h, trend_24h, trend_72h, trend_168h)
        if len(signals) < 100: continue
        
        hit_rates = calculate_hit_rates(df_24h, signals)
        
        pos_8state = get_positions_8state(signals, hit_rates)
        pos_168h = get_positions_168h_only(signals)
        
        bt_8state = backtest(df_24h, signals, pos_8state, hit_rates)
        bt_168h = backtest(df_24h, signals, pos_168h, hit_rates)
        
        results.append(GridResult(ma_24, ma_72, ma_168, entry, exit, bt_8state, bt_168h))
    
    print(f"\nCompleted {len(results)} combinations in {time.time()-start:.1f}s")
    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results(results: List[GridResult], strategy: str, n: int = 15):
    if strategy == '8state':
        sorted_r = sorted(results, key=lambda x: x.score_8state, reverse=True)
        get_bt = lambda x: x.bt_8state
    else:
        sorted_r = sorted(results, key=lambda x: x.score_168h, reverse=True)
        get_bt = lambda x: x.bt_168h_only
    
    print(f"\n{'MA24':>5} {'MA72':>5} {'MA168':>6} {'Entry':>6} {'Exit':>5} │ "
          f"{'Return':>9} {'B&H':>9} {'Alpha':>9} │ {'Sharpe':>7} {'B&H_Sh':>7} │ {'MaxDD':>7} {'AvgPos':>7}")
    print("-" * 110)
    
    for r in sorted_r[:n]:
        bt = get_bt(r)
        print(f"{r.ma_24h:>5} {r.ma_72h:>5} {r.ma_168h:>6} {r.entry_buffer*100:>5.1f}% {r.exit_buffer*100:>4.1f}% │ "
              f"{bt.total_return*100:>+8.0f}% {bt.bh_return*100:>+8.0f}% {bt.alpha*100:>+8.0f}% │ "
              f"{bt.sharpe:>7.2f} {bt.bh_sharpe:>7.2f} │ {bt.max_dd*100:>6.1f}% {bt.avg_position*100:>6.1f}%")


def print_hit_rates(hit_rates: Dict):
    print("\n    HIT RATE MATRIX:")
    print("    " + "-" * 65)
    print(f"    {'Perm':<8} {'N':>6} {'Hit%':>7} {'AvgWin':>8} {'AvgLoss':>9} {'Edge':>7} {'Pos':>6}")
    print("    " + "-" * 65)
    
    names = {0: 'D', 1: 'U'}
    for perm, d in sorted(hit_rates.items(), key=lambda x: x[1]['hit_rate'], reverse=True):
        perm_name = f"{names[perm[0]]}/{names[perm[1]]}/{names[perm[2]]}"
        edge = (d['hit_rate'] - 0.5) * 100
        pos = "100%" if d['sufficient'] and d['hit_rate'] > 0.5 else ("0%" if d['sufficient'] else "50%*")
        print(f"    {perm_name:<8} {d['n']:>6} {d['hit_rate']*100:>6.1f}% {d['avg_win']*100:>+7.2f}% "
              f"{-d['avg_loss']*100:>8.2f}% {edge:>+6.1f}% {pos:>6}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("MULTI-TIMEFRAME GRID SEARCH - FULLY CORRECTED")
    print("=" * 80)
    
    print("""
    FIXES APPLIED:
    1. Signal shift: Higher TF signals shifted by 1 period before ffill
    2. Backtest timing: position[T-1] * return[T] (correct alignment)
    3. Hit rates: Measure forward return from T to T+1
    """)
    
    # Load data
    print("-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    db = Database()
    df_1h = db.get_ohlcv(PAIR, start=DATA_START, end=DATA_END)
    print(f"Loaded {len(df_1h):,} 1h bars: {df_1h.index.min().date()} to {df_1h.index.max().date()}")
    
    df_24h = resample_ohlcv(df_1h, TF_EXECUTION)
    df_72h = resample_ohlcv(df_1h, TF_MEDIUM)
    df_168h = resample_ohlcv(df_1h, TF_SLOW)
    print(f"Resampled: 24h={len(df_24h)}, 72h={len(df_72h)}, 168h={len(df_168h)}")
    
    # Grid search
    print("\n" + "=" * 80)
    print("GRID SEARCH")
    print("=" * 80)
    
    results = run_grid_search(df_24h, df_72h, df_168h)
    
    # 8-state results
    print("\n" + "=" * 80)
    print("RESULTS: 8-STATE BINARY STRATEGY")
    print("=" * 80)
    print_results(results, '8state')
    
    best_8 = max(results, key=lambda x: x.score_8state)
    bt_8 = best_8.bt_8state
    print(f"\n    BEST 8-STATE: MA({best_8.ma_24h}/{best_8.ma_72h}/{best_8.ma_168h}), "
          f"Hyst({best_8.entry_buffer*100:.1f}%/{best_8.exit_buffer*100:.1f}%)")
    print(f"    Return: {bt_8.total_return*100:+.0f}% | Alpha: {bt_8.alpha*100:+.0f}% | "
          f"Sharpe: {bt_8.sharpe:.2f} | MaxDD: {bt_8.max_dd*100:.1f}%")
    print_hit_rates(bt_8.hit_rates)
    
    # 168h-only results
    print("\n" + "=" * 80)
    print("RESULTS: 168H-ONLY STRATEGY")
    print("=" * 80)
    print_results(results, '168h_only')
    
    best_168 = max(results, key=lambda x: x.score_168h)
    bt_168 = best_168.bt_168h_only
    print(f"\n    BEST 168H-ONLY: MA_168h={best_168.ma_168h}, "
          f"Hyst({best_168.entry_buffer*100:.1f}%/{best_168.exit_buffer*100:.1f}%)")
    print(f"    Return: {bt_168.total_return*100:+.0f}% | Alpha: {bt_168.alpha*100:+.0f}% | "
          f"Sharpe: {bt_168.sharpe:.2f} | MaxDD: {bt_168.max_dd*100:.1f}%")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"""
    ┌───────────────────────────────────────────────────────────────────┐
    │  METRIC              8-STATE        168H-ONLY        B&H         │
    ├───────────────────────────────────────────────────────────────────┤
    │  Return          {bt_8.total_return*100:>+9.0f}%      {bt_168.total_return*100:>+9.0f}%    {bt_8.bh_return*100:>+9.0f}%   │
    │  Alpha           {bt_8.alpha*100:>+9.0f}%      {bt_168.alpha*100:>+9.0f}%           0%   │
    │  Sharpe          {bt_8.sharpe:>9.2f}       {bt_168.sharpe:>9.2f}      {bt_8.bh_sharpe:>9.2f}   │
    │  Max Drawdown    {bt_8.max_dd*100:>9.1f}%      {bt_168.max_dd*100:>9.1f}%      {bt_8.bh_max_dd*100:>9.1f}%   │
    │  Avg Position    {bt_8.avg_position*100:>9.1f}%      {bt_168.avg_position*100:>9.1f}%        100.0%   │
    └───────────────────────────────────────────────────────────────────┘
    """)
    
    # Sanity check
    print("=" * 80)
    print("SANITY CHECK")
    print("=" * 80)
    ratio_8 = (1 + bt_8.total_return) / (1 + bt_8.bh_return)
    ratio_168 = (1 + bt_168.total_return) / (1 + bt_168.bh_return)
    print(f"""
    8-STATE vs B&H:    {ratio_8:.2f}x
    168H-ONLY vs B&H:  {ratio_168:.2f}x
    
    EXPECTED: 0.5x to 1.5x for realistic strategy
    If ratio >> 2x, there may still be issues.
    """)
    
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()