#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward Validation: 8-STATE STRATEGY (Expanding Window)
=============================================================
Tests 8-state strategy with PROPER out-of-sample validation.

Key: Hit rates are calculated using ONLY historical data at each point.
This eliminates in-sample bias that inflated previous results.

Compares:
    1. 8-state (expanding window hit rates)
    2. 168h-only (baseline)
    3. Buy & Hold

Usage:
    python walkforward_8state.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

PAIR = 'XBTUSD'
DATA_START = '2017-01-01'
DATA_END = '2025-11-30'

# MA periods (best from grid search)
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2

# Hysteresis
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005

# Walk-forward settings
MIN_HISTORY_DAYS = 365  # Minimum days before trading (to build hit rate estimates)
TEST_MONTHS = 6         # Months per test window
STEP_MONTHS = 6         # Step between windows

# Strategy settings
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_PERM = 20  # Minimum samples to trust a permutation

# Trading costs
TOTAL_COST = 0.0015


# =============================================================================
# CORE FUNCTIONS
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
    current = 1  # Start UP
    
    for i in range(len(df)):
        if pd.isna(ma.iloc[i]):
            labels.iloc[i] = current
            continue
        
        price = close.iloc[i]
        
        if current == 1:  # UP
            if price < ma.iloc[i] * (1 - exit_buffer) and price < ma.iloc[i] * (1 - entry_buffer):
                current = 0  # DOWN
        else:  # DOWN
            if price > ma.iloc[i] * (1 + exit_buffer) and price > ma.iloc[i] * (1 + entry_buffer):
                current = 1  # UP
        
        labels.iloc[i] = current
    
    return labels


def generate_signals(df_24h: pd.DataFrame, df_72h: pd.DataFrame, 
                     df_168h: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all trend signals with CORRECTED alignment.
    Signals are shifted to avoid look-ahead bias.
    """
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    aligned = pd.DataFrame(index=df_24h.index)
    
    # Shift by 1 period before forward-fill (no look-ahead)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    
    return aligned.dropna().astype(int)


# =============================================================================
# EXPANDING WINDOW HIT RATES
# =============================================================================

def calculate_hit_rates_up_to(
    df_24h: pd.DataFrame,
    signals: pd.DataFrame,
    end_idx: int,
) -> Dict[Tuple[int, int, int], Dict]:
    """
    Calculate hit rates using data UP TO end_idx (exclusive).
    This ensures no look-ahead bias.
    """
    all_perms = list(product([0, 1], repeat=3))
    
    # Get historical subset
    hist_signals = signals.iloc[:end_idx]
    
    # Forward returns: return from T to T+1
    forward_returns = df_24h['close'].pct_change().shift(-1)
    hist_returns = forward_returns.loc[hist_signals.index]
    
    hit_rates = {}
    
    for perm in all_perms:
        mask = (
            (hist_signals['trend_24h'] == perm[0]) &
            (hist_signals['trend_72h'] == perm[1]) &
            (hist_signals['trend_168h'] == perm[2])
        )
        
        perm_returns = hist_returns[mask].dropna()
        
        n = len(perm_returns)
        if n > 0:
            n_wins = (perm_returns > 0).sum()
            hit_rate = n_wins / n
        else:
            hit_rate = 0.5
            n_wins = 0
        
        hit_rates[perm] = {
            'n': n,
            'n_wins': n_wins,
            'hit_rate': hit_rate,
            'sufficient': n >= MIN_SAMPLES_PER_PERM,
        }
    
    return hit_rates


# =============================================================================
# POSITION GENERATION
# =============================================================================

def get_position_8state(perm: Tuple[int, int, int], 
                        hit_rates: Dict[Tuple[int, int, int], Dict]) -> float:
    """Get position based on historical hit rate for this permutation."""
    data = hit_rates[perm]
    
    if not data['sufficient']:
        return 0.50  # Uncertain - half position
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00  # Favorable - full position
    else:
        return 0.00  # Unfavorable - no position


def get_position_168h_only(trend_168h: int) -> float:
    """Simple 168h rule: 100% if UP, 0% if DOWN."""
    return 1.0 if trend_168h == 1 else 0.0


# =============================================================================
# BACKTESTING
# =============================================================================

@dataclass
class WindowMetrics:
    """Metrics for a single walk-forward window."""
    window_num: int
    test_start: str
    test_end: str
    # 8-state metrics
    return_8state: float
    sharpe_8state: float
    max_dd_8state: float
    avg_pos_8state: float
    # 168h-only metrics
    return_168h: float
    sharpe_168h: float
    max_dd_168h: float
    avg_pos_168h: float
    # Buy & hold metrics
    return_bh: float
    sharpe_bh: float
    max_dd_bh: float
    # Derived
    alpha_8state: float
    alpha_168h: float
    # Hit rates at end of training
    final_hit_rates: Dict


def backtest_window_expanding(
    df_24h: pd.DataFrame,
    signals: pd.DataFrame,
    train_end_idx: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> WindowMetrics:
    """
    Backtest a single window using expanding-window hit rates.
    
    Hit rates are calculated using data up to train_end_idx.
    Then applied to test period [test_start, test_end).
    """
    
    # Calculate hit rates from training period only
    hit_rates = calculate_hit_rates_up_to(df_24h, signals, train_end_idx)
    
    # Filter to test period
    test_mask = (signals.index >= test_start) & (signals.index < test_end)
    test_signals = signals[test_mask]
    test_prices = df_24h.loc[test_signals.index, 'close']
    
    if len(test_signals) < 10:
        return None
    
    # Generate positions for both strategies
    positions_8state = pd.Series(index=test_signals.index, dtype=float)
    positions_168h = pd.Series(index=test_signals.index, dtype=float)
    
    for idx in test_signals.index:
        perm = (
            int(test_signals.loc[idx, 'trend_24h']),
            int(test_signals.loc[idx, 'trend_72h']),
            int(test_signals.loc[idx, 'trend_168h']),
        )
        
        positions_8state.loc[idx] = get_position_8state(perm, hit_rates)
        positions_168h.loc[idx] = get_position_168h_only(test_signals.loc[idx, 'trend_168h'])
    
    # Returns
    returns = test_prices.pct_change()
    valid_idx = returns.dropna().index
    returns = returns.loc[valid_idx]
    
    # Lagged positions (position at T-1 applied to return at T)
    lagged_8state = positions_8state.shift(1).loc[valid_idx].fillna(0)
    lagged_168h = positions_168h.shift(1).loc[valid_idx].fillna(0)
    
    # Transaction costs
    costs_8state = lagged_8state.diff().abs().fillna(0) * TOTAL_COST
    costs_168h = lagged_168h.diff().abs().fillna(0) * TOTAL_COST
    
    # Strategy returns
    strat_returns_8state = lagged_8state * returns - costs_8state
    strat_returns_168h = lagged_168h * returns - costs_168h
    bh_returns = returns
    
    # Equity curves
    equity_8state = (1 + strat_returns_8state).cumprod()
    equity_168h = (1 + strat_returns_168h).cumprod()
    equity_bh = (1 + bh_returns).cumprod()
    
    # Total returns
    total_return_8state = equity_8state.iloc[-1] - 1
    total_return_168h = equity_168h.iloc[-1] - 1
    total_return_bh = equity_bh.iloc[-1] - 1
    
    # Sharpe ratios (annualized)
    bars_per_year = 365
    sharpe_8state = (strat_returns_8state.mean() / strat_returns_8state.std() * np.sqrt(bars_per_year)
                     if strat_returns_8state.std() > 0 else 0)
    sharpe_168h = (strat_returns_168h.mean() / strat_returns_168h.std() * np.sqrt(bars_per_year)
                   if strat_returns_168h.std() > 0 else 0)
    sharpe_bh = (bh_returns.mean() / bh_returns.std() * np.sqrt(bars_per_year)
                 if bh_returns.std() > 0 else 0)
    
    # Max drawdowns
    def calc_max_dd(equity):
        rolling_max = equity.expanding().max()
        return ((equity - rolling_max) / rolling_max).min()
    
    max_dd_8state = calc_max_dd(equity_8state)
    max_dd_168h = calc_max_dd(equity_168h)
    max_dd_bh = calc_max_dd(equity_bh)
    
    return WindowMetrics(
        window_num=0,
        test_start=test_start.strftime('%Y-%m-%d'),
        test_end=test_end.strftime('%Y-%m-%d'),
        return_8state=total_return_8state,
        sharpe_8state=sharpe_8state,
        max_dd_8state=max_dd_8state,
        avg_pos_8state=lagged_8state.mean(),
        return_168h=total_return_168h,
        sharpe_168h=sharpe_168h,
        max_dd_168h=max_dd_168h,
        avg_pos_168h=lagged_168h.mean(),
        return_bh=total_return_bh,
        sharpe_bh=sharpe_bh,
        max_dd_bh=max_dd_bh,
        alpha_8state=total_return_8state - total_return_bh,
        alpha_168h=total_return_168h - total_return_bh,
        final_hit_rates=hit_rates,
    )


# =============================================================================
# WALK-FORWARD
# =============================================================================

def run_walk_forward(df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                     df_168h: pd.DataFrame) -> List[WindowMetrics]:
    """Run walk-forward validation with expanding hit rates."""
    
    # Generate signals
    signals = generate_signals(df_24h, df_72h, df_168h)
    
    # Generate windows
    min_date = signals.index.min()
    max_date = signals.index.max()
    
    windows = []
    train_end = min_date + pd.DateOffset(days=MIN_HISTORY_DAYS)
    test_start = train_end
    test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
    
    window_num = 1
    while test_end <= max_date:
        # Find train_end_idx in signals
        train_end_idx = signals.index.get_indexer([train_end], method='ffill')[0]
        if train_end_idx < 0:
            train_end_idx = 0
        
        windows.append((window_num, train_end_idx, test_start, test_end))
        
        # Step forward
        train_end = train_end + pd.DateOffset(months=STEP_MONTHS)
        test_start = test_start + pd.DateOffset(months=STEP_MONTHS)
        test_end = test_end + pd.DateOffset(months=STEP_MONTHS)
        window_num += 1
    
    print(f"\nRunning {len(windows)} walk-forward windows...")
    print(f"Min history: {MIN_HISTORY_DAYS} days, Test: {TEST_MONTHS} months, Step: {STEP_MONTHS} months")
    print("-" * 120)
    print(f"{'Win':>4} {'Period':<23} │ {'8-State':>9} {'Alpha':>8} │ {'168h':>9} {'Alpha':>8} │ {'B&H':>9} │ {'8St>168h':>9}")
    print("-" * 120)
    
    results = []
    
    for window_num, train_end_idx, test_start, test_end in windows:
        metrics = backtest_window_expanding(
            df_24h, signals, train_end_idx, test_start, test_end
        )
        
        if metrics is None:
            continue
        
        metrics.window_num = window_num
        results.append(metrics)
        
        # Print progress
        alpha_8_mark = "✓" if metrics.alpha_8state > 0 else "✗"
        alpha_168_mark = "✓" if metrics.alpha_168h > 0 else "✗"
        beats_168h = "✓" if metrics.return_8state > metrics.return_168h else "✗"
        
        print(f"{window_num:>4} {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} │ "
              f"{metrics.return_8state*100:>+8.1f}% {metrics.alpha_8state*100:>+7.1f}%{alpha_8_mark} │ "
              f"{metrics.return_168h*100:>+8.1f}% {metrics.alpha_168h*100:>+7.1f}%{alpha_168_mark} │ "
              f"{metrics.return_bh*100:>+8.1f}% │ "
              f"{(metrics.return_8state - metrics.return_168h)*100:>+8.1f}%{beats_168h}")
    
    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_summary(results: List[WindowMetrics]):
    """Print comprehensive summary."""
    
    n = len(results)
    
    # 8-state metrics
    alphas_8 = [r.alpha_8state for r in results]
    sharpes_8 = [r.sharpe_8state for r in results]
    
    # 168h metrics
    alphas_168 = [r.alpha_168h for r in results]
    sharpes_168 = [r.sharpe_168h for r in results]
    
    # B&H metrics
    sharpes_bh = [r.sharpe_bh for r in results]
    
    # Comparisons
    alpha_pos_8 = sum(1 for a in alphas_8 if a > 0)
    alpha_pos_168 = sum(1 for a in alphas_168 if a > 0)
    sharpe_wins_8 = sum(1 for s, b in zip(sharpes_8, sharpes_bh) if s > b)
    sharpe_wins_168 = sum(1 for s, b in zip(sharpes_168, sharpes_bh) if s > b)
    
    # 8-state vs 168h direct comparison
    beats_168h = sum(1 for r in results if r.return_8state > r.return_168h)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  WALK-FORWARD SUMMARY                                                       │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  Total Windows: {n:<59} │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                           8-STATE              168H-ONLY                    │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  Alpha+ Rate:          {alpha_pos_8:>3}/{n} ({100*alpha_pos_8/n:>5.1f}%)          {alpha_pos_168:>3}/{n} ({100*alpha_pos_168/n:>5.1f}%)             │
    │  Mean Alpha:           {np.mean(alphas_8)*100:>+7.1f}%              {np.mean(alphas_168)*100:>+7.1f}%                │
    │  Sharpe Beat B&H:      {sharpe_wins_8:>3}/{n} ({100*sharpe_wins_8/n:>5.1f}%)          {sharpe_wins_168:>3}/{n} ({100*sharpe_wins_168/n:>5.1f}%)             │
    │  Mean Sharpe:          {np.mean(sharpes_8):>7.2f}               {np.mean(sharpes_168):>7.2f}                 │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  8-STATE vs 168H-ONLY:                                                      │
    │    8-State Beats 168h: {beats_168h:>3}/{n} ({100*beats_168h/n:>5.1f}%)                                       │
    │    Mean Difference:    {np.mean([r.return_8state - r.return_168h for r in results])*100:>+7.1f}%                                        │
    └─────────────────────────────────────────────────────────────────────────────┘
    """)


def print_regime_analysis(results: List[WindowMetrics]):
    """Analyze by market regime."""
    
    print("\n" + "=" * 80)
    print("REGIME ANALYSIS")
    print("=" * 80)
    
    bull = [r for r in results if r.return_bh > 0.20]
    bear = [r for r in results if r.return_bh < -0.20]
    sideways = [r for r in results if -0.20 <= r.return_bh <= 0.20]
    
    for windows, name in [
        (bull, "BULL (B&H > +20%)"),
        (bear, "BEAR (B&H < -20%)"),
        (sideways, "SIDEWAYS (-20% to +20%)"),
    ]:
        if not windows:
            continue
        
        alphas_8 = [w.alpha_8state for w in windows]
        alphas_168 = [w.alpha_168h for w in windows]
        beats_168h = sum(1 for w in windows if w.return_8state > w.return_168h)
        
        alpha_pos_8 = sum(1 for a in alphas_8 if a > 0)
        alpha_pos_168 = sum(1 for a in alphas_168 if a > 0)
        
        print(f"""
    {name}:
      Windows:              {len(windows)}
      
      8-STATE:
        Alpha+ Rate:        {alpha_pos_8}/{len(windows)} ({100*alpha_pos_8/len(windows):.0f}%)
        Mean Alpha:         {np.mean(alphas_8)*100:+.1f}%
        
      168H-ONLY:
        Alpha+ Rate:        {alpha_pos_168}/{len(windows)} ({100*alpha_pos_168/len(windows):.0f}%)
        Mean Alpha:         {np.mean(alphas_168)*100:+.1f}%
        
      8-STATE vs 168H:
        Beats 168h:         {beats_168h}/{len(windows)} ({100*beats_168h/len(windows):.0f}%)
        Mean Diff:          {np.mean([w.return_8state - w.return_168h for w in windows])*100:+.1f}%
""")


def print_hit_rate_evolution(results: List[WindowMetrics]):
    """Show how hit rates evolved over time."""
    
    print("\n" + "=" * 80)
    print("HIT RATE EVOLUTION (End of each training period)")
    print("=" * 80)
    
    names = {0: 'D', 1: 'U'}
    all_perms = list(product([0, 1], repeat=3))
    
    # Print header
    print(f"\n    {'Perm':<8}", end="")
    for r in results[:6]:  # First 6 windows
        print(f" {'W'+str(r.window_num):>7}", end="")
    print(f" {'Mean':>7} {'Std':>6}")
    print("    " + "-" * 70)
    
    for perm in sorted(all_perms, key=lambda p: -results[-1].final_hit_rates[p]['hit_rate']):
        perm_name = f"{names[perm[0]]}/{names[perm[1]]}/{names[perm[2]]}"
        
        hit_rates = [r.final_hit_rates[perm]['hit_rate'] for r in results]
        
        print(f"    {perm_name:<8}", end="")
        for hr in hit_rates[:6]:
            print(f" {hr*100:>6.1f}%", end="")
        print(f" {np.mean(hit_rates)*100:>6.1f}% {np.std(hit_rates)*100:>5.1f}%")


def print_verdict(results: List[WindowMetrics]):
    """Print final verdict comparing 8-state to 168h-only."""
    
    n = len(results)
    
    beats_168h = sum(1 for r in results if r.return_8state > r.return_168h)
    beats_168h_rate = beats_168h / n
    
    mean_diff = np.mean([r.return_8state - r.return_168h for r in results])
    
    alpha_pos_8 = sum(1 for r in results if r.alpha_8state > 0)
    alpha_pos_168 = sum(1 for r in results if r.alpha_168h > 0)
    
    # Determine verdict
    if beats_168h_rate >= 0.6 and mean_diff > 0.05:
        verdict_8state = "ADDS VALUE - 8-state consistently outperforms 168h-only"
    elif beats_168h_rate >= 0.5:
        verdict_8state = "MARGINAL - 8-state slightly better but not consistent"
    else:
        verdict_8state = "NO EDGE - 8-state does not improve on 168h-only"
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║  VERDICT                                                                      ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  8-STATE vs B&H:                                                              ║
    ║    Alpha Positive:     {alpha_pos_8:>3}/{n} ({100*alpha_pos_8/n:>5.1f}%)                                        ║
    ║    Mean Alpha:         {np.mean([r.alpha_8state for r in results])*100:>+7.1f}%                                             ║
    ║                                                                               ║
    ║  168H-ONLY vs B&H:                                                            ║
    ║    Alpha Positive:     {alpha_pos_168:>3}/{n} ({100*alpha_pos_168/n:>5.1f}%)                                        ║
    ║    Mean Alpha:         {np.mean([r.alpha_168h for r in results])*100:>+7.1f}%                                             ║
    ║                                                                               ║
    ║  8-STATE vs 168H-ONLY:                                                        ║
    ║    8-State Wins:       {beats_168h:>3}/{n} ({100*beats_168h/n:>5.1f}%)                                        ║
    ║    Mean Improvement:   {mean_diff*100:>+7.1f}%                                             ║
    ║                                                                               ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║  {verdict_8state:<77} ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("WALK-FORWARD VALIDATION: 8-STATE vs 168H-ONLY (Expanding Window)")
    print("=" * 100)
    
    print(f"""
    CONFIGURATION:
        Pair:           {PAIR}
        Data:           {DATA_START} to {DATA_END}
        
    8-STATE STRATEGY:
        MA Periods:     24h={MA_PERIOD_24H}, 72h={MA_PERIOD_72H}, 168h={MA_PERIOD_168H}
        Hysteresis:     Entry={ENTRY_BUFFER*100:.1f}%, Exit={EXIT_BUFFER*100:.1f}%
        Hit Rate Threshold: {HIT_RATE_THRESHOLD*100:.0f}%
        Min Samples:    {MIN_SAMPLES_PER_PERM} per permutation
        
    WALK-FORWARD:
        Min History:    {MIN_HISTORY_DAYS} days (before first trade)
        Test Window:    {TEST_MONTHS} months
        Step:           {STEP_MONTHS} months
        
    KEY DIFFERENCE FROM PREVIOUS TESTS:
        Hit rates calculated using EXPANDING WINDOW - only historical data.
        No look-ahead bias in hit rate calculation.
    """)
    
    # Load data
    print("-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    db = Database()
    df_1h = db.get_ohlcv(PAIR, start=DATA_START, end=DATA_END)
    print(f"Loaded {len(df_1h):,} 1h bars")
    print(f"Date range: {df_1h.index.min().date()} to {df_1h.index.max().date()}")
    
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    print(f"Resampled: 24h={len(df_24h)}, 72h={len(df_72h)}, 168h={len(df_168h)} bars")
    
    # Walk-forward
    print("\n" + "=" * 100)
    print("WALK-FORWARD VALIDATION")
    print("=" * 100)
    
    results = run_walk_forward(df_24h, df_72h, df_168h)
    
    # Results
    print_summary(results)
    print_regime_analysis(results)
    print_hit_rate_evolution(results)
    print_verdict(results)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()