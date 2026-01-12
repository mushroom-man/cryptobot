#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed Walk-Forward Analysis: Deployment Tier Pairs
======================================================
Detailed window-by-window analysis for pairs recommended for deployment:
    - XLMUSD (STRONG)
    - ZECUSD (STRONG)
    - ETCUSD (MODERATE)
    - ETHUSD (MODERATE)
    - XMRUSD (MODERATE)
    - ADAUSD (MODERATE)

Using LOCKED XBTUSD parameters.

Usage:
    python walkforward_deploy_tier.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from itertools import product
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# LOCKED PARAMETERS (FROM XBTUSD OPTIMIZATION)
# =============================================================================

MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005

# Walk-forward settings
MIN_HISTORY_DAYS = 365
TEST_MONTHS = 6
STEP_MONTHS = 6
MIN_SAMPLES_PER_PERM = 20
HIT_RATE_THRESHOLD = 0.50

# Trading costs
TOTAL_COST = 0.0015

# Deployment tier pairs
DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']


# =============================================================================
# CORE FUNCTIONS
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


def generate_signals(df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                     df_168h: pd.DataFrame) -> pd.DataFrame:
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    
    return aligned.dropna().astype(int)


def calculate_hit_rates_up_to(df_24h: pd.DataFrame, signals: pd.DataFrame,
                               end_idx: int) -> Dict[Tuple[int, int, int], Dict]:
    all_perms = list(product([0, 1], repeat=3))
    
    hist_signals = signals.iloc[:end_idx]
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
            hit_rate, n_wins = 0.5, 0
        
        hit_rates[perm] = {
            'n': n, 'n_wins': n_wins, 'hit_rate': hit_rate,
            'sufficient': n >= MIN_SAMPLES_PER_PERM,
        }
    
    return hit_rates


def get_position_8state(perm: Tuple[int, int, int],
                        hit_rates: Dict[Tuple[int, int, int], Dict]) -> float:
    data = hit_rates[perm]
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.00


def get_position_168h_only(trend_168h: int) -> float:
    return 1.0 if trend_168h == 1 else 0.0


# =============================================================================
# BACKTEST WINDOW
# =============================================================================

@dataclass
class WindowResult:
    window_num: int
    test_start: str
    test_end: str
    return_8state: float
    return_168h: float
    return_bh: float
    alpha_8state: float
    alpha_168h: float
    sharpe_8state: float
    sharpe_168h: float
    sharpe_bh: float
    max_dd_8state: float
    max_dd_168h: float
    max_dd_bh: float
    avg_pos_8state: float
    avg_pos_168h: float


def backtest_window(df_24h: pd.DataFrame, signals: pd.DataFrame,
                    hit_rates: Dict, test_start: pd.Timestamp,
                    test_end: pd.Timestamp) -> Optional[WindowResult]:
    
    test_mask = (signals.index >= test_start) & (signals.index < test_end)
    test_signals = signals[test_mask]
    test_prices = df_24h.loc[test_signals.index, 'close']
    
    if len(test_signals) < 10:
        return None
    
    # Generate positions
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
    
    lagged_8state = positions_8state.shift(1).loc[valid_idx].fillna(0)
    lagged_168h = positions_168h.shift(1).loc[valid_idx].fillna(0)
    
    costs_8state = lagged_8state.diff().abs().fillna(0) * TOTAL_COST
    costs_168h = lagged_168h.diff().abs().fillna(0) * TOTAL_COST
    
    strat_returns_8state = lagged_8state * returns - costs_8state
    strat_returns_168h = lagged_168h * returns - costs_168h
    bh_returns = returns
    
    # Equity
    equity_8state = (1 + strat_returns_8state).cumprod()
    equity_168h = (1 + strat_returns_168h).cumprod()
    equity_bh = (1 + bh_returns).cumprod()
    
    total_8state = equity_8state.iloc[-1] - 1
    total_168h = equity_168h.iloc[-1] - 1
    total_bh = equity_bh.iloc[-1] - 1
    
    # Sharpe
    bars_per_year = 365
    sharpe_8state = (strat_returns_8state.mean() / strat_returns_8state.std() * np.sqrt(bars_per_year)
                     if strat_returns_8state.std() > 0 else 0)
    sharpe_168h = (strat_returns_168h.mean() / strat_returns_168h.std() * np.sqrt(bars_per_year)
                   if strat_returns_168h.std() > 0 else 0)
    sharpe_bh = (bh_returns.mean() / bh_returns.std() * np.sqrt(bars_per_year)
                 if bh_returns.std() > 0 else 0)
    
    # Max drawdown
    def calc_max_dd(equity):
        rolling_max = equity.expanding().max()
        return ((equity - rolling_max) / rolling_max).min()
    
    max_dd_8state = calc_max_dd(equity_8state)
    max_dd_168h = calc_max_dd(equity_168h)
    max_dd_bh = calc_max_dd(equity_bh)
    
    return WindowResult(
        window_num=0,
        test_start=test_start.strftime('%Y-%m-%d'),
        test_end=test_end.strftime('%Y-%m-%d'),
        return_8state=total_8state,
        return_168h=total_168h,
        return_bh=total_bh,
        alpha_8state=total_8state - total_bh,
        alpha_168h=total_168h - total_bh,
        sharpe_8state=sharpe_8state,
        sharpe_168h=sharpe_168h,
        sharpe_bh=sharpe_bh,
        max_dd_8state=max_dd_8state,
        max_dd_168h=max_dd_168h,
        max_dd_bh=max_dd_bh,
        avg_pos_8state=lagged_8state.mean(),
        avg_pos_168h=lagged_168h.mean(),
    )


# =============================================================================
# WALK-FORWARD FOR SINGLE PAIR
# =============================================================================

def run_walk_forward(pair: str, df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                     df_168h: pd.DataFrame) -> List[WindowResult]:
    
    signals = generate_signals(df_24h, df_72h, df_168h)
    
    min_date = signals.index.min()
    max_date = signals.index.max()
    
    windows = []
    train_end = min_date + pd.DateOffset(days=MIN_HISTORY_DAYS)
    test_start = train_end
    test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
    
    window_num = 1
    while test_end <= max_date:
        train_end_idx = signals.index.get_indexer([train_end], method='ffill')[0]
        if train_end_idx < 0:
            train_end_idx = 0
        windows.append((window_num, train_end_idx, test_start, test_end))
        
        train_end += pd.DateOffset(months=STEP_MONTHS)
        test_start += pd.DateOffset(months=STEP_MONTHS)
        test_end += pd.DateOffset(months=STEP_MONTHS)
        window_num += 1
    
    results = []
    for window_num, train_end_idx, test_start, test_end in windows:
        hit_rates = calculate_hit_rates_up_to(df_24h, signals, train_end_idx)
        window_result = backtest_window(df_24h, signals, hit_rates, test_start, test_end)
        
        if window_result:
            window_result.window_num = window_num
            results.append(window_result)
    
    return results


def print_pair_results(pair: str, results: List[WindowResult]):
    """Print detailed results for a single pair."""
    
    n = len(results)
    
    # Header
    print(f"\n{'═' * 120}")
    print(f"  {pair} - DETAILED WALK-FORWARD ANALYSIS ({n} windows)")
    print(f"{'═' * 120}")
    
    # Window-by-window table
    print(f"\n{'Win':>4} {'Period':<23} │ {'8-State':>9} {'Alpha':>8} │ {'168h':>9} {'Alpha':>8} │ {'B&H':>9} │ {'8St>168h':>9} {'8St>B&H':>8}")
    print("-" * 120)
    
    for r in results:
        beats_168h = "✓" if r.return_8state > r.return_168h else ""
        beats_bh = "✓" if r.alpha_8state > 0 else ""
        
        print(f"{r.window_num:>4} {r.test_start} to {r.test_end} │ "
              f"{r.return_8state*100:>+8.1f}% {r.alpha_8state*100:>+7.1f}% │ "
              f"{r.return_168h*100:>+8.1f}% {r.alpha_168h*100:>+7.1f}% │ "
              f"{r.return_bh*100:>+8.1f}% │ "
              f"{(r.return_8state - r.return_168h)*100:>+8.1f}%{beats_168h} "
              f"{r.alpha_8state*100:>+7.1f}%{beats_bh}")
    
    # Summary statistics
    alphas_8 = [r.alpha_8state for r in results]
    alphas_168 = [r.alpha_168h for r in results]
    sharpes_8 = [r.sharpe_8state for r in results]
    sharpes_168 = [r.sharpe_168h for r in results]
    sharpes_bh = [r.sharpe_bh for r in results]
    max_dds_8 = [r.max_dd_8state for r in results]
    max_dds_168 = [r.max_dd_168h for r in results]
    max_dds_bh = [r.max_dd_bh for r in results]
    
    alpha_pos_8 = sum(1 for a in alphas_8 if a > 0)
    alpha_pos_168 = sum(1 for a in alphas_168 if a > 0)
    sharpe_wins_8 = sum(1 for s, b in zip(sharpes_8, sharpes_bh) if s > b)
    sharpe_wins_168 = sum(1 for s, b in zip(sharpes_168, sharpes_bh) if s > b)
    dd_wins_8 = sum(1 for d, b in zip(max_dds_8, max_dds_bh) if d > b)
    beats_168h = sum(1 for r in results if r.return_8state > r.return_168h)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  {pair} SUMMARY                                                               │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │                              8-STATE              168H-ONLY           B&H       │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Alpha+ Rate:            {alpha_pos_8:>3}/{n} ({100*alpha_pos_8/n:>5.1f}%)       {alpha_pos_168:>3}/{n} ({100*alpha_pos_168/n:>5.1f}%)           -       │
    │  Mean Alpha:             {np.mean(alphas_8)*100:>+7.1f}%           {np.mean(alphas_168)*100:>+7.1f}%              -       │
    │  Sharpe Beat B&H:        {sharpe_wins_8:>3}/{n} ({100*sharpe_wins_8/n:>5.1f}%)       {sharpe_wins_168:>3}/{n} ({100*sharpe_wins_168/n:>5.1f}%)           -       │
    │  Mean Sharpe:            {np.mean(sharpes_8):>7.2f}             {np.mean(sharpes_168):>7.2f}           {np.mean(sharpes_bh):>7.2f}   │
    │  Better Drawdown:        {dd_wins_8:>3}/{n} ({100*dd_wins_8/n:>5.1f}%)           -                   -       │
    │  Mean Max DD:            {np.mean(max_dds_8)*100:>7.1f}%           {np.mean(max_dds_168)*100:>7.1f}%         {np.mean(max_dds_bh)*100:>7.1f}%   │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  8-STATE vs 168H-ONLY:                                                          │
    │    Beats 168h:           {beats_168h:>3}/{n} ({100*beats_168h/n:>5.1f}%)                                            │
    │    Mean Improvement:     {np.mean([r.return_8state - r.return_168h for r in results])*100:>+7.1f}%                                            │
    └─────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Regime analysis
    bull = [r for r in results if r.return_bh > 0.20]
    bear = [r for r in results if r.return_bh < -0.20]
    sideways = [r for r in results if -0.20 <= r.return_bh <= 0.20]
    
    print("    REGIME ANALYSIS:")
    print("    " + "-" * 70)
    
    for windows, name in [(bull, "BULL"), (bear, "BEAR"), (sideways, "SIDEWAYS")]:
        if not windows:
            print(f"    {name:<12}: No windows")
            continue
        
        a8 = [w.alpha_8state for w in windows]
        a168 = [w.alpha_168h for w in windows]
        beats = sum(1 for w in windows if w.return_8state > w.return_168h)
        
        print(f"    {name:<12}: {len(windows):>2} windows | "
              f"8St α+ {sum(1 for a in a8 if a > 0)}/{len(windows)} ({100*sum(1 for a in a8 if a > 0)/len(windows):>4.0f}%) | "
              f"Mean α {np.mean(a8)*100:>+6.1f}% | "
              f"Beats 168h {beats}/{len(windows)} ({100*beats/len(windows):>4.0f}%)")
    
    return {
        'pair': pair,
        'n_windows': n,
        'alpha_pos_rate_8state': alpha_pos_8 / n,
        'mean_alpha_8state': np.mean(alphas_8),
        'sharpe_win_rate_8state': sharpe_wins_8 / n,
        'mean_sharpe_8state': np.mean(sharpes_8),
        'beats_168h_rate': beats_168h / n,
        'mean_diff_vs_168h': np.mean([r.return_8state - r.return_168h for r in results]),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 120)
    print("DETAILED WALK-FORWARD ANALYSIS: DEPLOYMENT TIER PAIRS")
    print("=" * 120)
    
    print(f"""
    PAIRS ANALYZED (Recommended for Deployment):
        {', '.join(DEPLOY_PAIRS)}
    
    LOCKED PARAMETERS (from XBTUSD optimization):
        MA Periods:     24h={MA_PERIOD_24H}, 72h={MA_PERIOD_72H}, 168h={MA_PERIOD_168H}
        Hysteresis:     Entry={ENTRY_BUFFER*100:.1f}%, Exit={EXIT_BUFFER*100:.1f}%
        
    WALK-FORWARD:
        Min History:    {MIN_HISTORY_DAYS} days
        Test Window:    {TEST_MONTHS} months
        Step:           {STEP_MONTHS} months
    """)
    
    db = Database()
    all_summaries = []
    
    for pair in DEPLOY_PAIRS:
        print(f"\n{'─' * 60}")
        print(f"Loading {pair}...")
        
        try:
            df_1h = db.get_ohlcv(pair)
            print(f"  {len(df_1h):,} 1h bars: {df_1h.index.min().date()} to {df_1h.index.max().date()}")
            
            df_24h = resample_ohlcv(df_1h, '24h')
            df_72h = resample_ohlcv(df_1h, '72h')
            df_168h = resample_ohlcv(df_1h, '168h')
            
            results = run_walk_forward(pair, df_24h, df_72h, df_168h)
            
            if len(results) < 2:
                print(f"  ⚠ Insufficient windows. Skipping.")
                continue
            
            summary = print_pair_results(pair, results)
            all_summaries.append(summary)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Final summary
    print("\n" + "=" * 120)
    print("DEPLOYMENT TIER SUMMARY")
    print("=" * 120)
    
    print(f"\n{'Pair':<10} {'Windows':>8} {'8St α+':>10} {'Mean α':>10} {'Sharpe Win':>12} {'Beats 168h':>12}")
    print("-" * 70)
    
    for s in all_summaries:
        print(f"{s['pair']:<10} {s['n_windows']:>8} {s['alpha_pos_rate_8state']*100:>9.1f}% "
              f"{s['mean_alpha_8state']*100:>+9.1f}% {s['sharpe_win_rate_8state']*100:>11.1f}% "
              f"{s['beats_168h_rate']*100:>11.1f}%")
    
    # Aggregate
    avg_alpha_pos = np.mean([s['alpha_pos_rate_8state'] for s in all_summaries])
    avg_alpha = np.mean([s['mean_alpha_8state'] for s in all_summaries])
    avg_beats_168h = np.mean([s['beats_168h_rate'] for s in all_summaries])
    
    print("-" * 70)
    print(f"{'AVERAGE':<10} {'-':>8} {avg_alpha_pos*100:>9.1f}% {avg_alpha*100:>+9.1f}% "
          f"{'-':>12} {avg_beats_168h*100:>11.1f}%")
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  DEPLOYMENT RECOMMENDATION                                                                            ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  Average Alpha+ Rate:     {avg_alpha_pos*100:>5.1f}%                                                                     ║
    ║  Average Mean Alpha:      {avg_alpha*100:>+5.1f}%                                                                     ║
    ║  Average Beats 168h:      {avg_beats_168h*100:>5.1f}%                                                                     ║
    ║                                                                                                       ║
    ║  VERDICT: {'READY FOR DEPLOYMENT' if avg_alpha_pos >= 0.5 else 'NEEDS MORE ANALYSIS':<84} ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n" + "=" * 120)
    print("ANALYSIS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
    

