#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Pair 8-State Strategy Validation (Option A)
==================================================
Applies XBTUSD-optimized parameters to ALL pairs in the database.

This is a TRUE out-of-sample test:
- Parameters were fit ONLY to XBTUSD
- If strategy works on other pairs, it's robust
- If it fails, parameters are Bitcoin-specific

LOCKED PARAMETERS (from XBTUSD optimization):
    MA Periods:     24h=24, 72h=8, 168h=2
    Hysteresis:     Entry=2.0%, Exit=0.5%

Usage:
    python multi_pair_validation.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
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

# Minimum data requirements
MIN_BARS_24H = 730  # At least 2 years of daily data


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
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
    """Generate signals with LOCKED parameters."""
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
    """Calculate hit rates using data UP TO end_idx (expanding window)."""
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
    """Get position based on historical hit rate."""
    data = hit_rates[perm]
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.00


def get_position_168h_only(trend_168h: int) -> float:
    """Simple 168h rule."""
    return 1.0 if trend_168h == 1 else 0.0


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
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


@dataclass
class PairResult:
    """Aggregate results for a single pair."""
    pair: str
    data_start: str
    data_end: str
    n_bars: int
    n_windows: int
    # 8-state metrics
    alpha_pos_rate_8state: float
    mean_alpha_8state: float
    sharpe_win_rate_8state: float
    mean_sharpe_8state: float
    # 168h metrics
    alpha_pos_rate_168h: float
    mean_alpha_168h: float
    sharpe_win_rate_168h: float
    mean_sharpe_168h: float
    # Comparison
    beats_168h_rate: float
    mean_diff_vs_168h: float
    # B&H reference
    mean_sharpe_bh: float
    total_bh_return: float
    # Verdict
    verdict: str


def backtest_window(df_24h: pd.DataFrame, signals: pd.DataFrame,
                    hit_rates: Dict, test_start: pd.Timestamp,
                    test_end: pd.Timestamp) -> Optional[WindowResult]:
    """Backtest a single window."""
    
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
    )


def run_walk_forward_for_pair(pair: str, df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                               df_168h: pd.DataFrame) -> Optional[PairResult]:
    """Run walk-forward validation for a single pair."""
    
    # Generate signals
    signals = generate_signals(df_24h, df_72h, df_168h)
    
    if len(signals) < MIN_BARS_24H:
        return None
    
    # Generate windows
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
    
    if len(windows) < 2:
        return None
    
    # Run walk-forward
    results = []
    for window_num, train_end_idx, test_start, test_end in windows:
        hit_rates = calculate_hit_rates_up_to(df_24h, signals, train_end_idx)
        window_result = backtest_window(df_24h, signals, hit_rates, test_start, test_end)
        
        if window_result:
            window_result.window_num = window_num
            results.append(window_result)
    
    if len(results) < 2:
        return None
    
    # Aggregate metrics
    n = len(results)
    
    alphas_8 = [r.alpha_8state for r in results]
    alphas_168 = [r.alpha_168h for r in results]
    sharpes_8 = [r.sharpe_8state for r in results]
    sharpes_168 = [r.sharpe_168h for r in results]
    sharpes_bh = [r.sharpe_bh for r in results]
    
    alpha_pos_8 = sum(1 for a in alphas_8 if a > 0)
    alpha_pos_168 = sum(1 for a in alphas_168 if a > 0)
    sharpe_wins_8 = sum(1 for s, b in zip(sharpes_8, sharpes_bh) if s > b)
    sharpe_wins_168 = sum(1 for s, b in zip(sharpes_168, sharpes_bh) if s > b)
    beats_168h = sum(1 for r in results if r.return_8state > r.return_168h)
    
    # Total B&H return
    total_bh = (1 + pd.Series([r.return_bh for r in results])).prod() - 1
    
    # Verdict
    alpha_rate_8 = alpha_pos_8 / n
    if alpha_rate_8 >= 0.6 and beats_168h / n >= 0.5:
        verdict = "STRONG"
    elif alpha_rate_8 >= 0.5:
        verdict = "MODERATE"
    elif alpha_rate_8 >= 0.4:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    
    return PairResult(
        pair=pair,
        data_start=df_24h.index.min().strftime('%Y-%m-%d'),
        data_end=df_24h.index.max().strftime('%Y-%m-%d'),
        n_bars=len(df_24h),
        n_windows=n,
        alpha_pos_rate_8state=alpha_pos_8 / n,
        mean_alpha_8state=np.mean(alphas_8),
        sharpe_win_rate_8state=sharpe_wins_8 / n,
        mean_sharpe_8state=np.mean(sharpes_8),
        alpha_pos_rate_168h=alpha_pos_168 / n,
        mean_alpha_168h=np.mean(alphas_168),
        sharpe_win_rate_168h=sharpe_wins_168 / n,
        mean_sharpe_168h=np.mean(sharpes_168),
        beats_168h_rate=beats_168h / n,
        mean_diff_vs_168h=np.mean([r.return_8state - r.return_168h for r in results]),
        mean_sharpe_bh=np.mean(sharpes_bh),
        total_bh_return=total_bh,
        verdict=verdict,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("MULTI-PAIR 8-STATE VALIDATION (XBTUSD PARAMETERS)")
    print("=" * 100)
    
    print(f"""
    LOCKED PARAMETERS (from XBTUSD optimization):
        MA Periods:     24h={MA_PERIOD_24H}, 72h={MA_PERIOD_72H}, 168h={MA_PERIOD_168H}
        Hysteresis:     Entry={ENTRY_BUFFER*100:.1f}%, Exit={EXIT_BUFFER*100:.1f}%
        
    WALK-FORWARD:
        Min History:    {MIN_HISTORY_DAYS} days
        Test Window:    {TEST_MONTHS} months
        Step:           {STEP_MONTHS} months
        
    This is a TRUE out-of-sample test: parameters were fit ONLY to XBTUSD.
    """)
    
    # =========================================================================
    # LOAD DATABASE
    # =========================================================================
    
    print("-" * 60)
    print("LOADING DATABASE")
    print("-" * 60)
    
    db = Database()
    
    # Get all pairs and their data ranges
    summary = db.summary()
    print(f"\nFound {len(summary)} pair(s) in database:\n")
    print(summary.to_string(index=False))
    
    pairs = summary['pair'].unique().tolist()
    
    # =========================================================================
    # PROCESS EACH PAIR
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("WALK-FORWARD VALIDATION BY PAIR")
    print("=" * 100)
    
    all_results = []
    
    for pair in pairs:
        print(f"\n{'─' * 80}")
        print(f"PROCESSING: {pair}")
        print(f"{'─' * 80}")
        
        # Load data
        try:
            df_1h = db.get_ohlcv(pair)
            
            if len(df_1h) < MIN_BARS_24H * 24:  # Need at least 2 years of hourly data
                print(f"  ⚠ Insufficient data ({len(df_1h)} hours). Skipping.")
                continue
            
            print(f"  Loaded {len(df_1h):,} 1h bars: {df_1h.index.min().date()} to {df_1h.index.max().date()}")
            
            # Resample
            df_24h = resample_ohlcv(df_1h, '24h')
            df_72h = resample_ohlcv(df_1h, '72h')
            df_168h = resample_ohlcv(df_1h, '168h')
            
            print(f"  Resampled: 24h={len(df_24h)}, 72h={len(df_72h)}, 168h={len(df_168h)} bars")
            
            if len(df_24h) < MIN_BARS_24H:
                print(f"  ⚠ Insufficient 24h bars ({len(df_24h)}). Skipping.")
                continue
            
            # Run walk-forward
            result = run_walk_forward_for_pair(pair, df_24h, df_72h, df_168h)
            
            if result is None:
                print(f"  ⚠ Insufficient windows. Skipping.")
                continue
            
            all_results.append(result)
            
            # Print summary for this pair
            print(f"\n  RESULTS ({result.n_windows} windows):")
            print(f"    8-STATE:    Alpha+ {result.alpha_pos_rate_8state*100:>5.1f}% | "
                  f"Mean α {result.mean_alpha_8state*100:>+6.1f}% | "
                  f"Sharpe {result.mean_sharpe_8state:>5.2f}")
            print(f"    168H-ONLY:  Alpha+ {result.alpha_pos_rate_168h*100:>5.1f}% | "
                  f"Mean α {result.mean_alpha_168h*100:>+6.1f}% | "
                  f"Sharpe {result.mean_sharpe_168h:>5.2f}")
            print(f"    B&H:        Sharpe {result.mean_sharpe_bh:>5.2f} | "
                  f"Total Return {result.total_bh_return*100:>+8.1f}%")
            print(f"    8-State beats 168h: {result.beats_168h_rate*100:.1f}%")
            print(f"    VERDICT: {result.verdict}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    
    if not all_results:
        print("\nNo pairs had sufficient data for analysis.")
        return
    
    # Create summary DataFrame
    df_results = pd.DataFrame([asdict(r) for r in all_results])
    
    # Print summary table
    print(f"\n{'Pair':<12} {'Windows':>8} {'8St α+':>8} {'168h α+':>8} {'8St>168h':>9} {'Verdict':<10}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r.pair:<12} {r.n_windows:>8} {r.alpha_pos_rate_8state*100:>7.1f}% "
              f"{r.alpha_pos_rate_168h*100:>7.1f}% {r.beats_168h_rate*100:>8.1f}% "
              f"{r.verdict:<10}")
    
    # Aggregate stats
    print("\n" + "-" * 70)
    
    n_pairs = len(all_results)
    strong = sum(1 for r in all_results if r.verdict == "STRONG")
    moderate = sum(1 for r in all_results if r.verdict == "MODERATE")
    weak = sum(1 for r in all_results if r.verdict == "WEAK")
    fail = sum(1 for r in all_results if r.verdict == "FAIL")
    
    avg_alpha_8 = np.mean([r.mean_alpha_8state for r in all_results])
    avg_alpha_168 = np.mean([r.mean_alpha_168h for r in all_results])
    avg_beats_168h = np.mean([r.beats_168h_rate for r in all_results])
    
    print(f"""
    AGGREGATE RESULTS ({n_pairs} pairs):
    
    VERDICT DISTRIBUTION:
        STRONG:     {strong:>3} ({100*strong/n_pairs:>5.1f}%)
        MODERATE:   {moderate:>3} ({100*moderate/n_pairs:>5.1f}%)
        WEAK:       {weak:>3} ({100*weak/n_pairs:>5.1f}%)
        FAIL:       {fail:>3} ({100*fail/n_pairs:>5.1f}%)
    
    AVERAGE METRICS:
        8-State Mean Alpha:     {avg_alpha_8*100:>+6.1f}%
        168h-Only Mean Alpha:   {avg_alpha_168*100:>+6.1f}%
        8-State Beats 168h:     {avg_beats_168h*100:>5.1f}%
    """)
    
    # Overall verdict
    if strong + moderate >= n_pairs * 0.6:
        overall = "STRATEGY IS ROBUST - Works across multiple pairs"
    elif strong + moderate >= n_pairs * 0.4:
        overall = "STRATEGY IS PARTIALLY ROBUST - Works on some pairs"
    else:
        overall = "STRATEGY IS NOT ROBUST - May be Bitcoin-specific"
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║  OVERALL VERDICT: {overall:<58} ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Save results to CSV
    output_file = "/mnt/user-data/outputs/multi_pair_8state_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
