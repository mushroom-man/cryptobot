#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32-State Hit Rate Calculator
=============================
Calculates hit rates for all 32 states:
- 8 price states (price vs MA24/MA72/MA168)
- 4 MA alignment states (MA72 vs MA24, MA168 vs MA24)

8 × 4 = 32 total states

Usage:
    python calculate_32state_hit_rates.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005
MIN_SAMPLES = 20  # Minimum samples per state


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
        ma_val = ma.iloc[i]
        
        if current == 1:
            if price < ma_val * (1 - exit_buffer) and price < ma_val * (1 - entry_buffer):
                current = 0
        else:
            if price > ma_val * (1 + exit_buffer) and price > ma_val * (1 + entry_buffer):
                current = 1
        
        labels.iloc[i] = current
    
    return labels


def generate_signals(df_24h, df_72h, df_168h):
    """Generate all signals for 32-state system."""
    
    # Price vs MA signals (8 states)
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    # MA values for alignment
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Align to 24h index
    aligned = pd.DataFrame(index=df_24h.index)
    
    # Price vs MA (shifted to prevent look-ahead)
    aligned['price_vs_ma24'] = trend_24h.shift(1)
    aligned['price_vs_ma72'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['price_vs_ma168'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    
    # MA alignment (shifted to prevent look-ahead)
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


def get_alignment_label(ma72_above, ma168_above):
    """Get human-readable alignment label."""
    if ma72_above == 0 and ma168_above == 0:
        return "BULLISH"   # MA24 > MA72 and MA24 > MA168
    elif ma72_above == 1 and ma168_above == 1:
        return "BEARISH"   # MA24 < MA72 and MA24 < MA168
    elif ma72_above == 0 and ma168_above == 1:
        return "MIXED-1"   # MA24 > MA72 but MA24 < MA168
    else:
        return "MIXED-2"   # MA24 < MA72 but MA24 > MA168


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("32-STATE HIT RATE ANALYSIS")
    print("=" * 100)
    
    print("""
    32 STATES = 8 Price States × 4 MA Alignment States
    
    PRICE STATES (Price vs MA):
    ───────────────────────────
    U = Price ABOVE MA (Uptrend)
    D = Price BELOW MA (Downtrend)
    
    Format: Price vs MA24 / Price vs MA72 / Price vs MA168
    Example: U/D/D = Price above MA24, below MA72, below MA168
    
    MA ALIGNMENT STATES:
    ────────────────────
    BULLISH  = MA24 > MA72 AND MA24 > MA168  (short-term leading)
    BEARISH  = MA24 < MA72 AND MA24 < MA168  (short-term lagging)
    MIXED-1  = MA24 > MA72 but MA24 < MA168
    MIXED-2  = MA24 < MA72 but MA24 > MA168
    """)
    
    db = Database()
    
    # Aggregate hit rates across all pairs
    all_hit_rates = {}
    total_samples = 0
    
    print("\nLoading data...")
    for pair in DEPLOY_PAIRS:
        print(f"  {pair}...", end=" ")
        
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        returns = df_24h['close'].pct_change()
        signals = generate_signals(df_24h, df_72h, df_168h)
        
        # Align returns and signals
        common_idx = returns.index.intersection(signals.index)
        aligned_returns = returns.loc[common_idx]
        aligned_signals = signals.loc[common_idx]
        forward_returns = aligned_returns.shift(-1)
        
        print(f"{len(aligned_signals)} days")
        total_samples += len(aligned_signals)
        
        # Calculate hit rates for all 32 states
        for price_perm in product([0, 1], repeat=3):
            for ma_perm in product([0, 1], repeat=2):
                state_key = (price_perm, ma_perm)
                
                mask = (
                    (aligned_signals['price_vs_ma24'] == price_perm[0]) &
                    (aligned_signals['price_vs_ma72'] == price_perm[1]) &
                    (aligned_signals['price_vs_ma168'] == price_perm[2]) &
                    (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
                    (aligned_signals['ma168_above_ma24'] == ma_perm[1])
                )
                
                state_returns = forward_returns[mask].dropna()
                n = len(state_returns)
                hits = (state_returns > 0).sum()
                
                if state_key not in all_hit_rates:
                    all_hit_rates[state_key] = {'n': 0, 'hits': 0}
                
                all_hit_rates[state_key]['n'] += n
                all_hit_rates[state_key]['hits'] += hits
    
    # Display results
    print(f"\nTotal samples: {total_samples:,}")
    print("\n" + "=" * 100)
    print("HIT RATES BY STATE (Sorted by Hit Rate)")
    print("=" * 100)
    
    # Sort by hit rate
    sorted_states = sorted(
        all_hit_rates.items(),
        key=lambda x: x[1]['hits'] / x[1]['n'] if x[1]['n'] > 0 else 0,
        reverse=True
    )
    
    print(f"\n  {'#':<4} {'Price':<10} {'MA Align':<10} {'Hit Rate':>10} {'n':>10} {'Action':>10}")
    print("  " + "-" * 60)
    
    sufficient_count = 0
    invest_count = 0
    avoid_count = 0
    
    for i, (state_key, data) in enumerate(sorted_states, 1):
        price_perm, ma_perm = state_key
        
        n = data['n']
        hr = data['hits'] / n if n > 0 else 0
        
        # Price state label
        p24 = 'U' if price_perm[0] == 1 else 'D'
        p72 = 'U' if price_perm[1] == 1 else 'D'
        p168 = 'U' if price_perm[2] == 1 else 'D'
        price_label = f"{p24}/{p72}/{p168}"
        
        # MA alignment label
        ma_label = get_alignment_label(ma_perm[0], ma_perm[1])
        
        # Determine action
        sufficient = n >= MIN_SAMPLES * len(DEPLOY_PAIRS)
        if sufficient:
            sufficient_count += 1
            if hr > 0.50:
                action = "INVEST"
                invest_count += 1
            else:
                action = "AVOID"
                avoid_count += 1
        else:
            action = "SKIP"
        
        # Highlight strong signals
        if hr >= 0.55 and sufficient:
            marker = "★"
        elif hr <= 0.45 and sufficient:
            marker = "☆"
        else:
            marker = " "
        
        print(f"  {i:<4} {price_label:<10} {ma_label:<10} {hr*100:>9.1f}% {n:>10,} {action:>10} {marker}")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"""
    Total States:        32
    Sufficient Data:     {sufficient_count}/32 ({sufficient_count/32*100:.0f}%)
    
    INVEST signals:      {invest_count} states (hit rate > 50%)
    AVOID signals:       {avoid_count} states (hit rate ≤ 50%)
    SKIP (insufficient): {32 - sufficient_count} states
    """)
    
    # Group by alignment
    print("\n" + "-" * 60)
    print("HIT RATES BY MA ALIGNMENT")
    print("-" * 60)
    
    alignment_stats = {}
    for (price_perm, ma_perm), data in all_hit_rates.items():
        alignment = get_alignment_label(ma_perm[0], ma_perm[1])
        if alignment not in alignment_stats:
            alignment_stats[alignment] = {'n': 0, 'hits': 0}
        alignment_stats[alignment]['n'] += data['n']
        alignment_stats[alignment]['hits'] += data['hits']
    
    print(f"\n  {'Alignment':<12} {'Hit Rate':>10} {'n':>12}")
    print("  " + "-" * 40)
    for alignment in ['BULLISH', 'BEARISH', 'MIXED-1', 'MIXED-2']:
        if alignment in alignment_stats:
            data = alignment_stats[alignment]
            hr = data['hits'] / data['n'] if data['n'] > 0 else 0
            print(f"  {alignment:<12} {hr*100:>9.1f}% {data['n']:>12,}")
    
    # Group by price state
    print("\n" + "-" * 60)
    print("HIT RATES BY PRICE STATE")
    print("-" * 60)
    
    price_stats = {}
    for (price_perm, ma_perm), data in all_hit_rates.items():
        p24 = 'U' if price_perm[0] == 1 else 'D'
        p72 = 'U' if price_perm[1] == 1 else 'D'
        p168 = 'U' if price_perm[2] == 1 else 'D'
        price_label = f"{p24}/{p72}/{p168}"
        
        if price_label not in price_stats:
            price_stats[price_label] = {'n': 0, 'hits': 0}
        price_stats[price_label]['n'] += data['n']
        price_stats[price_label]['hits'] += data['hits']
    
    print(f"\n  {'Price State':<12} {'Hit Rate':>10} {'n':>12}")
    print("  " + "-" * 40)
    sorted_price = sorted(price_stats.items(), 
                         key=lambda x: x[1]['hits']/x[1]['n'] if x[1]['n'] > 0 else 0,
                         reverse=True)
    for price_label, data in sorted_price:
        hr = data['hits'] / data['n'] if data['n'] > 0 else 0
        print(f"  {price_label:<12} {hr*100:>9.1f}% {data['n']:>12,}")
    
    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    
    # Top 5 states
    print("\n  TOP 5 STATES (Highest Hit Rate):")
    print("  " + "-" * 50)
    for i, (state_key, data) in enumerate(sorted_states[:5], 1):
        price_perm, ma_perm = state_key
        p24 = 'U' if price_perm[0] == 1 else 'D'
        p72 = 'U' if price_perm[1] == 1 else 'D'
        p168 = 'U' if price_perm[2] == 1 else 'D'
        price_label = f"{p24}/{p72}/{p168}"
        ma_label = get_alignment_label(ma_perm[0], ma_perm[1])
        hr = data['hits'] / data['n'] if data['n'] > 0 else 0
        print(f"  {i}. {price_label} + {ma_label:<8} → {hr*100:.1f}% hit rate (n={data['n']:,})")
    
    # Bottom 5 states
    print("\n  BOTTOM 5 STATES (Lowest Hit Rate):")
    print("  " + "-" * 50)
    for i, (state_key, data) in enumerate(sorted_states[-5:], 1):
        price_perm, ma_perm = state_key
        p24 = 'U' if price_perm[0] == 1 else 'D'
        p72 = 'U' if price_perm[1] == 1 else 'D'
        p168 = 'U' if price_perm[2] == 1 else 'D'
        price_label = f"{p24}/{p72}/{p168}"
        ma_label = get_alignment_label(ma_perm[0], ma_perm[1])
        hr = data['hits'] / data['n'] if data['n'] > 0 else 0
        print(f"  {i}. {price_label} + {ma_label:<8} → {hr*100:.1f}% hit rate (n={data['n']:,})")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    return all_hit_rates


if __name__ == "__main__":
    hit_rates = main()
