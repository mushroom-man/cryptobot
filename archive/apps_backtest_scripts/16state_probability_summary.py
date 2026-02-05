#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16-State Probability Summary
============================
Display P(up) for each of the 16 states across all assets.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path('/home/johnhenry/cryptobot')
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from cryptobot.data.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005


# =============================================================================
# DATA PROCESSING
# =============================================================================

def resample_ohlcv(df, timeframe):
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def label_trend_binary(df, ma_period, entry_buffer, exit_buffer):
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


def generate_signals(df_24h, df_72h, df_168h):
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


def calculate_state_probabilities(returns, signals):
    """Calculate hit rate for each of the 16 states."""
    all_price_perms = list(product([0, 1], repeat=2))
    all_ma_perms = list(product([0, 1], repeat=2))
    
    common_idx = returns.index.intersection(signals.index)
    aligned_returns = returns.loc[common_idx]
    aligned_signals = signals.loc[common_idx]
    
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    results = []
    
    for price_perm in all_price_perms:
        for ma_perm in all_ma_perms:
            mask = (
                (aligned_signals['trend_24h'] == price_perm[0]) &
                (aligned_signals['trend_168h'] == price_perm[1]) &
                (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
                (aligned_signals['ma168_above_ma24'] == ma_perm[1])
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            k = (perm_returns > 0).sum() if n > 0 else 0
            
            hit_rate = k / n if n > 0 else 0.5
            
            # State interpretation
            t24 = "UP" if price_perm[0] == 1 else "DN"
            t168 = "UP" if price_perm[1] == 1 else "DN"
            ma72 = ">" if ma_perm[0] == 1 else "<"
            ma168 = ">" if ma_perm[1] == 1 else "<"
            
            state_id = price_perm[0] * 8 + price_perm[1] * 4 + ma_perm[0] * 2 + ma_perm[1]
            
            results.append({
                'state_id': state_id,
                'trend_24h': price_perm[0],
                'trend_168h': price_perm[1],
                'ma72_vs_ma24': ma_perm[0],
                'ma168_vs_ma24': ma_perm[1],
                'state_desc': f"T24:{t24} T168:{t168} MA72{ma72}MA24 MA168{ma168}MA24",
                'n': n,
                'k': k,
                'hit_rate': hit_rate,
            })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("16-STATE PROBABILITY SUMMARY")
    print("=" * 90)
    
    db = Database()
    
    # Collect all results
    all_results = []
    
    print("\nLoading data and calculating probabilities...")
    
    for pair in DEPLOY_PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        signals = generate_signals(df_24h, df_72h, df_168h)
        returns = df_24h['close'].pct_change()
        
        state_probs = calculate_state_probabilities(returns, signals)
        state_probs['asset'] = pair
        all_results.append(state_probs)
        
        print(f"{len(signals)} days")
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Per-asset summary
    print("\n" + "=" * 90)
    print("PER-ASSET STATE PROBABILITIES")
    print("=" * 90)
    
    for pair in DEPLOY_PAIRS:
        asset_data = combined[combined['asset'] == pair].sort_values('state_id')
        
        print(f"\n  {pair}")
        print("  " + "-" * 80)
        print(f"  {'State':<6} {'Description':<40} {'N':>8} {'K':>8} {'P(up)':>10}")
        print("  " + "-" * 80)
        
        for _, row in asset_data.iterrows():
            marker = " *" if row['hit_rate'] > 0.55 else (" !" if row['hit_rate'] < 0.45 else "  ")
            print(f"  {row['state_id']:<6} {row['state_desc']:<40} {row['n']:>8} {row['k']:>8} {row['hit_rate']:>9.1%}{marker}")
    
    # Cross-asset average
    print("\n" + "=" * 90)
    print("CROSS-ASSET AVERAGE PROBABILITIES")
    print("=" * 90)
    
    avg_by_state = combined.groupby('state_id').agg({
        'trend_24h': 'first',
        'trend_168h': 'first',
        'ma72_vs_ma24': 'first',
        'ma168_vs_ma24': 'first',
        'state_desc': 'first',
        'n': 'sum',
        'k': 'sum',
        'hit_rate': 'mean',
    }).reset_index()
    
    # Recalculate pooled hit rate
    avg_by_state['pooled_hit_rate'] = avg_by_state['k'] / avg_by_state['n']
    
    print(f"\n  {'State':<6} {'Description':<40} {'Total N':>10} {'Avg P(up)':>12} {'Pooled P(up)':>14}")
    print("  " + "-" * 85)
    
    for _, row in avg_by_state.sort_values('state_id').iterrows():
        marker = " *" if row['pooled_hit_rate'] > 0.55 else (" !" if row['pooled_hit_rate'] < 0.45 else "  ")
        print(f"  {row['state_id']:<6} {row['state_desc']:<40} {row['n']:>10} {row['hit_rate']:>11.1%} {row['pooled_hit_rate']:>13.1%}{marker}")
    
    # Summary stats
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    high_prob_states = avg_by_state[avg_by_state['pooled_hit_rate'] > 0.55]
    low_prob_states = avg_by_state[avg_by_state['pooled_hit_rate'] < 0.45]
    neutral_states = avg_by_state[(avg_by_state['pooled_hit_rate'] >= 0.45) & (avg_by_state['pooled_hit_rate'] <= 0.55)]
    
    print(f"\n  States with P(up) > 55% (LONG):  {len(high_prob_states)} states")
    for _, row in high_prob_states.iterrows():
        print(f"    State {row['state_id']}: {row['pooled_hit_rate']:.1%} (n={row['n']})")
    
    print(f"\n  States with P(up) < 45% (SHORT): {len(low_prob_states)} states")
    for _, row in low_prob_states.iterrows():
        print(f"    State {row['state_id']}: {row['pooled_hit_rate']:.1%} (n={row['n']})")
    
    print(f"\n  States with 45% <= P(up) <= 55% (NEUTRAL): {len(neutral_states)} states")
    
    print(f"\n  Overall pooled hit rate: {avg_by_state['k'].sum() / avg_by_state['n'].sum():.1%}")
    
    return combined, avg_by_state


if __name__ == "__main__":
    combined, avg_by_state = main()