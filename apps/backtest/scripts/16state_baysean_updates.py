#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Bayesian Model - Combined (OPTIMISED)
===================================================
Daily Beta priors with hourly Bayesian updates.
Optimised for speed.

Optimisations:
    - Likelihoods recalculated weekly (not daily)
    - Beta params cached monthly
    - Vectorised operations where possible
    - Pre-computed daily outcomes

Usage:
    python hierarchical_bayesian_combined_fast.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from cryptobot.data.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

ASSETS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']
OUTPUT_DIR = PROJECT_ROOT / 'apps' / 'backtest' / 'outputs'

# MA Parameters
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Beta prior
ALPHA_0 = 10
BETA_0 = 10

# Rolling windows
EVIDENCE_WINDOW_DAYS = 60
LIKELIHOOD_WINDOW_DAYS = 60

# Recalc frequency (speed optimisation)
BETA_RECALC_DAYS = 7      # Weekly
LIKELIHOOD_RECALC_DAYS = 7  # Weekly

BURN_IN_DAYS = 60

# Lambda values
LAMBDA_VALUES = [0.3, 0.5, 0.7, 0.9, 1.0]


# =============================================================================
# 16-STATE SIGNAL GENERATION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def label_trend_binary(close: pd.Series, ma: pd.Series,
                       entry_buffer: float, exit_buffer: float) -> pd.Series:
    """Vectorised where possible, loop where necessary."""
    labels = np.ones(len(close), dtype=int)
    current = 1
    
    for i in range(len(close)):
        if np.isnan(ma.iloc[i]):
            labels[i] = current
            continue
        
        price = close.iloc[i]
        ma_val = ma.iloc[i]
        
        if current == 1:
            if price < ma_val * (1 - exit_buffer) and price < ma_val * (1 - entry_buffer):
                current = 0
        else:
            if price > ma_val * (1 + exit_buffer) and price > ma_val * (1 + entry_buffer):
                current = 1
        
        labels[i] = current
    
    return pd.Series(labels, index=close.index)


def generate_signals_fast(df_1h: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate signals and pre-compute daily outcomes."""
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    trend_24h = label_trend_binary(df_24h['close'], ma_24h, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h['close'], ma_168h, ENTRY_BUFFER, EXIT_BUFFER)
    
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    signals = pd.DataFrame(index=df_24h.index)
    signals['trend_24h'] = trend_24h.shift(1)
    signals['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    signals['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    signals['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    signals = signals.dropna().astype(int)
    
    # Pre-compute state keys as integers (faster lookup)
    signals['state_int'] = (signals['trend_24h'] * 8 + 
                           signals['trend_168h'] * 4 + 
                           signals['ma72_above_ma24'] * 2 + 
                           signals['ma168_above_ma24'])
    
    # Daily outcomes
    daily_returns = df_24h['close'].pct_change()
    outcomes = (daily_returns > 0).astype(int)
    
    return signals, outcomes, daily_returns


# =============================================================================
# BETA PARAMETERS (VECTORISED)
# =============================================================================

def calculate_beta_params_fast(
    signals: pd.DataFrame,
    outcomes: pd.Series,
    current_idx: int,
    dates: np.ndarray,
    window_days: int = EVIDENCE_WINDOW_DAYS
) -> np.ndarray:
    """
    Calculate Beta parameters for all 16 states.
    Returns array of shape (16, 2) with [alpha, beta] for each state.
    """
    current_date = dates[current_idx]
    window_start = current_date - np.timedelta64(window_days, 'D')
    
    # Filter to window
    mask = (dates >= window_start) & (dates < current_date)
    window_indices = np.where(mask)[0]
    
    # Initialize with base prior
    params = np.full((16, 2), [ALPHA_0, BETA_0], dtype=float)
    
    if len(window_indices) == 0:
        return params
    
    # Get states and outcomes in window
    window_states = signals['state_int'].values[window_indices]
    window_outcomes = outcomes.values[window_indices]
    
    # Count successes and failures per state
    for state in range(16):
        state_mask = window_states == state
        if state_mask.any():
            params[state, 0] += window_outcomes[state_mask].sum()  # alpha += successes
            params[state, 1] += (~window_outcomes[state_mask].astype(bool)).sum()  # beta += failures
    
    return params


# =============================================================================
# HOURLY LIKELIHOODS (CACHED)
# =============================================================================

def calculate_likelihoods_fast(
    hourly_returns: np.ndarray,
    hourly_hours: np.ndarray,
    hourly_dates: np.ndarray,
    daily_outcomes: Dict,
    end_date: np.datetime64,
    window_days: int = LIKELIHOOD_WINDOW_DAYS
) -> np.ndarray:
    """
    Calculate likelihoods. Returns array (24, 2) with 
    [p_up_given_day_up, p_up_given_day_down] per hour.
    """
    start_date = end_date - np.timedelta64(window_days, 'D')
    
    mask = (hourly_dates >= start_date) & (hourly_dates < end_date)
    
    if mask.sum() < 100:
        return np.full((24, 2), 0.5)
    
    # Filter data
    returns = hourly_returns[mask]
    hours = hourly_hours[mask]
    dates = hourly_dates[mask]
    
    likelihoods = np.full((24, 2), 0.5)
    
    for hour in range(24):
        hour_mask = hours == hour
        if hour_mask.sum() < 20:
            continue
        
        hour_returns = returns[hour_mask]
        hour_dates = dates[hour_mask]
        
        # Get day outcomes for these hours
        day_up = np.array([daily_outcomes.get(d, 0.5) for d in hour_dates])
        
        # Split by day outcome
        day_up_mask = day_up == 1
        day_down_mask = day_up == 0
        
        if day_up_mask.sum() >= 10:
            likelihoods[hour, 0] = (hour_returns[day_up_mask] > 0).mean()
        
        if day_down_mask.sum() >= 10:
            likelihoods[hour, 1] = (hour_returns[day_down_mask] > 0).mean()
    
    return likelihoods


# =============================================================================
# BAYESIAN UPDATE (VECTORISED FOR DAY)
# =============================================================================

def process_day_fast(
    hourly_returns: np.ndarray,
    hourly_hours: np.ndarray,
    prior: float,
    likelihoods: np.ndarray,
    lambda_: float
) -> float:
    """Process all hours in a day, return end-of-day posterior."""
    posterior = prior
    
    for i in range(len(hourly_returns)):
        ret = hourly_returns[i]
        hour = hourly_hours[i]
        
        if np.isnan(ret):
            continue
        
        p_up_day_up = likelihoods[hour, 0]
        p_up_day_down = likelihoods[hour, 1]
        
        prior_down = 1 - posterior
        
        if ret > 0:
            num = p_up_day_up * posterior
            denom = p_up_day_up * posterior + p_up_day_down * prior_down
        else:
            p_down_day_up = 1 - p_up_day_up
            p_down_day_down = 1 - p_up_day_down
            num = p_down_day_up * posterior
            denom = p_down_day_up * posterior + p_down_day_down * prior_down
        
        if denom > 0:
            raw_posterior = num / denom
            raw_posterior = np.clip(raw_posterior, 0.01, 0.99)
        else:
            raw_posterior = posterior
        
        # EWMA
        posterior = lambda_ * raw_posterior + (1 - lambda_) * posterior
    
    return posterior


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_asset_fast(db: Database, pair: str, lambda_: float) -> Tuple[List[Dict], List[Dict]]:
    """Optimised analysis for a single asset."""
    
    # Load data
    df_1h = db.get_ohlcv(pair)
    if len(df_1h) == 0:
        raise ValueError(f"No data for {pair}")
    
    # Pre-compute everything
    signals, outcomes, daily_returns = generate_signals_fast(df_1h)
    
    # Prepare hourly data as numpy arrays
    df_1h['hourly_return'] = df_1h['close'].pct_change()
    hourly_returns = df_1h['hourly_return'].values
    hourly_hours = df_1h.index.hour.values
    hourly_dates = df_1h.index.normalize().values
    
    # Daily outcomes dict for fast lookup
    daily_outcomes_dict = {d: int(v) for d, v in zip(outcomes.index.values, outcomes.values)}
    
    # Get common dates
    dates = signals.index.values
    n_dates = len(dates)
    
    min_start = BURN_IN_DAYS + LIKELIHOOD_WINDOW_DAYS
    if n_dates < min_start:
        raise ValueError(f"Insufficient data for {pair}")
    
    hourly_predictions = []
    daily_predictions = []
    
    # Caches
    beta_cache = None
    likelihood_cache = None
    last_beta_idx = -999
    last_likelihood_idx = -999
    
    for i in range(min_start, n_dates):
        date = dates[i]
        
        # Recalc beta params weekly
        if i - last_beta_idx >= BETA_RECALC_DAYS:
            beta_cache = calculate_beta_params_fast(signals, outcomes, i, dates)
            last_beta_idx = i
        
        # Recalc likelihoods weekly
        if i - last_likelihood_idx >= LIKELIHOOD_RECALC_DAYS:
            likelihood_cache = calculate_likelihoods_fast(
                hourly_returns, hourly_hours, hourly_dates,
                daily_outcomes_dict, date
            )
            last_likelihood_idx = i
        
        # Get state and prior
        state_int = signals['state_int'].iloc[i]
        alpha, beta = beta_cache[state_int]
        prior = alpha / (alpha + beta)
        prior_std = np.sqrt((alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1)))
        
        # Actual outcome
        actual = outcomes.iloc[i]
        
        # Daily baseline
        daily_predictions.append({
            'date': date,
            'asset': pair,
            'predicted_prob': prior,
            'actual_outcome': actual,
        })
        
        # Get hourly data for this day
        day_mask = hourly_dates == date
        day_returns = hourly_returns[day_mask]
        day_hours = hourly_hours[day_mask]
        
        if len(day_returns) == 0:
            continue
        
        # Process day
        end_posterior = process_day_fast(day_returns, day_hours, prior, likelihood_cache, lambda_)
        
        hourly_predictions.append({
            'date': date,
            'asset': pair,
            'predicted_prob': end_posterior,
            'prior': prior,
            'actual_outcome': actual,
        })
    
    return hourly_predictions, daily_predictions


def run_lambda_analysis(db: Database) -> pd.DataFrame:
    """Test lambda values across all assets."""
    results = []
    
    for lambda_ in LAMBDA_VALUES:
        print(f"\n  Testing λ = {lambda_}...")
        
        for pair in ASSETS:
            print(f"    {pair}...", end=" ", flush=True)
            
            try:
                hourly_preds, daily_preds = analyze_asset_fast(db, pair, lambda_)
                
                # Calculate metrics
                hourly_probs = np.array([p['predicted_prob'] for p in hourly_preds])
                hourly_actuals = np.array([p['actual_outcome'] for p in hourly_preds])
                daily_probs = np.array([p['predicted_prob'] for p in daily_preds])
                daily_actuals = np.array([p['actual_outcome'] for p in daily_preds])
                
                hourly_acc = np.mean((hourly_probs > 0.5) == (hourly_actuals == 1))
                daily_acc = np.mean((daily_probs > 0.5) == (daily_actuals == 1))
                
                hourly_brier = np.mean((hourly_probs - hourly_actuals)**2)
                daily_brier = np.mean((daily_probs - daily_actuals)**2)
                
                improvement = hourly_acc - daily_acc
                
                print(f"hourly={hourly_acc:.3f}, daily={daily_acc:.3f}, Δ={improvement:+.3f}")
                
                results.append({
                    'lambda': lambda_,
                    'asset': pair,
                    'hourly_accuracy': hourly_acc,
                    'daily_accuracy': daily_acc,
                    'improvement': improvement,
                    'hourly_brier': hourly_brier,
                    'daily_brier': daily_brier,
                    'n_days': len(hourly_preds),
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("HIERARCHICAL BAYESIAN MODEL - COMBINED (FAST)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Daily Prior:       Beta({ALPHA_0}, {BETA_0}) + {EVIDENCE_WINDOW_DAYS}-day evidence")
    print(f"  Hourly Likelihood: Rolling {LIKELIHOOD_WINDOW_DAYS}-day (recalc every {LIKELIHOOD_RECALC_DAYS} days)")
    print(f"  Beta recalc:       Every {BETA_RECALC_DAYS} days")
    print(f"  Lambda values:     {LAMBDA_VALUES}")
    
    print("\nConnecting to database...")
    db = Database()
    
    print("\nRunning analysis:")
    print("-" * 70)
    
    results_df = run_lambda_analysis(db)
    
    if len(results_df) == 0:
        print("No results generated.")
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    summary = results_df.groupby('lambda').agg({
        'hourly_accuracy': 'mean',
        'daily_accuracy': 'mean',
        'improvement': 'mean',
        'hourly_brier': 'mean',
        'daily_brier': 'mean',
    }).round(4)
    
    print("\nMean metrics by λ:")
    print("-" * 70)
    print(f"{'Lambda':<8} {'Hourly':>10} {'Daily':>10} {'Δ':>10} {'Brier H':>12} {'Brier D':>12}")
    print("-" * 70)
    
    for lambda_ in LAMBDA_VALUES:
        if lambda_ in summary.index:
            row = summary.loc[lambda_]
            print(f"{lambda_:<8} {row['hourly_accuracy']:>10.4f} {row['daily_accuracy']:>10.4f} "
                  f"{row['improvement']:>+10.4f} {row['hourly_brier']:>12.4f} {row['daily_brier']:>12.4f}")
    
    best_lambda = summary['improvement'].idxmax()
    best_improvement = summary.loc[best_lambda, 'improvement']
    
    print("\n" + "-" * 70)
    print(f"Best λ = {best_lambda} (improvement: {best_improvement:+.4f})")
    print("-" * 70)
    
    # Per-asset
    print(f"\nPer-asset results for λ = {best_lambda}:")
    print("-" * 70)
    
    best_results = results_df[results_df['lambda'] == best_lambda]
    for _, row in best_results.iterrows():
        print(f"  {row['asset']:<10} hourly={row['hourly_accuracy']:.4f} daily={row['daily_accuracy']:.4f} "
              f"Δ={row['improvement']:+.4f} n={row['n_days']}")
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / 'hierarchical_bayesian_combined_fast_results.csv'
    results_df.to_csv(results_path, index=False, float_format='%.4f')
    
    print(f"\nResults saved to: {results_path}")
    
    return results_df, summary


if __name__ == "__main__":
    results_df, summary = main()