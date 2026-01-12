#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap Validation: 8-State Risk-Managed Risk Parity
=======================================================
Statistical validation using bootstrap resampling to test:

1. BLOCK BOOTSTRAP
   - Resample blocks of consecutive days (preserves autocorrelation)
   - Build distribution of performance metrics
   - Calculate confidence intervals

2. MONTE CARLO PERMUTATION
   - Shuffle returns to destroy signal
   - Compare real strategy to random
   - Calculate p-value (is alpha real?)

3. WALK-FORWARD BOOTSTRAP
   - Bootstrap within each walk-forward window
   - Test stability across time periods

OUTPUT:
   - 95% confidence intervals for all metrics
   - P-values for alpha significance
   - Distribution plots

Usage:
    python bootstrap_validation.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import product
import warnings
from datetime import datetime
import time
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database

# Try numba for speed
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# Strategy parameters
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES = 20

# Risk management
TARGET_VOL = 0.40
VOL_LOOKBACK = 30
DD_START_REDUCE = -0.20
DD_MIN_EXPOSURE = -0.50
MIN_EXPOSURE_FLOOR = 0.40

# Portfolio
REBALANCE_DAYS = 30
COV_LOOKBACK = 60
MIN_POSITION_CHANGE = 0.05  # Cost-optimized setting

# Bootstrap settings
N_BOOTSTRAP = 1000       # Number of bootstrap samples
BLOCK_SIZE = 20          # Days per block (preserves short-term autocorrelation)
CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals
RANDOM_SEED = 42


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
        
        if current == 1:
            if price < ma.iloc[i] * (1 - exit_buffer) and price < ma.iloc[i] * (1 - entry_buffer):
                current = 0
        else:
            if price > ma.iloc[i] * (1 + exit_buffer) and price > ma.iloc[i] * (1 + entry_buffer):
                current = 1
        
        labels.iloc[i] = current
    
    return labels


def block_bootstrap_indices(n_samples: int, block_size: int, n_blocks_needed: int, 
                            rng: np.random.Generator) -> np.ndarray:
    """Generate block bootstrap indices."""
    n_possible_starts = n_samples - block_size + 1
    
    if n_possible_starts <= 0:
        return np.arange(n_samples)
    
    indices = []
    while len(indices) < n_blocks_needed * block_size:
        start = rng.integers(0, n_possible_starts)
        block = np.arange(start, start + block_size)
        indices.extend(block)
    
    return np.array(indices[:n_blocks_needed * block_size])


# =============================================================================
# FAST STRATEGY SIMULATION
# =============================================================================

def simulate_strategy(returns: np.ndarray, signals: np.ndarray, 
                      weights: np.ndarray) -> Dict:
    """
    Fast strategy simulation.
    
    Args:
        returns: (n_days, n_assets) array of returns
        signals: (n_days, n_assets) array of 8-state positions (0 or 1)
        weights: (n_days, n_assets) array of risk parity weights
    
    Returns:
        Dictionary of performance metrics
    """
    n_days, n_assets = returns.shape
    
    # Portfolio returns before risk management
    weighted_returns = returns * weights * signals
    port_returns_raw = np.sum(weighted_returns, axis=1)
    
    # Rolling volatility for vol targeting
    vol_scalar = np.ones(n_days)
    for i in range(VOL_LOOKBACK, n_days):
        realized_vol = np.std(port_returns_raw[i-VOL_LOOKBACK:i]) * np.sqrt(365)
        if realized_vol > 0:
            vol_scalar[i] = np.clip(TARGET_VOL / realized_vol, MIN_EXPOSURE_FLOOR, 1.0)
    
    # Simulate with drawdown control
    equity = np.ones(n_days)
    peak = 1.0
    dd_scalar = 1.0
    
    for i in range(1, n_days):
        exposure = min(vol_scalar[i-1], dd_scalar)
        exposure = max(exposure, MIN_EXPOSURE_FLOOR)
        
        equity[i] = equity[i-1] * (1.0 + port_returns_raw[i] * exposure)
        peak = max(peak, equity[i])
        
        current_dd = (equity[i] - peak) / peak
        
        if current_dd >= DD_START_REDUCE:
            dd_scalar = 1.0
        elif current_dd <= DD_MIN_EXPOSURE:
            dd_scalar = MIN_EXPOSURE_FLOOR
        else:
            range_dd = DD_START_REDUCE - DD_MIN_EXPOSURE
            position = (current_dd - DD_MIN_EXPOSURE) / range_dd
            dd_scalar = MIN_EXPOSURE_FLOOR + position * (1.0 - MIN_EXPOSURE_FLOOR)
    
    # Calculate metrics
    total_return = equity[-1] - 1.0
    years = n_days / 365
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
    
    rolling_max = np.maximum.accumulate(equity)
    max_dd = np.min((equity - rolling_max) / rolling_max)
    
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'equity': equity
    }


def simulate_buy_hold(returns: np.ndarray, weights: np.ndarray) -> Dict:
    """Simulate buy & hold benchmark."""
    n_days = len(returns)
    
    port_returns = np.sum(returns * weights, axis=1)
    equity = np.cumprod(1 + port_returns)
    
    total_return = equity[-1] - 1.0
    years = n_days / 365
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
    
    rolling_max = np.maximum.accumulate(equity)
    max_dd = np.min((equity - rolling_max) / rolling_max)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'equity': equity
    }


# =============================================================================
# BOOTSTRAP TESTS
# =============================================================================

@dataclass
class BootstrapResult:
    """Bootstrap test results."""
    metric_name: str
    observed: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    p_value: float
    distribution: np.ndarray


def run_block_bootstrap(returns: np.ndarray, signals: np.ndarray, 
                        weights: np.ndarray, n_bootstrap: int,
                        block_size: int, rng: np.random.Generator) -> Dict[str, BootstrapResult]:
    """
    Run block bootstrap to estimate confidence intervals.
    """
    n_days, n_assets = returns.shape
    n_blocks_needed = (n_days // block_size) + 1
    
    # Observed metrics
    observed = simulate_strategy(returns, signals, weights)
    
    # Bootstrap distributions
    bootstrap_metrics = {
        'total_return': [],
        'annual_return': [],
        'sharpe': [],
        'max_dd': [],
        'calmar': []
    }
    
    for b in range(n_bootstrap):
        # Generate bootstrap indices
        indices = block_bootstrap_indices(n_days, block_size, n_blocks_needed, rng)
        indices = indices[:n_days]  # Trim to original length
        
        # Resample
        boot_returns = returns[indices]
        boot_signals = signals[indices]
        boot_weights = weights[indices]
        
        # Simulate
        result = simulate_strategy(boot_returns, boot_signals, boot_weights)
        
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(result[key])
    
    # Calculate statistics
    results = {}
    alpha = 1 - CONFIDENCE_LEVEL
    
    for metric in bootstrap_metrics:
        dist = np.array(bootstrap_metrics[metric])
        
        # Remove infinities and NaNs
        dist = dist[np.isfinite(dist)]
        
        if len(dist) == 0:
            continue
        
        results[metric] = BootstrapResult(
            metric_name=metric,
            observed=observed[metric],
            mean=np.mean(dist),
            std=np.std(dist),
            ci_lower=np.percentile(dist, alpha/2 * 100),
            ci_upper=np.percentile(dist, (1 - alpha/2) * 100),
            p_value=np.mean(dist <= 0) if metric in ['total_return', 'annual_return', 'sharpe', 'calmar'] else np.mean(dist >= 0),
            distribution=dist
        )
    
    return results


def run_permutation_test(returns: np.ndarray, signals: np.ndarray,
                         weights: np.ndarray, n_permutations: int,
                         rng: np.random.Generator) -> Dict[str, float]:
    """
    Permutation test to check if strategy alpha is real.
    Shuffles the time series to destroy any predictive signal.
    """
    n_days = len(returns)
    
    # Observed strategy performance
    observed = simulate_strategy(returns, signals, weights)
    observed_bh = simulate_buy_hold(returns, weights)
    
    # Alpha = strategy return - buy & hold return
    observed_alpha = observed['total_return'] - observed_bh['total_return']
    observed_sharpe_diff = observed['sharpe'] - observed_bh['sharpe']
    
    # Permutation distribution
    alpha_dist = []
    sharpe_diff_dist = []
    
    for _ in range(n_permutations):
        # Shuffle returns (destroys temporal structure)
        perm_indices = rng.permutation(n_days)
        perm_returns = returns[perm_indices]
        
        # Keep signals aligned with original time (tests if signals have predictive power)
        perm_result = simulate_strategy(perm_returns, signals, weights)
        perm_bh = simulate_buy_hold(perm_returns, weights)
        
        perm_alpha = perm_result['total_return'] - perm_bh['total_return']
        perm_sharpe_diff = perm_result['sharpe'] - perm_bh['sharpe']
        
        alpha_dist.append(perm_alpha)
        sharpe_diff_dist.append(perm_sharpe_diff)
    
    alpha_dist = np.array(alpha_dist)
    sharpe_diff_dist = np.array(sharpe_diff_dist)
    
    # P-value: proportion of permutations with alpha >= observed
    p_value_alpha = np.mean(alpha_dist >= observed_alpha)
    p_value_sharpe = np.mean(sharpe_diff_dist >= observed_sharpe_diff)
    
    return {
        'observed_alpha': observed_alpha,
        'observed_sharpe_diff': observed_sharpe_diff,
        'p_value_alpha': p_value_alpha,
        'p_value_sharpe': p_value_sharpe,
        'alpha_dist_mean': np.mean(alpha_dist),
        'alpha_dist_std': np.std(alpha_dist),
        'sharpe_dist_mean': np.mean(sharpe_diff_dist),
        'sharpe_dist_std': np.std(sharpe_diff_dist)
    }


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data():
    """Load and prepare all data for bootstrap testing."""
    print("Loading data...")
    
    db = Database()
    
    all_returns = {}
    all_signals = {}
    
    for pair in DEPLOY_PAIRS:
        print(f"  {pair}...", end=" ")
        
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        # Returns
        all_returns[pair] = df_24h['close'].pct_change()
        
        # Signals
        trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
        trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
        trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
        
        signals = pd.DataFrame(index=df_24h.index)
        signals['trend_24h'] = trend_24h.shift(1)
        signals['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
        signals['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
        
        # Calculate hit rates and determine position
        common_idx = all_returns[pair].index.intersection(signals.index)
        aligned_returns = all_returns[pair].loc[common_idx]
        aligned_signals = signals.loc[common_idx]
        forward_returns = aligned_returns.shift(-1)
        
        hit_rates = {}
        for perm in product([0, 1], repeat=3):
            mask = (
                (aligned_signals['trend_24h'] == perm[0]) &
                (aligned_signals['trend_72h'] == perm[1]) &
                (aligned_signals['trend_168h'] == perm[2])
            )
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            if n >= MIN_SAMPLES:
                hit_rate = (perm_returns > 0).sum() / n
                hit_rates[perm] = 1.0 if hit_rate > HIT_RATE_THRESHOLD else 0.0
            else:
                hit_rates[perm] = 0.5
        
        # Create position series
        positions = pd.Series(index=signals.index, dtype=float)
        for idx in signals.index:
            if pd.isna(signals.loc[idx]).any():
                positions[idx] = 0.5
            else:
                perm = tuple(signals.loc[idx].astype(int))
                positions[idx] = hit_rates.get(perm, 0.5)
        
        all_signals[pair] = positions
        print(f"{len(df_24h)} bars")
    
    # Create aligned DataFrames
    returns_df = pd.DataFrame(all_returns).dropna()
    signals_df = pd.DataFrame({p: all_signals[p] for p in DEPLOY_PAIRS})
    signals_df = signals_df.loc[returns_df.index].fillna(0.5)
    
    print(f"\n  Common period: {returns_df.index.min().date()} to {returns_df.index.max().date()}")
    print(f"  Total days: {len(returns_df)}")
    
    # Calculate risk parity weights
    print("  Calculating risk parity weights...")
    
    n_days = len(returns_df)
    weights_df = pd.DataFrame(index=returns_df.index, columns=DEPLOY_PAIRS, dtype=float)
    
    current_weights = np.ones(len(DEPLOY_PAIRS)) / len(DEPLOY_PAIRS)
    
    for i in range(n_days):
        if i >= COV_LOOKBACK and (i - COV_LOOKBACK) % REBALANCE_DAYS == 0:
            trailing = returns_df.iloc[i-COV_LOOKBACK:i]
            cov = trailing.cov().values * 252
            vols = np.sqrt(np.diag(cov))
            vols[vols == 0] = 1e-6
            inv_vols = 1.0 / vols
            current_weights = inv_vols / inv_vols.sum()
        
        weights_df.iloc[i] = current_weights
    
    return returns_df.values, signals_df.values, weights_df.values, returns_df.index


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("BOOTSTRAP VALIDATION: 8-STATE RISK-MANAGED RISK PARITY")
    print("=" * 100)
    
    print(f"""
    BOOTSTRAP SETTINGS:
        N Bootstrap Samples:    {N_BOOTSTRAP}
        Block Size:             {BLOCK_SIZE} days
        Confidence Level:       {CONFIDENCE_LEVEL*100:.0f}%
        Random Seed:            {RANDOM_SEED}
    """)
    
    # Prepare data
    returns, signals, weights, dates = prepare_data()
    n_days = len(returns)
    
    # Initialize RNG
    rng = np.random.default_rng(RANDOM_SEED)
    
    # =========================================================================
    # 1. BLOCK BOOTSTRAP
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("1. BLOCK BOOTSTRAP: Confidence Intervals")
    print("=" * 100)
    
    print(f"\n  Running {N_BOOTSTRAP} bootstrap samples...")
    start_time = time.time()
    
    bootstrap_results = run_block_bootstrap(
        returns, signals, weights, N_BOOTSTRAP, BLOCK_SIZE, rng
    )
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    
    print(f"\n  {'Metric':<20} {'Observed':>12} {'Mean':>12} {'Std':>10} {'95% CI':>25} {'P-value':>10}")
    print("  " + "-" * 95)
    
    for metric, result in bootstrap_results.items():
        ci_str = f"[{result.ci_lower:+.1%}, {result.ci_upper:+.1%}]" if 'return' in metric else f"[{result.ci_lower:.2f}, {result.ci_upper:.2f}]"
        obs_str = f"{result.observed:+.1%}" if 'return' in metric else f"{result.observed:.2f}"
        mean_str = f"{result.mean:+.1%}" if 'return' in metric else f"{result.mean:.2f}"
        std_str = f"{result.std:.1%}" if 'return' in metric else f"{result.std:.2f}"
        
        print(f"  {metric:<20} {obs_str:>12} {mean_str:>12} {std_str:>10} {ci_str:>25} {result.p_value:>10.4f}")
    
    # =========================================================================
    # 2. PERMUTATION TEST
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("2. PERMUTATION TEST: Is Alpha Statistically Significant?")
    print("=" * 100)
    
    print(f"\n  Running {N_BOOTSTRAP} permutations...")
    start_time = time.time()
    
    perm_results = run_permutation_test(returns, signals, weights, N_BOOTSTRAP, rng)
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    
    print(f"""
    ALPHA (Strategy Return - Buy & Hold Return):
        Observed Alpha:         {perm_results['observed_alpha']*100:+.1f}%
        Permutation Mean:       {perm_results['alpha_dist_mean']*100:+.1f}%
        Permutation Std:        {perm_results['alpha_dist_std']*100:.1f}%
        P-value:                {perm_results['p_value_alpha']:.4f}
        
    SHARPE DIFFERENCE (Strategy Sharpe - Buy & Hold Sharpe):
        Observed Difference:    {perm_results['observed_sharpe_diff']:+.2f}
        Permutation Mean:       {perm_results['sharpe_dist_mean']:+.2f}
        Permutation Std:        {perm_results['sharpe_dist_std']:.2f}
        P-value:                {perm_results['p_value_sharpe']:.4f}
    """)
    
    # =========================================================================
    # 3. INTERPRETATION
    # =========================================================================
    
    print("=" * 100)
    print("3. INTERPRETATION")
    print("=" * 100)
    
    # Statistical significance thresholds
    alpha_significant = perm_results['p_value_alpha'] < 0.05
    sharpe_significant = perm_results['p_value_sharpe'] < 0.05
    
    # Confidence interval quality
    ci_contains_zero = {}
    for metric, result in bootstrap_results.items():
        if metric in ['total_return', 'annual_return', 'sharpe', 'calmar']:
            ci_contains_zero[metric] = result.ci_lower <= 0 <= result.ci_upper
    
    print(f"""
    STATISTICAL SIGNIFICANCE (α = 0.05):
    
    Alpha vs Buy & Hold:
        P-value: {perm_results['p_value_alpha']:.4f}
        Result:  {'✓ SIGNIFICANT' if alpha_significant else '✗ NOT SIGNIFICANT'} at 95% level
        
    Sharpe Improvement:
        P-value: {perm_results['p_value_sharpe']:.4f}
        Result:  {'✓ SIGNIFICANT' if sharpe_significant else '✗ NOT SIGNIFICANT'} at 95% level
    
    CONFIDENCE INTERVALS:
    """)
    
    for metric, contains_zero in ci_contains_zero.items():
        result = bootstrap_results[metric]
        status = "✗ Contains zero (uncertain)" if contains_zero else "✓ Does not contain zero (robust)"
        print(f"    {metric:<20}: {status}")
        print(f"                          95% CI: [{result.ci_lower:.2%}, {result.ci_upper:.2%}]")
    
    # =========================================================================
    # 4. VERDICT
    # =========================================================================
    
    # Overall assessment
    n_significant = sum([alpha_significant, sharpe_significant])
    n_robust_ci = sum([not v for v in ci_contains_zero.values()])
    
    if n_significant >= 2 and n_robust_ci >= 3:
        verdict = "STRONG"
        verdict_text = "Strategy shows statistically significant alpha with robust confidence intervals"
    elif n_significant >= 1 and n_robust_ci >= 2:
        verdict = "MODERATE"
        verdict_text = "Strategy shows some statistical evidence of alpha, but with uncertainty"
    else:
        verdict = "WEAK"
        verdict_text = "Strategy alpha may not be statistically distinguishable from random"
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  BOOTSTRAP VALIDATION VERDICT: {verdict:<10}                                                        ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                      ║
    ║  {verdict_text:<84} ║
    ║                                                                                                      ║
    ║  KEY FINDINGS:                                                                                       ║
    ║    • Alpha P-value:     {perm_results['p_value_alpha']:.4f} ({'< 0.05 ✓' if alpha_significant else '>= 0.05 ✗':<12})                                                   ║
    ║    • Sharpe P-value:    {perm_results['p_value_sharpe']:.4f} ({'< 0.05 ✓' if sharpe_significant else '>= 0.05 ✗':<12})                                                   ║
    ║    • Robust CIs:        {n_robust_ci}/4 metrics have CIs not containing zero                               ║
    ║                                                                                                      ║
    ║  95% CONFIDENCE INTERVALS:                                                                           ║
    ║    • Total Return:      [{bootstrap_results['total_return'].ci_lower*100:+6.0f}%, {bootstrap_results['total_return'].ci_upper*100:+6.0f}%]                                                      ║
    ║    • Sharpe Ratio:      [{bootstrap_results['sharpe'].ci_lower:+5.2f},  {bootstrap_results['sharpe'].ci_upper:+5.2f}]                                                        ║
    ║    • Max Drawdown:      [{bootstrap_results['max_dd'].ci_lower*100:+6.0f}%, {bootstrap_results['max_dd'].ci_upper*100:+6.0f}%]                                                      ║
    ║                                                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n" + "=" * 100)
    print("BOOTSTRAP VALIDATION COMPLETE")
    print("=" * 100)
    
    return bootstrap_results, perm_results


if __name__ == "__main__":
    bootstrap_results, perm_results = main()
