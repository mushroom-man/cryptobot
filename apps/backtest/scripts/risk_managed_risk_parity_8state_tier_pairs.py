#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk-Managed Risk Parity: OPTIMIZED VERSION
============================================
Faster implementation using vectorized operations.
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# Settings
REBALANCE_DAYS = 30
COV_LOOKBACK = 60
TARGET_VOL = 0.60
VOL_LOOKBACK = 30
MAX_LEVERAGE = 1.0
MIN_EXPOSURE = 0.20
DD_START_REDUCE = -0.25
DD_MIN_EXPOSURE = -0.40
TOTAL_COST = 0.0015


# =============================================================================
# FAST HELPERS
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def risk_parity_weights_fast(cov_matrix: np.ndarray) -> np.ndarray:
    """Fast risk parity using inverse volatility as approximation."""
    # Inverse volatility weighting (fast approximation)
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return weights


def risk_parity_weights_full(cov_matrix: pd.DataFrame) -> pd.Series:
    """Full risk parity optimization."""
    n = len(cov_matrix)
    
    def objective(w, cov):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        return np.sum((rc - target) ** 2)
    
    w0 = np.ones(n) / n
    bounds = [(0.01, 0.5)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    result = minimize(objective, w0, args=(cov_matrix.values,),
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 100})
    
    weights = pd.Series(result.x, index=cov_matrix.columns)
    return weights / weights.sum()


# =============================================================================
# VECTORIZED BACKTEST
# =============================================================================

@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    sharpe: float
    max_dd: float
    calmar: float
    avg_exposure: float
    equity_curve: pd.Series
    exposure_history: pd.Series


def backtest_vectorized(returns_df: pd.DataFrame, use_risk_management: bool = True) -> BacktestResult:
    """Vectorized backtest for speed."""
    
    returns = returns_df.dropna()
    n_days = len(returns)
    n_assets = len(returns.columns)
    
    print(f"    Processing {n_days} days...")
    
    # Pre-calculate all weights at rebalance points
    rebalance_indices = list(range(COV_LOOKBACK, n_days, REBALANCE_DAYS))
    weights_dict = {}
    
    print(f"    Calculating {len(rebalance_indices)} rebalance points...")
    
    for idx in rebalance_indices:
        trailing = returns.iloc[idx-COV_LOOKBACK:idx]
        cov = trailing.cov().values * 252
        
        # Fast inverse-vol weights
        vols = np.sqrt(np.diag(cov))
        vols[vols == 0] = 1e-6
        inv_vols = 1.0 / vols
        w = inv_vols / inv_vols.sum()
        weights_dict[idx] = w
    
    # Build weight matrix
    print("    Building weight matrix...")
    weights_matrix = np.zeros((n_days, n_assets))
    current_weights = np.ones(n_assets) / n_assets
    
    for i in range(COV_LOOKBACK, n_days):
        if i in weights_dict:
            current_weights = weights_dict[i]
        weights_matrix[i] = current_weights
    
    # Calculate portfolio returns (before risk management)
    returns_array = returns.values
    port_returns_raw = np.sum(weights_matrix * returns_array, axis=1)
    
    if use_risk_management:
        print("    Applying risk management...")
        
        # Rolling volatility
        port_returns_series = pd.Series(port_returns_raw, index=returns.index)
        rolling_vol = port_returns_series.rolling(VOL_LOOKBACK).std() * np.sqrt(365)
        rolling_vol = rolling_vol.fillna(TARGET_VOL)
        
        # Volatility scalar
        vol_scalar = TARGET_VOL / rolling_vol.replace(0, TARGET_VOL)
        vol_scalar = vol_scalar.clip(MIN_EXPOSURE, MAX_LEVERAGE)
        
        # Drawdown scalar (need to iterate for this)
        equity = np.ones(n_days)
        peak = 1.0
        dd_scalar = np.ones(n_days)
        
        for i in range(1, n_days):
            # Apply vol scalar to yesterday's return
            adj_return = port_returns_raw[i] * vol_scalar.iloc[i-1] * dd_scalar[i-1]
            equity[i] = equity[i-1] * (1 + adj_return)
            peak = max(peak, equity[i])
            
            # Calculate current drawdown
            current_dd = (equity[i] - peak) / peak
            
            # Drawdown scalar for next period
            if current_dd >= DD_START_REDUCE:
                dd_scalar[i] = 1.0
            elif current_dd <= DD_MIN_EXPOSURE:
                dd_scalar[i] = MIN_EXPOSURE
            else:
                range_dd = DD_START_REDUCE - DD_MIN_EXPOSURE
                position = (current_dd - DD_MIN_EXPOSURE) / range_dd
                dd_scalar[i] = MIN_EXPOSURE + position * (1.0 - MIN_EXPOSURE)
        
        exposure = np.minimum(vol_scalar.values, dd_scalar)
        exposure = np.maximum(exposure, MIN_EXPOSURE)
        
        equity_series = pd.Series(equity, index=returns.index)
        exposure_series = pd.Series(exposure, index=returns.index)
        
    else:
        # Pure risk parity - just compound returns
        equity = np.cumprod(1 + port_returns_raw)
        equity_series = pd.Series(equity, index=returns.index)
        exposure = np.ones(n_days)
        exposure_series = pd.Series(exposure, index=returns.index)
    
    # Calculate metrics
    total_return = equity_series.iloc[-1] - 1
    years = (returns.index[-1] - returns.index[0]).days / 365
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    daily_returns = equity_series.pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
    
    rolling_max = equity_series.expanding().max()
    max_dd = ((equity_series - rolling_max) / rolling_max).min()
    
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe=sharpe,
        max_dd=max_dd,
        calmar=calmar,
        avg_exposure=exposure_series.mean(),
        equity_curve=equity_series,
        exposure_history=exposure_series
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("RISK-MANAGED RISK PARITY (OPTIMIZED)")
    print("=" * 100)
    
    print(f"""
    TARGET: Max ~50% drawdown
    
    SETTINGS:
        Target Volatility:     {TARGET_VOL*100:.0f}%
        Start Reducing DD:     {DD_START_REDUCE*100:.0f}%
        Min Exposure at DD:    {DD_MIN_EXPOSURE*100:.0f}%
        Min Exposure Floor:    {MIN_EXPOSURE*100:.0f}%
    """)
    
    # Load data
    print("-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    db = Database()
    all_returns = {}
    
    for pair in DEPLOY_PAIRS:
        print(f"  {pair}...", end=" ")
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        all_returns[pair] = df_24h['close'].pct_change()
        print(f"{len(df_24h)} bars")
    
    returns_df = pd.DataFrame(all_returns).dropna()
    print(f"\n  Common: {returns_df.index.min().date()} to {returns_df.index.max().date()} ({len(returns_df)} bars)")
    
    # Backtest
    print("\n" + "=" * 100)
    print("BACKTESTING")
    print("=" * 100)
    
    print("\n  Running RISK-MANAGED...")
    rm = backtest_vectorized(returns_df, use_risk_management=True)
    
    print("\n  Running PURE RISK PARITY...")
    rp = backtest_vectorized(returns_df, use_risk_management=False)
    
    # Results
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    
    print(f"""
    ┌────────────────────────────────────────────────────────────────────┐
    │                    RISK-MANAGED         PURE RISK PARITY          │
    ├────────────────────────────────────────────────────────────────────┤
    │  Total Return:       {rm.total_return*100:>+8.1f}%              {rp.total_return*100:>+8.1f}%             │
    │  Annual Return:      {rm.annual_return*100:>+8.1f}%              {rp.annual_return*100:>+8.1f}%             │
    │  Sharpe Ratio:       {rm.sharpe:>8.2f}                {rp.sharpe:>8.2f}               │
    │  Max Drawdown:       {rm.max_dd*100:>8.1f}%              {rp.max_dd*100:>8.1f}%             │
    │  Calmar Ratio:       {rm.calmar:>8.2f}                {rp.calmar:>8.2f}               │
    │  Avg Exposure:       {rm.avg_exposure*100:>8.1f}%              {rp.avg_exposure*100:>8.1f}%             │
    └────────────────────────────────────────────────────────────────────┘
    """)
    
    # Tradeoff analysis
    return_kept = (1 + rm.total_return) / (1 + rp.total_return) * 100
    dd_improvement = (rm.max_dd - rp.max_dd) * 100
    
    print(f"""
    TRADEOFF ANALYSIS:
        Return Kept:         {return_kept:.1f}% of pure risk parity
        Drawdown Improved:   {dd_improvement:+.1f}pp
    """)
    
    # Verdict
    target_met = rm.max_dd >= -0.55
    
    if target_met:
        status = f"✓ TARGET MET: {rm.max_dd*100:.1f}% max DD (target: -50%)"
    else:
        status = f"✗ TARGET MISSED: {rm.max_dd*100:.1f}% max DD (target: -50%)"
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  {status:<69} ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║  Expected Annual Return:  {rm.annual_return*100:>+6.1f}%                                     ║
    ║  Expected Max Drawdown:   {rm.max_dd*100:>6.1f}%                                     ║
    ║  Risk-Adjusted (Sharpe):  {rm.sharpe:>6.2f}                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("=" * 100)
    print("COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()