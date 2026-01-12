#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Portfolio Optimization: 8-State Multi-Asset
====================================================
Combines multiple optimization techniques:

1. REGIME-SPECIFIC CORRELATIONS
   - Calculate correlations separately for bull/bear/sideways
   - Use appropriate correlation matrix based on current regime

2. RISK PARITY BASE
   - Equal risk contribution from each asset
   - Robust, doesn't require return estimates

3. KELLY OVERLAY
   - Adjust weights based on 8-state edge per asset
   - Higher edge = higher allocation

4. CORRELATION PENALTY
   - Reduce exposure when cross-asset correlations spike
   - Protects against crash correlation = 1

DEPLOYMENT PAIRS:
    XLMUSD, ZECUSD, ETCUSD, ETHUSD, XMRUSD, ADAUSD

Usage:
    python portfolio_optimization.py
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
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

# 8-State parameters (locked from XBTUSD)
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005

# Portfolio settings
DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']
MIN_HISTORY_DAYS = 365
MIN_SAMPLES_PER_PERM = 20
HIT_RATE_THRESHOLD = 0.50

# Kelly settings
KELLY_FRACTION = 0.25  # Quarter Kelly for safety

# Correlation penalty settings
CORR_THRESHOLD = 0.7   # Start penalizing above this
CORR_PENALTY_SCALE = 0.5  # Reduce exposure by this factor at corr=1

# Trading costs
TOTAL_COST = 0.0015

# Rebalancing
REBALANCE_DAYS = 30  # Rebalance monthly


# =============================================================================
# CORE FUNCTIONS (8-STATE)
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


def calculate_hit_rates(returns: pd.Series, signals: pd.DataFrame,
                        end_idx: int = None) -> Dict[Tuple[int, int, int], Dict]:
    """Calculate hit rates for 8-state strategy."""
    all_perms = list(product([0, 1], repeat=3))
    
    # Align returns to signals index
    common_idx = returns.index.intersection(signals.index)
    aligned_returns = returns.loc[common_idx]
    aligned_signals = signals.loc[common_idx]
    
    if end_idx is not None:
        aligned_signals = aligned_signals.iloc[:end_idx]
        aligned_returns = aligned_returns.iloc[:end_idx]
    
    # Forward returns
    forward_returns = aligned_returns.shift(-1)
    
    hit_rates = {}
    
    for perm in all_perms:
        mask = (
            (aligned_signals['trend_24h'] == perm[0]) &
            (aligned_signals['trend_72h'] == perm[1]) &
            (aligned_signals['trend_168h'] == perm[2])
        )
        
        perm_returns = forward_returns[mask].dropna()
        n = len(perm_returns)
        
        if n > 0:
            n_wins = (perm_returns > 0).sum()
            hit_rate = n_wins / n
            avg_win = perm_returns[perm_returns > 0].mean() if n_wins > 0 else 0
            avg_loss = abs(perm_returns[perm_returns < 0].mean()) if (n - n_wins) > 0 else 0
        else:
            hit_rate, n_wins, avg_win, avg_loss = 0.5, 0, 0.02, 0.02
        
        hit_rates[perm] = {
            'n': n, 'hit_rate': hit_rate,
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'sufficient': n >= MIN_SAMPLES_PER_PERM,
        }
    
    return hit_rates


def get_8state_position(perm: Tuple[int, int, int],
                        hit_rates: Dict[Tuple[int, int, int], Dict]) -> float:
    """Get base position for 8-state."""
    data = hit_rates[perm]
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.00


def get_kelly_fraction(perm: Tuple[int, int, int],
                       hit_rates: Dict[Tuple[int, int, int], Dict]) -> float:
    """Calculate Kelly fraction for a permutation."""
    data = hit_rates[perm]
    
    if not data['sufficient'] or data['avg_loss'] == 0:
        return 0.5  # Neutral
    
    p = data['hit_rate']
    b = data['avg_win'] / data['avg_loss'] if data['avg_loss'] > 0 else 1
    
    # Kelly formula: f = (p*b - (1-p)) / b
    kelly = (p * b - (1 - p)) / b if b > 0 else 0
    
    # Clamp to [0, 1] and apply fraction
    kelly = max(0, min(1, kelly)) * KELLY_FRACTION
    
    return kelly


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime(returns_30d: float) -> str:
    """Detect market regime based on trailing 30-day returns."""
    if returns_30d > 0.20:
        return 'BULL'
    elif returns_30d < -0.20:
        return 'BEAR'
    else:
        return 'SIDEWAYS'


def calculate_regime_correlations(returns_df: pd.DataFrame, 
                                   lookback: int = 90) -> Dict[str, pd.DataFrame]:
    """Calculate correlation matrices for different regimes."""
    
    # Calculate 30-day rolling returns for regime detection (use BTC as reference)
    if 'XBTUSD' in returns_df.columns:
        reference = returns_df['XBTUSD']
    else:
        reference = returns_df.mean(axis=1)
    
    rolling_30d = reference.rolling(30).sum()
    
    # Categorize each day into regime
    regimes = rolling_30d.apply(lambda x: 'BULL' if x > 0.20 else ('BEAR' if x < -0.20 else 'SIDEWAYS'))
    
    corr_matrices = {}
    
    for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
        regime_mask = regimes == regime
        regime_returns = returns_df[regime_mask]
        
        if len(regime_returns) > 30:
            corr_matrices[regime] = regime_returns.corr()
        else:
            corr_matrices[regime] = returns_df.corr()  # Fallback to full sample
    
    return corr_matrices


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

def risk_parity_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Calculate risk parity weights (equal risk contribution).
    
    Each asset contributes equally to total portfolio variance.
    """
    n = len(cov_matrix)
    
    def risk_budget_objective(weights, cov):
        port_var = weights @ cov @ weights
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk
        target_risk = port_var / n
        return np.sum((risk_contrib - target_risk) ** 2)
    
    # Initial guess: equal weight
    w0 = np.ones(n) / n
    
    # Constraints: weights sum to 1, all positive
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.5) for _ in range(n)]  # Min 1%, max 50% per asset
    
    result = minimize(
        risk_budget_objective,
        w0,
        args=(cov_matrix.values,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    weights = pd.Series(result.x, index=cov_matrix.columns)
    return weights / weights.sum()  # Normalize


def minimum_variance_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    """Calculate minimum variance portfolio weights."""
    n = len(cov_matrix)
    
    def portfolio_variance(weights, cov):
        return weights @ cov @ weights
    
    w0 = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.5) for _ in range(n)]
    
    result = minimize(
        portfolio_variance,
        w0,
        args=(cov_matrix.values,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    weights = pd.Series(result.x, index=cov_matrix.columns)
    return weights / weights.sum()


def calculate_correlation_penalty(corr_matrix: pd.DataFrame) -> float:
    """
    Calculate penalty factor based on average pairwise correlation.
    Higher correlation = lower exposure.
    """
    # Get upper triangle (exclude diagonal)
    n = len(corr_matrix)
    upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
    avg_corr = np.mean(upper_tri)
    
    if avg_corr <= CORR_THRESHOLD:
        return 1.0  # No penalty
    
    # Linear penalty from threshold to 1
    penalty = 1 - CORR_PENALTY_SCALE * (avg_corr - CORR_THRESHOLD) / (1 - CORR_THRESHOLD)
    return max(0.3, penalty)  # Floor at 30% exposure


# =============================================================================
# HYBRID STRATEGY
# =============================================================================

@dataclass
class PortfolioState:
    """Current portfolio state."""
    date: pd.Timestamp
    weights: Dict[str, float]
    positions: Dict[str, float]  # 8-state positions per asset
    kelly_fractions: Dict[str, float]
    regime: str
    correlation_penalty: float
    total_exposure: float


def calculate_hybrid_weights(
    base_weights: pd.Series,
    positions_8state: Dict[str, float],
    kelly_fractions: Dict[str, float],
    correlation_penalty: float
) -> pd.Series:
    """
    Calculate final hybrid weights combining all factors.
    
    Formula:
        weight[i] = base_weight[i] * position_8state[i] * (1 + kelly_adj[i]) * corr_penalty
        
    Where:
        - base_weight: from risk parity
        - position_8state: 0, 0.5, or 1 based on current state
        - kelly_adj: adjustment based on edge
        - corr_penalty: global reduction when correlations high
    """
    hybrid = pd.Series(index=base_weights.index, dtype=float)
    
    for asset in base_weights.index:
        base = base_weights[asset]
        pos_8state = positions_8state.get(asset, 0.5)
        kelly = kelly_fractions.get(asset, 0.5)
        
        # Kelly adjustment: scale from 0.5 (neutral) to 1.5 (max edge)
        kelly_multiplier = 0.5 + kelly  # Range: 0.5 to ~0.75 with quarter Kelly
        
        # Combine
        weight = base * pos_8state * kelly_multiplier * correlation_penalty
        hybrid[asset] = weight
    
    # Normalize to sum to correlation_penalty (allows cash position)
    total = hybrid.sum()
    if total > 0:
        hybrid = hybrid / total * correlation_penalty
    
    return hybrid


# =============================================================================
# BACKTESTING
# =============================================================================

@dataclass 
class BacktestResult:
    """Portfolio backtest results."""
    total_return: float
    annual_return: float
    sharpe: float
    max_dd: float
    calmar: float
    avg_exposure: float
    n_rebalances: int
    equity_curve: pd.Series
    weights_history: pd.DataFrame


def backtest_portfolio(
    returns_df: pd.DataFrame,
    signals_dict: Dict[str, pd.DataFrame],
    all_hit_rates: Dict[str, Dict],
    strategy: str = 'hybrid'
) -> BacktestResult:
    """
    Backtest portfolio strategy.
    
    Strategies:
        - 'hybrid': Full hybrid approach
        - 'risk_parity': Risk parity only (no 8-state)
        - 'equal_weight': Simple equal weight
        - 'equal_8state': Equal weight with 8-state overlay
    """
    
    # Align all returns to common index
    common_idx = returns_df.dropna().index
    returns = returns_df.loc[common_idx]
    
    # Initialize
    equity = 1.0
    equity_curve = [equity]
    equity_dates = [common_idx[0]]
    
    weights_history = []
    last_rebalance = common_idx[0]
    
    # Calculate regime correlations
    regime_corr = calculate_regime_correlations(returns)
    
    # Rolling covariance for risk parity (60-day)
    cov_lookback = 60
    
    n_rebalances = 0
    total_exposure = []
    
    for i in range(cov_lookback, len(common_idx)):
        date = common_idx[i]
        
        # Check if rebalance needed
        days_since_rebalance = (date - last_rebalance).days
        should_rebalance = days_since_rebalance >= REBALANCE_DAYS or i == cov_lookback
        
        if should_rebalance:
            # Calculate trailing covariance
            trailing_returns = returns.iloc[i-cov_lookback:i]
            cov_matrix = trailing_returns.cov() * 252  # Annualize
            
            # Detect current regime
            trailing_30d = returns.iloc[i-30:i].sum()
            avg_return = trailing_30d.mean()
            regime = detect_regime(avg_return)
            
            # Get regime-appropriate correlation matrix
            corr_matrix = regime_corr.get(regime, trailing_returns.corr())
            
            # Calculate correlation penalty
            corr_penalty = calculate_correlation_penalty(corr_matrix)
            
            if strategy == 'hybrid':
                # Base weights from risk parity
                base_weights = risk_parity_weights(cov_matrix)
                
                # Get 8-state positions and Kelly fractions for each asset
                positions_8state = {}
                kelly_fractions = {}
                
                for asset in DEPLOY_PAIRS:
                    if asset not in signals_dict:
                        continue
                    
                    signals = signals_dict[asset]
                    hit_rates = all_hit_rates[asset]
                    
                    # Get signal at this date (find nearest prior)
                    try:
                        prior_signals = signals[signals.index <= date]
                        if len(prior_signals) > 0:
                            sig = prior_signals.iloc[-1]
                            perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
                        else:
                            perm = (1, 1, 1)  # Default to all up
                    except Exception:
                        perm = (1, 1, 1)  # Default on error
                    
                    positions_8state[asset] = get_8state_position(perm, hit_rates)
                    kelly_fractions[asset] = get_kelly_fraction(perm, hit_rates)
                
                # Calculate hybrid weights
                weights = calculate_hybrid_weights(base_weights, positions_8state, 
                                                   kelly_fractions, corr_penalty)
                
            elif strategy == 'risk_parity':
                weights = risk_parity_weights(cov_matrix)
                
            elif strategy == 'equal_weight':
                n = len(cov_matrix)
                weights = pd.Series(1/n, index=cov_matrix.columns)
                
            elif strategy == 'equal_8state':
                n = len(cov_matrix)
                base_weights = pd.Series(1/n, index=cov_matrix.columns)
                
                positions_8state = {}
                for asset in DEPLOY_PAIRS:
                    if asset not in signals_dict:
                        continue
                    signals = signals_dict[asset]
                    hit_rates = all_hit_rates[asset]
                    
                    try:
                        prior_signals = signals[signals.index <= date]
                        if len(prior_signals) > 0:
                            sig = prior_signals.iloc[-1]
                            perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
                        else:
                            perm = (1, 1, 1)
                    except Exception:
                        perm = (1, 1, 1)
                    
                    positions_8state[asset] = get_8state_position(perm, hit_rates)
                
                # Apply 8-state overlay
                weights = base_weights.copy()
                for asset in weights.index:
                    weights[asset] *= positions_8state.get(asset, 0.5)
                
                # Normalize
                if weights.sum() > 0:
                    weights = weights / weights.sum() * 0.8  # Cap at 80% exposure
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            last_rebalance = date
            n_rebalances += 1
            
            weights_history.append({
                'date': date,
                'regime': regime if strategy == 'hybrid' else 'N/A',
                'corr_penalty': corr_penalty if strategy == 'hybrid' else 1.0,
                **{f'w_{a}': weights.get(a, 0) for a in DEPLOY_PAIRS}
            })
        
        # Calculate portfolio return
        daily_return = returns.loc[date]
        port_return = sum(weights.get(a, 0) * daily_return.get(a, 0) for a in DEPLOY_PAIRS)
        
        # Transaction costs on rebalance
        if should_rebalance and i > cov_lookback:
            turnover = sum(abs(weights.get(a, 0) - prev_weights.get(a, 0)) for a in DEPLOY_PAIRS)
            port_return -= turnover * TOTAL_COST
        
        prev_weights = weights.copy()
        
        # Update equity
        equity *= (1 + port_return)
        equity_curve.append(equity)
        equity_dates.append(date)
        total_exposure.append(weights.sum())
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=equity_dates)
    daily_returns = equity_series.pct_change().dropna()
    
    total_return = equity_series.iloc[-1] - 1
    years = (equity_dates[-1] - equity_dates[0]).days / 365
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
    
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe=sharpe,
        max_dd=max_dd,
        calmar=calmar,
        avg_exposure=np.mean(total_exposure),
        n_rebalances=n_rebalances,
        equity_curve=equity_series,
        weights_history=pd.DataFrame(weights_history)
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("HYBRID PORTFOLIO OPTIMIZATION: 8-STATE MULTI-ASSET")
    print("=" * 100)
    
    print(f"""
    COMPONENTS:
        1. Risk Parity base weights
        2. 8-State position overlay
        3. Kelly fraction adjustment
        4. Correlation penalty
        
    PAIRS: {', '.join(DEPLOY_PAIRS)}
    
    SETTINGS:
        Kelly Fraction:     {KELLY_FRACTION}
        Corr Threshold:     {CORR_THRESHOLD}
        Rebalance Period:   {REBALANCE_DAYS} days
    """)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    db = Database()
    
    all_returns = {}
    all_signals = {}
    all_hit_rates = {}
    
    for pair in DEPLOY_PAIRS:
        print(f"\n  Loading {pair}...")
        
        df_1h = db.get_ohlcv(pair)
        print(f"    {len(df_1h):,} 1h bars: {df_1h.index.min().date()} to {df_1h.index.max().date()}")
        
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        # Generate signals
        signals = generate_signals(df_24h, df_72h, df_168h)
        
        # Calculate returns
        returns = df_24h['close'].pct_change()
        
        # Calculate hit rates (full history for now)
        hit_rates = calculate_hit_rates(returns, signals)
        
        all_returns[pair] = returns
        all_signals[pair] = signals
        all_hit_rates[pair] = hit_rates
    
    # Create aligned returns DataFrame
    returns_df = pd.DataFrame(all_returns)
    
    # Drop rows with any NaN (ensures all assets have data)
    returns_df = returns_df.dropna()
    print(f"\n  Common date range: {returns_df.index.min().date()} to {returns_df.index.max().date()}")
    print(f"  Total bars: {len(returns_df)}")
    
    # =========================================================================
    # BACKTEST STRATEGIES
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("BACKTESTING STRATEGIES")
    print("=" * 100)
    
    strategies = ['hybrid', 'equal_8state', 'risk_parity', 'equal_weight']
    results = {}
    
    for strategy in strategies:
        print(f"\n  Running {strategy}...")
        result = backtest_portfolio(returns_df, all_signals, all_hit_rates, strategy)
        results[strategy] = result
        print(f"    Return: {result.total_return*100:+.1f}% | Sharpe: {result.sharpe:.2f} | MaxDD: {result.max_dd*100:.1f}%")
    
    # =========================================================================
    # BUY & HOLD BENCHMARK
    # =========================================================================
    
    print("\n  Calculating Buy & Hold benchmark...")
    bh_equity = (1 + returns_df.mean(axis=1)).cumprod()
    bh_return = bh_equity.iloc[-1] - 1
    bh_returns_daily = returns_df.mean(axis=1)
    bh_sharpe = bh_returns_daily.mean() / bh_returns_daily.std() * np.sqrt(365)
    bh_rolling_max = bh_equity.expanding().max()
    bh_max_dd = ((bh_equity - bh_rolling_max) / bh_rolling_max).min()
    
    print(f"    Return: {bh_return*100:+.1f}% | Sharpe: {bh_sharpe:.2f} | MaxDD: {bh_max_dd*100:.1f}%")
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Strategy':<20} {'Return':>12} {'Annual':>10} {'Sharpe':>8} {'MaxDD':>10} {'Calmar':>8} {'Exposure':>10}")
    print("-" * 90)
    
    for strategy, r in results.items():
        print(f"{strategy:<20} {r.total_return*100:>+11.1f}% {r.annual_return*100:>+9.1f}% "
              f"{r.sharpe:>8.2f} {r.max_dd*100:>9.1f}% {r.calmar:>8.2f} {r.avg_exposure*100:>9.1f}%")
    
    print(f"{'B&H (equal)':<20} {bh_return*100:>+11.1f}% {'-':>10} "
          f"{bh_sharpe:>8.2f} {bh_max_dd*100:>9.1f}% {'-':>8} {'100.0%':>10}")
    
    # =========================================================================
    # HYBRID STRATEGY DETAILS
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("HYBRID STRATEGY DETAILS")
    print("=" * 100)
    
    hybrid_result = results['hybrid']
    weights_df = hybrid_result.weights_history
    
    if len(weights_df) > 0:
        print("\n  AVERAGE WEIGHTS:")
        for pair in DEPLOY_PAIRS:
            col = f'w_{pair}'
            if col in weights_df.columns:
                avg_weight = weights_df[col].mean()
                print(f"    {pair}: {avg_weight*100:.1f}%")
        
        print(f"\n  REGIME DISTRIBUTION:")
        if 'regime' in weights_df.columns:
            regime_counts = weights_df['regime'].value_counts()
            for regime, count in regime_counts.items():
                print(f"    {regime}: {count} ({100*count/len(weights_df):.1f}%)")
        
        print(f"\n  CORRELATION PENALTY:")
        if 'corr_penalty' in weights_df.columns:
            print(f"    Mean:    {weights_df['corr_penalty'].mean():.2f}")
            print(f"    Min:     {weights_df['corr_penalty'].min():.2f}")
            print(f"    Max:     {weights_df['corr_penalty'].max():.2f}")
    
    # =========================================================================
    # VERDICT
    # =========================================================================
    
    hybrid_vs_bh = (1 + hybrid_result.total_return) / (1 + bh_return)
    sharpe_improvement = hybrid_result.sharpe - bh_sharpe
    dd_improvement = hybrid_result.max_dd - bh_max_dd
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  HYBRID STRATEGY VERDICT                                                                              ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                       ║
    ║  vs Buy & Hold:                                                                                       ║
    ║    Return Ratio:      {hybrid_vs_bh:>6.2f}x                                                                     ║
    ║    Sharpe Diff:       {sharpe_improvement:>+6.2f}                                                                     ║
    ║    MaxDD Improvement: {dd_improvement*100:>+6.1f}pp                                                                   ║
    ║                                                                                                       ║
    ║  vs Equal Weight 8-State:                                                                             ║
    ║    Return Diff:       {(hybrid_result.total_return - results['equal_8state'].total_return)*100:>+6.1f}%                                                                    ║
    ║    Sharpe Diff:       {hybrid_result.sharpe - results['equal_8state'].sharpe:>+6.2f}                                                                     ║
    ║                                                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()