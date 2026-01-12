#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directional 32-State Strategy Backtest
======================================
Standalone backtest for the directional 32-state trading strategy.

Features:
- 32-state regime classification (8 price states × 4 MA alignment)
- Expanding window hit rate calculation (no look-ahead bias)
- Risk parity asset weighting
- Volatility scaling (target 40% annual vol)
- Drawdown protection
- Transaction costs

Documented Performance: +66% annual, 1.99 Sharpe, -19% max DD

Usage:
    python directional_backtest.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked from validation)
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005

# Hit rate parameters
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_STATE = 20
MIN_TRAINING_DAYS = 365
HIT_RATE_RECALC_DAYS = 30

# Risk management
TARGET_VOL = 0.40
VOL_LOOKBACK = 30
DD_START_REDUCE = -0.20
DD_MIN_EXPOSURE = -0.50
MIN_EXPOSURE_FLOOR = 0.40
MAX_LEVERAGE = 1.0
COV_LOOKBACK = 60
REBALANCE_DAYS = 30

# Backtest parameters
INITIAL_CAPITAL = 100000.0
TRADING_FEE = 0.0010
SLIPPAGE = 0.0005
TOTAL_COST = TRADING_FEE + SLIPPAGE


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestResult:
    """Results from backtest."""
    equity_curve: pd.Series
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    realized_vol: float
    trading_costs: float
    n_days: int


# =============================================================================
# SIGNAL GENERATION
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


def generate_32state_signals(df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                              df_168h: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 32-state signals.
    
    8 price states × 4 MA alignment states = 32 total states
    SHIFT BEFORE REINDEX to prevent look-ahead bias.
    """
    # Price vs MA signals
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    # MA values for alignment
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Align to 24h index
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    # Build aligned DataFrame - SHIFT BEFORE REINDEX (critical)
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


def calculate_expanding_hit_rates(returns_history: pd.Series, 
                                   signals_history: pd.DataFrame) -> Dict:
    """
    Calculate 32-state hit rates using ONLY historical data.
    No look-ahead bias.
    """
    all_price_perms = list(product([0, 1], repeat=3))
    all_ma_perms = list(product([0, 1], repeat=2))
    
    if len(returns_history) < MIN_SAMPLES_PER_STATE:
        return {(p, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for p in all_price_perms for m in all_ma_perms}
    
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < MIN_SAMPLES_PER_STATE:
        return {(p, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for p in all_price_perms for m in all_ma_perms}
    
    aligned_returns = returns_history.loc[common_idx]
    aligned_signals = signals_history.loc[common_idx]
    
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]
    
    hit_rates = {}
    
    for price_perm in all_price_perms:
        for ma_perm in all_ma_perms:
            mask = (
                (aligned_signals['trend_24h'] == price_perm[0]) &
                (aligned_signals['trend_72h'] == price_perm[1]) &
                (aligned_signals['trend_168h'] == price_perm[2]) &
                (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
                (aligned_signals['ma168_above_ma24'] == ma_perm[1])
            )
            
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            
            if n > 0:
                hit_rate = (perm_returns > 0).sum() / n
            else:
                hit_rate = 0.5
            
            hit_rates[(price_perm, ma_perm)] = {
                'n': n,
                'hit_rate': hit_rate,
                'sufficient': n >= MIN_SAMPLES_PER_STATE,
            }
    
    return hit_rates


def get_32state_position(price_perm: Tuple[int, int, int], 
                         ma_perm: Tuple[int, int],
                         hit_rates: Dict) -> float:
    """
    Get position signal for 32-state.
    
    Returns:
        1.00: INVEST (hit rate > threshold and sufficient samples)
        0.50: SKIP (insufficient samples)
        0.00: AVOID (hit rate <= threshold)
    """
    key = (price_perm, ma_perm)
    data = hit_rates.get(key, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        return 0.50  # Skip
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00  # Invest
    else:
        return 0.00  # Avoid


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database) -> Dict:
    """Load and prepare all data."""
    data = {}
    
    print("\n  Loading data...", flush=True)
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ", flush=True)
        
        df_1h = db.get_ohlcv(pair)
        df_24h = resample_ohlcv(df_1h, '24h')
        df_72h = resample_ohlcv(df_1h, '72h')
        df_168h = resample_ohlcv(df_1h, '168h')
        
        signals = generate_32state_signals(df_24h, df_72h, df_168h)
        returns = df_24h['close'].pct_change()
        
        data[pair] = {
            'prices': df_24h,
            'returns': returns,
            'signals': signals,
        }
        
        print(f"{len(df_24h)} days", flush=True)
    
    print("  Data loading complete.", flush=True)
    return data


# =============================================================================
# HIT RATE HISTORY CALCULATION
# =============================================================================

def calculate_position_history(data: Dict) -> Tuple[List, Dict]:
    """
    Calculate position signals for all assets over time.
    Returns: dates list, positions dict
    """
    # Find common dates
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))
    
    all_dates = sorted(list(all_dates))
    
    print(f"\n  Calculating position history for {len(all_dates)} common dates...", flush=True)
    print(f"  (Recalculating hit rates every {HIT_RATE_RECALC_DAYS} days)", flush=True)
    
    positions = {pair: {} for pair in DEPLOY_PAIRS}
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_idx = {pair: -999 for pair in DEPLOY_PAIRS}
    
    last_progress = 0
    recalc_count = 0
    
    # Track state distribution
    invest_count = 0
    skip_count = 0
    avoid_count = 0
    
    for i, date in enumerate(all_dates):
        progress = int((i / len(all_dates)) * 100)
        if progress >= last_progress + 10:
            print(f"    Progress: {progress}%", flush=True)
            last_progress = progress
        
        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            
            if date not in signals.index:
                continue
            
            sig = signals.loc[date]
            price_perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
            ma_perm = (int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24']))
            
            returns_idx = returns.index.get_loc(date) if date in returns.index else None
            
            if returns_idx is None or returns_idx < MIN_TRAINING_DAYS:
                positions[pair][date] = 0.50  # Skip during warmup
                skip_count += 1
                continue
            
            # Recalculate hit rates periodically
            if returns_idx - last_recalc_idx[pair] >= HIT_RATE_RECALC_DAYS:
                hist_returns = returns.iloc[:returns_idx]
                hist_signals = signals[signals.index < date]
                hit_rate_cache[pair] = calculate_expanding_hit_rates(hist_returns, hist_signals)
                last_recalc_idx[pair] = returns_idx
                recalc_count += 1
            
            pos = get_32state_position(price_perm, ma_perm, hit_rate_cache[pair])
            positions[pair][date] = pos
            
            if pos == 1.0:
                invest_count += 1
            elif pos == 0.5:
                skip_count += 1
            else:
                avoid_count += 1
    
    print(f"    Progress: 100%", flush=True)
    print(f"    Total recalculations: {recalc_count}", flush=True)
    
    total_signals = invest_count + skip_count + avoid_count
    print(f"\n  Signal Distribution:", flush=True)
    print(f"    INVEST: {invest_count:>6} ({invest_count/total_signals*100:.1f}%)", flush=True)
    print(f"    SKIP:   {skip_count:>6} ({skip_count/total_signals*100:.1f}%)", flush=True)
    print(f"    AVOID:  {avoid_count:>6} ({avoid_count/total_signals*100:.1f}%)", flush=True)
    
    return all_dates, positions


# =============================================================================
# RISK MANAGEMENT
# =============================================================================

def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Calculate risk parity weights for assets (inverse volatility)."""
    vols = returns_df.std() * np.sqrt(365)
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return weights


# Note: Vol scaling and DD protection are now inlined in run_backtest()
# to match the original implementation exactly (using MARKET returns for vol scaling)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(data: Dict, positions: Dict, dates: List) -> BacktestResult:
    """Run the directional backtest."""
    
    print("\n  Running backtest...", flush=True)
    
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    equity_curve = {}
    
    # Risk parity weights for assets
    returns_dict = {pair: data[pair]['returns'] for pair in DEPLOY_PAIRS}
    returns_df = pd.DataFrame(returns_dict).dropna()
    asset_weights = pd.Series({pair: 1.0/len(DEPLOY_PAIRS) for pair in DEPLOY_PAIRS})
    
    prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    total_costs = 0.0
    last_rebalance_idx = 0
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        
        # Monthly rebalance of asset weights
        if i - last_rebalance_idx >= REBALANCE_DAYS:
            lookback_returns = returns_df.loc[:date].tail(COV_LOOKBACK)
            if len(lookback_returns) >= 30:
                asset_weights = calculate_risk_parity_weights(lookback_returns)
            last_rebalance_idx = i
        
        # Calculate base exposure from signals
        base_exposure = 0.0
        asset_exposures = {}
        
        for pair in DEPLOY_PAIRS:
            if date in positions[pair]:
                pos = positions[pair][date]
                asset_exposures[pair] = pos * asset_weights[pair]
                base_exposure += asset_exposures[pair]
        
        # Normalize base exposure
        if base_exposure > 1.0:
            for pair in asset_exposures:
                asset_exposures[pair] /= base_exposure
            base_exposure = 1.0
        
        # ─────────────────────────────────────────────────────────────
        # VOLATILITY SCALING - Use MARKET returns, not strategy returns
        # This matches the original implementation
        # ─────────────────────────────────────────────────────────────
        vol_scalar = 1.0
        if date in returns_df.index:
            idx = returns_df.index.get_loc(date)
            if idx >= VOL_LOOKBACK:
                # Use equal-weight MARKET returns (not strategy returns!)
                market_returns = returns_df.iloc[:idx].mean(axis=1)
                realized_vol = market_returns.iloc[-VOL_LOOKBACK:].std() * np.sqrt(365)
                if realized_vol > 0:
                    vol_scalar = TARGET_VOL / realized_vol
                    vol_scalar = np.clip(vol_scalar, MIN_EXPOSURE_FLOOR, MAX_LEVERAGE)
        
        # ─────────────────────────────────────────────────────────────
        # DRAWDOWN PROTECTION
        # Note: current_dd is NEGATIVE in original (equity - peak)/peak
        # ─────────────────────────────────────────────────────────────
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0  # NEGATIVE
        
        if current_dd >= DD_START_REDUCE:  # e.g., -0.10 >= -0.20
            dd_scalar = 1.0
        elif current_dd <= DD_MIN_EXPOSURE:  # e.g., -0.60 <= -0.50
            dd_scalar = MIN_EXPOSURE_FLOOR
        else:
            range_dd = DD_START_REDUCE - DD_MIN_EXPOSURE  # -0.20 - (-0.50) = 0.30
            position = (current_dd - DD_MIN_EXPOSURE) / range_dd
            dd_scalar = MIN_EXPOSURE_FLOOR + position * (1.0 - MIN_EXPOSURE_FLOOR)
        
        # Apply risk scalars to exposure
        risk_scalar = min(vol_scalar, dd_scalar)
        risk_scalar = max(risk_scalar, MIN_EXPOSURE_FLOOR)
        
        for pair in asset_exposures:
            asset_exposures[pair] *= risk_scalar
        
        # Calculate returns with costs
        daily_return = 0.0
        daily_cost = 0.0
        
        for pair in DEPLOY_PAIRS:
            curr_exp = asset_exposures.get(pair, 0.0)
            prev_exp = prev_exposures.get(pair, 0.0)
            
            # Trading cost
            change = abs(curr_exp - prev_exp)
            if change > 0.01:
                daily_cost += change * equity * TOTAL_COST
            
            # Return
            if curr_exp > 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    daily_return += curr_exp * ret
            
            prev_exposures[pair] = curr_exp
        
        pnl = equity * daily_return - daily_cost
        equity += pnl
        total_costs += daily_cost
        equity_curve[date] = equity
        
        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
    
    print("    Complete.", flush=True)
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    returns_series = equity_series.pct_change().dropna()
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(365) if returns_series.std() > 0 else 0
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
    realized_vol = returns_series.std() * np.sqrt(365)
    
    return BacktestResult(
        equity_curve=equity_series,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        realized_vol=realized_vol,
        trading_costs=total_costs,
        n_days=len(dates),
    )


# =============================================================================
# BUY & HOLD BENCHMARK
# =============================================================================

def calculate_buy_and_hold(data: Dict, dates: List) -> pd.Series:
    """Calculate equal-weight buy & hold benchmark."""
    equity = INITIAL_CAPITAL
    weight = 1.0 / len(DEPLOY_PAIRS)
    equity_curve = {}
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        
        daily_return = 0.0
        for pair in DEPLOY_PAIRS:
            if next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    daily_return += weight * ret
        
        equity *= (1 + daily_return)
        equity_curve[date] = equity
    
    return pd.Series(equity_curve)


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def display_results(result: BacktestResult, bh_curve: pd.Series, dates: List):
    """Display backtest results."""
    
    # B&H metrics
    bh_total_return = (bh_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    bh_annual = (1 + bh_total_return) ** (1 / years) - 1
    
    bh_returns = bh_curve.pct_change().dropna()
    bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(365) if bh_returns.std() > 0 else 0
    
    bh_peak = bh_curve.expanding().max()
    bh_dd = ((bh_peak - bh_curve) / bh_peak).max()
    
    print("\n" + "=" * 80)
    print("DIRECTIONAL 32-STATE BACKTEST RESULTS")
    print("=" * 80)
    
    print(f"""
    ┌────────────────────────┬─────────────────┬─────────────────┐
    │  METRIC                │  STRATEGY       │  BUY & HOLD     │
    ├────────────────────────┼─────────────────┼─────────────────┤
    │  Total Return          │  {result.total_return*100:>+12.0f}%  │  {bh_total_return*100:>+12.0f}%  │
    │  Annual Return         │  {result.annual_return*100:>+12.1f}%  │  {bh_annual*100:>+12.1f}%  │
    │  Sharpe Ratio          │  {result.sharpe_ratio:>14.2f}  │  {bh_sharpe:>14.2f}  │
    │  Max Drawdown          │  {result.max_drawdown*100:>13.1f}%  │  {bh_dd*100:>13.1f}%  │
    │  Calmar Ratio          │  {result.calmar_ratio:>14.2f}  │  {bh_annual/bh_dd if bh_dd > 0 else 0:>14.2f}  │
    │  Realized Volatility   │  {result.realized_vol*100:>13.1f}%  │                 │
    ├────────────────────────┼─────────────────┼─────────────────┤
    │  Final Equity          │  ${result.equity_curve.iloc[-1]:>12,.0f}  │  ${bh_curve.iloc[-1]:>12,.0f}  │
    │  Trading Costs         │  ${result.trading_costs:>12,.0f}  │           $0    │
    │  Trading Days          │  {result.n_days:>14,}  │                 │
    └────────────────────────┴─────────────────┴─────────────────┘
    """)
    
    print("  CONFIGURATION:")
    print(f"    Assets:              {', '.join(DEPLOY_PAIRS)}")
    print(f"    Target Volatility:   {TARGET_VOL*100:.0f}%")
    print(f"    DD Protection Start: {abs(DD_START_REDUCE)*100:.0f}%")
    print(f"    Min Exposure Floor:  {MIN_EXPOSURE_FLOOR*100:.0f}%")


def plot_results(result: BacktestResult, bh_curve: pd.Series, filename: str = 'directional_backtest.png'):
    """Generate equity curve chart."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curves
        ax1 = axes[0]
        ax1.plot(result.equity_curve.index, result.equity_curve.values, 
                 label='Directional 32-State', color='#2C5282', linewidth=2)
        ax1.plot(bh_curve.index, bh_curve.values, 
                 label='Buy & Hold', color='#C53030', linewidth=2, linestyle='--')
        ax1.set_title('Directional 32-State Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Drawdown
        ax2 = axes[1]
        equity = result.equity_curve
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='#C53030', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='#C53030', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(top=0)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\n  Chart saved to: {filename}")
        plt.close()
        
    except ImportError:
        print("\n  (matplotlib not available - skipping chart)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80, flush=True)
    print("DIRECTIONAL 32-STATE STRATEGY BACKTEST", flush=True)
    print("=" * 80, flush=True)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  STRATEGY CONFIGURATION                                            ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  • 32-state regime classification                                  ║
    ║  • Expanding window hit rate calculation (no look-ahead)           ║
    ║  • Risk parity asset weighting                                     ║
    ║  • Volatility scaling (target 40% annual vol)                      ║
    ║  • Drawdown protection (-20% to -50%)                              ║
    ║  • Transaction costs: 0.15% per trade                              ║
    ╚════════════════════════════════════════════════════════════════════╝
    """, flush=True)
    
    print("  Connecting to database...", flush=True)
    db = Database()
    print("  Connected.", flush=True)
    
    # Load data
    data = load_all_data(db)
    
    # Calculate position history
    dates, positions = calculate_position_history(data)
    
    print(f"\n  Trading period: {dates[0].date()} to {dates[-1].date()}", flush=True)
    
    # Run backtest
    result = run_backtest(data, positions, dates)
    
    # Calculate benchmark
    print("\n  Calculating buy & hold benchmark...", flush=True)
    bh_curve = calculate_buy_and_hold(data, dates)
    
    # Display results
    display_results(result, bh_curve, dates)
    
    # Plot
    plot_results(result, bh_curve)
    
    print("\n" + "=" * 80, flush=True)
    print("BACKTEST COMPLETE", flush=True)
    print("=" * 80, flush=True)
    
    return result, bh_curve


if __name__ == "__main__":
    result, bh_curve = main()