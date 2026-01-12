#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap Validation for 32-State Strategy
============================================
Runs bootstrap resampling on the equity curve to establish
confidence intervals for key performance metrics.

This answers: "How confident are we in these results?"

Usage:
    python bootstrap_32state_validation.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')
sys.path.insert(0, 'D:/cryptobot_docker/scripts')

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

N_BOOTSTRAP = 10000        # Number of bootstrap samples
CONFIDENCE_LEVEL = 0.95   # 95% confidence interval
BLOCK_SIZE = 20           # Block size for block bootstrap (preserve autocorrelation)

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_PERM = 20

# Expanding Window
MIN_TRAINING_DAYS = 365
HIT_RATE_RECALC_DAYS = 30

# Risk Management
COV_LOOKBACK = 60
TARGET_VOL = 0.40
VOL_LOOKBACK = 30
DD_START_REDUCE = -0.20
DD_MIN_EXPOSURE = -0.50
MIN_EXPOSURE_FLOOR = 0.40
MAX_LEVERAGE = 1.0

# Trading Costs
TRADING_FEE = 0.0010
SLIPPAGE = 0.0005
TOTAL_COST_PER_TRADE = TRADING_FEE + SLIPPAGE


# =============================================================================
# BACKTESTER FUNCTIONS (from 32-state backtester)
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for cost optimization."""
    name: str
    portfolio_method: str = 'risk_parity'
    signal_persistence_days: int = 1
    max_position_change_per_day: float = 1.0
    min_position_change: float = 0.01
    rebalance_days: int = 30
    use_intermediate_positions: bool = False


@dataclass
class BacktestResult:
    """Backtest results."""
    config_name: str
    total_return: float
    annual_return: float
    sharpe: float
    max_dd: float
    calmar: float
    total_costs: float
    cost_drag: float
    total_trades: int
    avg_trades_per_month: float
    turnover: float
    equity_curve: pd.Series


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


def generate_32state_signals(df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                              df_168h: pd.DataFrame) -> pd.DataFrame:
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)
    
    return aligned.dropna().astype(int)


def calculate_expanding_hit_rates(returns_history: pd.Series, 
                                   signals_history: pd.DataFrame) -> Dict:
    all_price_perms = list(product([0, 1], repeat=3))
    all_ma_perms = list(product([0, 1], repeat=2))
    
    if len(returns_history) < MIN_SAMPLES_PER_PERM:
        return {(p, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False} 
                for p in all_price_perms for m in all_ma_perms}
    
    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < MIN_SAMPLES_PER_PERM:
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
                'sufficient': n >= MIN_SAMPLES_PER_PERM,
            }
    
    return hit_rates


def get_32state_position(price_perm: Tuple[int, int, int], 
                         ma_perm: Tuple[int, int],
                         hit_rates: Dict,
                         use_intermediate: bool = False) -> float:
    key = (price_perm, ma_perm)
    data = hit_rates.get(key, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.25 if use_intermediate else 0.00


def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    cov = returns_df.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return pd.Series(weights, index=returns_df.columns)


def calculate_vol_scalar(returns: pd.Series) -> float:
    if len(returns) < VOL_LOOKBACK:
        return 1.0
    realized_vol = returns.iloc[-VOL_LOOKBACK:].std() * np.sqrt(365)
    if realized_vol <= 0:
        return 1.0
    scalar = TARGET_VOL / realized_vol
    return np.clip(scalar, MIN_EXPOSURE_FLOOR, MAX_LEVERAGE)


def calculate_dd_scalar(current_dd: float) -> float:
    if current_dd >= DD_START_REDUCE:
        return 1.0
    elif current_dd <= DD_MIN_EXPOSURE:
        return MIN_EXPOSURE_FLOOR
    else:
        range_dd = DD_START_REDUCE - DD_MIN_EXPOSURE
        position = (current_dd - DD_MIN_EXPOSURE) / range_dd
        return MIN_EXPOSURE_FLOOR + position * (1.0 - MIN_EXPOSURE_FLOOR)


class Backtester32State:
    """Backtester using 32-state signal system with expanding window."""
    
    def __init__(self, config: OptimizationConfig, initial_capital: float = 100000.0):
        self.config = config
        self.initial_capital = initial_capital
        self.db = Database()
        
        self.prices: Dict[str, pd.DataFrame] = {}
        self.returns: Dict[str, pd.Series] = {}
        self.signals: Dict[str, pd.DataFrame] = {}
        
        self.cached_hit_rates: Dict[str, Dict] = {pair: {} for pair in DEPLOY_PAIRS}
        self.last_recalc_idx: Dict[str, int] = {pair: -999 for pair in DEPLOY_PAIRS}
        
        self.cash = initial_capital
        self.positions: Dict[str, float] = {pair: 0.0 for pair in DEPLOY_PAIRS}
        self.current_weights: Dict[str, float] = {pair: 0.0 for pair in DEPLOY_PAIRS}
        self.equity = initial_capital
        self.peak_equity = initial_capital
        
        self.signal_history: Dict[str, List[float]] = {pair: [] for pair in DEPLOY_PAIRS}
        
        self.total_costs = 0.0
        self.total_turnover = 0.0
        self.total_trades = 0
        
    def load_data(self):
        for pair in DEPLOY_PAIRS:
            df_1h = self.db.get_ohlcv(pair)
            df_24h = resample_ohlcv(df_1h, '24h')
            df_72h = resample_ohlcv(df_1h, '72h')
            df_168h = resample_ohlcv(df_1h, '168h')
            
            self.prices[pair] = df_24h
            self.returns[pair] = df_24h['close'].pct_change()
            self.signals[pair] = generate_32state_signals(df_24h, df_72h, df_168h)
        
        self.returns_df = pd.DataFrame(self.returns).dropna()
    
    def get_expanding_hit_rates(self, pair: str, current_idx: int) -> Dict:
        if current_idx - self.last_recalc_idx[pair] >= HIT_RATE_RECALC_DAYS:
            dates = self.returns_df.index.tolist()
            current_date = dates[current_idx]
            
            returns_history = self.returns[pair][self.returns[pair].index <= current_date]
            signals_history = self.signals[pair][self.signals[pair].index <= current_date]
            
            self.cached_hit_rates[pair] = calculate_expanding_hit_rates(
                returns_history, signals_history
            )
            self.last_recalc_idx[pair] = current_idx
        
        return self.cached_hit_rates[pair]
    
    def get_persistent_signal(self, pair: str, raw_signal: float) -> float:
        history = self.signal_history[pair]
        history.append(raw_signal)
        
        if len(history) > self.config.signal_persistence_days:
            history.pop(0)
        
        if len(history) < self.config.signal_persistence_days:
            return history[-1]
        
        if all(s == history[-1] for s in history):
            return history[-1]
        else:
            return history[0]
    
    def apply_position_smoothing(self, current: float, target: float) -> float:
        max_change = self.config.max_position_change_per_day
        diff = target - current
        if abs(diff) <= max_change:
            return target
        else:
            return current + np.sign(diff) * max_change
    
    def should_trade(self, current_weight: float, target_weight: float) -> bool:
        return abs(target_weight - current_weight) >= self.config.min_position_change
    
    def execute_trade(self, pair: str, target_value: float, current_value: float):
        trade_value = abs(target_value - current_value)
        if trade_value < 100:
            return
        cost = trade_value * TOTAL_COST_PER_TRADE
        self.positions[pair] = target_value
        self.cash -= (target_value - current_value) + cost
        self.total_costs += cost
        self.total_turnover += trade_value
        self.total_trades += 1
    
    def run(self) -> BacktestResult:
        self.load_data()
        
        dates = self.returns_df.index.tolist()
        n_days = len(dates)
        
        portfolio_weights = pd.Series(1/len(DEPLOY_PAIRS), index=DEPLOY_PAIRS)
        last_rebalance = None
        
        equity_history = []
        min_warmup = max(COV_LOOKBACK, VOL_LOOKBACK, MIN_TRAINING_DAYS)
        last_progress = 0
        
        for i, date in enumerate(dates):
            progress = int((i / n_days) * 100)
            if progress >= last_progress + 10:
                print(f"      Progress: {progress}%", end="")
                last_progress = progress
            
            if i < min_warmup:
                equity_history.append(self.equity)
                continue
            
            for pair in DEPLOY_PAIRS:
                if pair in self.returns_df.columns and self.positions[pair] > 0:
                    ret = self.returns_df.loc[date, pair]
                    self.positions[pair] *= (1 + ret)
            
            self.equity = self.cash + sum(self.positions.values())
            self.peak_equity = max(self.peak_equity, self.equity)
            current_dd = (self.equity - self.peak_equity) / self.peak_equity
            
            should_rebalance = (last_rebalance is None or 
                               (date - last_rebalance).days >= self.config.rebalance_days)
            
            if should_rebalance:
                trailing_returns = self.returns_df.iloc[i-COV_LOOKBACK:i]
                portfolio_weights = calculate_risk_parity_weights(trailing_returns)
                last_rebalance = date
            
            target_32state = {}
            
            for pair in DEPLOY_PAIRS:
                signals = self.signals[pair]
                prior_signals = signals[signals.index <= date]
                if len(prior_signals) > 0:
                    sig = prior_signals.iloc[-1]
                    price_perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
                    ma_perm = (int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24']))
                else:
                    price_perm = (1, 1, 1)
                    ma_perm = (0, 0)
                
                hit_rates = self.get_expanding_hit_rates(pair, i)
                raw_signal = get_32state_position(price_perm, ma_perm, hit_rates, 
                                                   self.config.use_intermediate_positions)
                persistent_signal = self.get_persistent_signal(pair, raw_signal)
                target_32state[pair] = persistent_signal
            
            port_returns = self.returns_df.iloc[:i+1].mean(axis=1)
            vol_scalar = calculate_vol_scalar(port_returns)
            dd_scalar = calculate_dd_scalar(current_dd)
            exposure = max(min(vol_scalar, dd_scalar), MIN_EXPOSURE_FLOOR)
            
            target_weights = {}
            for pair in DEPLOY_PAIRS:
                target_weights[pair] = portfolio_weights[pair] * target_32state[pair] * exposure
            
            for pair in DEPLOY_PAIRS:
                current_weight = self.current_weights[pair]
                raw_target = target_weights[pair]
                smoothed_target = self.apply_position_smoothing(current_weight, raw_target)
                
                if self.should_trade(current_weight, smoothed_target):
                    current_value = self.positions[pair]
                    target_value = self.equity * smoothed_target
                    self.execute_trade(pair, target_value, current_value)
                    self.current_weights[pair] = smoothed_target
            
            self.equity = self.cash + sum(self.positions.values())
            equity_history.append(self.equity)
        
        print()
        
        equity_series = pd.Series(equity_history, index=dates)
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        years = n_days / 365
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        daily_returns = equity_series.pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
        
        rolling_max = equity_series.expanding().max()
        max_dd = ((equity_series - rolling_max) / rolling_max).min()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        cost_drag = (self.total_costs / self.initial_capital) / years if years > 0 else 0
        turnover = (self.total_turnover / self.initial_capital) / years if years > 0 else 0
        avg_trades_per_month = self.total_trades / (years * 12) if years > 0 else 0
        
        return BacktestResult(
            config_name=self.config.name,
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            max_dd=max_dd,
            calmar=calmar,
            total_costs=self.total_costs,
            cost_drag=cost_drag,
            total_trades=self.total_trades,
            avg_trades_per_month=avg_trades_per_month,
            turnover=turnover,
            equity_curve=equity_series
        )


# =============================================================================
# BOOTSTRAP FUNCTIONS
# =============================================================================

@dataclass
class BootstrapResult:
    """Bootstrap validation results."""
    metric: str
    point_estimate: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    
    
def block_bootstrap_returns(returns: pd.Series, block_size: int = 20) -> pd.Series:
    """
    Block bootstrap to preserve autocorrelation in returns.
    
    Instead of sampling individual days, we sample blocks of consecutive days.
    This preserves the time-series structure of volatility clustering.
    """
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    
    # Generate random block start indices
    max_start = n - block_size
    if max_start <= 0:
        # Not enough data for block bootstrap, fall back to regular
        indices = np.random.choice(n, size=n, replace=True)
    else:
        block_starts = np.random.randint(0, max_start, size=n_blocks)
        
        # Build bootstrapped series from blocks
        indices = []
        for start in block_starts:
            indices.extend(range(start, min(start + block_size, n)))
        indices = indices[:n]  # Trim to original length
    
    return returns.iloc[indices].reset_index(drop=True)


def calculate_metrics_from_returns(returns: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics from a return series."""
    
    # Build equity curve
    equity = (1 + returns).cumprod()
    
    # Total and annual return
    total_return = equity.iloc[-1] - 1
    years = len(returns) / 365
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Sharpe ratio
    sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
    
    # Max drawdown
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')
    
    return {
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': total_return,
    }


def run_bootstrap(equity_curve: pd.Series, n_bootstrap: int = 1000,
                  block_size: int = 20, confidence: float = 0.95) -> Dict[str, BootstrapResult]:
    """
    Run bootstrap validation on an equity curve.
    
    Returns confidence intervals for all key metrics.
    """
    # Get daily returns from equity curve
    returns = equity_curve.pct_change().dropna()
    
    # Point estimates from actual data
    point_estimates = calculate_metrics_from_returns(returns)
    
    # Bootstrap samples
    bootstrap_metrics = {metric: [] for metric in point_estimates.keys()}
    
    print(f"  Running {n_bootstrap} bootstrap samples...", end=" ")
    
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"{i+1}", end=" ")
        
        # Resample returns
        boot_returns = block_bootstrap_returns(returns, block_size)
        
        # Calculate metrics
        metrics = calculate_metrics_from_returns(boot_returns)
        
        for metric, value in metrics.items():
            bootstrap_metrics[metric].append(value)
    
    print("Done!")
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    results = {}
    
    for metric, values in bootstrap_metrics.items():
        values = np.array(values)
        
        results[metric] = BootstrapResult(
            metric=metric,
            point_estimate=point_estimates[metric],
            mean=np.mean(values),
            std=np.std(values),
            ci_lower=np.percentile(values, alpha/2 * 100),
            ci_upper=np.percentile(values, (1 - alpha/2) * 100),
        )
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("BOOTSTRAP VALIDATION: 32-STATE STRATEGY")
    print("=" * 100)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║  BOOTSTRAP VALIDATION                                                         ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  Method: Block Bootstrap (preserves autocorrelation)                          ║
    ║  Samples: {N_BOOTSTRAP:,}                                                             ║
    ║  Block Size: {BLOCK_SIZE} days                                                        ║
    ║  Confidence Level: {CONFIDENCE_LEVEL*100:.0f}%                                                      ║
    ║                                                                               ║
    ║  This establishes confidence intervals for performance metrics.               ║
    ║  Wider intervals = less certainty in results.                                 ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run the best configuration to get equity curve
    print("-" * 60)
    print("STEP 1: Running 32STATE_GRADUAL_RP backtest...")
    print("-" * 60)
    
    config = OptimizationConfig(
        name="32STATE_GRADUAL_RP",
        portfolio_method='risk_parity',
        signal_persistence_days=1,
        max_position_change_per_day=0.25,
        min_position_change=0.01,
        rebalance_days=30,
        use_intermediate_positions=False
    )
    
    backtester = Backtester32State(config)
    result = backtester.run()
    
    print(f"\n  Point Estimates:")
    print(f"    Annual Return: {result.annual_return*100:+.1f}%")
    print(f"    Sharpe Ratio:  {result.sharpe:.2f}")
    print(f"    Max Drawdown:  {result.max_dd*100:.1f}%")
    print(f"    Calmar Ratio:  {result.calmar:.2f}")
    
    # Run bootstrap
    print("\n" + "-" * 60)
    print("STEP 2: Running Bootstrap Validation...")
    print("-" * 60)
    
    bootstrap_results = run_bootstrap(
        result.equity_curve,
        n_bootstrap=N_BOOTSTRAP,
        block_size=BLOCK_SIZE,
        confidence=CONFIDENCE_LEVEL
    )
    
    # Display results
    print("\n" + "=" * 100)
    print(f"BOOTSTRAP RESULTS ({CONFIDENCE_LEVEL*100:.0f}% CONFIDENCE INTERVALS)")
    print("=" * 100)
    
    print(f"\n  {'Metric':<20} {'Point Est':>12} {'Mean':>12} {'Std':>10} {'CI Lower':>12} {'CI Upper':>12}")
    print("  " + "-" * 80)
    
    key_metrics = ['annual_return', 'sharpe', 'max_dd', 'calmar', 'win_rate', 'profit_factor']
    
    for metric in key_metrics:
        r = bootstrap_results[metric]
        
        if metric in ['annual_return', 'max_dd', 'win_rate']:
            print(f"  {metric:<20} {r.point_estimate*100:>+11.1f}% {r.mean*100:>+11.1f}% "
                  f"{r.std*100:>9.1f}% {r.ci_lower*100:>+11.1f}% {r.ci_upper*100:>+11.1f}%")
        else:
            print(f"  {metric:<20} {r.point_estimate:>12.2f} {r.mean:>12.2f} "
                  f"{r.std:>10.2f} {r.ci_lower:>12.2f} {r.ci_upper:>12.2f}")
    
    # Summary interpretation
    ar = bootstrap_results['annual_return']
    sr = bootstrap_results['sharpe']
    md = bootstrap_results['max_dd']
    ca = bootstrap_results['calmar']
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║  INTERPRETATION                                                               ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  ANNUAL RETURN:                                                               ║
    ║    95% CI: [{ar.ci_lower*100:+.1f}%, {ar.ci_upper*100:+.1f}%]                                               ║
    ║    We are 95% confident annual returns fall in this range.                    ║
    ║                                                                               ║
    ║  SHARPE RATIO:                                                                ║
    ║    95% CI: [{sr.ci_lower:.2f}, {sr.ci_upper:.2f}]                                                    ║
    ║    {"Excellent" if sr.ci_lower > 1.5 else "Good" if sr.ci_lower > 1.0 else "Acceptable" if sr.ci_lower > 0.5 else "Weak"}: Lower bound {">" if sr.ci_lower > 1.0 else "<"} 1.0 (institutional threshold)                       ║
    ║                                                                               ║
    ║  MAX DRAWDOWN:                                                                ║
    ║    95% CI: [{md.ci_lower*100:.1f}%, {md.ci_upper*100:.1f}%]                                              ║
    ║    Worst case drawdown could reach {md.ci_lower*100:.1f}%                                  ║
    ║                                                                               ║
    ║  CALMAR RATIO:                                                                ║
    ║    95% CI: [{ca.ci_lower:.2f}, {ca.ci_upper:.2f}]                                                    ║
    ║    {"Excellent" if ca.ci_lower > 2.0 else "Good" if ca.ci_lower > 1.0 else "Acceptable" if ca.ci_lower > 0.5 else "Weak"} risk-adjusted returns                                             ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Statistical significance
    print("\n" + "-" * 60)
    print("STATISTICAL SIGNIFICANCE")
    print("-" * 60)
    
    # Is Sharpe significantly > 0?
    sharpe_t_stat = sr.mean / sr.std if sr.std > 0 else 0
    sharpe_significant = sr.ci_lower > 0
    
    # Is annual return significantly > 0?
    return_significant = ar.ci_lower > 0
    
    print(f"""
    Sharpe > 0:     {"YES ✓" if sharpe_significant else "NO ✗"} (lower CI = {sr.ci_lower:.2f})
    Return > 0:     {"YES ✓" if return_significant else "NO ✗"} (lower CI = {ar.ci_lower*100:+.1f}%)
    Sharpe > 1.0:   {"YES ✓" if sr.ci_lower > 1.0 else "NO ✗"} (lower CI = {sr.ci_lower:.2f})
    Sharpe > 1.5:   {"YES ✓" if sr.ci_lower > 1.5 else "NO ✗"} (lower CI = {sr.ci_lower:.2f})
    """)
    
    print("=" * 100)
    print("BOOTSTRAP VALIDATION COMPLETE")
    print("=" * 100)
    
    return bootstrap_results


if __name__ == "__main__":
    results = main()

