
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost-Optimized Backtester: Reduced Turnover Version
====================================================
Implements several techniques to reduce trading costs:

1. SIGNAL PERSISTENCE: Require 8-state signal to hold for N days
2. POSITION SMOOTHING: Gradual transitions instead of 0→100%
3. TRADE BANDS: Only trade if position change exceeds threshold
4. REBALANCE FREQUENCY: Configurable (30, 60, 90 days)
5. MIN TRADE SIZE: Higher threshold to avoid small trades

Compares original vs optimized to show cost savings.

Usage:
    python backtest_cost_optimized.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# 8-State Parameters (locked)
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_PERM = 20

# Risk Parity (will be varied)
COV_LOOKBACK = 60

# Risk Management (optimized)
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
# OPTIMIZATION PARAMETERS
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for cost optimization."""
    name: str
    
    # Signal persistence: require signal to hold for N days
    signal_persistence_days: int = 1  # 1 = no persistence (original)
    
    # Position smoothing: max position change per day
    max_position_change_per_day: float = 1.0  # 1.0 = instant (original)
    
    # Trade bands: only trade if change exceeds this
    min_position_change: float = 0.01  # 1% (original)
    
    # Rebalance frequency
    rebalance_days: int = 30  # 30 = monthly (original)
    
    # Use intermediate positions (0.5 instead of 0)
    use_intermediate_positions: bool = False


# Define configurations to test
CONFIGS = [
    OptimizationConfig(
        name="ORIGINAL",
        signal_persistence_days=1,
        max_position_change_per_day=1.0,
        min_position_change=0.01,
        rebalance_days=30,
        use_intermediate_positions=False
    ),
    OptimizationConfig(
        name="SIGNAL_PERSIST_3D",
        signal_persistence_days=3,
        max_position_change_per_day=1.0,
        min_position_change=0.01,
        rebalance_days=30,
        use_intermediate_positions=False
    ),
    OptimizationConfig(
        name="GRADUAL_25PCT",
        signal_persistence_days=1,
        max_position_change_per_day=0.25,
        min_position_change=0.01,
        rebalance_days=30,
        use_intermediate_positions=False
    ),
    OptimizationConfig(
        name="WIDE_BANDS_5PCT",
        signal_persistence_days=1,
        max_position_change_per_day=1.0,
        min_position_change=0.05,
        rebalance_days=30,
        use_intermediate_positions=False
    ),
    OptimizationConfig(
        name="QUARTERLY_REBAL",
        signal_persistence_days=1,
        max_position_change_per_day=1.0,
        min_position_change=0.01,
        rebalance_days=90,
        use_intermediate_positions=False
    ),
    OptimizationConfig(
        name="INTERMEDIATE_POS",
        signal_persistence_days=1,
        max_position_change_per_day=1.0,
        min_position_change=0.01,
        rebalance_days=30,
        use_intermediate_positions=True
    ),
    OptimizationConfig(
        name="COMBINED_LIGHT",
        signal_persistence_days=2,
        max_position_change_per_day=0.50,
        min_position_change=0.03,
        rebalance_days=60,
        use_intermediate_positions=False
    ),
    OptimizationConfig(
        name="COMBINED_AGGRESSIVE",
        signal_persistence_days=3,
        max_position_change_per_day=0.25,
        min_position_change=0.05,
        rebalance_days=90,
        use_intermediate_positions=True
    ),
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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
    cost_drag: float  # Annual %
    total_trades: int
    avg_trades_per_month: float
    turnover: float  # Annual %
    equity_curve: pd.Series


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


def generate_signals_for_pair(df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                               df_168h: pd.DataFrame) -> pd.DataFrame:
    """Generate 8-state signals."""
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    
    return aligned.dropna().astype(int)


def calculate_hit_rates(returns: pd.Series, signals: pd.DataFrame) -> Dict[Tuple[int, int, int], Dict]:
    """Calculate hit rates."""
    all_perms = list(product([0, 1], repeat=3))
    
    common_idx = returns.index.intersection(signals.index)
    aligned_returns = returns.loc[common_idx]
    aligned_signals = signals.loc[common_idx]
    
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
            hit_rate = (perm_returns > 0).sum() / n
        else:
            hit_rate = 0.5
        
        hit_rates[perm] = {
            'n': n,
            'hit_rate': hit_rate,
            'sufficient': n >= MIN_SAMPLES_PER_PERM,
        }
    
    return hit_rates


def get_8state_position(perm: Tuple[int, int, int], hit_rates: Dict, 
                        use_intermediate: bool = False) -> float:
    """Get position for 8-state."""
    data = hit_rates.get(perm, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        # Use 0.25 instead of 0 if intermediate positions enabled
        return 0.25 if use_intermediate else 0.00


def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Calculate risk parity weights."""
    cov = returns_df.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return pd.Series(weights, index=returns_df.columns)


def calculate_vol_scalar(returns: pd.Series) -> float:
    """Calculate volatility scalar."""
    if len(returns) < VOL_LOOKBACK:
        return 1.0
    
    realized_vol = returns.iloc[-VOL_LOOKBACK:].std() * np.sqrt(365)
    
    if realized_vol <= 0:
        return 1.0
    
    scalar = TARGET_VOL / realized_vol
    return np.clip(scalar, MIN_EXPOSURE_FLOOR, MAX_LEVERAGE)


def calculate_dd_scalar(current_dd: float) -> float:
    """Calculate drawdown scalar."""
    if current_dd >= DD_START_REDUCE:
        return 1.0
    elif current_dd <= DD_MIN_EXPOSURE:
        return MIN_EXPOSURE_FLOOR
    else:
        range_dd = DD_START_REDUCE - DD_MIN_EXPOSURE
        position = (current_dd - DD_MIN_EXPOSURE) / range_dd
        return MIN_EXPOSURE_FLOOR + position * (1.0 - MIN_EXPOSURE_FLOOR)


# =============================================================================
# BACKTESTER
# =============================================================================

class CostOptimizedBacktester:
    """Backtester with cost optimization options."""
    
    def __init__(self, config: OptimizationConfig, initial_capital: float = 100000.0):
        self.config = config
        self.initial_capital = initial_capital
        self.db = Database()
        
        # Data
        self.prices: Dict[str, pd.DataFrame] = {}
        self.returns: Dict[str, pd.Series] = {}
        self.signals: Dict[str, pd.DataFrame] = {}
        self.hit_rates: Dict[str, Dict] = {}
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, float] = {pair: 0.0 for pair in DEPLOY_PAIRS}
        self.current_weights: Dict[str, float] = {pair: 0.0 for pair in DEPLOY_PAIRS}
        self.equity = initial_capital
        self.peak_equity = initial_capital
        
        # Signal persistence tracking
        self.signal_history: Dict[str, List[float]] = {pair: [] for pair in DEPLOY_PAIRS}
        
        # Costs
        self.total_costs = 0.0
        self.total_turnover = 0.0
        self.total_trades = 0
        
    def load_data(self):
        """Load data."""
        for pair in DEPLOY_PAIRS:
            df_1h = self.db.get_ohlcv(pair)
            df_24h = resample_ohlcv(df_1h, '24h')
            df_72h = resample_ohlcv(df_1h, '72h')
            df_168h = resample_ohlcv(df_1h, '168h')
            
            self.prices[pair] = df_24h
            self.returns[pair] = df_24h['close'].pct_change()
            self.signals[pair] = generate_signals_for_pair(df_24h, df_72h, df_168h)
            self.hit_rates[pair] = calculate_hit_rates(self.returns[pair], self.signals[pair])
        
        self.returns_df = pd.DataFrame(self.returns).dropna()
        
    def get_persistent_signal(self, pair: str, raw_signal: float) -> float:
        """Apply signal persistence filter."""
        history = self.signal_history[pair]
        history.append(raw_signal)
        
        # Keep only recent history
        if len(history) > self.config.signal_persistence_days:
            history.pop(0)
        
        # Require all recent signals to agree
        if len(history) < self.config.signal_persistence_days:
            return history[-1]  # Not enough history, use current
        
        # Check if all signals in window are the same
        if all(s == history[-1] for s in history):
            return history[-1]
        else:
            # Return previous stable signal (last before change)
            return history[0]
    
    def apply_position_smoothing(self, current: float, target: float) -> float:
        """Apply gradual position changes."""
        max_change = self.config.max_position_change_per_day
        
        diff = target - current
        
        if abs(diff) <= max_change:
            return target
        else:
            return current + np.sign(diff) * max_change
    
    def should_trade(self, current_weight: float, target_weight: float) -> bool:
        """Check if trade meets minimum threshold."""
        return abs(target_weight - current_weight) >= self.config.min_position_change
    
    def execute_trade(self, pair: str, target_value: float, current_value: float):
        """Execute trade with costs."""
        trade_value = abs(target_value - current_value)
        
        if trade_value < 100:  # Minimum $100 trade
            return
        
        cost = trade_value * TOTAL_COST_PER_TRADE
        
        self.positions[pair] = target_value
        self.cash -= (target_value - current_value) + cost
        
        self.total_costs += cost
        self.total_turnover += trade_value
        self.total_trades += 1
    
    def run(self) -> BacktestResult:
        """Run backtest."""
        self.load_data()
        
        dates = self.returns_df.index.tolist()
        n_days = len(dates)
        
        rp_weights = pd.Series(1/len(DEPLOY_PAIRS), index=DEPLOY_PAIRS)
        last_rebalance = None
        
        equity_history = []
        
        for i, date in enumerate(dates):
            
            # Skip warmup
            if i < max(COV_LOOKBACK, VOL_LOOKBACK):
                equity_history.append(self.equity)
                continue
            
            # Update positions with returns
            for pair in DEPLOY_PAIRS:
                if pair in self.returns_df.columns and self.positions[pair] > 0:
                    ret = self.returns_df.loc[date, pair]
                    self.positions[pair] *= (1 + ret)
            
            self.equity = self.cash + sum(self.positions.values())
            self.peak_equity = max(self.peak_equity, self.equity)
            current_dd = (self.equity - self.peak_equity) / self.peak_equity
            
            # Rebalance weights if needed
            should_rebalance = (last_rebalance is None or 
                               (date - last_rebalance).days >= self.config.rebalance_days)
            
            if should_rebalance:
                trailing_returns = self.returns_df.iloc[i-COV_LOOKBACK:i]
                rp_weights = calculate_risk_parity_weights(trailing_returns)
                last_rebalance = date
            
            # Get 8-state signals with persistence
            target_8state = {}
            
            for pair in DEPLOY_PAIRS:
                signals = self.signals[pair]
                
                prior_signals = signals[signals.index <= date]
                if len(prior_signals) > 0:
                    sig = prior_signals.iloc[-1]
                    perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
                else:
                    perm = (1, 1, 1)
                
                raw_signal = get_8state_position(
                    perm, self.hit_rates[pair], 
                    self.config.use_intermediate_positions
                )
                
                # Apply signal persistence
                persistent_signal = self.get_persistent_signal(pair, raw_signal)
                target_8state[pair] = persistent_signal
            
            # Risk management
            port_returns = self.returns_df.iloc[:i+1].mean(axis=1)
            vol_scalar = calculate_vol_scalar(port_returns)
            dd_scalar = calculate_dd_scalar(current_dd)
            exposure = max(min(vol_scalar, dd_scalar), MIN_EXPOSURE_FLOOR)
            
            # Calculate target weights
            target_weights = {}
            for pair in DEPLOY_PAIRS:
                target_weights[pair] = rp_weights[pair] * target_8state[pair] * exposure
            
            # Apply position smoothing and execute trades
            for pair in DEPLOY_PAIRS:
                current_weight = self.current_weights[pair]
                raw_target = target_weights[pair]
                
                # Apply smoothing
                smoothed_target = self.apply_position_smoothing(current_weight, raw_target)
                
                # Check if should trade
                if self.should_trade(current_weight, smoothed_target):
                    current_value = self.positions[pair]
                    target_value = self.equity * smoothed_target
                    
                    self.execute_trade(pair, target_value, current_value)
                    self.current_weights[pair] = smoothed_target
            
            self.equity = self.cash + sum(self.positions.values())
            equity_history.append(self.equity)
        
        # Calculate metrics
        equity_series = pd.Series(equity_history, index=dates)
        
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        years = n_days / 365
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        daily_returns = equity_series.pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
        
        rolling_max = equity_series.expanding().max()
        max_dd = ((equity_series - rolling_max) / rolling_max).min()
        
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        cost_drag = (self.total_costs / self.initial_capital) / years
        turnover = (self.total_turnover / self.initial_capital) / years
        avg_trades_per_month = self.total_trades / (years * 12)
        
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
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("COST-OPTIMIZED BACKTESTER: COMPARISON")
    print("=" * 100)
    
    print(f"""
    TESTING COST REDUCTION STRATEGIES:
    
    1. ORIGINAL:              No optimization (baseline)
    2. SIGNAL_PERSIST_3D:     Require signal to hold 3 days
    3. GRADUAL_25PCT:         Max 25% position change per day
    4. WIDE_BANDS_5PCT:       Only trade if >5% weight change
    5. QUARTERLY_REBAL:       Rebalance every 90 days
    6. INTERMEDIATE_POS:      Use 25% instead of 0% for avoid signals
    7. COMBINED_LIGHT:        Moderate combination
    8. COMBINED_AGGRESSIVE:   Maximum cost reduction
    """)
    
    # Load data once
    print("-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    db = Database()
    
    # Pre-load for first run display
    for pair in DEPLOY_PAIRS:
        print(f"  {pair}...", end=" ")
        df = db.get_ohlcv(pair)
        print(f"{len(df)} bars")
    
    # Run all configurations
    print("\n" + "=" * 100)
    print("RUNNING BACKTESTS")
    print("=" * 100)
    
    results = []
    
    for config in CONFIGS:
        print(f"\n  Testing: {config.name}...")
        backtester = CostOptimizedBacktester(config)
        result = backtester.run()
        results.append(result)
        print(f"    Return: {result.total_return*100:+.1f}% | "
              f"MaxDD: {result.max_dd*100:.1f}% | "
              f"Trades: {result.total_trades:,} | "
              f"Costs: ${result.total_costs:,.0f}")
    
    # Results comparison
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Config':<25} {'Return':>10} {'Annual':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Costs':>12} {'Drag':>8}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.config_name:<25} {r.total_return*100:>+9.1f}% {r.annual_return*100:>+9.1f}% "
              f"{r.sharpe:>8.2f} {r.max_dd*100:>7.1f}% {r.total_trades:>8,} "
              f"${r.total_costs:>11,.0f} {r.cost_drag*100:>7.2f}%")
    
    # Calculate improvements vs original
    original = results[0]
    
    print("\n" + "-" * 100)
    print("IMPROVEMENTS vs ORIGINAL")
    print("-" * 100)
    
    print(f"\n{'Config':<25} {'Return Δ':>12} {'Trades Δ':>12} {'Costs Δ':>15} {'Efficiency':>12}")
    print("-" * 100)
    
    for r in results[1:]:
        return_diff = (r.total_return - original.total_return) * 100
        trades_diff = r.total_trades - original.total_trades
        trades_pct = (trades_diff / original.total_trades) * 100
        costs_diff = r.total_costs - original.total_costs
        costs_pct = (costs_diff / original.total_costs) * 100
        
        # Efficiency: return preserved per cost saved
        if costs_diff < 0:
            efficiency = return_diff / abs(costs_pct) if costs_pct != 0 else 0
        else:
            efficiency = float('-inf')
        
        print(f"{r.config_name:<25} {return_diff:>+11.1f}pp {trades_pct:>+11.1f}% "
              f"${costs_diff:>+14,.0f} ({costs_pct:>+5.1f}%) {efficiency:>+11.2f}")
    
    # Find best
    best_efficiency = max(results[1:], key=lambda r: r.total_return / (r.total_costs + 1))
    best_return = max(results[1:], key=lambda r: r.total_return)
    best_costs = min(results[1:], key=lambda r: r.total_costs)
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  RECOMMENDATIONS                                                                                     ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                      ║
    ║  BEST RETURN:        {best_return.config_name:<20} Return: {best_return.total_return*100:+.1f}%  Costs: ${best_return.total_costs:,.0f}          ║
    ║  LOWEST COSTS:       {best_costs.config_name:<20} Return: {best_costs.total_return*100:+.1f}%  Costs: ${best_costs.total_costs:,.0f}          ║
    ║  BEST EFFICIENCY:    {best_efficiency.config_name:<20} Return: {best_efficiency.total_return*100:+.1f}%  Costs: ${best_efficiency.total_costs:,.0f}          ║
    ║                                                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Detailed look at best options
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS: TOP 3 CONFIGURATIONS")
    print("=" * 100)
    
    # Sort by Calmar ratio
    sorted_results = sorted(results, key=lambda r: r.calmar, reverse=True)
    
    for i, r in enumerate(sorted_results[:3], 1):
        config = next(c for c in CONFIGS if c.name == r.config_name)
        
        print(f"""
    #{i} {r.config_name}
    ────────────────────────────────────────
    Settings:
        Signal Persistence:     {config.signal_persistence_days} days
        Max Position Change:    {config.max_position_change_per_day*100:.0f}%/day
        Min Trade Threshold:    {config.min_position_change*100:.0f}%
        Rebalance Frequency:    {config.rebalance_days} days
        Intermediate Positions: {config.use_intermediate_positions}
    
    Performance:
        Total Return:           {r.total_return*100:+.1f}%
        Annual Return:          {r.annual_return*100:+.1f}%
        Sharpe Ratio:           {r.sharpe:.2f}
        Max Drawdown:           {r.max_dd*100:.1f}%
        Calmar Ratio:           {r.calmar:.2f}
    
    Trading:
        Total Trades:           {r.total_trades:,}
        Trades/Month:           {r.avg_trades_per_month:.1f}
        Annual Turnover:        {r.turnover*100:.0f}%
        Total Costs:            ${r.total_costs:,.0f}
        Annual Cost Drag:       {r.cost_drag*100:.2f}%
        """)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    results = main()