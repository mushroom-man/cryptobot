#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected 8-State Backtester: Matching Original Implementation
===============================================================
Fixes identified from original code:

1. DYNAMIC HIT RATES: Calculate per-pair hit rates from actual data
2. CORRECT HYSTERESIS: Use both buffers with AND condition
3. CORRECT SIGNAL ALIGNMENT: Shift BEFORE reindex
4. POSITION SIMULATION: Track actual dollar positions, not just weights

This version matches the original backtester that produced the PDF results.

Compares:
    - Risk Parity (original)
    - Minimum Variance (proposed improvement)
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import product
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION (MATCHING ORIGINAL)
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

# Risk Parity / Min Variance
COV_LOOKBACK = 60
REBALANCE_DAYS = 30

# Risk Management (from original)
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

# Minimum Variance Parameters
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.50
SHRINKAGE = 0.1


# =============================================================================
# CORRECTED HELPER FUNCTIONS (FROM ORIGINAL)
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def label_trend_binary(df: pd.DataFrame, ma_period: int,
                       entry_buffer: float, exit_buffer: float) -> pd.Series:
    """
    CORRECTED: Binary trend detection with hysteresis.
    
    Key difference from my version:
        - Uses BOTH buffers with AND condition for switching
        - Original: price < ma * (1 - exit_buffer) AND price < ma * (1 - entry_buffer)
    """
    close = df['close']
    ma = close.rolling(ma_period).mean()
    
    labels = pd.Series(index=df.index, dtype=int)
    current = 1  # Start bullish
    
    for i in range(len(df)):
        if pd.isna(ma.iloc[i]):
            labels.iloc[i] = current
            continue
        
        price = close.iloc[i]
        ma_val = ma.iloc[i]
        
        if current == 1:  # Currently UP
            # CORRECTED: Both conditions must be true
            if price < ma_val * (1 - exit_buffer) and price < ma_val * (1 - entry_buffer):
                current = 0
        else:  # Currently DOWN
            # CORRECTED: Both conditions must be true
            if price > ma_val * (1 + exit_buffer) and price > ma_val * (1 + entry_buffer):
                current = 1
        
        labels.iloc[i] = current
    
    return labels


def generate_signals_for_pair(df_24h: pd.DataFrame, df_72h: pd.DataFrame,
                               df_168h: pd.DataFrame) -> pd.DataFrame:
    """
    CORRECTED: Generate 8-state signals with proper alignment.
    
    Key difference: Shift BEFORE reindex (not after)
    """
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    aligned = pd.DataFrame(index=df_24h.index)
    
    # CORRECTED: Shift FIRST, then reindex
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    
    return aligned.dropna().astype(int)


def calculate_hit_rates(returns: pd.Series, signals: pd.DataFrame) -> Dict[Tuple[int, int, int], Dict]:
    """
    CORRECTED: Calculate hit rates dynamically from actual data.
    
    This is the KEY difference - original uses dynamic hit rates per pair,
    NOT a static lookup table from the PDF.
    """
    all_perms = list(product([0, 1], repeat=3))
    
    common_idx = returns.index.intersection(signals.index)
    aligned_returns = returns.loc[common_idx]
    aligned_signals = signals.loc[common_idx]
    
    # Forward returns (next period's return)
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


def get_8state_position(perm: Tuple[int, int, int], hit_rates: Dict) -> float:
    """
    CORRECTED: Get position based on dynamic hit rates.
    
    NOT using static lookup table - uses actual calculated hit rates.
    """
    data = hit_rates.get(perm, {'sufficient': False, 'hit_rate': 0.5})
    
    if not data['sufficient']:
        return 0.50  # Not enough samples
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00  # Good hit rate → INVEST
    else:
        return 0.00  # Bad hit rate → AVOID


def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Calculate risk parity weights (inverse volatility)."""
    cov = returns_df.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return pd.Series(weights, index=returns_df.columns)


def calculate_minimum_variance_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Calculate minimum variance weights."""
    cov = returns_df.cov().values * 252
    
    # Apply shrinkage
    n = cov.shape[0]
    avg_var = np.trace(cov) / n
    target = np.eye(n) * avg_var
    cov = (1 - SHRINKAGE) * cov + SHRINKAGE * target
    
    def objective(w):
        return w @ cov @ w
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n
    w0 = np.ones(n) / n
    
    result = minimize(
        objective, w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500}
    )
    
    weights = result.x
    weights = np.maximum(weights, 0)
    
    if weights.sum() < 1e-8:
        return pd.Series(np.ones(n) / n, index=returns_df.columns)
    
    return pd.Series(weights / weights.sum(), index=returns_df.columns)


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
# CORRECTED BACKTESTER
# =============================================================================

@dataclass
class BacktestResult:
    """Backtest results."""
    name: str
    total_return: float
    annual_return: float
    sharpe: float
    max_dd: float
    calmar: float
    total_costs: float
    cost_drag: float
    total_trades: int
    avg_exposure: float
    equity_curve: pd.Series


class CorrectedBacktester:
    """
    Backtester matching the original implementation exactly.
    
    Key corrections:
        1. Dynamic hit rates per pair
        2. Correct hysteresis logic
        3. Correct signal alignment
        4. Dollar-based position tracking
    """
    
    def __init__(self, use_minimum_variance: bool = False, initial_capital: float = 100000.0):
        self.use_minimum_variance = use_minimum_variance
        self.initial_capital = initial_capital
        self.db = Database()
        
        # Data storage
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
        
        # Costs
        self.total_costs = 0.0
        self.total_turnover = 0.0
        self.total_trades = 0
    
    def load_data(self):
        """Load and prepare all data."""
        print(f"    Loading data...")
        
        for pair in DEPLOY_PAIRS:
            df_1h = self.db.get_ohlcv(pair)
            df_24h = resample_ohlcv(df_1h, '24h')
            df_72h = resample_ohlcv(df_1h, '72h')
            df_168h = resample_ohlcv(df_1h, '168h')
            
            self.prices[pair] = df_24h
            self.returns[pair] = df_24h['close'].pct_change()
            self.signals[pair] = generate_signals_for_pair(df_24h, df_72h, df_168h)
            
            # CORRECTED: Calculate hit rates dynamically per pair
            self.hit_rates[pair] = calculate_hit_rates(self.returns[pair], self.signals[pair])
        
        self.returns_df = pd.DataFrame(self.returns).dropna()
        print(f"    Loaded {len(self.returns_df)} aligned days")
    
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
        
        # Initialize weights
        weights = pd.Series(1/len(DEPLOY_PAIRS), index=DEPLOY_PAIRS)
        last_rebalance = None
        
        equity_history = []
        exposure_history = []
        
        print(f"    Running simulation...")
        
        for i, date in enumerate(dates):
            
            # Skip warmup
            if i < max(COV_LOOKBACK, VOL_LOOKBACK):
                equity_history.append(self.equity)
                exposure_history.append(0.0)
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
                               (date - last_rebalance).days >= REBALANCE_DAYS)
            
            if should_rebalance:
                trailing_returns = self.returns_df.iloc[i-COV_LOOKBACK:i]
                
                if self.use_minimum_variance:
                    weights = calculate_minimum_variance_weights(trailing_returns)
                else:
                    weights = calculate_risk_parity_weights(trailing_returns)
                
                last_rebalance = date
            
            # Get 8-state signals (CORRECTED: using dynamic hit rates)
            target_8state = {}
            
            for pair in DEPLOY_PAIRS:
                signals = self.signals[pair]
                
                prior_signals = signals[signals.index <= date]
                if len(prior_signals) > 0:
                    sig = prior_signals.iloc[-1]
                    perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
                else:
                    perm = (1, 1, 1)
                
                # CORRECTED: Use dynamic hit rates per pair
                raw_signal = get_8state_position(perm, self.hit_rates[pair])
                target_8state[pair] = raw_signal
            
            # Risk management
            port_returns = self.returns_df.iloc[:i+1].mean(axis=1)
            vol_scalar = calculate_vol_scalar(port_returns)
            dd_scalar = calculate_dd_scalar(current_dd)
            exposure = max(min(vol_scalar, dd_scalar), MIN_EXPOSURE_FLOOR)
            
            exposure_history.append(exposure)
            
            # Calculate target weights
            target_weights = {}
            for pair in DEPLOY_PAIRS:
                target_weights[pair] = weights[pair] * target_8state[pair] * exposure
            
            # Execute trades
            for pair in DEPLOY_PAIRS:
                current_weight = self.current_weights[pair]
                target_weight = target_weights[pair]
                
                # Check if should trade (min 1% change)
                if abs(target_weight - current_weight) >= 0.01:
                    current_value = self.positions[pair]
                    target_value = self.equity * target_weight
                    
                    self.execute_trade(pair, target_value, current_value)
                    self.current_weights[pair] = target_weight
            
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
        
        cost_drag = (self.total_costs / self.initial_capital) / years if years > 0 else 0
        avg_exposure = np.mean(exposure_history) if exposure_history else 0
        
        name = "MINIMUM_VARIANCE" if self.use_minimum_variance else "RISK_PARITY"
        
        return BacktestResult(
            name=name,
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            max_dd=max_dd,
            calmar=calmar,
            total_costs=self.total_costs,
            cost_drag=cost_drag,
            total_trades=self.total_trades,
            avg_exposure=avg_exposure,
            equity_curve=equity_series
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("CORRECTED 8-STATE BACKTESTER")
    print("=" * 100)
    
    print(f"""
    CORRECTIONS APPLIED:
    
    1. DYNAMIC HIT RATES: Calculated per pair from actual data
       (Original used this, NOT the static PDF lookup table)
    
    2. HYSTERESIS LOGIC: Both buffers with AND condition
       if price < ma * (1 - exit) AND price < ma * (1 - entry): switch
    
    3. SIGNAL ALIGNMENT: Shift BEFORE reindex
       trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    
    4. POSITION TRACKING: Dollar-based, not weight-based
    """)
    
    # Run Risk Parity
    print("\n" + "-" * 60)
    print("RUNNING RISK PARITY (ORIGINAL)")
    print("-" * 60)
    
    rp_backtester = CorrectedBacktester(use_minimum_variance=False)
    rp_result = rp_backtester.run()
    
    print(f"    Total Return:  {rp_result.total_return*100:+.1f}%")
    print(f"    Annual Return: {rp_result.annual_return*100:+.1f}%")
    print(f"    Sharpe:        {rp_result.sharpe:.2f}")
    print(f"    Max Drawdown:  {rp_result.max_dd*100:.1f}%")
    
    # Run Minimum Variance
    print("\n" + "-" * 60)
    print("RUNNING MINIMUM VARIANCE (PROPOSED)")
    print("-" * 60)
    
    mv_backtester = CorrectedBacktester(use_minimum_variance=True)
    mv_result = mv_backtester.run()
    
    print(f"    Total Return:  {mv_result.total_return*100:+.1f}%")
    print(f"    Annual Return: {mv_result.annual_return*100:+.1f}%")
    print(f"    Sharpe:        {mv_result.sharpe:.2f}")
    print(f"    Max Drawdown:  {mv_result.max_dd*100:.1f}%")
    
    # Comparison
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)
    
    print(f"""
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                         RISK PARITY       MINIMUM VARIANCE        DIFFERENCE         │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │  Total Return:          {rp_result.total_return*100:>+8.0f}%           {mv_result.total_return*100:>+8.0f}%           {(mv_result.total_return-rp_result.total_return)*100:>+8.0f}%      │
    │  Annual Return:         {rp_result.annual_return*100:>+8.1f}%           {mv_result.annual_return*100:>+8.1f}%           {(mv_result.annual_return-rp_result.annual_return)*100:>+8.1f}%      │
    │  Sharpe Ratio:          {rp_result.sharpe:>8.2f}             {mv_result.sharpe:>8.2f}             {mv_result.sharpe-rp_result.sharpe:>+8.2f}       │
    │  Max Drawdown:          {rp_result.max_dd*100:>8.1f}%           {mv_result.max_dd*100:>8.1f}%           {(mv_result.max_dd-rp_result.max_dd)*100:>+8.1f}%      │
    │  Calmar Ratio:          {rp_result.calmar:>8.2f}             {mv_result.calmar:>8.2f}             {mv_result.calmar-rp_result.calmar:>+8.2f}       │
    │  Avg Exposure:          {rp_result.avg_exposure*100:>8.1f}%           {mv_result.avg_exposure*100:>8.1f}%           {(mv_result.avg_exposure-rp_result.avg_exposure)*100:>+8.1f}%      │
    │  Total Trades:          {rp_result.total_trades:>8,}             {mv_result.total_trades:>8,}             {mv_result.total_trades-rp_result.total_trades:>+8,}       │
    │  Total Costs:           ${rp_result.total_costs:>7,.0f}            ${mv_result.total_costs:>7,.0f}            ${mv_result.total_costs-rp_result.total_costs:>+7,.0f}       │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Comparison to PDF target
    print("\n" + "-" * 60)
    print("COMPARISON TO PDF TARGET")
    print("-" * 60)
    
    print(f"""
    PDF TARGET (Risk Parity):
        Annual Return:    +39.7%
        Sharpe:           1.31
        Max Drawdown:     -31.0%
        Avg Exposure:     61.0%
        Trades:           1,620
    
    MY RISK PARITY RESULT:
        Annual Return:    {rp_result.annual_return*100:+.1f}%
        Sharpe:           {rp_result.sharpe:.2f}
        Max Drawdown:     {rp_result.max_dd*100:.1f}%
        Avg Exposure:     {rp_result.avg_exposure*100:.1f}%
        Trades:           {rp_result.total_trades:,}
    
    GAP:
        Annual Return:    {rp_result.annual_return*100 - 39.7:+.1f}pp
        Sharpe:           {rp_result.sharpe - 1.31:+.2f}
        Max Drawdown:     {rp_result.max_dd*100 - (-31.0):+.1f}pp
    """)
    
    # Verdict - use Calmar (return/drawdown) as primary metric
    mv_wins = mv_result.calmar > rp_result.calmar
    
    # Count wins across metrics
    mv_score = 0
    mv_score += 1 if mv_result.annual_return > rp_result.annual_return else 0
    mv_score += 1 if mv_result.sharpe > rp_result.sharpe else 0
    mv_score += 1 if mv_result.max_dd > rp_result.max_dd else 0  # Less negative = better
    mv_score += 1 if mv_result.calmar > rp_result.calmar else 0
    
    if mv_wins or mv_score >= 3:
        verdict = "✓ MINIMUM VARIANCE OUTPERFORMS"
        recommendation = "Use Minimum Variance allocation"
    else:
        verdict = "✗ RISK PARITY OUTPERFORMS"
        recommendation = "Keep Risk Parity allocation"
    
    print(f"    Score: MV wins {mv_score}/4 metrics")
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║  {verdict:<84} ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                      ║
    ║  RECOMMENDATION: {recommendation:<66} ║
    ║                                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("=" * 100)
    print("COMPLETE")
    print("=" * 100)
    
    return rp_result, mv_result


if __name__ == "__main__":
    rp, mv = main()