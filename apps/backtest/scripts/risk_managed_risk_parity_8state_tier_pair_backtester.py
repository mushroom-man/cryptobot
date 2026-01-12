# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Backtester: Risk-Managed Risk Parity with Trading Costs
===================================================================
Complete backtesting framework that accurately models:

1. Trading fees (maker/taker)
2. Slippage
3. Rebalancing costs
4. Exposure change costs

STRATEGY: 8-State Risk-Managed Risk Parity
PAIRS: XLMUSD, ZECUSD, ETCUSD, ETHUSD, XMRUSD, ADAUSD

Usage:
    python backtest_with_costs.py
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from itertools import product
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

# Assets
DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# 8-State Parameters (locked from XBTUSD optimization)
MA_PERIOD_24H = 24
MA_PERIOD_72H = 8
MA_PERIOD_168H = 2
ENTRY_BUFFER = 0.02
EXIT_BUFFER = 0.005
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_PERM = 20

# Risk Parity Parameters
REBALANCE_DAYS = 30
COV_LOOKBACK = 60

# Risk Management Parameters (optimized)
TARGET_VOL = 0.40
VOL_LOOKBACK = 30
DD_START_REDUCE = -0.20
DD_MIN_EXPOSURE = -0.50
MIN_EXPOSURE_FLOOR = 0.40
MAX_LEVERAGE = 1.0

# Trading Costs
TRADING_FEE = 0.0010      # 0.10% per trade (Kraken taker fee)
SLIPPAGE = 0.0005         # 0.05% slippage estimate
TOTAL_COST_PER_TRADE = TRADING_FEE + SLIPPAGE  # 0.15% total

# Minimum trade threshold (don't trade if change < this)
MIN_TRADE_THRESHOLD = 0.01  # 1% position change minimum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Trade:
    """Single trade record."""
    timestamp: pd.Timestamp
    pair: str
    direction: str  # 'BUY' or 'SELL'
    size: float     # Dollar amount
    price: float    # Asset price
    fee: float      # Trading fee
    slippage: float # Slippage cost
    total_cost: float
    reason: str     # 'REBALANCE', '8STATE', 'RISK_MGMT'


@dataclass
class DailySnapshot:
    """Daily portfolio state."""
    timestamp: pd.Timestamp
    equity: float
    cash: float
    positions: Dict[str, float]  # pair -> dollar value
    weights: Dict[str, float]    # pair -> weight
    exposure: float
    drawdown: float
    regime: str
    vol_scalar: float
    dd_scalar: float
    trades_today: int
    costs_today: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Performance metrics
    total_return: float
    total_return_gross: float  # Before costs
    annual_return: float
    sharpe: float
    max_dd: float
    calmar: float
    
    # Cost analysis
    total_costs: float
    total_fees: float
    total_slippage: float
    cost_drag: float  # Annual cost as % of equity
    
    # Trading stats
    total_trades: int
    avg_trades_per_month: float
    avg_trade_size: float
    turnover: float  # Annual turnover as % of equity
    
    # Time series
    equity_curve: pd.Series
    equity_curve_gross: pd.Series
    drawdown_series: pd.Series
    exposure_series: pd.Series
    
    # Detailed logs
    trades: List[Trade]
    daily_snapshots: List[DailySnapshot]


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
    """Generate 8-state signals for a single pair."""
    trend_24h = label_trend_binary(df_24h, MA_PERIOD_24H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_72h = label_trend_binary(df_72h, MA_PERIOD_72H, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary(df_168h, MA_PERIOD_168H, ENTRY_BUFFER, EXIT_BUFFER)
    
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_72h'] = trend_72h.shift(1).reindex(df_24h.index, method='ffill')
    aligned['trend_168h'] = trend_168h.shift(1).reindex(df_24h.index, method='ffill')
    
    return aligned.dropna().astype(int)


def calculate_hit_rates(returns: pd.Series, signals: pd.DataFrame) -> Dict[Tuple[int, int, int], Dict]:
    """Calculate hit rates for all 8 states."""
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


def get_8state_position(perm: Tuple[int, int, int], hit_rates: Dict) -> float:
    """Get position for 8-state."""
    data = hit_rates.get(perm, {'sufficient': False, 'hit_rate': 0.5})
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.00


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


def detect_regime(returns_30d: float) -> str:
    """Detect market regime."""
    if returns_30d > 0.20:
        return 'BULL'
    elif returns_30d < -0.20:
        return 'BEAR'
    else:
        return 'SIDEWAYS'


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Production backtester with trading costs."""
    
    def __init__(self, initial_capital: float = 100000.0):
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
        self.equity = initial_capital
        self.peak_equity = initial_capital
        
        # Tracking
        self.trades: List[Trade] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.equity_history = []
        self.equity_history_gross = []
        
        # Costs
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_turnover = 0.0
        
    def load_data(self):
        """Load and prepare all data."""
        print("Loading data...")
        
        for pair in DEPLOY_PAIRS:
            print(f"  {pair}...", end=" ")
            
            # Load OHLCV
            df_1h = self.db.get_ohlcv(pair)
            df_24h = resample_ohlcv(df_1h, '24h')
            df_72h = resample_ohlcv(df_1h, '72h')
            df_168h = resample_ohlcv(df_1h, '168h')
            
            # Store prices
            self.prices[pair] = df_24h
            
            # Calculate returns
            self.returns[pair] = df_24h['close'].pct_change()
            
            # Generate signals
            self.signals[pair] = generate_signals_for_pair(df_24h, df_72h, df_168h)
            
            # Calculate hit rates
            self.hit_rates[pair] = calculate_hit_rates(self.returns[pair], self.signals[pair])
            
            print(f"{len(df_24h)} bars")
        
        # Create aligned returns DataFrame
        self.returns_df = pd.DataFrame(self.returns).dropna()
        print(f"\n  Common period: {self.returns_df.index.min().date()} to {self.returns_df.index.max().date()}")
        print(f"  Total days: {len(self.returns_df)}")
        
    def execute_trade(self, timestamp: pd.Timestamp, pair: str, 
                      target_value: float, current_value: float,
                      price: float, reason: str) -> Optional[Trade]:
        """Execute a single trade with costs."""
        
        trade_value = target_value - current_value
        
        # Check minimum trade threshold
        if abs(trade_value) < self.equity * MIN_TRADE_THRESHOLD:
            return None
        
        direction = 'BUY' if trade_value > 0 else 'SELL'
        size = abs(trade_value)
        
        # Calculate costs
        fee = size * TRADING_FEE
        slippage = size * SLIPPAGE
        total_cost = fee + slippage
        
        # Update state
        self.positions[pair] = target_value
        self.cash -= trade_value + total_cost
        
        # Track costs
        self.total_fees += fee
        self.total_slippage += slippage
        self.total_turnover += size
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            pair=pair,
            direction=direction,
            size=size,
            price=price,
            fee=fee,
            slippage=slippage,
            total_cost=total_cost,
            reason=reason
        )
        
        self.trades.append(trade)
        return trade
    
    def run(self) -> BacktestResult:
        """Run the full backtest."""
        
        self.load_data()
        
        print("\nRunning backtest...")
        
        dates = self.returns_df.index.tolist()
        n_days = len(dates)
        
        # Risk parity weights (will be updated monthly)
        rp_weights = pd.Series(1/len(DEPLOY_PAIRS), index=DEPLOY_PAIRS)
        last_rebalance = None
        
        # Track gross equity (before costs)
        gross_equity = self.initial_capital
        
        for i, date in enumerate(dates):
            
            # Skip warmup period
            if i < max(COV_LOOKBACK, VOL_LOOKBACK):
                self.equity_history.append(self.equity)
                self.equity_history_gross.append(gross_equity)
                continue
            
            # =================================================================
            # STEP 1: Update positions with market returns
            # =================================================================
            
            daily_pnl = 0.0
            daily_pnl_gross = 0.0
            
            for pair in DEPLOY_PAIRS:
                if pair in self.returns_df.columns and self.positions[pair] > 0:
                    ret = self.returns_df.loc[date, pair]
                    pnl = self.positions[pair] * ret
                    daily_pnl += pnl
                    daily_pnl_gross += pnl
                    self.positions[pair] *= (1 + ret)
            
            self.equity = self.cash + sum(self.positions.values())
            gross_equity += daily_pnl_gross
            
            # Update peak
            self.peak_equity = max(self.peak_equity, self.equity)
            current_dd = (self.equity - self.peak_equity) / self.peak_equity
            
            # =================================================================
            # STEP 2: Check if rebalancing needed (monthly)
            # =================================================================
            
            should_rebalance = (last_rebalance is None or 
                               (date - last_rebalance).days >= REBALANCE_DAYS)
            
            if should_rebalance:
                # Calculate new risk parity weights
                trailing_returns = self.returns_df.iloc[i-COV_LOOKBACK:i]
                rp_weights = calculate_risk_parity_weights(trailing_returns)
                last_rebalance = date
            
            # =================================================================
            # STEP 3: Calculate 8-state positions for each pair
            # =================================================================
            
            positions_8state = {}
            
            for pair in DEPLOY_PAIRS:
                signals = self.signals[pair]
                
                # Get signal for this date
                prior_signals = signals[signals.index <= date]
                if len(prior_signals) > 0:
                    sig = prior_signals.iloc[-1]
                    perm = (int(sig['trend_24h']), int(sig['trend_72h']), int(sig['trend_168h']))
                else:
                    perm = (1, 1, 1)  # Default
                
                positions_8state[pair] = get_8state_position(perm, self.hit_rates[pair])
            
            # =================================================================
            # STEP 4: Calculate risk management scalars
            # =================================================================
            
            # Portfolio returns for vol calculation
            port_returns = self.returns_df.iloc[:i+1].mean(axis=1)
            vol_scalar = calculate_vol_scalar(port_returns)
            dd_scalar = calculate_dd_scalar(current_dd)
            
            # Combined exposure
            exposure = min(vol_scalar, dd_scalar)
            exposure = max(exposure, MIN_EXPOSURE_FLOOR)
            
            # =================================================================
            # STEP 5: Calculate target positions
            # =================================================================
            
            target_positions = {}
            
            for pair in DEPLOY_PAIRS:
                # Base weight × 8-state × exposure
                target_weight = rp_weights[pair] * positions_8state[pair] * exposure
                target_positions[pair] = self.equity * target_weight
            
            # =================================================================
            # STEP 6: Execute trades
            # =================================================================
            
            trades_today = 0
            costs_today = 0.0
            
            for pair in DEPLOY_PAIRS:
                current_value = self.positions[pair]
                target_value = target_positions[pair]
                
                # Get current price
                if date in self.prices[pair].index:
                    price = self.prices[pair].loc[date, 'close']
                else:
                    continue
                
                # Determine reason
                if should_rebalance:
                    reason = 'REBALANCE'
                elif abs(positions_8state[pair] - (current_value / self.equity if self.equity > 0 else 0)) > 0.1:
                    reason = '8STATE'
                else:
                    reason = 'RISK_MGMT'
                
                # Execute trade
                trade = self.execute_trade(date, pair, target_value, current_value, price, reason)
                
                if trade:
                    trades_today += 1
                    costs_today += trade.total_cost
            
            # Update equity after costs
            self.equity = self.cash + sum(self.positions.values())
            
            # =================================================================
            # STEP 7: Record daily snapshot
            # =================================================================
            
            # Detect regime
            if i >= 30:
                returns_30d = self.returns_df.iloc[i-30:i].mean(axis=1).sum()
                regime = detect_regime(returns_30d)
            else:
                regime = 'SIDEWAYS'
            
            snapshot = DailySnapshot(
                timestamp=date,
                equity=self.equity,
                cash=self.cash,
                positions=self.positions.copy(),
                weights={p: self.positions[p]/self.equity if self.equity > 0 else 0 
                        for p in DEPLOY_PAIRS},
                exposure=exposure,
                drawdown=current_dd,
                regime=regime,
                vol_scalar=vol_scalar,
                dd_scalar=dd_scalar,
                trades_today=trades_today,
                costs_today=costs_today
            )
            self.daily_snapshots.append(snapshot)
            
            self.equity_history.append(self.equity)
            self.equity_history_gross.append(gross_equity)
            
            # Progress
            if (i + 1) % 500 == 0:
                print(f"  Day {i+1}/{n_days} - Equity: ${self.equity:,.0f} - Trades: {len(self.trades)}")
        
        # =================================================================
        # CALCULATE FINAL METRICS
        # =================================================================
        
        print("\nCalculating metrics...")
        
        equity_series = pd.Series(self.equity_history, index=dates)
        equity_series_gross = pd.Series(self.equity_history_gross, index=dates)
        
        # Returns
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        total_return_gross = (gross_equity - self.initial_capital) / self.initial_capital
        
        years = n_days / 365
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        daily_returns = equity_series.pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
        
        rolling_max = equity_series.expanding().max()
        drawdown_series = (equity_series - rolling_max) / rolling_max
        max_dd = drawdown_series.min()
        
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Cost analysis
        total_costs = self.total_fees + self.total_slippage
        cost_drag = (total_costs / self.initial_capital) / years  # Annual cost %
        
        # Trading stats
        total_trades = len(self.trades)
        avg_trades_per_month = total_trades / (years * 12) if years > 0 else 0
        avg_trade_size = self.total_turnover / total_trades if total_trades > 0 else 0
        turnover = (self.total_turnover / self.initial_capital) / years if years > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            total_return_gross=total_return_gross,
            annual_return=annual_return,
            sharpe=sharpe,
            max_dd=max_dd,
            calmar=calmar,
            total_costs=total_costs,
            total_fees=self.total_fees,
            total_slippage=self.total_slippage,
            cost_drag=cost_drag,
            total_trades=total_trades,
            avg_trades_per_month=avg_trades_per_month,
            avg_trade_size=avg_trade_size,
            turnover=turnover,
            equity_curve=equity_series,
            equity_curve_gross=equity_series_gross,
            drawdown_series=drawdown_series,
            exposure_series=pd.Series([s.exposure for s in self.daily_snapshots], 
                                      index=[s.timestamp for s in self.daily_snapshots]),
            trades=self.trades,
            daily_snapshots=self.daily_snapshots
        )


# =============================================================================
# REPORTING
# =============================================================================

def print_report(result: BacktestResult):
    """Print detailed backtest report."""
    
    print("\n" + "=" * 100)
    print("BACKTEST RESULTS: 8-STATE RISK-MANAGED RISK PARITY")
    print("=" * 100)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  PERFORMANCE SUMMARY                                                            │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Total Return (Net):       {result.total_return*100:>+10.1f}%                                       │
    │  Total Return (Gross):     {result.total_return_gross*100:>+10.1f}%                                       │
    │  Annual Return:            {result.annual_return*100:>+10.1f}%                                       │
    │  Sharpe Ratio:             {result.sharpe:>10.2f}                                         │
    │  Max Drawdown:             {result.max_dd*100:>10.1f}%                                       │
    │  Calmar Ratio:             {result.calmar:>10.2f}                                         │
    └─────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  TRADING COSTS ANALYSIS                                                         │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Total Costs:              ${result.total_costs:>10,.0f}                                       │
    │    - Trading Fees:         ${result.total_fees:>10,.0f}                                       │
    │    - Slippage:             ${result.total_slippage:>10,.0f}                                       │
    │  Annual Cost Drag:         {result.cost_drag*100:>10.2f}%                                       │
    │  Cost Impact on Return:    {(result.total_return_gross - result.total_return)*100:>10.1f}pp                                      │
    └─────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  TRADING STATISTICS                                                             │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Total Trades:             {result.total_trades:>10,}                                         │
    │  Avg Trades/Month:         {result.avg_trades_per_month:>10.1f}                                         │
    │  Avg Trade Size:           ${result.avg_trade_size:>10,.0f}                                       │
    │  Annual Turnover:          {result.turnover*100:>10.0f}%                                       │
    └─────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Trade breakdown by reason
    trades_by_reason = {}
    for trade in result.trades:
        if trade.reason not in trades_by_reason:
            trades_by_reason[trade.reason] = {'count': 0, 'volume': 0, 'cost': 0}
        trades_by_reason[trade.reason]['count'] += 1
        trades_by_reason[trade.reason]['volume'] += trade.size
        trades_by_reason[trade.reason]['cost'] += trade.total_cost
    
    print("    TRADES BY REASON:")
    print("    " + "-" * 70)
    print(f"    {'Reason':<15} {'Count':>10} {'Volume':>15} {'Cost':>15}")
    print("    " + "-" * 70)
    for reason, data in trades_by_reason.items():
        print(f"    {reason:<15} {data['count']:>10,} ${data['volume']:>14,.0f} ${data['cost']:>14,.0f}")
    
    # Trade breakdown by pair
    trades_by_pair = {}
    for trade in result.trades:
        if trade.pair not in trades_by_pair:
            trades_by_pair[trade.pair] = {'count': 0, 'volume': 0, 'cost': 0}
        trades_by_pair[trade.pair]['count'] += 1
        trades_by_pair[trade.pair]['volume'] += trade.size
        trades_by_pair[trade.pair]['cost'] += trade.total_cost
    
    print("\n    TRADES BY PAIR:")
    print("    " + "-" * 70)
    print(f"    {'Pair':<15} {'Count':>10} {'Volume':>15} {'Cost':>15}")
    print("    " + "-" * 70)
    for pair in DEPLOY_PAIRS:
        if pair in trades_by_pair:
            data = trades_by_pair[pair]
            print(f"    {pair:<15} {data['count']:>10,} ${data['volume']:>14,.0f} ${data['cost']:>14,.0f}")
    
    # Exposure analysis
    if result.daily_snapshots:
        exposures = [s.exposure for s in result.daily_snapshots]
        print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  EXPOSURE ANALYSIS                                                              │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │  Average Exposure:         {np.mean(exposures)*100:>10.1f}%                                       │
    │  Min Exposure:             {np.min(exposures)*100:>10.1f}%                                       │
    │  Max Exposure:             {np.max(exposures)*100:>10.1f}%                                       │
    └─────────────────────────────────────────────────────────────────────────────────┘
        """)
    
    print("\n" + "=" * 100)
    print("BACKTEST COMPLETE")
    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("PRODUCTION BACKTESTER: 8-STATE RISK-MANAGED RISK PARITY")
    print("=" * 100)
    
    print(f"""
    STRATEGY PARAMETERS:
        8-State:        MA(24/8/2), Hysteresis 2%/0.5%
        Risk Parity:    Monthly rebalance, 60-day lookback
        Risk Mgmt:      Vol Target 40%, DD Control -20%/-50%, Floor 40%
    
    TRADING COSTS:
        Trading Fee:    {TRADING_FEE*100:.2f}%
        Slippage:       {SLIPPAGE*100:.2f}%
        Total:          {TOTAL_COST_PER_TRADE*100:.2f}% per trade
    
    PAIRS: {', '.join(DEPLOY_PAIRS)}
    """)
    
    # Run backtest
    backtester = Backtester(initial_capital=100000)
    result = backtester.run()
    
    # Print report
    print_report(result)
    
    return result


if __name__ == "__main__":
    result = main()

