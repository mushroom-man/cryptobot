# -*- coding: utf-8 -*-
"""
CryptoBot - Backtest Runner
============================
FILE: apps/backtest/runner.py

Main orchestrator for event-driven backtesting.

Supports multiple sizers:
    - RiskParitySizer: Portfolio-level inverse-volatility allocation (RECOMMENDED)
    - KellySizer: Single-asset Kelly criterion sizing
    - AvoidDangerSizer: Binary in/out based on regime danger

The same sizers are used in both backtest and live trading to ensure consistency.

Usage:
    from apps.backtest.runner import BacktestRunner, BacktestConfig
    from cryptobot.risk import RiskParitySizer, RiskParityConfig
    
    # Setup with Risk Parity (recommended)
    sizer = RiskParitySizer(RiskParityConfig())
    
    runner = BacktestRunner(
        config=BacktestConfig(initial_capital=100_000),
        sizer=sizer,
    )
    
    # Run backtest
    results = runner.run(df, predictions, features_df)
    results.print_summary()
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
import pandas as pd
import numpy as np
from tqdm import tqdm

from cryptobot.shared.core.bar import Bar
from cryptobot.shared.core.order import Order
from cryptobot.shared.core.portfolio import Portfolio
from cryptobot.shared.core.engine import TradingEngine, Action, ActionType
from cryptobot.shared.risk.manager import RiskManager
from cryptobot.research.backtest.executor import SimulatedExecutor

# Import sizers - support both old and new
try:
    from cryptobot.risk.risk_parity import RiskParitySizer, RiskParityConfig
except ImportError:
    RiskParitySizer = None
    RiskParityConfig = None

try:
    from cryptobot.risk.position import KellySizer, SizingConfig
except ImportError:
    from cryptobot.shared.sizing.kelly import KellySizer, SizingConfig


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    
    # Capital
    initial_capital: float = 100_000.0
    
    # Execution
    slippage_bps: float = 10.0
    commission_bps: float = 10.0
    
    # Trading
    pair: str = "XBTUSD"
    pairs: List[str] = field(default_factory=lambda: ["XBTUSD"])
    min_trade_threshold: float = 0.01    # Minimum position change to trade
    
    # Sizer type
    sizer_type: str = "kelly"            # "kelly", "risk_parity", "avoid_danger"
    
    # Kelly-specific settings
    kelly_fraction: float = 0.25
    
    # Risk Parity settings (from validated backtest)
    max_exposure: float = 2.0
    target_vol: float = 0.40
    
    # Features
    feature_cols: List[str] = field(default_factory=lambda: [
        'ma_score', 'regime_hybrid_simple', 'garch_vol_simple',
        'rolling_vol_168h', 'price_vs_sma_24',
    ])
    
    # Progress
    show_progress: bool = True
    progress_interval: int = 10000       # Update progress every N bars
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_capital': self.initial_capital,
            'slippage_bps': self.slippage_bps,
            'commission_bps': self.commission_bps,
            'pair': self.pair,
            'pairs': self.pairs,
            'min_trade_threshold': self.min_trade_threshold,
            'sizer_type': self.sizer_type,
            'kelly_fraction': self.kelly_fraction,
            'max_exposure': self.max_exposure,
            'target_vol': self.target_vol,
            'feature_cols': self.feature_cols,
        }


@dataclass 
class BacktestResults:
    """Results from a backtest run."""
    
    # Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading
    total_trades: int = 0
    win_rate: float = 0.0
    avg_trade_return: float = 0.0
    
    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0
    
    # Capital
    initial_capital: float = 0.0
    final_equity: float = 0.0
    
    # Data
    equity_curve: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    actions: Optional[List[Dict]] = None
    
    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_bars: int = 0
    duration_days: float = 0.0
    sizer_type: str = ""
    
    # Risk Parity specific
    risk_parity_weights: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'total_return_pct': self.total_return * 100,
            'annualized_return': self.annualized_return,
            'annualized_return_pct': self.annualized_return * 100,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'avg_trade_return': self.avg_trade_return,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_costs': self.total_costs,
            'initial_capital': self.initial_capital,
            'final_equity': self.final_equity,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_bars': self.total_bars,
            'duration_days': self.duration_days,
            'sizer_type': self.sizer_type,
        }
    
    def print_summary(self):
        """Print results summary."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Sizer:            {self.sizer_type}")
        print(f"Period:           {self.start_date.date()} to {self.end_date.date()}")
        print(f"Duration:         {self.duration_days:.0f} days ({self.total_bars:,} bars)")
        print("-"*60)
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Final Equity:     ${self.final_equity:,.2f}")
        print(f"Total Return:     {self.total_return*100:+.2f}%")
        print(f"Annual Return:    {self.annualized_return*100:+.2f}%")
        print("-"*60)
        print(f"Sharpe Ratio:     {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:    {self.sortino_ratio:.2f}")
        print(f"Max Drawdown:     {self.max_drawdown*100:.2f}%")
        print(f"Calmar Ratio:     {self.calmar_ratio:.2f}")
        print("-"*60)
        print(f"Total Trades:     {self.total_trades}")
        print(f"Win Rate:         {self.win_rate*100:.1f}%")
        print(f"Total Costs:      ${self.total_costs:,.2f}")
        print("="*60)


class BacktestRunner:
    """
    Main backtest orchestrator.
    
    Two-phase approach for speed:
        Phase 1 (Vectorized): Pre-compute features and predictions
        Phase 2 (Event Loop): Simulate portfolio, risk, execution
    
    Supports multiple sizers:
        - RiskParitySizer for portfolio-level allocation
        - KellySizer for single-asset sizing
    
    The event loop uses shared code (TradingEngine.process_bar) that
    is identical to production, ensuring backtest/live consistency.
    """
    
    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        executor: Optional[SimulatedExecutor] = None,
        risk_manager: Optional[RiskManager] = None,
        sizer: Optional[Union[KellySizer, 'RiskParitySizer']] = None,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtest runner.
        
        Args:
            portfolio: Portfolio to use (creates new if None)
            executor: Order executor (creates default if None)
            risk_manager: Risk manager (creates default if None)
            sizer: Position sizer (creates based on config if None)
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        
        # Initialize components
        self.portfolio = portfolio or Portfolio(
            initial_capital=self.config.initial_capital
        )
        
        self.executor = executor or SimulatedExecutor(
            slippage_bps=self.config.slippage_bps,
            commission_bps=self.config.commission_bps,
        )
        
        self.risk_manager = risk_manager or RiskManager()
        
        # Create sizer based on config
        if sizer is not None:
            self.sizer = sizer
            self.sizer_type = type(sizer).__name__
        else:
            self.sizer, self.sizer_type = self._create_sizer()
        
        # Check if using portfolio-level sizer
        self.is_portfolio_sizer = RiskParitySizer is not None and isinstance(self.sizer, RiskParitySizer)
        
        # Create trading engine (only used for non-portfolio sizers)
        if not self.is_portfolio_sizer:
            self.engine = TradingEngine(
                portfolio=self.portfolio,
                risk_manager=self.risk_manager,
                sizer=self.sizer,
                min_trade_threshold=self.config.min_trade_threshold,
            )
        else:
            self.engine = None
        
        # Results storage
        self.results: Optional[BacktestResults] = None
    
    def _create_sizer(self):
        """Create sizer based on config."""
        sizer_type = self.config.sizer_type.lower()
        
        if sizer_type == "risk_parity":
            if RiskParitySizer is None:
                raise ImportError("RiskParitySizer not available. Install cryptobot.risk module.")
            
            rp_config = RiskParityConfig(
                max_exposure=self.config.max_exposure,
                target_vol=self.config.target_vol,
                pairs=self.config.pairs,
            )
            return RiskParitySizer(config=rp_config), "RiskParitySizer"
        
        elif sizer_type == "kelly":
            sizing_config = SizingConfig(kelly_fraction=self.config.kelly_fraction)
            return KellySizer(config=sizing_config), "KellySizer"
        
        else:
            # Default to Kelly
            sizing_config = SizingConfig(kelly_fraction=self.config.kelly_fraction)
            return KellySizer(config=sizing_config), "KellySizer"
    
    def run(
        self,
        df: pd.DataFrame,
        predictions: Optional[pd.Series] = None,
        features_df: Optional[pd.DataFrame] = None,
        pair: Optional[str] = None,
    ) -> BacktestResults:
        """
        Run backtest on historical data.
        
        Args:
            df: OHLCV DataFrame with DatetimeIndex
            predictions: Pre-computed predictions (optional)
            features_df: Pre-computed features DataFrame (optional)
            pair: Trading pair (overrides config)
        
        Returns:
            BacktestResults with performance metrics
        """
        pair = pair or self.config.pair
        
        # Validate inputs
        df = self._validate_data(df)
        
        # Merge predictions and features if provided
        if predictions is not None:
            df['prediction'] = predictions
        
        if features_df is not None:
            for col in features_df.columns:
                if col not in df.columns:
                    df[col] = features_df[col]
        
        # Check we have predictions
        if 'prediction' not in df.columns:
            raise ValueError("No predictions provided. Pass predictions parameter or include 'prediction' column in df.")
        
        # Drop NaN rows
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'prediction']
        df = df.dropna(subset=required_cols)
        
        if len(df) == 0:
            raise ValueError("No valid data after dropping NaN rows")
        
        # Reset components
        self.portfolio = Portfolio(initial_capital=self.config.initial_capital)
        if self.engine is not None:
            self.engine.portfolio = self.portfolio
        self.executor.reset_statistics()
        
        # Run appropriate event loop
        if self.is_portfolio_sizer:
            self._run_portfolio_event_loop(df, pair)
        else:
            self._run_event_loop(df, pair)
        
        # Calculate results
        self.results = self._calculate_results(df)
        
        return self.results
    
    def run_multi_asset(
        self,
        data: Dict[str, pd.DataFrame],
        predictions: Dict[str, pd.Series],
        features: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> BacktestResults:
        """
        Run multi-asset backtest with Risk Parity.
        
        Args:
            data: Dict of pair -> OHLCV DataFrame
            predictions: Dict of pair -> predictions Series
            features: Dict of pair -> features DataFrame (optional)
        
        Returns:
            BacktestResults with performance metrics
        """
        if not self.is_portfolio_sizer:
            raise ValueError("Multi-asset backtest requires RiskParitySizer")
        
        # Validate and prepare data
        pairs = list(data.keys())
        
        # Align all data to common index
        all_dates = None
        for pair, df in data.items():
            df = self._validate_data(df)
            if 'prediction' not in df.columns and pair in predictions:
                df['prediction'] = predictions[pair]
            if features and pair in features:
                for col in features[pair].columns:
                    if col not in df.columns:
                        df[col] = features[pair][col]
            data[pair] = df
            
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        # Sort dates
        all_dates = sorted(list(all_dates))
        
        if len(all_dates) == 0:
            raise ValueError("No overlapping dates across pairs")
        
        # Build returns DataFrame for Risk Parity
        returns_data = {}
        for pair, df in data.items():
            returns_data[pair] = df['close'].pct_change()
        returns_df = pd.DataFrame(returns_data)
        
        # Reset components
        self.portfolio = Portfolio(initial_capital=self.config.initial_capital)
        self.executor.reset_statistics()
        self.sizer.reset()
        
        # Run portfolio event loop
        self._run_multi_asset_event_loop(data, returns_df, all_dates, pairs)
        
        # Calculate results
        combined_df = list(data.values())[0].loc[all_dates]
        self.results = self._calculate_results(combined_df)
        self.results.risk_parity_weights = self.sizer.asset_weights.copy()
        
        return self.results
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data."""
        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")
        
        # Sort by time
        df = df.sort_index()
        
        return df.copy()
    
    def _run_event_loop(self, df: pd.DataFrame, pair: str):
        """
        Run the event-driven simulation loop for single-asset sizer.
        
        This is where the magic happens - we iterate bar by bar,
        using the same process_bar() logic as production.
        """
        n_bars = len(df)
        
        # Setup progress bar
        if self.config.show_progress:
            pbar = tqdm(total=n_bars, desc="Backtesting", unit="bars")
        
        # Get feature columns that exist in df
        feature_cols = [c for c in self.config.feature_cols if c in df.columns]
        
        # Main loop
        for i, (timestamp, row) in enumerate(df.iterrows()):
            # Create bar
            bar = Bar(
                timestamp=timestamp,
                pair=pair,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
            )
            
            # Get features for this bar
            features = {}
            for col in feature_cols:
                val = row.get(col)
                if pd.notna(val):
                    features[col] = val
            
            # Get prediction for this bar
            prediction = row.get('prediction')
            if pd.isna(prediction):
                prediction = None
            
            # Process bar using shared engine code
            # This is identical to what runs in production
            self.engine.process_bar(
                bar=bar,
                features=features,
                executor=self.executor,
                prediction=prediction,
            )
            
            # Update progress
            if self.config.show_progress:
                pbar.update(1)
        
        if self.config.show_progress:
            pbar.close()
    
    def _run_portfolio_event_loop(self, df: pd.DataFrame, pair: str):
        """
        Run event loop for portfolio-level sizer (single pair).
        """
        n_bars = len(df)
        
        # Setup progress bar
        if self.config.show_progress:
            pbar = tqdm(total=n_bars, desc="Backtesting (Risk Parity)", unit="bars")
        
        # Build returns for weight calculation
        returns = df['close'].pct_change()
        returns_df = pd.DataFrame({pair: returns})
        
        # Track state
        equity = self.config.initial_capital
        peak_equity = equity
        current_position = 0.0
        cash = equity
        
        equity_history = []
        
        # Main loop
        for i, (timestamp, row) in enumerate(df.iterrows()):
            price = row['close']
            prediction = row.get('prediction', 0.5)
            
            if pd.isna(prediction):
                prediction = 0.5
            
            # Convert prediction to signal (0-1)
            signal = prediction if prediction > 0.5 else 0.0
            
            # Calculate target position using Risk Parity
            self.sizer.calculate_weights(returns_df.loc[:timestamp], timestamp)
            target_position = self.sizer.calculate_position(
                pair=pair,
                signal=signal,
                weight=self.sizer.asset_weights.get(pair, 1.0),
                equity=equity,
                peak_equity=peak_equity,
                returns_df=returns_df.loc[:timestamp],
                current_date=timestamp,
                price=price,
            )
            
            # Execute trade if needed
            position_diff = target_position - current_position
            if abs(position_diff) * price > self.config.min_trade_threshold * equity:
                # Calculate costs
                trade_value = abs(position_diff) * price
                slippage = trade_value * (self.config.slippage_bps / 10000)
                commission = trade_value * (self.config.commission_bps / 10000)
                total_cost = slippage + commission
                
                # Update cash
                if position_diff > 0:  # Buying
                    cash -= (position_diff * price + total_cost)
                else:  # Selling
                    cash += (abs(position_diff) * price - total_cost)
                
                current_position = target_position
                
                # Record trade
                self.portfolio.record_trade({
                    'timestamp': timestamp,
                    'pair': pair,
                    'side': 'BUY' if position_diff > 0 else 'SELL',
                    'size': abs(position_diff),
                    'price': price,
                    'cost': total_cost,
                })
            
            # Update equity
            position_value = current_position * price
            equity = cash + position_value
            peak_equity = max(peak_equity, equity)
            
            equity_history.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': cash,
                'position_value': position_value,
            })
            
            # Update progress
            if self.config.show_progress:
                pbar.update(1)
        
        if self.config.show_progress:
            pbar.close()
        
        # Store equity curve
        self.portfolio._equity_history = equity_history
        self.portfolio._equity = equity
    
    def _run_multi_asset_event_loop(
        self,
        data: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame,
        dates: List,
        pairs: List[str],
    ):
        """
        Run event loop for multi-asset Risk Parity backtest.
        """
        n_bars = len(dates)
        
        # Setup progress bar
        if self.config.show_progress:
            pbar = tqdm(total=n_bars, desc="Backtesting (Multi-Asset Risk Parity)", unit="bars")
        
        # Track state
        equity = self.config.initial_capital
        peak_equity = equity
        current_positions = {pair: 0.0 for pair in pairs}
        cash = equity
        
        equity_history = []
        
        # Main loop
        for timestamp in dates:
            # Get prices and predictions for this timestamp
            prices = {}
            signals = {}
            
            for pair in pairs:
                df = data[pair]
                if timestamp in df.index:
                    prices[pair] = df.loc[timestamp, 'close']
                    pred = df.loc[timestamp].get('prediction', 0.5)
                    signals[pair] = pred if pred > 0.5 else 0.0
            
            # Calculate all positions using Risk Parity
            target_positions = self.sizer.calculate_all_positions(
                signals=signals,
                equity=equity,
                peak_equity=peak_equity,
                returns_df=returns_df.loc[:timestamp],
                current_date=timestamp,
                prices=prices,
            )
            
            # Execute trades
            for pair in pairs:
                if pair not in target_positions or pair not in prices:
                    continue
                
                target = target_positions[pair]
                current = current_positions[pair]
                price = prices[pair]
                
                position_diff = target - current
                if abs(position_diff) * price > self.config.min_trade_threshold * equity:
                    # Calculate costs
                    trade_value = abs(position_diff) * price
                    total_cost = trade_value * ((self.config.slippage_bps + self.config.commission_bps) / 10000)
                    
                    # Update cash
                    if position_diff > 0:
                        cash -= (position_diff * price + total_cost)
                    else:
                        cash += (abs(position_diff) * price - total_cost)
                    
                    current_positions[pair] = target
            
            # Update equity
            position_value = sum(
                current_positions[pair] * prices.get(pair, 0)
                for pair in pairs
            )
            equity = cash + position_value
            peak_equity = max(peak_equity, equity)
            
            equity_history.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': cash,
                'position_value': position_value,
            })
            
            # Update progress
            if self.config.show_progress:
                pbar.update(1)
        
        if self.config.show_progress:
            pbar.close()
        
        # Store results
        self.portfolio._equity_history = equity_history
        self.portfolio._equity = equity
    
    def _calculate_results(self, df: pd.DataFrame) -> BacktestResults:
        """Calculate backtest results and metrics."""
        results = BacktestResults()
        results.sizer_type = self.sizer_type
        
        # Get equity curve
        if hasattr(self.portfolio, '_equity_history') and self.portfolio._equity_history:
            equity_curve = pd.DataFrame(self.portfolio._equity_history)
            equity_curve = equity_curve.set_index('timestamp')
        else:
            equity_curve = self.portfolio.get_equity_curve()
        
        if len(equity_curve) == 0:
            return results
        
        # Basic info
        results.initial_capital = self.config.initial_capital
        results.final_equity = equity_curve['equity'].iloc[-1] if 'equity' in equity_curve.columns else self.config.initial_capital
        results.total_return = (results.final_equity - results.initial_capital) / results.initial_capital
        
        # Time info
        results.start_date = df.index[0]
        results.end_date = df.index[-1]
        results.total_bars = len(df)
        results.duration_days = (results.end_date - results.start_date).days
        
        # Annualized return
        if results.duration_days > 0:
            years = results.duration_days / 365.25
            if years > 0:
                results.annualized_return = (1 + results.total_return) ** (1/years) - 1
        
        # Risk metrics from equity curve
        if 'equity' in equity_curve.columns and len(equity_curve) > 1:
            returns = equity_curve['equity'].pct_change().dropna()
            
            if len(returns) > 0 and returns.std() > 0:
                # Sharpe (annualized, assuming hourly data)
                results.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(8760)
                
                # Sortino (downside deviation)
                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    results.sortino_ratio = returns.mean() / downside.std() * np.sqrt(8760)
            
            # Max drawdown
            rolling_max = equity_curve['equity'].cummax()
            drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
            results.max_drawdown = abs(drawdown.min())
            
            # Calmar ratio
            if results.max_drawdown > 0:
                results.calmar_ratio = results.annualized_return / results.max_drawdown
        
        # Trade statistics
        if hasattr(self.portfolio, 'trade_count'):
            results.total_trades = self.portfolio.trade_count
        
        # Costs
        exec_stats = self.executor.get_statistics()
        results.total_commission = exec_stats.get('total_commission', 0)
        results.total_slippage = exec_stats.get('total_slippage', 0)
        results.total_costs = exec_stats.get('total_costs', 0)
        
        # Store DataFrames
        results.equity_curve = equity_curve
        
        return results
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve from last run."""
        if hasattr(self.portfolio, '_equity_history') and self.portfolio._equity_history:
            return pd.DataFrame(self.portfolio._equity_history).set_index('timestamp')
        return self.portfolio.get_equity_curve()
    
    def get_trades(self) -> pd.DataFrame:
        """Get trades from last run."""
        return self.portfolio.get_trades()
    
    def get_actions(self) -> List[Dict]:
        """Get action history from last run."""
        if self.engine:
            return self.engine.get_action_history()
        return []
    
    def __repr__(self) -> str:
        return (
            f"BacktestRunner(capital=${self.config.initial_capital:,.0f}, "
            f"sizer={self.sizer_type}, "
            f"slippage={self.config.slippage_bps}bps)"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    predictions: pd.Series,
    features_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 100_000,
    slippage_bps: float = 10,
    commission_bps: float = 10,
    kelly_fraction: float = 0.25,
    pair: str = "XBTUSD",
    show_progress: bool = True,
) -> BacktestResults:
    """
    Convenience function to run a backtest with Kelly sizer.
    
    Args:
        df: OHLCV DataFrame
        predictions: Model predictions (P(up))
        features_df: Features for sizing/risk (optional)
        initial_capital: Starting capital
        slippage_bps: Slippage in basis points
        commission_bps: Commission in basis points
        kelly_fraction: Kelly fraction for sizing
        pair: Trading pair
        show_progress: Show progress bar
    
    Returns:
        BacktestResults
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        pair=pair,
        sizer_type="kelly",
        kelly_fraction=kelly_fraction,
        show_progress=show_progress,
    )
    
    runner = BacktestRunner(config=config)
    
    return runner.run(df, predictions=predictions, features_df=features_df)


def run_risk_parity_backtest(
    df: pd.DataFrame,
    predictions: pd.Series,
    features_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 100_000,
    slippage_bps: float = 10,
    commission_bps: float = 10,
    max_exposure: float = 2.0,
    target_vol: float = 0.40,
    pair: str = "XBTUSD",
    show_progress: bool = True,
) -> BacktestResults:
    """
    Convenience function to run a backtest with Risk Parity sizer.
    
    Uses the validated parameters from 16state_combined_backtest.py.
    
    Args:
        df: OHLCV DataFrame
        predictions: Model predictions (P(up))
        features_df: Features (optional)
        initial_capital: Starting capital
        slippage_bps: Slippage in basis points
        commission_bps: Commission in basis points
        max_exposure: Maximum exposure (2.0 = 200%)
        target_vol: Target volatility (0.40 = 40%)
        pair: Trading pair
        show_progress: Show progress bar
    
    Returns:
        BacktestResults
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        pair=pair,
        pairs=[pair],
        sizer_type="risk_parity",
        max_exposure=max_exposure,
        target_vol=target_vol,
        show_progress=show_progress,
    )
    
    runner = BacktestRunner(config=config)
    
    return runner.run(df, predictions=predictions, features_df=features_df)


def quick_backtest(
    df: pd.DataFrame,
    predictions: pd.Series,
    **kwargs,
) -> BacktestResults:
    """
    Quick backtest with minimal setup.
    
    Just provide data and predictions, uses defaults for everything else.
    """
    return run_backtest(df, predictions, **kwargs)