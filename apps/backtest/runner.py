# -*- coding: utf-8 -*-
"""
CryptoBot - Backtest Runner
============================
Main orchestrator for event-driven backtesting.

Runs TradingEngine over historical data with:
    - Pre-computed features and predictions (vectorized, fast)
    - Event loop for realistic portfolio/risk simulation
    - Comprehensive metrics and reporting

Usage:
    from cryptobot.research.backtest import BacktestRunner, SimulatedExecutor
    from cryptobot.shared.core import Portfolio, TradingEngine
    from cryptobot.shared.risk import RiskManager
    from cryptobot.shared.sizing import KellySizer
    
    # Setup
    portfolio = Portfolio(initial_capital=100_000)
    executor = SimulatedExecutor(slippage_bps=10, commission_bps=10)
    risk = RiskManager()
    sizer = KellySizer()
    
    # Create runner
    runner = BacktestRunner(
        portfolio=portfolio,
        executor=executor,
        risk_manager=risk,
        sizer=sizer,
    )
    
    # Run backtest
    results = runner.run(df, predictions, features)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from tqdm import tqdm

from cryptobot.shared.core.bar import Bar
from cryptobot.shared.core.order import Order
from cryptobot.shared.core.portfolio import Portfolio
from cryptobot.shared.core.engine import TradingEngine, Action, ActionType
from cryptobot.shared.risk.manager import RiskManager
from cryptobot.shared.sizing.kelly import KellySizer
from cryptobot.research.backtest.executor import SimulatedExecutor


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
    min_trade_threshold: float = 0.01    # Minimum position change to trade
    
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
            'min_trade_threshold': self.min_trade_threshold,
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
        }
    
    def print_summary(self):
        """Print results summary."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
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
    
    The event loop uses shared code (TradingEngine.process_bar) that
    is identical to production, ensuring backtest/live consistency.
    """
    
    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        executor: Optional[SimulatedExecutor] = None,
        risk_manager: Optional[RiskManager] = None,
        sizer: Optional[KellySizer] = None,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtest runner.
        
        Args:
            portfolio: Portfolio to use (creates new if None)
            executor: Order executor (creates default if None)
            risk_manager: Risk manager (creates default if None)
            sizer: Position sizer (creates default if None)
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
        self.sizer = sizer or KellySizer()
        
        # Create trading engine
        self.engine = TradingEngine(
            portfolio=self.portfolio,
            risk_manager=self.risk_manager,
            sizer=self.sizer,
            min_trade_threshold=self.config.min_trade_threshold,
        )
        
        # Results storage
        self.results: Optional[BacktestResults] = None
    
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
        self.engine.portfolio = self.portfolio
        self.executor.reset_statistics()
        
        # Run event loop
        self._run_event_loop(df, pair)
        
        # Calculate results
        self.results = self._calculate_results(df)
        
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
        Run the event-driven simulation loop.
        
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
    
    def _calculate_results(self, df: pd.DataFrame) -> BacktestResults:
        """Calculate backtest results and metrics."""
        results = BacktestResults()
        
        # Get equity curve
        equity_curve = self.portfolio.get_equity_curve()
        trades_df = self.portfolio.get_trades()
        
        if len(equity_curve) == 0:
            return results
        
        # Basic info
        results.initial_capital = self.config.initial_capital
        results.final_equity = self.portfolio.equity
        results.total_return = self.portfolio.total_return
        
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
        results.max_drawdown = self.portfolio.max_drawdown
        
        # Trade statistics
        results.total_trades = self.portfolio.trade_count
        
        if len(trades_df) > 0:
            # Calculate per-trade P&L
            # This is simplified - for accurate win rate, need to track round trips
            results.win_rate = 0.5  # Placeholder
            results.avg_trade_return = 0.0
        
        # Costs
        exec_stats = self.executor.get_statistics()
        results.total_commission = exec_stats['total_commission']
        results.total_slippage = exec_stats['total_slippage']
        results.total_costs = exec_stats['total_costs']
        
        # Store DataFrames
        results.equity_curve = equity_curve
        results.trades = trades_df
        results.actions = self.engine.get_action_history()
        
        return results
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve from last run."""
        return self.portfolio.get_equity_curve()
    
    def get_trades(self) -> pd.DataFrame:
        """Get trades from last run."""
        return self.portfolio.get_trades()
    
    def get_actions(self) -> List[Dict]:
        """Get action history from last run."""
        return self.engine.get_action_history()
    
    def __repr__(self) -> str:
        return (
            f"BacktestRunner(capital=${self.config.initial_capital:,.0f}, "
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
    Convenience function to run a backtest.
    
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
    
    Example:
        results = run_backtest(
            df=ohlcv_data,
            predictions=model.predict(X),
            initial_capital=100_000,
            kelly_fraction=0.25,
        )
        results.print_summary()
    """
    from cryptobot.shared.sizing.kelly import SizingConfig
    
    config = BacktestConfig(
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        pair=pair,
        show_progress=show_progress,
    )
    
    sizing_config = SizingConfig(kelly_fraction=kelly_fraction)
    sizer = KellySizer(config=sizing_config)
    
    runner = BacktestRunner(config=config, sizer=sizer)
    
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
