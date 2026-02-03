#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryptoBot - Live/Paper Trading Runner
======================================
FILE: apps/live/trader.py

Trading runner using pluggable Strategy + Executor pattern.

Strategies:
    - momentum: Validated 16-state momentum (2.5% buffer, 3h confirm, S12/14 exclusion)

Execution Modes:
    - paper: Simulated execution with slippage
    - live: Real execution via Kraken API

Validated Performance (momentum strategy):
    - Annual Return: +58.6%
    - Sharpe Ratio: 3.04
    - Max Drawdown: -12.2%

Usage:
    python -m apps.live.trader --strategy momentum --mode paper
    python -m apps.live.trader --strategy momentum --mode live
    python -m apps.live.trader --strategy momentum --mode paper --hourly
    python -m apps.live.trader --dry-run

Scheduling:
    --hourly flag: Update data first, then run trading logic.
    
    The 3h confirmation filter is inside MomentumStrategy. To catch signals
    when they confirm, trading logic MUST run every hour. The strategy
    internally tracks state persistence and only triggers trades when
    confirmation completes.
    
    Cron example (run every hour):
        0 * * * * cd ~/cryptobot && python -m apps.live.trader --strategy momentum --mode paper --hourly

    Daily reports are sent at midnight Melbourne time.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from cryptobot.data.database import Database
from cryptobot.data.kraken import KrakenAPI
from cryptobot.risk.risk_parity import RiskParitySizer, RiskParityConfig
from cryptobot.risk.var import calculate_var
from cryptobot.reports import DailyReport, ReportSender

# Strategy and Executor imports
from cryptobot.strategies import MomentumStrategy
from cryptobot.executors import PaperExecutor, LiveExecutor
from cryptobot.types.bar import Bar
from cryptobot.types.order import Order

# Melbourne timezone for display
try:
    from zoneinfo import ZoneInfo
    MELBOURNE_TZ = ZoneInfo('Australia/Melbourne')
except ImportError:
    import pytz
    MELBOURNE_TZ = pytz.timezone('Australia/Melbourne')


def to_melbourne(dt: datetime) -> datetime:
    """Convert datetime to Melbourne timezone for display."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MELBOURNE_TZ)


def melbourne_now() -> datetime:
    """Get current time in Melbourne timezone."""
    return datetime.now(MELBOURNE_TZ)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: str = None, level: str = "INFO"):
    """Configure logging to console and optional file."""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger('trader')


# =============================================================================
# TRADE CLASSIFICATION
# =============================================================================

def classify_trade(current_position: float, target_position: float) -> str:
    """Classify trade as ENTRY/EXIT/INCREASE/DECREASE."""
    had = abs(current_position) > 0.0001
    has = abs(target_position) > 0.0001
    
    if not had and has:
        return 'ENTRY'
    elif had and not has:
        return 'EXIT'
    elif target_position > current_position:
        return 'INCREASE'
    else:
        return 'DECREASE'


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


# =============================================================================
# DATA UPDATE (runs every hour)
# =============================================================================

def update_data(config: dict, logger: logging.Logger) -> int:
    """
    Fetch latest OHLCV data from Kraken for all pairs.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Total rows inserted
    """
    logger.info("-" * 40)
    logger.info("UPDATING OHLCV DATA")
    logger.info("-" * 40)
    
    api = KrakenAPI()
    total_rows = 0
    
    for pair in config['pairs']:
        try:
            rows = api.fetch_and_store(pair)
            logger.info(f"  {pair}: +{rows} rows")
            total_rows += rows
        except Exception as e:
            logger.error(f"  {pair}: FAILED - {e}")
    
    logger.info(f"Data update complete: +{total_rows} total rows")
    return total_rows


# =============================================================================
# RETURNS DATA BUILDER
# =============================================================================

def build_returns_df(db: Database, pairs: List[str], lookback_days: int = 90) -> pd.DataFrame:
    """
    Build DataFrame of daily returns for Risk Parity weight calculation.
    
    Args:
        db: Database instance
        pairs: List of trading pairs
        lookback_days: Days of history to load
    
    Returns:
        DataFrame with columns = pairs, index = date, values = daily returns
    """
    returns_data = {}
    start_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    
    for pair in pairs:
        try:
            df = db.get_ohlcv(pair, start=start_date)
            if df is not None and len(df) > 0:
                daily = df['close'].resample('24h').last().ffill()
                returns = daily.pct_change().dropna()
                returns_data[pair] = returns
        except Exception as e:
            logging.getLogger('trader').warning(f"Could not load returns for {pair}: {e}")
    
    if not returns_data:
        return pd.DataFrame()
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna(how='all')
    
    return returns_df


# =============================================================================
# TRADING RUNNER
# =============================================================================

class TradingRunner:
    """
    Trading runner using pluggable Strategy + Executor pattern.
    
    Strategy determines WHAT to trade (signal generation).
    Executor determines HOW to trade (paper vs live).
    
    This separation allows:
    - Same strategy for paper and live trading
    - Easy addition of new strategies
    - Clean testing of strategies before going live
    """
    
    def __init__(
        self, 
        config: dict, 
        dry_run: bool = False,
        strategy_name: str = 'momentum',
        mode: str = 'paper'
    ):
        """
        Initialize the runner.
        
        Args:
            config: Configuration dictionary
            dry_run: If True, don't write to database
            strategy_name: Which strategy to use ('momentum', etc.)
            mode: Execution mode ('paper' or 'live')
        """
        self.config = config
        self.dry_run = dry_run
        self.mode = mode
        self.strategy_name = strategy_name
        
        self.pairs = config['pairs']
        self.initial_capital = config['position']['initial_capital']
        self.lookback_days = config['data']['lookback_days']
        self.trading_cost_bps = config['execution']['trading_cost_bps']
        self.slippage_bps = config['execution'].get('slippage_bps', 10)
        
        # Initialize database
        self.db = Database()
        self.logger = logging.getLogger('trader')
        
        # Initialize strategy
        if strategy_name == 'momentum':
            self.strategy = MomentumStrategy()
            self.logger.info(f"Strategy: MomentumStrategy (validated 16-state)")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Initialize executor
        if mode == 'paper':
            self.executor = PaperExecutor(
                slippage_bps=self.slippage_bps,
                commission_bps=self.trading_cost_bps,
            )
            self.logger.info(f"Executor: PaperExecutor (simulated)")
        elif mode == 'live':
            self.executor = LiveExecutor(KrakenAPI())
            self.logger.info(f"Executor: LiveExecutor (REAL TRADING)")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Initialize Risk Parity Sizer
        try:
            risk_parity_config_path = PROJECT_ROOT / "cryptobot" / "configs" / "risk" / "risk_parity.yaml"
            if risk_parity_config_path.exists():
                self.sizer_config = RiskParityConfig.from_yaml(str(risk_parity_config_path))
                self.sizer_config.pairs = self.pairs
            else:
                self.sizer_config = RiskParityConfig(pairs=self.pairs)
        except Exception as e:
            self.logger.warning(f"Could not load risk_parity.yaml: {e}, using defaults")
            self.sizer_config = RiskParityConfig(pairs=self.pairs)
        
        self.sizer = RiskParitySizer(config=self.sizer_config)
        
        # Track results for this run
        self.signals_generated = []
        self.trades_executed = []
        self.risk_parity_weights = {}
    
    def run(self) -> dict:
        """
        Execute the trading run.
        
        Returns:
            Summary dict with results
        """
        run_time = datetime.now(timezone.utc)
        run_time_melb = to_melbourne(run_time)
        
        self.logger.info("=" * 60)
        self.logger.info(f"TRADING RUN - {run_time_melb.strftime('%Y-%m-%d %H:%M:%S')} Melbourne")
        self.logger.info(f"Strategy: {self.strategy_name} | Mode: {self.mode.upper()}")
        self.logger.info("=" * 60)
        
        if self.dry_run:
            self.logger.info("*** DRY RUN MODE - No database writes ***")
        
        # Load persisted strategy state
        loaded = self.strategy.load_state_for_pairs(self.db, self.pairs)
        self.logger.info(f"Loaded strategy state for {loaded} pairs")
        
        # Get current equity (or initialize)
        equity = self._get_or_init_equity(run_time)
        self.logger.info(f"Current equity: ${equity['total_equity']:,.2f}")
        self.logger.info(f"Peak equity: ${equity['peak_equity']:,.2f}")
        
        # Build returns DataFrame for Risk Parity
        self.logger.info("-" * 40)
        self.logger.info("Building returns data for Risk Parity...")
        returns_df = build_returns_df(self.db, self.pairs, lookback_days=90)
        
        if len(returns_df) < 20:
            self.logger.warning("Insufficient return data for Risk Parity weights - using equal weights")
        else:
            self.logger.info(f"Returns data: {len(returns_df)} days, {len(returns_df.columns)} pairs")
        
        # Calculate Risk Parity weights
        current_date = pd.Timestamp(run_time)
        self.risk_parity_weights = self.sizer.calculate_weights(returns_df, current_date)
        
        if not self.dry_run:
            self.db.record_weights(self.risk_parity_weights, timestamp=run_time)
        
        self.logger.info("-" * 40)
        self.logger.info("RISK PARITY WEIGHTS:")
        for pair, weight in self.risk_parity_weights.items():
            self.logger.info(f"  {pair}: {weight:.1%}")
        
        # Log sizer scalars
        vol_scalar = self.sizer.calculate_vol_scalar(returns_df, current_date)
        dd_scalar = self.sizer.calculate_dd_scalar(equity['total_equity'], equity['peak_equity'])
        self.logger.info(f"Vol scalar: {vol_scalar:.2f}, DD scalar: {dd_scalar:.2f}")
        
        # Collect signals and prices for all pairs first
        pair_data = {}
        for pair in self.pairs:
            self.logger.info("-" * 40)
            self.logger.info(f"Processing {pair}...")
            
            try:
                data = self._get_pair_signal(pair, run_time)
                if data:
                    pair_data[pair] = data
            except Exception as e:
                self.logger.error(f"Error processing {pair}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                continue
        
        # Calculate all positions using Risk Parity
        signals = {pair: data['signal_position'] for pair, data in pair_data.items()}
        prices = {pair: data['price'] for pair, data in pair_data.items()}
        
        target_positions = self.sizer.calculate_all_positions(
            signals=signals,
            equity=equity['total_equity'],
            peak_equity=equity['peak_equity'],
            returns_df=returns_df,
            current_date=current_date,
            prices=prices,
        )
        
        # Execute trades for each pair
        for pair, data in pair_data.items():
            target_position = target_positions.get(pair, 0.0)
            self._process_pair_trade(
                pair=pair,
                target_position=target_position,
                price=data['price'],
                signal_type=data['signal_type'],
                state=data['state'],
                duration=data['duration'],
                equity=equity,
                run_time=run_time
            )
        
        
        # Summary
        summary = self._generate_summary()
        self._log_summary(summary)
        
        # Save strategy state for next run
        if not self.dry_run:
            saved = self.strategy.save_state(self.db)
            self.logger.info(f"Saved strategy state for {saved} pairs")
        
        return summary
    
    def _get_or_init_equity(self, run_time: datetime) -> dict:
        """Get current equity or initialize with starting capital."""
        
        equity = self.db.get_latest_equity()
        
        if equity is None:
            self.logger.info("No existing equity - initializing with starting capital")
            equity = {
                'total_equity': self.initial_capital,
                'cash': self.initial_capital,
                'invested': 0.0,
                'daily_pnl': 0.0,
                'drawdown': 0.0,
                'peak_equity': self.initial_capital
            }
            
            if not self.dry_run:
                self.db.record_equity(**equity, timestamp=run_time)
        
        return equity
    
    def _get_pair_signal(self, pair: str, run_time: datetime) -> Optional[dict]:
        """
        Get signal data for a single pair using the strategy.
        
        Returns:
            Dict with signal info, or None if insufficient data
        """
        # 1. Fetch data
        start_date = run_time - timedelta(days=self.lookback_days)
        df_1h = self.db.get_ohlcv(pair, start=start_date.isoformat())
        
        if df_1h is None or len(df_1h) < 168:  # Need at least 1 week of hourly data
            self.logger.warning(f"{pair}: Insufficient data ({len(df_1h) if df_1h is not None else 0} rows)")
            return None
        
        self.logger.info(f"{pair}: Loaded {len(df_1h)} hourly bars")
        
        # 2. Compute MAs for strategy
        # Resample to get multi-timeframe MAs
        df_24h = df_1h['close'].resample('24h').last().ffill()
        df_72h = df_1h['close'].resample('72h').last().ffill()
        df_168h = df_1h['close'].resample('168h').last().ffill()
        
        ma_24h = df_24h.rolling(16).mean().iloc[-1]
        ma_72h = df_72h.rolling(6).mean().iloc[-1]
        ma_168h = df_168h.rolling(2).mean().iloc[-1]
        current_price = df_1h['close'].iloc[-1]
        
        # 3. Get signal from strategy
        features = {
            'pair': pair,
            'price': current_price,
            'ma_24h': ma_24h,
            'ma_72h': ma_72h,
            'ma_168h': ma_168h,
        }
        
        position_multiplier = self.strategy.predict(features)
        
        # Get state info for logging
        pair_state = self.strategy.get_pair_state(pair)
        state_num = pair_state.confirmed_state if pair_state else 0
        duration = pair_state.duration_hours if pair_state else 0
        
        # Map multiplier to signal type
        if position_multiplier == 0:
            if pair_state and pair_state.confirmed_state in {12, 14}:
                signal_type = "EXCLUDED"
            elif pair_state and pair_state.confirmed_state < 8:
                signal_type = "BEARISH"
            else:
                signal_type = "FLAT"
        elif position_multiplier == 1.5:
            signal_type = "BOOSTED_LONG"
        else:
            signal_type = "LONG"
        
        # Determine action description
        if position_multiplier == 1.5:
            action = "🚀 BOOSTED LONG (150%)"
        elif position_multiplier == 1.0:
            action = "✅ LONG (100%)"
        elif pair_state and pair_state.confirmed_state in {12, 14}:
            action = "⛔ EXCLUDED (exhaustion state)"
        else:
            action = "⏸️ FLAT (bearish)"
        
        self.logger.info(
            f"{pair}: {action} | State={state_num}, Duration={duration}h"
        )
        
        return {
            'signal_position': position_multiplier,
            'signal_type': signal_type,
            'price': current_price,
            'state': state_num,
            'duration': duration,
        }
    
    def _process_pair_trade(
        self,
        pair: str,
        target_position: float,
        price: float,
        signal_type: str,
        state: int,
        duration: int,
        equity: dict,
        run_time: datetime,
    ):
        """Process trade for a single pair after position sizing."""
        
        # Record signal
        if not self.dry_run:
            self.db.record_signal(
                pair=pair,
                strategy=self.strategy_name,
                signal=signal_type,
                target_position=target_position,
                confidence=0.55 if target_position > 0 else 0.45,
                regime=state,
                prediction=0.55 if target_position > 0 else 0.45,
                timestamp=run_time
            )
        
        self.signals_generated.append({
            'pair': pair,
            'signal': signal_type,
            'state': state,
            'duration': duration,
            'target_position': target_position,
            'weight': self.risk_parity_weights.get(pair, 0),
        })
        
        # Get current position
        current_pos = self.db.get_position(pair)
        current_position = current_pos['position'] if current_pos else 0.0
        
        # Check if trade needed
        position_diff = target_position - current_position
        
        if abs(position_diff) > 0.001:  # Threshold to avoid tiny trades
            self._execute_trade(
                pair=pair,
                current_position=current_position,
                target_position=target_position,
                price=price,
                signal_type=signal_type,
                run_time=run_time
            )
        else:
            self.logger.info(f"{pair}: No trade needed (position unchanged)")
        
        # Update portfolio snapshot
        if not self.dry_run:
            entry_price = current_pos['entry_price'] if current_pos and current_pos.get('entry_price') else price
            unrealized_pnl = target_position * (price - entry_price) if target_position != 0 and entry_price else 0
            
            self.db.record_portfolio_snapshot(
                pair=pair,
                position=target_position,
                entry_price=entry_price if target_position != 0 else None,
                current_price=price,
                unrealized_pnl=unrealized_pnl,
                timestamp=run_time
            )
    
    def _execute_trade(
        self,
        pair: str,
        current_position: float,
        target_position: float,
        price: float,
        signal_type: str,
        run_time: datetime
    ):
        """Execute trade via executor (paper or live)."""
        
        # Create order
        order = Order.from_target_position(
            pair=pair,
            current_position=current_position,
            target_position=target_position,
            timestamp=run_time,
            reference_price=price,
            reason=signal_type,
        )
        
        if order is None:
            return  # No position change needed
        
        # Create bar for executor
        bar = Bar(
            timestamp=run_time,
            pair=pair,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0,
        )
        
        # Execute via executor (paper or live - same interface!)
        fill = self.executor.execute(order, bar)
        
        # Record to database
        if not self.dry_run:
            trade_type = classify_trade(current_position, target_position)
            self.db.record_trade(
                pair=pair,
                strategy=self.strategy_name,
                direction=order.side.value.upper(),
                size=fill.fill_size,
                price=fill.fill_price,
                slippage_bps=self.slippage_bps,
                transaction_cost=fill.total_cost,
                execution_type=self.mode,
                notes=f"Signal: {signal_type}, Weight: {self.risk_parity_weights.get(pair, 0):.1%}",
                trade_type=trade_type
            )
        
        self.trades_executed.append({
            'pair': pair,
            'direction': order.side.value.upper(),
            'size': fill.fill_size,
            'price': fill.fill_price,
            'value': fill.notional_value,
            'cost': fill.total_cost,
            'trading_cost': fill.commission,
            'slippage': fill.slippage,
        })
    
    def _update_equity(self, equity: dict, run_time: datetime):
        """Update equity snapshot after all trades."""
        
        if self.dry_run:
            return
        
        # Get all current positions
        positions_df = self.db.get_current_positions()
        
        # Calculate invested value
        invested = 0.0
        
        if positions_df is not None and len(positions_df) > 0:
            for _, row in positions_df.iterrows():
                if row['position'] and row['position'] != 0:
                    live_price = self.db.get_latest_price(row['pair'])
                    if live_price:
                        invested += abs(row['position'] * live_price)
                    elif row['current_price']:
                        invested += abs(row['position'] * row['current_price'])
        
        # Calculate cash change from today's trades
        cash_change = 0.0
        for t in self.trades_executed:
            if t['direction'] == 'BUY':
                cash_change -= t['value']
            else:
                cash_change += t['value']
            cash_change -= t['cost']
        
        # Update equity
        cash = equity['cash'] + cash_change
        total_equity = cash + invested
        
        # Track peak and drawdown
        peak_equity = max(equity.get('peak_equity', total_equity), total_equity)
        drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0
        
        # Calculate daily PnL
        prev_equity = equity.get('total_equity', total_equity)
        daily_pnl = total_equity - prev_equity
        
        self.db.record_equity(
            total_equity=total_equity,
            cash=cash,
            invested=invested,
            daily_pnl=daily_pnl,
            drawdown=drawdown,
            peak_equity=peak_equity,
            timestamp=run_time
        )
        
        self.logger.info(f"Equity updated: ${total_equity:,.2f} (PnL: ${daily_pnl:+,.2f})")
    
    def _generate_summary(self) -> dict:
        """Generate run summary."""
        return {
            'run_time': datetime.now(timezone.utc).isoformat(),
            'strategy': self.strategy_name,
            'mode': self.mode,
            'pairs_processed': len(self.signals_generated),
            'signals': self.signals_generated,
            'trades_executed': len(self.trades_executed),
            'trades': self.trades_executed,
            'total_trade_value': sum(t['value'] for t in self.trades_executed),
            'total_costs': sum(t['cost'] for t in self.trades_executed),
            'total_trading_costs': sum(t.get('trading_cost', 0) for t in self.trades_executed),
            'total_slippage': sum(t.get('slippage', 0) for t in self.trades_executed),
            'risk_parity_weights': self.risk_parity_weights.copy(),
            'sizer_stats': self.sizer.get_stats(),
            'active_boosts': self.strategy.get_active_boosts() if hasattr(self.strategy, 'get_active_boosts') else [],
            'excluded_pairs': self.strategy.get_excluded_pairs() if hasattr(self.strategy, 'get_excluded_pairs') else [],
        }
    
    def _log_summary(self, summary: dict):
        """Log run summary."""
        self.logger.info("=" * 60)
        self.logger.info("RUN SUMMARY")
        self.logger.info("=" * 60)
        
        # Get current equity and positions
        equity = self.db.get_latest_equity()
        positions_df = self.db.get_current_positions()
        
        # Calculate invested value
        invested_value = 0.0
        if positions_df is not None and len(positions_df) > 0:
            for _, p in positions_df.iterrows():
                if p['position'] and p['current_price']:
                    invested_value += abs(p['position'] * p['current_price'])
        
        # EQUITY SECTION
        self.logger.info("-" * 40)
        self.logger.info("EQUITY:")
        if equity:
            cash_balance = equity['total_equity'] - invested_value
            daily_pnl = equity.get('daily_pnl', 0) or 0
            self.logger.info(f"  Total Equity Value:     ${equity['total_equity']:>12,.2f}")
            self.logger.info(f"  Cash Balance:           ${cash_balance:>12,.2f}")
            self.logger.info(f"  Invested Value:         ${invested_value:>12,.2f}")
            self.logger.info(f"  Capital Deployed Today: ${summary['total_trade_value']:>12,.2f}")
            self.logger.info(f"  Daily P&L (24h):        ${daily_pnl:>+12,.2f}")
        
        # RISK METRICS SECTION
        self.logger.info("-" * 40)
        self.logger.info("RISK METRICS:")
        if equity:
            drawdown_pct = (equity.get('drawdown') or 0) * 100
            self.logger.info(f"  Current Drawdown:       {drawdown_pct:>10.2f}%")
            self.logger.info(f"  Max Allowed DD:         {self.config['risk']['max_drawdown']*100:>10.1f}%")
        
        # VaR
        var_metrics = calculate_var(self.db, self.pairs)
        self.logger.info(f"  VaR (95%, 1-day):       ${var_metrics.get('var_95', 0):>10,.2f}")
        self.logger.info(f"  VaR (99%, 1-day):       ${var_metrics.get('var_99', 0):>10,.2f}")
        self.logger.info(f"  Portfolio Vol:          {var_metrics.get('portfolio_vol', 0)*100:>10.1f}% ann.")
        
        # TRADES SECTION
        self.logger.info("-" * 40)
        self.logger.info("TODAY'S TRADES:")
        self.logger.info(f"  Trades Executed:        {summary['trades_executed']}")
        self.logger.info(f"  Total Value:            ${summary['total_trade_value']:>12,.2f}")
        self.logger.info(f"  Trading Costs:          ${summary.get('total_trading_costs', 0):>12,.2f}")
        self.logger.info(f"  Slippage:               ${summary.get('total_slippage', 0):>12,.2f}")
        
        if summary.get('trades'):
            for t in summary['trades']:
                self.logger.info(f"  • {t['pair']}: {t['direction']} {t['size']:.6f} @ ${t['price']:,.2f}")
        
        # POSITIONS SECTION
        self.logger.info("-" * 40)
        self.logger.info("CURRENT POSITIONS:")
        if positions_df is not None and len(positions_df) > 0:
            has_positions = False
            for _, p in positions_df.iterrows():
                if p['position'] and abs(p['position']) > 0.0001:
                    has_positions = True
                    value = p['position'] * p['current_price'] if p['current_price'] else 0
                    pnl = p['unrealized_pnl'] if p['unrealized_pnl'] else 0
                    self.logger.info(f"  {p['pair']:8s}: {p['position']:>12.4f} (${value:>10,.2f}, P&L: ${pnl:>+8,.2f})")
            if not has_positions:
                self.logger.info("  No open positions")
        else:
            self.logger.info("  No open positions")
        
        # SIGNALS SECTION
        self.logger.info("-" * 40)
        self.logger.info("SIGNALS:")
        if self.signals_generated:
            for sig in self.signals_generated:
                if sig['signal'] == 'BOOSTED_LONG':
                    status = "🚀"
                elif sig['signal'] == 'LONG':
                    status = "✅"
                elif sig['signal'] == 'EXCLUDED':
                    status = "⛔"
                else:
                    status = "⏸️"
                self.logger.info(
                    f"  {status} {sig['pair']:8s}: {sig['signal']:12s} "
                    f"(S{sig['state']}, {sig['duration']}h, Target: {sig['target_position']:.4f})"
                )
        
        # BOOST STATUS
        if summary.get('active_boosts'):
            self.logger.info("-" * 40)
            self.logger.info(f"🚀 ACTIVE BOOSTS: {', '.join(summary['active_boosts'])}")
        
        if summary.get('excluded_pairs'):
            self.logger.info("-" * 40)
            self.logger.info(f"⛔ EXCLUDED (S12/14): {', '.join(summary['excluded_pairs'])}")
        
        # RISK PARITY WEIGHTS
        self.logger.info("-" * 40)
        self.logger.info("RISK PARITY WEIGHTS:")
        for pair, weight in self.risk_parity_weights.items():
            self.logger.info(f"  {pair:8s}: {weight:>6.1%}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='CryptoBot Trading Runner (Strategy + Executor Pattern)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file (default: apps/live/config.yaml)'
    )
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='momentum',
        choices=['momentum'],
        help='Trading strategy (default: momentum)'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='paper',
        choices=['paper', 'live'],
        help='Execution mode (default: paper)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Dry run mode - no database writes'
    )
    parser.add_argument(
        '--hourly',
        action='store_true',
        help='Hourly mode: update data first, then run trading logic'
    )
    parser.add_argument(
        '--report-hour',
        type=int,
        default=0,
        help='Hour (Melbourne time) to send daily report (default: 0 = midnight)'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Setup logging
    log_file = config.get('logging', {}).get('file')
    log_level = config.get('logging', {}).get('level', 'INFO')
    logger = setup_logging(log_file, log_level)
    
    # Get current hour in Melbourne for report decision
    current_hour_melbourne = melbourne_now().hour
    
    logger.info("=" * 60)
    logger.info(f"CryptoBot Runner Started: {melbourne_now().strftime('%Y-%m-%d %H:%M:%S')} Melbourne")
    logger.info(f"Strategy: {args.strategy} | Mode: {args.mode.upper()}")
    logger.info(f"Schedule: {'HOURLY' if args.hourly else 'DIRECT'}")
    logger.info("=" * 60)
    
    try:
        # STEP 1: Update data if --hourly flag is set
        if args.hourly:
            update_data(config, logger)
        
        # STEP 2: Always run trading logic
        # The 3h confirmation is inside MomentumStrategy - we must check every hour
        # to catch when confirmation triggers
        logger.info("-" * 40)
        logger.info("RUNNING TRADING LOGIC...")
        logger.info("-" * 40)
        
        runner = TradingRunner(
            config,
            dry_run=args.dry_run,
            strategy_name=args.strategy,
            mode=args.mode,
        )
        summary = runner.run()
        
        # STEP 3: Send daily report at specified hour only
        if args.hourly and current_hour_melbourne == args.report_hour:
            if config.get('notifications', {}).get('enabled'):
                logger.info("-" * 40)
                logger.info(f"SENDING DAILY REPORT (hour={args.report_hour})")
                logger.info("-" * 40)
                report = DailyReport(runner.db, config)
                sender = ReportSender(config)
                sender.send_all(
                    report.generate(),
                    subject=f"CryptoBot {args.strategy.title()} Report - {melbourne_now().strftime('%Y-%m-%d')}"
                )
        
        logger.info("=" * 60)
        logger.info("Runner completed successfully")
        logger.info("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Runner failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()