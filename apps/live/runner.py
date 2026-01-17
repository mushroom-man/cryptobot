#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryptoBot - Paper Trading Runner
=================================
Hourly runner for the 16-state strategy.

Usage:
    python -m apps.live.runner              # Run trading once (manual)
    python -m apps.live.runner --dry-run    # Show what would happen
    python -m apps.live.runner --hourly     # Hourly mode: update data, trade at 00:xx UTC

Designed to run hourly via systemd timer with --hourly flag.
"""

import sys
import os
import argparse
import logging
import smtplib
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from cryptobot.data.database import Database
from cryptobot.data.kraken import KrakenAPI
from cryptobot.signals.generator import SignalGenerator

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
        force=True  # Override any existing config
    )
    
    return logging.getLogger('paper_trader')


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
# VALUE AT RISK CALCULATION
# =============================================================================

def calculate_var(db: Database, pairs: List[str], confidence: float = 0.95, horizon_days: int = 1) -> Dict:
    """
    Calculate Value at Risk for current portfolio.
    
    Args:
        db: Database instance
        pairs: List of trading pairs
        confidence: Confidence level (0.95 = 95%)
        horizon_days: Time horizon in days
    
    Returns:
        Dict with VaR metrics
    """
    try:
        positions_df = db.get_current_positions()
        equity_data = db.get_latest_equity()
        
        if positions_df is None or len(positions_df) == 0:
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
        
        # Get historical returns for VaR calculation
        returns_data = {}
        for pair in pairs:
            df = db.get_ohlcv(pair, start=(datetime.utcnow() - timedelta(days=365)).isoformat())
            if df is not None and len(df) > 0:
                # Daily returns from hourly data
                daily = df['close'].resample('24h').last().pct_change().dropna()
                returns_data[pair] = daily
        
        if not returns_data:
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
        
        # Combine into DataFrame
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 30:
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
        
        # Get position weights
        total_invested = equity_data['invested'] if equity_data and equity_data['invested'] > 0 else 1
        weights = {}
        
        for _, row in positions_df.iterrows():
            pair = row['pair']
            if pair in returns_df.columns and row['position'] and row['current_price']:
                position_value = abs(row['position'] * row['current_price'])
                weights[pair] = position_value / total_invested
        
        if not weights:
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
        
        # Portfolio returns
        portfolio_returns = sum(
            returns_df[pair] * weight 
            for pair, weight in weights.items() 
            if pair in returns_df.columns
        )
        
        # VaR calculations
        var_95 = np.percentile(portfolio_returns, (1 - 0.95) * 100) * np.sqrt(horizon_days)
        var_99 = np.percentile(portfolio_returns, (1 - 0.99) * 100) * np.sqrt(horizon_days)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Convert to dollar amounts
        total_equity = equity_data['total_equity'] if equity_data else 0
        
        return {
            'var_95': var_95 * total_equity,  # 95% VaR in dollars
            'var_99': var_99 * total_equity,  # 99% VaR in dollars
            'var_95_pct': var_95,             # 95% VaR as percentage
            'var_99_pct': var_99,             # 99% VaR as percentage
            'portfolio_vol': portfolio_vol,   # Annualized volatility
        }
    
    except Exception as e:
        logging.getLogger('paper_trader').error(f"VaR calculation failed: {e}")
        return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}


# =============================================================================
# DAILY REPORT
# =============================================================================

def generate_report(db: Database, config: dict, trading_summary: dict = None) -> str:
    """
    Generate comprehensive daily report.
    
    Args:
        db: Database instance
        config: Configuration dictionary
        trading_summary: Summary from trading run (optional)
    
    Returns:
        Formatted report string
    """
    # Get data
    equity = db.get_latest_equity()
    positions_df = db.get_current_positions()
    
    # Calculate VaR
    var_metrics = calculate_var(db, config['pairs'])
    
    # Get today's trades
    today = datetime.utcnow().date()
    trades_df = db.get_trades(start=today.isoformat())
    
    # Build report
    lines = [
        "=" * 50,
        "📊 CRYPTOBOT DAILY REPORT",
        f"📅 {today.isoformat()}",
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC",
        "=" * 50,
        "",
        "💰 EQUITY",
        "-" * 30,
    ]
    
    if equity:
        lines.extend([
            f"  Total Equity:  ${equity['total_equity']:>12,.2f}",
            f"  Cash Reserve:  ${equity['cash']:>12,.2f}",
            f"  Invested:      ${equity['invested']:>12,.2f}",
            f"  Daily P&L:     ${equity['daily_pnl']:>+12,.2f}",
            f"  Peak Equity:   ${equity['peak_equity']:>12,.2f}",
        ])
    else:
        lines.append("  No equity data available")
    
    lines.extend([
        "",
        "📉 RISK METRICS",
        "-" * 30,
    ])
    
    if equity:
        drawdown_pct = equity['drawdown'] * 100 if equity['drawdown'] else 0
        lines.extend([
            f"  Current Drawdown:  {drawdown_pct:>10.2f}%",
            f"  Max Allowed DD:    {config['risk']['max_drawdown']*100:>10.1f}%",
            f"  VaR (95%, 1-day):  ${var_metrics.get('var_95', 0):>10,.2f}",
            f"  VaR (99%, 1-day):  ${var_metrics.get('var_99', 0):>10,.2f}",
            f"  Portfolio Vol:     {var_metrics.get('portfolio_vol', 0)*100:>10.1f}% ann.",
        ])
    
    lines.extend([
        "",
        "📈 TODAY'S TRADES",
        "-" * 30,
    ])
    
    if trading_summary and trading_summary.get('trades'):
        lines.append(f"  Trades Executed: {trading_summary['trades_executed']}")
        lines.append(f"  Total Value:     ${trading_summary['total_trade_value']:,.2f}")
        lines.append(f"  Total Costs:     ${trading_summary['total_costs']:,.2f}")
        lines.append("")
        for t in trading_summary['trades']:
            lines.append(f"  • {t['pair']}: {t['direction']} {t['size']:.6f} @ ${t['price']:,.2f}")
    elif trades_df is not None and len(trades_df) > 0:
        lines.append(f"  Trades: {len(trades_df)}")
        for _, t in trades_df.head(10).iterrows():
            lines.append(f"  • {t['pair']}: {t['direction']} @ ${t['price']:,.2f}")
    else:
        lines.append("  No trades executed today")
    
    lines.extend([
        "",
        "📊 CURRENT POSITIONS",
        "-" * 30,
    ])
    
    if positions_df is not None and len(positions_df) > 0:
        has_positions = False
        for _, p in positions_df.iterrows():
            if p['position'] and abs(p['position']) > 0.0001:
                has_positions = True
                value = p['position'] * p['current_price'] if p['current_price'] else 0
                pnl = p['unrealized_pnl'] if p['unrealized_pnl'] else 0
                lines.append(
                    f"  {p['pair']:8s}: {p['position']:>10.6f} "
                    f"(${value:>10,.2f}, P&L: ${pnl:>+8,.2f})"
                )
        if not has_positions:
            lines.append("  No open positions")
    else:
        lines.append("  No open positions")
    
    lines.extend([
        "",
        "📋 SIGNALS",
        "-" * 30,
    ])
    
    if trading_summary and trading_summary.get('signals'):
        for sig in trading_summary['signals']:
            status = "✓" if sig['position'] >= 0.5 else "✗"
            lines.append(
                f"  {status} {sig['pair']:8s}: {sig['signal']:12s} "
                f"(HR: {sig['hit_rate']:.1%})"
            )
    else:
        lines.append("  No signal data")
    
    lines.extend([
        "",
        "=" * 50,
        "End of Report",
        "=" * 50,
    ])
    
    return "\n".join(lines)


def send_report(db: Database, config: dict, trading_summary: dict = None, logger: logging.Logger = None) -> bool:
    """
    Generate and send daily report via Pushover and Email.
    
    Args:
        db: Database instance
        config: Configuration dictionary
        trading_summary: Summary from trading run
        logger: Logger instance
    
    Returns:
        True if at least one notification sent successfully
    """
    if logger is None:
        logger = logging.getLogger('paper_trader')
    
    notifications = config.get('notifications', {})
    
    if not notifications.get('enabled', False):
        logger.info("Notifications disabled in config")
        return False
    
    # Generate report
    report = generate_report(db, config, trading_summary)
    logger.info("Daily Report Generated:")
    logger.info("\n" + report)
    
    success = False
    
    # =========================================================================
    # Send Pushover (supports multiple accounts)
    # =========================================================================
    pushover_list = notifications.get('pushover', [])
    
    # Handle old single-account format for backward compatibility
    if not pushover_list and notifications.get('pushover_user_key'):
        pushover_list = [{
            'user_key': notifications.get('pushover_user_key'),
            'api_token': notifications.get('pushover_api_token')
        }]
    
    for i, pushover in enumerate(pushover_list):
        user_key = pushover.get('user_key') or os.getenv('PUSHOVER_USER_KEY')
        api_token = pushover.get('api_token') or os.getenv('PUSHOVER_API_TOKEN')
        
        if not user_key or not api_token:
            continue
        
        try:
            # Pushover has 1024 char limit - send condensed version
            equity = db.get_latest_equity()
            short_msg = [
                f"📊 CryptoBot {datetime.utcnow().strftime('%Y-%m-%d')}",
            ]
            if equity:
                short_msg.extend([
                    f"💰 Equity: ${equity['total_equity']:,.0f}",
                    f"📈 P&L: ${equity['daily_pnl']:+,.0f}",
                    f"📉 DD: {equity['drawdown']*100:.1f}%",
                ])
            if trading_summary:
                short_msg.append(f"🔄 Trades: {trading_summary['trades_executed']}")
            
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": api_token,
                    "user": user_key,
                    "title": "CryptoBot Daily Report",
                    "message": "\n".join(short_msg),
                },
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"✅ Pushover #{i+1} sent")
                success = True
            else:
                logger.error(f"❌ Pushover #{i+1} failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Pushover #{i+1} error: {e}")
    
    # =========================================================================
    # Send Email (supports multiple recipients)
    # =========================================================================
    email_config = notifications.get('email', {})
    
    # Handle old flat format for backward compatibility
    if not email_config:
        email_config = {
            'sender': notifications.get('email_sender'),
            'password': notifications.get('email_password'),
            'recipients': [notifications.get('email_recipient')] if notifications.get('email_recipient') else [],
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587
        }
    
    sender = email_config.get('sender') or os.getenv('EMAIL_SENDER')
    password = email_config.get('password') or os.getenv('EMAIL_PASSWORD')
    recipients = email_config.get('recipients', [])
    smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
    smtp_port = email_config.get('smtp_port', 587)
    
    # Also check env for additional recipients
    env_recipient = os.getenv('EMAIL_RECIPIENT')
    if env_recipient and env_recipient not in recipients:
        recipients.append(env_recipient)
    
    if sender and password and recipients:
        try:
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"CryptoBot Daily Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
            msg.attach(MIMEText(report, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipients, msg.as_string())
            
            logger.info(f"✅ Email sent to {len(recipients)} recipient(s)")
            success = True
        except Exception as e:
            logger.error(f"❌ Email error: {e}")
    
    return success


# =============================================================================
# PAPER TRADING RUNNER
# =============================================================================

class PaperTradingRunner:
    """
    Paper trading runner for the 16-state strategy.
    
    Runs daily to:
    1. Fetch latest data
    2. Generate signals for each pair
    3. Record signals to database
    4. Compare to current positions
    5. Log paper trades
    6. Update portfolio snapshots
    """
    
    def __init__(self, config: dict, dry_run: bool = False):
        """
        Initialize the runner.
        
        Args:
            config: Configuration dictionary
            dry_run: If True, don't write to database
        """
        self.config = config
        self.dry_run = dry_run
        
        self.pairs = config['pairs']
        self.strategy_name = config['strategy']['name']
        self.min_training_months = config['strategy']['min_training_months']
        self.use_filter = config['strategy']['use_ma72_filter']
        
        self.initial_capital = config['position']['initial_capital']
        self.lookback_days = config['data']['lookback_days']
        self.trading_cost_bps = config['execution']['trading_cost_bps']
        
        # Initialize components
        self.db = Database()
        self.signal_generator = SignalGenerator(use_filter=self.use_filter)
        
        self.logger = logging.getLogger('paper_trader')
        
        # Track results for this run
        self.signals_generated = []
        self.trades_executed = []
    
    def run(self) -> dict:
        """
        Execute the daily paper trading run.
        
        Returns:
            Summary dict with results
        """
        run_time = datetime.utcnow()
        self.logger.info("=" * 60)
        self.logger.info(f"PAPER TRADING RUN - {run_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        self.logger.info("=" * 60)
        
        if self.dry_run:
            self.logger.info("*** DRY RUN MODE - No database writes ***")
        
        # Get current equity (or initialize)
        equity = self._get_or_init_equity()
        self.logger.info(f"Current equity: ${equity['total_equity']:,.2f}")
        
        # Process each pair
        for pair in self.pairs:
            self.logger.info("-" * 40)
            self.logger.info(f"Processing {pair}...")
            
            try:
                self._process_pair(pair, equity, run_time)
            except Exception as e:
                self.logger.error(f"Error processing {pair}: {e}")
                continue
        
        # Update equity snapshot
        self._update_equity(equity, run_time)
        
        # Summary
        summary = self._generate_summary()
        self._log_summary(summary)
        
        return summary
    
    def _get_or_init_equity(self) -> dict:
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
                self.db.record_equity(**equity)
        
        return equity
    
    def _process_pair(self, pair: str, equity: dict, run_time: datetime):
        """Process a single trading pair."""
        
        # 1. Fetch data
        start_date = run_time - timedelta(days=self.lookback_days)
        df_1h = self.db.get_ohlcv(pair, start=start_date.isoformat())
        
        if len(df_1h) < 168:  # Need at least 1 week of hourly data
            self.logger.warning(f"{pair}: Insufficient data ({len(df_1h)} rows)")
            return
        
        self.logger.info(f"{pair}: Loaded {len(df_1h)} hourly bars")
        
        # 2. Generate signals
        signals_df = self.signal_generator.generate_signals(df_1h)
        returns = self.signal_generator.get_returns(df_1h)
        
        if len(signals_df) == 0:
            self.logger.warning(f"{pair}: No signals generated")
            return
        
        # Get signal for most recent date
        latest_date = signals_df.index[-1]
        data_start = signals_df.index[0]
        
        position, details = self.signal_generator.get_position_for_date(
            signals_df, 
            returns, 
            latest_date, 
            data_start,
            self.min_training_months
        )
        
        # 3. Record signal
        signal_type = details.get('simple_state', 'UNKNOWN')
        hit_rate = details.get('hit_rate', 0.5)
        filtered = details.get('filtered', False)
        
        self.logger.info(
            f"{pair}: Signal={signal_type}, Position={position:.1f}, "
            f"HitRate={hit_rate:.1%}, Filtered={filtered}"
        )
        
        if not self.dry_run:
            self.db.record_signal(
                pair=pair,
                strategy=self.strategy_name,
                signal=signal_type,
                target_position=position,
                confidence=hit_rate,
                regime=self._encode_state(details),
                prediction=hit_rate,
                timestamp=run_time
            )
        
        self.signals_generated.append({
            'pair': pair,
            'signal': signal_type,
            'position': position,
            'hit_rate': hit_rate,
            'filtered': filtered
        })
        
        # 4. Compare to current position
        current_pos = self.db.get_position(pair)
        current_position = current_pos['position'] if current_pos else 0.0
        
        # 5. Determine if trade needed
        current_price = df_1h['close'].iloc[-1]
        target_position = self._calculate_target_position(position, equity, current_price)
        
        position_diff = target_position - current_position
        
        if abs(position_diff) > 0.001:  # Threshold to avoid tiny trades
            self._execute_paper_trade(
                pair=pair,
                current_position=current_position,
                target_position=target_position,
                price=current_price,
                signal_type=signal_type,
                run_time=run_time
            )
        else:
            self.logger.info(f"{pair}: No trade needed (position unchanged)")
        
        # 6. Update portfolio snapshot
        if not self.dry_run:
            entry_price = current_pos['entry_price'] if current_pos else current_price
            unrealized_pnl = target_position * (current_price - entry_price) if target_position != 0 else 0
            
            self.db.record_portfolio_snapshot(
                pair=pair,
                position=target_position,
                entry_price=entry_price if target_position != 0 else None,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                timestamp=run_time
            )
    
    def _calculate_target_position(
        self, 
        signal_position: float, 
        equity: dict, 
        price: float
    ) -> float:
        """
        Calculate target position size based on signal and equity.
        
        Args:
            signal_position: Signal (1.0=invest, 0.5=skip, 0.0=avoid)
            equity: Current equity dict
            price: Current price
        
        Returns:
            Target position in base units
        """
        if signal_position <= 0:
            return 0.0
        
        # Allocate based on signal strength
        max_pct = self.config['position']['max_position_pct']
        allocation = equity['total_equity'] * max_pct * signal_position
        
        # Convert to position size
        position = allocation / price
        
        return position
    
    def _execute_paper_trade(
        self,
        pair: str,
        current_position: float,
        target_position: float,
        price: float,
        signal_type: str,
        run_time: datetime
    ):
        """Execute a paper trade (log and record)."""
        
        position_diff = target_position - current_position
        direction = "BUY" if position_diff > 0 else "SELL"
        size = abs(position_diff)
        value = size * price
        cost = value * (self.trading_cost_bps / 10000)
        
        self.logger.info(
            f"{pair}: PAPER TRADE - {direction} {size:.6f} @ ${price:,.2f} "
            f"(value: ${value:,.2f}, cost: ${cost:.2f})"
        )
        
        if not self.dry_run:
            self.db.record_trade(
                pair=pair,
                strategy=self.strategy_name,
                direction=direction,
                size=size,
                price=price,
                slippage_bps=0,
                transaction_cost=cost,
                execution_type="paper",
                notes=f"Signal: {signal_type}"
            )
        
        self.trades_executed.append({
            'pair': pair,
            'direction': direction,
            'size': size,
            'price': price,
            'value': value,
            'cost': cost
        })
    
    def _encode_state(self, details: dict) -> int:
        """Encode state tuple as integer for storage."""
        state = details.get('active_state')
        if state is None:
            return None
        
        # Encode 4-tuple as single int: t24*8 + t168*4 + ma72*2 + ma168
        return state[0]*8 + state[1]*4 + state[2]*2 + state[3]
    
    def _update_equity(self, equity: dict, run_time: datetime):
        """Update equity snapshot after all trades."""
        
        if self.dry_run:
            return
        
        # Get all current positions
        positions_df = self.db.get_current_positions()
        
        # Calculate invested value
        invested = 0.0
        unrealized_pnl = 0.0
        
        if len(positions_df) > 0:
            for _, row in positions_df.iterrows():
                if row['position'] and row['position'] != 0:
                    invested += abs(row['position'] * row['current_price'])
                    if row['unrealized_pnl']:
                        unrealized_pnl += row['unrealized_pnl']
        
        # Calculate cash change from today's trades
        cash_change = 0.0
        for t in self.trades_executed:
            if t['direction'] == 'BUY':
                cash_change -= t['value']  # Buying reduces cash
            else:
                cash_change += t['value']  # Selling increases cash
            cash_change -= t['cost']  # Costs always reduce cash
        
        # Update equity
        cash = equity['cash'] + cash_change
        total_equity = cash + invested + unrealized_pnl
        
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
            'run_time': datetime.utcnow().isoformat(),
            'pairs_processed': len(self.signals_generated),
            'signals': self.signals_generated,
            'trades_executed': len(self.trades_executed),
            'trades': self.trades_executed,
            'total_trade_value': sum(t['value'] for t in self.trades_executed),
            'total_costs': sum(t['cost'] for t in self.trades_executed)
        }
    
    def _log_summary(self, summary: dict):
        """Log run summary."""
        self.logger.info("=" * 60)
        self.logger.info("RUN SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Pairs processed: {summary['pairs_processed']}")
        self.logger.info(f"Trades executed: {summary['trades_executed']}")
        self.logger.info(f"Total trade value: ${summary['total_trade_value']:,.2f}")
        self.logger.info(f"Total costs: ${summary['total_costs']:,.2f}")
        
        # Signal breakdown
        if self.signals_generated:
            self.logger.info("-" * 40)
            self.logger.info("SIGNALS:")
            for sig in self.signals_generated:
                self.logger.info(
                    f"  {sig['pair']}: {sig['signal']} "
                    f"(pos={sig['position']:.1f}, hr={sig['hit_rate']:.1%})"
                )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='CryptoBot Paper Trading Runner'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file (default: apps/live/config.yaml)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Dry run mode - no database writes'
    )
    parser.add_argument(
        '--hourly',
        action='store_true',
        help='Hourly mode: update data every run, trade only at 00:xx UTC'
    )
    parser.add_argument(
        '--trading-hour',
        type=int,
        default=0,
        help='Hour (UTC) to run trading (default: 0 = midnight)'
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
    
    current_hour = datetime.utcnow().hour
    
    logger.info("=" * 60)
    logger.info(f"CryptoBot Runner Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"Mode: {'HOURLY' if args.hourly else 'DIRECT'}")
    logger.info("=" * 60)
    
    try:
        if args.hourly:
            # HOURLY MODE
            # 1. Always update data
            update_data(config, logger)
            
            # 2. Trade only at specified hour
            if current_hour == args.trading_hour:
                logger.info("-" * 40)
                logger.info(f"TRADING HOUR ({args.trading_hour}:xx UTC) - Running trades...")
                logger.info("-" * 40)
                
                runner = PaperTradingRunner(config, dry_run=args.dry_run)
                summary = runner.run()
                
                # 3. Send report after trading
                send_report(runner.db, config, summary, logger)
            else:
                logger.info(f"Not trading hour (current: {current_hour}, trading: {args.trading_hour})")
                logger.info("Data updated. Skipping trades.")
        
        else:
            # DIRECT MODE - just run trading
            runner = PaperTradingRunner(config, dry_run=args.dry_run)
            summary = runner.run()
            
            # Optionally send report
            if config.get('notifications', {}).get('enabled'):
                send_report(runner.db, config, summary, logger)
        
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