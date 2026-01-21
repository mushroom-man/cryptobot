# -*- coding: utf-8 -*-
"""
CryptoBot - Daily Report Generator
==================================
Database-driven daily report generation.

Pulls all data from database tables:
- equity: Current equity snapshot
- portfolio: Current positions
- signals: Latest signals per pair
- trades: Today's executed trades
- ohlcv: Current prices

Usage:
    from cryptobot.reports.daily_report import DailyReport
    
    report = DailyReport(db, config)
    content = report.generate()
    print(content)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Timezone handling
try:
    from zoneinfo import ZoneInfo
    MELBOURNE_TZ = ZoneInfo('Australia/Melbourne')
except ImportError:
    import pytz
    MELBOURNE_TZ = pytz.timezone('Australia/Melbourne')


def melbourne_now() -> datetime:
    """Get current time in Melbourne timezone."""
    return datetime.now(MELBOURNE_TZ)


@dataclass
class ReportData:
    """Container for all report data."""
    timestamp: datetime
    
    # Equity
    total_equity: float
    cash: float
    invested: float
    daily_pnl: float
    drawdown: float
    peak_equity: float
    
    # Positions
    positions: List[Dict[str, Any]]
    
    # Signals
    signals: List[Dict[str, Any]]
    
    # Trades
    trades: List[Dict[str, Any]]
    
    # Risk metrics
    var_95: float
    var_99: float
    portfolio_vol: float
    
    # Config
    max_drawdown: float
    pairs: List[str]


class DailyReport:
    """
    Database-driven daily report generator.
    
    All data is pulled fresh from the database, ensuring
    reports are accurate regardless of when they're generated.
    """
    
    def __init__(self, db, config: dict):
        """
        Initialise daily report generator.
        
        Args:
            db: Database instance
            config: Configuration dictionary
        """
        self.db = db
        self.config = config
        self.pairs = config.get('pairs', [])
    
    def gather_data(self) -> ReportData:
        """
        Gather all data needed for the report from database.
        
        Calculates LIVE mark-to-market equity rather than using
        potentially stale values from equity table.
        
        Returns:
            ReportData with all fields populated
        """
        now = melbourne_now()
        
        # Get last recorded equity (for peak equity reference)
        equity = self._get_equity()
        
        # Get positions with current prices (live mark-to-market)
        positions = self._get_positions_with_prices()
        
        # Calculate LIVE values
        invested = sum(p['value'] for p in positions if p['value'])
        unrealised_pnl = sum(p['unrealised_pnl'] for p in positions)
        
        # Calculate cash from first principles:
        # cash = initial_capital - cost_of_positions - trading_costs
        initial_capital = self.config.get('position', {}).get('initial_capital', 100000)
        
        # Cost of positions = sum(entry_price × position_size)
        cost_of_positions = sum(
            p['entry_price'] * abs(p['position']) 
            for p in positions 
            if p['entry_price'] and p['position']
        )
        
        # Estimate total trading costs paid
        trading_cost_bps = self.config.get('execution', {}).get('trading_cost_bps', 20)
        slippage_bps = self.config.get('execution', {}).get('slippage_bps', 5)
        total_cost_bps = trading_cost_bps + slippage_bps
        estimated_costs = cost_of_positions * (total_cost_bps / 10000)
        
        # Current cash
        cash = initial_capital - cost_of_positions - estimated_costs
        
        # LIVE total equity = cash + position values (mark-to-market)
        total_equity = cash + invested
        
        # Get peak equity for drawdown calculation
        peak_equity = equity.get('peak_equity', initial_capital) if equity else initial_capital
        peak_equity = max(peak_equity, total_equity)  # Update if new high
        
        # Calculate LIVE drawdown from peak
        if peak_equity > 0:
            drawdown = (peak_equity - total_equity) / peak_equity
        else:
            drawdown = 0
        
        # Calculate LIVE daily P&L
        # Compare current equity to last recorded equity (or initial if none)
        if equity:
            previous_equity = equity.get('total_equity', initial_capital)
        else:
            previous_equity = initial_capital
        daily_pnl = total_equity - previous_equity
        
        # Get today's trades for display
        trades = self._get_todays_trades()
        
        # Get signals (latest per pair)
        signals = self._get_latest_signals()
        
        # Calculate VaR using live equity
        live_equity = {'total_equity': total_equity, 'peak_equity': peak_equity}
        var_metrics = self._calculate_var(live_equity, positions)
        
        return ReportData(
            timestamp=now,
            total_equity=total_equity,
            cash=cash,
            invested=invested,
            daily_pnl=daily_pnl,
            drawdown=drawdown,
            peak_equity=peak_equity,
            positions=positions,
            signals=signals,
            trades=trades,
            var_95=var_metrics.get('var_95', 0),
            var_99=var_metrics.get('var_99', 0),
            portfolio_vol=var_metrics.get('portfolio_vol', 0),
            max_drawdown=self.config.get('risk', {}).get('max_drawdown', 0.20),
            pairs=self.pairs,
        )
    
    def _get_equity(self) -> Optional[Dict]:
        """Get latest equity snapshot."""
        return self.db.get_latest_equity()
    
    def _get_positions_with_prices(self) -> List[Dict]:
        """
        Get current positions with live prices and calculated P&L.
        
        Fetches current prices from ohlcv table and calculates
        unrealised P&L based on entry price.
        """
        positions = []
        
        # Get positions from portfolio table
        positions_df = self.db.get_current_positions()
        
        if positions_df is None or len(positions_df) == 0:
            return positions
        
        for _, row in positions_df.iterrows():
            pair = row['pair']
            position = row['position'] or 0
            entry_price = row['entry_price']
            
            # Skip zero positions
            if abs(position) < 0.0001:
                continue
            
            # Get current price from database
            current_price = self.db.get_latest_price(pair)
            if current_price is None:
                current_price = row.get('current_price', 0)
            
            # Calculate value and P&L
            value = abs(position * current_price) if current_price else 0
            
            if entry_price and current_price and position:
                unrealised_pnl = position * (current_price - entry_price)
            else:
                unrealised_pnl = 0
            
            positions.append({
                'pair': pair,
                'position': position,
                'entry_price': entry_price,
                'current_price': current_price,
                'value': value,
                'unrealised_pnl': unrealised_pnl,
            })
        
        return positions
    
    def _get_latest_signals(self) -> List[Dict]:
        """
        Get latest signal for each pair.
        
        Queries signals table for the most recent signal per pair.
        """
        signals = []
        
        for pair in self.pairs:
            signal = self.db.get_latest_signal(pair)
            if signal:
                # Convert confidence to hit rate percentage
                hit_rate = signal.get('confidence', 0)
                
                signals.append({
                    'pair': pair,
                    'signal': signal.get('signal', 'UNKNOWN'),
                    'hit_rate': hit_rate,
                    'target_position': signal.get('target_position', 0),
                    'regime': signal.get('regime'),
                    'timestamp': signal.get('timestamp'),
                })
            else:
                signals.append({
                    'pair': pair,
                    'signal': 'NO_DATA',
                    'hit_rate': 0,
                    'target_position': 0,
                    'regime': None,
                    'timestamp': None,
                })
        
        return signals
    
    def _get_todays_trades(self) -> List[Dict]:
        """Get trades executed today (Melbourne time)."""
        # Get start of today Melbourne time, convert to UTC for query
        today_melb = melbourne_now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        trades_df = self.db.get_trades(start=today_melb.isoformat())
        
        trades = []
        if trades_df is not None and len(trades_df) > 0:
            for _, row in trades_df.iterrows():
                trades.append({
                    'pair': row['pair'],
                    'direction': row['direction'],
                    'size': row['size'],
                    'price': row['price'],
                    'timestamp': row.get('timestamp'),
                })
        
        return trades
    
    def _get_trades_since(self, since_timestamp) -> List[Dict]:
        """Get all trades since a given timestamp."""
        trades_df = self.db.get_trades(start=since_timestamp.isoformat() if hasattr(since_timestamp, 'isoformat') else str(since_timestamp))
        
        trades = []
        if trades_df is not None and len(trades_df) > 0:
            for _, row in trades_df.iterrows():
                trades.append({
                    'pair': row['pair'],
                    'direction': row['direction'],
                    'size': row['size'],
                    'price': row['price'],
                    'timestamp': row.get('timestamp'),
                })
        
        return trades
    
    def _calculate_var(self, equity: Dict, positions: List[Dict]) -> Dict:
        """
        Calculate Value at Risk metrics.
        
        Uses historical simulation with 1-year lookback.
        """
        try:
            if not equity or not positions:
                return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
            
            total_invested = sum(p['value'] for p in positions)
            if total_invested <= 0:
                return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
            
            # Get historical returns
            returns_data = {}
            start_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
            
            for pair in self.pairs:
                df = self.db.get_ohlcv(pair, start=start_date)
                if df is not None and len(df) > 0:
                    daily = df['close'].resample('24h').last().ffill().pct_change().dropna()
                    returns_data[pair] = daily
            
            if not returns_data:
                return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if len(returns_df) < 30:
                return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
            
            # Get position weights
            weights = {}
            for p in positions:
                if p['pair'] in returns_df.columns and p['value']:
                    weights[p['pair']] = p['value'] / total_invested
            
            if not weights:
                return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
            
            # Portfolio returns
            portfolio_returns = sum(
                returns_df[pair] * weight
                for pair, weight in weights.items()
                if pair in returns_df.columns
            )
            
            # VaR calculations (1-day horizon)
            var_95 = np.percentile(portfolio_returns, 5)  # 5th percentile for losses
            var_99 = np.percentile(portfolio_returns, 1)  # 1st percentile
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            
            total_equity = equity.get('total_equity', total_invested)
            
            return {
                'var_95': abs(var_95 * total_equity),
                'var_99': abs(var_99 * total_equity),
                'portfolio_vol': portfolio_vol,
            }
        
        except Exception as e:
            print(f"VaR calculation error: {e}")
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
    
    def generate(self) -> str:
        """
        Generate the full daily report.
        
        Returns:
            Formatted report string
        """
        data = self.gather_data()
        return self._format_report(data)
    
    def _format_report(self, data: ReportData) -> str:
        """Format report data into readable string."""
        
        lines = [
            "=" * 50,
            "📊 CRYPTOBOT DAILY REPORT",
            f"📅 {data.timestamp.strftime('%Y-%m-%d')}",
            f"⏰ {data.timestamp.strftime('%H:%M:%S')} Melbourne",
            "=" * 50,
            "",
            "💰 EQUITY",
            "-" * 30,
            f"  Total Equity Value:    ${data.total_equity:>12,.2f}",
            f"  Cash Balance:          ${data.cash:>12,.2f}",
            f"  Invested Value:        ${data.invested:>12,.2f}",
            f"  Daily P&L (24h):       ${data.daily_pnl:>+12,.2f}",
            "",
            "📉 RISK METRICS",
            "-" * 30,
            f"  Current Drawdown:      {data.drawdown * 100:>10.2f}%",
            f"  Max Allowed DD:        {data.max_drawdown * 100:>10.1f}%",
            f"  VaR (95%, 1-day):      ${data.var_95:>10,.2f}",
            f"  VaR (99%, 1-day):      ${data.var_99:>10,.2f}",
            f"  Portfolio Vol:         {data.portfolio_vol * 100:>10.1f}% ann.",
            "",
            "📈 TODAY'S TRADES",
            "-" * 30,
        ]
        
        if data.trades:
            lines.append(f"  Trades Executed: {len(data.trades)}")
            total_value = sum(t['size'] * t['price'] for t in data.trades)
            lines.append(f"  Total Value:     ${total_value:,.2f}")
            lines.append("")
            for t in data.trades:
                value = t['size'] * t['price']
                lines.append(
                    f"  • {t['pair']}: {t['direction']} {t['size']:.6f} @ ${t['price']:,.2f} (${value:,.2f})"
                )
        else:
            lines.append("  No trades executed today")
        
        lines.extend([
            "",
            "📊 CURRENT POSITIONS",
            "-" * 30,
        ])
        
        if data.positions:
            for p in data.positions:
                lines.append(
                    f"  {p['pair']:8s}: {p['position']:>12.4f} "
                    f"(${p['value']:>10,.2f}, P&L: ${p['unrealised_pnl']:>+8,.2f})"
                )
        else:
            lines.append("  No open positions")
        
        lines.extend([
            "",
            "📡 SIGNALS",
            "-" * 30,
        ])
        
        if data.signals:
            for sig in data.signals:
                if sig['signal'] in ['STRONG_BUY', 'BUY']:
                    icon = "✅"
                elif sig['signal'] in ['SELL', 'STRONG_SELL']:
                    icon = "⛔"
                elif sig['signal'] == 'NO_DATA':
                    icon = "❓"
                else:
                    icon = "⏸️"
                
                target = sig.get('target_position', 0) or 0
                hr = sig.get('hit_rate', 0) or 0
                
                lines.append(
                    f"  {icon} {sig['pair']:8s}: {sig['signal']:12s} "
                    f"(HR: {hr:.1%}, Target: {target:.2f})"
                )
        else:
            lines.append("  No signal data available")
        
        lines.extend([
            "",
            "=" * 50,
            "End of Report",
            "=" * 50,
        ])
        
        return "\n".join(lines)
    
    def generate_dict(self) -> Dict[str, Any]:
        """
        Generate report as dictionary (for JSON/API use).
        
        Returns:
            Dictionary with all report data
        """
        data = self.gather_data()
        
        return {
            'timestamp': data.timestamp.isoformat(),
            'equity': {
                'total': data.total_equity,
                'cash': data.cash,
                'invested': data.invested,
                'daily_pnl': data.daily_pnl,
                'drawdown': data.drawdown,
                'peak': data.peak_equity,
            },
            'risk': {
                'var_95': data.var_95,
                'var_99': data.var_99,
                'portfolio_vol': data.portfolio_vol,
                'max_drawdown': data.max_drawdown,
            },
            'positions': data.positions,
            'signals': data.signals,
            'trades': data.trades,
        }