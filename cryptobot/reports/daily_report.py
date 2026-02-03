# -*- coding: utf-8 -*-
"""
CryptoBot - Daily Report Generator
==================================
Partner-friendly daily report with clear, non-technical language.

Usage:
    from cryptobot.reports.daily_report import DailyReport
    
    report = DailyReport(db, config)
    content = report.generate()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .pair_names import get_friendly_name, format_holding

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


def to_melbourne(dt: datetime) -> datetime:
    """Convert datetime to Melbourne timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MELBOURNE_TZ)


@dataclass
class TradeDetail:
    """Individual trade record for reporting."""
    timestamp: datetime
    pair: str
    friendly_name: str
    direction: str  # BUY or SELL
    trade_type: str  # ENTRY, EXIT, INCREASE, DECREASE
    size: float
    price: float
    value: float
    cost: float


@dataclass
class DailyReportData:
    """Container for daily report data."""
    timestamp: datetime
    
    # Portfolio values
    total_equity: float
    yesterday_equity: float
    daily_change: float
    daily_change_pct: float
    
    # Performance periods
    week_change: float
    week_change_pct: float
    month_change: float
    month_change_pct: float
    inception_change: float
    inception_change_pct: float
    
    # Risk
    drawdown_pct: float
    max_drawdown_pct: float
    risk_status: str
    
    # Activity
    trades_total: int
    trades_entry: int
    trades_exit: int
    trades_rebalance: int
    total_costs: float
    
    # Individual trades
    trades: List[TradeDetail] = field(default_factory=list)
    
    # Holdings
    holdings: List[Dict[str, Any]] = field(default_factory=list)
    cash: float = 0.0
    
    # Config
    inception_date: datetime = None


class DailyReport:
    """
    Partner-friendly daily report generator.
    
    Pulls live data from database and formats in clear,
    non-technical language suitable for partners.
    """
    
    def __init__(self, db, config: dict):
        """
        Initialize daily report generator.
        
        Args:
            db: Database instance
            config: Configuration dictionary
        """
        self.db = db
        self.config = config
        self.pairs = config.get('pairs', [])
        self.initial_capital = config.get('position', {}).get('initial_capital', 100000)
        self.max_drawdown = config.get('risk', {}).get('max_drawdown', 0.20)
        
        # Inception date - set this when starting fresh
        self.inception_date = config.get('inception_date')
        if isinstance(self.inception_date, str):
            self.inception_date = pd.to_datetime(self.inception_date)
    
    def gather_data(self) -> DailyReportData:
        """Gather all data needed for the daily report."""
        now = melbourne_now()
        
        # Get current equity
        current_equity = self._get_current_equity()
        total_equity = current_equity.get('total_equity', self.initial_capital)
        
        # Get historical equity for comparisons
        yesterday_equity = self._get_equity_days_ago(1)
        week_ago_equity = self._get_equity_days_ago(7)
        month_ago_equity = self._get_equity_days_ago(30)
        inception_equity = self.initial_capital
        
        # Calculate changes
        daily_change = total_equity - yesterday_equity
        daily_change_pct = (daily_change / yesterday_equity * 100) if yesterday_equity else 0
        
        week_change = total_equity - week_ago_equity
        week_change_pct = (week_change / week_ago_equity * 100) if week_ago_equity else 0
        
        month_change = total_equity - month_ago_equity
        month_change_pct = (month_change / month_ago_equity * 100) if month_ago_equity else 0
        
        inception_change = total_equity - inception_equity
        inception_change_pct = (inception_change / inception_equity * 100) if inception_equity else 0
        
        # Risk status
        peak_equity = current_equity.get('peak_equity', total_equity)
        drawdown_pct = ((peak_equity - total_equity) / peak_equity * 100) if peak_equity else 0
        
        if drawdown_pct < self.max_drawdown * 50:  # Less than 50% of limit
            risk_status = "Normal operations"
        elif drawdown_pct < self.max_drawdown * 80:  # Less than 80% of limit
            risk_status = "Elevated - monitoring closely"
        else:
            risk_status = "⚠️ High - approaching limits"
        
        # Today's trades (summary and details)
        trade_summary, trade_details = self._get_todays_trades()
        
        # Current holdings with live prices
        holdings, cash = self._get_holdings()
        
        return DailyReportData(
            timestamp=now,
            total_equity=total_equity,
            yesterday_equity=yesterday_equity,
            daily_change=daily_change,
            daily_change_pct=daily_change_pct,
            week_change=week_change,
            week_change_pct=week_change_pct,
            month_change=month_change,
            month_change_pct=month_change_pct,
            inception_change=inception_change,
            inception_change_pct=inception_change_pct,
            drawdown_pct=drawdown_pct,
            max_drawdown_pct=self.max_drawdown * 100,
            risk_status=risk_status,
            trades_total=trade_summary['total'],
            trades_entry=trade_summary['entry'],
            trades_exit=trade_summary['exit'],
            trades_rebalance=trade_summary['rebalance'],
            total_costs=trade_summary['costs'],
            trades=trade_details,
            holdings=holdings,
            cash=cash,
            inception_date=self.inception_date,
        )
    
    def _get_current_equity(self) -> Dict:
        """Get current equity with live mark-to-market."""
        # Get positions with live prices
        holdings, cash = self._get_holdings()
        
        invested = sum(h['value'] for h in holdings)
        total_equity = cash + invested
        
        # Get peak from database
        db_equity = self.db.get_latest_equity()
        peak_equity = db_equity.get('peak_equity', total_equity) if db_equity else total_equity
        peak_equity = max(peak_equity, total_equity)
        
        return {
            'total_equity': total_equity,
            'cash': cash,
            'invested': invested,
            'peak_equity': peak_equity,
        }
    
    def _get_equity_days_ago(self, days: int) -> float:
        """Get equity value from N days ago."""
        target_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        try:
            df = self.db.get_equity_history(start=target_date)
            if df is not None and len(df) > 0:
                return df['total_equity'].iloc[0]
        except Exception:
            pass
        
        return self.initial_capital
    
    def _get_holdings(self) -> tuple:
        """
        Get current holdings with live prices.
        
        Returns:
            Tuple of (holdings list, cash balance)
        """
        holdings = []
        
        positions_df = self.db.get_current_positions()
        
        if positions_df is None or len(positions_df) == 0:
            return holdings, self.initial_capital
        
        total_invested = 0
        
        for _, row in positions_df.iterrows():
            pair = row['pair']
            position = row['position'] or 0
            entry_price = row['entry_price']
            
            if abs(position) < 0.0001:
                continue
            
            # Get LIVE price from OHLCV
            current_price = self.db.get_latest_price(pair)
            if current_price is None:
                current_price = row.get('current_price', 0)
            
            value = abs(position * current_price) if current_price else 0
            total_invested += value
            
            holdings.append({
                'pair': pair,
                'friendly_name': get_friendly_name(pair),
                'position': position,
                'current_price': current_price,
                'value': value,
            })
        
        # Sort by value descending
        holdings.sort(key=lambda x: x['value'], reverse=True)
        
        # Calculate percentages
        total_value = total_invested + self._get_cash_balance(total_invested)
        for h in holdings:
            h['percentage'] = (h['value'] / total_value * 100) if total_value else 0
        
        cash = self._get_cash_balance(total_invested)
        
        return holdings, cash
    
    def _get_cash_balance(self, invested: float) -> float:
        """Calculate cash balance."""
        db_equity = self.db.get_latest_equity()
        if db_equity and db_equity.get('total_equity'):
            return db_equity['total_equity'] - invested
        return self.initial_capital - invested
    
    def _get_todays_trades(self) -> tuple:
        """
        Get today's trade summary and details.
        
        Returns:
            Tuple of (summary dict, list of TradeDetail)
        """
        today_start = melbourne_now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        trades_df = self.db.get_trades(start=today_start.isoformat())
        
        summary = {
            'total': 0,
            'entry': 0,
            'exit': 0,
            'rebalance': 0,
            'costs': 0.0,
        }
        
        trade_details = []
        
        if trades_df is None or len(trades_df) == 0:
            return summary, trade_details
        
        summary['total'] = len(trades_df)
        summary['costs'] = trades_df['transaction_cost'].sum() if 'transaction_cost' in trades_df else 0
        
        if 'trade_type' in trades_df.columns:
            type_counts = trades_df['trade_type'].value_counts()
            summary['entry'] = type_counts.get('ENTRY', 0)
            summary['exit'] = type_counts.get('EXIT', 0)
            summary['rebalance'] = type_counts.get('INCREASE', 0) + type_counts.get('DECREASE', 0)
        else:
            summary['rebalance'] = summary['total']
        
        # Build individual trade details
        for _, row in trades_df.iterrows():
            # Get timestamp and convert to Melbourne
            ts = row.get('timestamp') or row.get('created_at')
            if ts is not None:
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                ts = to_melbourne(ts)
            else:
                ts = melbourne_now()
            
            pair = row.get('pair', 'UNKNOWN')
            direction = row.get('direction', 'BUY')
            trade_type = row.get('trade_type', 'TRADE')
            size = row.get('size', 0)
            price = row.get('price', 0)
            value = abs(size * price) if size and price else 0
            cost = row.get('transaction_cost', 0)
            
            trade_details.append(TradeDetail(
                timestamp=ts,
                pair=pair,
                friendly_name=get_friendly_name(pair),
                direction=direction,
                trade_type=trade_type,
                size=size,
                price=price,
                value=value,
                cost=cost,
            ))
        
        # Sort by timestamp (oldest first)
        trade_details.sort(key=lambda x: x.timestamp)
        
        return summary, trade_details
    
    def generate(self) -> str:
        """Generate the formatted daily report."""
        data = self.gather_data()
        return self._format_report(data)
    
    def _format_report(self, data: DailyReportData) -> str:
        """Format report data into partner-friendly string."""
        
        # Day name for header
        day_name = data.timestamp.strftime('%A, %d %B %Y')
        
        lines = [
            "=" * 50,
            "📊 CRYPTOBOT DAILY REPORT",
            f"📅 {day_name}",
            "=" * 50,
            "",
            "💰 PORTFOLIO VALUE",
            f"   Today:           ${data.total_equity:>14,.2f}",
            f"   Yesterday:       ${data.yesterday_equity:>14,.2f}",
            f"   Daily Change:    ${data.daily_change:>+14,.2f} ({data.daily_change_pct:+.2f}%)",
            "",
            "📈 PERFORMANCE",
            f"   This Week:       ${data.week_change:>+14,.2f} ({data.week_change_pct:+.2f}%)",
            f"   This Month:      ${data.month_change:>+14,.2f} ({data.month_change_pct:+.2f}%)",
            f"   Since Start:     ${data.inception_change:>+14,.2f} ({data.inception_change_pct:+.2f}%)",
            "",
            "📉 RISK STATUS",
        ]
        
        # Risk status with icon
        if data.drawdown_pct < data.max_drawdown_pct * 0.5:
            risk_icon = "✅"
        elif data.drawdown_pct < data.max_drawdown_pct * 0.8:
            risk_icon = "⚠️"
        else:
            risk_icon = "🚨"
        
        lines.extend([
            f"   {risk_icon} Drawdown: {data.drawdown_pct:.1f}% (limit: {data.max_drawdown_pct:.0f}%)",
            f"   Status: {data.risk_status}",
            "",
            "🔄 TODAY'S ACTIVITY",
            f"   Total Trades: {data.trades_total}",
        ])
        
        if data.trades_total > 0:
            if data.trades_entry > 0:
                lines.append(f"   New Positions:    {data.trades_entry}")
            if data.trades_exit > 0:
                lines.append(f"   Exits:            {data.trades_exit}")
            if data.trades_rebalance > 0:
                lines.append(f"   Rebalancing:      {data.trades_rebalance}")
            lines.append(f"   Total Costs:      ${data.total_costs:,.2f}")
            
            # Individual trade details
            lines.extend([
                "",
                "   ─────────────────────────────────────────────",
                "   TRADE DETAILS",
                "   ─────────────────────────────────────────────",
            ])
            
            for trade in data.trades:
                time_str = trade.timestamp.strftime('%H:%M')
                
                # Direction icon
                if trade.direction == 'BUY':
                    dir_icon = "🟢"
                    action = "Bought"
                else:
                    dir_icon = "🔴"
                    action = "Sold"
                
                # Trade type description
                if trade.trade_type == 'ENTRY':
                    type_desc = "(New Position)"
                elif trade.trade_type == 'EXIT':
                    type_desc = "(Closed)"
                elif trade.trade_type == 'INCREASE':
                    type_desc = "(Added)"
                elif trade.trade_type == 'DECREASE':
                    type_desc = "(Reduced)"
                else:
                    type_desc = ""
                
                lines.append(
                    f"   {time_str} {dir_icon} {action} {trade.friendly_name}"
                )
                lines.append(
                    f"         {trade.size:.6f} @ ${trade.price:,.2f} = ${trade.value:,.2f} {type_desc}"
                )
            
            lines.append("   ─────────────────────────────────────────────")
        else:
            lines.append("   No trades today")
        
        lines.extend([
            "",
            "📊 CURRENT HOLDINGS",
        ])
        
        if data.holdings:
            for h in data.holdings:
                lines.append(f"   {h['friendly_name']:24s} ${h['value']:>10,.0f} ({h['percentage']:>4.0f}%)")
            lines.append(f"   {'Cash':24s} ${data.cash:>10,.0f}")
        else:
            lines.append("   No positions held")
            lines.append(f"   {'Cash':24s} ${data.cash:>10,.0f}")
        
        lines.extend([
            "",
            "=" * 50,
        ])
        
        return "\n".join(lines)
    
    def generate_dict(self) -> Dict[str, Any]:
        """Generate report as dictionary for API/JSON use."""
        data = self.gather_data()
        
        return {
            'timestamp': data.timestamp.isoformat(),
            'portfolio': {
                'total': data.total_equity,
                'yesterday': data.yesterday_equity,
                'daily_change': data.daily_change,
                'daily_change_pct': data.daily_change_pct,
            },
            'performance': {
                'week': data.week_change,
                'week_pct': data.week_change_pct,
                'month': data.month_change,
                'month_pct': data.month_change_pct,
                'inception': data.inception_change,
                'inception_pct': data.inception_change_pct,
            },
            'risk': {
                'drawdown_pct': data.drawdown_pct,
                'max_drawdown_pct': data.max_drawdown_pct,
                'status': data.risk_status,
            },
            'activity': {
                'total_trades': data.trades_total,
                'entries': data.trades_entry,
                'exits': data.trades_exit,
                'rebalancing': data.trades_rebalance,
                'costs': data.total_costs,
            },
            'trades': [
                {
                    'timestamp': t.timestamp.isoformat(),
                    'pair': t.pair,
                    'friendly_name': t.friendly_name,
                    'direction': t.direction,
                    'trade_type': t.trade_type,
                    'size': t.size,
                    'price': t.price,
                    'value': t.value,
                    'cost': t.cost,
                }
                for t in data.trades
            ],
            'holdings': data.holdings,
            'cash': data.cash,
        }