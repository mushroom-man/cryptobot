# -*- coding: utf-8 -*-
"""
CryptoBot - Weekly Report Generator
===================================
Partner-friendly weekly report with performance summary.

Sent Monday mornings at 12:05 Melbourne time.

Usage:
    from cryptobot.reports.weekly_report import WeeklyReport
    
    report = WeeklyReport(db, config)
    content = report.generate()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .pair_names import get_friendly_name

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
class WeeklyReportData:
    """Container for weekly report data."""
    timestamp: datetime
    week_start: datetime
    week_end: datetime
    
    # Portfolio summary
    start_equity: float
    end_equity: float
    weekly_change: float
    weekly_change_pct: float
    
    # Daily breakdown
    best_day: Dict[str, Any]
    worst_day: Dict[str, Any]
    
    # Performance periods
    month_change: float
    month_change_pct: float
    year_change: float
    year_change_pct: float
    
    # Risk
    max_drawdown_pct: float
    peak_value: float
    risk_status: str
    
    # Activity
    total_trades: int
    trades_entry: int
    trades_exit: int
    trades_rebalance: int
    total_costs: float
    
    # Holdings change
    holdings_start: List[Dict[str, Any]]
    holdings_end: List[Dict[str, Any]]
    cash_start: float
    cash_end: float


class WeeklyReport:
    """
    Partner-friendly weekly report generator.
    
    Summarizes the past week's performance in clear,
    non-technical language suitable for partners.
    """
    
    def __init__(self, db, config: dict):
        """
        Initialize weekly report generator.
        
        Args:
            db: Database instance
            config: Configuration dictionary
        """
        self.db = db
        self.config = config
        self.pairs = config.get('pairs', [])
        self.initial_capital = config.get('position', {}).get('initial_capital', 100000)
        self.max_drawdown = config.get('risk', {}).get('max_drawdown', 0.20)
    
    def gather_data(self) -> WeeklyReportData:
        """Gather all data needed for the weekly report."""
        now = melbourne_now()
        
        # Calculate week boundaries (Monday to Sunday)
        # If today is Monday, report on the previous week
        days_since_monday = now.weekday()
        if days_since_monday == 0:  # It's Monday
            week_end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59)
            week_start = (week_end - timedelta(days=6)).replace(hour=0, minute=0, second=0)
        else:
            # Mid-week - report on current partial week
            week_start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0)
            week_end = now
        
        # Get equity at start and end of week
        start_equity = self._get_equity_at(week_start)
        end_equity = self._get_equity_at(week_end)
        
        weekly_change = end_equity - start_equity
        weekly_change_pct = (weekly_change / start_equity * 100) if start_equity else 0
        
        # Get daily equity for best/worst day
        daily_equity = self._get_daily_equity(week_start, week_end)
        best_day, worst_day = self._find_best_worst_days(daily_equity)
        
        # Performance periods
        month_ago_equity = self._get_equity_days_ago(30)
        year_ago_equity = self._get_equity_days_ago(365)
        
        month_change = end_equity - month_ago_equity
        month_change_pct = (month_change / month_ago_equity * 100) if month_ago_equity else 0
        
        year_change = end_equity - year_ago_equity
        year_change_pct = (year_change / year_ago_equity * 100) if year_ago_equity else 0
        
        # Risk metrics
        peak_value = self._get_peak_equity()
        max_drawdown_pct = ((peak_value - end_equity) / peak_value * 100) if peak_value else 0
        
        if max_drawdown_pct < self.max_drawdown * 50:
            risk_status = "Normal"
        elif max_drawdown_pct < self.max_drawdown * 80:
            risk_status = "Elevated"
        else:
            risk_status = "High"
        
        # Week's trading activity
        trade_summary = self._get_week_trades(week_start, week_end)
        
        # Holdings comparison
        holdings_start = self._get_holdings_at(week_start)
        holdings_end, cash_end = self._get_current_holdings()
        cash_start = start_equity - sum(h['value'] for h in holdings_start)
        
        return WeeklyReportData(
            timestamp=now,
            week_start=week_start,
            week_end=week_end,
            start_equity=start_equity,
            end_equity=end_equity,
            weekly_change=weekly_change,
            weekly_change_pct=weekly_change_pct,
            best_day=best_day,
            worst_day=worst_day,
            month_change=month_change,
            month_change_pct=month_change_pct,
            year_change=year_change,
            year_change_pct=year_change_pct,
            max_drawdown_pct=max_drawdown_pct,
            peak_value=peak_value,
            risk_status=risk_status,
            total_trades=trade_summary['total'],
            trades_entry=trade_summary['entry'],
            trades_exit=trade_summary['exit'],
            trades_rebalance=trade_summary['rebalance'],
            total_costs=trade_summary['costs'],
            holdings_start=holdings_start,
            holdings_end=holdings_end,
            cash_start=cash_start,
            cash_end=cash_end,
        )
    
    def _get_equity_at(self, dt: datetime) -> float:
        """Get equity value at a specific datetime."""
        try:
            df = self.db.get_equity_history(
                start=(dt - timedelta(hours=24)).isoformat(),
                end=(dt + timedelta(hours=24)).isoformat()
            )
            if df is not None and len(df) > 0:
                # Find closest to target datetime
                df_reset = df.reset_index()
                df_reset['diff'] = abs(pd.to_datetime(df_reset['timestamp']) - pd.to_datetime(dt))
                closest = df_reset.loc[df_reset['diff'].idxmin()]
                return closest['total_equity']
        except Exception:
            pass
        
        return self.initial_capital
    
    def _get_equity_days_ago(self, days: int) -> float:
        """Get equity value from N days ago."""
        target_date = datetime.now(timezone.utc) - timedelta(days=days)
        return self._get_equity_at(target_date)
    
    def _get_daily_equity(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Get daily equity values for a date range."""
        try:
            df = self.db.get_equity_history(start=start.isoformat(), end=end.isoformat())
            if df is not None and len(df) > 0:
                df_reset = df.reset_index()
                df_reset['date'] = pd.to_datetime(df_reset['timestamp']).dt.date
                daily = df_reset.groupby('date')['total_equity'].last().reset_index()
                return daily
        except Exception:
            pass
        
        return pd.DataFrame(columns=['date', 'total_equity'])
    
    def _find_best_worst_days(self, daily_equity: pd.DataFrame) -> tuple:
        """Find best and worst performing days."""
        best_day = {'day': 'N/A', 'change': 0}
        worst_day = {'day': 'N/A', 'change': 0}
        
        if len(daily_equity) < 2:
            return best_day, worst_day
        
        daily_equity = daily_equity.sort_values('date')
        daily_equity['change'] = daily_equity['total_equity'].diff()
        daily_equity['day_name'] = pd.to_datetime(daily_equity['date']).dt.strftime('%A')
        
        # Remove first row (NaN change)
        daily_equity = daily_equity.dropna(subset=['change'])
        
        if len(daily_equity) == 0:
            return best_day, worst_day
        
        best_idx = daily_equity['change'].idxmax()
        worst_idx = daily_equity['change'].idxmin()
        
        best_row = daily_equity.loc[best_idx]
        worst_row = daily_equity.loc[worst_idx]
        
        best_day = {'day': best_row['day_name'], 'change': best_row['change']}
        worst_day = {'day': worst_row['day_name'], 'change': worst_row['change']}
        
        return best_day, worst_day
    
    def _get_peak_equity(self) -> float:
        """Get all-time peak equity."""
        db_equity = self.db.get_latest_equity()
        if db_equity and db_equity.get('peak_equity'):
            return db_equity['peak_equity']
        return self.initial_capital
    
    def _get_week_trades(self, start: datetime, end: datetime) -> Dict:
        """Get trade summary for the week."""
        trades_df = self.db.get_trades(start=start.isoformat(), end=end.isoformat())
        
        summary = {
            'total': 0,
            'entry': 0,
            'exit': 0,
            'rebalance': 0,
            'costs': 0.0,
        }
        
        if trades_df is None or len(trades_df) == 0:
            return summary
        
        summary['total'] = len(trades_df)
        summary['costs'] = trades_df['transaction_cost'].sum() if 'transaction_cost' in trades_df else 0
        
        if 'trade_type' in trades_df.columns:
            type_counts = trades_df['trade_type'].value_counts()
            summary['entry'] = type_counts.get('ENTRY', 0)
            summary['exit'] = type_counts.get('EXIT', 0)
            summary['rebalance'] = type_counts.get('INCREASE', 0) + type_counts.get('DECREASE', 0)
        else:
            summary['rebalance'] = summary['total']
        
        return summary
    
    def _get_holdings_at(self, dt: datetime) -> List[Dict]:
        """Get holdings at a specific datetime (approximation from equity)."""
        # For now, return empty - this is hard to reconstruct historically
        # Could be enhanced with historical portfolio snapshots
        return []
    
    def _get_current_holdings(self) -> tuple:
        """Get current holdings with live prices."""
        holdings = []
        
        positions_df = self.db.get_current_positions()
        
        if positions_df is None or len(positions_df) == 0:
            return holdings, self.initial_capital
        
        total_invested = 0
        
        for _, row in positions_df.iterrows():
            pair = row['pair']
            position = row['position'] or 0
            
            if abs(position) < 0.0001:
                continue
            
            current_price = self.db.get_latest_price(pair)
            if current_price is None:
                current_price = row.get('current_price', 0)
            
            value = abs(position * current_price) if current_price else 0
            total_invested += value
            
            holdings.append({
                'pair': pair,
                'friendly_name': get_friendly_name(pair),
                'value': value,
            })
        
        holdings.sort(key=lambda x: x['value'], reverse=True)
        
        db_equity = self.db.get_latest_equity()
        if db_equity and db_equity.get('total_equity'):
            cash = db_equity['total_equity'] - total_invested
        else:
            cash = self.initial_capital - total_invested
        
        return holdings, cash
    
    def generate(self, posture_block: str = None) -> str:
        """Generate the formatted weekly report.
        
        Args:
            posture_block: Optional pre-formatted weekly posture text
                from weekly_posture.generate_weekly_posture()
        """
        data = self.gather_data()
        return self._format_report(data, posture_block=posture_block)
    
    def _format_report(self, data: WeeklyReportData, posture_block: str = None) -> str:
        """Format report data into partner-friendly string."""
        
        week_label = f"{data.week_start.strftime('%d %b')} - {data.week_end.strftime('%d %b %Y')}"
        
        lines = [
            "=" * 50,
            "📊 CRYPTOBOT WEEKLY REPORT",
            f"📅 Week of {week_label}",
            "=" * 50,
            "",
            "💰 PORTFOLIO SUMMARY",
            f"   Start of Week:   ${data.start_equity:>14,.2f}",
            f"   End of Week:     ${data.end_equity:>14,.2f}",
            f"   Weekly Change:   ${data.weekly_change:>+14,.2f} ({data.weekly_change_pct:+.2f}%)",
            "",
            "📈 PERFORMANCE",
            f"   Best Day:        ${data.best_day['change']:>+14,.2f} ({data.best_day['day']})",
            f"   Worst Day:       ${data.worst_day['change']:>+14,.2f} ({data.worst_day['day']})",
            "",
            f"   Month to Date:   ${data.month_change:>+14,.2f} ({data.month_change_pct:+.2f}%)",
            f"   Year to Date:    ${data.year_change:>+14,.2f} ({data.year_change_pct:+.2f}%)",
            "",
            "📉 RISK SUMMARY",
        ]
        
        # Risk status with icon
        if data.risk_status == "Normal":
            risk_icon = "✅"
        elif data.risk_status == "Elevated":
            risk_icon = "⚠️"
        else:
            risk_icon = "🚨"
        
        lines.extend([
            f"   Max Drawdown:    {data.max_drawdown_pct:>14.1f}%",
            f"   Peak Value:      ${data.peak_value:>14,.2f}",
            f"   Status:          {risk_icon} {data.risk_status}",
            "",
            "🔄 WEEKLY ACTIVITY",
            f"   Total Trades:    {data.total_trades:>14}",
        ])
        
        if data.total_trades > 0:
            if data.trades_entry > 0:
                lines.append(f"   New Positions:   {data.trades_entry:>14}")
            if data.trades_exit > 0:
                lines.append(f"   Exits:           {data.trades_exit:>14}")
            if data.trades_rebalance > 0:
                lines.append(f"   Rebalancing:     {data.trades_rebalance:>14}")
            lines.append(f"   Total Costs:     ${data.total_costs:>14,.2f}")
        
        lines.extend([
            "",
            "📊 CURRENT HOLDINGS",
        ])
        
        if data.holdings_end:
            total_invested = sum(h['value'] for h in data.holdings_end)
            total_value = total_invested + data.cash_end
            
            for h in data.holdings_end:
                pct = (h['value'] / total_value * 100) if total_value else 0
                lines.append(f"   {h['friendly_name']:24s} ${h['value']:>10,.0f} ({pct:>4.0f}%)")
            
            cash_pct = (data.cash_end / total_value * 100) if total_value else 0
            lines.append(f"   {'Cash':24s} ${data.cash_end:>10,.0f} ({cash_pct:>4.0f}%)")
        else:
            lines.append(f"   {'Cash':24s} ${data.cash_end:>10,.0f}")
        
        # Weekly posture (context panel + templates)
        if posture_block:
            lines.extend([
                "",
                posture_block,
            ])
        
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
            'week_start': data.week_start.isoformat(),
            'week_end': data.week_end.isoformat(),
            'portfolio': {
                'start': data.start_equity,
                'end': data.end_equity,
                'change': data.weekly_change,
                'change_pct': data.weekly_change_pct,
            },
            'performance': {
                'best_day': data.best_day,
                'worst_day': data.worst_day,
                'month': data.month_change,
                'month_pct': data.month_change_pct,
                'year': data.year_change,
                'year_pct': data.year_change_pct,
            },
            'risk': {
                'max_drawdown_pct': data.max_drawdown_pct,
                'peak_value': data.peak_value,
                'status': data.risk_status,
            },
            'activity': {
                'total_trades': data.total_trades,
                'entries': data.trades_entry,
                'exits': data.trades_exit,
                'rebalancing': data.trades_rebalance,
                'costs': data.total_costs,
            },
            'holdings': data.holdings_end,
            'cash': data.cash_end,
        }