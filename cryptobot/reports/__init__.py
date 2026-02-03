# -*- coding: utf-8 -*-
"""
CryptoBot - Reports Module
==========================
Partner-friendly reporting for daily and weekly summaries.

Usage:
    from cryptobot.reports import DailyReport, WeeklyReport, ReportSender
    
    # Generate daily report
    daily = DailyReport(db, config)
    content = daily.generate()
    
    # Generate weekly report
    weekly = WeeklyReport(db, config)
    content = weekly.generate()
    
    # Send reports
    sender = ReportSender(config)
    sender.send_email(content, subject="Daily Report")
"""

from .daily_report import DailyReport
from .weekly_report import WeeklyReport
from .sender import ReportSender
from .pair_names import get_friendly_name, get_short_name

__all__ = [
    'DailyReport',
    'WeeklyReport', 
    'ReportSender',
    'get_friendly_name',
    'get_short_name',
]