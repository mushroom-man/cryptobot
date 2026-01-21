# -*- coding: utf-8 -*-
"""
CryptoBot Reports Module
========================
Standalone reporting system for CryptoBot.

Decoupled from runner.py to enable:
- Reports independent of trading runs
- Historical report generation
- Multiple report types
- Testable in isolation

Usage:
    from cryptobot.reports import DailyReport, ReportSender
    
    # Generate report from database
    report = DailyReport(db, config)
    content = report.generate()
    
    # Send via email/pushover
    sender = ReportSender(config)
    sender.send(content, subject="CryptoBot Daily Report")
"""

from cryptobot.reports.daily_report import DailyReport
from cryptobot.reports.sender import ReportSender

__all__ = ['DailyReport', 'ReportSender']