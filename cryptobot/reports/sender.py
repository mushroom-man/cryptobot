# -*- coding: utf-8 -*-
"""
CryptoBot - Report Sender
=========================
Send reports via Email and Pushover.

Usage:
    from cryptobot.reports.sender import ReportSender
    
    sender = ReportSender(config)
    sender.send_email(report_text, subject="Daily Report")
    sender.send_pushover(short_message)
    sender.send_all(report_text, short_message)
"""

import smtplib
import os
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class ReportSender:
    """
    Multi-channel report sender.
    
    Supports:
    - Email via SMTP (Gmail)
    - Pushover mobile notifications
    """
    
    def __init__(self, config: dict):
        """
        Initialise sender with configuration.
        
        Args:
            config: Configuration dictionary with notifications section
        """
        self.config = config
        self.notifications = config.get('notifications', {})
        self.enabled = self.notifications.get('enabled', False)
    
    def send_email(
        self,
        content: str,
        subject: str = None,
        recipients: List[str] = None,
    ) -> bool:
        """
        Send report via email.
        
        Args:
            content: Report text content
            subject: Email subject (defaults to dated subject)
            recipients: Override recipient list (optional)
        
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.info("Notifications disabled")
            return False
        
        email_config = self.notifications.get('email', {})
        
        sender = email_config.get('sender') or os.getenv('EMAIL_SENDER')
        password = email_config.get('password') or os.getenv('EMAIL_PASSWORD')
        smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = email_config.get('smtp_port', 587)
        
        if recipients is None:
            recipients = email_config.get('recipients', [])
            env_recipient = os.getenv('EMAIL_RECIPIENT')
            if env_recipient and env_recipient not in recipients:
                recipients.append(env_recipient)
        
        if not sender or not password or not recipients:
            logger.warning("Email not configured (missing sender, password, or recipients)")
            return False
        
        if subject is None:
            subject = f"CryptoBot Daily Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        
        try:
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            msg.attach(MIMEText(content, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipients, msg.as_string())
            
            logger.info(f"✅ Email sent to {len(recipients)} recipient(s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Email error: {e}")
            return False
    
    def send_pushover(
        self,
        message: str,
        title: str = "CryptoBot Report",
    ) -> bool:
        """
        Send notification via Pushover.
        
        Args:
            message: Short notification message
            title: Notification title
        
        Returns:
            True if at least one notification sent
        """
        if not self.enabled:
            logger.info("Notifications disabled")
            return False
        
        pushover_config = self.notifications.get('pushover', {})
        
        # Handle multiple config formats
        pushover_users = []
        
        if isinstance(pushover_config, list):
            pushover_users = pushover_config
        elif isinstance(pushover_config, dict):
            if 'users' in pushover_config:
                pushover_users = pushover_config.get('users', [])
            elif 'user_key' in pushover_config:
                pushover_users = [pushover_config]
        
        # Check environment variables
        if not pushover_users:
            user_key = os.getenv('PUSHOVER_USER_KEY')
            api_token = os.getenv('PUSHOVER_API_TOKEN')
            if user_key and api_token:
                pushover_users = [{'user_key': user_key, 'api_token': api_token}]
        
        if not pushover_users:
            logger.info("Pushover not configured")
            return False
        
        # Get shared token if specified
        shared_api_token = None
        if isinstance(pushover_config, dict):
            shared_api_token = pushover_config.get('api_token')
        
        success = False
        
        for i, user in enumerate(pushover_users):
            user_key = user.get('user_key')
            api_token = user.get('api_token') or shared_api_token or os.getenv('PUSHOVER_API_TOKEN')
            
            if not user_key or not api_token:
                continue
            
            try:
                response = requests.post(
                    "https://api.pushover.net/1/messages.json",
                    data={
                        "token": api_token,
                        "user": user_key,
                        "title": title,
                        "message": message,
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
        
        return success
    
    def send_all(
        self,
        report_content: str,
        short_message: str = None,
        subject: str = None,
    ) -> Dict[str, bool]:
        """
        Send report via all configured channels.
        
        Args:
            report_content: Full report text
            short_message: Short message for push notifications
            subject: Email subject
        
        Returns:
            Dict with channel -> success status
        """
        results = {}
        
        # Send email
        results['email'] = self.send_email(report_content, subject=subject)
        
        # Send pushover if short message provided
        if short_message:
            results['pushover'] = self.send_pushover(short_message)
        
        return results
    
    @staticmethod
    def format_short_message(
        total_equity: float,
        daily_pnl: float,
        drawdown: float,
        trades_count: int,
    ) -> str:
        """
        Format a short message for push notifications.
        
        Args:
            total_equity: Current total equity
            daily_pnl: Daily P&L
            drawdown: Current drawdown (as decimal, e.g., 0.05 for 5%)
            trades_count: Number of trades today
        
        Returns:
            Formatted short message
        """
        lines = [
            f"💰 ${total_equity:,.0f}",
            f"📈 P&L: ${daily_pnl:+,.0f}",
            f"📉 DD: {drawdown * 100:.1f}%",
            f"🔄 Trades: {trades_count}",
        ]
        return "\n".join(lines)