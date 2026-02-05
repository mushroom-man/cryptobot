#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CryptoBot - Send Status Email (One-Off)
========================================
Sends a snapshot of current system status as an email.
Pulls live data from the database — not a dummy.

Usage:
    cd ~/cryptobot
    python scripts/send_status.py
    python scripts/send_status.py --config cryptobot/config/settings.yaml
"""

import sys
import os
import argparse
from datetime import datetime, timezone

# Ensure project root is on path
project_root = os.path.expanduser('~/cryptobot')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from cryptobot.config.loader import load_config
from cryptobot.data.database import get_database
from cryptobot.reports.sender import ReportSender

# Timezone
try:
    from zoneinfo import ZoneInfo
    MELBOURNE_TZ = ZoneInfo('Australia/Melbourne')
except ImportError:
    import pytz
    MELBOURNE_TZ = pytz.timezone('Australia/Melbourne')


def melbourne_now():
    return datetime.now(MELBOURNE_TZ)


def build_status_email(db, config):
    """Build status email content from live DB state."""
    
    now = melbourne_now()
    pairs = config.get('pairs', [])
    initial_capital = config.get('position', {}).get('initial_capital', 100000)
    
    lines = [
        f"CRYPTOBOT STATUS - {now.strftime('%Y-%m-%d %H:%M')} Melbourne",
        "=" * 50,
        "",
    ]
    
    # --- Equity ---
    equity = db.get_latest_equity()
    if equity:
        total = equity.get('total_equity', initial_capital)
        peak = equity.get('peak_equity', total)
        dd = equity.get('drawdown', 0)
        dd_pct = dd * 100 if dd < 1 else dd  # Handle both ratio and percentage
        daily_pnl = equity.get('daily_pnl', 0)
        
        lines.append("PORTFOLIO")
        lines.append("-" * 50)
        lines.append(f"  Total Equity:      ${total:>14,.2f}")
        lines.append(f"  Peak Equity:       ${peak:>14,.2f}")
        lines.append(f"  Daily P&L:         ${daily_pnl:>+14,.2f}")
        lines.append(f"  Drawdown:          {dd_pct:>13.1f}%")
        lines.append("")
    else:
        lines.append("PORTFOLIO")
        lines.append("-" * 50)
        lines.append(f"  Total Equity:      ${initial_capital:>14,.2f} (initial)")
        lines.append("")
    
    # --- Current Positions ---
    positions_df = db.get_current_positions()
    lines.append("POSITIONS")
    lines.append("-" * 50)
    
    n_active = 0
    net_exposure = 0.0
    long_exposure = 0.0
    short_exposure = 0.0
    
    if positions_df is not None and len(positions_df) > 0:
        for _, row in positions_df.iterrows():
            pair = row['pair']
            position = row.get('position', 0) or 0
            
            if abs(position) < 0.0001:
                continue
            
            n_active += 1
            current_price = db.get_latest_price(pair) or 0
            market_value = abs(position * current_price)
            direction = "SHORT" if position < 0 else "LONG"
            
            if position > 0:
                long_exposure += market_value
            else:
                short_exposure += market_value
            
            # Net P&L for this position: short profits when price falls
            net_value = position * current_price  # negative for shorts
            net_exposure += net_value
            
            lines.append(f"  {pair:10s} {direction:5s}  {abs(position):>12.6f} @ ${current_price:>10,.2f} = ${market_value:>10,.2f}")
    
    if n_active == 0:
        lines.append("  No open positions (all cash)")
    else:
        lines.append("")
        if long_exposure > 0:
            lines.append(f"  Long Exposure:     ${long_exposure:>14,.2f}")
        if short_exposure > 0:
            lines.append(f"  Short Exposure:    ${short_exposure:>14,.2f}")
        
        # Calculate live equity: cash + net_exposure
        cash = equity.get('cash', initial_capital) if equity else initial_capital
        live_equity = cash + net_exposure
        lines.append(f"  Net Equity (live): ${live_equity:>14,.2f}")
    
    lines.append("")
    
    # --- Latest Signals ---
    lines.append("SIGNALS")
    lines.append("-" * 50)
    
    for pair in pairs:
        sig = db.get_latest_signal(pair)
        if sig:
            signal = sig.get('signal', 'UNKNOWN')
            regime = sig.get('regime', '?')
            confidence = sig.get('confidence', 0) or 0
            
            # Signal icon
            if signal in ('LONG', 'BOOSTED_LONG'):
                icon = "\U0001f7e2"
            elif signal == 'SHORT':
                icon = "\U0001f534"
            else:
                icon = "\u26aa"
            
            lines.append(f"  {icon} {pair:10s} {signal:12s} S{regime}  HR: {confidence:.0%}")
        else:
            lines.append(f"  \u26aa {pair:10s} No signal recorded")
    
    lines.append("")
    
    # --- Risk Parity Weights ---
    weights_df = db.get_current_weights()
    if weights_df is not None and len(weights_df) > 0:
        lines.append("RISK PARITY WEIGHTS")
        lines.append("-" * 50)
        for _, row in weights_df.iterrows():
            pair = row.get('pair', '')
            weight = row.get('weight', 0)
            lines.append(f"  {pair:10s} {weight:>6.1%}")
        lines.append("")
    
    # --- Posture (from latest signals) ---
    try:
        from cryptobot.reports.posture import PostureData, get_posture_messages
        
        # Build minimal posture data from DB state
        signals_list = []
        for pair in pairs:
            sig = db.get_latest_signal(pair)
            if sig:
                signals_list.append({
                    'pair': pair,
                    'signal': sig.get('signal', 'FLAT'),
                    'state': sig.get('regime', 0),
                    'duration': 0,  # Unknown from DB query
                    'target_position': sig.get('target_position', 0),
                    'weight': 0,
                    'quality_scalar': sig.get('confidence', 0) or 0,
                })
        
        if signals_list:
            from cryptobot.reports.posture import build_posture_data
            
            n_active_positions = 0
            if positions_df is not None:
                for _, row in positions_df.iterrows():
                    if row.get('position') and abs(row['position']) > 0.0001:
                        n_active_positions += 1
            
            posture_data = build_posture_data(
                signals=signals_list,
                trades_executed=[],
                vol_scalar=1.0,
                dd_scalar=1.0,
                equity=equity or {'total_equity': initial_capital, 'peak_equity': initial_capital},
                initial_capital=initial_capital,
                previous_position_count=n_active_positions,
                max_drawdown_pct=config.get('risk', {}).get('max_drawdown', 0.20) * 100,
                transition_pairs=[],
            )
            posture_msgs = get_posture_messages(posture_data)
            
            if posture_msgs:
                lines.append("STRATEGY POSTURE")
                lines.append("-" * 50)
                for msg in posture_msgs:
                    lines.append(f"  {msg}")
                    lines.append("")
    except Exception as e:
        lines.append(f"  (Posture unavailable: {e})")
        lines.append("")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def main():
    try:
        parser = argparse.ArgumentParser(description='Send CryptoBot status email')
        parser.add_argument('--config', '-c', type=str,
                            default=os.path.join(project_root, 'cryptobot', 'config', 'settings.yaml'),
                            help='Path to config file')
        args = parser.parse_args()
        config_path = args.config
    except SystemExit:
        # Thonny may pass unexpected args — use default
        config_path = os.path.join(project_root, 'cryptobot', 'config', 'settings.yaml')
    
    config = load_config(config_path)
    db = get_database()
    
    content = build_status_email(db, config)
    
    # Print to console
    print(content)
    print()
    
    # Send email
    if config.get('notifications', {}).get('enabled'):
        sender = ReportSender(config)
        subject = f"CryptoBot Status - {melbourne_now().strftime('%Y-%m-%d %H:%M')}"
        sender.send_email(content, subject=subject)
        print(f"✅ Email sent: {subject}")
    else:
        print("⚠️  Notifications disabled in config. Status printed above but not emailed.")


if __name__ == "__main__":
    main()