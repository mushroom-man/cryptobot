# -*- coding: utf-8 -*-
"""
CryptoBot - Value at Risk Calculator
====================================
Portfolio risk metrics including VaR and volatility.

Usage:
    from cryptobot.risk.var import calculate_var
    
    var_metrics = calculate_var(db, pairs=['ETHUSD', 'XBTUSD'])
    print(f"95% VaR: ${var_metrics['var_95']:,.2f}")
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def calculate_var(
    db,
    pairs: List[str],
    confidence: float = 0.95,
    horizon_days: int = 1
) -> Dict:
    """
    Calculate Value at Risk for current portfolio.
    
    Args:
        db: Database instance
        pairs: List of trading pairs
        confidence: Confidence level (0.95 = 95%)
        horizon_days: Time horizon in days
    
    Returns:
        Dict with VaR metrics:
            - var_95: 95% VaR in dollars
            - var_99: 99% VaR in dollars
            - var_95_pct: 95% VaR as percentage
            - var_99_pct: 99% VaR as percentage
            - portfolio_vol: Annualized volatility
    """
    try:
        positions_df = db.get_current_positions()
        equity_data = db.get_latest_equity()
        
        if positions_df is None or len(positions_df) == 0:
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
        
        # Calculate total invested from current positions
        total_invested = 0.0
        for _, row in positions_df.iterrows():
            pair = row['pair']
            position = row.get('position', 0)
            
            if position:
                # Get live price from OHLCV
                live_price = db.get_latest_price(pair)
                if live_price is None:
                    live_price = row.get('current_price', 0)
                
                if live_price:
                    total_invested += abs(position * live_price)
        
        if total_invested <= 0:
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
        
        # Get historical returns for VaR calculation
        returns_data = {}
        start_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        
        for pair in pairs:
            df = db.get_ohlcv(pair, start=start_date)
            if df is not None and len(df) > 0:
                # Daily returns from hourly data
                daily = df['close'].resample('24h').last().ffill().pct_change().dropna()
                returns_data[pair] = daily
        
        if not returns_data:
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
        
        # Combine into DataFrame
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 30:
            return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}
        
        # Get position weights
        weights = {}
        
        for _, row in positions_df.iterrows():
            pair = row['pair']
            position = row.get('position', 0)
            
            if pair in returns_df.columns and position:
                # Get live price
                live_price = db.get_latest_price(pair)
                if live_price is None:
                    live_price = row.get('current_price', 0)
                
                if live_price:
                    position_value = abs(position * live_price)
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
        total_equity = equity_data['total_equity'] if equity_data else total_invested
        
        return {
            'var_95': var_95 * total_equity,    # 95% VaR in dollars
            'var_99': var_99 * total_equity,    # 99% VaR in dollars
            'var_95_pct': var_95,               # 95% VaR as percentage
            'var_99_pct': var_99,               # 99% VaR as percentage
            'portfolio_vol': portfolio_vol,     # Annualized volatility
        }
    
    except Exception as e:
        logger.error(f"VaR calculation failed: {e}")
        return {'var_95': 0, 'var_99': 0, 'portfolio_vol': 0}