# -*- coding: utf-8 -*-
"""
CryptoBot - Portfolio Management
=================================
Tracks positions, cash, equity, and P&L.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from cryptobot.shared.core.bar import Bar
from cryptobot.shared.core.order import Order, OrderSide
from cryptobot.shared.core.fill import Fill


@dataclass
class Position:
    """
    Single position in a trading pair.
    
    Attributes:
        pair: Trading pair
        size: Position size (positive=long, negative=short)
        avg_entry_price: Average entry price
        unrealized_pnl: Current unrealized P&L
        realized_pnl: Cumulative realized P&L for this pair
    """
    pair: str
    size: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    current_price: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        return self.size < 0
    
    @property
    def is_flat(self) -> bool:
        return abs(self.size) < 1e-10
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.size * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Cost basis of position."""
        return self.size * self.avg_entry_price
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    def update_price(self, price: float):
        """Update current price and recalculate unrealized P&L."""
        self.current_price = price
        if not self.is_flat:
            self.unrealized_pnl = (price - self.avg_entry_price) * self.size
    
    def apply_fill(self, fill: Fill) -> float:
        """
        Apply fill to position.
        
        Returns:
            Realized P&L from this fill
        """
        realized = 0.0
        
        if fill.is_buy:
            if self.size < 0:
                # Covering short
                cover_size = min(fill.fill_size, abs(self.size))
                realized = (self.avg_entry_price - fill.fill_price) * cover_size
                self.size += cover_size
                remaining = fill.fill_size - cover_size
                
                if remaining > 0:
                    # Opening long with remainder
                    self.avg_entry_price = fill.fill_price
                    self.size = remaining
            else:
                # Adding to long
                if self.size > 0:
                    # Weighted average entry
                    total_cost = (self.size * self.avg_entry_price) + (fill.fill_size * fill.fill_price)
                    self.size += fill.fill_size
                    self.avg_entry_price = total_cost / self.size
                else:
                    # Opening new long
                    self.size = fill.fill_size
                    self.avg_entry_price = fill.fill_price
        
        else:  # Sell
            if self.size > 0:
                # Closing long
                close_size = min(fill.fill_size, self.size)
                realized = (fill.fill_price - self.avg_entry_price) * close_size
                self.size -= close_size
                remaining = fill.fill_size - close_size
                
                if remaining > 0:
                    # Opening short with remainder
                    self.avg_entry_price = fill.fill_price
                    self.size = -remaining
            else:
                # Adding to short
                if self.size < 0:
                    # Weighted average entry
                    total_cost = (abs(self.size) * self.avg_entry_price) + (fill.fill_size * fill.fill_price)
                    self.size -= fill.fill_size
                    self.avg_entry_price = total_cost / abs(self.size)
                else:
                    # Opening new short
                    self.size = -fill.fill_size
                    self.avg_entry_price = fill.fill_price
        
        self.realized_pnl += realized
        self.update_price(self.current_price or fill.fill_price)
        
        return realized
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pair': self.pair,
            'size': self.size,
            'avg_entry_price': self.avg_entry_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
        }


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    drawdown: float
    positions: Dict[str, float]  # pair -> size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'equity': self.equity,
            'cash': self.cash,
            'positions_value': self.positions_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'drawdown': self.drawdown,
            **{f'pos_{k}': v for k, v in self.positions.items()},
        }


class Portfolio:
    """
    Portfolio manager - tracks all positions and P&L.
    
    Attributes:
        initial_capital: Starting capital
        cash: Current cash balance
        positions: Dict of Position objects by pair
    """
    
    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        
        # Tracking
        self.peak_equity = initial_capital
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.trade_count = 0
        
        # History
        self.snapshots: List[PortfolioSnapshot] = []
        self.fills: List[Fill] = []
    
    # =========================================================================
    # Position Access
    # =========================================================================
    
    def get_position(self, pair: str) -> float:
        """Get current position size for pair (0 if no position)."""
        if pair in self.positions:
            return self.positions[pair].size
        return 0.0
    
    def get_position_obj(self, pair: str) -> Position:
        """Get Position object for pair (creates if doesn't exist)."""
        if pair not in self.positions:
            self.positions[pair] = Position(pair=pair)
        return self.positions[pair]
    
    def get_all_positions(self) -> Dict[str, float]:
        """Get all non-zero positions."""
        return {
            pair: pos.size 
            for pair, pos in self.positions.items() 
            if not pos.is_flat
        }
    
    # =========================================================================
    # Price Updates
    # =========================================================================
    
    def update_price(self, pair: str, price: float):
        """Update price for a single pair."""
        if pair in self.positions:
            self.positions[pair].update_price(price)
    
    def update_prices(self, prices: Dict[str, float]):
        """Update prices for multiple pairs."""
        for pair, price in prices.items():
            self.update_price(pair, price)
    
    def mark_to_market(self, bar: Bar):
        """Update portfolio with new bar data."""
        self.update_price(bar.pair, bar.close)
    
    # =========================================================================
    # Order Execution
    # =========================================================================
    
    def apply_fill(self, fill: Fill):
        """
        Apply a fill to the portfolio.
        
        Updates:
            - Position size and avg entry
            - Cash (for realized P&L and commission)
            - Tracking stats
        """
        position = self.get_position_obj(fill.pair)
        
        # Apply to position
        realized_pnl = position.apply_fill(fill)
        
        # Update cash
        self.cash += realized_pnl - fill.commission
        
        # Track costs
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage
        self.trade_count += 1
        
        # Store fill
        fill.update_order_status()
        self.fills.append(fill)
    
    # =========================================================================
    # Portfolio Metrics
    # =========================================================================
    
    @property
    def positions_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def equity(self) -> float:
        """Total portfolio equity (cash + positions)."""
        return self.cash + self.positions_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Total realized P&L across all positions."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (equity - initial capital)."""
        return self.equity - self.initial_capital
    
    @property
    def total_return(self) -> float:
        """Total return as decimal (0.10 = 10%)."""
        return self.total_pnl / self.initial_capital
    
    @property
    def total_costs(self) -> float:
        """Total trading costs."""
        return self.total_commission + self.total_slippage
    
    @property
    def drawdown(self) -> float:
        """Current drawdown from peak (as positive decimal)."""
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from history."""
        if not self.snapshots:
            return self.drawdown
        return max(s.drawdown for s in self.snapshots)
    
    # =========================================================================
    # Position Sizing Helpers
    # =========================================================================
    
    def get_position_pct(self, pair: str) -> float:
        """Get position as percentage of equity."""
        pos = self.get_position(pair)
        if pair in self.positions:
            value = abs(pos * self.positions[pair].current_price)
            return value / self.equity if self.equity > 0 else 0.0
        return 0.0
    
    def get_total_exposure(self) -> float:
        """Get total exposure as percentage of equity."""
        total_value = sum(
            abs(pos.market_value) 
            for pos in self.positions.values()
        )
        return total_value / self.equity if self.equity > 0 else 0.0
    
    # =========================================================================
    # Snapshots
    # =========================================================================
    
    def record_snapshot(self, timestamp: datetime):
        """Record current portfolio state."""
        # Update peak
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            equity=self.equity,
            cash=self.cash,
            positions_value=self.positions_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_pnl=self.total_pnl,
            drawdown=self.drawdown,
            positions=self.get_all_positions(),
        )
        self.snapshots.append(snapshot)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()
        
        df = pd.DataFrame([s.to_dict() for s in self.snapshots])
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.fills:
            return pd.DataFrame()
        
        return pd.DataFrame([f.to_dict() for f in self.fills])
    
    # =========================================================================
    # Position Management
    # =========================================================================
    
    def flatten_position(self, pair: str, timestamp: datetime, 
                         price: float) -> Optional[Order]:
        """Create order to flatten a position."""
        pos = self.get_position(pair)
        if abs(pos) < 1e-10:
            return None
        
        return Order.from_target_position(
            pair=pair,
            current_position=pos,
            target_position=0.0,
            timestamp=timestamp,
            reference_price=price,
            reason="flatten"
        )
    
    def flatten_all(self, timestamp: datetime, 
                    prices: Dict[str, float]) -> List[Order]:
        """Create orders to flatten all positions."""
        orders = []
        for pair, pos in self.positions.items():
            if not pos.is_flat:
                order = self.flatten_position(
                    pair, timestamp, prices.get(pair, pos.current_price)
                )
                if order:
                    orders.append(order)
        return orders
    
    # =========================================================================
    # Results
    # =========================================================================
    
    def get_results(self) -> Dict[str, Any]:
        """Get portfolio results summary."""
        equity_curve = self.get_equity_curve()
        
        results = {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity,
            'total_return': self.total_return,
            'total_return_pct': self.total_return * 100,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_costs': self.total_costs,
            'trade_count': self.trade_count,
            'final_positions': self.get_all_positions(),
        }
        
        # Calculate Sharpe if we have equity curve
        if len(equity_curve) > 1:
            returns = equity_curve['equity'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Annualize assuming hourly data
                sharpe = returns.mean() / returns.std() * np.sqrt(8760)
                results['sharpe_ratio'] = sharpe
                
                # Sortino (downside deviation)
                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    sortino = returns.mean() / downside.std() * np.sqrt(8760)
                    results['sortino_ratio'] = sortino
        
        return results
    
    def print_summary(self):
        """Print portfolio summary."""
        results = self.get_results()
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Initial Capital:  ${results['initial_capital']:,.2f}")
        print(f"Final Equity:     ${results['final_equity']:,.2f}")
        print(f"Total Return:     {results['total_return_pct']:+.2f}%")
        print(f"Max Drawdown:     {results['max_drawdown_pct']:.2f}%")
        
        if 'sharpe_ratio' in results:
            print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        if 'sortino_ratio' in results:
            print(f"Sortino Ratio:    {results['sortino_ratio']:.2f}")
        
        print(f"\nTrades:           {results['trade_count']}")
        print(f"Total Costs:      ${results['total_costs']:,.2f}")
        print("="*60)
    
    def __repr__(self) -> str:
        return f"Portfolio(equity=${self.equity:,.2f}, positions={len(self.get_all_positions())})"
