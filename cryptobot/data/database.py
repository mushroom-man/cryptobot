# -*- coding: utf-8 -*-
"""
CryptoBot - Database Reader
============================
Query OHLCV data and other tables from TimescaleDB.

Usage:
    from cryptobot.datasources.database import Database
    
    db = Database()
    df = db.get_ohlcv("XBTUSD", start="2020-01-01", end="2024-12-31")
"""

import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from typing import Optional, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://cryptobot:Carljung#3001@localhost:5432/cryptobot"
)


class Database:
    """Database interface for CryptoBot."""
    
    def __init__(self, connection_url: str = None):
        """
        Initialize database connection.
        
        Args:
            connection_url: PostgreSQL connection string. 
                           Uses DATABASE_URL env var if not provided.
        """
        self.connection_url = connection_url or DATABASE_URL
        self._engine = None
    
    @property
    def engine(self):
        """Lazy-load database engine."""
        if self._engine is None:
            self._engine = create_engine(self.connection_url)
        return self._engine
    
    # =========================================================================
    # OHLCV Data
    # =========================================================================
    
    def get_ohlcv(
        self,
        pair: str,
        start: str = None,
        end: str = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a trading pair.
        
        Args:
            pair: Trading pair (e.g., "XBTUSD")
            start: Start date (ISO format or datetime)
            end: End date (ISO format or datetime)
            limit: Maximum rows to return
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, volume_quote
        """
        query = """
            SELECT timestamp, open, high, low, close, volume, volume_quote
            FROM ohlcv
            WHERE pair = :pair
        """
        params = {"pair": pair}
        
        if start:
            query += " AND timestamp >= :start"
            params["start"] = start
        
        if end:
            query += " AND timestamp <= :end"
            params["end"] = end
        
        query += " ORDER BY timestamp"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    def get_latest_price(self, pair: str) -> Optional[float]:
        """Get the most recent close price for a pair."""
        query = """
            SELECT close 
            FROM ohlcv 
            WHERE pair = :pair 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"pair": pair})
            row = result.fetchone()
            return row[0] if row else None
    
    def get_available_pairs(self) -> List[str]:
        """Get list of all pairs in the database."""
        query = "SELECT DISTINCT pair FROM ohlcv ORDER BY pair"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return [row[0] for row in result]
    
    def get_data_range(self, pair: str) -> dict:
        """Get date range and row count for a pair."""
        query = """
            SELECT 
                MIN(timestamp) as start,
                MAX(timestamp) as end,
                COUNT(*) as rows
            FROM ohlcv 
            WHERE pair = :pair
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"pair": pair})
            row = result.fetchone()
            if row:
                return {
                    "start": row[0],
                    "end": row[1],
                    "rows": row[2]
                }
            return {"start": None, "end": None, "rows": 0}
    
    # =========================================================================
    # Features
    # =========================================================================
    
    def save_features(self, df: pd.DataFrame, pair: str) -> int:
        """
        Save computed features to the features table.
        
        Args:
            df: DataFrame with features (index should be timestamp)
            pair: Trading pair
        
        Returns:
            Number of rows inserted
        """
        df_to_save = df.copy()
        df_to_save['pair'] = pair
        
        if df_to_save.index.name == 'timestamp':
            df_to_save = df_to_save.reset_index()
        
        df_to_save.to_sql(
            'features',
            self.engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        return len(df_to_save)
    
    def get_features(
        self,
        pair: str,
        start: str = None,
        end: str = None
    ) -> pd.DataFrame:
        """Get computed features for a pair."""
        query = """
            SELECT *
            FROM features
            WHERE pair = :pair
        """
        params = {"pair": pair}
        
        if start:
            query += " AND timestamp >= :start"
            params["start"] = start
        
        if end:
            query += " AND timestamp <= :end"
            params["end"] = end
        
        query += " ORDER BY timestamp"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    # =========================================================================
    # Trades
    # =========================================================================
    
    def record_trade(
        self,
        pair: str,
        strategy: str,
        direction: str,
        size: float,
        price: float,
        slippage_bps: float = 0,
        transaction_cost: float = 0,
        execution_type: str = "backtest",
        order_id: str = None,
        notes: str = None
    ) -> int:
        """
        Record a trade in the database.
        
        Returns:
            Trade ID
        """
        query = """
            INSERT INTO trades 
            (timestamp, pair, strategy, direction, size, price, slippage_bps, 
             transaction_cost, execution_type, order_id, notes)
            VALUES 
            (NOW(), :pair, :strategy, :direction, :size, :price, :slippage_bps,
             :transaction_cost, :execution_type, :order_id, :notes)
            RETURNING id
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {
                "pair": pair,
                "strategy": strategy,
                "direction": direction,
                "size": size,
                "price": price,
                "slippage_bps": slippage_bps,
                "transaction_cost": transaction_cost,
                "execution_type": execution_type,
                "order_id": order_id,
                "notes": notes
            })
            conn.commit()
            return result.fetchone()[0]
    
    def get_trades(
        self,
        pair: str = None,
        strategy: str = None,
        start: str = None,
        end: str = None
    ) -> pd.DataFrame:
        """Get trade history."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = {}
        
        if pair:
            query += " AND pair = :pair"
            params["pair"] = pair
        
        if strategy:
            query += " AND strategy = :strategy"
            params["strategy"] = strategy
        
        if start:
            query += " AND timestamp >= :start"
            params["start"] = start
        
        if end:
            query += " AND timestamp <= :end"
            params["end"] = end
        
        query += " ORDER BY timestamp"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        return df
    
    # =========================================================================
    # Backtest Results
    # =========================================================================
    
    def save_backtest_result(
        self,
        strategy: str,
        pair: str,
        test_start: str,
        test_end: str,
        total_return: float,
        annual_return: float,
        sharpe_ratio: float,
        sortino_ratio: float,
        max_drawdown: float,
        num_trades: int,
        win_rate: float,
        config: dict = None,
        notes: str = None
    ) -> int:
        """Save backtest results to database."""
        import json
        
        query = """
            INSERT INTO backtest_results 
            (strategy, pair, test_start, test_end, total_return, annual_return,
             sharpe_ratio, sortino_ratio, max_drawdown, num_trades, win_rate, config, notes)
            VALUES 
            (:strategy, :pair, :test_start, :test_end, :total_return, :annual_return,
             :sharpe_ratio, :sortino_ratio, :max_drawdown, :num_trades, :win_rate, :config, :notes)
            RETURNING id
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {
                "strategy": strategy,
                "pair": pair,
                "test_start": test_start,
                "test_end": test_end,
                "total_return": total_return,
                "annual_return": annual_return,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "num_trades": num_trades,
                "win_rate": win_rate,
                "config": json.dumps(config) if config else None,
                "notes": notes
            })
            conn.commit()
            return result.fetchone()[0]
    
    def get_backtest_results(
        self,
        strategy: str = None,
        pair: str = None
    ) -> pd.DataFrame:
        """Get backtest history."""
        query = "SELECT * FROM backtest_results WHERE 1=1"
        params = {}
        
        if strategy:
            query += " AND strategy = :strategy"
            params["strategy"] = strategy
        
        if pair:
            query += " AND pair = :pair"
            params["pair"] = pair
        
        query += " ORDER BY run_timestamp DESC"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        return df
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def execute(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute arbitrary SQL query and return DataFrame."""
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params or {})
        return df
    
    def summary(self) -> pd.DataFrame:
        """Get summary of all data in the database."""
        query = """
            SELECT 
                pair,
                source,
                MIN(timestamp) as start,
                MAX(timestamp) as end,
                COUNT(*) as rows
            FROM ohlcv 
            GROUP BY pair, source
            ORDER BY pair
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df


# Convenience function
def get_database() -> Database:
    """Get a Database instance."""
    return Database()
