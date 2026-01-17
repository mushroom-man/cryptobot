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
    # Signals
    # =========================================================================
    
    def record_signal(
        self,
        pair: str,
        strategy: str,
        signal: str = None,
        target_position: float = None,
        confidence: float = None,
        regime: int = None,
        prediction: float = None,
        timestamp: datetime = None
    ) -> bool:
        """
        Record a trading signal in the database.
        
        Args:
            pair: Trading pair (e.g., "XBTUSD")
            strategy: Strategy name (e.g., "16state_directional")
            signal: Signal type (e.g., "STRONG_BUY", "BUY", "SELL", "STRONG_SELL")
            target_position: Target position size (1.0=full, 0.5=half, 0.0=none)
            confidence: Confidence score / hit rate (0.0 to 1.0)
            regime: Regime state as integer
            prediction: Model prediction value
            timestamp: Signal timestamp (defaults to NOW())
        
        Returns:
            True if successful
        """
        query = """
            INSERT INTO signals 
            (timestamp, pair, strategy, signal, target_position, confidence, regime, prediction)
            VALUES 
            (COALESCE(:timestamp, NOW()), :pair, :strategy, :signal, :target_position, 
             :confidence, :regime, :prediction)
            ON CONFLICT (timestamp, pair, strategy) 
            DO UPDATE SET
                signal = EXCLUDED.signal,
                target_position = EXCLUDED.target_position,
                confidence = EXCLUDED.confidence,
                regime = EXCLUDED.regime,
                prediction = EXCLUDED.prediction
        """
        with self.engine.connect() as conn:
            conn.execute(text(query), {
                "timestamp": timestamp,
                "pair": pair,
                "strategy": strategy,
                "signal": signal,
                "target_position": target_position,
                "confidence": confidence,
                "regime": regime,
                "prediction": prediction
            })
            conn.commit()
            return True
    
    def get_signals(
        self,
        pair: str = None,
        strategy: str = None,
        start: str = None,
        end: str = None
    ) -> pd.DataFrame:
        """
        Get signal history from the database.
        
        Args:
            pair: Filter by trading pair (optional)
            strategy: Filter by strategy name (optional)
            start: Start date filter (optional)
            end: End date filter (optional)
        
        Returns:
            DataFrame with signal history
        """
        query = "SELECT * FROM signals WHERE 1=1"
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
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
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
    # Portfolio
    # =========================================================================
    
    def record_portfolio_snapshot(
        self,
        pair: str,
        position: float,
        entry_price: float = None,
        current_price: float = None,
        unrealized_pnl: float = None,
        timestamp: datetime = None
    ) -> bool:
        """
        Record a portfolio position snapshot.
        
        Args:
            pair: Trading pair
            position: Position size (positive=long, negative=short, 0=flat)
            entry_price: Average entry price
            current_price: Current market price
            unrealized_pnl: Unrealized profit/loss
            timestamp: Snapshot timestamp (defaults to NOW())
        
        Returns:
            True if successful
        """
        query = """
            INSERT INTO portfolio 
            (timestamp, pair, position, entry_price, current_price, unrealized_pnl)
            VALUES 
            (COALESCE(:timestamp, NOW()), :pair, :position, :entry_price, 
             :current_price, :unrealized_pnl)
            ON CONFLICT (timestamp, pair) 
            DO UPDATE SET
                position = EXCLUDED.position,
                entry_price = EXCLUDED.entry_price,
                current_price = EXCLUDED.current_price,
                unrealized_pnl = EXCLUDED.unrealized_pnl
        """
        with self.engine.connect() as conn:
            conn.execute(text(query), {
                "timestamp": timestamp,
                "pair": pair,
                "position": position,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl
            })
            conn.commit()
            return True
    
    def get_current_positions(self) -> pd.DataFrame:
        """
        Get the most recent position for each pair.
        
        Uses the current_positions view defined in init_db.sql.
        
        Returns:
            DataFrame with current positions
        """
        query = "SELECT * FROM current_positions"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df
    
    def get_position(self, pair: str) -> Optional[dict]:
        """
        Get the most recent position for a specific pair.
        
        Args:
            pair: Trading pair
        
        Returns:
            Dict with position details or None if no position
        """
        query = """
            SELECT * FROM portfolio 
            WHERE pair = :pair 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"pair": pair})
            row = result.fetchone()
            if row:
                return {
                    "timestamp": row[0],
                    "pair": row[1],
                    "position": row[2],
                    "entry_price": row[3],
                    "current_price": row[4],
                    "unrealized_pnl": row[5]
                }
            return None
    
    # =========================================================================
    # Equity
    # =========================================================================
    
    def record_equity(
        self,
        total_equity: float,
        cash: float = None,
        invested: float = None,
        daily_pnl: float = None,
        drawdown: float = None,
        peak_equity: float = None,
        timestamp: datetime = None
    ) -> bool:
        """
        Record an equity snapshot.
        
        Args:
            total_equity: Total account value
            cash: Cash balance
            invested: Value invested in positions
            daily_pnl: Day's profit/loss
            drawdown: Current drawdown from peak
            peak_equity: All-time high equity
            timestamp: Snapshot timestamp (defaults to NOW())
        
        Returns:
            True if successful
        """
        query = """
            INSERT INTO equity 
            (timestamp, total_equity, cash, invested, daily_pnl, drawdown, peak_equity)
            VALUES 
            (COALESCE(:timestamp, NOW()), :total_equity, :cash, :invested, 
             :daily_pnl, :drawdown, :peak_equity)
            ON CONFLICT (timestamp) 
            DO UPDATE SET
                total_equity = EXCLUDED.total_equity,
                cash = EXCLUDED.cash,
                invested = EXCLUDED.invested,
                daily_pnl = EXCLUDED.daily_pnl,
                drawdown = EXCLUDED.drawdown,
                peak_equity = EXCLUDED.peak_equity
        """
        with self.engine.connect() as conn:
            conn.execute(text(query), {
                "timestamp": timestamp,
                "total_equity": total_equity,
                "cash": cash,
                "invested": invested,
                "daily_pnl": daily_pnl,
                "drawdown": drawdown,
                "peak_equity": peak_equity
            })
            conn.commit()
            return True
    
    def get_latest_equity(self) -> Optional[dict]:
        """
        Get the most recent equity snapshot.
        
        Returns:
            Dict with equity details or None
        """
        query = "SELECT * FROM latest_equity"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            row = result.fetchone()
            if row:
                return {
                    "timestamp": row[0],
                    "total_equity": row[1],
                    "cash": row[2],
                    "invested": row[3],
                    "daily_pnl": row[4],
                    "drawdown": row[5],
                    "peak_equity": row[6]
                }
            return None
    
    def get_equity_history(
        self,
        start: str = None,
        end: str = None
    ) -> pd.DataFrame:
        """
        Get equity history.
        
        Args:
            start: Start date filter (optional)
            end: End date filter (optional)
        
        Returns:
            DataFrame with equity history
        """
        query = "SELECT * FROM equity WHERE 1=1"
        params = {}
        
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
