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


def _to_python(val):
    """Convert numpy types to native Python types for database compatibility."""
    if val is None:
        return None
    if hasattr(val, 'item'):  # numpy scalars have .item() method
        return val.item()
    return val


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
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
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
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
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
        notes: str = None,
        trade_type: str = None 
    ) -> int:
        """
        Record a trade in the database.
        
        Returns:
            Trade ID
        """
        query = """
            INSERT INTO trades 
            (timestamp, pair, strategy, direction, size, price, slippage_bps, 
             transaction_cost, execution_type, order_id, notes, trade_type)
            VALUES 
            (NOW(), :pair, :strategy, :direction, :size, :price, :slippage_bps,
             :transaction_cost, :execution_type, :order_id, :notes, :trade_type)
            RETURNING id
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {
                "pair": pair,
                "strategy": strategy,
                "direction": direction,
                "size": _to_python(size),
                "price": _to_python(price),
                "slippage_bps": _to_python(slippage_bps),
                "transaction_cost": _to_python(transaction_cost),
                "execution_type": execution_type,
                "order_id": order_id,
                "notes": notes,
                "trade_type": trade_type
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
                "target_position": _to_python(target_position),
                "confidence": _to_python(confidence),
                "regime": _to_python(regime),
                "prediction": _to_python(prediction)
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
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            df = df.set_index('timestamp')
        
        return df
    
    def get_latest_signal(self, pair: str, strategy: str = None) -> Optional[dict]:
        """
        Get the most recent signal for a specific pair.
        
        Args:
            pair: Trading pair
            strategy: Strategy name filter (optional)
        
        Returns:
            Dict with signal details or None if no signal found
        """
        query = """
            SELECT * FROM signals 
            WHERE pair = :pair
        """
        params = {"pair": pair}
        
        if strategy:
            query += " AND strategy = :strategy"
            params["strategy"] = strategy
        
        query += " ORDER BY timestamp DESC LIMIT 1"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            row = result.fetchone()
            if row:
                return {
                    "timestamp": row[0],
                    "pair": row[1],
                    "strategy": row[2],
                    "signal": row[3],
                    "target_position": row[4],
                    "confidence": row[5],
                    "regime": row[6],
                    "prediction": row[7]
                }
            return None
    
    # =============================================================================
    # Weights
    # =============================================================================
    # Add this section to database.py after the Signals section (around line 450)
    # and before the Portfolio section.

    def record_weight(
        self,
        pair: str,
        weight: float,
        strategy: str = '16state',
        timestamp: datetime = None
    ) -> bool:
        """
        Record a risk parity weight for a pair.
        
        Args:
            pair: Trading pair (e.g., "XMRUSD")
            weight: Risk parity weight (0.0 to 1.0)
            strategy: Strategy name (default: '16state')
            timestamp: Weight calculation timestamp (defaults to NOW())
        
        Returns:
            True if successful
        """
        query = """
            INSERT INTO weights 
            (timestamp, pair, weight, strategy)
            VALUES 
            (COALESCE(:timestamp, NOW()), :pair, :weight, :strategy)
            ON CONFLICT (timestamp, pair) 
            DO UPDATE SET
                weight = EXCLUDED.weight,
                strategy = EXCLUDED.strategy
        """
        with self.engine.connect() as conn:
            conn.execute(text(query), {
                "timestamp": timestamp,
                "pair": pair,
                "weight": _to_python(weight),
                "strategy": strategy
            })
            conn.commit()
            return True

    def record_weights(
        self,
        weights: dict,
        strategy: str = '16state',
        timestamp: datetime = None
    ) -> int:
        """
        Record multiple risk parity weights at once.
        
        Args:
            weights: Dict of {pair: weight} (e.g., {"XMRUSD": 0.15, "ETHUSD": 0.25})
            strategy: Strategy name (default: '16state')
            timestamp: Weight calculation timestamp (defaults to NOW())
        
        Returns:
            Number of weights recorded
        """
        count = 0
        for pair, weight in weights.items():
            if self.record_weight(pair, weight, strategy, timestamp):
                count += 1
        return count

    def get_weights(
        self,
        pair: str = None,
        strategy: str = None,
        start: str = None,
        end: str = None
    ) -> pd.DataFrame:
        """
        Get weight history from the database.
        
        Args:
            pair: Filter by trading pair (optional)
            strategy: Filter by strategy name (optional)
            start: Start date filter (optional)
            end: End date filter (optional)
        
        Returns:
            DataFrame with weight history
        """
        query = "SELECT * FROM weights WHERE 1=1"
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
        
        query += " ORDER BY timestamp DESC"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        
        return df

    def get_current_weights(self) -> pd.DataFrame:
        """
        Get the most recent weight for each pair.
        
        Uses the current_weights view.
        
        Returns:
            DataFrame with current weights per pair
        """
        query = "SELECT * FROM current_weights"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df

    def get_weights_at(self, timestamp: datetime) -> dict:
        """
        Get weights that were active at a specific timestamp.
        
        Finds the most recent weights recorded before or at the given time.
        
        Args:
            timestamp: Point in time to query
        
        Returns:
            Dict of {pair: weight}
        """
        query = """
            SELECT DISTINCT ON (pair) pair, weight
            FROM weights
            WHERE timestamp <= :timestamp
            ORDER BY pair, timestamp DESC
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"timestamp": timestamp})
            return {row[0]: row[1] for row in result.fetchall()}
        
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
                "total_return": _to_python(total_return),
                "annual_return": _to_python(annual_return),
                "sharpe_ratio": _to_python(sharpe_ratio),
                "sortino_ratio": _to_python(sortino_ratio),
                "max_drawdown": _to_python(max_drawdown),
                "num_trades": _to_python(num_trades),
                "win_rate": _to_python(win_rate),
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
                "position": _to_python(position),
                "entry_price": _to_python(entry_price),
                "current_price": _to_python(current_price),
                "unrealized_pnl": _to_python(unrealized_pnl)
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
                "total_equity": _to_python(total_equity),
                "cash": _to_python(cash),
                "invested": _to_python(invested),
                "daily_pnl": _to_python(daily_pnl),
                "drawdown": _to_python(drawdown),
                "peak_equity": _to_python(peak_equity)
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
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            df = df.set_index('timestamp')
        
        return df
    
    
    # =========================================================================
    # Strategy State
    # =========================================================================
    
    def get_strategy_state(self, pair: str, strategy: str = 'momentum') -> Optional[dict]:
        """
        Get persisted strategy state for a pair.
        
        Args:
            pair: Trading pair
            strategy: Strategy name (default: 'momentum')
        
        Returns:
            Dict with state fields, or None if not found
        """
        query = """
            SELECT pair, strategy, confirmed_state, duration_hours,
                   pending_state, pending_hours, trend_24h, trend_168h, updated_at
            FROM strategy_state
            WHERE pair = :pair AND strategy = :strategy
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"pair": pair, "strategy": strategy})
            row = result.fetchone()
            if row:
                return {
                    'pair': row[0],
                    'strategy': row[1],
                    'confirmed_state': row[2],
                    'duration_hours': row[3],
                    'pending_state': row[4],
                    'pending_hours': row[5],
                    'trend_24h': row[6],
                    'trend_168h': row[7],
                    'updated_at': row[8],
                }
        return None
    
    def save_strategy_state(
        self,
        pair: str,
        confirmed_state: int,
        duration_hours: int,
        pending_state: Optional[int],
        pending_hours: int,
        trend_24h: int,
        trend_168h: int,
        strategy: str = 'momentum'
    ) -> bool:
        """
        Save strategy state for a pair (upsert).
        
        Args:
            pair: Trading pair
            confirmed_state: Current confirmed state (0-15)
            duration_hours: Hours in current state
            pending_state: State awaiting confirmation (or None)
            pending_hours: Hours pending state has persisted
            trend_24h: Hysteresis trend state for 24h MA
            trend_168h: Hysteresis trend state for 168h MA
            strategy: Strategy name (default: 'momentum')
        
        Returns:
            True if successful
        """
        query = """
            INSERT INTO strategy_state 
                (pair, strategy, confirmed_state, duration_hours, 
                 pending_state, pending_hours, trend_24h, trend_168h, updated_at)
            VALUES 
                (:pair, :strategy, :confirmed_state, :duration_hours,
                 :pending_state, :pending_hours, :trend_24h, :trend_168h, NOW())
            ON CONFLICT (pair) DO UPDATE SET
                strategy = EXCLUDED.strategy,
                confirmed_state = EXCLUDED.confirmed_state,
                duration_hours = EXCLUDED.duration_hours,
                pending_state = EXCLUDED.pending_state,
                pending_hours = EXCLUDED.pending_hours,
                trend_24h = EXCLUDED.trend_24h,
                trend_168h = EXCLUDED.trend_168h,
                updated_at = NOW()
        """
        with self.engine.connect() as conn:
            conn.execute(text(query), {
                "pair": pair,
                "strategy": strategy,
                "confirmed_state": _to_python(confirmed_state),
                "duration_hours": _to_python(duration_hours),
                "pending_state": _to_python(pending_state),
                "pending_hours": _to_python(pending_hours),
                "trend_24h": _to_python(trend_24h),
                "trend_168h": _to_python(trend_168h),
            })
            conn.commit()
            return True
    
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