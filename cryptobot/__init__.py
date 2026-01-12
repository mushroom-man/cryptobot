# -*- coding: utf-8 -*-
"""
CryptoBot - Cryptocurrency Trading System
==========================================

A systematic trading system combining quantitative finance 
with machine learning for cryptocurrency markets.

Modules:
    shared: Core components (used by both backtest and production)
        - core: Bar, Order, Fill, Portfolio, TradingEngine
        - risk: CircuitBreaker, StopLoss, PositionLimits, RiskManager
        - sizing: KellySizer, FixedSizer
    
    research: Backtesting and model development
        - backtest: BacktestRunner, SimulatedExecutor
        - training: LogisticModel, RandomForestModel, XGBoostModel
        - analysis: Feature analysis and selection
    
    config: Configuration management
        - ConfigManager, load_config
    
    datasources: Data loading
        - database, csv_loader, kraken_api
    
    features: Feature engineering
        - FeatureEngine

Quick Start:
    from cryptobot.research import run_backtest, RandomForestModel, create_target
    from cryptobot.shared import Portfolio, RiskManager, KellySizer
    from cryptobot.config import ConfigManager
    
    # Load config
    manager = ConfigManager("configs/")
    strategy = manager.get_strategy_config()
    
    # Train model
    df['target'] = create_target(df, horizon=24)
    model = RandomForestModel()
    model.fit(X_train, y_train)
    
    # Backtest
    predictions = model.predict_proba(df)
    results = run_backtest(df, predictions)
    results.print_summary()

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                        RESEARCH                             │
    │   (Backtesting, Training, Analysis)                         │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                         SHARED                              │
    │   (Core, Risk, Sizing - identical in backtest & production) │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                       PRODUCTION                            │
    │   (Live trading - future)                                   │
    └─────────────────────────────────────────────────────────────┘
"""

__version__ = "0.1.0"
__author__ = "John"

# Version info
VERSION = __version__
