# -*- coding: utf-8 -*-
"""
CryptoBot - Configuration Module
=================================
YAML-based configuration management.

Config Classes:
    StrategyConfig: Position sizing, Kelly settings
    RiskConfig: Circuit breakers, stops, limits
    BacktestConfig: Capital, execution costs
    FeatureConfig: Feature list, target settings
    ModelConfig: Model type and parameters

Quick Start:
    from cryptobot.config import ConfigManager
    
    # Load configs
    manager = ConfigManager("configs/")
    strategy = manager.get_strategy_config()
    risk = manager.get_risk_config()
    
    # With overrides
    strategy = manager.get_strategy_config(
        name="default",
        overrides={"kelly_fraction": 0.30}
    )

Simple Loading:
    from cryptobot.config import load_config
    
    config = load_config("configs/strategy/default.yaml")

Create Defaults:
    from cryptobot.config import create_default_configs
    
    create_default_configs("configs/")
"""

from cryptobot.config.loader import (
    # Loading utilities
    load_yaml,
    save_yaml,
    load_config,
    merge_configs,
    # Config classes
    StrategyConfig,
    RiskConfig,
    BacktestConfig,
    FeatureConfig,
    ModelConfig,
    # Manager
    ConfigManager,
    # Utilities
    create_default_configs,
    print_config,
)

__all__ = [
    # Loading
    "load_yaml",
    "save_yaml",
    "load_config",
    "merge_configs",
    # Config classes
    "StrategyConfig",
    "RiskConfig",
    "BacktestConfig",
    "FeatureConfig",
    "ModelConfig",
    # Manager
    "ConfigManager",
    # Utilities
    "create_default_configs",
    "print_config",
]
