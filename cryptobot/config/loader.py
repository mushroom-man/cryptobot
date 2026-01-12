# -*- coding: utf-8 -*-
"""
CryptoBot - Configuration Management
=====================================
YAML-based configuration loading and management.

Provides:
    - Load/save YAML configs
    - Merge configs (base + overrides)
    - Create typed config objects
    - Validate configurations

Usage:
    from cryptobot.config import load_config, ConfigManager
    
    # Load single config
    config = load_config("configs/strategy/default.yaml")
    
    # Load with overrides
    config = load_config(
        "configs/strategy/default.yaml",
        overrides={"kelly_fraction": 0.30}
    )
    
    # Use ConfigManager for full system config
    manager = ConfigManager("configs/")
    strategy_config = manager.get_strategy_config()
    risk_config = manager.get_risk_config()
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml
import json
from datetime import datetime


# =============================================================================
# Config Loading Utilities
# =============================================================================

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def save_yaml(config: Dict[str, Any], path: Union[str, Path]):
    """Save config to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configs, with override taking precedence.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load config from YAML with optional overrides.
    
    Args:
        path: Path to YAML config file
        overrides: Dictionary of values to override
    
    Returns:
        Merged configuration dictionary
    """
    config = load_yaml(path)
    
    if overrides:
        config = merge_configs(config, overrides)
    
    return config


# =============================================================================
# Typed Config Classes
# =============================================================================

@dataclass
class StrategyConfig:
    """Strategy configuration."""
    
    # Trading
    pair: str = "XBTUSD"
    pairs: List[str] = field(default_factory=lambda: ["XBTUSD"])
    
    # Position sizing
    kelly_fraction: float = 0.25
    max_position: float = 0.30
    min_position: float = 0.01
    p_threshold: float = 0.50
    
    # Regime multipliers
    regime_multipliers: Dict[int, float] = field(default_factory=lambda: {
        0: 0.8, 1: 1.0, 2: 0.5, 3: 0.3
    })
    
    # MA multipliers
    ma_multipliers: Dict[int, float] = field(default_factory=lambda: {
        0: 0.1, 1: 0.4, 2: 0.7, 3: 1.0
    })
    
    # Volatility scaling
    use_vol_scaling: bool = True
    target_vol: float = 0.20
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StrategyConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # Circuit breakers
    daily_loss_limit: float = -0.05
    weekly_loss_limit: float = -0.10
    monthly_loss_limit: float = -0.15
    max_drawdown: float = -0.25
    
    # Stop losses
    atr_multiplier: float = 2.0
    garch_multiplier: float = 2.5
    signal_threshold: float = 0.35
    max_loss_per_trade: float = 0.03
    use_trailing_stops: bool = False
    trailing_stop_pct: float = 0.02
    
    # Position limits
    max_position_pct: float = 0.30
    max_total_exposure: float = 3.0
    max_leverage: float = 3.0
    max_correlated_exposure: float = 0.50
    correlation_threshold: float = 0.80
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RiskConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    
    # Capital
    initial_capital: float = 100_000.0
    
    # Execution
    slippage_bps: float = 10.0
    commission_bps: float = 10.0
    
    # Trading
    min_trade_threshold: float = 0.01
    
    # Data
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BacktestConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureConfig:
    """Feature configuration."""
    
    # Feature list
    features: List[str] = field(default_factory=lambda: [
        'ma_score',
        'regime_hybrid_simple',
        'rolling_vol_168h',
        'garch_vol_simple',
        'price_vs_sma_24',
    ])
    
    # Target
    target_horizon: int = 24
    target_type: str = 'binary'
    
    # MA periods
    ma_periods: List[int] = field(default_factory=lambda: [6, 24, 72])
    
    # Volatility
    vol_lookback: int = 168
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # Model type
    model_type: str = 'random_forest'
    
    # Random Forest params
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    
    # Logistic params
    C: float = 1.0
    
    # XGBoost params
    learning_rate: float = 0.1
    
    # Training
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Config Manager
# =============================================================================

class ConfigManager:
    """
    Centralized configuration management.
    
    Loads configs from a directory structure:
        configs/
        ├── strategy/
        │   ├── default.yaml
        │   └── aggressive.yaml
        ├── risk/
        │   ├── default.yaml
        │   └── conservative.yaml
        ├── backtest/
        │   └── default.yaml
        ├── features/
        │   └── default.yaml
        └── model/
            └── default.yaml
    """
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def _get_config_path(self, category: str, name: str = "default") -> Path:
        """Get path to config file."""
        return self.config_dir / category / f"{name}.yaml"
    
    def _load_cached(self, category: str, name: str = "default") -> Dict[str, Any]:
        """Load config with caching."""
        cache_key = f"{category}/{name}"
        
        if cache_key not in self._cache:
            path = self._get_config_path(category, name)
            if path.exists():
                self._cache[cache_key] = load_yaml(path)
            else:
                self._cache[cache_key] = {}
        
        return self._cache[cache_key]
    
    def get_strategy_config(
        self, 
        name: str = "default",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> StrategyConfig:
        """Get strategy configuration."""
        config = self._load_cached("strategy", name)
        if overrides:
            config = merge_configs(config, overrides)
        return StrategyConfig.from_dict(config)
    
    def get_risk_config(
        self,
        name: str = "default",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> RiskConfig:
        """Get risk configuration."""
        config = self._load_cached("risk", name)
        if overrides:
            config = merge_configs(config, overrides)
        return RiskConfig.from_dict(config)
    
    def get_backtest_config(
        self,
        name: str = "default",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> BacktestConfig:
        """Get backtest configuration."""
        config = self._load_cached("backtest", name)
        if overrides:
            config = merge_configs(config, overrides)
        return BacktestConfig.from_dict(config)
    
    def get_feature_config(
        self,
        name: str = "default",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> FeatureConfig:
        """Get feature configuration."""
        config = self._load_cached("features", name)
        if overrides:
            config = merge_configs(config, overrides)
        return FeatureConfig.from_dict(config)
    
    def get_model_config(
        self,
        name: str = "default",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ModelConfig:
        """Get model configuration."""
        config = self._load_cached("model", name)
        if overrides:
            config = merge_configs(config, overrides)
        return ModelConfig.from_dict(config)
    
    def get_full_config(
        self,
        strategy: str = "default",
        risk: str = "default",
        backtest: str = "default",
        features: str = "default",
        model: str = "default",
    ) -> Dict[str, Any]:
        """Get all configs combined."""
        return {
            'strategy': self.get_strategy_config(strategy).to_dict(),
            'risk': self.get_risk_config(risk).to_dict(),
            'backtest': self.get_backtest_config(backtest).to_dict(),
            'features': self.get_feature_config(features).to_dict(),
            'model': self.get_model_config(model).to_dict(),
        }
    
    def save_config(
        self,
        config: Union[Dict[str, Any], Any],
        category: str,
        name: str,
    ):
        """Save config to file."""
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        
        path = self._get_config_path(category, name)
        save_yaml(config, path)
        
        # Clear cache
        cache_key = f"{category}/{name}"
        self._cache.pop(cache_key, None)
    
    def list_configs(self, category: str) -> List[str]:
        """List available configs in a category."""
        category_dir = self.config_dir / category
        if not category_dir.exists():
            return []
        
        return [p.stem for p in category_dir.glob("*.yaml")]
    
    def clear_cache(self):
        """Clear config cache."""
        self._cache.clear()


# =============================================================================
# Default Config Generation
# =============================================================================

def create_default_configs(config_dir: Union[str, Path] = "configs"):
    """
    Create default configuration files.
    
    Run once to set up initial configs.
    """
    config_dir = Path(config_dir)
    
    # Strategy configs
    strategy_default = StrategyConfig()
    save_yaml(strategy_default.to_dict(), config_dir / "strategy" / "default.yaml")
    
    strategy_aggressive = StrategyConfig(
        kelly_fraction=0.40,
        max_position=0.50,
        regime_multipliers={0: 1.0, 1: 1.0, 2: 0.7, 3: 0.5},
    )
    save_yaml(strategy_aggressive.to_dict(), config_dir / "strategy" / "aggressive.yaml")
    
    strategy_conservative = StrategyConfig(
        kelly_fraction=0.15,
        max_position=0.20,
        regime_multipliers={0: 0.5, 1: 0.8, 2: 0.3, 3: 0.1},
    )
    save_yaml(strategy_conservative.to_dict(), config_dir / "strategy" / "conservative.yaml")
    
    # Risk configs
    risk_default = RiskConfig()
    save_yaml(risk_default.to_dict(), config_dir / "risk" / "default.yaml")
    
    risk_conservative = RiskConfig(
        daily_loss_limit=-0.03,
        weekly_loss_limit=-0.07,
        monthly_loss_limit=-0.10,
        max_drawdown=-0.15,
        max_loss_per_trade=0.02,
    )
    save_yaml(risk_conservative.to_dict(), config_dir / "risk" / "conservative.yaml")
    
    # Backtest configs
    backtest_default = BacktestConfig()
    save_yaml(backtest_default.to_dict(), config_dir / "backtest" / "default.yaml")
    
    backtest_realistic = BacktestConfig(
        slippage_bps=15.0,
        commission_bps=26.0,  # Kraken taker fee
    )
    save_yaml(backtest_realistic.to_dict(), config_dir / "backtest" / "realistic.yaml")
    
    # Feature configs
    feature_default = FeatureConfig()
    save_yaml(feature_default.to_dict(), config_dir / "features" / "default.yaml")
    
    # Model configs
    model_default = ModelConfig()
    save_yaml(model_default.to_dict(), config_dir / "model" / "default.yaml")
    
    model_rf = ModelConfig(
        model_type='random_forest',
        n_estimators=200,
        max_depth=15,
    )
    save_yaml(model_rf.to_dict(), config_dir / "model" / "random_forest.yaml")
    
    model_xgb = ModelConfig(
        model_type='xgboost',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
    )
    save_yaml(model_xgb.to_dict(), config_dir / "model" / "xgboost.yaml")
    
    print(f"Created default configs in {config_dir}/")


def print_config(config: Union[Dict[str, Any], Any], title: str = "Configuration"):
    """Pretty print a configuration."""
    if hasattr(config, 'to_dict'):
        config = config.to_dict()
    
    print(f"\n{'='*50}")
    print(title)
    print('='*50)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
