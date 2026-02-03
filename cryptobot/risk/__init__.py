# -*- coding: utf-8 -*-
"""
CryptoBot - Risk Management Module
===================================
Position sizing, risk limits, and portfolio protection.

Sizers:
    - RiskParitySizer: Portfolio-level inverse-volatility allocation (RECOMMENDED)
    - KellySizer: Single-asset Kelly criterion sizing
    - AvoidDangerSizer: Binary in/out based on regime danger
    - FixedSizer: Constant position size
    - VolatilityTargetSizer: Single-asset volatility targeting

Risk Controls:
    - PositionLimits: Max position, exposure, and correlation limits
    - RiskManager: Unified risk management (circuit breakers, stops, limits)
    - CircuitBreaker: Portfolio-level loss limits
    - StopLossManager: Position-level stop losses

Usage:
    # For portfolio-level sizing (16-state strategy)
    from cryptobot.risk import RiskParitySizer, RiskParityConfig
    
    config = RiskParityConfig()
    sizer = RiskParitySizer(config)
    positions = sizer.calculate_all_positions(signals, equity, peak_equity, returns_df, date, prices)
    
    # For single-asset sizing
    from cryptobot.risk import KellySizer
    sizer = KellySizer()
    position = sizer.calculate(prediction, features, portfolio, pair)
"""

# =============================================================================
# Base classes
# =============================================================================
from cryptobot.risk.base_sizer import (
    BaseSizer,
    BasePortfolioSizer,
    PositionSizerProtocol,
)

# =============================================================================
# Risk Parity (portfolio-level) - PRIMARY SIZER FOR 16-STATE STRATEGY
# =============================================================================
from cryptobot.risk.risk_parity import (
    RiskParitySizer,
    RiskParityConfig,
    create_risk_parity_sizer,
    create_conservative_risk_parity_sizer,
    create_aggressive_risk_parity_sizer,
)

# =============================================================================
# VaR
# =============================================================================
from .var import calculate_var

# =============================================================================
# Kelly sizers (single-asset)
# =============================================================================
from cryptobot.risk.position import (
    KellySizer,
    SizingConfig,
    FixedSizer,
    VolatilityTargetSizer,
    create_kelly_sizer,
    create_conservative_sizer,
    create_aggressive_sizer,
)

# =============================================================================
# Avoid Danger sizer
# =============================================================================
from cryptobot.risk.drawdown import (
    AvoidDangerSizer,
    AvoidDangerConfig,
    create_v16_sizer,
)

# =============================================================================
# Position limits
# =============================================================================
from cryptobot.risk.limits import (
    PositionLimits,
    LimitResult,
)

# =============================================================================
# Risk manager
# =============================================================================
from cryptobot.risk.manager import (
    RiskManager,
    RiskConfig,
    create_risk_manager,
    create_conservative_risk_manager,
    create_aggressive_risk_manager,
)

# =============================================================================
# Circuit breaker
# =============================================================================
from cryptobot.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerAction,
    CircuitBreakerState,
)

# =============================================================================
# Stops
# =============================================================================
from cryptobot.risk.stops import (
    StopLossManager,
    StopTriggered,
    StopType,
)


__all__ = [
    # Base
    'BaseSizer',
    'BasePortfolioSizer',
    'PositionSizerProtocol',
    
    # Risk Parity (primary for 16-state)
    'RiskParitySizer',
    'RiskParityConfig',
    'create_risk_parity_sizer',
    'create_conservative_risk_parity_sizer',
    'create_aggressive_risk_parity_sizer',
    
    # Kelly (single-asset)
    'KellySizer',
    'SizingConfig',
    'FixedSizer',
    'VolatilityTargetSizer',
    'create_kelly_sizer',
    'create_conservative_sizer',
    'create_aggressive_sizer',
    
    # Avoid Danger
    'AvoidDangerSizer',
    'AvoidDangerConfig',
    'create_v16_sizer',
    
    # Limits
    'PositionLimits',
    'LimitResult',
    
    # Risk Manager
    'RiskManager',
    'RiskConfig',
    'create_risk_manager',
    'create_conservative_risk_manager',
    'create_aggressive_risk_manager',
    
    # Circuit Breaker
    'CircuitBreaker',
    'CircuitBreakerAction',
    'CircuitBreakerState',
    
    # Stops
    'StopLossManager',
    'StopTriggered',
    'StopType',
]