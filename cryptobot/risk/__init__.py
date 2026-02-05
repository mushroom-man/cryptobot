# -*- coding: utf-8 -*-
"""CryptoBot - Risk Management Module"""

from cryptobot.risk.risk_parity import RiskParitySizer, RiskParityConfig
from cryptobot.risk.var import calculate_var
from cryptobot.risk.circuit_breaker import CircuitBreaker, CircuitBreakerAction, CircuitBreakerState
from cryptobot.risk.drawdown import AvoidDangerSizer, AvoidDangerConfig
