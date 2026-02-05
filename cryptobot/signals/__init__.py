# -*- coding: utf-8 -*-
"""
CryptoBot - Signals Module
===========================
16-state regime detection, hit rate calculation, and quality filtering.

Shared engine modules (single source of truth for backtest and live):

    regime.py    — RegimeClassifier: bar-by-bar classification with hysteresis
    features.py  — build_24h_signals(): batch 24h signal generation for training
    hit_rates.py — HitRateCalculator: expanding-window hit rates per state
    quality.py   — QualityFilter: dist_ma168 position sizing + dead zone

Validated Configuration:
    - 16 states (4 binary components: trend_24h, trend_168h, ma72>ma24, ma168>ma24)
    - NO_MA72_ONLY filter (suppresses MA72-only transitions)
    - MA periods: 24h=16, 72h=6, 168h=2
    - Hysteresis: entry=1.5%, exit=0.5% (AND logic)

Usage:
    # Direct submodule imports (preferred):
    from cryptobot.signals.regime import RegimeClassifier, RegimeConfig
    from cryptobot.signals.features import build_24h_signals
    from cryptobot.signals.hit_rates import HitRateCalculator
    from cryptobot.signals.quality import QualityFilter

    # Or via package:
    from cryptobot.signals import RegimeClassifier, build_24h_signals
"""

# =========================================================================
# Regime classification
# =========================================================================
from cryptobot.signals.regime import (
    RegimeClassifier,
    RegimeConfig,
    RegimeState,
)

# =========================================================================
# Batch signal generation (24h resolution, for hit rate training)
# =========================================================================
from cryptobot.signals.features import (
    build_24h_signals,
)

# =========================================================================
# Hit rate calculation
# =========================================================================
from cryptobot.signals.hit_rates import (
    HitRateCalculator,
    HitRateConfig,
)

# =========================================================================
# Quality filter
# =========================================================================
from cryptobot.signals.quality import (
    QualityFilter,
    QualityConfig,
)


__all__ = [
    # Regime
    'RegimeClassifier',
    'RegimeConfig',
    'RegimeState',

    # Features
    'build_24h_signals',

    # Hit rates
    'HitRateCalculator',
    'HitRateConfig',

    # Quality
    'QualityFilter',
    'QualityConfig',
]