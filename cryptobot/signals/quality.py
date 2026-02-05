# -*- coding: utf-8 -*-
"""
CryptoBot - Quality Filter
============================
FILE: cryptobot/signals/quality.py

Modulates position size based on distance to MA(168h).
Prevents overextended entries and suppresses micro-rebalances.

Single source of truth — imported by both backtest and live system.

Quality Scalar (matches backtest v03 lines 560–588 exactly):
    Longs:  dist ≤ -0.04 → 1.0 (near/below MA, full conviction)
            dist ≥ +0.05 → 0.5 (overextended, reduce)
            between → linear interpolation
    Shorts: dist ≤ -0.04 → 0.3 (oversold, weak short)
            dist ≥ +0.04 → 1.0 (exhaustion above MA, full short)
            between → linear interpolation

Dead Zone (matches backtest v03 lines 1109–1127):
    Suppresses micro-adjustments when direction is unchanged and
    relative exposure change is below 15%. Direction changes
    (long↔short, long↔flat, etc.) always execute immediately.
    Prevents the 87× trade explosion discovered during hourly
    cycle testing.

Usage:
    from cryptobot.signals.quality import QualityFilter, QualityConfig

    qf = QualityFilter()

    # Compute scalar
    scalar = qf.compute_scalar(dist_ma168=0.02, direction=1)
    # scalar → 0.83 (linearly interpolated)

    # Check dead zone
    should_trade = qf.should_rebalance(
        pair='ETHUSD', curr_exposure=0.15, prev_exposure=0.14)
    # should_trade → False (7% change < 15% threshold)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class QualityConfig:
    """
    Immutable quality filter configuration.

    Validated parameters — matches backtest v03 lines 139–145.
    """
    # Long scalar thresholds
    long_full: float = -0.04     # dist ≤ this → scalar 1.0
    long_min: float = 0.05       # dist ≥ this → scalar = long_floor
    long_floor: float = 0.50     # minimum scalar for longs

    # Short scalar thresholds
    short_weak: float = -0.04    # dist ≤ this → scalar = short_floor
    short_full: float = 0.04     # dist ≥ this → scalar 1.0
    short_floor: float = 0.30    # minimum scalar for shorts

    # Dead zone
    rebalance_threshold: float = 0.15  # 15% relative change required


# =============================================================================
# QUALITY FILTER
# =============================================================================

class QualityFilter:
    """
    Quality filter: dist_ma168 scalar + rebalance dead zone.

    The scalar modulates position size based on how far price has
    moved from MA(168h). The dead zone suppresses noisy
    micro-adjustments on the hourly cycle.

    Args:
        config: QualityConfig with thresholds.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()

    # -----------------------------------------------------------------
    # Scalar calculation (matches backtest v03 lines 560–588)
    # -----------------------------------------------------------------

    def compute_scalar(self, dist_ma168: float, direction: int) -> float:
        """
        Map distance-to-MA(168h) to a position sizing scalar.

        Args:
            dist_ma168: (price - MA168) / MA168. Positive = above MA.
            direction:  +1 for long, -1 for short, 0 for flat.

        Returns:
            Scalar between floor and 1.0. Returns 1.0 for NaN or flat.
        """
        if dist_ma168 is None or (isinstance(dist_ma168, float)
                                  and math.isnan(dist_ma168)):
            return 1.0

        if direction > 0:
            return self._scalar_long(dist_ma168)
        elif direction < 0:
            return self._scalar_short(dist_ma168)
        else:
            return 1.0

    def compute_dist_ma168(self, price: float, ma_168h: float) -> float:
        """
        Calculate distance to MA(168h).

        Args:
            price:   Current close price.
            ma_168h: 168-hour moving average value.

        Returns:
            (price - ma_168h) / ma_168h, or NaN if ma_168h is invalid.
        """
        if ma_168h is None or ma_168h == 0:
            return float('nan')
        return (price - ma_168h) / ma_168h

    # -----------------------------------------------------------------
    # Dead zone (matches backtest v03 lines 1109–1127)
    # -----------------------------------------------------------------

    def should_rebalance(
        self,
        curr_exposure: float,
        prev_exposure: float,
    ) -> bool:
        """
        Check if a rebalance should execute or be suppressed.

        Direction changes always execute. Same-direction adjustments
        below the threshold are suppressed.

        Args:
            curr_exposure: Target exposure (signed: positive=long, negative=short).
            prev_exposure: Current exposure (signed).

        Returns:
            True if the trade should execute, False if suppressed.
        """
        # Direction change → always execute
        direction_changed = (
            (curr_exposure > 0) != (prev_exposure > 0)
            or (curr_exposure < 0) != (prev_exposure < 0)
            or (curr_exposure == 0) != (prev_exposure == 0)
        )

        if direction_changed:
            return True

        # Same direction — check relative change
        if prev_exposure == 0:
            return True  # entering from flat

        relative_change = abs(curr_exposure - prev_exposure) / abs(prev_exposure)

        if relative_change < self.config.rebalance_threshold:
            return False

        return True

    # -----------------------------------------------------------------
    # Internal: linear interpolation
    # -----------------------------------------------------------------

    def _scalar_long(self, dist: float) -> float:
        """Long scalar: full at/below long_full, floor at/above long_min."""
        cfg = self.config

        if dist <= cfg.long_full:
            return 1.0
        elif dist >= cfg.long_min:
            return cfg.long_floor
        else:
            frac = (dist - cfg.long_full) / (cfg.long_min - cfg.long_full)
            return 1.0 - frac * (1.0 - cfg.long_floor)

    def _scalar_short(self, dist: float) -> float:
        """Short scalar: floor at/below short_weak, full at/above short_full."""
        cfg = self.config

        if dist <= cfg.short_weak:
            return cfg.short_floor
        elif dist >= cfg.short_full:
            return 1.0
        else:
            frac = (dist - cfg.short_weak) / (cfg.short_full - cfg.short_weak)
            return cfg.short_floor + frac * (1.0 - cfg.short_floor)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"QualityFilter("
                f"long=[{self.config.long_full}, {self.config.long_min}]→"
                f"{self.config.long_floor}, "
                f"short=[{self.config.short_weak}, {self.config.short_full}]→"
                f"{self.config.short_floor}, "
                f"dead_zone={self.config.rebalance_threshold:.0%})")