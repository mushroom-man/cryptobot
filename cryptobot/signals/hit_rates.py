# -*- coding: utf-8 -*-
"""
CryptoBot - Hit Rate Calculator
=================================
FILE: cryptobot/signals/hit_rates.py

Expanding-window hit rate calculation per 16-state regime.
Computes the historical probability that a directional trade is
profitable for each state, then maps to a position signal.

Single source of truth — imported by both backtest and live system.

Hit Rate Logic (matches backtest v03 lines 492–547 exactly):
    - Uses 24h bars only (statistical validity, not 1h)
    - Forward return: next-bar 24h return (shift -1)
    - Hit rate: count(return > 0) / count(all)
    - Default for empty/rare states: 0.5 (agnostic)
    - Minimum samples: 20 per state for 'sufficient' flag

Signal Mapping:
    hit_rate > 0.50 + sufficient  →  +1.0 (long)
    hit_rate ≤ 0.50 + sufficient  →  -0.5 (short, scaled by short_size_scalar)
    insufficient samples          →   0.0 (flat)

Caching:
    Hit rates are cached per pair and recalculated monthly.
    The expanding window is cut at the current month boundary.

Usage:
    from cryptobot.signals.hit_rates import HitRateCalculator, HitRateConfig

    calc = HitRateCalculator()

    # Update cache (recalculates if month changed)
    calc.update('ETHUSD', signals_24h, returns_24h, current_date)

    # Get direction signal for a state
    signal = calc.get_signal('ETHUSD', state.state_key)
    # signal → +1.0, -0.5, or 0.0
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Type alias for state key: ((trend_24h, trend_168h), (ma72, ma168))
StateKey = Tuple[Tuple[int, int], Tuple[int, int]]

# All 16 possible state keys
ALL_TREND_PERMS = list(product([0, 1], repeat=2))
ALL_MA_PERMS = list(product([0, 1], repeat=2))
ALL_STATE_KEYS = [(t, m) for t in ALL_TREND_PERMS for m in ALL_MA_PERMS]

# Default entry for states with no data
_DEFAULT_ENTRY = {'n': 0, 'hit_rate': 0.5, 'sufficient': False}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class HitRateConfig:
    """
    Immutable hit rate configuration.

    Validated parameters — matches backtest v03.
    """
    hit_rate_threshold: float = 0.50
    min_samples: int = 20            # MIN_SAMPLES_PER_STATE
    min_training_months: int = 12    # minimum history before trading
    short_size_scalar: float = 0.50  # shorts at 50% of long sizing


# =============================================================================
# CALCULATOR
# =============================================================================

class HitRateCalculator:
    """
    Expanding-window hit rate calculator with monthly caching.

    Maintains per-pair hit rate tables and recalculates when the
    calendar month changes. All calculations use 24h bar data.

    Args:
        config: HitRateConfig with thresholds and scalars.
    """

    def __init__(self, config: Optional[HitRateConfig] = None):
        self.config = config or HitRateConfig()
        self._cache: Dict[str, Dict[StateKey, dict]] = {}
        self._last_recalc: Dict[str, Tuple[int, int]] = {}  # pair → (year, month)

    # -----------------------------------------------------------------
    # Cache management
    # -----------------------------------------------------------------

    def update(
        self,
        pair: str,
        signals_24h: pd.DataFrame,
        returns_24h: pd.Series,
        current_date: pd.Timestamp,
    ) -> bool:
        """
        Update hit rate cache for a pair. Recalculates if the month changed.

        Args:
            pair:         Trading pair identifier.
            signals_24h:  DataFrame with columns: trend_24h, trend_168h,
                          ma72_above_ma24, ma168_above_ma24 (24h index).
            returns_24h:  Series of 24h close-to-close returns (same index).
            current_date: Current timestamp for month check.

        Returns:
            True if recalculation was performed, False if cache was used.
        """
        current_month = (current_date.year, current_date.month)

        if self._last_recalc.get(pair) == current_month:
            return False

        # Cut expanding window at start of current month
        cutoff = current_date.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0)
        hist_signals = signals_24h[signals_24h.index < cutoff]
        hist_returns = returns_24h[returns_24h.index < cutoff]

        self._cache[pair] = self._calculate_hit_rates(
            hist_returns, hist_signals)
        self._last_recalc[pair] = current_month

        # Log summary
        sufficient = sum(
            1 for v in self._cache[pair].values() if v['sufficient'])
        logger.info(
            f"{pair}: hit rates recalculated — "
            f"{sufficient}/16 states sufficient, "
            f"cutoff={cutoff.strftime('%Y-%m-%d')}")

        return True

    def force_recalc(
        self,
        pair: str,
        signals_24h: pd.DataFrame,
        returns_24h: pd.Series,
        current_date: pd.Timestamp,
    ) -> None:
        """
        Force recalculation regardless of month. Useful for initialisation.

        Args:
            Same as update().
        """
        cutoff = current_date.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0)
        hist_signals = signals_24h[signals_24h.index < cutoff]
        hist_returns = returns_24h[returns_24h.index < cutoff]

        self._cache[pair] = self._calculate_hit_rates(
            hist_returns, hist_signals)
        self._last_recalc[pair] = (current_date.year, current_date.month)

    # -----------------------------------------------------------------
    # Lookups
    # -----------------------------------------------------------------

    def get_hit_rate(
        self, pair: str, state_key: StateKey
    ) -> Dict[str, Any]:
        """
        Get hit rate data for a specific state.

        Args:
            pair:      Trading pair identifier.
            state_key: ((trend_24h, trend_168h), (ma72, ma168)).

        Returns:
            Dict with 'n', 'hit_rate', 'sufficient'.
            Returns default (0.5, insufficient) if pair not cached.
        """
        pair_rates = self._cache.get(pair)
        if pair_rates is None:
            logger.warning(f"{pair}: no cached hit rates — returning default")
            return _DEFAULT_ENTRY.copy()
        return pair_rates.get(state_key, _DEFAULT_ENTRY.copy())

    def get_signal(
        self,
        pair: str,
        state_key: StateKey,
        allow_shorts: bool = True,
    ) -> float:
        """
        Get position signal for a state.

        Combines hit rate lookup with direction mapping and short scaling.

        Args:
            pair:         Trading pair identifier.
            state_key:    ((trend_24h, trend_168h), (ma72, ma168)).
            allow_shorts: If False, bearish states return 0.0 instead of -0.5.

        Returns:
            +1.0 (long), -short_size_scalar (short), or 0.0 (flat).
        """
        data = self.get_hit_rate(pair, state_key)

        if not data['sufficient']:
            return 0.0

        if data['hit_rate'] > self.config.hit_rate_threshold:
            return 1.0
        else:
            if allow_shorts:
                return -self.config.short_size_scalar
            else:
                return 0.0

    def get_all_hit_rates(
        self, pair: str
    ) -> Dict[StateKey, Dict[str, Any]]:
        """
        Get the full 16-state hit rate table for a pair.

        Returns:
            Dict mapping state_key → {n, hit_rate, sufficient}.
            Empty dict if pair not cached.
        """
        return self._cache.get(pair, {}).copy()

    def has_minimum_training(
        self,
        current_date: pd.Timestamp,
        data_start: pd.Timestamp,
    ) -> bool:
        """
        Check if enough historical data exists for trading.

        Args:
            current_date: Current timestamp.
            data_start:   Earliest available data timestamp.

        Returns:
            True if data spans at least min_training_months.
        """
        months = ((current_date.year - data_start.year) * 12
                  + (current_date.month - data_start.month))
        return months >= self.config.min_training_months

    def is_cached(self, pair: str) -> bool:
        """Check if a pair has cached hit rates."""
        return pair in self._cache

    # -----------------------------------------------------------------
    # Core calculation (matches backtest v03 lines 492–527 exactly)
    # -----------------------------------------------------------------

    @staticmethod
    def _calculate_hit_rates(
        returns_history: pd.Series,
        signals_history: pd.DataFrame,
        min_samples: int = None,
    ) -> Dict[StateKey, Dict[str, Any]]:
        """
        Calculate expanding-window hit rates for all 16 states.

        Logic matches backtest v03 exactly:
            1. Intersect signal and return indices
            2. Forward returns = shift(-1) of aligned returns
            3. Per state: hit_rate = count(return > 0) / n

        Args:
            returns_history: 24h close-to-close returns.
            signals_history: 24h signal DataFrame with trend/MA columns.
            min_samples:     Override minimum samples (default: 20).

        Returns:
            Dict mapping state_key → {n, hit_rate, sufficient}.
        """
        if min_samples is None:
            min_samples = 20  # matches MIN_SAMPLES_PER_STATE

        # Early exit: insufficient total data
        if len(returns_history) < min_samples:
            return {key: _DEFAULT_ENTRY.copy() for key in ALL_STATE_KEYS}

        common_idx = returns_history.index.intersection(signals_history.index)
        if len(common_idx) < min_samples:
            return {key: _DEFAULT_ENTRY.copy() for key in ALL_STATE_KEYS}

        aligned_returns = returns_history.loc[common_idx]
        aligned_signals = signals_history.loc[common_idx]

        # Forward return: the return AFTER being in this state
        forward_returns = aligned_returns.shift(-1).iloc[:-1]
        aligned_signals = aligned_signals.iloc[:-1]

        hit_rates = {}
        for trend_perm in ALL_TREND_PERMS:
            for ma_perm in ALL_MA_PERMS:
                mask = (
                    (aligned_signals['trend_24h'] == trend_perm[0])
                    & (aligned_signals['trend_168h'] == trend_perm[1])
                    & (aligned_signals['ma72_above_ma24'] == ma_perm[0])
                    & (aligned_signals['ma168_above_ma24'] == ma_perm[1])
                )
                perm_returns = forward_returns[mask].dropna()
                n = len(perm_returns)
                hit_rate = (
                    (perm_returns > 0).sum() / n if n > 0 else 0.5)

                hit_rates[(trend_perm, ma_perm)] = {
                    'n': n,
                    'hit_rate': hit_rate,
                    'sufficient': n >= min_samples,
                }

        return hit_rates

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def reset(self, pair: str = None) -> None:
        """Clear cached hit rates for one pair or all pairs."""
        if pair:
            self._cache.pop(pair, None)
            self._last_recalc.pop(pair, None)
        else:
            self._cache.clear()
            self._last_recalc.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics for debugging."""
        stats = {}
        for pair, rates in self._cache.items():
            sufficient = sum(1 for v in rates.values() if v['sufficient'])
            total_n = sum(v['n'] for v in rates.values())
            stats[pair] = {
                'sufficient_states': sufficient,
                'total_samples': total_n,
                'last_recalc': self._last_recalc.get(pair),
            }
        return stats

    def __repr__(self) -> str:
        return (f"HitRateCalculator(pairs={len(self._cache)}, "
                f"threshold={self.config.hit_rate_threshold}, "
                f"min_samples={self.config.min_samples})")