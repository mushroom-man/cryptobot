# -*- coding: utf-8 -*-
"""
CryptoBot - Regime Classification Engine
==========================================
FILE: cryptobot/signals/regime.py

Pure 16-state regime classifier. Takes price + MAs, applies hysteresis,
returns state integer and binary components. No signal logic, no position
sizing, no hit rates — just classification.

Hysteresis Logic (LOCKED — matches backtest v03 exactly):
    AND-logic with 1.5% entry / 0.5% exit buffers.
    Exit bullish:  price < ma*(1-exit)  AND price < ma*(1-entry)  → bearish
    Enter bullish: price > ma*(1+exit)  AND price > ma*(1+entry)  → bullish
    Effective: symmetric ±1.5% (entry buffer always dominates)

State Construction:
    Bit 3 (×8):  trend_24h        — price vs MA(24h) with hysteresis
    Bit 2 (×4):  trend_168h       — price vs MA(168h) with hysteresis
    Bit 1 (×2):  ma72_above_ma24  — MA(72h) > MA(24h), no hysteresis
    Bit 0 (×1):  ma168_above_ma24 — MA(168h) > MA(24h), no hysteresis

Modes:
    classify_bar()    — streaming: one bar at a time, maintains state per pair
    classify_series() — vectorized: full price series, leaves state at final bar

Usage:
    from cryptobot.signals.regime import RegimeClassifier, RegimeConfig

    classifier = RegimeClassifier()

    # Streaming (live):
    state = classifier.classify_bar('ETHUSD', price, ma_24h, ma_72h, ma_168h)
    # state.state_int → 15, state.components → (1, 1, 1, 1)

    # Vectorized (backtest):
    states = classifier.classify_series(prices, ma_24h, ma_72h, ma_168h)
    # list of RegimeState objects
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RegimeConfig:
    """
    Immutable regime classification configuration.

    Validated parameters — matches backtest v03 Config E.
    Single source of truth for hysteresis buffers and MA periods.

    Future: add entry_buffer_short / exit_buffer_short for
    per-direction hysteresis experiments.
    """
    entry_buffer: float = 0.015   # 1.5%
    exit_buffer: float = 0.005    # 0.5%

    # Moving average lookback periods (in bars of their native timeframe)
    ma_period_24h: int = 16       # 16-bar MA on 24h bars
    ma_period_72h: int = 6        # 6-bar MA on 72h bars
    ma_period_168h: int = 2       # 2-bar MA on 168h bars


# =============================================================================
# RETURN TYPE
# =============================================================================

@dataclass(frozen=True)
class RegimeState:
    """
    Immutable regime classification result.

    Attributes:
        state_int:         Integer 0–15 encoding all four binary components.
        trend_24h:         1 = bullish (price above MA24h), 0 = bearish.
        trend_168h:        1 = bullish (price above MA168h), 0 = bearish.
        ma72_above_ma24:   1 = MA72h > MA24h, 0 = MA72h <= MA24h.
        ma168_above_ma24:  1 = MA168h > MA24h, 0 = MA168h <= MA24h.
    """
    state_int: int
    trend_24h: int
    trend_168h: int
    ma72_above_ma24: int
    ma168_above_ma24: int

    @property
    def components(self) -> Tuple[int, int, int, int]:
        """Component tuple for hit rate lookups: (t24, t168, ma72, ma168)."""
        return (self.trend_24h, self.trend_168h,
                self.ma72_above_ma24, self.ma168_above_ma24)

    @property
    def trend_perm(self) -> Tuple[int, int]:
        """Trend permutation for hit rate key: (trend_24h, trend_168h)."""
        return (self.trend_24h, self.trend_168h)

    @property
    def ma_perm(self) -> Tuple[int, int]:
        """MA permutation for hit rate key: (ma72_above_ma24, ma168_above_ma24)."""
        return (self.ma72_above_ma24, self.ma168_above_ma24)

    @property
    def state_key(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Full key for hit rate lookups: (trend_perm, ma_perm)."""
        return (self.trend_perm, self.ma_perm)

    def __repr__(self) -> str:
        return (f"RegimeState(S{self.state_int}, "
                f"t24={self.trend_24h}, t168={self.trend_168h}, "
                f"ma72={self.ma72_above_ma24}, ma168={self.ma168_above_ma24})")


# =============================================================================
# INTERNAL: per-pair hysteresis state
# =============================================================================

@dataclass
class _PairHysteresis:
    """Mutable hysteresis state for a single pair."""
    trend_24h: int = 1   # default bullish (matches backtest initialisation)
    trend_168h: int = 1


# =============================================================================
# CLASSIFIER
# =============================================================================

class RegimeClassifier:
    """
    16-state regime classifier with hysteresis.

    Maintains per-pair hysteresis state for the two trend components.
    The MA comparison components are stateless.

    Args:
        config: RegimeConfig with buffer parameters.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._states: Dict[str, _PairHysteresis] = {}

    # -----------------------------------------------------------------
    # Streaming interface (live trading)
    # -----------------------------------------------------------------

    def classify_bar(
        self,
        pair: str,
        price: float,
        ma_24h: float,
        ma_72h: float,
        ma_168h: float,
    ) -> RegimeState:
        """
        Classify a single bar for one pair.

        Updates internal hysteresis state. Call once per hourly bar.

        Args:
            pair:    Trading pair identifier.
            price:   Current close price.
            ma_24h:  24-hour moving average value.
            ma_72h:  72-hour moving average value.
            ma_168h: 168-hour moving average value.

        Returns:
            RegimeState with state integer and all four components.
        """
        hyst = self._get_or_create(pair)

        # Update hysteresis trend components
        hyst.trend_24h = self._apply_hysteresis(
            price, ma_24h, hyst.trend_24h)
        hyst.trend_168h = self._apply_hysteresis(
            price, ma_168h, hyst.trend_168h)

        # Stateless MA comparisons
        ma72_above = 1 if ma_72h > ma_24h else 0
        ma168_above = 1 if ma_168h > ma_24h else 0

        # Build state
        state_int = (hyst.trend_24h * 8
                     + hyst.trend_168h * 4
                     + ma72_above * 2
                     + ma168_above * 1)

        return RegimeState(
            state_int=state_int,
            trend_24h=hyst.trend_24h,
            trend_168h=hyst.trend_168h,
            ma72_above_ma24=ma72_above,
            ma168_above_ma24=ma168_above,
        )

    # -----------------------------------------------------------------
    # Vectorized interface (backtesting)
    # -----------------------------------------------------------------

    def classify_series(
        self,
        pair: str,
        prices: list,
        ma_24h: list,
        ma_72h: list,
        ma_168h: list,
    ) -> List[RegimeState]:
        """
        Classify a full price series for one pair.

        Iterates through bars in order, applying the same hysteresis
        logic as classify_bar(). On completion, internal state for
        this pair reflects the final bar — ready for streaming
        continuation.

        Args:
            pair:    Trading pair identifier.
            prices:  Sequence of close prices.
            ma_24h:  Sequence of 24h MA values (same length as prices).
            ma_72h:  Sequence of 72h MA values (same length as prices).
            ma_168h: Sequence of 168h MA values (same length as prices).

        Returns:
            List of RegimeState, one per bar.
        """
        n = len(prices)
        if not (len(ma_24h) == len(ma_72h) == len(ma_168h) == n):
            raise ValueError(
                f"All input sequences must have equal length. "
                f"Got prices={n}, ma_24h={len(ma_24h)}, "
                f"ma_72h={len(ma_72h)}, ma_168h={len(ma_168h)}")

        results = []
        for i in range(n):
            state = self.classify_bar(
                pair, prices[i], ma_24h[i], ma_72h[i], ma_168h[i])
            results.append(state)

        return results

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def get_state(self, pair: str) -> Optional[RegimeState]:
        """
        Get the last classified state for a pair.

        Returns None if the pair has never been classified. Note: this
        returns only the hysteresis components — the MA comparisons
        are not stored because they are stateless.
        """
        hyst = self._states.get(pair)
        if hyst is None:
            return None
        # Cannot reconstruct full RegimeState without current MA values,
        # so return partial info via trend state only.
        logger.debug(
            f"{pair}: stored hysteresis — "
            f"trend_24h={hyst.trend_24h}, trend_168h={hyst.trend_168h}")
        return None  # caller should use classify_bar for full state

    def get_hysteresis(self, pair: str) -> Optional[Tuple[int, int]]:
        """
        Get raw hysteresis values for a pair.

        Returns:
            (trend_24h, trend_168h) or None if pair not initialised.
        """
        hyst = self._states.get(pair)
        if hyst is None:
            return None
        return (hyst.trend_24h, hyst.trend_168h)

    def get_all_states(self) -> Dict[str, Tuple[int, int]]:
        """Get hysteresis state for all pairs: {pair: (trend_24h, trend_168h)}."""
        return {pair: (h.trend_24h, h.trend_168h)
                for pair, h in self._states.items()}

    def load_state(self, pair: str, trend_24h: int, trend_168h: int) -> None:
        """
        Restore persisted hysteresis state for a pair.

        Call this on startup before classify_bar() to ensure continuity
        from the previous session. Logs the restored state for audit.

        Args:
            pair:       Trading pair identifier.
            trend_24h:  Persisted trend state for 24h MA (0 or 1).
            trend_168h: Persisted trend state for 168h MA (0 or 1).
        """
        self._states[pair] = _PairHysteresis(
            trend_24h=trend_24h, trend_168h=trend_168h)
        logger.info(
            f"{pair}: loaded hysteresis — "
            f"trend_24h={trend_24h}, trend_168h={trend_168h}")

    def reset(self, pair: str = None) -> None:
        """
        Reset hysteresis state.

        Args:
            pair: Reset a single pair, or all pairs if None.
        """
        if pair:
            self._states.pop(pair, None)
        else:
            self._states.clear()

    # -----------------------------------------------------------------
    # Core hysteresis logic (LOCKED — matches backtest v03 line 321-347)
    # -----------------------------------------------------------------

    def _apply_hysteresis(
        self, price: float, ma: float, current: int
    ) -> int:
        """
        Apply AND-logic hysteresis to a single trend component.

        AND-logic (backtest v03 validated):
            Exit bullish:  price < ma*(1-exit)  AND price < ma*(1-entry)
            Enter bullish: price > ma*(1+exit)  AND price > ma*(1+entry)

        Since entry_buffer > exit_buffer, the AND condition means:
            Exit:  binding threshold is ma*(1-entry) = ma*0.985
            Enter: binding threshold is ma*(1+entry) = ma*1.015
            Effective: symmetric ±1.5%

        Args:
            price:   Current close price.
            ma:      Moving average value.
            current: Current trend state (0=bearish, 1=bullish).

        Returns:
            Updated trend state (0 or 1).
        """
        entry = self.config.entry_buffer
        exit_ = self.config.exit_buffer

        if current == 1:  # Currently bullish
            if (price < ma * (1 - exit_)
                    and price < ma * (1 - entry)):
                return 0
        else:  # Currently bearish
            if (price > ma * (1 + exit_)
                    and price > ma * (1 + entry)):
                return 1

        return current

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _get_or_create(self, pair: str) -> _PairHysteresis:
        """Get existing hysteresis state or create with default (bullish)."""
        if pair not in self._states:
            self._states[pair] = _PairHysteresis()
        return self._states[pair]

    def __repr__(self) -> str:
        return (f"RegimeClassifier(pairs={len(self._states)}, "
                f"entry={self.config.entry_buffer:.1%}, "
                f"exit={self.config.exit_buffer:.1%})")