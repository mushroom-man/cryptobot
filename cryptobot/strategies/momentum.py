# -*- coding: utf-8 -*-
"""
CryptoBot - Momentum Strategy (Thin Wrapper)
==============================================
FILE: cryptobot/strategies/momentum.py

Orchestrates the shared signal engine (regime.py, hit_rates.py, quality.py)
to produce position signals. This is the single strategy entry point for
both live trading (via trader.py) and backtesting.

Signal Flow:
    1. regime.classify_bar()     â†’ RegimeState (16-state classification)
    2. NO_MA72_ONLY filter       â†’ suppress MA72-only transitions
    3. hit_rates.get_signal()    â†’ +1.0 / -0.5 / 0.0 (direction from hit rates)
    4. boost check               â†’ S11 â‰¥ 12h â†’ Ã—1.5
    4.5 monthly trend veto       -> bearish monthly -> longs vetoed to flat
    5. quality.compute_scalar()  â†’ scale by dist_ma168
    6. return SignalResult        â†’ multiplier + metadata

Config E (Production Reference):
    - MA Periods: 16 / 6 / 2  (24h / 72h / 168h)
    - Hysteresis: AND-logic, 1.5% entry / 0.5% exit (symmetric Â±1.5%)
    - Confirmation: 0h (removed â€” raw state = confirmed state)
    - Boost: State 11 â‰¥ 12h â†’ 1.5Ã—
    - Quality: dist_ma168 linear scalar, 15% dead zone
    - Short sizing: 0.5Ã— via hit_rates
    - Signal logic: dynamic hit rates (no hardcoded states)

Implements Predictor protocol:
    predict(features) â†’ SignalResult

Usage:
    from cryptobot.strategies.momentum import MomentumStrategy

    strategy = MomentumStrategy()
    strategy.load_state_for_pairs(db, pairs)
    strategy.init_hit_rates('ETHUSD', signals_24h, returns_24h, current_date)

    result = strategy.predict({
        'pair': 'ETHUSD',
        'price': 3200.0,
        'ma_24h': 3150.0,
        'ma_72h': 3100.0,
        'ma_168h': 3050.0,
    })
    # result.multiplier â†’ +0.85 (long, quality-reduced)
    # result.direction  â†’ 'LONG'
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

from cryptobot.signals.regime import RegimeClassifier, RegimeConfig, RegimeState
from cryptobot.signals.hit_rates import HitRateCalculator, HitRateConfig
from cryptobot.signals.quality import QualityFilter, QualityConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class MomentumConfig:
    """
    Immutable momentum strategy configuration.

    Covers boost logic, transition filter, and monthly veto.
    Regime classification (including MA periods and hysteresis buffers),
    hit rate thresholds, and quality filter parameters are configured
    via their respective module configs. MA periods live in RegimeConfig
    (single source of truth).
    """
    # Boost (validated Config E)
    boost_state: int = 11
    boost_threshold_hours: int = 12
    boost_multiplier: float = 1.5

    # NO_MA72_ONLY filter
    use_signal_filter: bool = True

    # Monthly trend veto (validated Config H — Sharpe 2.07)
    # When enabled, bearish monthly trend vetoes long positions to flat.
    # Monthly trend value is computed by features.get_current_monthly_trend()
    # and passed into predict() via features dict.
    use_monthly_veto: bool = True


# =============================================================================
# SIGNAL RESULT
# =============================================================================

@dataclass(frozen=True)
class SignalResult:
    """
    Complete signal output from predict().

    Attributes:
        multiplier:         Final position multiplier (signed, quality-adjusted).
                            Positive = long, negative = short, zero = flat.
                            Examples: +1.0, +1.5, +0.75, -0.5, -0.35, 0.0
        state_int:          Active regime state (0â€“15, post-filter).
        raw_state_int:      Raw regime state from classifier (pre-filter).
        direction:          Sign of multiplier: +1, -1, or 0.
        hit_rate:           Hit rate for the active state.
        hit_rate_n:         Sample count for the active state.
        hit_rate_sufficient: Whether min sample threshold was met.
        base_signal:        Signal before boost and quality (+1.0 / -0.5 / 0.0).
        quality_scalar:     Quality scalar applied (0.3â€“1.0).
        is_boosted:         Whether S11 boost is active.
        duration_hours:     Hours in current active state.
        was_filtered:       True if NO_MA72_ONLY suppressed a transition this bar.
        monthly_vetoed:     True if long was vetoed by bearish monthly trend.
    """
    multiplier: float
    state_int: int
    raw_state_int: int
    direction: int
    hit_rate: float
    hit_rate_n: int
    hit_rate_sufficient: bool
    base_signal: float
    quality_scalar: float
    is_boosted: bool
    duration_hours: int
    was_filtered: bool
    monthly_vetoed: bool = False

    @property
    def signal_type(self) -> str:
        """Human-readable signal type for reporting."""
        if self.monthly_vetoed:
            return 'MONTHLY_VETOED'
        elif self.multiplier == 0:
            return 'FLAT'
        elif self.is_boosted:
            return 'BOOSTED_LONG'
        elif self.multiplier > 0:
            return 'LONG'
        else:
            return 'SHORT'


# =============================================================================
# INTERNAL STATE TRACKING (per pair)
# =============================================================================

@dataclass
class _PairTracker:
    """
    Mutable per-pair state.

    Tracks active state (post-filter), duration (for boost), and
    previous raw components (for NO_MA72_ONLY filter).

    IMPORTANT: Duration tracking is SEPARATED from filter logic,
    matching backtest v03 lines 988â€“998. Duration compares the
    active_state_int across bars, independent of how the filter
    updated the active components.
    """
    pair: str

    # Active state: post-NO_MA72_ONLY filter
    active_components: Optional[Tuple[int, int, int, int]] = None
    active_state_int: Optional[int] = None

    # Duration: tracks consecutive hours at the same active_state_int
    duration_hours: int = 0

    # Previous RAW components (for NO_MA72_ONLY filter â€” always updated)
    prev_raw_components: Optional[Tuple[int, int, int, int]] = None

    # Boost tracking
    was_boosted: bool = False


# =============================================================================
# MOMENTUM STRATEGY
# =============================================================================

class MomentumStrategy:
    """
    Momentum strategy â€” thin wrapper over the shared signal engine.

    Orchestrates regime classification, hit rate lookup, quality
    filtering, boost logic, and the NO_MA72_ONLY transition filter.

    Args:
        momentum_config: Boost and filter parameters.
        regime_config:   Hysteresis buffer parameters (passed to RegimeClassifier).
        hit_rate_config: Hit rate thresholds and short scalar (passed to HitRateCalculator).
        quality_config:  Quality filter thresholds and dead zone (passed to QualityFilter).
    """

    def __init__(
        self,
        momentum_config: Optional[MomentumConfig] = None,
        regime_config: Optional[RegimeConfig] = None,
        hit_rate_config: Optional[HitRateConfig] = None,
        quality_config: Optional[QualityConfig] = None,
    ):
        self.config = momentum_config or MomentumConfig()
        self.classifier = RegimeClassifier(regime_config)
        self.hit_rates = HitRateCalculator(hit_rate_config)
        self.quality = QualityFilter(quality_config)

        self._trackers: Dict[str, _PairTracker] = {}

    # -----------------------------------------------------------------
    # Hit rate management (call before predict)
    # -----------------------------------------------------------------

    def init_hit_rates(
        self,
        pair: str,
        signals_24h,
        returns_24h,
        current_date,
    ) -> bool:
        """
        Initialise or update hit rates for a pair.

        Must be called at least once per pair before predict().
        Subsequent calls only recalculate when the month changes.

        Args:
            pair:         Trading pair identifier.
            signals_24h:  DataFrame with columns: trend_24h, trend_168h,
                          ma72_above_ma24, ma168_above_ma24 (24h bar index).
            returns_24h:  Series of 24h close-to-close returns (same index).
            current_date: Current timestamp for month-boundary check.

        Returns:
            True if recalculation was performed.
        """
        return self.hit_rates.update(pair, signals_24h, returns_24h, current_date)

    def force_init_hit_rates(
        self,
        pair: str,
        signals_24h,
        returns_24h,
        current_date,
    ) -> None:
        """Force hit rate recalculation regardless of month. For initialisation."""
        self.hit_rates.force_recalc(pair, signals_24h, returns_24h, current_date)

    def has_minimum_training(self, current_date, data_start) -> bool:
        """Check if enough historical data exists for trading."""
        return self.hit_rates.has_minimum_training(current_date, data_start)

    # -----------------------------------------------------------------
    # Core signal generation
    # -----------------------------------------------------------------

    def predict(self, features: Dict) -> SignalResult:
        """
        Generate position signal from features.

        Implements the full signal flow:
            1. Regime classification (with hysteresis)
            2. NO_MA72_ONLY transition filter
            3. Hit rate lookup â†’ base signal
            4. Boost check (S11 â‰¥ 12h)
            5. Quality filter (dist_ma168 scalar)

        Args:
            features: Dict with keys:
                pair:    Trading pair (default: 'DEFAULT')
                price:   Current close price
                ma_24h:  24-hour moving average
                ma_72h:  72-hour moving average
                ma_168h: 168-hour moving average

        Returns:
            SignalResult with final multiplier and metadata.
        """
        pair = features.get('pair', 'DEFAULT')
        price = features.get('price') or features.get('close')
        ma_24h = features.get('ma_24h')
        ma_72h = features.get('ma_72h')
        ma_168h = features.get('ma_168h')

        # Validate inputs
        if price is None or ma_24h is None or ma_168h is None:
            logger.warning(f"{pair}: missing required features")
            return self._flat_result(pair)

        if ma_72h is None:
            ma_72h = ma_24h  # degrade gracefully; MA72 comparison â†’ 0

        tracker = self._get_or_create(pair)

        # =================================================================
        # STEP 1: Regime classification
        # =================================================================
        regime: RegimeState = self.classifier.classify_bar(
            pair, price, ma_24h, ma_72h, ma_168h)
        raw_components = regime.components

        # =================================================================
        # STEP 2: NO_MA72_ONLY filter
        # (matches backtest v03 lines 928â€“956 with confirmation_hours=0)
        #
        # Filter decides whether active_components updates.
        # Duration tracking is SEPARATE (below).
        # =================================================================
        was_filtered = self._apply_filter(tracker, raw_components, regime.state_int)

        # Always update prev_raw (matches backtest line 956)
        tracker.prev_raw_components = raw_components

        # =================================================================
        # DURATION TRACKING (matches backtest v03 lines 988â€“998)
        # Separated from filter â€” compares active_state_int across bars.
        # =================================================================
        current_active_int = _components_to_int(tracker.active_components)

        if current_active_int == tracker.active_state_int:
            tracker.duration_hours += 1
        else:
            old_int = tracker.active_state_int
            tracker.active_state_int = current_active_int
            tracker.duration_hours = 1
            tracker.was_boosted = False

            if old_int is not None:
                logger.info(
                    f"{pair}: state transition "
                    f"S{old_int} â†’ S{current_active_int}")

        # =================================================================
        # STEP 3: Hit rate lookup â†’ base signal
        # =================================================================
        if not self.hit_rates.is_cached(pair):
            logger.warning(f"{pair}: hit rates not initialised â€” flat")
            return self._flat_result(pair, tracker=tracker,
                                     raw_state_int=regime.state_int)

        active_key = self._components_to_key(tracker.active_components)
        hr_data = self.hit_rates.get_hit_rate(pair, active_key)
        base_signal = self.hit_rates.get_signal(pair, active_key)

        # =================================================================
        # STEP 4: Boost (S11 â‰¥ 12h, long only)
        # (matches backtest v03 lines 1000â€“1011)
        # =================================================================
        is_boosted = False
        signal = base_signal

        if (signal > 0
                and tracker.active_state_int == self.config.boost_state
                and tracker.duration_hours >= self.config.boost_threshold_hours):
            signal *= self.config.boost_multiplier
            is_boosted = True

            if not tracker.was_boosted:
                tracker.was_boosted = True
                logger.info(
                    f"{pair}: ðŸš€ BOOST activated â€” "
                    f"S{tracker.active_state_int} "
                    f"@ {tracker.duration_hours}h")

        # =================================================================
        # STEP 4.5: Monthly trend veto (Config H, v05)
        # Bearish monthly trend -> veto longs to flat.
        # Shorts pass through unchanged.
        # Monthly trend is computed by features.get_current_monthly_trend()
        # and passed in via features dict. Default 1 (bullish) if absent.
        # =================================================================
        monthly_vetoed = False
        if self.config.use_monthly_veto:
            monthly_trend = features.get('monthly_trend', 1)
            if monthly_trend == 0 and signal > 0:
                logger.info(
                    f"{pair}: MONTHLY VETO - bearish trend, "
                    f"long signal {signal:+.2f} vetoed to flat")
                signal = 0.0
                monthly_vetoed = True

        # =================================================================
        # STEP 5: Quality filter (dist_ma168 scalar)
        # (matches backtest v03 lines 1037â€“1052)
        # Uses 24h bar close (not hourly) to prevent intra-day churn.
        # Quality scalar should only update when a new 24h bar completes,
        # matching backtest resolution and preventing unnecessary rebalances.
        # =================================================================
        direction = 1 if signal > 0 else (-1 if signal < 0 else 0)
        quality_price = features.get('close_24h', price)  # Fallback to hourly if missing
        dist_ma168 = self.quality.compute_dist_ma168(quality_price, ma_168h)
        quality_scalar = self.quality.compute_scalar(dist_ma168, direction)

        multiplier = signal * quality_scalar

        # =================================================================
        # STEP 6: Build result
        # =================================================================
        result = SignalResult(
            multiplier=multiplier,
            state_int=tracker.active_state_int,
            raw_state_int=regime.state_int,
            direction=direction,
            hit_rate=hr_data.get('hit_rate', 0.5),
            hit_rate_n=hr_data.get('n', 0),
            hit_rate_sufficient=hr_data.get('sufficient', False),
            base_signal=base_signal,
            quality_scalar=quality_scalar,
            is_boosted=is_boosted,
            duration_hours=tracker.duration_hours,
            was_filtered=was_filtered,
            monthly_vetoed=monthly_vetoed,
        )

        self._log_signal(pair, result)
        return result

    # -----------------------------------------------------------------
    # NO_MA72_ONLY filter (matches backtest v03 lines 466â€“485, 928â€“956)
    # -----------------------------------------------------------------

    def _apply_filter(
        self,
        tracker: _PairTracker,
        raw_components: Tuple[int, int, int, int],
        raw_state_int: int,
    ) -> bool:
        """
        Apply NO_MA72_ONLY filter. Updates tracker.active_components.

        Returns True if a transition was suppressed (filtered).

        Logic (backtest v03, confirmation_hours=0):
            1. First bar (prev is None): initialise active = raw
            2. Raw changed from prev:
               a. Only MA72 changed â†’ suppress (active unchanged)
               b. Meaningful transition AND different from active â†’ update active
            3. Raw unchanged: do nothing (active unchanged)
        """
        was_filtered = False

        if tracker.prev_raw_components is None:
            # First bar â€” initialise (backtest line 949â€“951)
            tracker.active_components = raw_components

        elif raw_components != tracker.prev_raw_components:
            # Raw state changed â€” check filter
            if (self.config.use_signal_filter
                    and _only_ma72_changed(
                        tracker.prev_raw_components, raw_components)):
                # Suppress: only MA72 changed (backtest line 937â€“938)
                was_filtered = True
                logger.debug(
                    f"{tracker.pair}: NO_MA72_ONLY suppressed â€” "
                    f"raw S{raw_state_int}, active stays "
                    f"S{tracker.active_state_int}")

            elif raw_components != tracker.active_components:
                # Meaningful transition to a new state (backtest line 939â€“943)
                tracker.active_components = raw_components

            # else: filter passed but raw == active â€” no update needed

        # else: raw == prev â€” no transition, active unchanged

        return was_filtered

    # -----------------------------------------------------------------
    # State persistence
    # -----------------------------------------------------------------

    def load_state_for_pairs(self, db, pairs: list) -> int:
        """
        Load persisted state from database for specific pairs.

        Restores:
            - Classifier hysteresis (trend_24h, trend_168h)
            - Active state + duration (for boost tracking)
            - Previous raw state (for NO_MA72_ONLY filter)

        DB column mapping (reuses existing schema, no migration):
            confirmed_state â†’ active_state_int
            duration_hours  â†’ duration_hours
            trend_24h       â†’ classifier hysteresis
            trend_168h      â†’ classifier hysteresis
            pending_state   â†’ prev_raw_state_int (REPURPOSED)
            pending_hours   â†’ unused (ignored on load)

        Args:
            db: Database instance.
            pairs: List of trading pairs.

        Returns:
            Number of pairs loaded.
        """
        loaded = 0
        for pair in pairs:
            state_data = db.get_strategy_state(pair, strategy='momentum')
            if state_data:
                # Restore classifier hysteresis
                self.classifier.load_state(
                    pair,
                    trend_24h=state_data['trend_24h'],
                    trend_168h=state_data['trend_168h'],
                )

                # Restore tracker
                active_int = state_data['confirmed_state']
                prev_raw_int = state_data.get('pending_state')

                self._trackers[pair] = _PairTracker(
                    pair=pair,
                    active_components=_int_to_components(active_int),
                    active_state_int=active_int,
                    duration_hours=state_data['duration_hours'],
                    prev_raw_components=(
                        _int_to_components(prev_raw_int)
                        if prev_raw_int is not None else None),
                    was_boosted=False,
                )

                loaded += 1
                logger.info(
                    f"{pair}: loaded state â€” S{active_int} "
                    f"({state_data['duration_hours']}h), "
                    f"t24={state_data['trend_24h']}, "
                    f"t168={state_data['trend_168h']}")
            else:
                self._trackers[pair] = _PairTracker(pair=pair)
                logger.info(f"{pair}: initialised fresh state")

        return loaded

    def save_state(self, db) -> int:
        """
        Save current state to database for all pairs.

        See load_state_for_pairs() for column mapping.

        Args:
            db: Database instance.

        Returns:
            Number of pairs saved.
        """
        saved = 0
        for pair, tracker in self._trackers.items():
            hysteresis = self.classifier.get_hysteresis(pair)
            if hysteresis is None:
                logger.warning(f"{pair}: no hysteresis state â€” skipping save")
                continue

            trend_24h, trend_168h = hysteresis

            # Repurpose pending_state column for prev_raw_state_int
            prev_raw_int = None
            if tracker.prev_raw_components is not None:
                prev_raw_int = _components_to_int(tracker.prev_raw_components)

            db.save_strategy_state(
                pair=pair,
                confirmed_state=tracker.active_state_int or 0,
                duration_hours=tracker.duration_hours,
                pending_state=prev_raw_int,   # REPURPOSED column
                pending_hours=0,              # unused
                trend_24h=trend_24h,
                trend_168h=trend_168h,
                strategy='momentum',
            )
            saved += 1
            logger.debug(
                f"{pair}: saved â€” S{tracker.active_state_int} "
                f"({tracker.duration_hours}h)")

        return saved

    # -----------------------------------------------------------------
    # Reporting / introspection
    # -----------------------------------------------------------------

    def get_pair_state(self, pair: str) -> Optional[dict]:
        """Get current state summary for a pair."""
        tracker = self._trackers.get(pair)
        if tracker is None or tracker.active_state_int is None:
            return None
        return {
            'state_int': tracker.active_state_int,
            'duration_hours': tracker.duration_hours,
            'is_boosted': tracker.was_boosted,
            'active_components': tracker.active_components,
        }

    def get_all_states(self) -> Dict[str, dict]:
        """Get state summary for all pairs."""
        return {pair: self.get_pair_state(pair)
                for pair in self._trackers
                if self._trackers[pair].active_state_int is not None}

    def get_active_boosts(self) -> List[str]:
        """Get list of pairs with active boost."""
        return [
            pair for pair, t in self._trackers.items()
            if (t.active_state_int == self.config.boost_state
                and t.duration_hours >= self.config.boost_threshold_hours)
        ]

    def get_pending_boosts(self) -> Dict[str, int]:
        """Get pairs approaching boost with hours remaining."""
        result = {}
        for pair, t in self._trackers.items():
            if (t.active_state_int == self.config.boost_state
                    and t.duration_hours < self.config.boost_threshold_hours):
                result[pair] = (self.config.boost_threshold_hours
                                - t.duration_hours)
        return result

    def get_short_pairs(self) -> List[str]:
        """Get list of pairs currently in short-signal states."""
        result = []
        for pair, t in self._trackers.items():
            if t.active_components is None:
                continue
            key = self._components_to_key(t.active_components)
            if self.hit_rates.is_cached(pair):
                signal = self.hit_rates.get_signal(pair, key)
                if signal < 0:
                    result.append(pair)
        return result

    def describe_state(self, pair: str) -> str:
        """Get human-readable state description."""
        tracker = self._trackers.get(pair)
        if tracker is None or tracker.active_state_int is None:
            return f"{pair}: No state"

        s = tracker.active_state_int
        d = tracker.duration_hours

        # Determine direction from hit rates if available
        if tracker.active_components and self.hit_rates.is_cached(pair):
            key = self._components_to_key(tracker.active_components)
            signal = self.hit_rates.get_signal(pair, key)
            is_boosted = (
                signal > 0
                and s == self.config.boost_state
                and d >= self.config.boost_threshold_hours)
            if is_boosted:
                regime = "BOOSTED_LONG"
            elif signal > 0:
                regime = "LONG"
            elif signal < 0:
                regime = "SHORT"
            else:
                regime = "FLAT"
        else:
            regime = "UNKNOWN"

        return f"{pair}: S{s} ({d}h) â†’ {regime}"

    def reset(self, pair: str = None) -> None:
        """Reset state for one pair or all pairs."""
        if pair:
            self._trackers.pop(pair, None)
            self.classifier.reset(pair)
            self.hit_rates.reset(pair)
        else:
            self._trackers.clear()
            self.classifier.reset()
            self.hit_rates.reset()

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_or_create(self, pair: str) -> _PairTracker:
        """Get existing tracker or create fresh one."""
        if pair not in self._trackers:
            self._trackers[pair] = _PairTracker(pair=pair)
        return self._trackers[pair]

    def _flat_result(
        self,
        pair: str,
        tracker: _PairTracker = None,
        raw_state_int: int = 0,
    ) -> SignalResult:
        """Build a flat SignalResult for edge cases."""
        return SignalResult(
            multiplier=0.0,
            state_int=tracker.active_state_int if tracker else 0,
            raw_state_int=raw_state_int,
            direction=0,
            hit_rate=0.5,
            hit_rate_n=0,
            hit_rate_sufficient=False,
            base_signal=0.0,
            quality_scalar=1.0,
            is_boosted=False,
            duration_hours=tracker.duration_hours if tracker else 0,
            was_filtered=False,
            monthly_vetoed=False,
        )

    @staticmethod
    def _components_to_key(
        components: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Build hit rate lookup key from components tuple."""
        return ((components[0], components[1]),
                (components[2], components[3]))

    def _log_signal(self, pair: str, result: SignalResult) -> None:
        """Log signal for debugging."""
        boost = " ðŸš€" if result.is_boosted else ""
        filt = " âš¡FILTERED" if result.was_filtered else ""
        qual = (f" Q={result.quality_scalar:.2f}"
                if result.quality_scalar < 1.0 else "")
        veto = " MONTHLY_VETO" if result.monthly_vetoed else ""

        logger.debug(
            f"{pair}: S{result.state_int}({result.duration_hours}h) "
            f"raw=S{result.raw_state_int} â†’ {result.signal_type} "
            f"{result.multiplier:+.3f} "
            f"(base={result.base_signal:+.1f}, "
            f"hr={result.hit_rate:.2f}, n={result.hit_rate_n})"
            f"{qual}{boost}{filt}{veto}")

    def __repr__(self) -> str:
        active = len([
            t for t in self._trackers.values()
            if t.active_state_int is not None
        ])
        return (f"MomentumStrategy(pairs={len(self._trackers)}, "
                f"active={active})")


# =============================================================================
# MODULE-LEVEL HELPERS (shared by filter and persistence)
# =============================================================================

def _only_ma72_changed(
    prev: Tuple[int, int, int, int],
    curr: Tuple[int, int, int, int],
) -> bool:
    """
    Check if MA72 was the sole component that changed.

    Matches backtest v03 lines 476â€“485 exactly.
    """
    t24_changed = prev[0] != curr[0]
    t168_changed = prev[1] != curr[1]
    ma72_changed = prev[2] != curr[2]
    ma168_changed = prev[3] != curr[3]

    return (ma72_changed
            and not t24_changed
            and not t168_changed
            and not ma168_changed)


def _components_to_int(components: Tuple[int, int, int, int]) -> int:
    """State integer from component tuple: (t24, t168, ma72, ma168) â†’ 0â€“15."""
    return (components[0] * 8 + components[1] * 4
            + components[2] * 2 + components[3])


def _int_to_components(state_int: int) -> Tuple[int, int, int, int]:
    """Component tuple from state integer: 0â€“15 â†’ (t24, t168, ma72, ma168)."""
    return (
        (state_int >> 3) & 1,
        (state_int >> 2) & 1,
        (state_int >> 1) & 1,
        state_int & 1,
    )