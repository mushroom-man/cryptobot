#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long/Short Backtest v03 — Hourly Cycle with Confirmation Filter
================================================================
CHANGES FROM v02 (daily cycle):
  - Main loop now iterates over 1h bars (matching live bot cycle)
  - Signals computed at 1h resolution: price checked vs forward-filled MAs every hour
  - Position changes and PnL accrue hourly
  - New CONFIRMATION_HOURS parameter (0=disabled, 1-4): state must persist N
    additional hours before position changes. Allows optimisation.
  - Hit rates still calculated from 24h bars (statistical validity, sample size)
  - Sharpe ratio computed from daily-resampled equity (avoids inflated hourly Sharpe)
  - Rollover costs adjusted to per-hour
  - Duration tracking in hours (displayed as days)

NOTE: This produces DIFFERENT results than the daily version.
  1h signals detect MA crossovers faster (1h lag vs 24h lag).
  Compare against v02 daily baseline to understand impact.

Position Logic:
    LONG:  hit_rate > 0.50 (sufficient samples) → long at risk-parity weight
    SHORT: hit_rate ≤ 0.50 (sufficient samples) → short at 50% sizing
    FLAT:  insufficient samples → no position

Quality Filter (Config C):
    Modulates position SIZE based on distance to MA(168h).
    Longs:  near/below MA168 → full size (1.0), overextended → half (0.5)
    Shorts: oversold below MA168 → minimal (0.3), exhaustion above → full (1.0)

Kraken Cost Model:
    LONG:  0.26% trading + 0.10% slippage per side = 0.72% round trip
    SHORT: 0.26% trading + 0.10% slippage per side + 0.02% margin open
           + 0.005%/hour rollover (0.12%/day ÷ 24)

Configurations:
    BH = Buy & hold — equal weight, monthly rebalance
    A  = Long-only baseline — hit_rate > 0.50 → long, else flat
    B  = Long/Short — hit_rate ≤ 0.50 + sufficient → short at 50%
    C  = Long/Short + Quality Filter — dist_ma168 sizing modulation
    D  = Full Strategy — L/S + Quality + 3h Confirmation + S11 Boost
    E  = Boost Only — L/S + Quality + S11 Boost (no confirmation)

Usage:
    python backtest_long_short_full_analytics_03.py
"""

from pathlib import Path
import sys
import os

try:
    PROJECT_ROOT = Path(__file__).parent
    while PROJECT_ROOT != PROJECT_ROOT.parent:
        if (PROJECT_ROOT / 'cryptobot' / 'data').exists():
            break
        PROJECT_ROOT = PROJECT_ROOT.parent
    else:
        PROJECT_ROOT = Path(os.path.expanduser('~/cryptobot'))
except NameError:
    PROJECT_ROOT = Path(os.path.expanduser('~/cryptobot'))

sys.path.insert(0, str(PROJECT_ROOT))
from cryptobot.data.database import Database

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
RUN_CONFIGS = ['BH', 'A', 'B', 'C', 'D', 'E']

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked from validation)
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

# Hysteresis buffers (validated)
ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Hit rate / portfolio parameters
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_STATE = 20
MIN_TRAINING_MONTHS = 12
HIT_RATE_RECALC_MONTHS = 1
VOL_LOOKBACK_MONTHS = 1
COV_LOOKBACK_MONTHS = 2

# --- NEW: Confirmation filter ---
# Number of additional hours a new state must persist before confirmation.
# 0 = immediate (no confirmation, equivalent to v02 daily behaviour)
# 1-4 = state must hold for N extra hours after initial detection
CONFIRMATION_HOURS = 0

# Exposure limits
DIR_MAX_EXPOSURE = 2.0
DIR_TARGET_VOL = 0.40
DIR_DD_START_REDUCE = -0.20
DIR_DD_MIN_EXPOSURE = -0.50
DIR_MIN_EXPOSURE_FLOOR = 0.40

INITIAL_CAPITAL = 100000.0
MIN_TRADE_SIZE = 100.0

# Kraken cost model
KRAKEN_TRADING_FEE = 0.0026     # 0.26% taker per side
KRAKEN_SLIPPAGE = 0.0010        # 0.10% estimated per side
KRAKEN_MARGIN_OPEN = 0.0002     # 0.02% one-time on short entry
KRAKEN_ROLLOVER_4H = 0.0002     # 0.02% per 4 hours
KRAKEN_ROLLOVER_DAY = KRAKEN_ROLLOVER_4H * 6   # 0.12% per day
KRAKEN_ROLLOVER_HOUR = KRAKEN_ROLLOVER_DAY / 24  # 0.005% per hour
COST_PER_SIDE = KRAKEN_TRADING_FEE + KRAKEN_SLIPPAGE  # 0.36%

# Short configuration
SHORT_SIZE_SCALAR = 0.50    # Shorts at 50% of equivalent long sizing

# Boost configuration (from validated momentum.py)
BOOST_STATE = 11            # State integer that gets boosted
BOOST_THRESHOLD_HOURS = 12  # Hours in state before boost activates
BOOST_MULTIPLIER = 1.5      # Position multiplier when boosted

# Rebalance dead zone: suppress micro-adjustments within same direction.
# Only rebalance when relative exposure change exceeds this threshold.
# Prevents quality filter churn on hourly cycle (scalar twitches every hour
# as price moves, generating 87x more trades without meaningful sizing change).
# Direction changes (long→short, long→flat, etc.) always execute immediately.
REBALANCE_THRESHOLD = 0.15  # relative change required to trigger rebalance

# Quality filter thresholds (from quintile analysis of dist_ma168)
QUALITY_LONG_FULL = -0.04
QUALITY_LONG_MIN = 0.05
QUALITY_LONG_FLOOR = 0.50
QUALITY_SHORT_WEAK = -0.04
QUALITY_SHORT_FULL = 0.04
QUALITY_SHORT_FLOOR = 0.30

# Strategy configurations
STRATEGY_CONFIGS = {
    'BH': {
        'name': 'Buy & Hold',
        'allow_shorts': False,
        'use_quality_filter': False,
        'is_buy_hold': True,
        'description': 'Equal weight, monthly rebalance, fully invested',
        'confirmation_hours': 0,
        'use_boost': False,
    },
    'A': {
        'name': 'Long Only',
        'allow_shorts': False,
        'use_quality_filter': False,
        'is_buy_hold': False,
        'description': '16-state, hit_rate > 0.50 → long, else flat',
        'confirmation_hours': 0,
        'use_boost': False,
    },
    'B': {
        'name': 'Long/Short',
        'allow_shorts': True,
        'use_quality_filter': False,
        'is_buy_hold': False,
        'description': 'L/S: hit_rate ≤ 0.50 + sufficient → short at 50%',
        'confirmation_hours': 0,
        'use_boost': False,
    },
    'C': {
        'name': 'L/S + Quality',
        'allow_shorts': True,
        'use_quality_filter': True,
        'is_buy_hold': False,
        'description': 'L/S + quality filter: dist_ma168 modulates position sizing',
        'confirmation_hours': 0,
        'use_boost': False,
    },
    'D': {
        'name': 'L/S+Qual+Conf+Boost',
        'allow_shorts': True,
        'use_quality_filter': True,
        'is_buy_hold': False,
        'description': 'Full strategy: L/S + quality + 3h confirmation + S11 boost',
        'confirmation_hours': 3,
        'use_boost': True,
    },
    'E': {
        'name': 'L/S+Qual+Boost',
        'allow_shorts': True,
        'use_quality_filter': True,
        'is_buy_hold': False,
        'description': 'L/S + quality + S11 boost (NO confirmation)',
        'confirmation_hours': 0,
        'use_boost': True,
    },
}

# State labels for reporting
STATE_NAMES = {
    0:  'All bearish',
    1:  'Bear trends, MA168>MA24',
    2:  'Bear trends, MA72>MA24',
    3:  'Bear trends, both MAs above',
    4:  'Bear 24h, Bull 168h, MAs below',
    5:  'Bear 24h, Bull 168h, MA168>MA24',
    6:  'Bear 24h, Bull 168h, MA72>MA24',
    7:  'Bear 24h, Bull 168h, both MAs above',
    8:  'Bull 24h, Bear 168h, MAs below',
    9:  'Bull 24h, Bear 168h, MA168>MA24',
    10: 'Bull 24h, Bear 168h, MA72>MA24',
    11: 'Bull 24h, Bear 168h, both MAs above',
    12: 'All bull, MAs below (EXHAUSTION)',
    13: 'All bull, MA168>MA24',
    14: 'All bull, MA72>MA24 (EXHAUSTION)',
    15: 'All bull, all MAs above',
}


# =============================================================================
# CALENDAR HELPERS
# =============================================================================

def get_month_start(date: pd.Timestamp, months_back: int = 0) -> pd.Timestamp:
    target = date - pd.DateOffset(months=months_back)
    return target.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_months_of_data(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def has_minimum_training(current_date: pd.Timestamp, data_start: pd.Timestamp,
                          min_months: int) -> bool:
    return get_months_of_data(data_start, current_date) >= min_months


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestResult:
    name: str
    config: str = ""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_raw: float = 0.0
    calmar_ratio: float = 0.0
    # Trade counts
    n_long_trades: int = 0
    n_short_trades: int = 0
    n_total_trades: int = 0
    # Cost breakdown
    long_trading_costs: float = 0.0
    short_trading_costs: float = 0.0
    short_rollover_costs: float = 0.0
    total_costs: float = 0.0
    # Signal stats
    signals_filtered: int = 0
    signals_total: int = 0
    confirmations_cancelled: int = 0    # NEW: pending states that never confirmed
    rebalances_suppressed: int = 0     # NEW: micro-adjustments filtered by dead zone
    # Position duration (internal: hours; displayed as days)
    short_hours: int = 0
    short_states_used: Dict = field(default_factory=dict)
    avg_short_duration_hours: float = 0.0
    long_hours: int = 0
    avg_long_duration_hours: float = 0.0
    # Series
    equity_curve: Optional[pd.Series] = None
    yearly_returns: Dict[int, float] = field(default_factory=dict)
    # PnL attribution
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    short_pnl_gross: float = 0.0
    # Quality filter stats
    avg_long_scalar: float = 1.0
    avg_short_scalar: float = 1.0
    quality_adjustments: int = 0
    # Boost stats
    boost_hours: int = 0              # Hours where boost was active
    boost_pnl: float = 0.0           # PnL attributable to boost multiplier
    boost_activations: int = 0        # Number of times boost activated

    # Convenience properties for display (hours → days)
    @property
    def short_days(self) -> float:
        return self.short_hours / 24.0

    @property
    def long_days(self) -> float:
        return self.long_hours / 24.0

    @property
    def avg_short_duration(self) -> float:
        return self.avg_short_duration_hours / 24.0

    @property
    def avg_long_duration(self) -> float:
        return self.avg_long_duration_hours / 24.0


# =============================================================================
# SIGNAL GENERATION — 1H RESOLUTION (for hourly trading loop)
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def label_trend_binary_rolling(
    close_series: pd.Series, ma_series: pd.Series,
    entry_buffer: float, exit_buffer: float
) -> pd.Series:
    labels = pd.Series(index=close_series.index, dtype=float)
    labels[:] = np.nan
    current = 1

    for i in range(len(close_series)):
        idx = close_series.index[i]
        price = close_series.iloc[i]
        ma = ma_series.iloc[i] if idx in ma_series.index else np.nan

        if pd.isna(ma):
            labels.iloc[i] = current
            continue

        if current == 1:
            if price < ma * (1 - exit_buffer) and price < ma * (1 - entry_buffer):
                current = 0
        else:
            if price > ma * (1 + exit_buffer) and price > ma * (1 + entry_buffer):
                current = 1

        labels.iloc[i] = current

    return labels.astype(int)


def generate_signals_1h(df_1h: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate 16-state signals at 1h resolution for hourly trading.

    MAs are computed on their native timeframe bars (24h, 72h, 168h),
    then forward-filled to 1h. Trend labels check the 1h close against
    the forward-filled MAs with hysteresis, allowing crossover detection
    at any hour rather than waiting for bar close.

    Returns:
        signals_1h:  DataFrame with trend_24h, trend_168h, ma72_above_ma24,
                     ma168_above_ma24, state_int — indexed on 1h bars
        features_1h: DataFrame with dist_ma168 — indexed on 1h bars
    """
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')

    # MAs at native timeframes
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()

    # Forward-fill all MAs to 1h resolution
    ma_24h_1h = ma_24h.reindex(df_1h.index, method='ffill')
    ma_72h_1h = ma_72h.reindex(df_1h.index, method='ffill')
    ma_168h_1h = ma_168h.reindex(df_1h.index, method='ffill')

    # Trend labels: 1h close vs forward-filled MAs (with hysteresis)
    # This allows crossover detection at any hour
    trend_24h = label_trend_binary_rolling(
        df_1h['close'], ma_24h_1h, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h = label_trend_binary_rolling(
        df_1h['close'], ma_168h_1h, ENTRY_BUFFER, EXIT_BUFFER)

    # MA comparisons at 1h (forward-filled, only change at bar boundaries)
    ma72_above_ma24 = (ma_72h_1h > ma_24h_1h).astype(int)
    ma168_above_ma24 = (ma_168h_1h > ma_24h_1h).astype(int)

    # Build signal DataFrame — shift(1) = 1 hour look-ahead prevention
    signals = pd.DataFrame(index=df_1h.index)
    signals['trend_24h'] = trend_24h.shift(1)
    signals['trend_168h'] = trend_168h.shift(1)
    signals['ma72_above_ma24'] = ma72_above_ma24.shift(1)
    signals['ma168_above_ma24'] = ma168_above_ma24.shift(1)

    signals = signals.dropna().astype(int)
    signals['state_int'] = (
        signals['trend_24h'] * 8 +
        signals['trend_168h'] * 4 +
        signals['ma72_above_ma24'] * 2 +
        signals['ma168_above_ma24'] * 1
    )

    # Quality filter feature: distance to MA(168h) at 1h resolution
    features = pd.DataFrame(index=df_1h.index)
    features['dist_ma168'] = ((df_1h['close'] - ma_168h_1h) / ma_168h_1h).shift(1)
    features = features.reindex(signals.index)

    return signals, features


# =============================================================================
# SIGNAL GENERATION — 24H RESOLUTION (for hit rate calculation)
# =============================================================================

def generate_signals_24h(df_1h: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate 16-state signals at 24h resolution for hit rate calculation.

    This preserves the original validated methodology: signals and returns
    both on 24h bars, ensuring hit rate estimates remain statistically valid.

    Returns:
        signals_24h: DataFrame with state components + state_int (24h index)
        df_24h:      Resampled 24h OHLCV
    """
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')

    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()

    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')

    trend_24h = label_trend_binary_rolling(
        df_24h['close'], ma_24h, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h_raw = label_trend_binary_rolling(
        df_168h['close'], df_168h['close'].rolling(MA_PERIOD_168H).mean(),
        ENTRY_BUFFER, EXIT_BUFFER
    )

    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_168h'] = trend_168h_raw.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)

    signals = aligned.dropna().astype(int)
    signals['state_int'] = (
        signals['trend_24h'] * 8 +
        signals['trend_168h'] * 4 +
        signals['ma72_above_ma24'] * 2 +
        signals['ma168_above_ma24'] * 1
    )

    return signals, df_24h


# =============================================================================
# SIGNAL FILTER (NO_MA72_ONLY) — unchanged
# =============================================================================

def should_trade_signal_16state(prev_state, curr_state, use_filter=True):
    if prev_state is None:
        return True
    if prev_state == curr_state:
        return False
    if not use_filter:
        return True
    if len(curr_state) != 4:
        return True

    trend_24h_changed = prev_state[0] != curr_state[0]
    trend_168h_changed = prev_state[1] != curr_state[1]
    ma72_changed = prev_state[2] != curr_state[2]
    ma168_changed = prev_state[3] != curr_state[3]

    only_ma72_changed = (ma72_changed and
                         not trend_24h_changed and
                         not trend_168h_changed and
                         not ma168_changed)
    return not only_ma72_changed


# =============================================================================
# HIT RATE CALCULATION — unchanged (still uses 24h data)
# =============================================================================

def calculate_expanding_hit_rates_16state(returns_history, signals_history):
    all_trend_perms = list(product([0, 1], repeat=2))
    all_ma_perms = list(product([0, 1], repeat=2))

    if len(returns_history) < MIN_SAMPLES_PER_STATE:
        return {(t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False}
                for t in all_trend_perms for m in all_ma_perms}

    common_idx = returns_history.index.intersection(signals_history.index)
    if len(common_idx) < MIN_SAMPLES_PER_STATE:
        return {(t, m): {'n': 0, 'hit_rate': 0.5, 'sufficient': False}
                for t in all_trend_perms for m in all_ma_perms}

    aligned_returns = returns_history.loc[common_idx]
    aligned_signals = signals_history.loc[common_idx]
    forward_returns = aligned_returns.shift(-1).iloc[:-1]
    aligned_signals = aligned_signals.iloc[:-1]

    hit_rates = {}
    for trend_perm in all_trend_perms:
        for ma_perm in all_ma_perms:
            mask = (
                (aligned_signals['trend_24h'] == trend_perm[0]) &
                (aligned_signals['trend_168h'] == trend_perm[1]) &
                (aligned_signals['ma72_above_ma24'] == ma_perm[0]) &
                (aligned_signals['ma168_above_ma24'] == ma_perm[1])
            )
            perm_returns = forward_returns[mask].dropna()
            n = len(perm_returns)
            hit_rate = (perm_returns > 0).sum() / n if n > 0 else 0.5

            hit_rates[(trend_perm, ma_perm)] = {
                'n': n, 'hit_rate': hit_rate,
                'sufficient': n >= MIN_SAMPLES_PER_STATE,
            }
    return hit_rates


def get_position_signal(state_key, hit_rates, allow_shorts: bool) -> float:
    """
    Get position signal from hit rates.
    Returns: +1.0 = long, -0.5 = short (SHORT_SIZE_SCALAR), 0.0 = flat
    """
    data = hit_rates.get(state_key, {'sufficient': False, 'hit_rate': 0.5})

    if not data['sufficient']:
        return 0.0

    if data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.0
    else:
        if allow_shorts:
            return -SHORT_SIZE_SCALAR
        else:
            return 0.0


def calculate_risk_parity_weights(returns_df):
    vols = returns_df.std() * np.sqrt(365)
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


# =============================================================================
# QUALITY FILTER — unchanged
# =============================================================================

def linear_quality_scalar(dist_ma168: float, direction: int) -> float:
    """
    Map distance-to-MA(168h) to a position sizing scalar.
    Longs:  dist ≤ -0.04 → 1.0, dist ≥ +0.05 → 0.5, between → linear
    Shorts: dist ≤ -0.04 → 0.3, dist ≥ +0.04 → 1.0, between → linear
    """
    if pd.isna(dist_ma168):
        return 1.0

    if direction > 0:  # Long
        if dist_ma168 <= QUALITY_LONG_FULL:
            return 1.0
        elif dist_ma168 >= QUALITY_LONG_MIN:
            return QUALITY_LONG_FLOOR
        else:
            frac = (dist_ma168 - QUALITY_LONG_FULL) / (QUALITY_LONG_MIN - QUALITY_LONG_FULL)
            return 1.0 - frac * (1.0 - QUALITY_LONG_FLOOR)

    elif direction < 0:  # Short
        if dist_ma168 <= QUALITY_SHORT_WEAK:
            return QUALITY_SHORT_FLOOR
        elif dist_ma168 >= QUALITY_SHORT_FULL:
            return 1.0
        else:
            frac = (dist_ma168 - QUALITY_SHORT_WEAK) / (QUALITY_SHORT_FULL - QUALITY_SHORT_WEAK)
            return QUALITY_SHORT_FLOOR + frac * (1.0 - QUALITY_SHORT_FLOOR)

    else:
        return 1.0


# =============================================================================
# DATA LOADING — returns both 1h and 24h data
# =============================================================================

def load_all_data(db: Database) -> Dict:
    """
    Load and prepare data at both resolutions:
      - 1h signals + features + returns: for hourly trading loop
      - 24h signals + returns: for hit rate calculation and risk parity
    """
    data = {}
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ", flush=True)
        df_1h = db.get_ohlcv(pair)

        # 1h resolution — for hourly trading loop
        signals_1h, features_1h = generate_signals_1h(df_1h)
        returns_1h = df_1h['close'].pct_change()

        # 24h resolution — for hit rate calculation + risk parity
        signals_24h, df_24h = generate_signals_24h(df_1h)
        returns_24h = df_24h['close'].pct_change()

        data[pair] = {
            'signals_1h': signals_1h,
            'features_1h': features_1h,
            'returns_1h': returns_1h,
            'signals_24h': signals_24h,
            'returns_24h': returns_24h,
            'prices_1h': df_1h,
            'prices_24h': df_24h,
        }
        n_hours = len(signals_1h)
        n_days = len(df_24h)
        print(f"{n_hours:,} hours / {n_days:,} days "
              f"({signals_1h.index[0].strftime('%Y-%m-%d')} to "
              f"{signals_1h.index[-1].strftime('%Y-%m-%d')})")
    return data


# =============================================================================
# BUY & HOLD BACKTEST — hourly cycle
# =============================================================================

def run_buy_hold(
    data: Dict,
    hourly_dates: List[pd.Timestamp],
) -> BacktestResult:
    """
    Buy & hold benchmark: equal weight, monthly rebalance.
    Runs on hourly cycle for fair comparison with strategy configs.
    """
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    equity_curve = {}
    total_costs = 0.0
    n_trades = 0

    n_pairs = len(DEPLOY_PAIRS)
    target_weight = 1.0 / n_pairs
    prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    last_rebalance_month = None

    # Yearly tracking
    yearly_pnl = {}
    yearly_start_equity = {}
    current_year = hourly_dates[0].year
    yearly_start_equity[current_year] = equity
    yearly_pnl[current_year] = 0.0
    equity_curve[hourly_dates[0]] = equity

    for i, ts in enumerate(hourly_dates[:-1]):
        next_ts = hourly_dates[i + 1]
        current_month = (ts.year, ts.month)

        if next_ts.year != current_year:
            current_year = next_ts.year
            yearly_start_equity[current_year] = equity
            yearly_pnl[current_year] = 0.0

        # Monthly rebalance to equal weight
        if last_rebalance_month != current_month:
            for pair in DEPLOY_PAIRS:
                curr_exp = target_weight
                prev_exp = prev_exposures[pair]
                trade_value = abs(curr_exp - prev_exp) * equity
                if trade_value >= MIN_TRADE_SIZE:
                    total_costs += trade_value * COST_PER_SIDE
                    n_trades += 1
                    prev_exposures[pair] = curr_exp
            last_rebalance_month = current_month

        # Calculate hourly PnL
        hourly_pnl = 0.0
        for pair in DEPLOY_PAIRS:
            exp = prev_exposures.get(pair, 0.0)
            returns_1h = data[pair]['returns_1h']
            if exp > 0 and next_ts in returns_1h.index:
                ret = returns_1h.loc[next_ts]
                if not pd.isna(ret):
                    hourly_pnl += equity * exp * ret

        equity += hourly_pnl
        yearly_pnl[next_ts.year] = yearly_pnl.get(next_ts.year, 0) + hourly_pnl
        equity_curve[next_ts] = equity

        if equity > peak_equity:
            peak_equity = equity

    # Metrics — Sharpe from daily-resampled equity
    equity_series = pd.Series(equity_curve)
    running_peak = equity_series.cummax()
    drawdowns = (running_peak - equity_series) / running_peak
    max_drawdown_raw = drawdowns.max()

    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (hourly_dates[-1] - hourly_dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Resample to daily for Sharpe calculation (avoids inflated hourly Sharpe)
    equity_daily = equity_series.resample('24h').last().dropna()
    returns_daily = equity_daily.pct_change().dropna()
    sharpe = (returns_daily.mean() / returns_daily.std() * np.sqrt(365)
              if returns_daily.std() > 0 else 0)
    calmar = annual_return / max_drawdown_raw if max_drawdown_raw > 0 else 0

    yearly_returns = {}
    for year in yearly_start_equity:
        if year in yearly_pnl:
            start_eq = yearly_start_equity[year]
            yearly_returns[year] = yearly_pnl[year] / start_eq if start_eq > 0 else 0

    return BacktestResult(
        name='Buy & Hold', config='BH',
        total_return=total_return, annual_return=annual_return,
        sharpe_ratio=sharpe, max_drawdown_raw=max_drawdown_raw, calmar_ratio=calmar,
        n_long_trades=n_trades, n_total_trades=n_trades,
        long_trading_costs=total_costs, total_costs=total_costs,
        equity_curve=equity_series, yearly_returns=yearly_returns,
        long_pnl=equity - INITIAL_CAPITAL + total_costs,
        long_hours=len(hourly_dates) * len(DEPLOY_PAIRS),
    )


# =============================================================================
# MAIN BACKTEST ENGINE — HOURLY CYCLE, LONG/SHORT WITH QUALITY FILTER
# =============================================================================

def run_backtest(
    data: Dict,
    hourly_dates: List[pd.Timestamp],
    data_start: pd.Timestamp,
    name: str = "Backtest",
    config: str = "",
    allow_shorts: bool = False,
    use_quality_filter: bool = False,
    use_dd_protection: bool = True,
    confirmation_hours: int = CONFIRMATION_HOURS,
    use_boost: bool = False,
) -> BacktestResult:
    """
    Run backtest with hourly cycle, optional shorts, quality filter,
    and configurable confirmation filter.

    Hourly cycle:
        Every 1h bar: compute state → apply filter → confirm → position → PnL
        Hit rates recalculated monthly from 24h data (statistical validity)
        Risk parity weights recalculated monthly from 24h returns

    Confirmation filter:
        confirmation_hours = 0: immediate state changes (no confirmation)
        confirmation_hours = N: new state must persist for N additional hours
        after initial detection before position changes.
    """
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    equity_curve = {}

    # Cost tracking
    long_trading_costs = 0.0
    short_trading_costs = 0.0
    short_rollover_costs = 0.0

    # Trade counts
    n_long_trades = 0
    n_short_trades = 0
    signals_filtered = 0
    signals_total = 0
    confirmations_cancelled = 0
    rebalances_suppressed = 0

    # PnL attribution
    long_pnl_total = 0.0
    short_pnl_gross_total = 0.0

    # Portfolio state — 24h returns for risk parity and vol scaling
    returns_dict_24h = {pair: data[pair]['returns_24h'] for pair in DEPLOY_PAIRS}
    returns_df_24h = pd.DataFrame(returns_dict_24h).dropna()
    asset_weights = pd.Series({pair: 1.0 / len(DEPLOY_PAIRS) for pair in DEPLOY_PAIRS})
    prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    last_rebalance_month = None

    # =========================================================================
    # STATE TRACKING (per pair)
    # =========================================================================
    # Raw state tracking (for NO_MA72_ONLY filter)
    prev_raw_states = {pair: None for pair in DEPLOY_PAIRS}

    # Confirmation filter state
    confirmed_states = {pair: None for pair in DEPLOY_PAIRS}
    pending_states = {pair: None for pair in DEPLOY_PAIRS}
    pending_hours_count = {pair: 0 for pair in DEPLOY_PAIRS}

    # Hit rate caches (recalculated monthly from 24h data)
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}

    # =========================================================================
    # DURATION TRACKING (in hours)
    # =========================================================================
    pair_direction = {pair: 0 for pair in DEPLOY_PAIRS}
    pair_regime_age = {pair: 0 for pair in DEPLOY_PAIRS}

    short_durations = []
    long_durations = []
    short_hours = 0
    long_hours = 0
    short_states_used = {}

    # Quality filter tracking
    long_scalars = []
    short_scalars = []
    quality_adjustments = 0

    # State duration tracking (for boost logic — tracks confirmed state integer)
    confirmed_state_ints = {pair: None for pair in DEPLOY_PAIRS}
    state_duration_hours = {pair: 0 for pair in DEPLOY_PAIRS}

    # Boost tracking
    boost_hours = 0
    boost_pnl_total = 0.0
    boost_activations = 0
    prev_boosted = {pair: False for pair in DEPLOY_PAIRS}

    # Yearly tracking
    yearly_pnl = {}
    yearly_start_equity = {}
    current_year = hourly_dates[0].year
    yearly_start_equity[current_year] = equity
    yearly_pnl[current_year] = 0.0
    equity_curve[hourly_dates[0]] = equity

    # =========================================================================
    # MAIN HOURLY LOOP
    # =========================================================================
    for i, ts in enumerate(hourly_dates[:-1]):
        next_ts = hourly_dates[i + 1]
        current_month = (ts.year, ts.month)

        # Year tracking
        if next_ts.year != current_year:
            current_year = next_ts.year
            yearly_start_equity[current_year] = equity
            yearly_pnl[current_year] = 0.0

        # Drawdown protection
        dd_scalar = 1.0
        if use_dd_protection:
            current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
            if current_dd >= DIR_DD_START_REDUCE:
                dd_scalar = 1.0
            elif current_dd <= DIR_DD_MIN_EXPOSURE:
                dd_scalar = DIR_MIN_EXPOSURE_FLOOR
            else:
                range_dd = DIR_DD_START_REDUCE - DIR_DD_MIN_EXPOSURE
                position = (current_dd - DIR_DD_MIN_EXPOSURE) / range_dd
                dd_scalar = DIR_MIN_EXPOSURE_FLOOR + position * (1.0 - DIR_MIN_EXPOSURE_FLOOR)

        # Monthly rebalance of risk parity weights (from 24h returns)
        if last_rebalance_month != current_month:
            lookback_start = get_month_start(ts, COV_LOOKBACK_MONTHS)
            lookback_end = get_month_start(ts, 0)
            lookback_returns = returns_df_24h.loc[lookback_start:lookback_end]
            if len(lookback_returns) >= 20:
                asset_weights = calculate_risk_parity_weights(lookback_returns)
            last_rebalance_month = current_month

        # =================================================================
        # CALCULATE EXPOSURES PER ASSET
        # =================================================================
        asset_exposures = {}
        total_long_exposure = 0.0
        total_short_exposure = 0.0

        for pair in DEPLOY_PAIRS:
            signals_1h = data[pair]['signals_1h']
            features_1h = data[pair]['features_1h']

            # 24h data for hit rate calculation
            signals_24h = data[pair]['signals_24h']
            returns_24h = data[pair]['returns_24h']

            if ts not in signals_1h.index:
                asset_exposures[pair] = 0.0
                continue

            sig = signals_1h.loc[ts]
            state_int = int(sig['state_int'])

            # Build raw 16-state tuple
            raw_state = (
                int(sig['trend_24h']), int(sig['trend_168h']),
                int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24'])
            )

            # =============================================================
            # CONFIRMATION FILTER
            # =============================================================
            prev_raw = prev_raw_states[pair]

            # Step 1: Check pending confirmation (before processing new transitions)
            if pending_states[pair] is not None:
                if raw_state == pending_states[pair]:
                    # State persisting — increment confirmation counter
                    pending_hours_count[pair] += 1
                    if pending_hours_count[pair] >= confirmation_hours:
                        # Confirmed! Update active state
                        confirmed_states[pair] = pending_states[pair]
                        pending_states[pair] = None
                        pending_hours_count[pair] = 0
                else:
                    # State changed away from pending — cancel confirmation
                    pending_states[pair] = None
                    pending_hours_count[pair] = 0
                    confirmations_cancelled += 1

            # Step 2: Check for new transition
            if raw_state != prev_raw:
                if prev_raw is not None:
                    signals_total += 1

                    # Apply NO_MA72_ONLY filter
                    should_trade = should_trade_signal_16state(
                        prev_raw, raw_state, True)

                    if not should_trade:
                        signals_filtered += 1
                    elif raw_state != confirmed_states[pair]:
                        # Filter passed, state is new (not already confirmed)
                        if confirmation_hours == 0:
                            # Immediate confirmation
                            confirmed_states[pair] = raw_state
                        else:
                            # Start confirmation countdown
                            pending_states[pair] = raw_state
                            pending_hours_count[pair] = 0
                else:
                    # First bar — initialise
                    if confirmation_hours == 0:
                        confirmed_states[pair] = raw_state
                    else:
                        pending_states[pair] = raw_state
                        pending_hours_count[pair] = 0

            prev_raw_states[pair] = raw_state

            # =============================================================
            # POSITION FROM CONFIRMED STATE
            # =============================================================
            active = confirmed_states[pair]
            if active is None:
                asset_exposures[pair] = 0.0
                continue

            trend_perm = (active[0], active[1])
            ma_perm = (active[2], active[3])
            state_key = (trend_perm, ma_perm)

            # Hit rate lookup (recalculated monthly from 24h data)
            if not has_minimum_training(ts, data_start, MIN_TRAINING_MONTHS):
                pos = 0.0
            else:
                current_m = (ts.year, ts.month)
                if last_recalc_month[pair] != current_m:
                    cutoff = get_month_start(ts, 0)
                    hist_ret = returns_24h[returns_24h.index < cutoff]
                    hist_sig = signals_24h[signals_24h.index < cutoff]
                    if len(hist_ret) > 0:
                        hit_rate_cache[pair] = calculate_expanding_hit_rates_16state(
                            hist_ret, hist_sig)
                        last_recalc_month[pair] = current_m
                pos = get_position_signal(state_key, hit_rate_cache[pair], allow_shorts)

            # =============================================================
            # STATE DURATION + BOOST LOGIC
            # =============================================================
            # Track how many consecutive hours this confirmed state integer
            # has been active (separate from direction duration tracking).
            active_int = active[0] * 8 + active[1] * 4 + active[2] * 2 + active[3] * 1

            if active_int == confirmed_state_ints[pair]:
                state_duration_hours[pair] += 1
            else:
                # State changed — reset duration
                confirmed_state_ints[pair] = active_int
                state_duration_hours[pair] = 1
                prev_boosted[pair] = False

            # Apply boost: State 11, duration >= threshold, long position
            is_boosted = False
            if (use_boost and
                active_int == BOOST_STATE and
                state_duration_hours[pair] >= BOOST_THRESHOLD_HOURS and
                pos > 0):
                pos *= BOOST_MULTIPLIER
                is_boosted = True
                boost_hours += 1
                if not prev_boosted[pair]:
                    boost_activations += 1
                    prev_boosted[pair] = True

            # =============================================================
            # DIRECTION DURATION TRACKING (in hours)
            # =============================================================
            if pos > 0:
                new_direction = 1
            elif pos < 0:
                new_direction = -1
            else:
                new_direction = 0

            old_direction = pair_direction[pair]

            if new_direction == old_direction:
                pair_regime_age[pair] += 1
            else:
                age = pair_regime_age[pair]
                if old_direction == -1 and age > 0:
                    short_durations.append(age)
                elif old_direction == 1 and age > 0:
                    long_durations.append(age)
                pair_direction[pair] = new_direction
                pair_regime_age[pair] = 1

            # =============================================================
            # QUALITY FILTER — MODULATE POSITION SIZE
            # =============================================================
            if use_quality_filter and pos != 0.0:
                dist = (features_1h.loc[ts, 'dist_ma168']
                        if ts in features_1h.index else np.nan)
                q_scalar = linear_quality_scalar(dist, new_direction)

                if q_scalar < 1.0:
                    quality_adjustments += 1

                pos *= q_scalar

                if new_direction > 0:
                    long_scalars.append(q_scalar)
                elif new_direction < 0:
                    short_scalars.append(q_scalar)

            # Track hours by direction
            if pos > 0:
                long_hours += 1
            elif pos < 0:
                short_hours += 1
                short_states_used[state_int] = short_states_used.get(state_int, 0) + 1

            # Apply risk parity weight
            weighted_pos = pos * asset_weights[pair]
            asset_exposures[pair] = weighted_pos

            if weighted_pos > 0:
                total_long_exposure += weighted_pos
            elif weighted_pos < 0:
                total_short_exposure += abs(weighted_pos)

        # Normalize long exposure if > 1.0
        if total_long_exposure > 1.0:
            long_scale = 1.0 / total_long_exposure
            for pair in asset_exposures:
                if asset_exposures[pair] > 0:
                    asset_exposures[pair] *= long_scale

        # Vol scaling (from 24h returns)
        vol_scalar = 1.0
        vol_lookback_start = get_month_start(ts, VOL_LOOKBACK_MONTHS)
        vol_lookback_end = get_month_start(ts, 0)
        vol_returns = returns_df_24h.loc[vol_lookback_start:vol_lookback_end]
        if len(vol_returns) >= 15:
            market_returns = vol_returns.mean(axis=1)
            realized_vol = market_returns.std() * np.sqrt(365)
            if realized_vol > 0:
                vol_scalar = DIR_TARGET_VOL / realized_vol
                vol_scalar = np.clip(vol_scalar, DIR_MIN_EXPOSURE_FLOOR, DIR_MAX_EXPOSURE)

        risk_scalar = min(vol_scalar, 1.0 / dd_scalar if dd_scalar > 0 else 1.0)
        risk_scalar = max(risk_scalar, DIR_MIN_EXPOSURE_FLOOR)

        for pair in asset_exposures:
            asset_exposures[pair] *= risk_scalar

        # =================================================================
        # EXECUTE TRADES + CALCULATE COSTS (hourly)
        # =================================================================
        hourly_pnl = 0.0
        hourly_long_cost = 0.0
        hourly_short_cost = 0.0
        hourly_rollover = 0.0
        hourly_long_pnl = 0.0
        hourly_short_pnl_gross = 0.0

        for pair in DEPLOY_PAIRS:
            curr_exp = asset_exposures.get(pair, 0.0)
            prev_exp = prev_exposures.get(pair, 0.0)

            # =============================================================
            # REBALANCE DEAD ZONE
            # Suppress micro-adjustments when direction is unchanged and
            # the relative exposure change is below threshold.
            # Direction changes (long↔short, long↔flat, etc.) ALWAYS execute.
            # =============================================================
            direction_changed = (
                (curr_exp > 0) != (prev_exp > 0) or   # sign flip
                (curr_exp < 0) != (prev_exp < 0) or   # sign flip
                (curr_exp == 0) != (prev_exp == 0)     # entering/exiting
            )

            if not direction_changed and prev_exp != 0:
                relative_change = abs(curr_exp - prev_exp) / abs(prev_exp)
                if relative_change < REBALANCE_THRESHOLD:
                    # Suppress: keep previous exposure, skip trade
                    curr_exp = prev_exp
                    asset_exposures[pair] = prev_exp
                    rebalances_suppressed += 1

            trade_value = abs(curr_exp - prev_exp) * equity

            if trade_value >= MIN_TRADE_SIZE:
                base_trade_cost = trade_value * COST_PER_SIDE

                entering_short = (curr_exp < 0 and prev_exp >= 0)
                increasing_short = (curr_exp < prev_exp < 0)

                if curr_exp < 0 or prev_exp < 0:
                    hourly_short_cost += base_trade_cost
                    n_short_trades += 1
                else:
                    hourly_long_cost += base_trade_cost
                    n_long_trades += 1

                if entering_short:
                    margin_fee = abs(curr_exp) * equity * KRAKEN_MARGIN_OPEN
                    hourly_short_cost += margin_fee
                elif increasing_short:
                    margin_fee = abs(curr_exp - prev_exp) * equity * KRAKEN_MARGIN_OPEN
                    hourly_short_cost += margin_fee

                prev_exposures[pair] = curr_exp
            else:
                curr_exp = prev_exp
                asset_exposures[pair] = prev_exp

            # PnL calculation (hourly return)
            returns_1h = data[pair]['returns_1h']
            if curr_exp != 0 and next_ts in returns_1h.index:
                ret = returns_1h.loc[next_ts]
                if not pd.isna(ret):
                    pnl = equity * curr_exp * ret
                    hourly_pnl += pnl

                    if curr_exp > 0:
                        hourly_long_pnl += pnl
                    else:
                        hourly_short_pnl_gross += pnl

            # Rollover cost on open short positions (per hour)
            if curr_exp < 0:
                rollover = abs(curr_exp) * equity * KRAKEN_ROLLOVER_HOUR
                hourly_rollover += rollover

        # Apply all costs
        total_hourly_cost = hourly_long_cost + hourly_short_cost + hourly_rollover
        hourly_pnl -= total_hourly_cost

        long_trading_costs += hourly_long_cost
        short_trading_costs += hourly_short_cost
        short_rollover_costs += hourly_rollover

        long_pnl_total += hourly_long_pnl
        short_pnl_gross_total += hourly_short_pnl_gross

        equity += hourly_pnl
        yearly_pnl[next_ts.year] = yearly_pnl.get(next_ts.year, 0) + hourly_pnl
        equity_curve[next_ts] = equity

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd

    # =====================================================================
    # CALCULATE FINAL METRICS
    # =====================================================================
    equity_series = pd.Series(equity_curve)
    running_peak = equity_series.cummax()
    drawdowns = (running_peak - equity_series) / running_peak
    max_drawdown_raw = drawdowns.max()

    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (hourly_dates[-1] - hourly_dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe from daily-resampled equity (avoids inflated hourly Sharpe)
    equity_daily = equity_series.resample('24h').last().dropna()
    returns_daily = equity_daily.pct_change().dropna()
    sharpe = (returns_daily.mean() / returns_daily.std() * np.sqrt(365)
              if returns_daily.std() > 0 else 0)
    calmar = annual_return / max_drawdown_raw if max_drawdown_raw > 0 else 0

    yearly_returns = {}
    for year in yearly_start_equity:
        if year in yearly_pnl:
            start_eq = yearly_start_equity[year]
            yearly_returns[year] = yearly_pnl[year] / start_eq if start_eq > 0 else 0

    total_costs = long_trading_costs + short_trading_costs + short_rollover_costs

    avg_short_dur = np.mean(short_durations) if short_durations else 0.0
    avg_long_dur = np.mean(long_durations) if long_durations else 0.0

    return BacktestResult(
        name=name, config=config,
        total_return=total_return, annual_return=annual_return,
        sharpe_ratio=sharpe, max_drawdown_raw=max_drawdown_raw, calmar_ratio=calmar,
        n_long_trades=n_long_trades, n_short_trades=n_short_trades,
        n_total_trades=n_long_trades + n_short_trades,
        long_trading_costs=long_trading_costs, short_trading_costs=short_trading_costs,
        short_rollover_costs=short_rollover_costs, total_costs=total_costs,
        signals_filtered=signals_filtered, signals_total=signals_total,
        confirmations_cancelled=confirmations_cancelled,
        rebalances_suppressed=rebalances_suppressed,
        short_hours=short_hours,
        short_states_used=short_states_used,
        avg_short_duration_hours=avg_short_dur,
        long_hours=long_hours, avg_long_duration_hours=avg_long_dur,
        equity_curve=equity_series, yearly_returns=yearly_returns,
        long_pnl=long_pnl_total,
        short_pnl=short_pnl_gross_total - short_trading_costs - short_rollover_costs,
        short_pnl_gross=short_pnl_gross_total,
        avg_long_scalar=np.mean(long_scalars) if long_scalars else 1.0,
        avg_short_scalar=np.mean(short_scalars) if short_scalars else 1.0,
        quality_adjustments=quality_adjustments,
        boost_hours=boost_hours,
        boost_activations=boost_activations,
    )


# =============================================================================
# DISPLAY
# =============================================================================

def display_result(result: BacktestResult):
    filter_pct = result.signals_filtered / result.signals_total * 100 if result.signals_total > 0 else 0

    print(f"""
    ═══════════════════════════════════════════════════════════════
    {result.name.upper()} [{result.config}]
    ═══════════════════════════════════════════════════════════════

    Performance:
    ─────────────────────────────────────────────────────────────────
    Total Return:         {result.total_return * 100:>+8.1f}%
    Annual Return:        {result.annual_return * 100:>+8.1f}%
    Sharpe Ratio:         {result.sharpe_ratio:>8.2f}  (from daily equity)
    Max Drawdown:         {result.max_drawdown_raw * 100:>8.1f}%
    Calmar Ratio:         {result.calmar_ratio:>8.2f}""")

    if result.config != 'BH':
        cfg = STRATEGY_CONFIGS.get(result.config, {})
        conf_hours = cfg.get('confirmation_hours', CONFIRMATION_HOURS)
        confirm_str = f"CONFIRMATION_HOURS={conf_hours}"
        boost_str = f", BOOST=S{BOOST_STATE}≥{BOOST_THRESHOLD_HOURS}h→{BOOST_MULTIPLIER}x" if cfg.get('use_boost', False) else ""
        print(f"""
    Trading Statistics:
    ─────────────────────────────────────────────────────────────────
    Long Trades:          {result.n_long_trades:>8,}
    Short Trades:         {result.n_short_trades:>8,}
    Total Trades:         {result.n_total_trades:>8,}
    Signals Filtered:     {result.signals_filtered:>8,} / {result.signals_total:>8,} ({filter_pct:.1f}%)
    Confirms Cancelled:   {result.confirmations_cancelled:>8,}
    Rebal. Suppressed:    {result.rebalances_suppressed:>8,}  (dead zone: {REBALANCE_THRESHOLD*100:.0f}%)
    Cycle:                    Hourly ({confirm_str}{boost_str})""")

        if result.boost_hours > 0:
            print(f"""
    Boost Statistics:
    ─────────────────────────────────────────────────────────────────
    Boost Hours:          {result.boost_hours:>8,}  ({result.boost_hours/24:.1f} days)
    Boost Activations:    {result.boost_activations:>8,}""")

    print(f"""
    Cost Breakdown (Kraken):
    ─────────────────────────────────────────────────────────────────
    Long Trading Costs:   ${result.long_trading_costs:>12,.0f}
    Short Trading Costs:  ${result.short_trading_costs:>12,.0f}
    Short Rollover Costs: ${result.short_rollover_costs:>12,.0f}
    Total Costs:          ${result.total_costs:>12,.0f}""")

    # Long position analysis
    if result.long_hours > 0:
        print(f"""
    Long Position Analysis:
    ─────────────────────────────────────────────────────────────────
    Long Pair-Hours:      {result.long_hours:>8,}  ({result.long_days:>8.1f} pair-days)
    Long PnL:             ${result.long_pnl:>12,.0f}
    Avg Long Duration:    {result.avg_long_duration_hours:>8.1f} hours ({result.avg_long_duration:>6.1f} days)""")

    # Short position analysis
    if result.short_hours > 0:
        print(f"""
    Short Position Analysis:
    ─────────────────────────────────────────────────────────────────
    Short Pair-Hours:     {result.short_hours:>8,}  ({result.short_days:>8.1f} pair-days)
    Short PnL (gross):    ${result.short_pnl_gross:>12,.0f}
    Short PnL (net):      ${result.short_pnl:>12,.0f}
    Avg Short Duration:   {result.avg_short_duration_hours:>8.1f} hours ({result.avg_short_duration:>6.1f} days)""")

        if result.short_states_used:
            print(f"\n    Short States Used:")
            for state in sorted(result.short_states_used.keys()):
                count = result.short_states_used[state]
                name = STATE_NAMES.get(state, '')
                hours = count
                print(f"      State {state:>2d} ({name:40s}): {hours:>6,} pair-hours ({hours/24:>7.1f} days)")

    # Quality filter stats
    if result.quality_adjustments > 0:
        print(f"""
    Quality Filter:
    ─────────────────────────────────────────────────────────────────
    Adjustments Made:     {result.quality_adjustments:>8,}
    Avg Long Scalar:      {result.avg_long_scalar:>8.3f}
    Avg Short Scalar:     {result.avg_short_scalar:>8.3f}""")

    if result.yearly_returns:
        print(f"""
    Yearly Returns:
    ─────────────────────────────────────────────────────────────────""")
        for year in sorted(result.yearly_returns.keys()):
            ret = result.yearly_returns[year]
            print(f"      {year}:  {ret * 100:>+8.1f}%")


def display_comparison(results: Dict[str, BacktestResult]):
    print("\n" + "=" * 130)
    print("CONFIGURATION COMPARISON")
    print("=" * 130)

    print(f"""
    ┌────────┬──────────────────────┬───────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────────────┐
    │ Config │ Name                 │ Tot.Ret   │ Ann.Ret  │ Sharpe   │ Max DD   │ Calmar   │ Trades   │ Total Costs       │
    ├────────┼──────────────────────┼───────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────────────────┤""")

    baseline_sharpe = results.get('A', results[list(results.keys())[0]]).sharpe_ratio

    for config in RUN_CONFIGS:
        if config not in results:
            continue
        r = results[config]
        sharpe_diff = r.sharpe_ratio - baseline_sharpe
        diff_str = f"({sharpe_diff:+.2f})" if config != 'A' else ""
        cost_str = f"${r.total_costs:>10,.0f}"

        print(f"    │ {config:6s} │ {r.name:20s} │ "
              f"{r.total_return * 100:>+8.1f}% │ {r.annual_return * 100:>+7.1f}% │ "
              f"{r.sharpe_ratio:>5.2f} {diff_str:>6s} │ {r.max_drawdown_raw * 100:>7.1f}% │ "
              f"{r.calmar_ratio:>7.2f} │ {r.n_total_trades:>8,} │ {cost_str:>17s} │")

    print(f"    └────────┴──────────────────────┴───────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────────────┘")

    # Long/Short PnL attribution
    any_shorts = any(results.get(c, BacktestResult(name='')).short_hours > 0 for c in RUN_CONFIGS)
    if any_shorts:
        print(f"""
    LONG/SHORT PnL ATTRIBUTION:
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    {'Config':<8s} {'Long PnL':>12s} {'L Days':>8s} {'Avg L Dur':>10s} {'Short Gross':>12s} {'S Costs':>12s} {'Short Net':>12s} {'S Days':>8s} {'Avg S Dur':>10s} {'Avg L Scl':>10s} {'Avg S Scl':>10s}
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────""")

        for config in RUN_CONFIGS:
            if config not in results:
                continue
            r = results[config]
            short_costs = r.short_trading_costs + r.short_rollover_costs
            print(f"    {config:<8s} ${r.long_pnl:>11,.0f} {r.long_days:>8.1f} {r.avg_long_duration:>9.1f}d "
                  f"${r.short_pnl_gross:>11,.0f} ${short_costs:>11,.0f} ${r.short_pnl:>11,.0f} "
                  f"{r.short_days:>8.1f} {r.avg_short_duration:>9.1f}d "
                  f"{r.avg_long_scalar:>9.3f} {r.avg_short_scalar:>9.3f}")

    # Yearly comparison
    print(f"""
    YEARLY RETURNS:
    ─────────────────────────────────────────────────────────────────────────────────────────""")

    all_years = sorted(set(y for r in results.values() for y in r.yearly_returns.keys()))
    header = f"    {'Year':>6s}"
    for config in RUN_CONFIGS:
        if config in results:
            header += f"  {config:>8s}"
    print(header)
    print(f"    {'─' * (8 + 10 * len(results))}")

    for year in all_years:
        row = f"    {year:>6d}"
        for config in RUN_CONFIGS:
            if config in results:
                r = results[config]
                ret = r.yearly_returns.get(year, 0)
                row += f"  {ret * 100:>+7.1f}%"
        print(row)

    # Quality filter impact (C vs B)
    if 'C' in results and 'B' in results:
        b = results['B']
        c = results['C']
        sharpe_delta = c.sharpe_ratio - b.sharpe_ratio
        return_delta = c.annual_return - b.annual_return
        dd_delta = c.max_drawdown_raw - b.max_drawdown_raw
        cost_delta = c.total_costs - b.total_costs
        short_net_delta = c.short_pnl - b.short_pnl
        long_delta = c.long_pnl - b.long_pnl

        print(f"""
    QUALITY FILTER IMPACT (C vs B):
    ─────────────────────────────────────────────────────────────────────────────────────────
      Sharpe:      {sharpe_delta:+.3f}  ({b.sharpe_ratio:.2f} → {c.sharpe_ratio:.2f})
      Ann. Return: {return_delta * 100:+.1f}pp  ({b.annual_return * 100:.1f}% → {c.annual_return * 100:.1f}%)
      Max DD:      {dd_delta * 100:+.1f}pp  ({b.max_drawdown_raw * 100:.1f}% → {c.max_drawdown_raw * 100:.1f}%)
      Total Costs: ${cost_delta:>+12,.0f}
      Long PnL:    ${long_delta:>+12,.0f}
      Short Net:   ${short_net_delta:>+12,.0f}
      Avg Scalars: Long={c.avg_long_scalar:.3f}  Short={c.avg_short_scalar:.3f}""")

    # Confirmation + Boost impact (D vs C)
    if 'D' in results and 'C' in results:
        c = results['C']
        d = results['D']
        sharpe_delta = d.sharpe_ratio - c.sharpe_ratio
        return_delta = d.annual_return - c.annual_return
        dd_delta = d.max_drawdown_raw - c.max_drawdown_raw
        cost_delta = d.total_costs - c.total_costs

        print(f"""
    CONFIRMATION + BOOST IMPACT (D vs C):
    ─────────────────────────────────────────────────────────────────────────────────────────
      Sharpe:      {sharpe_delta:+.3f}  ({c.sharpe_ratio:.2f} → {d.sharpe_ratio:.2f})
      Ann. Return: {return_delta * 100:+.1f}pp  ({c.annual_return * 100:.1f}% → {d.annual_return * 100:.1f}%)
      Max DD:      {dd_delta * 100:+.1f}pp  ({c.max_drawdown_raw * 100:.1f}% → {d.max_drawdown_raw * 100:.1f}%)
      Total Costs: ${cost_delta:>+12,.0f}
      Boost Hours: {d.boost_hours:>8,}  ({d.boost_hours/24:.1f} days)
      Boost Acts:  {d.boost_activations:>8,}
      Confirmation: 3h  |  Boost: S{BOOST_STATE}≥{BOOST_THRESHOLD_HOURS}h→{BOOST_MULTIPLIER}x""")

    # Boost-only impact (E vs C) — isolates boost without confirmation
    if 'E' in results and 'C' in results:
        c = results['C']
        e = results['E']
        sharpe_delta = e.sharpe_ratio - c.sharpe_ratio
        return_delta = e.annual_return - c.annual_return
        dd_delta = e.max_drawdown_raw - c.max_drawdown_raw
        cost_delta = e.total_costs - c.total_costs

        print(f"""
    BOOST-ONLY IMPACT (E vs C):
    ─────────────────────────────────────────────────────────────────────────────────────────
      Sharpe:      {sharpe_delta:+.3f}  ({c.sharpe_ratio:.2f} → {e.sharpe_ratio:.2f})
      Ann. Return: {return_delta * 100:+.1f}pp  ({c.annual_return * 100:.1f}% → {e.annual_return * 100:.1f}%)
      Max DD:      {dd_delta * 100:+.1f}pp  ({c.max_drawdown_raw * 100:.1f}% → {e.max_drawdown_raw * 100:.1f}%)
      Total Costs: ${cost_delta:>+12,.0f}
      Boost Hours: {e.boost_hours:>8,}  ({e.boost_hours/24:.1f} days)
      Boost Acts:  {e.boost_activations:>8,}
      Boost: S{BOOST_STATE}≥{BOOST_THRESHOLD_HOURS}h→{BOOST_MULTIPLIER}x  |  Confirmation: NONE""")

    # Confirmation-only impact (D vs E) — isolates confirmation effect
    if 'D' in results and 'E' in results:
        e = results['E']
        d = results['D']
        sharpe_delta = d.sharpe_ratio - e.sharpe_ratio
        return_delta = d.annual_return - e.annual_return
        dd_delta = d.max_drawdown_raw - e.max_drawdown_raw

        print(f"""
    CONFIRMATION-ONLY IMPACT (D vs E):
    ─────────────────────────────────────────────────────────────────────────────────────────
      Sharpe:      {sharpe_delta:+.3f}  ({e.sharpe_ratio:.2f} → {d.sharpe_ratio:.2f})
      Ann. Return: {return_delta * 100:+.1f}pp  ({e.annual_return * 100:.1f}% → {d.annual_return * 100:.1f}%)
      Max DD:      {dd_delta * 100:+.1f}pp  ({e.max_drawdown_raw * 100:.1f}% → {d.max_drawdown_raw * 100:.1f}%)
      Effect:      Isolates 3h confirmation filter (boost held constant)""")

    # Best config
    best_config = max(results.keys(), key=lambda k: results[k].sharpe_ratio)
    best = results[best_config]
    best_cfg = STRATEGY_CONFIGS.get(best_config, {})
    best_conf = best_cfg.get('confirmation_hours', CONFIRMATION_HOURS)
    best_boost = "Yes" if best_cfg.get('use_boost', False) else "No"
    print(f"""
    ═══════════════════════════════════════════════════════════════════════════════════════════
    BEST CONFIGURATION: {best_config} ({best.name})
    ─────────────────────────────────────────────────────────────────────────────────────────
    Sharpe: {best.sharpe_ratio:.2f}  |  Return: {best.annual_return * 100:+.1f}%  |  Max DD: {best.max_drawdown_raw * 100:.1f}%  |  Calmar: {best.calmar_ratio:.2f}
    Sharpe improvement over long-only (A): {best.sharpe_ratio - baseline_sharpe:+.2f}
    Cycle: Hourly | Confirmation: {best_conf}h | Boost: {best_boost}
    ═══════════════════════════════════════════════════════════════════════════════════════════""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 130)
    print("LONG/SHORT BACKTEST v03 — HOURLY CYCLE WITH CONFIRMATION FILTER")
    print("=" * 130)

    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  LONG/SHORT STRATEGY — HOURLY CYCLE                                                                                    ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                         ║
    ║  16-state regime model | 6 pairs | Expanding-window hit rates | Kraken cost model                                      ║
    ║                                                                                                                         ║
    ║  CYCLE: Hourly (1h bars) — signals detect MA crossovers at any hour                                                    ║
    ║  CONFIRMATION: {CONFIRMATION_HOURS}h — new state must persist {CONFIRMATION_HOURS} additional hour(s) before position change                          ║
    ║  HIT RATES: Calculated from 24h bars (statistical validity, unchanged from v02)                                        ║
    ║  SHARPE: Computed from daily-resampled equity (comparable to v02)                                                      ║
    ║                                                                                                                         ║
    ║  POSITION LOGIC:                                                                                                        ║
    ║    hit_rate > 0.50 + sufficient samples → LONG  (+1.0 × risk parity weight)                                            ║
    ║    hit_rate ≤ 0.50 + sufficient samples → SHORT (-0.5 × risk parity weight)                                            ║
    ║    insufficient samples                 → FLAT  (no position)                                                           ║
    ║                                                                                                                         ║
    ║  QUALITY FILTER (Config C, D):                                                                                         ║
    ║    Distance to MA(168h) modulates position sizing within regime direction.                                               ║
    ║    Rebalance dead zone: {REBALANCE_THRESHOLD*100:.0f}% — suppress micro-adjustments below this relative change threshold                    ║
    ║                                                                                                                         ║
    ║  CONFIRMATION + BOOST (Config D):                                                                                       ║
    ║    Confirmation: 3h — new state must persist 3 hours before position change                                             ║
    ║    Boost: State {BOOST_STATE} held ≥{BOOST_THRESHOLD_HOURS}h → {BOOST_MULTIPLIER}× position multiplier                                                          ║
    ║                                                                                                                         ║
    ║  KRAKEN COSTS:                                                                                                          ║
    ║    Long:  {COST_PER_SIDE*100:.2f}% per side = {COST_PER_SIDE*2*100:.2f}% round trip                                                                    ║
    ║    Short: {COST_PER_SIDE*100:.2f}% per side + {KRAKEN_MARGIN_OPEN*100:.2f}% margin open + {KRAKEN_ROLLOVER_HOUR*100:.4f}%/hour rollover                                          ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"  Configurations to test: {', '.join(RUN_CONFIGS)}")
    print(f"  Default confirmation filter: {CONFIRMATION_HOURS} hours (Config D uses 3h)")
    print(f"  Rebalance dead zone: {REBALANCE_THRESHOLD*100:.0f}%")
    print(f"  Boost: State {BOOST_STATE} ≥{BOOST_THRESHOLD_HOURS}h → {BOOST_MULTIPLIER}× (Config D only)")
    print()

    # Connect to database
    print("  Connecting to database...")
    db = Database()

    # Load data (both 1h and 24h resolutions)
    print("\n  Loading data + features (1h + 24h)...")
    data = load_all_data(db)

    # Find common hourly timestamps across all pairs
    all_hours = None
    for pair in DEPLOY_PAIRS:
        hours = data[pair]['signals_1h'].index
        if all_hours is None:
            all_hours = set(hours)
        else:
            all_hours = all_hours.intersection(set(hours))

    hourly_dates = sorted(list(all_hours))
    data_start = hourly_dates[0]
    years = (hourly_dates[-1] - hourly_dates[0]).days / 365.25
    n_days = (hourly_dates[-1] - hourly_dates[0]).days
    print(f"\n  Common timestamps: {len(hourly_dates):,} hours ({n_days:,} days)")
    print(f"  Period: {hourly_dates[0].strftime('%Y-%m-%d %H:%M')} to "
          f"{hourly_dates[-1].strftime('%Y-%m-%d %H:%M')} = {years:.1f} years")

    # Run each configuration
    results = {}

    for config in RUN_CONFIGS:
        cfg = STRATEGY_CONFIGS[config]

        print(f"\n{'=' * 70}")
        print(f"  RUNNING CONFIG {config}: {cfg['name']}")
        print(f"  {cfg['description']}")
        print(f"{'=' * 70}")

        if cfg['is_buy_hold']:
            result = run_buy_hold(data=data, hourly_dates=hourly_dates)
        else:
            result = run_backtest(
                data=data,
                hourly_dates=hourly_dates,
                data_start=data_start,
                name=cfg['name'],
                config=config,
                allow_shorts=cfg['allow_shorts'],
                use_quality_filter=cfg['use_quality_filter'],
                confirmation_hours=cfg.get('confirmation_hours', CONFIRMATION_HOURS),
                use_boost=cfg.get('use_boost', False),
            )

        results[config] = result
        display_result(result)

    # Show comparison
    if len(results) > 1:
        display_comparison(results)

    print("\n" + "=" * 130)
    print("BACKTEST COMPLETE")
    print("=" * 130)

    return results


if __name__ == "__main__":
    results = main()