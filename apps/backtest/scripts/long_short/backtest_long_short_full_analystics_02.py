#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long/Short Backtest with Quality Filter
=========================================
Production backtest: long-only, long/short, and long/short with linear
quality filter. Includes buy-and-hold benchmark.

Position Logic:
    LONG:  hit_rate > 0.50 (sufficient samples) → long at risk-parity weight
    SHORT: hit_rate ≤ 0.50 (sufficient samples) → short at 50% sizing
    FLAT:  insufficient samples → no position

Quality Filter (Config C):
    Modulates position SIZE based on distance to MA(168h).
    Longs:  near/below MA168 → full size (1.0), overextended → half (0.5)
    Shorts: oversold below MA168 → minimal (0.3), exhaustion above → full (1.0)
    Regime system determines DIRECTION. Quality filter modulates SIZING.

Kraken Cost Model:
    LONG:  0.26% trading + 0.10% slippage per side = 0.72% round trip
    SHORT: 0.26% trading + 0.10% slippage per side + 0.02% margin open
           + 0.12%/day rollover (0.02% per 4h × 6)

Configurations:
    BH = Buy & hold — equal weight, monthly rebalance
    A  = Long-only baseline — hit_rate > 0.50 → long, else flat
    B  = Long/Short — hit_rate ≤ 0.50 + sufficient → short at 50%
    C  = Long/Short + Quality Filter — dist_ma168 sizing modulation

Usage:
    python backtest_long_short_full_analytics_02.py
"""

from pathlib import Path
import sys
import os

try:
    PROJECT_ROOT = Path(__file__).parent
    # Walk up until we find the cryptobot package
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
RUN_CONFIGS = ['BH', 'A', 'B', 'C']

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
KRAKEN_ROLLOVER_DAY = KRAKEN_ROLLOVER_4H * 6  # 0.12% per day
COST_PER_SIDE = KRAKEN_TRADING_FEE + KRAKEN_SLIPPAGE  # 0.36%

# Short configuration
SHORT_SIZE_SCALAR = 0.50    # Shorts at 50% of equivalent long sizing

# Quality filter thresholds (from quintile analysis of dist_ma168)
# Longs: near/below MA168 = high quality, overextended = low quality
QUALITY_LONG_FULL = -0.04       # dist_ma168 ≤ this → scalar 1.0
QUALITY_LONG_MIN = 0.05         # dist_ma168 ≥ this → scalar 0.5
QUALITY_LONG_FLOOR = 0.50       # minimum scalar for longs
# Shorts: oversold = low quality (shorts lose), exhaustion = high quality
QUALITY_SHORT_WEAK = -0.04      # dist_ma168 ≤ this → scalar 0.3
QUALITY_SHORT_FULL = 0.04       # dist_ma168 ≥ this → scalar 1.0
QUALITY_SHORT_FLOOR = 0.30      # minimum scalar for shorts

# Strategy configurations
STRATEGY_CONFIGS = {
    'BH': {
        'name': 'Buy & Hold',
        'allow_shorts': False,
        'use_quality_filter': False,
        'is_buy_hold': True,
        'description': 'Equal weight, monthly rebalance, fully invested',
    },
    'A': {
        'name': 'Long Only',
        'allow_shorts': False,
        'use_quality_filter': False,
        'is_buy_hold': False,
        'description': '16-state, hit_rate > 0.50 → long, else flat',
    },
    'B': {
        'name': 'Long/Short',
        'allow_shorts': True,
        'use_quality_filter': False,
        'is_buy_hold': False,
        'description': 'L/S: hit_rate ≤ 0.50 + sufficient → short at 50%, hold until flip',
    },
    'C': {
        'name': 'L/S + Quality',
        'allow_shorts': True,
        'use_quality_filter': True,
        'is_buy_hold': False,
        'description': 'L/S + quality filter: dist_ma168 modulates position sizing',
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
    # Short stats
    short_days: int = 0
    short_states_used: Dict = field(default_factory=dict)
    avg_short_duration: float = 0.0
    # Long stats
    long_days: int = 0
    avg_long_duration: float = 0.0
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


# =============================================================================
# SIGNAL GENERATION + FEATURE ENGINEERING
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


def generate_signals_and_features(df_1h: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate 16-state signals AND dist_ma168 feature for quality filter.

    Returns:
        signals:  DataFrame with binary state components + state_int
        df_24h:   Resampled 24h OHLCV
        features: DataFrame with dist_ma168 (aligned to 24h bars)
    """
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')

    # Moving averages
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()

    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')

    # Binary trend signals
    trend_24h = label_trend_binary_rolling(df_24h['close'], ma_24h, ENTRY_BUFFER, EXIT_BUFFER)
    trend_168h_raw = label_trend_binary_rolling(
        df_168h['close'], df_168h['close'].rolling(MA_PERIOD_168H).mean(),
        ENTRY_BUFFER, EXIT_BUFFER
    )

    # Build signal DataFrame
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

    # =====================================================================
    # QUALITY FILTER FEATURE: distance to MA(168h)
    # =====================================================================
    feat = pd.DataFrame(index=df_24h.index)
    close = df_24h['close']
    feat['dist_ma168'] = ((close - ma_168h_aligned) / ma_168h_aligned).shift(1)
    features = feat.reindex(signals.index)

    return signals, df_24h, features


# =============================================================================
# SIGNAL FILTER (NO_MA72_ONLY)
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
# HIT RATE CALCULATION
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

    Returns:
        +1.0  = long
        -0.5  = short (SHORT_SIZE_SCALAR)
         0.0  = flat (insufficient data OR bearish without shorts)
    """
    data = hit_rates.get(state_key, {'sufficient': False, 'hit_rate': 0.5})

    if not data['sufficient']:
        return 0.0  # Insufficient data → flat (never trade without evidence)

    if data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.0  # Bullish → long
    else:
        if allow_shorts:
            return -SHORT_SIZE_SCALAR  # Bearish + sufficient → short
        else:
            return 0.0  # Bearish without shorts → flat


def calculate_risk_parity_weights(returns_df):
    vols = returns_df.std() * np.sqrt(365)
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


# =============================================================================
# QUALITY FILTER
# =============================================================================

def linear_quality_scalar(dist_ma168: float, direction: int) -> float:
    """
    Map distance-to-MA(168h) to a position sizing scalar.

    Derived from quintile analysis showing monotonic relationship between
    dist_ma168 and forward returns within Long/Short macro states.

    Longs:  dist ≤ -0.04 → 1.0 (near MA, fresh regime, full conviction)
            dist ≥ +0.05 → 0.5 (overextended, reduce exposure)
            Between → linear interpolation

    Shorts: dist ≤ -0.04 → 0.3 (oversold, shorts lose, minimal exposure)
            dist ≥ +0.04 → 1.0 (exhaustion, shorts profitable, full conviction)
            Between → linear interpolation
    """
    if pd.isna(dist_ma168):
        return 1.0  # No data → no adjustment

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
        return 1.0  # Flat → no adjustment


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database) -> Dict:
    data = {}
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ", flush=True)
        df_1h = db.get_ohlcv(pair)
        signals, df_24h, features = generate_signals_and_features(df_1h)
        returns = df_24h['close'].pct_change()
        data[pair] = {
            'prices': df_24h,
            'signals': signals,
            'returns': returns,
            'features': features,
        }
        print(f"{len(df_24h)} days ({signals.index[0].date()} to {signals.index[-1].date()})")
    return data


# =============================================================================
# BUY & HOLD BACKTEST
# =============================================================================

def run_buy_hold(
    data: Dict,
    dates: List[pd.Timestamp],
) -> BacktestResult:
    """
    Buy & hold benchmark: equal weight across all pairs, monthly rebalance.
    No regime detection, no vol scaling, no DD protection.
    Uses Kraken cost model for rebalance trades.
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
    current_year = dates[0].year
    yearly_start_equity[current_year] = equity
    yearly_pnl[current_year] = 0.0
    equity_curve[dates[0]] = equity

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        current_month = (date.year, date.month)

        if next_date.year != current_year:
            current_year = next_date.year
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

        # Calculate PnL
        daily_pnl = 0.0
        for pair in DEPLOY_PAIRS:
            exp = prev_exposures.get(pair, 0.0)
            if exp > 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    daily_pnl += equity * exp * ret

        equity += daily_pnl
        yearly_pnl[next_date.year] = yearly_pnl.get(next_date.year, 0) + daily_pnl
        equity_curve[next_date] = equity

        if equity > peak_equity:
            peak_equity = equity

    # Metrics
    equity_series = pd.Series(equity_curve)
    running_peak = equity_series.cummax()
    drawdowns = (running_peak - equity_series) / running_peak
    max_drawdown_raw = drawdowns.max()

    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    returns_series = equity_series.pct_change().dropna()
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(365) if returns_series.std() > 0 else 0
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
        long_days=len(dates) * len(DEPLOY_PAIRS),
    )


# =============================================================================
# MAIN BACKTEST ENGINE — LONG/SHORT WITH QUALITY FILTER
# =============================================================================

def run_backtest(
    data: Dict,
    dates: List[pd.Timestamp],
    data_start: pd.Timestamp,
    name: str = "Backtest",
    config: str = "",
    allow_shorts: bool = False,
    use_quality_filter: bool = False,
    use_dd_protection: bool = True,
) -> BacktestResult:
    """
    Run backtest with optional short positions and quality filter.

    Position logic:
        hit_rate > 0.50 + sufficient → long (+1.0)
        hit_rate ≤ 0.50 + sufficient → short (-SHORT_SIZE_SCALAR) if allow_shorts
        insufficient data           → flat (0.0)

    Quality filter (when enabled):
        After direction is determined, pos *= linear_quality_scalar(dist_ma168, direction)
        This reduces sizing for low-quality entries while preserving direction.
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

    # PnL attribution
    long_pnl_total = 0.0
    short_pnl_gross_total = 0.0

    # Portfolio state
    returns_dict = {pair: data[pair]['returns'] for pair in DEPLOY_PAIRS}
    returns_df = pd.DataFrame(returns_dict).dropna()
    asset_weights = pd.Series({pair: 1.0 / len(DEPLOY_PAIRS) for pair in DEPLOY_PAIRS})
    prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    last_rebalance_month = None

    # Filter state tracking
    prev_signal_states = {pair: None for pair in DEPLOY_PAIRS}
    active_states = {pair: None for pair in DEPLOY_PAIRS}

    # Hit rate caches
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}

    # =========================================================================
    # DURATION TRACKING PER PAIR
    # =========================================================================
    pair_direction = {pair: 0 for pair in DEPLOY_PAIRS}
    pair_regime_age = {pair: 0 for pair in DEPLOY_PAIRS}

    short_durations = []
    long_durations = []
    short_days = 0
    long_days = 0
    short_states_used = {}

    # =========================================================================
    # QUALITY FILTER TRACKING
    # =========================================================================
    long_scalars = []
    short_scalars = []
    quality_adjustments = 0

    # Yearly tracking
    yearly_pnl = {}
    yearly_start_equity = {}
    current_year = dates[0].year
    yearly_start_equity[current_year] = equity
    yearly_pnl[current_year] = 0.0
    equity_curve[dates[0]] = equity

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        current_month = (date.year, date.month)

        # Year tracking
        if next_date.year != current_year:
            current_year = next_date.year
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

        # Monthly rebalance of risk parity weights
        if last_rebalance_month != current_month:
            lookback_start = get_month_start(date, COV_LOOKBACK_MONTHS)
            lookback_end = get_month_start(date, 0)
            lookback_returns = returns_df.loc[lookback_start:lookback_end]
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
            signals = data[pair]['signals']
            returns = data[pair]['returns']
            features = data[pair]['features']

            if date not in signals.index:
                asset_exposures[pair] = 0.0
                continue

            sig = signals.loc[date]
            state_int = int(sig['state_int'])

            # Build 16-state tuple
            current_state = (
                int(sig['trend_24h']), int(sig['trend_168h']),
                int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24'])
            )

            # Apply NO_MA72_ONLY filter
            prev_state = prev_signal_states[pair]
            if current_state != prev_state:
                signals_total += 1
                should_trade = should_trade_signal_16state(prev_state, current_state, True)
                if should_trade:
                    active_states[pair] = current_state
                else:
                    signals_filtered += 1
            prev_signal_states[pair] = current_state

            # Use active state for position
            active = active_states[pair]
            if active is None:
                active = current_state
                active_states[pair] = current_state

            # Get position from 16-state hit rates
            trend_perm = (active[0], active[1])
            ma_perm = (active[2], active[3])
            state_key = (trend_perm, ma_perm)

            if not has_minimum_training(date, data_start, MIN_TRAINING_MONTHS):
                pos = 0.0  # No trading without minimum training data
            else:
                current_m = (date.year, date.month)
                if last_recalc_month[pair] != current_m:
                    cutoff = get_month_start(date, 0)
                    hist_ret = returns[returns.index < cutoff]
                    hist_sig = signals[signals.index < cutoff]
                    if len(hist_ret) > 0:
                        hit_rate_cache[pair] = calculate_expanding_hit_rates_16state(
                            hist_ret, hist_sig)
                        last_recalc_month[pair] = current_m
                pos = get_position_signal(state_key, hit_rate_cache[pair], allow_shorts)

            # =============================================================
            # DURATION TRACKING
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
                dist = features.loc[date, 'dist_ma168'] if date in features.index else np.nan
                q_scalar = linear_quality_scalar(dist, new_direction)

                if q_scalar < 1.0:
                    quality_adjustments += 1

                pos *= q_scalar

                if new_direction > 0:
                    long_scalars.append(q_scalar)
                elif new_direction < 0:
                    short_scalars.append(q_scalar)

            # Track days by direction
            if pos > 0:
                long_days += 1
            elif pos < 0:
                short_days += 1
                short_states_used[state_int] = short_states_used.get(state_int, 0) + 1

            # Apply risk parity weight
            weighted_pos = pos * asset_weights[pair]
            asset_exposures[pair] = weighted_pos

            if weighted_pos > 0:
                total_long_exposure += weighted_pos
            elif weighted_pos < 0:
                total_short_exposure += abs(weighted_pos)

        # Normalize long exposure if > 1.0 (preserve short sizing separately)
        if total_long_exposure > 1.0:
            long_scale = 1.0 / total_long_exposure
            for pair in asset_exposures:
                if asset_exposures[pair] > 0:
                    asset_exposures[pair] *= long_scale

        # Vol scaling
        vol_scalar = 1.0
        if date in returns_df.index:
            vol_lookback_start = get_month_start(date, VOL_LOOKBACK_MONTHS)
            vol_lookback_end = get_month_start(date, 0)
            vol_returns = returns_df.loc[vol_lookback_start:vol_lookback_end]
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
        # EXECUTE TRADES + CALCULATE COSTS
        # =================================================================
        daily_pnl = 0.0
        daily_long_cost = 0.0
        daily_short_cost = 0.0
        daily_rollover = 0.0
        daily_long_pnl = 0.0
        daily_short_pnl_gross = 0.0

        for pair in DEPLOY_PAIRS:
            curr_exp = asset_exposures.get(pair, 0.0)
            prev_exp = prev_exposures.get(pair, 0.0)

            trade_value = abs(curr_exp - prev_exp) * equity

            if trade_value >= MIN_TRADE_SIZE:
                # Cost calculation
                base_trade_cost = trade_value * COST_PER_SIDE

                # Determine trade type for cost attribution
                entering_short = (curr_exp < 0 and prev_exp >= 0)
                increasing_short = (curr_exp < prev_exp < 0)

                if curr_exp < 0 or prev_exp < 0:
                    daily_short_cost += base_trade_cost
                    n_short_trades += 1
                else:
                    daily_long_cost += base_trade_cost
                    n_long_trades += 1

                # Margin opening fee for new/increased shorts
                if entering_short:
                    margin_fee = abs(curr_exp) * equity * KRAKEN_MARGIN_OPEN
                    daily_short_cost += margin_fee
                elif increasing_short:
                    margin_fee = abs(curr_exp - prev_exp) * equity * KRAKEN_MARGIN_OPEN
                    daily_short_cost += margin_fee

                prev_exposures[pair] = curr_exp
            else:
                curr_exp = prev_exp
                asset_exposures[pair] = prev_exp

            # PnL calculation
            if curr_exp != 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    pnl = equity * curr_exp * ret
                    daily_pnl += pnl

                    if curr_exp > 0:
                        daily_long_pnl += pnl
                    else:
                        daily_short_pnl_gross += pnl

            # Rollover cost on open short positions
            if curr_exp < 0:
                rollover = abs(curr_exp) * equity * KRAKEN_ROLLOVER_DAY
                daily_rollover += rollover

        # Apply all costs
        total_daily_cost = daily_long_cost + daily_short_cost + daily_rollover
        daily_pnl -= total_daily_cost

        long_trading_costs += daily_long_cost
        short_trading_costs += daily_short_cost
        short_rollover_costs += daily_rollover

        long_pnl_total += daily_long_pnl
        short_pnl_gross_total += daily_short_pnl_gross

        equity += daily_pnl
        yearly_pnl[next_date.year] = yearly_pnl.get(next_date.year, 0) + daily_pnl
        equity_curve[next_date] = equity

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
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    returns_series = equity_series.pct_change().dropna()
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(365) if returns_series.std() > 0 else 0
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
        short_days=short_days,
        short_states_used=short_states_used,
        avg_short_duration=avg_short_dur,
        long_days=long_days, avg_long_duration=avg_long_dur,
        equity_curve=equity_series, yearly_returns=yearly_returns,
        long_pnl=long_pnl_total,
        short_pnl=short_pnl_gross_total - short_trading_costs - short_rollover_costs,
        short_pnl_gross=short_pnl_gross_total,
        avg_long_scalar=np.mean(long_scalars) if long_scalars else 1.0,
        avg_short_scalar=np.mean(short_scalars) if short_scalars else 1.0,
        quality_adjustments=quality_adjustments,
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
    Sharpe Ratio:         {result.sharpe_ratio:>8.2f}
    Max Drawdown:         {result.max_drawdown_raw * 100:>8.1f}%
    Calmar Ratio:         {result.calmar_ratio:>8.2f}""")

    if result.config != 'BH':
        print(f"""
    Trading Statistics:
    ─────────────────────────────────────────────────────────────────
    Long Trades:          {result.n_long_trades:>8,}
    Short Trades:         {result.n_short_trades:>8,}
    Total Trades:         {result.n_total_trades:>8,}
    Signals Filtered:     {result.signals_filtered:>8,} / {result.signals_total:>8,} ({filter_pct:.1f}%)""")

    print(f"""
    Cost Breakdown (Kraken):
    ─────────────────────────────────────────────────────────────────
    Long Trading Costs:   ${result.long_trading_costs:>12,.0f}
    Short Trading Costs:  ${result.short_trading_costs:>12,.0f}
    Short Rollover Costs: ${result.short_rollover_costs:>12,.0f}
    Total Costs:          ${result.total_costs:>12,.0f}""")

    # Long position analysis
    if result.long_days > 0:
        print(f"""
    Long Position Analysis:
    ─────────────────────────────────────────────────────────────────
    Long Pair-Days:       {result.long_days:>8,}
    Long PnL:             ${result.long_pnl:>12,.0f}
    Avg Long Duration:    {result.avg_long_duration:>8.1f} days""")

    # Short position analysis
    if result.short_days > 0:
        print(f"""
    Short Position Analysis:
    ─────────────────────────────────────────────────────────────────
    Short Pair-Days:      {result.short_days:>8,}
    Short PnL (gross):    ${result.short_pnl_gross:>12,.0f}
    Short PnL (net):      ${result.short_pnl:>12,.0f}
    Avg Short Duration:   {result.avg_short_duration:>8.1f} days""")

        if result.short_states_used:
            print(f"\n    Short States Used:")
            for state in sorted(result.short_states_used.keys()):
                count = result.short_states_used[state]
                name = STATE_NAMES.get(state, '')
                print(f"      State {state:>2d} ({name:40s}): {count:>5,} pair-days")

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

    # Main comparison table
    print(f"""
    ┌────────┬─────────────────┬───────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────────────┐
    │ Config │ Name            │ Tot.Ret   │ Ann.Ret  │ Sharpe   │ Max DD   │ Calmar   │ Trades   │ Total Costs       │
    ├────────┼─────────────────┼───────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────────────────┤""")

    baseline_sharpe = results.get('A', results[list(results.keys())[0]]).sharpe_ratio

    for config in RUN_CONFIGS:
        if config not in results:
            continue
        r = results[config]
        sharpe_diff = r.sharpe_ratio - baseline_sharpe
        diff_str = f"({sharpe_diff:+.2f})" if config != 'A' else ""
        cost_str = f"${r.total_costs:>10,.0f}"

        print(f"    │ {config:6s} │ {r.name:15s} │ "
              f"{r.total_return * 100:>+8.1f}% │ {r.annual_return * 100:>+7.1f}% │ "
              f"{r.sharpe_ratio:>5.2f} {diff_str:>6s} │ {r.max_drawdown_raw * 100:>7.1f}% │ "
              f"{r.calmar_ratio:>7.2f} │ {r.n_total_trades:>8,} │ {cost_str:>17s} │")

    print(f"    └────────┴─────────────────┴───────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────────────┘")

    # Long/Short PnL attribution
    any_shorts = any(results.get(c, BacktestResult(name='')).short_days > 0 for c in RUN_CONFIGS)
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
            print(f"    {config:<8s} ${r.long_pnl:>11,.0f} {r.long_days:>8,} {r.avg_long_duration:>9.1f}d "
                  f"${r.short_pnl_gross:>11,.0f} ${short_costs:>11,.0f} ${r.short_pnl:>11,.0f} "
                  f"{r.short_days:>8,} {r.avg_short_duration:>9.1f}d "
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

    # Best config
    best_config = max(results.keys(), key=lambda k: results[k].sharpe_ratio)
    best = results[best_config]
    print(f"""
    ═══════════════════════════════════════════════════════════════════════════════════════════
    BEST CONFIGURATION: {best_config} ({best.name})
    ─────────────────────────────────────────────────────────────────────────────────────────
    Sharpe: {best.sharpe_ratio:.2f}  |  Return: {best.annual_return * 100:+.1f}%  |  Max DD: {best.max_drawdown_raw * 100:.1f}%  |  Calmar: {best.calmar_ratio:.2f}
    Sharpe improvement over long-only (A): {best.sharpe_ratio - baseline_sharpe:+.2f}
    ═══════════════════════════════════════════════════════════════════════════════════════════""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 130)
    print("LONG/SHORT BACKTEST WITH QUALITY FILTER")
    print("=" * 130)

    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  LONG/SHORT STRATEGY WITH QUALITY FILTER                                                                                ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                         ║
    ║  16-state regime model | 6 pairs | Expanding-window hit rates | Kraken cost model                                      ║
    ║                                                                                                                         ║
    ║  POSITION LOGIC:                                                                                                        ║
    ║    hit_rate > 0.50 + sufficient samples → LONG  (+1.0 × risk parity weight)                                            ║
    ║    hit_rate ≤ 0.50 + sufficient samples → SHORT (-0.5 × risk parity weight)                                            ║
    ║    insufficient samples                 → FLAT  (no position)                                                           ║
    ║                                                                                                                         ║
    ║  QUALITY FILTER (Config C):                                                                                             ║
    ║    Distance to MA(168h) modulates position sizing within regime direction.                                               ║
    ║    Longs:  near/below MA168 → full size (1.0), overextended → half (0.5)                                               ║
    ║    Shorts: oversold below MA168 → minimal (0.3), exhaustion above → full (1.0)                                         ║
    ║                                                                                                                         ║
    ║  KRAKEN COSTS:                                                                                                          ║
    ║    Long:  {COST_PER_SIDE*100:.2f}% per side = {COST_PER_SIDE*2*100:.2f}% round trip                                                                    ║
    ║    Short: {COST_PER_SIDE*100:.2f}% per side + {KRAKEN_MARGIN_OPEN*100:.2f}% margin open + {KRAKEN_ROLLOVER_DAY*100:.2f}%/day rollover                                              ║
    ║                                                                                                                         ║
    ║  CONFIGURATIONS:                                                                                                        ║
    ║    BH = Buy & hold (equal weight, monthly rebalance)                                                                    ║
    ║    A  = Long-only (hit_rate > 0.50 → long, else flat)                                                                  ║
    ║    B  = Long/Short (hit_rate ≤ 0.50 + sufficient → short at 50%, hold until flip)                                      ║
    ║    C  = Long/Short + Quality Filter (dist_ma168 sizing modulation)                                                      ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"  Configurations to test: {', '.join(RUN_CONFIGS)}")
    print()

    # Connect to database
    print("  Connecting to database...")
    db = Database()

    # Load data once (all configs use the same signals + features)
    print("\n  Loading data + features...")
    data = load_all_data(db)

    # Find common dates
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))

    dates = sorted(list(all_dates))
    data_start = dates[0]
    years = (dates[-1] - dates[0]).days / 365.25
    print(f"\n  Common dates: {len(dates)} ({dates[0].date()} to {dates[-1].date()}) = {years:.1f} years")

    # Run each configuration
    results = {}

    for config in RUN_CONFIGS:
        cfg = STRATEGY_CONFIGS[config]

        print(f"\n{'=' * 70}")
        print(f"  RUNNING CONFIG {config}: {cfg['name']}")
        print(f"  {cfg['description']}")
        print(f"{'=' * 70}")

        if cfg['is_buy_hold']:
            result = run_buy_hold(data=data, dates=dates)
        else:
            result = run_backtest(
                data=data,
                dates=dates,
                data_start=data_start,
                name=cfg['name'],
                config=config,
                allow_shorts=cfg['allow_shorts'],
                use_quality_filter=cfg['use_quality_filter'],
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