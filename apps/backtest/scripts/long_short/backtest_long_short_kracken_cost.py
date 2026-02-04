#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long/Short Backtest with Kraken Cost Model
=============================================
Full backtest comparing long-only vs long+short strategies
using the monthly MA(25) context filter and regime-based short signals.

Kraken Cost Model:
    LONG:  0.26% trading + 0.10% slippage per side = 0.72% round trip
    SHORT: 0.26% trading + 0.10% slippage per side + 0.02% margin open
           + 0.12%/day rollover (0.02% per 4h × 6)

Short Candidates (from regime profitability research):
    Tier 1 (extreme): States 8, 10, 12, 14
        - Avg return -1.6% to -2.5%/day, hit rate 66-71%, BE < 0.5 days
    Tier 2 (moderate): States 13, 15
        - Avg return -0.8% to -1.4%/day, hit rate 57-62%, BE ~0.6-1.1 days
    Tier 3 (weak): States 4, 6
        - Avg return -0.8%/day, hit rate 57%, BE ~1.1 days

Configurations:
    A = Baseline: 16-state long only (no monthly filter), Kraken costs
    B = Long + monthly MA(25) filter, Kraken costs, no shorts
    C = Long + monthly filter + Tier 1 shorts (8, 10, 12, 14)
    D = Long + monthly filter + Tier 1+2 shorts (8, 10, 12, 13, 14, 15)
    E = Long + monthly filter + All strong shorts (4, 6, 8, 10, 12, 13, 14, 15)

Usage:
    Open in Thonny, hit Run.
"""

from pathlib import Path
import sys
import os

try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
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
# THONNY / IDE CONFIGURATION
# =============================================================================
RUN_CONFIGS = ['A', 'B', 'C', 'D', 'E']

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# =============================================================================
# MA PARAMETERS (locked from validation)
# =============================================================================
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

# Monthly MA (Phase 2 winner — robust across P20-P50)
MONTHLY_MA_PERIOD = 25
MONTHLY_UPDATE_FREQ = '168h'

# Hysteresis buffers (validated)
ENTRY_BUFFER = 0.025
EXIT_BUFFER = 0.005

# =============================================================================
# HIT RATE / PORTFOLIO PARAMETERS
# =============================================================================
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

# =============================================================================
# KRAKEN COST MODEL
# =============================================================================
KRAKEN_TRADING_FEE = 0.0026     # 0.26% taker per side
KRAKEN_SLIPPAGE = 0.0010        # 0.10% estimated per side
KRAKEN_MARGIN_OPEN = 0.0002     # 0.02% one-time on short entry
KRAKEN_ROLLOVER_4H = 0.0002     # 0.02% per 4 hours
KRAKEN_ROLLOVER_DAY = KRAKEN_ROLLOVER_4H * 6  # 0.12% per day

# Per-side cost (entry or exit)
COST_PER_SIDE = KRAKEN_TRADING_FEE + KRAKEN_SLIPPAGE  # 0.36%

# =============================================================================
# SHORT CONFIGURATION
# =============================================================================
SHORT_SIZE_SCALAR = 0.50  # Shorts at 50% of equivalent long sizing

# Short candidate tiers (from regime profitability research)
TIER_1_STATES = {8, 10, 12, 14}         # Extreme: exhaustion + bull traps
TIER_2_STATES = {13, 15}                 # Moderate: bullish 16-state, monthly disagrees
TIER_3_STATES = {4, 6}                   # Weak: bearish 24h, bullish 168h

# Integer state encoding: state = trend_24h*8 + trend_168h*4 + ma72>ma24*2 + ma168>ma24

# =============================================================================
# STRATEGY CONFIGURATIONS
# =============================================================================
STRATEGY_CONFIGS = {
    'A': {
        'name': 'Long only (no filter)',
        'has_monthly': False,
        'short_states': set(),
        'description': 'Baseline: 16-state, Kraken costs, no monthly filter',
    },
    'B': {
        'name': 'Long + monthly P25',
        'has_monthly': True,
        'short_states': set(),
        'description': 'Monthly MA(25) filter: bearish = flat',
    },
    'C': {
        'name': 'L+S Tier 1 (8,10,12,14)',
        'has_monthly': True,
        'short_states': TIER_1_STATES,
        'description': 'Monthly filter + Tier 1 shorts: exhaustion & bull traps',
    },
    'D': {
        'name': 'L+S Tier 1+2',
        'has_monthly': True,
        'short_states': TIER_1_STATES | TIER_2_STATES,
        'description': 'Monthly filter + Tier 1+2 shorts (8,10,12,13,14,15)',
    },
    'E': {
        'name': 'L+S All strong',
        'has_monthly': True,
        'short_states': TIER_1_STATES | TIER_2_STATES | TIER_3_STATES,
        'description': 'Monthly filter + All strong shorts (4,6,8,10,12,13,14,15)',
    },
}

# State labels
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
    # Monthly filter
    monthly_bullish_pct: float = 0.0
    monthly_bearish_pct: float = 0.0
    disagreement_count: int = 0
    filtered_by_monthly: int = 0
    # Short stats
    short_days: int = 0          # Total pair-days spent short
    short_states_used: Dict = field(default_factory=dict)  # {state: count}
    # Series
    equity_curve: Optional[pd.Series] = None
    yearly_returns: Dict[int, float] = field(default_factory=dict)
    # Long/short PnL attribution
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    short_pnl_gross: float = 0.0   # Before short costs


# =============================================================================
# SIGNAL GENERATION
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


def generate_signals_with_monthly(
    df_1h: pd.DataFrame,
    include_monthly: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 16-state signals + monthly context + integer state."""
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')

    # 24h MA
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    trend_24h = label_trend_binary_rolling(df_24h['close'], ma_24h, ENTRY_BUFFER, EXIT_BUFFER)

    # 72h MA
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')

    # 168h MA
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    trend_168h_raw = label_trend_binary_rolling(
        df_168h['close'], df_168h['close'].rolling(MA_PERIOD_168H).mean(),
        ENTRY_BUFFER, EXIT_BUFFER
    )

    # Build signals
    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_168h'] = trend_168h_raw.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)

    # Monthly MA
    if include_monthly:
        ma_hours = MONTHLY_MA_PERIOD * 24
        ma_monthly_hourly = df_1h['close'].rolling(ma_hours).mean()
        ma_monthly_at_freq = ma_monthly_hourly.resample(MONTHLY_UPDATE_FREQ).last()
        close_at_freq = df_1h['close'].resample(MONTHLY_UPDATE_FREQ).last()
        monthly_trend_freq = (close_at_freq > ma_monthly_at_freq).astype(int).dropna()
        monthly_trend = monthly_trend_freq.reindex(df_24h.index, method='ffill')
        aligned['monthly_trend'] = monthly_trend.shift(1)

    signals = aligned.dropna().astype(int)

    # Add integer state for fast short lookups
    signals['state_int'] = (
        signals['trend_24h'] * 8 +
        signals['trend_168h'] * 4 +
        signals['ma72_above_ma24'] * 2 +
        signals['ma168_above_ma24'] * 1
    )

    return signals, df_24h


# =============================================================================
# SIGNAL FILTER (NO_MA72_ONLY — unchanged)
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
# HIT RATE CALCULATION (unchanged)
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


def get_position(state_key, hit_rates):
    data = hit_rates.get(state_key, {'sufficient': False, 'hit_rate': 0.5})
    if not data['sufficient']:
        return 0.50
    elif data['hit_rate'] > HIT_RATE_THRESHOLD:
        return 1.00
    else:
        return 0.00


def calculate_risk_parity_weights(returns_df):
    vols = returns_df.std() * np.sqrt(365)
    vols[vols == 0] = 1e-6
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database, include_monthly: bool = True) -> Dict:
    data = {}
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ", flush=True)
        df_1h = db.get_ohlcv(pair)
        signals, df_24h = generate_signals_with_monthly(df_1h, include_monthly=include_monthly)
        returns = df_24h['close'].pct_change()
        data[pair] = {'prices': df_24h, 'signals': signals, 'returns': returns}
        print(f"{len(df_24h)} days")
    return data


# =============================================================================
# BACKTEST ENGINE — LONG/SHORT WITH KRAKEN COSTS
# =============================================================================

def run_backtest(
    data: Dict,
    dates: List[pd.Timestamp],
    data_start: pd.Timestamp,
    name: str = "Backtest",
    config: str = "",
    has_monthly: bool = False,
    short_states: Set[int] = None,
    use_dd_protection: bool = True,
) -> BacktestResult:
    """
    Run backtest with optional monthly filter and short positions.

    Position logic:
        - 16-state model determines long signals (unchanged)
        - Monthly bearish → override longs to flat
        - Monthly bearish + state in short_states → SHORT at 50% sizing
        - Monthly bullish → normal 16-state longs
    """
    if short_states is None:
        short_states = set()

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

    # State tracking
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

    # Monthly filter tracking
    monthly_bullish_days = 0
    monthly_bearish_days = 0
    disagreement_count = 0
    filtered_by_monthly = 0

    # Short tracking
    short_days = 0
    short_states_used = {}

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

        # =====================================================================
        # CALCULATE EXPOSURES PER ASSET
        # =====================================================================
        asset_exposures = {}
        total_long_exposure = 0.0
        total_short_exposure = 0.0

        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']

            if date not in signals.index:
                asset_exposures[pair] = 0.0
                continue

            sig = signals.loc[date]

            # Build 16-state
            current_state = (
                int(sig['trend_24h']), int(sig['trend_168h']),
                int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24'])
            )
            state_int = int(sig['state_int'])

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

            # Get position from 16-state model
            trend_perm = (active[0], active[1])
            ma_perm = (active[2], active[3])
            state_key = (trend_perm, ma_perm)

            if not has_minimum_training(date, data_start, MIN_TRAINING_MONTHS):
                pos = 0.50
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
                pos = get_position(state_key, hit_rate_cache[pair])

            # =================================================================
            # MONTHLY CONTEXT + SHORT LOGIC
            # =================================================================
            if has_monthly and 'monthly_trend' in sig.index:
                monthly_trend = int(sig['monthly_trend'])

                if monthly_trend == 1:
                    monthly_bullish_days += 1
                    # Monthly bullish — normal long operation
                else:
                    monthly_bearish_days += 1

                    # Track disagreement
                    if pos > 0.5:
                        disagreement_count += 1
                        filtered_by_monthly += 1

                    # SHORT CHECK: Is this state a short candidate?
                    if state_int in short_states:
                        # Short at reduced sizing
                        pos = -SHORT_SIZE_SCALAR
                        # Track which states we're shorting
                        short_states_used[state_int] = short_states_used.get(state_int, 0) + 1
                    else:
                        # Monthly bearish but not a short candidate → flat
                        pos = 0.0

            # Apply risk parity weight
            weighted_pos = pos * asset_weights[pair]
            asset_exposures[pair] = weighted_pos

            if weighted_pos > 0:
                total_long_exposure += weighted_pos
            else:
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

        # =====================================================================
        # EXECUTE TRADES + CALCULATE COSTS
        # =====================================================================
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
                # ---------------------------------------------------------
                # COST CALCULATION
                # ---------------------------------------------------------
                base_trade_cost = trade_value * COST_PER_SIDE

                # Determine if we're entering/exiting shorts
                entering_short = (curr_exp < 0 and prev_exp >= 0)
                increasing_short = (curr_exp < prev_exp < 0)
                exiting_short = (curr_exp >= 0 and prev_exp < 0)
                exiting_long = (curr_exp <= 0 and prev_exp > 0)
                entering_long = (curr_exp > 0 and prev_exp <= 0)
                adjusting_long = (curr_exp > prev_exp > 0) or (0 < curr_exp < prev_exp)

                # Base trade cost applies to all trades
                if curr_exp < 0 or prev_exp < 0:
                    # Short-related trade
                    daily_short_cost += base_trade_cost
                    n_short_trades += 1
                else:
                    # Long-only trade
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

            # ---------------------------------------------------------
            # PnL CALCULATION
            # ---------------------------------------------------------
            if curr_exp != 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    pnl = equity * curr_exp * ret
                    daily_pnl += pnl

                    if curr_exp > 0:
                        daily_long_pnl += pnl
                    else:
                        daily_short_pnl_gross += pnl

            # ---------------------------------------------------------
            # ROLLOVER COST (daily charge on open short positions)
            # ---------------------------------------------------------
            if curr_exp < 0:
                rollover = abs(curr_exp) * equity * KRAKEN_ROLLOVER_DAY
                daily_rollover += rollover
                short_days += 1

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

    # Yearly returns
    yearly_returns = {}
    for year in yearly_start_equity:
        if year in yearly_pnl:
            start_eq = yearly_start_equity[year]
            yearly_returns[year] = yearly_pnl[year] / start_eq if start_eq > 0 else 0

    # Monthly stats
    total_monthly_days = monthly_bullish_days + monthly_bearish_days
    monthly_bullish_pct = monthly_bullish_days / total_monthly_days * 100 if total_monthly_days > 0 else 0
    monthly_bearish_pct = monthly_bearish_days / total_monthly_days * 100 if total_monthly_days > 0 else 0

    total_costs = long_trading_costs + short_trading_costs + short_rollover_costs

    return BacktestResult(
        name=name, config=config,
        total_return=total_return, annual_return=annual_return,
        sharpe_ratio=sharpe, max_drawdown_raw=max_drawdown_raw, calmar_ratio=calmar,
        n_long_trades=n_long_trades, n_short_trades=n_short_trades,
        n_total_trades=n_long_trades + n_short_trades,
        long_trading_costs=long_trading_costs, short_trading_costs=short_trading_costs,
        short_rollover_costs=short_rollover_costs, total_costs=total_costs,
        signals_filtered=signals_filtered, signals_total=signals_total,
        monthly_bullish_pct=monthly_bullish_pct, monthly_bearish_pct=monthly_bearish_pct,
        disagreement_count=disagreement_count, filtered_by_monthly=filtered_by_monthly,
        short_days=short_days, short_states_used=short_states_used,
        equity_curve=equity_series, yearly_returns=yearly_returns,
        long_pnl=long_pnl_total, short_pnl=short_pnl_gross_total - short_trading_costs - short_rollover_costs,
        short_pnl_gross=short_pnl_gross_total,
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

    Performance Metrics:
    ─────────────────────────────────────────────────────────────────
    Annual Return:        {result.annual_return * 100:>+8.1f}%
    Sharpe Ratio:         {result.sharpe_ratio:>8.2f}
    Max Drawdown:         {result.max_drawdown_raw * 100:>8.1f}%
    Calmar Ratio:         {result.calmar_ratio:>8.2f}

    Trading Statistics:
    ─────────────────────────────────────────────────────────────────
    Long Trades:          {result.n_long_trades:>8,}
    Short Trades:         {result.n_short_trades:>8,}
    Total Trades:         {result.n_total_trades:>8,}
    Signals Filtered:     {result.signals_filtered:>8,} / {result.signals_total:>8,} ({filter_pct:.1f}%)

    Cost Breakdown (Kraken):
    ─────────────────────────────────────────────────────────────────
    Long Trading Costs:   ${result.long_trading_costs:>12,.0f}
    Short Trading Costs:  ${result.short_trading_costs:>12,.0f}
    Short Rollover Costs: ${result.short_rollover_costs:>12,.0f}
    Total Costs:          ${result.total_costs:>12,.0f}""")

    if result.short_days > 0:
        print(f"""
    Short Position Analysis:
    ─────────────────────────────────────────────────────────────────
    Short Pair-Days:      {result.short_days:>8,}
    Short PnL (gross):    ${result.short_pnl_gross:>12,.0f}
    Short PnL (net):      ${result.short_pnl:>12,.0f}
    Long PnL:             ${result.long_pnl:>12,.0f}""")

        if result.short_states_used:
            print(f"\n    Short States Used:")
            for state in sorted(result.short_states_used.keys()):
                count = result.short_states_used[state]
                print(f"      State {state:>2d} ({STATE_NAMES.get(state, ''):40s}): {count:>5,} pair-days")

    if result.monthly_bullish_pct > 0 or result.monthly_bearish_pct > 0:
        print(f"""
    Monthly Filter:
    ─────────────────────────────────────────────────────────────────
    Time Bullish:         {result.monthly_bullish_pct:>7.1f}%
    Time Bearish:         {result.monthly_bearish_pct:>7.1f}%
    Disagreements:        {result.disagreement_count:>8,}
    Positions Filtered:   {result.filtered_by_monthly:>8,}""")

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

    baseline_sharpe = results.get('A', results[list(results.keys())[0]]).sharpe_ratio

    print(f"""
    ┌────────┬───────────────────────────┬──────────┬───────────────┬──────────┬──────────┬──────────┬──────────┬───────────────────────┐
    │ Config │ Name                      │ Ann.Ret  │ Sharpe        │ Max DD   │ Calmar   │ L Trades │ S Trades │ Costs                 │
    ├────────┼───────────────────────────┼──────────┼───────────────┼──────────┼──────────┼──────────┼──────────┼───────────────────────┤""")

    for config in RUN_CONFIGS:
        if config not in results:
            continue
        r = results[config]
        sharpe_diff = r.sharpe_ratio - baseline_sharpe
        diff_str = f"({sharpe_diff:+.2f})" if config != 'A' else ""
        cost_str = f"${r.total_costs:>10,.0f}"

        print(f"    │ {config:6s} │ {r.name:25s} │ {r.annual_return * 100:>+7.1f}% │ "
              f"{r.sharpe_ratio:>5.2f} {diff_str:>7s} │ {r.max_drawdown_raw * 100:>7.1f}% │ "
              f"{r.calmar_ratio:>7.2f} │ {r.n_long_trades:>8,} │ {r.n_short_trades:>8,} │ {cost_str:>21s} │")

    print(f"    └────────┴───────────────────────────┴──────────┴───────────────┴──────────┴──────────┴──────────┴──────────┴───────────────────────┘")

    # Short PnL comparison
    any_shorts = any(r.short_days > 0 for r in results.values())
    if any_shorts:
        print(f"""
    SHORT PnL ATTRIBUTION:
    ─────────────────────────────────────────────────────────────────────────────────────────
    {'Config':<8s} {'Long PnL':>12s} {'Short Gross':>12s} {'Short Costs':>12s} {'Short Net':>12s} {'Short Days':>12s}
    ─────────────────────────────────────────────────────────────────────────────────────────""")

        for config in RUN_CONFIGS:
            if config not in results:
                continue
            r = results[config]
            short_costs = r.short_trading_costs + r.short_rollover_costs
            print(f"    {config:<8s} ${r.long_pnl:>11,.0f} ${r.short_pnl_gross:>11,.0f} "
                  f"${short_costs:>11,.0f} ${r.short_pnl:>11,.0f} {r.short_days:>11,}")

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

    # Best config
    best_config = max(results.keys(), key=lambda k: results[k].sharpe_ratio)
    best = results[best_config]
    print(f"""
    BEST CONFIGURATION: {best_config} ({best.name})
    ─────────────────────────────────────────────────────────────────
    Sharpe: {best.sharpe_ratio:.2f}  |  Return: {best.annual_return * 100:+.1f}%  |  Max DD: {best.max_drawdown_raw * 100:.1f}%  |  Calmar: {best.calmar_ratio:.2f}
    Sharpe improvement over baseline: {best.sharpe_ratio - baseline_sharpe:+.2f}""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 130)
    print("LONG/SHORT BACKTEST WITH KRAKEN COST MODEL")
    print("=" * 130)

    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  LONG/SHORT STRATEGY COMPARISON                                                                                        ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                         ║
    ║  Monthly MA(25) @ 168h update | 16-state model | 6 pairs | Kraken cost model                                           ║
    ║                                                                                                                         ║
    ║  KRAKEN COSTS:                                                                                                          ║
    ║    Long:  0.26% trading + 0.10% slippage per side = 0.72% round trip                                                   ║
    ║    Short: 0.26% trading + 0.10% slippage per side + 0.02% margin open + 0.12%/day rollover                             ║
    ║                                                                                                                         ║
    ║  CONFIGURATIONS:                                                                                                        ║
    ║    A = Baseline: long only, no monthly filter, Kraken costs                                                             ║
    ║    B = Long + monthly MA(25) filter (bearish = flat), no shorts                                                         ║
    ║    C = Long + monthly filter + Tier 1 shorts (states 8, 10, 12, 14)                                                    ║
    ║    D = Long + monthly filter + Tier 1+2 shorts (+ states 13, 15)                                                       ║
    ║    E = Long + monthly filter + All strong shorts (+ states 4, 6)                                                        ║
    ║                                                                                                                         ║
    ║  SHORT SIZING: 50% of equivalent long weight | Only when monthly bearish                                                ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"  Configurations to test: {', '.join(RUN_CONFIGS)}")
    print()

    # Connect to database
    print("  Connecting to database...")
    db = Database()

    # Load data once (all configs use the same data, just different logic)
    print("\n  Loading data (with monthly MA)...")
    data_with_monthly = load_all_data(db, include_monthly=True)

    # Also need data without monthly for baseline
    print("\n  Loading data (without monthly MA for baseline)...")
    data_no_monthly = load_all_data(db, include_monthly=False)

    # Find common dates (use the monthly dataset as reference)
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data_with_monthly[pair]['signals'].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))

    # Also intersect with no-monthly dates
    for pair in DEPLOY_PAIRS:
        dates = data_no_monthly[pair]['signals'].index
        all_dates = all_dates.intersection(set(dates))

    dates = sorted(list(all_dates))
    data_start = dates[0]
    print(f"\n  Common dates: {len(dates)} ({dates[0].date()} to {dates[-1].date()})")

    # Run each configuration
    results = {}

    for config in RUN_CONFIGS:
        cfg = STRATEGY_CONFIGS[config]

        print(f"\n{'=' * 70}")
        print(f"  RUNNING CONFIG {config}: {cfg['name']}")
        print(f"  {cfg['description']}")
        if cfg['short_states']:
            states_str = ', '.join(str(s) for s in sorted(cfg['short_states']))
            print(f"  Short states: [{states_str}]")
        print(f"{'=' * 70}")

        # Select appropriate data
        use_data = data_with_monthly if cfg['has_monthly'] else data_no_monthly

        result = run_backtest(
            data=use_data,
            dates=dates,
            data_start=data_start,
            name=cfg['name'],
            config=config,
            has_monthly=cfg['has_monthly'],
            short_states=cfg['short_states'],
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