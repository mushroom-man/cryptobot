#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward Validation — Long/Short Strategy
================================================
Validates the short state selection out-of-sample.

The regime profitability research (regime_short_research.py) identified
short candidates using the FULL dataset. This script tests whether those
same states would have been selected using only historical data at each
rebalancing point.

Walk-Forward Approach:
    1. Start with minimum 18 months of training data
    2. Every quarter, using ONLY data available at that point:
       - For each of 16 states, compute bearish-context statistics
       - Select short candidates based on criteria (hit rate, avg return, samples)
    3. Trade the next quarter using dynamically selected short states
    4. Step forward and repeat

Configurations:
    FIXED-C = Fixed Tier 1 shorts (8,10,12,14) — from full-sample research
    FIXED-E = Fixed All Strong shorts (4,6,8,10,12,13,14,15)
    WF-S    = Walk-forward strict: HR > 55%, avg ret < -0.10%, n ≥ 30
    WF-M    = Walk-forward moderate: HR > 52%, avg ret < -0.05%, n ≥ 20

Key Output:
    - Performance comparison: walk-forward vs fixed selection
    - State selection history: which states were selected at each quarter
    - Selection stability: how often do states enter/exit the short set
    - If WF matches fixed performance → robust, not overfit

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
RUN_CONFIGS = ['FIXED-C', 'FIXED-E', 'WF-S', 'WF-M']

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# =============================================================================
# MA PARAMETERS (locked)
# =============================================================================
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2
MONTHLY_MA_PERIOD = 25
MONTHLY_UPDATE_FREQ = '168h'
ENTRY_BUFFER = 0.025
EXIT_BUFFER = 0.005

# =============================================================================
# PORTFOLIO PARAMETERS
# =============================================================================
HIT_RATE_THRESHOLD = 0.50
MIN_SAMPLES_PER_STATE = 20
MIN_TRAINING_MONTHS = 12
HIT_RATE_RECALC_MONTHS = 1
VOL_LOOKBACK_MONTHS = 1
COV_LOOKBACK_MONTHS = 2

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
KRAKEN_TRADING_FEE = 0.0026
KRAKEN_SLIPPAGE = 0.0010
KRAKEN_MARGIN_OPEN = 0.0002
KRAKEN_ROLLOVER_4H = 0.0002
KRAKEN_ROLLOVER_DAY = KRAKEN_ROLLOVER_4H * 6
COST_PER_SIDE = KRAKEN_TRADING_FEE + KRAKEN_SLIPPAGE

# =============================================================================
# SHORT PARAMETERS
# =============================================================================
SHORT_SIZE_SCALAR = 0.50

# Walk-forward recalculation frequency
WF_RECALC_MONTHS = 3  # Quarterly

# Minimum training for short selection (longer than long hit rates)
WF_MIN_TRAINING_MONTHS = 18  # 18 months before we start shorting

# Walk-forward selection criteria
WF_CRITERIA = {
    'WF-S': {  # Strict
        'name': 'WF Strict',
        'min_samples': 30,
        'min_hit_rate': 0.55,
        'max_avg_return': -0.0010,  # avg return must be < -0.10%
        'description': 'HR>55%, ret<-0.10%, n≥30',
    },
    'WF-M': {  # Moderate
        'name': 'WF Moderate',
        'min_samples': 20,
        'min_hit_rate': 0.52,
        'max_avg_return': -0.0005,  # avg return must be < -0.05%
        'description': 'HR>52%, ret<-0.05%, n≥20',
    },
}

# Fixed configs for comparison
FIXED_CONFIGS = {
    'FIXED-C': {
        'name': 'Fixed Tier 1',
        'short_states': {8, 10, 12, 14},
        'description': 'Fixed states from full-sample research',
    },
    'FIXED-E': {
        'name': 'Fixed All Strong',
        'short_states': {4, 6, 8, 10, 12, 13, 14, 15},
        'description': 'Fixed states from full-sample research',
    },
}

# State labels
STATE_NAMES = {
    0: 'All bearish', 1: 'Bear, MA168>24', 2: 'Bear, MA72>24',
    3: 'Bear, both MAs above', 4: 'Bear24h Bull168h MAs below',
    5: 'Bear24h Bull168h MA168>24', 6: 'Bear24h Bull168h MA72>24',
    7: 'Bear24h Bull168h both above', 8: 'Bull24h Bear168h MAs below',
    9: 'Bull24h Bear168h MA168>24', 10: 'Bull24h Bear168h MA72>24',
    11: 'Bull24h Bear168h both above', 12: 'EXHAUSTION MAs below',
    13: 'All bull MA168>24', 14: 'EXHAUSTION MA72>24', 15: 'All bull all above',
}


# =============================================================================
# CALENDAR HELPERS
# =============================================================================

def get_month_start(date: pd.Timestamp, months_back: int = 0) -> pd.Timestamp:
    target = date - pd.DateOffset(months=months_back)
    return target.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_months_of_data(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def has_minimum_training(current_date, data_start, min_months):
    return get_months_of_data(data_start, current_date) >= min_months


def get_quarter(date):
    return (date.year, (date.month - 1) // 3)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WFSelection:
    """Record of a walk-forward short state selection."""
    date: pd.Timestamp
    selected_states: Set[int]
    state_details: Dict  # {state: {n, avg_ret, hit_rate}}


@dataclass
class BacktestResult:
    name: str
    config: str = ""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_raw: float = 0.0
    calmar_ratio: float = 0.0
    n_long_trades: int = 0
    n_short_trades: int = 0
    n_total_trades: int = 0
    long_trading_costs: float = 0.0
    short_trading_costs: float = 0.0
    short_rollover_costs: float = 0.0
    total_costs: float = 0.0
    signals_filtered: int = 0
    signals_total: int = 0
    short_days: int = 0
    short_states_used: Dict = field(default_factory=dict)
    equity_curve: Optional[pd.Series] = None
    yearly_returns: Dict[int, float] = field(default_factory=dict)
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    short_pnl_gross: float = 0.0
    # Walk-forward specific
    wf_selections: List = field(default_factory=list)  # List of WFSelection


# =============================================================================
# SIGNAL GENERATION (identical to long_short_backtest.py)
# =============================================================================

def resample_ohlcv(df, timeframe):
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def label_trend_binary_rolling(close_series, ma_series, entry_buffer, exit_buffer):
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


def generate_signals_with_monthly(df_1h):
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')

    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    trend_24h = label_trend_binary_rolling(df_24h['close'], ma_24h, ENTRY_BUFFER, EXIT_BUFFER)

    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_72h_aligned = ma_72h.reindex(df_24h.index, method='ffill')

    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    ma_168h_aligned = ma_168h.reindex(df_24h.index, method='ffill')
    trend_168h_raw = label_trend_binary_rolling(
        df_168h['close'], df_168h['close'].rolling(MA_PERIOD_168H).mean(),
        ENTRY_BUFFER, EXIT_BUFFER
    )

    aligned = pd.DataFrame(index=df_24h.index)
    aligned['trend_24h'] = trend_24h.shift(1)
    aligned['trend_168h'] = trend_168h_raw.shift(1).reindex(df_24h.index, method='ffill')
    aligned['ma72_above_ma24'] = (ma_72h_aligned > ma_24h).astype(int).shift(1)
    aligned['ma168_above_ma24'] = (ma_168h_aligned > ma_24h).astype(int).shift(1)

    ma_hours = MONTHLY_MA_PERIOD * 24
    ma_monthly_hourly = df_1h['close'].rolling(ma_hours).mean()
    ma_monthly_at_freq = ma_monthly_hourly.resample(MONTHLY_UPDATE_FREQ).last()
    close_at_freq = df_1h['close'].resample(MONTHLY_UPDATE_FREQ).last()
    monthly_trend_freq = (close_at_freq > ma_monthly_at_freq).astype(int).dropna()
    monthly_trend = monthly_trend_freq.reindex(df_24h.index, method='ffill')
    aligned['monthly_trend'] = monthly_trend.shift(1)

    signals = aligned.dropna().astype(int)
    signals['state_int'] = (
        signals['trend_24h'] * 8 + signals['trend_168h'] * 4 +
        signals['ma72_above_ma24'] * 2 + signals['ma168_above_ma24'] * 1
    )

    return signals, df_24h


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
    only_ma72_changed = (ma72_changed and not trend_24h_changed and
                         not trend_168h_changed and not ma168_changed)
    return not only_ma72_changed


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
# WALK-FORWARD SHORT STATE SELECTION
# =============================================================================

def select_short_states_wf(
    data: Dict,
    cutoff_date: pd.Timestamp,
    criteria: Dict,
) -> Tuple[Set[int], Dict]:
    """
    Select short candidate states using ONLY data before cutoff_date.

    For each of the 16 states, when monthly context is bearish:
        - Compute forward return stats from historical data
        - Apply selection criteria

    Returns:
        (set of selected state ints, details dict)
    """
    # Pool all pair data before cutoff
    all_returns = []

    for pair in DEPLOY_PAIRS:
        signals = data[pair]['signals']
        returns = data[pair]['returns']

        # Only use data strictly before cutoff
        hist_signals = signals[signals.index < cutoff_date]
        hist_returns = returns[returns.index < cutoff_date]

        if len(hist_signals) == 0:
            continue

        # Align and compute forward returns
        common_idx = hist_returns.index.intersection(hist_signals.index)
        if len(common_idx) < 20:
            continue

        aligned_ret = hist_returns.loc[common_idx]
        aligned_sig = hist_signals.loc[common_idx]

        # Forward return = next day's return (shift -1)
        fwd_ret = aligned_ret.shift(-1).iloc[:-1]
        aligned_sig = aligned_sig.iloc[:-1]

        # Filter to monthly bearish only
        if 'monthly_trend' in aligned_sig.columns:
            bearish_mask = aligned_sig['monthly_trend'] == 0
            bearish_sig = aligned_sig[bearish_mask]
            bearish_ret = fwd_ret[bearish_mask]

            for idx in bearish_sig.index:
                if idx in bearish_ret.index and not pd.isna(bearish_ret[idx]):
                    state_int = int(bearish_sig.loc[idx, 'state_int'])
                    all_returns.append({
                        'state': state_int,
                        'fwd_return': bearish_ret[idx],
                    })

    if not all_returns:
        return set(), {}

    df = pd.DataFrame(all_returns)

    # Analyze each state
    selected = set()
    details = {}

    for state in range(16):
        state_data = df[df['state'] == state]['fwd_return']
        n = len(state_data)

        if n == 0:
            details[state] = {'n': 0, 'avg_return': 0, 'hit_rate': 0, 'selected': False}
            continue

        avg_ret = state_data.mean()
        short_hit_rate = (state_data < 0).sum() / n  # % negative returns

        details[state] = {
            'n': n,
            'avg_return': avg_ret,
            'hit_rate': short_hit_rate,
            'selected': False,
        }

        # Apply criteria
        if (n >= criteria['min_samples'] and
            short_hit_rate >= criteria['min_hit_rate'] and
            avg_ret <= criteria['max_avg_return']):
            selected.add(state)
            details[state]['selected'] = True

    return selected, details


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(db: Database) -> Dict:
    data = {}
    for pair in DEPLOY_PAIRS:
        print(f"    {pair}...", end=" ", flush=True)
        df_1h = db.get_ohlcv(pair)
        signals, df_24h = generate_signals_with_monthly(df_1h)
        returns = df_24h['close'].pct_change()
        data[pair] = {'prices': df_24h, 'signals': signals, 'returns': returns}
        print(f"{len(df_24h)} days")
    return data


# =============================================================================
# BACKTEST ENGINE (with walk-forward short selection)
# =============================================================================

def run_backtest_wf(
    data: Dict,
    dates: List[pd.Timestamp],
    data_start: pd.Timestamp,
    name: str = "Backtest",
    config: str = "",
    fixed_short_states: Optional[Set[int]] = None,
    wf_criteria: Optional[Dict] = None,
    use_dd_protection: bool = True,
) -> BacktestResult:
    """
    Run backtest with either fixed or walk-forward short state selection.

    If fixed_short_states is provided: use those states for the entire backtest.
    If wf_criteria is provided: dynamically select short states each quarter.
    """
    is_walkforward = wf_criteria is not None
    active_short_states = fixed_short_states if fixed_short_states else set()

    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
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

    # PnL
    long_pnl_total = 0.0
    short_pnl_gross_total = 0.0

    # State tracking
    returns_dict = {pair: data[pair]['returns'] for pair in DEPLOY_PAIRS}
    returns_df = pd.DataFrame(returns_dict).dropna()
    asset_weights = pd.Series({pair: 1.0 / len(DEPLOY_PAIRS) for pair in DEPLOY_PAIRS})
    prev_exposures = {pair: 0.0 for pair in DEPLOY_PAIRS}
    last_rebalance_month = None

    prev_signal_states = {pair: None for pair in DEPLOY_PAIRS}
    active_states_tracker = {pair: None for pair in DEPLOY_PAIRS}
    hit_rate_cache = {pair: {} for pair in DEPLOY_PAIRS}
    last_recalc_month = {pair: None for pair in DEPLOY_PAIRS}

    # Monthly tracking
    monthly_bullish_days = 0
    monthly_bearish_days = 0
    disagreement_count = 0
    filtered_by_monthly = 0

    short_days = 0
    short_states_used = {}

    # Walk-forward tracking
    wf_selections = []
    last_wf_quarter = None

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

        # =================================================================
        # WALK-FORWARD: Quarterly short state re-selection
        # =================================================================
        if is_walkforward:
            current_quarter = get_quarter(date)
            if current_quarter != last_wf_quarter:
                if has_minimum_training(date, data_start, WF_MIN_TRAINING_MONTHS):
                    cutoff = get_month_start(date, 0)
                    selected, details = select_short_states_wf(
                        data, cutoff, wf_criteria)
                    active_short_states = selected
                    wf_selections.append(WFSelection(
                        date=date, selected_states=selected.copy(),
                        state_details=details.copy()))
                else:
                    active_short_states = set()
                last_wf_quarter = current_quarter

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

        # Risk parity rebalance
        if last_rebalance_month != current_month:
            lookback_start = get_month_start(date, COV_LOOKBACK_MONTHS)
            lookback_end = get_month_start(date, 0)
            lookback_returns = returns_df.loc[lookback_start:lookback_end]
            if len(lookback_returns) >= 20:
                asset_weights = calculate_risk_parity_weights(lookback_returns)
            last_rebalance_month = current_month

        # =================================================================
        # CALCULATE EXPOSURES
        # =================================================================
        asset_exposures = {}
        total_long_exposure = 0.0

        for pair in DEPLOY_PAIRS:
            signals = data[pair]['signals']
            returns = data[pair]['returns']

            if date not in signals.index:
                asset_exposures[pair] = 0.0
                continue

            sig = signals.loc[date]
            current_state = (
                int(sig['trend_24h']), int(sig['trend_168h']),
                int(sig['ma72_above_ma24']), int(sig['ma168_above_ma24'])
            )
            state_int = int(sig['state_int'])

            prev_state = prev_signal_states[pair]
            if current_state != prev_state:
                signals_total += 1
                should_trade = should_trade_signal_16state(prev_state, current_state, True)
                if should_trade:
                    active_states_tracker[pair] = current_state
                else:
                    signals_filtered += 1
            prev_signal_states[pair] = current_state

            active = active_states_tracker[pair]
            if active is None:
                active = current_state
                active_states_tracker[pair] = current_state

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

            # Monthly context + short logic
            monthly_trend = int(sig['monthly_trend']) if 'monthly_trend' in sig.index else 1

            if monthly_trend == 1:
                monthly_bullish_days += 1
            else:
                monthly_bearish_days += 1
                if pos > 0.5:
                    disagreement_count += 1
                    filtered_by_monthly += 1

                if state_int in active_short_states:
                    pos = -SHORT_SIZE_SCALAR
                    short_states_used[state_int] = short_states_used.get(state_int, 0) + 1
                else:
                    pos = 0.0

            weighted_pos = pos * asset_weights[pair]
            asset_exposures[pair] = weighted_pos
            if weighted_pos > 0:
                total_long_exposure += weighted_pos

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
        # EXECUTE TRADES + COSTS
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
                base_trade_cost = trade_value * COST_PER_SIDE
                entering_short = (curr_exp < 0 and prev_exp >= 0)
                increasing_short = (curr_exp < prev_exp < 0)

                if curr_exp < 0 or prev_exp < 0:
                    daily_short_cost += base_trade_cost
                    n_short_trades += 1
                else:
                    daily_long_cost += base_trade_cost
                    n_long_trades += 1

                if entering_short:
                    daily_short_cost += abs(curr_exp) * equity * KRAKEN_MARGIN_OPEN
                elif increasing_short:
                    daily_short_cost += abs(curr_exp - prev_exp) * equity * KRAKEN_MARGIN_OPEN

                prev_exposures[pair] = curr_exp
            else:
                curr_exp = prev_exp
                asset_exposures[pair] = prev_exp

            if curr_exp != 0 and next_date in data[pair]['returns'].index:
                ret = data[pair]['returns'].loc[next_date]
                if not pd.isna(ret):
                    pnl = equity * curr_exp * ret
                    daily_pnl += pnl
                    if curr_exp > 0:
                        daily_long_pnl += pnl
                    else:
                        daily_short_pnl_gross += pnl

            if curr_exp < 0:
                rollover = abs(curr_exp) * equity * KRAKEN_ROLLOVER_DAY
                daily_rollover += rollover
                short_days += 1

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

    # Final metrics
    equity_series = pd.Series(equity_curve)
    max_drawdown_raw = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
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

    return BacktestResult(
        name=name, config=config,
        total_return=total_return, annual_return=annual_return,
        sharpe_ratio=sharpe, max_drawdown_raw=max_drawdown_raw, calmar_ratio=calmar,
        n_long_trades=n_long_trades, n_short_trades=n_short_trades,
        n_total_trades=n_long_trades + n_short_trades,
        long_trading_costs=long_trading_costs, short_trading_costs=short_trading_costs,
        short_rollover_costs=short_rollover_costs, total_costs=total_costs,
        signals_filtered=signals_filtered, signals_total=signals_total,
        short_days=short_days, short_states_used=short_states_used,
        equity_curve=equity_series, yearly_returns=yearly_returns,
        long_pnl=long_pnl_total,
        short_pnl=short_pnl_gross_total - short_trading_costs - short_rollover_costs,
        short_pnl_gross=short_pnl_gross_total,
        wf_selections=wf_selections,
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
    Annual Return:        {result.annual_return * 100:>+8.1f}%
    Sharpe Ratio:         {result.sharpe_ratio:>8.2f}
    Max Drawdown:         {result.max_drawdown_raw * 100:>8.1f}%
    Calmar Ratio:         {result.calmar_ratio:>8.2f}

    Trades & Costs (Kraken):
    ─────────────────────────────────────────────────────────────────
    Long / Short / Total: {result.n_long_trades:>6,} / {result.n_short_trades:>6,} / {result.n_total_trades:>6,}
    Long Costs:           ${result.long_trading_costs:>12,.0f}
    Short Costs:          ${result.short_trading_costs + result.short_rollover_costs:>12,.0f}
    Total Costs:          ${result.total_costs:>12,.0f}""")

    if result.short_days > 0:
        print(f"""
    Short Attribution:
    ─────────────────────────────────────────────────────────────────
    Short Pair-Days:      {result.short_days:>8,}
    Short PnL (gross):    ${result.short_pnl_gross:>12,.0f}
    Short PnL (net):      ${result.short_pnl:>12,.0f}""")

        if result.short_states_used:
            print(f"    States shorted:       {sorted(result.short_states_used.keys())}")

    print(f"""
    Yearly Returns:
    ─────────────────────────────────────────────────────────────────""")
    for year in sorted(result.yearly_returns.keys()):
        print(f"      {year}:  {result.yearly_returns[year] * 100:>+8.1f}%")


def display_wf_history(result: BacktestResult):
    """Display walk-forward selection history."""
    if not result.wf_selections:
        return

    print(f"\n    Walk-Forward Selection History ({result.config}):")
    print(f"    {'─' * 100}")
    print(f"    {'Quarter':<14s} {'States Selected':<40s} {'Count':>6s} {'Overlap w/ Tier1':>16s} {'Overlap w/ AllS':>16s}")
    print(f"    {'─' * 100}")

    tier1 = {8, 10, 12, 14}
    all_strong = {4, 6, 8, 10, 12, 13, 14, 15}

    for sel in result.wf_selections:
        q_str = f"{sel.date.year} Q{(sel.date.month - 1) // 3 + 1}"
        states_str = str(sorted(sel.selected_states)) if sel.selected_states else "[]"

        # Overlap with fixed sets
        t1_overlap = len(sel.selected_states & tier1)
        t1_total = len(tier1)
        as_overlap = len(sel.selected_states & all_strong)
        as_total = len(all_strong)

        print(f"    {q_str:<14s} {states_str:<40s} {len(sel.selected_states):>6d} "
              f"{t1_overlap}/{t1_total:>14d} {as_overlap}/{as_total:>14d}")

    # Stability analysis
    print(f"\n    Selection Stability:")
    print(f"    {'─' * 80}")

    # Count how often each state was selected
    state_freq = {}
    total_quarters = len(result.wf_selections)
    for sel in result.wf_selections:
        for s in sel.selected_states:
            state_freq[s] = state_freq.get(s, 0) + 1

    if state_freq:
        print(f"    {'State':>7s} {'Name':<30s} {'Selected':>10s} {'Pct of Quarters':>16s} {'In Tier1':>10s} {'In AllS':>10s}")
        print(f"    {'─' * 80}")
        for state in range(16):
            freq = state_freq.get(state, 0)
            if freq > 0:
                pct = freq / total_quarters * 100
                in_t1 = "✓" if state in tier1 else ""
                in_as = "✓" if state in all_strong else ""
                name = STATE_NAMES.get(state, '')[:28]
                print(f"    {state:>7d} {name:<30s} {freq:>8d}/{total_quarters:<3d} {pct:>14.1f}% {in_t1:>10s} {in_as:>10s}")


def display_comparison(results: Dict[str, BacktestResult]):
    print("\n" + "=" * 130)
    print("CONFIGURATION COMPARISON")
    print("=" * 130)

    baseline_sharpe = results[RUN_CONFIGS[0]].sharpe_ratio

    print(f"""
    ┌──────────┬───────────────────────────┬──────────┬───────────────┬──────────┬──────────┬──────────┬──────────┬───────────────┐
    │ Config   │ Name                      │ Ann.Ret  │ Sharpe        │ Max DD   │ Calmar   │ L Trades │ S Trades │ Total Costs   │
    ├──────────┼───────────────────────────┼──────────┼───────────────┼──────────┼──────────┼──────────┼──────────┼───────────────┤""")

    for config in RUN_CONFIGS:
        if config not in results:
            continue
        r = results[config]
        sharpe_diff = r.sharpe_ratio - baseline_sharpe
        diff_str = f"({sharpe_diff:+.2f})" if config != RUN_CONFIGS[0] else ""

        print(f"    │ {config:<8s} │ {r.name:25s} │ {r.annual_return * 100:>+7.1f}% │ "
              f"{r.sharpe_ratio:>5.2f} {diff_str:>7s} │ {r.max_drawdown_raw * 100:>7.1f}% │ "
              f"{r.calmar_ratio:>7.2f} │ {r.n_long_trades:>8,} │ {r.n_short_trades:>8,} │ "
              f"${r.total_costs:>12,.0f} │")

    print(f"    └──────────┴───────────────────────────┴──────────┴───────────────┴──────────┴──────────┴──────────┴──────────┴───────────────┘")

    # Short PnL comparison
    print(f"""
    SHORT PnL ATTRIBUTION:
    ─────────────────────────────────────────────────────────────────────────────────────────
    {'Config':<10s} {'Long PnL':>12s} {'Short Gross':>12s} {'Short Costs':>12s} {'Short Net':>12s} {'S-Days':>8s}
    ─────────────────────────────────────────────────────────────────────────────────────────""")

    for config in RUN_CONFIGS:
        if config not in results:
            continue
        r = results[config]
        short_costs = r.short_trading_costs + r.short_rollover_costs
        print(f"    {config:<10s} ${r.long_pnl:>11,.0f} ${r.short_pnl_gross:>11,.0f} "
              f"${short_costs:>11,.0f} ${r.short_pnl:>11,.0f} {r.short_days:>7,}")

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
                ret = results[config].yearly_returns.get(year, 0)
                row += f"  {ret * 100:>+7.1f}%"
        print(row)

    # Walk-forward vs fixed comparison
    print(f"""
    ═══════════════════════════════════════════════════════════════
    WALK-FORWARD VALIDATION VERDICT
    ═══════════════════════════════════════════════════════════════""")

    fixed_c = results.get('FIXED-C')
    fixed_e = results.get('FIXED-E')
    wf_s = results.get('WF-S')
    wf_m = results.get('WF-M')

    if fixed_c and wf_s:
        diff = wf_s.sharpe_ratio - fixed_c.sharpe_ratio
        print(f"""
    WF-Strict vs Fixed-Tier1:
      Sharpe:  {wf_s.sharpe_ratio:.2f} vs {fixed_c.sharpe_ratio:.2f} (Δ {diff:+.2f})
      Return:  {wf_s.annual_return*100:+.1f}% vs {fixed_c.annual_return*100:+.1f}%
      Max DD:  {wf_s.max_drawdown_raw*100:.1f}% vs {fixed_c.max_drawdown_raw*100:.1f}%""")

        if abs(diff) < 0.3:
            print(f"      → SIMILAR: Walk-forward validates fixed selection")
        elif diff > 0.3:
            print(f"      → WF BETTER: Dynamic selection outperforms fixed")
        else:
            print(f"      → WF WORSE: Full-sample selection may be overfit")

    if fixed_e and wf_m:
        diff = wf_m.sharpe_ratio - fixed_e.sharpe_ratio
        print(f"""
    WF-Moderate vs Fixed-AllStrong:
      Sharpe:  {wf_m.sharpe_ratio:.2f} vs {fixed_e.sharpe_ratio:.2f} (Δ {diff:+.2f})
      Return:  {wf_m.annual_return*100:+.1f}% vs {fixed_e.annual_return*100:+.1f}%
      Max DD:  {wf_m.max_drawdown_raw*100:.1f}% vs {fixed_e.max_drawdown_raw*100:.1f}%""")

        if abs(diff) < 0.3:
            print(f"      → SIMILAR: Walk-forward validates fixed selection")
        elif diff > 0.3:
            print(f"      → WF BETTER: Dynamic selection outperforms fixed")
        else:
            print(f"      → WF WORSE: Full-sample selection may be overfit")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 130)
    print("WALK-FORWARD VALIDATION — LONG/SHORT STRATEGY")
    print("=" * 130)

    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  WALK-FORWARD VALIDATION                                                                                                ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                         ║
    ║  Tests whether short state selection is robust out-of-sample.                                                           ║
    ║                                                                                                                         ║
    ║  FIXED configs use states chosen from full-sample analysis (potential lookahead bias).                                  ║
    ║  WF configs dynamically select short states each quarter using ONLY historical data.                                    ║
    ║                                                                                                                         ║
    ║  CONFIGURATIONS:                                                                                                        ║
    ║    FIXED-C = Fixed Tier 1 (states 8,10,12,14) — from full-sample regime research                                      ║
    ║    FIXED-E = Fixed All Strong (states 4,6,8,10,12,13,14,15) — from full-sample                                        ║
    ║    WF-S    = Walk-forward strict: HR>55%, avg ret<-0.10%, n≥30, quarterly reselection                                  ║
    ║    WF-M    = Walk-forward moderate: HR>52%, avg ret<-0.05%, n≥20, quarterly reselection                                ║
    ║                                                                                                                         ║
    ║  VERDICT:                                                                                                               ║
    ║    WF ≈ Fixed  → Robust (not overfit, safe to deploy)                                                                  ║
    ║    WF > Fixed  → Dynamic selection adds value                                                                           ║
    ║    WF < Fixed  → Potential overfitting in full-sample analysis                                                          ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"  Configs: {', '.join(RUN_CONFIGS)}")
    print()

    # Load data
    print("  Connecting to database...")
    db = Database()
    print("\n  Loading data...")
    data = load_all_data(db)

    # Common dates
    all_dates = None
    for pair in DEPLOY_PAIRS:
        dates = data[pair]['signals'].index
        all_dates = set(dates) if all_dates is None else all_dates.intersection(set(dates))
    dates = sorted(list(all_dates))
    data_start = dates[0]
    print(f"\n  Common dates: {len(dates)} ({dates[0].date()} to {dates[-1].date()})")

    results = {}

    for config in RUN_CONFIGS:
        print(f"\n{'=' * 70}")
        print(f"  RUNNING CONFIG {config}")

        if config in FIXED_CONFIGS:
            cfg = FIXED_CONFIGS[config]
            print(f"  {cfg['description']}")
            print(f"  States: {sorted(cfg['short_states'])}")
            print(f"{'=' * 70}")

            result = run_backtest_wf(
                data=data, dates=dates, data_start=data_start,
                name=cfg['name'], config=config,
                fixed_short_states=cfg['short_states'],
            )

        elif config in WF_CRITERIA:
            criteria = WF_CRITERIA[config]
            print(f"  {criteria['description']}")
            print(f"  Quarterly reselection, {WF_MIN_TRAINING_MONTHS}mo min training")
            print(f"{'=' * 70}")

            result = run_backtest_wf(
                data=data, dates=dates, data_start=data_start,
                name=criteria['name'], config=config,
                wf_criteria=criteria,
            )

        results[config] = result
        display_result(result)

        # Show WF selection history
        if result.wf_selections:
            display_wf_history(result)

    # Comparison
    if len(results) > 1:
        display_comparison(results)

    print("\n" + "=" * 130)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 130)

    return results


if __name__ == "__main__":
    results = main()