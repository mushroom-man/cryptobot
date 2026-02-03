#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duration-Based Bayesian Transition Model
=========================================
Test P(transition | state, duration) using only:
  - Historical transition frequencies (prior)
  - Duration in current state (likelihood multiplier)

This simplified model emerged from finding that D, D', D'', D'''
had no predictive power (AUC ~0.50), while duration showed weak
but consistent signal (AUC ~0.544).

Architecture:
    Prior: P(transition | state) = historical transition rate per state
    Likelihood: P(duration | transition) vs P(duration | no transition)
    Posterior: P(transition | state, duration)

Validation:
    Expanding window - no look-ahead bias
    Train on [0, t], predict at t, compare to t+horizon

Usage:
    python duration_bayesian_model.py --pair ETHUSD
    python duration_bayesian_model.py --all-pairs
    python duration_bayesian_model.py --all-pairs --with-trading-sim
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from cryptobot.data.database import Database
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    print("Warning: Database not available.")

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score, brier_score_loss
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked from validation)
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Horizons to test (hours)
HORIZONS = [24, 48, 72]

# Minimum samples for reliable estimates
MIN_STATE_SAMPLES = 30
MIN_TRAINING_DAYS = 365

# Duration multiplier configurations to test
DURATION_CONFIGS = {
    'linear': {'type': 'linear', 'params': {}},
    'exponential': {'type': 'exponential', 'params': {'decay': 0.5}},
    'step': {'type': 'step', 'params': {'thresholds': [0.5, 1.0, 1.5, 2.0]}},
    'log': {'type': 'log', 'params': {}},
}

# Trading simulation parameters
INITIAL_CAPITAL = 100000.0
TRADING_COST = 0.0020


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StateStats:
    """Statistics for a single state."""
    state_id: int
    n_entries: int = 0
    n_transitions: int = 0
    total_duration_hours: float = 0.0
    durations: List[float] = field(default_factory=list)
    
    @property
    def avg_duration(self) -> float:
        return self.total_duration_hours / self.n_entries if self.n_entries > 0 else 24.0
    
    @property
    def transition_rate(self) -> float:
        """Daily transition rate."""
        return self.n_transitions / self.n_entries if self.n_entries > 0 else 0.5
    
    @property
    def median_duration(self) -> float:
        return np.median(self.durations) if self.durations else 24.0


@dataclass 
class TransitionMatrix:
    """16x16 transition matrix with duration stats per state."""
    pair: str
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((16, 16)))
    state_stats: Dict[int, StateStats] = field(default_factory=dict)
    
    def __post_init__(self):
        for i in range(16):
            self.state_stats[i] = StateStats(state_id=i)
    
    def add_transition(self, from_state: int, to_state: int, duration_hours: float):
        """Record a transition."""
        self.matrix[from_state, to_state] += 1
        self.state_stats[from_state].n_entries += 1
        self.state_stats[from_state].total_duration_hours += duration_hours
        self.state_stats[from_state].durations.append(duration_hours)
        if from_state != to_state:
            self.state_stats[from_state].n_transitions += 1
    
    def get_transition_probs(self, from_state: int) -> np.ndarray:
        """Get transition probabilities from a state."""
        row = self.matrix[from_state, :]
        total = row.sum()
        if total > 0:
            return row / total
        else:
            # Uniform if no data
            return np.ones(16) / 16
    
    def get_stay_prob(self, state: int) -> float:
        """P(stay in same state)."""
        probs = self.get_transition_probs(state)
        return probs[state]
    
    def get_leave_prob(self, state: int) -> float:
        """P(transition to different state)."""
        return 1.0 - self.get_stay_prob(state)


@dataclass
class ModelResult:
    """Results from model evaluation."""
    pair: str
    horizon: int
    duration_config: str
    
    # Calibration metrics
    auc: float = 0.0
    brier_score: float = 0.0
    
    # Binned calibration
    calibration_bins: Dict[str, Dict] = field(default_factory=dict)
    
    # Per-state performance
    state_aucs: Dict[int, float] = field(default_factory=dict)
    
    # Prediction stats
    n_predictions: int = 0
    n_transitions: int = 0
    avg_predicted_prob: float = 0.0
    avg_actual_rate: float = 0.0


# =============================================================================
# REGIME COMPUTATION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_regime_states_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 16-state regime at daily frequency.
    Returns DataFrame with daily state labels.
    """
    # Resample to timeframes
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    # Compute MAs
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Align to daily
    ma_72h_daily = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_daily = ma_168h.reindex(df_24h.index, method='ffill')
    
    # Trend detection with hysteresis
    def compute_trend(close, ma, entry_buf=ENTRY_BUFFER, exit_buf=EXIT_BUFFER):
        trend = []
        current = 1
        for i in range(len(close)):
            if pd.isna(ma.iloc[i]):
                trend.append(current)
                continue
            price = close.iloc[i]
            ma_val = ma.iloc[i]
            if current == 1:
                if price < ma_val * (1 - exit_buf) and price < ma_val * (1 - entry_buf):
                    current = 0
            else:
                if price > ma_val * (1 + exit_buf) and price > ma_val * (1 + entry_buf):
                    current = 1
            trend.append(current)
        return trend
    
    states = pd.DataFrame(index=df_24h.index)
    states['close'] = df_24h['close']
    states['ma_24h'] = ma_24h
    states['ma_72h'] = ma_72h_daily
    states['ma_168h'] = ma_168h_daily
    
    states['trend_24h'] = compute_trend(df_24h['close'], ma_24h)
    states['trend_168h'] = compute_trend(df_24h['close'], ma_168h_daily)
    states['ma72_above_ma24'] = (ma_72h_daily > ma_24h).astype(int)
    states['ma168_above_ma24'] = (ma_168h_daily > ma_24h).astype(int)
    
    # Combined state
    states['state'] = (
        states['trend_24h'] * 8 +
        states['trend_168h'] * 4 +
        states['ma72_above_ma24'] * 2 +
        states['ma168_above_ma24'] * 1
    )
    
    return states.dropna()


def compute_state_durations(states: pd.DataFrame) -> pd.DataFrame:
    """
    Add duration column: hours in current state.
    """
    state_series = states['state'].values
    
    durations = []
    current_duration = 0
    prev_state = None
    
    for state in state_series:
        if state == prev_state:
            current_duration += 24  # Daily data = 24 hours per bar
        else:
            current_duration = 24
        durations.append(current_duration)
        prev_state = state
    
    states = states.copy()
    states['duration_hours'] = durations
    
    return states


# =============================================================================
# TRANSITION MATRIX BUILDING
# =============================================================================

def build_transition_matrix(states: pd.DataFrame, end_date: pd.Timestamp) -> TransitionMatrix:
    """
    Build transition matrix from historical data up to end_date.
    """
    matrix = TransitionMatrix(pair="")
    
    # Filter to training period
    train_states = states[states.index < end_date]
    
    if len(train_states) < 2:
        return matrix
    
    state_series = train_states['state'].values
    duration_series = train_states['duration_hours'].values
    
    # Track state entries and durations
    current_state = state_series[0]
    entry_duration = 0
    
    for i in range(1, len(state_series)):
        prev_state = state_series[i-1]
        curr_state = state_series[i]
        entry_duration += 24
        
        if curr_state != prev_state:
            # Transition occurred
            matrix.add_transition(prev_state, curr_state, entry_duration)
            entry_duration = 0
        elif i == len(state_series) - 1:
            # End of data, record ongoing state
            matrix.add_transition(curr_state, curr_state, entry_duration)
    
    return matrix


# =============================================================================
# DURATION MULTIPLIER FUNCTIONS
# =============================================================================

def duration_multiplier_linear(duration: float, avg_duration: float) -> float:
    """
    Linear multiplier: P(transition) increases linearly with duration.
    
    ratio = duration / avg_duration
    multiplier = ratio (capped at 3.0)
    """
    ratio = duration / avg_duration if avg_duration > 0 else 1.0
    return min(ratio, 3.0)


def duration_multiplier_exponential(duration: float, avg_duration: float, decay: float = 0.5) -> float:
    """
    Exponential multiplier: P(transition) increases exponentially.
    
    multiplier = exp(decay * (ratio - 1))
    """
    ratio = duration / avg_duration if avg_duration > 0 else 1.0
    return min(np.exp(decay * (ratio - 1)), 5.0)


def duration_multiplier_step(duration: float, avg_duration: float, 
                              thresholds: List[float] = [0.5, 1.0, 1.5, 2.0]) -> float:
    """
    Step function multiplier.
    
    < 0.5x avg: 0.5
    0.5-1.0x avg: 0.75
    1.0-1.5x avg: 1.0
    1.5-2.0x avg: 1.5
    > 2.0x avg: 2.0
    """
    ratio = duration / avg_duration if avg_duration > 0 else 1.0
    
    if ratio < thresholds[0]:
        return 0.5
    elif ratio < thresholds[1]:
        return 0.75
    elif ratio < thresholds[2]:
        return 1.0
    elif ratio < thresholds[3]:
        return 1.5
    else:
        return 2.0


def duration_multiplier_log(duration: float, avg_duration: float) -> float:
    """
    Logarithmic multiplier: diminishing returns.
    
    multiplier = 1 + log(1 + ratio) / 2
    """
    ratio = duration / avg_duration if avg_duration > 0 else 1.0
    return 1.0 + np.log(1 + ratio) / 2


def get_duration_multiplier(duration: float, avg_duration: float, 
                            config: Dict) -> float:
    """Get duration multiplier based on config."""
    config_type = config['type']
    params = config.get('params', {})
    
    if config_type == 'linear':
        return duration_multiplier_linear(duration, avg_duration)
    elif config_type == 'exponential':
        return duration_multiplier_exponential(duration, avg_duration, **params)
    elif config_type == 'step':
        return duration_multiplier_step(duration, avg_duration, **params)
    elif config_type == 'log':
        return duration_multiplier_log(duration, avg_duration)
    else:
        return 1.0


# =============================================================================
# BAYESIAN PREDICTION
# =============================================================================

def predict_transition_prob(
    state: int,
    duration: float,
    matrix: TransitionMatrix,
    duration_config: Dict,
    horizon_hours: int = 24
) -> float:
    """
    Predict P(transition within horizon | state, duration).
    
    Uses Bayesian update:
        Prior: historical transition rate for this state
        Likelihood ratio: duration multiplier
        Posterior: P(transition | duration)
    """
    stats = matrix.state_stats[state]
    
    # Prior: base transition rate (per day)
    prior = stats.transition_rate
    
    # If insufficient data, return uninformative prior
    if stats.n_entries < MIN_STATE_SAMPLES:
        return 0.5
    
    # Likelihood multiplier based on duration
    avg_duration = stats.avg_duration
    multiplier = get_duration_multiplier(duration, avg_duration, duration_config)
    
    # Adjust for horizon (if predicting 48h, roughly double the base rate)
    horizon_factor = horizon_hours / 24.0
    
    # Posterior (simplified): prior * multiplier, capped at 0.95
    posterior = prior * multiplier * horizon_factor
    posterior = min(max(posterior, 0.01), 0.95)
    
    return posterior


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    states: pd.DataFrame,
    duration_config: Dict,
    horizon_hours: int = 24,
    train_ratio: float = 0.6
) -> ModelResult:
    """
    Evaluate transition prediction model using expanding window.
    """
    result = ModelResult(
        pair="",
        horizon=horizon_hours,
        duration_config=duration_config['type']
    )
    
    n_days = len(states)
    train_end_idx = int(n_days * train_ratio)
    
    predictions = []
    actuals = []
    state_predictions = {i: {'pred': [], 'actual': []} for i in range(16)}
    
    # Expanding window evaluation
    for i in range(train_end_idx, n_days - (horizon_hours // 24)):
        current_date = states.index[i]
        current_state = int(states.iloc[i]['state'])
        current_duration = states.iloc[i]['duration_hours']
        
        # Build matrix from data up to current date
        matrix = build_transition_matrix(states, current_date)
        
        # Predict
        pred_prob = predict_transition_prob(
            current_state, current_duration, matrix, duration_config, horizon_hours
        )
        
        # Actual: did transition occur within horizon?
        future_idx = i + (horizon_hours // 24)
        if future_idx < n_days:
            future_state = int(states.iloc[future_idx]['state'])
            actual = 1 if future_state != current_state else 0
        else:
            continue
        
        predictions.append(pred_prob)
        actuals.append(actual)
        state_predictions[current_state]['pred'].append(pred_prob)
        state_predictions[current_state]['actual'].append(actual)
    
    if len(predictions) < 100:
        return result
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Overall metrics
    try:
        result.auc = roc_auc_score(actuals, predictions)
    except:
        result.auc = 0.5
    
    result.brier_score = brier_score_loss(actuals, predictions)
    result.n_predictions = len(predictions)
    result.n_transitions = int(actuals.sum())
    result.avg_predicted_prob = predictions.mean()
    result.avg_actual_rate = actuals.mean()
    
    # Per-state AUC
    for state_id in range(16):
        state_preds = state_predictions[state_id]['pred']
        state_acts = state_predictions[state_id]['actual']
        if len(state_preds) >= 20 and len(set(state_acts)) > 1:
            try:
                result.state_aucs[state_id] = roc_auc_score(state_acts, state_preds)
            except:
                result.state_aucs[state_id] = 0.5
    
    # Calibration bins
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in bins:
        mask = (predictions >= low) & (predictions < high)
        if mask.sum() > 0:
            bin_preds = predictions[mask]
            bin_acts = actuals[mask]
            result.calibration_bins[f"{low:.1f}-{high:.1f}"] = {
                'n': int(mask.sum()),
                'avg_predicted': bin_preds.mean(),
                'avg_actual': bin_acts.mean(),
                'calibration_error': abs(bin_preds.mean() - bin_acts.mean())
            }
    
    return result


# =============================================================================
# TRADING SIMULATION
# =============================================================================

def run_trading_simulation(
    states: pd.DataFrame,
    returns: pd.Series,
    duration_config: Dict,
    horizon_hours: int = 24,
    train_ratio: float = 0.6
) -> Dict:
    """
    Simulate trading with position sizing based on P(transition).
    
    Position sizing:
        P(transition) < 0.2: 100% position
        P(transition) 0.2-0.4: 80% position
        P(transition) 0.4-0.6: 60% position
        P(transition) > 0.6: 30% position
    """
    n_days = len(states)
    train_end_idx = int(n_days * train_ratio)
    
    # Align returns with states
    common_idx = states.index.intersection(returns.index)
    states = states.loc[common_idx]
    returns = returns.loc[common_idx]
    n_days = len(states)
    train_end_idx = int(n_days * train_ratio)
    
    # Track equity
    equity_baseline = [INITIAL_CAPITAL]
    equity_adjusted = [INITIAL_CAPITAL]
    
    position_sizes = []
    transition_probs = []
    
    for i in range(train_end_idx, n_days - 1):
        current_date = states.index[i]
        next_date = states.index[i + 1]
        current_state = int(states.iloc[i]['state'])
        current_duration = states.iloc[i]['duration_hours']
        
        # Build matrix
        matrix = build_transition_matrix(states, current_date)
        
        # Predict transition probability
        pred_prob = predict_transition_prob(
            current_state, current_duration, matrix, duration_config, horizon_hours
        )
        transition_probs.append(pred_prob)
        
        # Position sizing based on P(transition)
        if pred_prob < 0.2:
            position_mult = 1.0
        elif pred_prob < 0.4:
            position_mult = 0.8
        elif pred_prob < 0.6:
            position_mult = 0.6
        else:
            position_mult = 0.3
        
        position_sizes.append(position_mult)
        
        # Get return
        if next_date in returns.index:
            daily_return = returns.loc[next_date]
        else:
            daily_return = 0.0
        
        # Baseline: always full position
        equity_baseline.append(equity_baseline[-1] * (1 + daily_return))
        
        # Adjusted: position sized by transition prob
        equity_adjusted.append(equity_adjusted[-1] * (1 + daily_return * position_mult))
    
    # Calculate metrics
    equity_baseline = np.array(equity_baseline)
    equity_adjusted = np.array(equity_adjusted)
    
    returns_baseline = np.diff(equity_baseline) / equity_baseline[:-1]
    returns_adjusted = np.diff(equity_adjusted) / equity_adjusted[:-1]
    
    def calc_metrics(returns_arr, equity_arr):
        total_return = (equity_arr[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
        years = len(returns_arr) / 365
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(365) if returns_arr.std() > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_dd = drawdown.max()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        }
    
    baseline_metrics = calc_metrics(returns_baseline, equity_baseline)
    adjusted_metrics = calc_metrics(returns_adjusted, equity_adjusted)
    
    return {
        'baseline': baseline_metrics,
        'adjusted': adjusted_metrics,
        'avg_position': np.mean(position_sizes),
        'avg_transition_prob': np.mean(transition_probs),
        'n_days': len(position_sizes),
    }


# =============================================================================
# STATE ANALYSIS
# =============================================================================

def analyze_state_durations(states: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Analyze duration characteristics per state."""
    matrix = build_transition_matrix(states, states.index[-1])
    
    rows = []
    for state_id in range(16):
        stats = matrix.state_stats[state_id]
        if stats.n_entries > 0:
            rows.append({
                'state': state_id,
                'n_entries': stats.n_entries,
                'n_transitions': stats.n_transitions,
                'transition_rate': stats.transition_rate,
                'avg_duration_hours': stats.avg_duration,
                'avg_duration_days': stats.avg_duration / 24,
                'median_duration_hours': stats.median_duration,
                'std_duration': np.std(stats.durations) if stats.durations else 0,
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(results: List[ModelResult], pair: str):
    """Display model evaluation results."""
    
    print(f"\n{'='*90}")
    print(f"DURATION-BASED BAYESIAN MODEL RESULTS: {pair}")
    print(f"{'='*90}")
    
    # Group by horizon
    horizons = sorted(set(r.horizon for r in results))
    configs = sorted(set(r.duration_config for r in results))
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  MODEL PERFORMANCE BY DURATION CONFIG AND HORIZON                               │
    ├────────────┬─────────┬─────────┬─────────┬────────────┬────────────┬────────────┤
    │ Config     │ Horizon │ AUC     │ Brier   │ Pred Prob  │ Actual Rate│ Calib Err  │
    ├────────────┼─────────┼─────────┼─────────┼────────────┼────────────┼────────────┤""")
    
    for config in configs:
        for horizon in horizons:
            r = next((x for x in results if x.duration_config == config and x.horizon == horizon), None)
            if r:
                # Calculate average calibration error
                if r.calibration_bins:
                    calib_err = np.mean([b['calibration_error'] for b in r.calibration_bins.values()])
                else:
                    calib_err = 0
                
                print(f"    │ {config:10s} │ {horizon:>5}h  │ {r.auc:>7.3f} │ {r.brier_score:>7.3f} │ "
                      f"{r.avg_predicted_prob:>9.1%}  │ {r.avg_actual_rate:>9.1%}  │ {calib_err:>9.3f}  │")
        print(f"    ├────────────┼─────────┼─────────┼─────────┼────────────┼────────────┼────────────┤")
    
    print(f"    └────────────┴─────────┴─────────┴─────────┴────────────┴────────────┴────────────┘")
    
    # Best config
    best = max(results, key=lambda x: x.auc)
    print(f"""
    BEST CONFIGURATION: {best.duration_config} @ {best.horizon}h
    ─────────────────────────────────────────
    AUC:            {best.auc:.3f}
    Brier Score:    {best.brier_score:.3f}
    N Predictions:  {best.n_predictions:,}
    N Transitions:  {best.n_transitions:,}
    """)


def display_calibration(results: List[ModelResult], pair: str):
    """Display calibration analysis."""
    
    # Find best config
    best = max(results, key=lambda x: x.auc)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  CALIBRATION ANALYSIS: {best.duration_config} @ {best.horizon}h                          │
    ├─────────────┬──────────┬──────────────┬──────────────┬─────────┤
    │ Prob Bin    │ N        │ Avg Predicted│ Avg Actual   │ Error   │
    ├─────────────┼──────────┼──────────────┼──────────────┼─────────┤""")
    
    for bin_name, bin_data in sorted(best.calibration_bins.items()):
        print(f"    │ {bin_name:11s} │ {bin_data['n']:>8,} │ {bin_data['avg_predicted']:>11.1%} │ "
              f"{bin_data['avg_actual']:>11.1%} │ {bin_data['calibration_error']:>7.3f} │")
    
    print(f"    └─────────────┴──────────┴──────────────┴──────────────┴─────────┘")


def display_state_analysis(state_df: pd.DataFrame, pair: str):
    """Display per-state duration analysis."""
    
    print(f"""
    ┌────────────────────────────────────────────────────────────────────────────────┐
    │  STATE DURATION ANALYSIS: {pair:6s}                                              │
    ├───────┬──────────┬────────────┬────────────┬──────────────┬───────────────────┤
    │ State │ N Entries│ Transitions│ Trans Rate │ Avg Duration │ Median Duration   │
    ├───────┼──────────┼────────────┼────────────┼──────────────┼───────────────────┤""")
    
    for _, row in state_df.iterrows():
        print(f"    │ {int(row['state']):>5} │ {int(row['n_entries']):>8} │ {int(row['n_transitions']):>10} │ "
              f"{row['transition_rate']:>9.1%}  │ {row['avg_duration_days']:>10.1f}d  │ "
              f"{row['median_duration_hours']/24:>15.1f}d  │")
    
    print(f"    └───────┴──────────┴────────────┴────────────┴──────────────┴───────────────────┘")


def display_trading_results(sim_results: Dict, pair: str, config: str):
    """Display trading simulation results."""
    
    baseline = sim_results['baseline']
    adjusted = sim_results['adjusted']
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  TRADING SIMULATION: {pair} ({config})                          │
    ├───────────────────┬──────────────────┬──────────────────────────┤
    │ Metric            │ Baseline         │ Transition-Adjusted      │
    ├───────────────────┼──────────────────┼──────────────────────────┤
    │ Annual Return     │ {baseline['annual_return']:>+14.1%}   │ {adjusted['annual_return']:>+22.1%}   │
    │ Sharpe Ratio      │ {baseline['sharpe']:>14.2f}   │ {adjusted['sharpe']:>22.2f}   │
    │ Max Drawdown      │ {baseline['max_drawdown']:>14.1%}   │ {adjusted['max_drawdown']:>22.1%}   │
    ├───────────────────┼──────────────────┴──────────────────────────┤
    │ Avg Position Size │ {sim_results['avg_position']:>43.1%}   │
    │ Avg P(transition) │ {sim_results['avg_transition_prob']:>43.1%}   │
    │ Trading Days      │ {sim_results['n_days']:>43,}   │
    └───────────────────┴─────────────────────────────────────────────┘""")
    
    # Improvement summary
    sharpe_diff = adjusted['sharpe'] - baseline['sharpe']
    dd_diff = adjusted['max_drawdown'] - baseline['max_drawdown']
    
    print(f"""
    IMPACT:
    ─────────────────────────────────────────
    Sharpe Change:    {sharpe_diff:+.2f}
    Drawdown Change:  {dd_diff:+.1%}
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Duration-Based Bayesian Transition Model')
    parser.add_argument('--pair', type=str, default=None, help='Analyze specific pair')
    parser.add_argument('--all-pairs', action='store_true', help='Analyze all pairs')
    parser.add_argument('--with-trading-sim', action='store_true', help='Include trading simulation')
    parser.add_argument('--horizon', type=int, default=None, help='Specific horizon (hours)')
    args = parser.parse_args()
    
    print("=" * 90)
    print("DURATION-BASED BAYESIAN TRANSITION MODEL")
    print("=" * 90)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════════════════════╗
    ║  MODEL DESCRIPTION                                                                     ║
    ╠════════════════════════════════════════════════════════════════════════════════════════╣
    ║  Simplified transition prediction using only:                                          ║
    ║    - Historical transition rate per state (prior)                                      ║
    ║    - Duration in current state (likelihood multiplier)                                 ║
    ║                                                                                        ║
    ║  Duration Configs:                                                                     ║
    ║    linear:      multiplier = duration / avg_duration                                   ║
    ║    exponential: multiplier = exp(0.5 * (ratio - 1))                                    ║
    ║    step:        discrete steps at 0.5x, 1x, 1.5x, 2x avg duration                      ║
    ║    log:         multiplier = 1 + log(1 + ratio) / 2                                    ║
    ║                                                                                        ║
    ║  Validation: Expanding window (no look-ahead bias)                                     ║
    ╚════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Determine pairs
    if args.pair:
        pairs = [args.pair]
    elif args.all_pairs:
        pairs = DEPLOY_PAIRS
    else:
        pairs = ['ETHUSD']
    
    # Determine horizons
    horizons = [args.horizon] if args.horizon else HORIZONS
    
    print(f"  Pairs: {pairs}")
    print(f"  Horizons: {horizons}")
    print(f"  Duration Configs: {list(DURATION_CONFIGS.keys())}")
    
    if not HAS_DATABASE:
        print("\n  ERROR: Database not available")
        return
    
    print("\n  Connecting to database...")
    db = Database()
    
    all_results = {}
    
    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"  ANALYZING: {pair}")
        print(f"{'='*60}")
        
        # Load data
        df_1h = db.get_ohlcv(pair)
        print(f"  Loaded {len(df_1h)} hourly bars")
        
        # Compute states
        states = compute_regime_states_daily(df_1h)
        states = compute_state_durations(states)
        print(f"  Computed {len(states)} daily states")
        
        # State analysis
        state_df = analyze_state_durations(states, pair)
        display_state_analysis(state_df, pair)
        
        # Evaluate models
        results = []
        for config_name, config in DURATION_CONFIGS.items():
            for horizon in horizons:
                print(f"  Testing {config_name} @ {horizon}h...", end=" ", flush=True)
                result = evaluate_model(states, config, horizon)
                result.pair = pair
                results.append(result)
                print(f"AUC={result.auc:.3f}")
        
        display_results(results, pair)
        display_calibration(results, pair)
        
        all_results[pair] = results
        
        # Trading simulation
        if args.with_trading_sim:
            print(f"\n  Running trading simulation...")
            
            # Get returns
            df_24h = resample_ohlcv(df_1h, '24h')
            returns = df_24h['close'].pct_change()
            
            # Use best config
            best = max(results, key=lambda x: x.auc)
            best_config = DURATION_CONFIGS[best.duration_config]
            
            sim_results = run_trading_simulation(
                states, returns, best_config, best.horizon
            )
            display_trading_results(sim_results, pair, best.duration_config)
    
    # Cross-pair summary
    if len(pairs) > 1:
        print(f"\n{'='*90}")
        print("CROSS-PAIR SUMMARY (24h Horizon)")
        print(f"{'='*90}")
        
        print(f"""
    ┌────────┬────────────┬─────────┬─────────┬─────────┬─────────┐
    │ Pair   │ Best Config│ AUC     │ Brier   │ Calib   │ States  │
    ├────────┼────────────┼─────────┼─────────┼─────────┼─────────┤""")
        
        for pair in pairs:
            results_24h = [r for r in all_results[pair] if r.horizon == 24]
            if results_24h:
                best = max(results_24h, key=lambda x: x.auc)
                n_states = len([s for s in best.state_aucs if best.state_aucs[s] > 0])
                calib_err = np.mean([b['calibration_error'] for b in best.calibration_bins.values()]) if best.calibration_bins else 0
                print(f"    │ {pair:6s} │ {best.duration_config:10s} │ {best.auc:>7.3f} │ "
                      f"{best.brier_score:>7.3f} │ {calib_err:>7.3f} │ {n_states:>7} │")
        
        print(f"    └────────┴────────────┴─────────┴─────────┴─────────┴─────────┘")
    
    print(f"\n{'='*90}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*90}")
    
    return all_results


if __name__ == "__main__":
    results = main()