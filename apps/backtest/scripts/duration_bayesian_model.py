#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duration-Based Bayesian Transition Model v2 (FIXED)
====================================================
Fixed version with correct state occupancy tracking.

Bug in v1: Only counted state entries when transitions occurred.
Fix: Count every day in a state as an observation.

Key Metrics:
    - n_days: Total days spent in this state
    - n_transitions: Number of times we LEFT this state
    - daily_transition_rate: n_transitions / n_days (should be << 100%)

Usage:
    python duration_bayesian_model_v2.py --all-pairs
    python duration_bayesian_model_v2.py --all-pairs --with-trading-sim
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

# Duration multiplier configurations
DURATION_CONFIGS = {
    'linear': {'type': 'linear'},
    'exponential': {'type': 'exponential', 'decay': 0.5},
    'step': {'type': 'step', 'thresholds': [0.5, 1.0, 1.5, 2.0]},
    'log': {'type': 'log'},
    'none': {'type': 'none'},  # Base rate only, no duration adjustment
}

# Trading simulation parameters
INITIAL_CAPITAL = 100000.0
TRADING_COST = 0.0020


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StateStats:
    """Statistics for a single state - FIXED VERSION."""
    state_id: int
    n_days: int = 0              # Total days spent in this state
    n_transitions_out: int = 0   # Number of times we LEFT this state
    durations: List[float] = field(default_factory=list)  # Duration of each visit
    
    @property
    def daily_transition_rate(self) -> float:
        """Probability of transitioning OUT on any given day."""
        if self.n_days == 0:
            return 0.5
        return self.n_transitions_out / self.n_days
    
    @property
    def avg_duration_days(self) -> float:
        """Average duration of visits to this state."""
        if not self.durations:
            return 2.0  # Default
        return np.mean(self.durations)
    
    @property
    def median_duration_days(self) -> float:
        if not self.durations:
            return 2.0
        return np.median(self.durations)


@dataclass 
class TransitionMatrix:
    """16x16 transition matrix with FIXED state tracking."""
    pair: str
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((16, 16)))
    state_stats: Dict[int, StateStats] = field(default_factory=dict)
    
    def __post_init__(self):
        for i in range(16):
            self.state_stats[i] = StateStats(state_id=i)
    
    def record_day_in_state(self, state: int):
        """Record one day spent in a state."""
        self.state_stats[state].n_days += 1
    
    def record_transition(self, from_state: int, to_state: int, duration_days: float):
        """Record a transition from one state to another."""
        self.matrix[from_state, to_state] += 1
        self.state_stats[from_state].n_transitions_out += 1
        self.state_stats[from_state].durations.append(duration_days)
    
    def get_transition_probs(self, from_state: int) -> np.ndarray:
        """Get transition probabilities from a state (given we leave)."""
        row = self.matrix[from_state, :]
        total = row.sum()
        if total > 0:
            return row / total
        return np.ones(16) / 16
    
    def get_daily_leave_prob(self, state: int) -> float:
        """P(leave state today | in state today)."""
        return self.state_stats[state].daily_transition_rate


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
    """Compute 16-state regime at daily frequency."""
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    ma_72h_daily = ma_72h.reindex(df_24h.index, method='ffill')
    ma_168h_daily = ma_168h.reindex(df_24h.index, method='ffill')
    
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
    
    states['state'] = (
        states['trend_24h'] * 8 +
        states['trend_168h'] * 4 +
        states['ma72_above_ma24'] * 2 +
        states['ma168_above_ma24'] * 1
    )
    
    return states.dropna()


def compute_state_durations(states: pd.DataFrame) -> pd.DataFrame:
    """Add duration column: days in current state."""
    state_series = states['state'].values
    
    durations = []
    current_duration = 0
    prev_state = None
    
    for state in state_series:
        if state == prev_state:
            current_duration += 1
        else:
            current_duration = 1
        durations.append(current_duration)
        prev_state = state
    
    states = states.copy()
    states['duration_days'] = durations
    
    return states


# =============================================================================
# TRANSITION MATRIX BUILDING - FIXED
# =============================================================================

def build_transition_matrix(states: pd.DataFrame, end_date: pd.Timestamp) -> TransitionMatrix:
    """
    Build transition matrix from historical data up to end_date.
    
    FIXED: Now properly counts:
        - Every day spent in each state (for base rate calculation)
        - Transitions out of each state (for transition probability)
    """
    matrix = TransitionMatrix(pair="")
    
    train_states = states[states.index < end_date]
    
    if len(train_states) < 2:
        return matrix
    
    state_series = train_states['state'].values
    
    # Track current visit duration
    current_state = state_series[0]
    visit_duration = 1
    
    # Record first day
    matrix.record_day_in_state(current_state)
    
    for i in range(1, len(state_series)):
        curr_state = state_series[i]
        prev_state = state_series[i-1]
        
        # Record this day in the current state
        matrix.record_day_in_state(curr_state)
        
        if curr_state != prev_state:
            # Transition occurred - record it with the duration of the visit that just ended
            matrix.record_transition(prev_state, curr_state, visit_duration)
            visit_duration = 1
        else:
            visit_duration += 1
    
    return matrix


# =============================================================================
# DURATION MULTIPLIER FUNCTIONS
# =============================================================================

def get_duration_multiplier(duration_days: float, avg_duration: float, config: Dict) -> float:
    """Get duration multiplier based on config."""
    config_type = config['type']
    
    if config_type == 'none':
        return 1.0
    
    if avg_duration <= 0:
        avg_duration = 2.0
    
    ratio = duration_days / avg_duration
    
    if config_type == 'linear':
        # Linear: multiplier proportional to time ratio
        return min(max(ratio, 0.1), 3.0)
    
    elif config_type == 'exponential':
        # Exponential: accelerating increase
        decay = config.get('decay', 0.5)
        return min(np.exp(decay * (ratio - 1)), 5.0)
    
    elif config_type == 'step':
        # Step function
        thresholds = config.get('thresholds', [0.5, 1.0, 1.5, 2.0])
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
    
    elif config_type == 'log':
        # Logarithmic: diminishing returns
        return 1.0 + np.log(1 + ratio) / 2
    
    return 1.0


# =============================================================================
# BAYESIAN PREDICTION
# =============================================================================

def predict_transition_prob(
    state: int,
    duration_days: float,
    matrix: TransitionMatrix,
    duration_config: Dict,
    horizon_days: int = 1
) -> float:
    """
    Predict P(transition within horizon | state, duration).
    
    FIXED formula:
        base_rate = daily_transition_rate (from historical data)
        adjusted_rate = base_rate * duration_multiplier
        horizon_prob = 1 - (1 - adjusted_rate)^horizon_days
    """
    stats = matrix.state_stats[state]
    
    # If insufficient data, return uninformative prior
    if stats.n_days < MIN_STATE_SAMPLES:
        return 0.5
    
    # Base daily transition rate (FIXED: now correctly calculated)
    base_rate = stats.daily_transition_rate
    
    # Duration multiplier
    avg_duration = stats.avg_duration_days
    multiplier = get_duration_multiplier(duration_days, avg_duration, duration_config)
    
    # Adjusted daily rate (capped at 0.95)
    adjusted_daily_rate = min(base_rate * multiplier, 0.95)
    
    # Convert to horizon probability
    # P(transition within N days) = 1 - P(no transition for N days)
    # P(no transition for N days) = (1 - daily_rate)^N
    horizon_prob = 1 - (1 - adjusted_daily_rate) ** horizon_days
    
    return min(max(horizon_prob, 0.01), 0.99)


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    states: pd.DataFrame,
    duration_config: Dict,
    horizon_days: int = 1,
    train_ratio: float = 0.6
) -> ModelResult:
    """Evaluate transition prediction model using expanding window."""
    
    config_name = duration_config['type']
    result = ModelResult(
        pair="",
        horizon=horizon_days * 24,
        duration_config=config_name
    )
    
    n_days = len(states)
    train_end_idx = int(n_days * train_ratio)
    
    predictions = []
    actuals = []
    state_predictions = {i: {'pred': [], 'actual': []} for i in range(16)}
    
    for i in range(train_end_idx, n_days - horizon_days):
        current_date = states.index[i]
        current_state = int(states.iloc[i]['state'])
        current_duration = states.iloc[i]['duration_days']
        
        # Build matrix from data up to current date
        matrix = build_transition_matrix(states, current_date)
        
        # Predict
        pred_prob = predict_transition_prob(
            current_state, current_duration, matrix, duration_config, horizon_days
        )
        
        # Actual: did transition occur within horizon?
        future_idx = i + horizon_days
        if future_idx < n_days:
            future_state = int(states.iloc[future_idx]['state'])
            # Check if ANY transition occurred in the horizon window
            states_in_window = states.iloc[i+1:future_idx+1]['state'].values
            actual = 1 if any(s != current_state for s in states_in_window) else 0
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
    result.avg_predicted_prob = float(predictions.mean())
    result.avg_actual_rate = float(actuals.mean())
    
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
                'avg_predicted': float(bin_preds.mean()),
                'avg_actual': float(bin_acts.mean()),
                'calibration_error': float(abs(bin_preds.mean() - bin_acts.mean()))
            }
    
    return result


# =============================================================================
# TRADING SIMULATION
# =============================================================================

def run_trading_simulation(
    states: pd.DataFrame,
    returns: pd.Series,
    duration_config: Dict,
    horizon_days: int = 1,
    train_ratio: float = 0.6
) -> Dict:
    """Simulate trading with position sizing based on P(transition)."""
    
    n_days = len(states)
    train_end_idx = int(n_days * train_ratio)
    
    common_idx = states.index.intersection(returns.index)
    states = states.loc[common_idx]
    returns = returns.loc[common_idx]
    n_days = len(states)
    train_end_idx = int(n_days * train_ratio)
    
    equity_baseline = [INITIAL_CAPITAL]
    equity_adjusted = [INITIAL_CAPITAL]
    
    position_sizes = []
    transition_probs = []
    
    for i in range(train_end_idx, n_days - 1):
        current_date = states.index[i]
        next_date = states.index[i + 1]
        current_state = int(states.iloc[i]['state'])
        current_duration = states.iloc[i]['duration_days']
        
        matrix = build_transition_matrix(states, current_date)
        
        pred_prob = predict_transition_prob(
            current_state, current_duration, matrix, duration_config, horizon_days
        )
        transition_probs.append(pred_prob)
        
        # Position sizing
        if pred_prob < 0.3:
            position_mult = 1.0
        elif pred_prob < 0.5:
            position_mult = 0.8
        elif pred_prob < 0.7:
            position_mult = 0.6
        else:
            position_mult = 0.3
        
        position_sizes.append(position_mult)
        
        if next_date in returns.index:
            daily_return = returns.loc[next_date]
        else:
            daily_return = 0.0
        
        equity_baseline.append(equity_baseline[-1] * (1 + daily_return))
        equity_adjusted.append(equity_adjusted[-1] * (1 + daily_return * position_mult))
    
    equity_baseline = np.array(equity_baseline)
    equity_adjusted = np.array(equity_adjusted)
    
    returns_baseline = np.diff(equity_baseline) / equity_baseline[:-1]
    returns_adjusted = np.diff(equity_adjusted) / equity_adjusted[:-1]
    
    def calc_metrics(returns_arr, equity_arr):
        total_return = (equity_arr[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
        years = len(returns_arr) / 365
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(365) if returns_arr.std() > 0 else 0
        
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
    """Analyze duration characteristics per state - FIXED VERSION."""
    matrix = build_transition_matrix(states, states.index[-1])
    
    rows = []
    for state_id in range(16):
        stats = matrix.state_stats[state_id]
        if stats.n_days > 0:
            rows.append({
                'state': state_id,
                'n_days': stats.n_days,
                'n_transitions': stats.n_transitions_out,
                'daily_trans_rate': stats.daily_transition_rate,
                'avg_duration_days': stats.avg_duration_days,
                'median_duration_days': stats.median_duration_days,
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# DISPLAY
# =============================================================================

def display_state_analysis(state_df: pd.DataFrame, pair: str):
    """Display per-state duration analysis - FIXED VERSION."""
    
    print(f"""
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │  STATE DURATION ANALYSIS: {pair:6s} (FIXED)                                           │
    ├───────┬──────────┬────────────┬──────────────┬────────────────┬─────────────────────┤
    │ State │ N Days   │ N Trans Out│ Daily Rate   │ Avg Duration   │ Median Duration     │
    ├───────┼──────────┼────────────┼──────────────┼────────────────┼─────────────────────┤""")
    
    for _, row in state_df.iterrows():
        print(f"    │ {int(row['state']):>5} │ {int(row['n_days']):>8} │ {int(row['n_transitions']):>10} │ "
              f"{row['daily_trans_rate']:>10.1%}   │ {row['avg_duration_days']:>12.1f}d  │ "
              f"{row['median_duration_days']:>17.1f}d  │")
    
    print(f"    └───────┴──────────┴────────────┴──────────────┴────────────────┴─────────────────────┘")
    
    # Summary stats
    total_days = state_df['n_days'].sum()
    total_trans = state_df['n_transitions'].sum()
    overall_rate = total_trans / total_days if total_days > 0 else 0
    
    print(f"""
    Summary:
    ─────────────────────────────────────────
    Total Days:              {total_days:,}
    Total Transitions:       {total_trans:,}
    Overall Daily Trans Rate: {overall_rate:.1%}
    """)


def display_results(results: List[ModelResult], pair: str):
    """Display model evaluation results."""
    
    print(f"\n{'='*90}")
    print(f"DURATION-BASED BAYESIAN MODEL RESULTS: {pair}")
    print(f"{'='*90}")
    
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
                if r.calibration_bins:
                    calib_err = np.mean([b['calibration_error'] for b in r.calibration_bins.values()])
                else:
                    calib_err = 0
                
                print(f"    │ {config:10s} │ {horizon:>5}h  │ {r.auc:>7.3f} │ {r.brier_score:>7.3f} │ "
                      f"{r.avg_predicted_prob:>9.1%}  │ {r.avg_actual_rate:>9.1%}  │ {calib_err:>9.3f}  │")
        print(f"    ├────────────┼─────────┼─────────┼─────────┼────────────┼────────────┼────────────┤")
    
    print(f"    └────────────┴─────────┴─────────┴─────────┴────────────┴────────────┴────────────┘")
    
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
    parser = argparse.ArgumentParser(description='Duration-Based Bayesian Transition Model v2')
    parser.add_argument('--pair', type=str, default=None, help='Analyze specific pair')
    parser.add_argument('--all-pairs', action='store_true', help='Analyze all pairs')
    parser.add_argument('--with-trading-sim', action='store_true', help='Include trading simulation')
    parser.add_argument('--horizon', type=int, default=None, help='Specific horizon (hours)')
    args = parser.parse_args()
    
    print("=" * 90)
    print("DURATION-BASED BAYESIAN TRANSITION MODEL v2 (FIXED)")
    print("=" * 90)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════════════════════╗
    ║  MODEL DESCRIPTION                                                                     ║
    ╠════════════════════════════════════════════════════════════════════════════════════════╣
    ║  FIXED: Now correctly calculates:                                                      ║
    ║    - n_days: Total days spent in each state                                            ║
    ║    - n_transitions: Times we LEFT each state                                           ║
    ║    - daily_rate: n_transitions / n_days (should be ~30-50%, not 100%)                  ║
    ║                                                                                        ║
    ║  Formula:                                                                              ║
    ║    base_rate = historical daily transition rate for state                              ║
    ║    adjusted_rate = base_rate × duration_multiplier                                     ║
    ║    P(transition in N days) = 1 - (1 - adjusted_rate)^N                                 ║
    ║                                                                                        ║
    ║  Duration Configs: none, linear, exponential, step, log                                ║
    ╚════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if args.pair:
        pairs = [args.pair]
    elif args.all_pairs:
        pairs = DEPLOY_PAIRS
    else:
        pairs = ['ETHUSD']
    
    # Convert horizons from hours to days
    if args.horizon:
        horizon_days_list = [args.horizon // 24]
    else:
        horizon_days_list = [1, 2, 3]  # 24h, 48h, 72h
    
    print(f"  Pairs: {pairs}")
    print(f"  Horizons (days): {horizon_days_list}")
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
        
        df_1h = db.get_ohlcv(pair)
        print(f"  Loaded {len(df_1h)} hourly bars")
        
        states = compute_regime_states_daily(df_1h)
        states = compute_state_durations(states)
        print(f"  Computed {len(states)} daily states")
        
        state_df = analyze_state_durations(states, pair)
        display_state_analysis(state_df, pair)
        
        results = []
        for config_name, config in DURATION_CONFIGS.items():
            for horizon_days in horizon_days_list:
                print(f"  Testing {config_name} @ {horizon_days*24}h...", end=" ", flush=True)
                result = evaluate_model(states, config, horizon_days)
                result.pair = pair
                results.append(result)
                print(f"AUC={result.auc:.3f}")
        
        display_results(results, pair)
        display_calibration(results, pair)
        
        all_results[pair] = results
        
        if args.with_trading_sim:
            print(f"\n  Running trading simulation...")
            
            df_24h = resample_ohlcv(df_1h, '24h')
            returns = df_24h['close'].pct_change()
            
            best = max(results, key=lambda x: x.auc)
            best_config = DURATION_CONFIGS[best.duration_config]
            
            sim_results = run_trading_simulation(
                states, returns, best_config, best.horizon // 24
            )
            display_trading_results(sim_results, pair, best.duration_config)
    
    if len(pairs) > 1:
        print(f"\n{'='*90}")
        print("CROSS-PAIR SUMMARY (24h Horizon)")
        print(f"{'='*90}")
        
        print(f"""
    ┌────────┬────────────┬─────────┬─────────┬─────────┬──────────────┐
    │ Pair   │ Best Config│ AUC     │ Brier   │ Calib   │ Actual Rate  │
    ├────────┼────────────┼─────────┼─────────┼─────────┼──────────────┤""")
        
        for pair in pairs:
            results_24h = [r for r in all_results[pair] if r.horizon == 24]
            if results_24h:
                best = max(results_24h, key=lambda x: x.auc)
                calib_err = np.mean([b['calibration_error'] for b in best.calibration_bins.values()]) if best.calibration_bins else 0
                print(f"    │ {pair:6s} │ {best.duration_config:10s} │ {best.auc:>7.3f} │ "
                      f"{best.brier_score:>7.3f} │ {calib_err:>7.3f} │ {best.avg_actual_rate:>10.1%}   │")
        
        print(f"    └────────┴────────────┴─────────┴─────────┴─────────┴──────────────┘")
    
    print(f"\n{'='*90}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*90}")
    
    return all_results


if __name__ == "__main__":
    results = main()