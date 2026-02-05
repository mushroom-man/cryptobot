#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transition Detection Feature Analysis
======================================
Analyze whether D, D', D'', D''' can predict regime transitions.

Compares two regime configurations:
    A = Baseline: 24h@24h, 72h@72h, 168h@168h (static)
    B = Hourly 24h: 24h@1h, 72h@72h, 168h@168h (hourly scan)

For each configuration:
    1. Compute hourly features (D, D', D'', D''')
    2. Label transitions (state change within N hours)
    3. Analyze feature distributions (transition vs no-transition)
    4. Test predictive power (logistic regression)
    5. Measure early warning lead time

Usage:
    python transition_detection_backtest.py --pair ETHUSD
    python transition_detection_backtest.py --all-pairs
    python transition_detection_backtest.py --horizon 24
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
    print("Warning: Database not available. Using sample data generation.")

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.preprocessing import StandardScaler
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked from validation)
MA_PERIOD_24H = 16      # 16 bars on 24h = 384 hours
MA_PERIOD_72H = 6       # 6 bars on 72h = 432 hours  
MA_PERIOD_168H = 2      # 2 bars on 168h = 336 hours

# Hourly equivalents
MA_HOURS_24H = MA_PERIOD_24H * 24    # 384 hours
MA_HOURS_72H = MA_PERIOD_72H * 72    # 432 hours
MA_HOURS_168H = MA_PERIOD_168H * 168  # 336 hours

# Regime detection buffers
ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Transition horizons to test (hours)
HORIZONS = [6, 12, 24, 48, 72]

# Smoothing windows for derivatives (hours)
SMOOTH_WINDOWS = [1, 3, 6]  # 1 = no smoothing

# Minimum training samples
MIN_TRAIN_SAMPLES = 500


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FeatureAnalysisResult:
    """Results from feature analysis."""
    pair: str
    config: str
    horizon: int
    n_transitions: int
    n_non_transitions: int
    transition_rate: float
    
    # Per-feature metrics
    feature_aucs: Dict[str, float] = field(default_factory=dict)
    feature_means_transition: Dict[str, float] = field(default_factory=dict)
    feature_means_no_transition: Dict[str, float] = field(default_factory=dict)
    
    # Combined model metrics
    model_auc: float = 0.0
    model_precision_at_50: float = 0.0
    model_recall_at_50: float = 0.0
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Early warning
    avg_lead_time_hours: float = 0.0


# =============================================================================
# MA AND REGIME COMPUTATION
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to specified timeframe."""
    return df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_mas_baseline(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAs using baseline configuration (all fixed timeframes).
    Returns hourly-aligned MA values.
    """
    # Resample to standard timeframes
    df_24h = resample_ohlcv(df_1h, '24h')
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    # Compute MAs on native timeframes
    ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Align to hourly (forward fill)
    mas = pd.DataFrame(index=df_1h.index)
    mas['ma_24h'] = ma_24h.reindex(df_1h.index, method='ffill')
    mas['ma_72h'] = ma_72h.reindex(df_1h.index, method='ffill')
    mas['ma_168h'] = ma_168h.reindex(df_1h.index, method='ffill')
    mas['price'] = df_1h['close']
    
    return mas


def compute_mas_hourly_24h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAs with hourly-updated 24h MA, fixed 72h and 168h.
    Returns hourly-aligned MA values.
    """
    # 24h MA: rolling on hourly data
    ma_24h_hourly = df_1h['close'].rolling(MA_HOURS_24H).mean()
    
    # 72h and 168h: fixed timeframes
    df_72h = resample_ohlcv(df_1h, '72h')
    df_168h = resample_ohlcv(df_1h, '168h')
    
    ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
    ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
    
    # Align to hourly
    mas = pd.DataFrame(index=df_1h.index)
    mas['ma_24h'] = ma_24h_hourly
    mas['ma_72h'] = ma_72h.reindex(df_1h.index, method='ffill')
    mas['ma_168h'] = ma_168h.reindex(df_1h.index, method='ffill')
    mas['price'] = df_1h['close']
    
    return mas


def label_trend_binary(close: float, ma: float, current_trend: int,
                       entry_buffer: float = ENTRY_BUFFER,
                       exit_buffer: float = EXIT_BUFFER) -> int:
    """Binary trend detection with hysteresis for single observation."""
    if pd.isna(ma):
        return current_trend
    
    if current_trend == 1:  # Currently bullish
        if close < ma * (1 - exit_buffer) and close < ma * (1 - entry_buffer):
            return 0
    else:  # Currently bearish
        if close > ma * (1 + exit_buffer) and close > ma * (1 + entry_buffer):
            return 1
    
    return current_trend


def compute_regime_states(mas: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 16-state regime from MA data.
    Returns DataFrame with hourly state labels.
    """
    states = pd.DataFrame(index=mas.index)
    
    # Trend detection with hysteresis
    trend_24h = []
    trend_168h = []
    current_24h = 1
    current_168h = 1
    
    for idx in mas.index:
        price = mas.loc[idx, 'price']
        ma24 = mas.loc[idx, 'ma_24h']
        ma168 = mas.loc[idx, 'ma_168h']
        
        current_24h = label_trend_binary(price, ma24, current_24h)
        current_168h = label_trend_binary(price, ma168, current_168h)
        
        trend_24h.append(current_24h)
        trend_168h.append(current_168h)
    
    states['trend_24h'] = trend_24h
    states['trend_168h'] = trend_168h
    
    # MA relationships
    states['ma72_above_ma24'] = (mas['ma_72h'] > mas['ma_24h']).astype(int)
    states['ma168_above_ma24'] = (mas['ma_168h'] > mas['ma_24h']).astype(int)
    
    # Combined 16-state encoding
    states['state'] = (
        states['trend_24h'] * 8 +
        states['trend_168h'] * 4 +
        states['ma72_above_ma24'] * 2 +
        states['ma168_above_ma24'] * 1
    )
    
    return states


# =============================================================================
# DERIVATIVE COMPUTATION
# =============================================================================

def compute_derivatives(price: pd.Series, ma: pd.Series, 
                        smooth_window: int = 1) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute D, D', D'', D''' relative to MA.
    
    Args:
        price: Price series
        ma: Moving average series
        smooth_window: Smoothing window for D before differencing (1 = no smoothing)
    
    Returns:
        Tuple of (D, D', D'', D''')
    """
    # Distance (0th derivative)
    D = (price - ma) / ma
    
    # Optional smoothing before differencing
    if smooth_window > 1:
        D = D.rolling(smooth_window, min_periods=1).mean()
    
    # Velocity (1st derivative)
    D_prime = D.diff()
    
    # Acceleration (2nd derivative)
    D_double_prime = D_prime.diff()
    
    # Jerk (3rd derivative)
    D_triple_prime = D_double_prime.diff()
    
    return D, D_prime, D_double_prime, D_triple_prime


def compute_all_features(mas: pd.DataFrame, smooth_window: int = 1) -> pd.DataFrame:
    """
    Compute all derivative features for all MAs.
    
    Returns DataFrame with columns:
        D_24, D_72, D_168
        D_prime_24, D_prime_72, D_prime_168
        D_double_prime_24, D_double_prime_72, D_double_prime_168
        D_triple_prime_24, D_triple_prime_72, D_triple_prime_168
        vol_ratio
        duration
    """
    features = pd.DataFrame(index=mas.index)
    
    price = mas['price']
    
    # Derivatives for each MA
    for ma_name, col in [('24', 'ma_24h'), ('72', 'ma_72h'), ('168', 'ma_168h')]:
        ma = mas[col]
        D, D_prime, D_double_prime, D_triple_prime = compute_derivatives(price, ma, smooth_window)
        
        features[f'D_{ma_name}'] = D
        features[f'D_prime_{ma_name}'] = D_prime
        features[f'D_double_prime_{ma_name}'] = D_double_prime
        features[f'D_triple_prime_{ma_name}'] = D_triple_prime
    
    # Volatility ratio: short-term / long-term
    returns = price.pct_change()
    vol_6h = returns.rolling(6).std()
    vol_24h = returns.rolling(24).std()
    features['vol_ratio'] = vol_6h / vol_24h
    
    return features


def add_duration_feature(features: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Add duration-in-state feature (hours in current state)."""
    state_series = states['state']
    
    # Count consecutive hours in same state
    duration = []
    current_duration = 0
    prev_state = None
    
    for idx in state_series.index:
        state = state_series.loc[idx]
        if state == prev_state:
            current_duration += 1
        else:
            current_duration = 1
        duration.append(current_duration)
        prev_state = state
    
    features['duration'] = duration
    
    return features


# =============================================================================
# TRANSITION LABELING
# =============================================================================

def label_transitions(states: pd.DataFrame, horizon_hours: int) -> pd.Series:
    """
    Label each hour with whether a transition occurs within horizon.
    
    Returns:
        Series with 1 = transition within horizon, 0 = no transition
    """
    state_series = states['state']
    
    # For each hour, check if state differs within horizon
    labels = []
    
    for i in range(len(state_series)):
        current_state = state_series.iloc[i]
        
        # Look ahead up to horizon hours
        future_end = min(i + horizon_hours, len(state_series))
        future_states = state_series.iloc[i+1:future_end]
        
        if len(future_states) == 0:
            labels.append(np.nan)
        elif (future_states != current_state).any():
            labels.append(1)
        else:
            labels.append(0)
    
    return pd.Series(labels, index=states.index)


def compute_lead_time(states: pd.DataFrame, features: pd.DataFrame,
                      threshold_feature: str, threshold_value: float,
                      horizon_hours: int = 24) -> float:
    """
    Compute average lead time: hours before transition that feature crosses threshold.
    """
    state_series = states['state']
    feature_series = features[threshold_feature]
    
    # Find transition points
    transitions = state_series.diff().abs() > 0
    transition_indices = transitions[transitions].index
    
    lead_times = []
    
    for trans_idx in transition_indices:
        # Look back up to horizon hours
        loc = states.index.get_loc(trans_idx)
        lookback_start = max(0, loc - horizon_hours)
        lookback_slice = feature_series.iloc[lookback_start:loc]
        
        # Find first hour where feature crossed threshold
        crossed = lookback_slice < threshold_value
        if crossed.any():
            first_cross = crossed.idxmax()
            lead_time = (trans_idx - first_cross).total_seconds() / 3600
            lead_times.append(lead_time)
    
    return np.mean(lead_times) if lead_times else 0.0


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================

def analyze_feature_distributions(features: pd.DataFrame, labels: pd.Series,
                                   feature_cols: List[str]) -> Dict:
    """
    Analyze feature distributions for transitions vs non-transitions.
    """
    # Align and drop NaNs
    valid_idx = labels.notna() & features[feature_cols].notna().all(axis=1)
    X = features.loc[valid_idx, feature_cols]
    y = labels.loc[valid_idx]
    
    results = {}
    
    for col in feature_cols:
        trans_vals = X.loc[y == 1, col]
        no_trans_vals = X.loc[y == 0, col]
        
        results[col] = {
            'mean_transition': trans_vals.mean(),
            'mean_no_transition': no_trans_vals.mean(),
            'std_transition': trans_vals.std(),
            'std_no_transition': no_trans_vals.std(),
            'diff': trans_vals.mean() - no_trans_vals.mean(),
        }
    
    return results


def compute_feature_aucs(features: pd.DataFrame, labels: pd.Series,
                          feature_cols: List[str]) -> Dict[str, float]:
    """
    Compute univariate AUC for each feature.
    """
    valid_idx = labels.notna() & features[feature_cols].notna().all(axis=1)
    X = features.loc[valid_idx, feature_cols]
    y = labels.loc[valid_idx]
    
    aucs = {}
    
    for col in feature_cols:
        try:
            # Handle both positive and negative correlations
            auc = roc_auc_score(y, X[col])
            # If AUC < 0.5, the feature predicts the opposite - flip it
            aucs[col] = max(auc, 1 - auc)
        except:
            aucs[col] = 0.5
    
    return aucs


def train_logistic_model(features: pd.DataFrame, labels: pd.Series,
                          feature_cols: List[str], 
                          train_ratio: float = 0.7) -> Tuple[LogisticRegression, Dict]:
    """
    Train logistic regression model using expanding window approach.
    
    Returns:
        Tuple of (model, metrics_dict)
    """
    # Align and drop NaNs
    valid_idx = labels.notna() & features[feature_cols].notna().all(axis=1)
    X = features.loc[valid_idx, feature_cols].values
    y = labels.loc[valid_idx].values
    
    if len(y) < MIN_TRAIN_SAMPLES:
        return None, {'error': 'Insufficient samples'}
    
    # Time-based split (expanding window)
    train_size = int(len(y) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.5
    
    # Precision at various thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find precision/recall at ~50% threshold
    idx_50 = np.argmin(np.abs(thresholds - 0.5)) if len(thresholds) > 0 else 0
    
    # Feature importance (coefficients)
    importance = dict(zip(feature_cols, np.abs(model.coef_[0])))
    
    metrics = {
        'auc': auc,
        'precision_at_50': precision[idx_50] if idx_50 < len(precision) else 0,
        'recall_at_50': recall[idx_50] if idx_50 < len(recall) else 0,
        'feature_importance': importance,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'transition_rate_train': y_train.mean(),
        'transition_rate_test': y_test.mean(),
    }
    
    return model, metrics


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_pair(df_1h: pd.DataFrame, pair: str, 
                 horizons: List[int] = HORIZONS,
                 smooth_window: int = 1) -> List[FeatureAnalysisResult]:
    """
    Run full transition detection analysis for one pair.
    """
    results = []
    
    # Two configurations to test
    configs = {
        'A_baseline': compute_mas_baseline,
        'B_hourly_24h': compute_mas_hourly_24h,
    }
    
    for config_name, ma_func in configs.items():
        print(f"\n    Config: {config_name}")
        
        # Compute MAs and states
        mas = ma_func(df_1h)
        states = compute_regime_states(mas)
        
        # Compute features
        features = compute_all_features(mas, smooth_window)
        features = add_duration_feature(features, states)
        
        # Feature columns
        feature_cols = [
            'D_24', 'D_72', 'D_168',
            'D_prime_24', 'D_prime_72', 'D_prime_168',
            'D_double_prime_24', 'D_double_prime_72', 'D_double_prime_168',
            'D_triple_prime_24', 'D_triple_prime_72', 'D_triple_prime_168',
            'vol_ratio', 'duration'
        ]
        
        for horizon in horizons:
            print(f"      Horizon: {horizon}h...", end=" ", flush=True)
            
            # Label transitions
            labels = label_transitions(states, horizon)
            
            # Count transitions
            valid_labels = labels.dropna()
            n_transitions = int(valid_labels.sum())
            n_non_transitions = int(len(valid_labels) - n_transitions)
            transition_rate = n_transitions / len(valid_labels) if len(valid_labels) > 0 else 0
            
            # Feature AUCs
            feature_aucs = compute_feature_aucs(features, labels, feature_cols)
            
            # Feature distributions
            distributions = analyze_feature_distributions(features, labels, feature_cols)
            
            # Train combined model
            model, metrics = train_logistic_model(features, labels, feature_cols)
            
            # Early warning lead time (using D''_24 as example)
            lead_time = compute_lead_time(
                states, features, 
                'D_double_prime_24', 
                threshold_value=-0.001,  # Negative acceleration
                horizon_hours=horizon
            )
            
            result = FeatureAnalysisResult(
                pair=pair,
                config=config_name,
                horizon=horizon,
                n_transitions=n_transitions,
                n_non_transitions=n_non_transitions,
                transition_rate=transition_rate,
                feature_aucs=feature_aucs,
                feature_means_transition={k: v['mean_transition'] for k, v in distributions.items()},
                feature_means_no_transition={k: v['mean_no_transition'] for k, v in distributions.items()},
                model_auc=metrics.get('auc', 0.5),
                model_precision_at_50=metrics.get('precision_at_50', 0),
                model_recall_at_50=metrics.get('recall_at_50', 0),
                feature_importance=metrics.get('feature_importance', {}),
                avg_lead_time_hours=lead_time,
            )
            
            results.append(result)
            print(f"AUC={result.model_auc:.3f}")
    
    return results


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(results: List[FeatureAnalysisResult]):
    """Display analysis results in formatted tables."""
    
    print("\n" + "=" * 100)
    print("TRANSITION DETECTION FEATURE ANALYSIS")
    print("=" * 100)
    
    # Group by pair and config
    pairs = sorted(set(r.pair for r in results))
    configs = sorted(set(r.config for r in results))
    horizons = sorted(set(r.horizon for r in results))
    
    # Summary table
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │  MODEL PERFORMANCE BY CONFIGURATION AND HORIZON                                             │
    ├────────┬─────────────────┬─────────┬───────────┬───────────┬───────────┬───────────────────┤
    │ Pair   │ Config          │ Horizon │ Trans Rate│ Model AUC │ Lead Time │ Best Feature AUC  │
    ├────────┼─────────────────┼─────────┼───────────┼───────────┼───────────┼───────────────────┤""")
    
    for pair in pairs:
        for config in configs:
            for horizon in horizons:
                r = next((x for x in results if x.pair == pair and x.config == config and x.horizon == horizon), None)
                if r:
                    best_feature = max(r.feature_aucs.items(), key=lambda x: x[1])
                    print(f"    │ {pair:6s} │ {config:15s} │ {horizon:>5}h  │ {r.transition_rate:>8.1%} │ "
                          f"{r.model_auc:>9.3f} │ {r.avg_lead_time_hours:>8.1f}h │ {best_feature[0]:12s} {best_feature[1]:.3f} │")
            print(f"    ├────────┼─────────────────┼─────────┼───────────┼───────────┼───────────┼───────────────────┤")
    
    print(f"""    └────────┴─────────────────┴─────────┴───────────┴───────────┴───────────┴───────────────────┘""")
    
    # Feature importance summary (aggregate across all results)
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  FEATURE IMPORTANCE (AVERAGE ACROSS ALL CONFIGS/HORIZONS)      │
    ├────────────────────────────┬──────────────┬─────────────────────┤
    │ Feature                    │ Avg AUC      │ Avg Importance      │
    ├────────────────────────────┼──────────────┼─────────────────────┤""")
    
    # Aggregate feature metrics
    feature_cols = [
        'D_24', 'D_72', 'D_168',
        'D_prime_24', 'D_prime_72', 'D_prime_168',
        'D_double_prime_24', 'D_double_prime_72', 'D_double_prime_168',
        'D_triple_prime_24', 'D_triple_prime_72', 'D_triple_prime_168',
        'vol_ratio', 'duration'
    ]
    
    for feat in feature_cols:
        aucs = [r.feature_aucs.get(feat, 0.5) for r in results]
        importances = [r.feature_importance.get(feat, 0) for r in results]
        avg_auc = np.mean(aucs)
        avg_imp = np.mean(importances)
        
        # Highlight derivatives
        marker = "***" if "double_prime" in feat or "triple_prime" in feat else ""
        print(f"    │ {feat:26s} │ {avg_auc:>10.3f}   │ {avg_imp:>17.3f}   │ {marker}")
    
    print(f"""    └────────────────────────────┴──────────────┴─────────────────────┘
    
    *** = D'' (acceleration) or D''' (jerk) features
    """)
    
    # Comparison: Baseline vs Hourly
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  BASELINE vs HOURLY 24H COMPARISON (24h Horizon)               │
    ├────────┬──────────────────┬──────────────────┬─────────────────┤
    │ Pair   │ Baseline AUC     │ Hourly 24h AUC   │ Difference      │
    ├────────┼──────────────────┼──────────────────┼─────────────────┤""")
    
    for pair in pairs:
        baseline = next((r for r in results if r.pair == pair and 'baseline' in r.config and r.horizon == 24), None)
        hourly = next((r for r in results if r.pair == pair and 'hourly' in r.config and r.horizon == 24), None)
        
        if baseline and hourly:
            diff = hourly.model_auc - baseline.model_auc
            diff_str = f"{diff:+.3f}" if diff != 0 else "  0.000"
            print(f"    │ {pair:6s} │ {baseline.model_auc:>14.3f}   │ {hourly.model_auc:>14.3f}   │ {diff_str:>13s}   │")
    
    print(f"""    └────────┴──────────────────┴──────────────────┴─────────────────┘""")


def display_derivative_comparison(results: List[FeatureAnalysisResult]):
    """Show D vs D' vs D'' vs D''' comparison."""
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │  DERIVATIVE ORDER COMPARISON (24h Horizon, Baseline Config)                         │
    ├────────┬──────────────┬──────────────┬──────────────┬──────────────┬───────────────┤
    │ MA     │ D (distance) │ D' (velocity)│ D'' (accel)  │ D''' (jerk)  │ Best          │
    ├────────┼──────────────┼──────────────┼──────────────┼──────────────┼───────────────┤""")
    
    baseline_24h = [r for r in results if 'baseline' in r.config and r.horizon == 24]
    
    if baseline_24h:
        # Average across pairs
        for ma in ['24', '72', '168']:
            d_aucs = [r.feature_aucs.get(f'D_{ma}', 0.5) for r in baseline_24h]
            d_prime_aucs = [r.feature_aucs.get(f'D_prime_{ma}', 0.5) for r in baseline_24h]
            d_double_aucs = [r.feature_aucs.get(f'D_double_prime_{ma}', 0.5) for r in baseline_24h]
            d_triple_aucs = [r.feature_aucs.get(f'D_triple_prime_{ma}', 0.5) for r in baseline_24h]
            
            avg_d = np.mean(d_aucs)
            avg_d_prime = np.mean(d_prime_aucs)
            avg_d_double = np.mean(d_double_aucs)
            avg_d_triple = np.mean(d_triple_aucs)
            
            avgs = {"D": avg_d, "D'": avg_d_prime, "D''": avg_d_double, "D'''": avg_d_triple}
            best = max(avgs.items(), key=lambda x: x[1])
            
            print(f"    │ {ma:6s} │ {avg_d:>10.3f}   │ {avg_d_prime:>10.3f}   │ "
                  f"{avg_d_double:>10.3f}   │ {avg_d_triple:>10.3f}   │ {best[0]:>11s}   │")
    
    print(f"""    └────────┴──────────────┴──────────────┴──────────────┴──────────────┴───────────────┘""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transition Detection Feature Analysis')
    parser.add_argument('--pair', type=str, default=None, 
                        help='Analyze specific pair (e.g., ETHUSD)')
    parser.add_argument('--all-pairs', action='store_true',
                        help='Analyze all deployment pairs')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Specific horizon to test (hours)')
    parser.add_argument('--smooth', type=int, default=1,
                        help='Smoothing window for derivatives (1=none)')
    args = parser.parse_args()
    
    print("=" * 100)
    print("TRANSITION DETECTION FEATURE ANALYSIS")
    print("=" * 100)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  OBJECTIVE                                                                                    ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  Test whether D, D', D'', D''' can predict regime transitions.                                ║
    ║                                                                                               ║
    ║  CONFIGURATIONS:                                                                              ║
    ║    A = Baseline: 24h@24h, 72h@72h, 168h@168h                                                  ║
    ║    B = Hourly 24h: 24h@1h, 72h@72h, 168h@168h                                                 ║
    ║                                                                                               ║
    ║  FEATURES:                                                                                    ║
    ║    D     = Distance from MA (normalized)                                                      ║
    ║    D'    = Velocity (rate of approach)                                                        ║
    ║    D''   = Acceleration (change in velocity)                                                  ║
    ║    D'''  = Jerk (change in acceleration)                                                      ║
    ║                                                                                               ║
    ║  HORIZONS: 6h, 12h, 24h, 48h, 72h                                                             ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Determine pairs to analyze
    if args.pair:
        pairs = [args.pair]
    elif args.all_pairs:
        pairs = DEPLOY_PAIRS
    else:
        pairs = ['ETHUSD']  # Default to one pair for quick testing
    
    # Determine horizons
    horizons = [args.horizon] if args.horizon else HORIZONS
    
    print(f"  Pairs: {pairs}")
    print(f"  Horizons: {horizons}")
    print(f"  Smoothing window: {args.smooth}")
    
    # Connect to database
    if HAS_DATABASE:
        print("\n  Connecting to database...")
        db = Database()
    else:
        print("\n  No database - would need sample data")
        return
    
    # Run analysis
    all_results = []
    
    for pair in pairs:
        print(f"\n  Analyzing {pair}...")
        
        # Load hourly data
        df_1h = db.get_ohlcv(pair)
        print(f"    Loaded {len(df_1h)} hourly bars")
        
        # Run analysis
        results = analyze_pair(df_1h, pair, horizons, args.smooth)
        all_results.extend(results)
    
    # Display results
    display_results(all_results)
    display_derivative_comparison(all_results)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    results = main()