#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State Transition Analysis
=========================
Analyze the 16x16 transition matrix to understand:
1. Where each state transitions to
2. Duration in state by entry source (path dependence)
3. Exit destinations by entry source
4. Statistical significance of patterns

Key Question: Does where you came FROM affect how long you STAY?

Usage:
    python state_transition_analysis.py --pair ETHUSD
    python state_transition_analysis.py --all-pairs
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
from scipy import stats
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'ETHUSD', 'XMRUSD', 'ADAUSD']

# MA Parameters (locked)
MA_PERIOD_24H = 16
MA_PERIOD_72H = 6
MA_PERIOD_168H = 2

ENTRY_BUFFER = 0.015
EXIT_BUFFER = 0.005

# Minimum samples for reliable statistics
MIN_SAMPLES_FOR_STATS = 10
MIN_SAMPLES_FOR_COMPARISON = 5

# State classification
BULLISH_STATES = [8, 9, 10, 11, 12, 13, 14, 15]  # trend_24h = 1
BEARISH_STATES = [0, 1, 2, 3, 4, 5, 6, 7]         # trend_24h = 0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PathStats:
    """Statistics for a specific transition path (from_state → to_state)."""
    from_state: int
    to_state: int
    count: int = 0
    durations: List[float] = field(default_factory=list)  # How long did to_state last?
    exit_states: List[int] = field(default_factory=list)   # Where did to_state go next?
    
    @property
    def avg_duration(self) -> float:
        return np.mean(self.durations) if self.durations else 0
    
    @property
    def median_duration(self) -> float:
        return np.median(self.durations) if self.durations else 0
    
    @property
    def std_duration(self) -> float:
        return np.std(self.durations) if len(self.durations) > 1 else 0


@dataclass
class StateAnalysis:
    """Complete analysis for a single state."""
    state_id: int
    total_visits: int = 0
    total_days: int = 0
    overall_avg_duration: float = 0
    
    # Duration by entry source
    entry_paths: Dict[int, PathStats] = field(default_factory=dict)
    
    # Exit destinations
    exit_counts: Dict[int, int] = field(default_factory=dict)


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


# =============================================================================
# TRANSITION ANALYSIS
# =============================================================================

def extract_state_visits(states: pd.DataFrame) -> List[Dict]:
    """
    Extract all state visits with entry source, duration, and exit destination.
    
    Returns list of:
        {'from': prev_state, 'state': current_state, 'to': next_state, 'duration': days}
    """
    state_series = states['state'].values
    visits = []
    
    # Track visits
    current_state = state_series[0]
    entry_idx = 0
    prev_state = None  # Unknown for first state
    
    for i in range(1, len(state_series)):
        if state_series[i] != current_state:
            # State ended - record the visit
            duration = i - entry_idx
            exit_state = state_series[i]
            
            visits.append({
                'from': prev_state,
                'state': current_state,
                'to': exit_state,
                'duration': duration,
                'entry_idx': entry_idx,
                'exit_idx': i
            })
            
            # Start new state
            prev_state = current_state
            current_state = state_series[i]
            entry_idx = i
    
    # Don't include final ongoing state (no exit yet)
    
    return visits


def build_transition_matrix(visits: List[Dict]) -> np.ndarray:
    """Build 16x16 transition count matrix."""
    matrix = np.zeros((16, 16), dtype=int)
    
    for v in visits:
        from_state = v['state']
        to_state = v['to']
        matrix[from_state, to_state] += 1
    
    return matrix


def analyze_path_dependence(visits: List[Dict]) -> Dict[int, StateAnalysis]:
    """
    Analyze duration and exit patterns by entry source.
    
    For each state, determine if duration depends on where you came from.
    """
    analyses = {i: StateAnalysis(state_id=i) for i in range(16)}
    
    for v in visits:
        state = v['state']
        from_state = v['from']
        to_state = v['to']
        duration = v['duration']
        
        analysis = analyses[state]
        analysis.total_visits += 1
        analysis.total_days += duration
        
        # Track by entry source
        if from_state is not None:
            if from_state not in analysis.entry_paths:
                analysis.entry_paths[from_state] = PathStats(from_state, state)
            
            path = analysis.entry_paths[from_state]
            path.count += 1
            path.durations.append(duration)
            path.exit_states.append(to_state)
        
        # Track exit destinations
        if to_state not in analysis.exit_counts:
            analysis.exit_counts[to_state] = 0
        analysis.exit_counts[to_state] += 1
    
    # Calculate overall average duration
    for state_id, analysis in analyses.items():
        if analysis.total_visits > 0:
            analysis.overall_avg_duration = analysis.total_days / analysis.total_visits
    
    return analyses


def test_duration_difference(durations1: List[float], durations2: List[float]) -> Tuple[float, float]:
    """
    Test if two sets of durations are significantly different.
    Returns (t_statistic, p_value).
    """
    if len(durations1) < MIN_SAMPLES_FOR_COMPARISON or len(durations2) < MIN_SAMPLES_FOR_COMPARISON:
        return 0.0, 1.0
    
    try:
        t_stat, p_val = stats.ttest_ind(durations1, durations2)
        return t_stat, p_val
    except:
        return 0.0, 1.0


# =============================================================================
# DISPLAY
# =============================================================================

def get_state_name(state: int) -> str:
    """Get descriptive name for state."""
    trend_24h = (state >> 3) & 1
    trend_168h = (state >> 2) & 1
    ma72_above = (state >> 1) & 1
    ma168_above = state & 1
    
    trend = "BULL" if trend_24h else "BEAR"
    return f"{trend}-{state}"


def display_transition_matrix(matrix: np.ndarray, pair: str):
    """Display the transition count matrix."""
    
    print(f"""
    ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  TRANSITION MATRIX: {pair} (counts)                                                            │
    └────────────────────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Header
    print("         To:", end="")
    for j in range(16):
        print(f"{j:>5}", end="")
    print("  │ Total")
    print("    From  ", "─" * 85)
    
    # Rows
    for i in range(16):
        row_sum = matrix[i, :].sum()
        if row_sum == 0:
            continue
        print(f"      {i:>2}  │", end="")
        for j in range(16):
            val = matrix[i, j]
            if val == 0:
                print("    .", end="")
            else:
                print(f"{val:>5}", end="")
        print(f"  │ {row_sum:>5}")
    
    print()


def display_transition_probabilities(matrix: np.ndarray, pair: str):
    """Display transition probabilities (normalized rows)."""
    
    print(f"""
    ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  TRANSITION PROBABILITIES: {pair} (P(to | from))                                               │
    └────────────────────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Header
    print("         To:", end="")
    for j in range(16):
        print(f"{j:>6}", end="")
    print()
    print("    From  ", "─" * 100)
    
    # Rows
    for i in range(16):
        row_sum = matrix[i, :].sum()
        if row_sum == 0:
            continue
        print(f"      {i:>2}  │", end="")
        for j in range(16):
            prob = matrix[i, j] / row_sum if row_sum > 0 else 0
            if prob == 0:
                print("     .", end="")
            elif prob >= 0.1:
                print(f" {prob:>4.0%}", end="")
            else:
                print(f" {prob:>4.1%}", end="")
        print()
    
    print()


def display_path_dependence(analyses: Dict[int, StateAnalysis], pair: str):
    """Display duration by entry source analysis."""
    
    print(f"""
    ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  PATH DEPENDENCE ANALYSIS: {pair}                                                              ║
    ║  Question: Does duration in state depend on where you came from?                               ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Find states with multiple entry sources and enough samples
    significant_findings = []
    
    for state_id in range(16):
        analysis = analyses[state_id]
        
        if analysis.total_visits < MIN_SAMPLES_FOR_STATS:
            continue
        
        # Get entry paths with enough samples
        valid_paths = [(src, path) for src, path in analysis.entry_paths.items() 
                       if path.count >= MIN_SAMPLES_FOR_COMPARISON]
        
        if len(valid_paths) < 2:
            continue
        
        print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  STATE {state_id} ({get_state_name(state_id)})                                              │
    │  Overall: {analysis.total_visits} visits, avg duration {analysis.overall_avg_duration:.1f} days              │
    ├──────────────┬─────────┬────────────┬────────────┬──────────────┤
    │ Entry From   │ Count   │ Avg Dur    │ Med Dur    │ Std Dev      │
    ├──────────────┼─────────┼────────────┼────────────┼──────────────┤""")
        
        path_durations = []
        for src, path in sorted(valid_paths, key=lambda x: -x[1].count):
            print(f"    │ State {src:>2}     │ {path.count:>7} │ {path.avg_duration:>8.1f}d  │ "
                  f"{path.median_duration:>8.1f}d  │ {path.std_duration:>10.1f}d │")
            path_durations.append((src, path))
        
        print(f"    └──────────────┴─────────┴────────────┴────────────┴──────────────┘")
        
        # Statistical comparison between top entry sources
        if len(path_durations) >= 2:
            # Compare the two most common entry sources
            src1, path1 = path_durations[0]
            src2, path2 = path_durations[1]
            
            t_stat, p_val = test_duration_difference(path1.durations, path2.durations)
            diff = path1.avg_duration - path2.avg_duration
            
            print(f"""
        Comparison: Entry from {src1} vs Entry from {src2}
        ─────────────────────────────────────────────────
        Duration difference: {diff:+.1f} days
        T-statistic: {t_stat:.2f}
        P-value: {p_val:.3f}
        Significant: {"YES ✓" if p_val < 0.05 else "No"}
        """)
            
            if p_val < 0.05:
                significant_findings.append({
                    'state': state_id,
                    'src1': src1,
                    'src2': src2,
                    'diff': diff,
                    'p_val': p_val
                })
    
    # Summary of significant findings
    if significant_findings:
        print(f"""
    ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  SIGNIFICANT FINDINGS (p < 0.05)                                                               ║
    ╠════════════════════════════════════════════════════════════════════════════════════════════════╣""")
        for f in significant_findings:
            longer = f['src1'] if f['diff'] > 0 else f['src2']
            shorter = f['src2'] if f['diff'] > 0 else f['src1']
            print(f"    ║  State {f['state']}: Entry from {longer} lasts {abs(f['diff']):.1f}d longer than entry from {shorter} (p={f['p_val']:.3f})  ║")
        print(f"    ╚════════════════════════════════════════════════════════════════════════════════════════════════╝")
    else:
        print(f"""
    ┌────────────────────────────────────────────────────────────────┐
    │  No statistically significant path dependence found (p < 0.05) │
    └────────────────────────────────────────────────────────────────┘
        """)


def display_exit_patterns(analyses: Dict[int, StateAnalysis], pair: str):
    """Display exit destination patterns by entry source."""
    
    print(f"""
    ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  EXIT PATTERN ANALYSIS: {pair}                                                                 ║
    ║  Question: Does exit destination depend on entry source?                                       ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    for state_id in range(16):
        analysis = analyses[state_id]
        
        if analysis.total_visits < MIN_SAMPLES_FOR_STATS:
            continue
        
        # Get entry paths with enough samples
        valid_paths = [(src, path) for src, path in analysis.entry_paths.items() 
                       if path.count >= MIN_SAMPLES_FOR_COMPARISON]
        
        if len(valid_paths) < 2:
            continue
        
        print(f"""
    State {state_id} ({get_state_name(state_id)}) - Exit destinations by entry source:
    ─────────────────────────────────────────────────────────────────""")
        
        for src, path in sorted(valid_paths, key=lambda x: -x[1].count):
            # Count exit destinations
            exit_counts = {}
            for exit_state in path.exit_states:
                if exit_state not in exit_counts:
                    exit_counts[exit_state] = 0
                exit_counts[exit_state] += 1
            
            # Sort by frequency
            sorted_exits = sorted(exit_counts.items(), key=lambda x: -x[1])
            
            # Format exit distribution
            exit_str = ", ".join([f"→{e}:{c}" for e, c in sorted_exits[:4]])
            
            # Bull vs Bear exit
            bull_exits = sum(c for e, c in exit_counts.items() if e in BULLISH_STATES)
            bear_exits = sum(c for e, c in exit_counts.items() if e in BEARISH_STATES)
            total = bull_exits + bear_exits
            
            if total > 0:
                bull_pct = bull_exits / total
                direction = "BULL" if bull_pct > 0.55 else "BEAR" if bull_pct < 0.45 else "NEUTRAL"
                print(f"      From {src:>2}: {exit_str:<30} │ {bull_pct:>5.0%} bull, {direction}")
        
        print()


def display_aggregate_summary(all_analyses: Dict[str, Dict[int, StateAnalysis]]):
    """Display cross-pair summary of path dependence."""
    
    print(f"""
    ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  CROSS-PAIR PATH DEPENDENCE SUMMARY                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Aggregate duration by state across all pairs
    state_durations = {i: [] for i in range(16)}
    
    for pair, analyses in all_analyses.items():
        for state_id, analysis in analyses.items():
            if analysis.total_visits >= MIN_SAMPLES_FOR_STATS:
                state_durations[state_id].append({
                    'pair': pair,
                    'avg_duration': analysis.overall_avg_duration,
                    'n_visits': analysis.total_visits
                })
    
    print(f"""
    ┌───────┬───────────────────────────────────────────────────────────────────────────────────┐
    │ State │ Duration by Pair                                                                  │
    ├───────┼───────────────────────────────────────────────────────────────────────────────────┤""")
    
    for state_id in range(16):
        durations = state_durations[state_id]
        if not durations:
            continue
        
        dur_str = "  ".join([f"{d['pair'][:3]}:{d['avg_duration']:.1f}d" for d in durations])
        avg_all = np.mean([d['avg_duration'] for d in durations])
        
        print(f"    │ {state_id:>5} │ {dur_str:<70} │ avg: {avg_all:.1f}d")
    
    print(f"    └───────┴───────────────────────────────────────────────────────────────────────────────────┘")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='State Transition Analysis')
    parser.add_argument('--pair', type=str, default=None, help='Analyze specific pair')
    parser.add_argument('--all-pairs', action='store_true', help='Analyze all pairs')
    parser.add_argument('--output', type=str, default='state_transition_results.txt', help='Output file')
    args = parser.parse_args()
    
    # Redirect output to file
    import io
    import sys
    
    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    print("=" * 100)
    print("STATE TRANSITION ANALYSIS")
    print("=" * 100)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  QUESTIONS WE'RE ANSWERING                                                                     ║
    ╠════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  1. Where does each state transition to? (16x16 matrix)                                        ║
    ║  2. Does duration depend on entry source? (path dependence)                                    ║
    ║  3. Does exit destination depend on entry source?                                              ║
    ║  4. Are these patterns consistent across pairs?                                                ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if args.pair:
        pairs = [args.pair]
    elif args.all_pairs:
        pairs = DEPLOY_PAIRS
    else:
        pairs = ['ETHUSD']
    
    print(f"  Pairs: {pairs}")
    
    if not HAS_DATABASE:
        sys.stdout = original_stdout
        print("ERROR: Database not available")
        return
    
    print("\n  Connecting to database...")
    db = Database()
    
    all_analyses = {}
    all_matrices = {}
    
    for pair in pairs:
        # Progress to console
        sys.stdout = original_stdout
        print(f"  Processing {pair}...", end=" ", flush=True)
        sys.stdout = output_buffer
        
        print(f"\n{'='*100}")
        print(f"  ANALYZING: {pair}")
        print(f"{'='*100}")
        
        # Load data
        df_1h = db.get_ohlcv(pair)
        print(f"  Loaded {len(df_1h)} hourly bars")
        
        # Compute states
        states = compute_regime_states_daily(df_1h)
        print(f"  Computed {len(states)} daily states")
        
        # Extract visits
        visits = extract_state_visits(states)
        print(f"  Extracted {len(visits)} state visits")
        
        # Build transition matrix
        matrix = build_transition_matrix(visits)
        all_matrices[pair] = matrix
        
        # Display transition matrix
        display_transition_matrix(matrix, pair)
        display_transition_probabilities(matrix, pair)
        
        # Analyze path dependence
        analyses = analyze_path_dependence(visits)
        all_analyses[pair] = analyses
        
        # Display findings
        display_path_dependence(analyses, pair)
        display_exit_patterns(analyses, pair)
        
        # Progress to console
        sys.stdout = original_stdout
        print("done")
        sys.stdout = output_buffer
    
    # Cross-pair summary
    if len(pairs) > 1:
        display_aggregate_summary(all_analyses)
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}")
    
    # Write to file
    sys.stdout = original_stdout
    output_text = output_buffer.getvalue()
    
    with open(args.output, 'w') as f:
        f.write(output_text)
    
    print(f"Results saved to: {args.output}")
    print(f"File size: {len(output_text):,} characters")
    
    # Print brief summary to console
    print("\n" + "=" * 100)
    print("COMPLETE TRANSITION ANALYSIS")
    print("=" * 100)
    
    for pair in pairs:
        analyses = all_analyses[pair]
        matrix = all_matrices[pair]
        
        print(f"\n{'='*100}")
        print(f"{pair}: TRANSITION MATRIX WITH DURATIONS")
        print("=" * 100)
        print(f"\n  From → To   | Count | P(to|from) | Avg Duration in 'To' | Significance")
        print(f"  " + "-" * 85)
        
        # Build complete path data
        path_data = []
        
        for from_state in range(16):
            row_total = matrix[from_state, :].sum()
            if row_total < 10:
                continue
                
            for to_state in range(16):
                count = matrix[from_state, to_state]
                if count < 5 or from_state == to_state:
                    continue
                
                prob = count / row_total
                
                # Get duration in to_state when entered from from_state
                to_analysis = analyses[to_state]
                if from_state in to_analysis.entry_paths:
                    path = to_analysis.entry_paths[from_state]
                    avg_dur = path.avg_duration
                    n_samples = path.count
                    
                    # Compare to overall duration in to_state
                    overall_dur = to_analysis.overall_avg_duration
                    diff = avg_dur - overall_dur
                    diff_pct = (diff / overall_dur * 100) if overall_dur > 0 else 0
                    
                    path_data.append({
                        'from': from_state,
                        'to': to_state,
                        'count': count,
                        'prob': prob,
                        'dur': avg_dur,
                        'overall_dur': overall_dur,
                        'diff': diff,
                        'diff_pct': diff_pct,
                        'n_samples': n_samples
                    })
        
        # Sort by count (most common paths first)
        path_data = sorted(path_data, key=lambda x: -x['count'])
        
        # Display top 20 paths
        for p in path_data[:25]:
            sig_marker = ""
            if abs(p['diff_pct']) > 20 and p['n_samples'] >= 10:
                sig_marker = "**" if p['diff'] > 0 else "--"
            elif abs(p['diff_pct']) > 10 and p['n_samples'] >= 10:
                sig_marker = "+" if p['diff'] > 0 else "-"
            
            from_type = "BULL" if p['from'] >= 8 else "BEAR"
            to_type = "BULL" if p['to'] >= 8 else "BEAR"
            
            print(f"  {p['from']:>2} → {p['to']:<2}    | {p['count']:>5} | {p['prob']:>9.0%}  | "
                  f"{p['dur']:>5.1f}d (vs {p['overall_dur']:.1f}d avg) {p['diff']:>+5.1f}d | "
                  f"{sig_marker:<3} {from_type}→{to_type}")
        
        # Summary stats for this pair
        print(f"\n  Key findings for {pair}:")
        
        # Paths where duration differs significantly from average
        long_paths = [p for p in path_data if p['diff_pct'] > 20 and p['n_samples'] >= 10]
        short_paths = [p for p in path_data if p['diff_pct'] < -20 and p['n_samples'] >= 10]
        
        if long_paths:
            print(f"    Paths that EXTEND duration (>20% longer):")
            for p in long_paths[:5]:
                print(f"      {p['from']:>2} → {p['to']:<2}: {p['dur']:.1f}d vs {p['overall_dur']:.1f}d avg ({p['diff']:+.1f}d)")
        
        if short_paths:
            print(f"    Paths that SHORTEN duration (>20% shorter):")
            for p in short_paths[:5]:
                print(f"      {p['from']:>2} → {p['to']:<2}: {p['dur']:.1f}d vs {p['overall_dur']:.1f}d avg ({p['diff']:+.1f}d)")
    
    # Cross-pair summary
    print("\n" + "=" * 100)
    print("CROSS-PAIR SUMMARY: MOST COMMON TRANSITIONS")
    print("=" * 100)
    
    # Aggregate transitions across pairs
    agg_transitions = {}
    for pair in pairs:
        matrix = all_matrices[pair]
        for from_state in range(16):
            row_total = matrix[from_state, :].sum()
            if row_total < 10:
                continue
            for to_state in range(16):
                if from_state == to_state:
                    continue
                count = matrix[from_state, to_state]
                if count >= 5:
                    key = (from_state, to_state)
                    if key not in agg_transitions:
                        agg_transitions[key] = {'pairs': [], 'probs': [], 'counts': []}
                    agg_transitions[key]['pairs'].append(pair)
                    agg_transitions[key]['probs'].append(count / row_total)
                    agg_transitions[key]['counts'].append(count)
    
    # Find transitions that occur in 4+ pairs
    common_trans = [(k, v) for k, v in agg_transitions.items() if len(v['pairs']) >= 4]
    common_trans = sorted(common_trans, key=lambda x: -np.mean(x[1]['probs']))
    
    print(f"\n  Transitions occurring in 4+ pairs (sorted by avg probability):")
    print(f"  {'From':>4} → {'To':<4} | {'Avg P':>7} | {'Pairs':>6} | {'Interpretation':<30}")
    print(f"  " + "-" * 70)
    
    for (from_s, to_s), data in common_trans[:20]:
        avg_prob = np.mean(data['probs'])
        n_pairs = len(data['pairs'])
        
        from_type = "BULL" if from_s >= 8 else "BEAR"
        to_type = "BULL" if to_s >= 8 else "BEAR"
        
        if from_type == to_type:
            interp = f"Stay {from_type}"
        else:
            interp = f"FLIP {from_type}→{to_type}"
        
        print(f"  {from_s:>4} → {to_s:<4} | {avg_prob:>6.0%}  | {n_pairs:>6}  | {interp:<30}")
    
    # Average duration by state
    print(f"\n" + "=" * 100)
    print("STATE DURATION SUMMARY (ALL PAIRS)")
    print("=" * 100)
    
    state_avg_durations = {i: [] for i in range(16)}
    state_daily_rates = {i: [] for i in range(16)}
    
    for pair, analyses in all_analyses.items():
        for state_id, analysis in analyses.items():
            if analysis.total_visits >= 10:
                state_avg_durations[state_id].append(analysis.overall_avg_duration)
                daily_rate = analysis.total_visits / analysis.total_days if analysis.total_days > 0 else 0
                state_daily_rates[state_id].append(daily_rate)
    
    print(f"\n  {'State':<8} {'Type':<6} {'Avg Dur':<10} {'Daily Exit Rate':<15} {'Stability':<12}")
    print(f"  " + "-" * 55)
    
    state_summary = []
    for state_id in range(16):
        durations = state_avg_durations[state_id]
        if durations:
            avg_dur = np.mean(durations)
            # Calculate implied daily exit rate from duration
            daily_exit = 1 / avg_dur if avg_dur > 0 else 0
            state_summary.append((state_id, avg_dur, daily_exit))
    
    state_summary = sorted(state_summary, key=lambda x: -x[1])
    
    for state_id, avg_dur, daily_exit in state_summary:
        bull_bear = "BULL" if state_id >= 8 else "BEAR"
        stability = "STABLE" if avg_dur > 2.5 else "MEDIUM" if avg_dur > 1.8 else "UNSTABLE"
        print(f"  {state_id:<8} {bull_bear:<6} {avg_dur:>6.1f}d    {daily_exit:>12.0%}      {stability:<12}")
    
    print(f"\nFull details saved to: {args.output}")
    
    return all_analyses, all_matrices


if __name__ == "__main__":
    analyses, matrices = main()