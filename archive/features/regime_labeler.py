# -*- coding: utf-8 -*-
"""
CryptoBot - Regime Labeling System
===================================
Label OHLCV bars as Up/Down/Flat with validation metrics.

Designed to run independently on each timeframe dataset.

Usage:
    from regime_labeler import RegimeLabeler, RegimeConfig
    
    # Configure with single MA
    config = RegimeConfig(
        ma_periods=[6],
        lookback=20,
        quantile_upper=0.90,
        quantile_lower=0.10,
        buffer_type='percent',
        buffer_size=0.10,
        min_duration=3,
    )
    
    # Configure with multiple MAs
    config = RegimeConfig(
        ma_periods=[6, 12, 24],
        ma_agreement='majority',  # 'any', 'majority', 'all'
        lookback=20,
        buffer_size=0.10,
    )
    
    # Label
    labeler = RegimeLabeler(config)
    labels = labeler.label(df)
    
    # Validate
    results = labeler.validate(df, labels)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RegimeConfig:
    """
    Hyperparameters for regime labeling.
    
    All parameters can be tuned independently per timeframe.
    """
    # --- MA Settings ---
    ma_periods: List[int] = field(default_factory=lambda: [6])  # List of MA periods
    ma_agreement: str = 'majority'  # 'any', 'majority', 'all'
    
    # --- Breakout Detection ---
    lookback: int = 20  # Bars for rolling range
    quantile_upper: float = 0.90  # Upper quantile for range
    quantile_lower: float = 0.10  # Lower quantile for range
    buffer_type: str = 'percent'  # 'percent', 'atr', 'mad'
    buffer_size: float = 0.10  # Buffer as fraction of range
    
    # --- Smoothing ---
    min_duration: int = 3  # Minimum bars for regime to be valid
    
    # --- Dollar Test ---
    position_up: float = 1.0  # Position size when Up
    position_flat: float = 0.5  # Position size when Flat
    position_down: float = 0.0  # Position size when Down
    initial_capital: float = 100.0  # Starting capital
    transaction_cost: float = 0.001  # Cost per trade (0.1%)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ma_periods': self.ma_periods,
            'ma_agreement': self.ma_agreement,
            'lookback': self.lookback,
            'quantile_upper': self.quantile_upper,
            'quantile_lower': self.quantile_lower,
            'buffer_type': self.buffer_type,
            'buffer_size': self.buffer_size,
            'min_duration': self.min_duration,
            'position_up': self.position_up,
            'position_flat': self.position_flat,
            'position_down': self.position_down,
            'initial_capital': self.initial_capital,
            'transaction_cost': self.transaction_cost,
        }


# =============================================================================
# Regime Labeler
# =============================================================================

class RegimeLabeler:
    """
    Label OHLCV bars as Up (+1), Down (-1), or Flat (0).
    
    Process:
        1. Compute SMA
        2. Detect breakouts using rolling quantiles + buffer
        3. Smooth labels (remove short regimes)
    """
    
    def __init__(self, config: RegimeConfig = None):
        """
        Initialize labeler.
        
        Args:
            config: RegimeConfig with hyperparameters
        """
        self.config = config or RegimeConfig()
    
    # -------------------------------------------------------------------------
    # Main Methods
    # -------------------------------------------------------------------------
    
    def label(self, df: pd.DataFrame, smooth: bool = True) -> pd.Series:
        """
        Label each bar as Up (+1), Down (-1), or Flat (0).
        
        Args:
            df: OHLCV DataFrame with 'close' column
            smooth: Apply smoothing to remove short regimes
        
        Returns:
            Series of labels (-1, 0, +1)
        """
        # Compute labels using multiple MAs
        labels = self._label_with_multiple_mas(df)
        
        # Smooth (optional)
        if smooth:
            labels = self._smooth_labels(labels)
        
        return labels
    
    def validate(
        self, 
        df: pd.DataFrame, 
        labels: pd.Series,
        forward_periods: List[int] = None,
    ) -> Dict:
        """
        Validate labels with statistical and economic metrics.
        
        Args:
            df: OHLCV DataFrame
            labels: Regime labels (-1, 0, +1)
            forward_periods: Periods for forward return analysis
        
        Returns:
            Dict with 'statistical' and 'economic' results
        """
        if forward_periods is None:
            forward_periods = [1, 5, 10]
        
        results = {
            'statistical': self._compute_statistical_metrics(df, labels, forward_periods),
            'economic': self._compute_dollar_test(df, labels),
            'config': self.config.to_dict(),
        }
        
        return results
    
    # -------------------------------------------------------------------------
    # Labeling Steps
    # -------------------------------------------------------------------------
    
    def _compute_ma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Simple Moving Average for given period."""
        return df['close'].rolling(
            window=period, 
            min_periods=period
        ).mean()
    
    def _compute_single_ma_labels(self, df: pd.DataFrame, ma: pd.Series) -> pd.Series:
        """
        Compute labels for a single MA.
        
        Returns raw labels (-1, 0, +1) before smoothing.
        """
        # Compute breakout bands
        upper_band, lower_band = self._compute_bands(ma)
        
        # Assign labels
        labels = pd.Series(0, index=ma.index)  # Default: Flat
        labels[ma > upper_band] = 1   # Up
        labels[ma < lower_band] = -1  # Down
        
        return labels
    
    def _combine_ma_labels(self, all_labels: pd.DataFrame) -> pd.Series:
        """
        Combine labels from multiple MAs based on agreement method.
        
        Args:
            all_labels: DataFrame where each column is labels from one MA
        
        Returns:
            Combined labels series
        """
        n_mas = len(all_labels.columns)
        agreement = self.config.ma_agreement
        
        # Sum of labels: +1 for Up, -1 for Down, 0 for Flat
        label_sum = all_labels.sum(axis=1)
        
        # Count non-flat votes
        up_count = (all_labels == 1).sum(axis=1)
        down_count = (all_labels == -1).sum(axis=1)
        
        # Initialize as Flat
        combined = pd.Series(0, index=all_labels.index)
        
        if agreement == 'any':
            # Any MA says Up/Down
            combined[up_count > 0] = 1
            combined[down_count > 0] = -1
            # If both Up and Down present, use majority
            both_mask = (up_count > 0) & (down_count > 0)
            combined[both_mask & (up_count > down_count)] = 1
            combined[both_mask & (down_count > up_count)] = -1
            combined[both_mask & (up_count == down_count)] = 0
            
        elif agreement == 'majority':
            # More than half agree
            threshold = n_mas / 2
            combined[up_count > threshold] = 1
            combined[down_count > threshold] = -1
            
        elif agreement == 'all':
            # All MAs must agree
            combined[up_count == n_mas] = 1
            combined[down_count == n_mas] = -1
            
        else:
            raise ValueError(f"Unknown ma_agreement: {agreement}")
        
        return combined
    
    def _label_with_multiple_mas(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute labels using multiple MAs.
        
        For each MA:
          1. Compute MA
          2. Apply breakout detection
          3. Get raw labels
        
        Then combine using agreement method.
        """
        ma_periods = self.config.ma_periods
        
        # Collect labels from each MA
        all_labels = pd.DataFrame(index=df.index)
        
        for period in ma_periods:
            ma = self._compute_ma(df, period)
            labels = self._compute_single_ma_labels(df, ma)
            all_labels[f'ma_{period}'] = labels
        
        # Combine
        combined = self._combine_ma_labels(all_labels)
        
        return combined
    
    def _compute_bands(self, ma: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Compute upper and lower bands for breakout detection.
        
        Uses rolling quantiles + buffer.
        """
        lookback = self.config.lookback
        q_upper = self.config.quantile_upper
        q_lower = self.config.quantile_lower
        
        # Rolling quantiles
        rolling_upper = ma.rolling(window=lookback, min_periods=lookback).quantile(q_upper)
        rolling_lower = ma.rolling(window=lookback, min_periods=lookback).quantile(q_lower)
        
        # Range for buffer calculation
        rolling_range = rolling_upper - rolling_lower
        
        # Compute buffer based on type
        buffer = self._compute_buffer(ma, rolling_range)
        
        # Final bands
        upper_band = rolling_upper + buffer
        lower_band = rolling_lower - buffer
        
        return upper_band, lower_band
    
    def _compute_buffer(self, ma: pd.Series, rolling_range: pd.Series) -> pd.Series:
        """
        Compute buffer based on config type.
        
        Options:
            - percent: buffer_size × rolling_range
            - atr: buffer_size × ATR (requires OHLC)
            - mad: buffer_size × MAD of MA
        """
        buffer_type = self.config.buffer_type
        buffer_size = self.config.buffer_size
        
        if buffer_type == 'percent':
            return buffer_size * rolling_range
        
        elif buffer_type == 'mad':
            # Median Absolute Deviation
            lookback = self.config.lookback
            ma_median = ma.rolling(window=lookback).median()
            mad = (ma - ma_median).abs().rolling(window=lookback).median()
            return buffer_size * mad * 1.4826  # Scale to std equivalent
        
        elif buffer_type == 'atr':
            # ATR not available without OHLC, fall back to percent
            return buffer_size * rolling_range
        
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}")
    
    def _smooth_labels(self, labels: pd.Series) -> pd.Series:
        """
        Smooth labels by removing short-lived regimes.
        
        If a regime lasts fewer than min_duration bars,
        revert to previous regime.
        """
        min_duration = self.config.min_duration
        
        if min_duration <= 1:
            return labels
        
        smoothed = labels.copy()
        
        # Find regime changes
        regime_changes = labels.diff().fillna(0) != 0
        change_indices = labels.index[regime_changes].tolist()
        
        if len(change_indices) < 2:
            return smoothed
        
        # Add start and end
        all_indices = [labels.index[0]] + change_indices + [labels.index[-1]]
        
        # Check each regime
        for i in range(1, len(all_indices) - 1):
            start = all_indices[i]
            end = all_indices[i + 1]
            
            # Get positions
            start_pos = labels.index.get_loc(start)
            end_pos = labels.index.get_loc(end)
            duration = end_pos - start_pos
            
            if duration < min_duration:
                # Too short - revert to previous regime
                prev_regime = smoothed.iloc[start_pos - 1] if start_pos > 0 else 0
                smoothed.iloc[start_pos:end_pos] = prev_regime
        
        return smoothed
    
    # -------------------------------------------------------------------------
    # Validation: Statistical Metrics
    # -------------------------------------------------------------------------
    
    def _compute_statistical_metrics(
        self, 
        df: pd.DataFrame, 
        labels: pd.Series,
        forward_periods: List[int],
    ) -> Dict:
        """
        Compute statistical validation metrics.
        
        Returns:
            - return_by_regime: Forward returns per regime
            - hit_rates: % correct direction
            - regime_stats: Duration, counts
            - transitions: Transition matrix
        """
        results = {}
        
        # --- Forward Returns by Regime ---
        returns_by_regime = {}
        for period in forward_periods:
            forward_ret = df['close'].pct_change(period).shift(-period)
            
            period_results = {}
            for regime in [-1, 0, 1]:
                mask = labels == regime
                if mask.sum() == 0:
                    continue
                    
                regime_returns = forward_ret[mask].dropna()
                
                period_results[regime] = {
                    'count': mask.sum(),
                    'mean_return': regime_returns.mean() if len(regime_returns) > 0 else np.nan,
                    'std_return': regime_returns.std() if len(regime_returns) > 0 else np.nan,
                    'median_return': regime_returns.median() if len(regime_returns) > 0 else np.nan,
                }
            
            returns_by_regime[f'{period}_bar'] = period_results
        
        results['returns_by_regime'] = returns_by_regime
        
        # --- Return Spread ---
        spreads = {}
        for period in forward_periods:
            forward_ret = df['close'].pct_change(period).shift(-period)
            
            up_ret = forward_ret[labels == 1].mean()
            down_ret = forward_ret[labels == -1].mean()
            flat_ret = forward_ret[labels == 0].mean()
            
            spreads[f'{period}_bar'] = {
                'up_minus_down': up_ret - down_ret if not (np.isnan(up_ret) or np.isnan(down_ret)) else np.nan,
                'up_minus_flat': up_ret - flat_ret if not (np.isnan(up_ret) or np.isnan(flat_ret)) else np.nan,
            }
        
        results['spreads'] = spreads
        
        # --- Hit Rates ---
        forward_ret_1 = df['close'].pct_change(1).shift(-1)
        
        hit_rates = {}
        for regime, expected_sign in [(1, 1), (-1, -1), (0, 0)]:
            mask = labels == regime
            if mask.sum() == 0:
                continue
            
            if regime == 1:  # Up should have positive returns
                hits = (forward_ret_1[mask] > 0).sum()
            elif regime == -1:  # Down should have negative returns
                hits = (forward_ret_1[mask] < 0).sum()
            else:  # Flat - no expectation
                hits = mask.sum() / 2  # 50% baseline
            
            hit_rates[regime] = hits / mask.sum() if mask.sum() > 0 else np.nan
        
        results['hit_rates'] = hit_rates
        
        # --- Kelly Inputs (avg_win, avg_loss per regime) ---
        kelly_inputs = {}
        
        for regime in [1, -1]:  # Up and Down only (Flat has no edge)
            mask = labels == regime
            if mask.sum() == 0:
                continue
            
            regime_returns = forward_ret_1[mask].dropna()
            
            if len(regime_returns) == 0:
                continue
            
            if regime == 1:  # Up regime: positive returns = win
                wins = regime_returns[regime_returns > 0]
                losses = regime_returns[regime_returns <= 0]
            else:  # Down regime: negative returns = win (for short)
                wins = regime_returns[regime_returns < 0].abs()  # Profit from short
                losses = regime_returns[regime_returns >= 0]  # Loss from short
            
            n_wins = len(wins)
            n_losses = len(losses)
            n_total = n_wins + n_losses
            
            kelly_inputs[regime] = {
                'n_samples': n_total,
                'n_wins': n_wins,
                'n_losses': n_losses,
                'hit_rate': n_wins / n_total if n_total > 0 else 0.5,
                'avg_win': wins.mean() if len(wins) > 0 else 0.0,
                'avg_loss': losses.abs().mean() if len(losses) > 0 else 0.0,
                'median_win': wins.median() if len(wins) > 0 else 0.0,
                'median_loss': losses.abs().median() if len(losses) > 0 else 0.0,
                'total_wins': wins.sum() if len(wins) > 0 else 0.0,
                'total_losses': losses.abs().sum() if len(losses) > 0 else 0.0,
            }
            
            # Compute win/loss ratio
            ki = kelly_inputs[regime]
            if ki['avg_loss'] > 0:
                ki['win_loss_ratio'] = ki['avg_win'] / ki['avg_loss']
            else:
                ki['win_loss_ratio'] = np.inf if ki['avg_win'] > 0 else 1.0
            
            # Compute raw Kelly fraction
            p = ki['hit_rate']
            b = ki['win_loss_ratio']
            if b > 0 and b != np.inf:
                ki['raw_kelly'] = (p * b - (1 - p)) / b
            else:
                ki['raw_kelly'] = 2 * p - 1  # Simplified when b=1 or inf
            
            ki['raw_kelly'] = max(0, ki['raw_kelly'])  # Can't be negative
            
            # Statistical confidence
            if n_total >= 10:
                se = np.sqrt(p * (1 - p) / n_total)
                z_score = (p - 0.5) / se if se > 0 else 0
                # Approximate confidence using normal CDF
                ki['confidence'] = 0.5 * (1 + np.tanh(z_score / 1.4))  # Approx normal CDF
            else:
                ki['confidence'] = 0.0
        
        results['kelly_inputs'] = kelly_inputs
        
        # --- Regime Statistics ---
        regime_stats = {}
        
        # Counts
        for regime in [-1, 0, 1]:
            regime_stats[regime] = {
                'count': (labels == regime).sum(),
                'pct': (labels == regime).mean(),
            }
        
        # Durations
        durations = self._compute_regime_durations(labels)
        for regime in [-1, 0, 1]:
            if regime in durations and len(durations[regime]) > 0:
                regime_stats[regime]['avg_duration'] = np.mean(durations[regime])
                regime_stats[regime]['median_duration'] = np.median(durations[regime])
                regime_stats[regime]['max_duration'] = np.max(durations[regime])
                regime_stats[regime]['min_duration'] = np.min(durations[regime])
        
        results['regime_stats'] = regime_stats
        
        # --- Transition Matrix ---
        results['transitions'] = self._compute_transition_matrix(labels)
        
        return results
    
    def _compute_regime_durations(self, labels: pd.Series) -> Dict[int, List[int]]:
        """Compute duration of each regime occurrence."""
        durations = {-1: [], 0: [], 1: []}
        
        current_regime = labels.iloc[0]
        current_duration = 1
        
        for i in range(1, len(labels)):
            if labels.iloc[i] == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = labels.iloc[i]
                current_duration = 1
        
        # Don't forget last regime
        durations[current_regime].append(current_duration)
        
        return durations
    
    def _compute_transition_matrix(self, labels: pd.Series) -> pd.DataFrame:
        """
        Compute regime transition probabilities.
        
        Returns DataFrame:
                    To_Down  To_Flat  To_Up
        From_Down    0.8      0.15     0.05
        From_Flat    0.1      0.8      0.1
        From_Up      0.05     0.15     0.8
        """
        transitions = pd.DataFrame(
            0.0, 
            index=['Down', 'Flat', 'Up'],
            columns=['To_Down', 'To_Flat', 'To_Up']
        )
        
        regime_map = {-1: 'Down', 0: 'Flat', 1: 'Up'}
        col_map = {-1: 'To_Down', 0: 'To_Flat', 1: 'To_Up'}
        
        # Count transitions
        for i in range(len(labels) - 1):
            from_regime = labels.iloc[i]
            to_regime = labels.iloc[i + 1]
            
            transitions.loc[regime_map[from_regime], col_map[to_regime]] += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1)
        for idx in transitions.index:
            if row_sums[idx] > 0:
                transitions.loc[idx] = transitions.loc[idx] / row_sums[idx]
        
        return transitions
    
    # -------------------------------------------------------------------------
    # Validation: Dollar Test
    # -------------------------------------------------------------------------
    
    def _compute_dollar_test(self, df: pd.DataFrame, labels: pd.Series) -> Dict:
        """
        Compare regime strategy vs buy-and-hold.
        
        Returns metrics for both strategies.
        """
        # Get config
        pos_up = self.config.position_up
        pos_flat = self.config.position_flat
        pos_down = self.config.position_down
        initial = self.config.initial_capital
        tx_cost = self.config.transaction_cost
        
        # Forward returns
        returns = df['close'].pct_change(1).shift(-1)
        
        # Position based on label
        position_map = {1: pos_up, 0: pos_flat, -1: pos_down}
        positions = labels.map(position_map)
        
        # Transaction costs (position changes)
        position_changes = positions.diff().abs().fillna(0)
        costs = position_changes * tx_cost
        
        # --- Regime Strategy ---
        regime_returns = positions * returns - costs
        regime_equity = initial * (1 + regime_returns).cumprod()
        regime_equity = regime_equity.fillna(initial)
        
        # --- Buy and Hold ---
        bh_returns = returns  # 100% position always
        bh_equity = initial * (1 + bh_returns).cumprod()
        bh_equity = bh_equity.fillna(initial)
        
        # --- Inverse (sanity check) ---
        inverse_map = {1: pos_down, 0: pos_flat, -1: pos_up}
        inverse_positions = labels.map(inverse_map)
        inverse_returns = inverse_positions * returns
        inverse_equity = initial * (1 + inverse_returns).cumprod()
        inverse_equity = inverse_equity.fillna(initial)
        
        # --- Compute Metrics ---
        results = {
            'regime_strategy': self._compute_equity_metrics(regime_equity, regime_returns, positions),
            'buy_and_hold': self._compute_equity_metrics(bh_equity, bh_returns, pd.Series(1.0, index=labels.index)),
            'inverse_check': self._compute_equity_metrics(inverse_equity, inverse_returns, inverse_positions),
        }
        
        # --- Comparison ---
        results['comparison'] = {
            'regime_vs_bh_return': results['regime_strategy']['total_return'] - results['buy_and_hold']['total_return'],
            'regime_vs_bh_sharpe': results['regime_strategy']['sharpe'] - results['buy_and_hold']['sharpe'],
            'regime_vs_bh_drawdown': results['buy_and_hold']['max_drawdown'] - results['regime_strategy']['max_drawdown'],
            'regime_beats_bh': results['regime_strategy']['total_return'] > results['buy_and_hold']['total_return'],
            'inverse_loses': results['inverse_check']['total_return'] < results['buy_and_hold']['total_return'],
        }
        
        # Store equity curves
        results['equity_curves'] = {
            'regime': regime_equity,
            'buy_and_hold': bh_equity,
            'inverse': inverse_equity,
        }
        
        return results
    
    def _compute_equity_metrics(
        self, 
        equity: pd.Series, 
        returns: pd.Series,
        positions: pd.Series,
    ) -> Dict:
        """Compute performance metrics for an equity curve."""
        
        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 0 else 0
        
        # Annualized (assuming hourly for flexibility)
        n_bars = len(equity)
        # We don't know TF here, so report raw
        
        # Sharpe (annualized assuming daily-ish)
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
        
        # Max Drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Time in market
        time_in_market = (positions > 0).mean()
        
        # Trade count (position changes)
        trades = (positions.diff().fillna(0) != 0).sum()
        
        return {
            'final_value': equity.iloc[-1] if len(equity) > 0 else self.config.initial_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'time_in_market': time_in_market,
            'trade_count': trades,
            'return_per_drawdown': abs(total_return / max_drawdown) if max_drawdown != 0 else np.inf,
        }


# =============================================================================
# Summary Printer
# =============================================================================

def print_validation_results(results: Dict, timeframe: str = None):
    """Pretty print validation results."""
    
    header = f"VALIDATION RESULTS"
    if timeframe:
        header += f" - {timeframe}"
    
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    
    # --- Config ---
    print("\nConfiguration:")
    config = results.get('config', {})
    print(f"  MA Periods: {config.get('ma_periods')}")
    print(f"  MA Agreement: {config.get('ma_agreement')}")
    print(f"  Lookback: {config.get('lookback')}")
    print(f"  Quantiles: [{config.get('quantile_lower')}, {config.get('quantile_upper')}]")
    print(f"  Buffer: {config.get('buffer_size')} ({config.get('buffer_type')})")
    print(f"  Min Duration: {config.get('min_duration')}")
    
    # --- Statistical ---
    stats = results.get('statistical', {})
    
    print("\n--- Regime Distribution ---")
    regime_stats = stats.get('regime_stats', {})
    regime_names = {-1: 'Down', 0: 'Flat', 1: 'Up'}
    for regime in [-1, 0, 1]:
        if regime in regime_stats:
            rs = regime_stats[regime]
            print(f"  {regime_names[regime]:5}: {rs.get('count', 0):5} bars ({rs.get('pct', 0)*100:.1f}%), "
                  f"avg duration: {rs.get('avg_duration', 0):.1f} bars")
    
    print("\n--- Forward Return Spreads ---")
    spreads = stats.get('spreads', {})
    for period, spread_data in spreads.items():
        up_down = spread_data.get('up_minus_down', np.nan)
        print(f"  {period}: Up-Down = {up_down*100:+.2f}%" if not np.isnan(up_down) else f"  {period}: N/A")
    
    print("\n--- Hit Rates ---")
    hit_rates = stats.get('hit_rates', {})
    for regime in [1, 0, -1]:
        if regime in hit_rates:
            print(f"  {regime_names[regime]:5}: {hit_rates[regime]*100:.1f}%")
    
    # --- Kelly Inputs ---
    kelly_inputs = stats.get('kelly_inputs', {})
    if kelly_inputs:
        print("\n--- Kelly Inputs ---")
        for regime in [1, -1]:
            if regime not in kelly_inputs:
                continue
            ki = kelly_inputs[regime]
            print(f"  {regime_names[regime]} Regime:")
            print(f"    Samples:       {ki.get('n_samples', 0):>6} ({ki.get('n_wins', 0)} wins, {ki.get('n_losses', 0)} losses)")
            print(f"    Hit Rate:      {ki.get('hit_rate', 0)*100:>6.1f}%")
            print(f"    Avg Win:       {ki.get('avg_win', 0)*100:>+6.2f}%")
            print(f"    Avg Loss:      {ki.get('avg_loss', 0)*100:>6.2f}%")
            print(f"    Win/Loss:      {ki.get('win_loss_ratio', 0):>6.2f}")
            print(f"    Raw Kelly:     {ki.get('raw_kelly', 0)*100:>6.1f}%")
            print(f"    Confidence:    {ki.get('confidence', 0)*100:>6.1f}%")
    
    print("\n--- Transition Matrix ---")
    transitions = stats.get('transitions')
    if transitions is not None:
        print(transitions.round(3).to_string())
    
    # --- Economic ---
    econ = results.get('economic', {})
    
    print("\n--- Dollar Test ---")
    print(f"  {'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Time In':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    
    for name in ['regime_strategy', 'buy_and_hold', 'inverse_check']:
        if name in econ:
            m = econ[name]
            print(f"  {name:<15} {m['total_return_pct']:>+9.1f}% {m['sharpe']:>8.2f} "
                  f"{m['max_drawdown_pct']:>+9.1f}% {m['time_in_market']*100:>7.0f}%")
    
    print("\n--- Comparison ---")
    comp = econ.get('comparison', {})
    regime_beats = "✓ YES" if comp.get('regime_beats_bh') else "✗ NO"
    inverse_loses = "✓ YES" if comp.get('inverse_loses') else "✗ NO"
    print(f"  Regime beats Buy&Hold: {regime_beats}")
    print(f"  Inverse loses money:   {inverse_loses}")
    
    print("\n" + "=" * 60)


# =============================================================================
# Convenience Function
# =============================================================================

def label_and_validate(
    df: pd.DataFrame, 
    config: RegimeConfig = None,
    timeframe: str = None,
    print_results: bool = True,
) -> Tuple[pd.Series, Dict]:
    """
    One-step labeling and validation.
    
    Args:
        df: OHLCV DataFrame
        config: RegimeConfig (uses defaults if None)
        timeframe: Name for display purposes
        print_results: Print summary
    
    Returns:
        (labels, results)
    """
    labeler = RegimeLabeler(config)
    labels = labeler.label(df)
    results = labeler.validate(df, labels)
    
    if print_results:
        print_validation_results(results, timeframe)
    
    return labels, results