#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kelly Inputs Calculator
========================
Derive empirical Kelly inputs and multipliers from regime validation.

Instead of arbitrary multipliers, this calculates:
    1. Raw Kelly fraction per regime (from hit rate + win/loss ratio)
    2. Statistical confidence per regime
    3. Recommended position sizing

Usage:
    1. Run this script
    2. Review Kelly inputs per TF/pair
    3. Use recommended multipliers in KellySizer
"""

import sys
sys.path.insert(0, 'D:/cryptobot_docker')

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from regime_labeler import RegimeLabeler, RegimeConfig
from cryptobot.datasources.database import Database


# =============================================================================
# CONFIGURATION
# =============================================================================

PAIRS = ['XBTUSD', 'ETHUSD']

START_DATE = '2022-01-01'
END_DATE = '2024-12-31'

# Optimized configs from grid search
CONFIGS = {
    '72h': RegimeConfig(
        ma_periods=[6],
        ma_agreement='majority',
        lookback=20,
        quantile_upper=0.95,
        quantile_lower=0.05,
        buffer_type='percent',
        buffer_size=0.05,
        min_duration=3,
    ),
    '24h': RegimeConfig(
        ma_periods=[6, 12, 24],
        ma_agreement='all',
        lookback=10,
        quantile_upper=0.95,
        quantile_lower=0.05,
        buffer_type='percent',
        buffer_size=0.05,
        min_duration=3,
    ),
    '12h': RegimeConfig(
        ma_periods=[6, 12],
        ma_agreement='majority',
        lookback=20,
        quantile_upper=0.95,
        quantile_lower=0.05,
        buffer_type='percent',
        buffer_size=0.10,
        min_duration=3,
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1h OHLCV to higher timeframe."""
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()
    return resampled


@dataclass
class KellyResult:
    """Kelly calculation result for one regime/TF/pair."""
    pair: str
    timeframe: str
    regime: int  # 1=Up, -1=Down
    
    # Raw data
    n_samples: int
    n_wins: int
    n_losses: int
    
    # Core metrics
    hit_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    
    # Kelly outputs
    raw_kelly: float
    confidence: float
    
    # Derived
    edge: float = 0.0
    adjusted_kelly: float = 0.0
    recommended_multiplier: float = 0.0
    
    def __post_init__(self):
        self.edge = self.hit_rate - 0.5
        
        # Adjusted Kelly = raw × confidence × safety factor
        safety_factor = 0.5  # Half Kelly
        self.adjusted_kelly = self.raw_kelly * self.confidence * safety_factor
        
        # Recommended multiplier (relative to full position)
        # Based on edge quality and sample size
        self._compute_multiplier()
    
    def _compute_multiplier(self):
        """Compute recommended position multiplier."""
        
        # No edge = no position
        if self.hit_rate <= 0.50 or self.raw_kelly <= 0:
            self.recommended_multiplier = 0.0
            return
        
        # Base: scale by edge quality
        edge_quality = min(1.0, self.edge / 0.15)  # Cap at 15% edge
        
        # Sample size factor
        if self.n_samples < 20:
            sample_factor = 0.3
        elif self.n_samples < 50:
            sample_factor = 0.6
        elif self.n_samples < 100:
            sample_factor = 0.8
        else:
            sample_factor = 1.0
        
        # Confidence factor
        conf_factor = self.confidence
        
        # Win/loss ratio factor (bonus for asymmetric payoffs)
        wl_factor = min(1.2, max(0.8, self.win_loss_ratio))
        
        # Combine
        self.recommended_multiplier = edge_quality * sample_factor * conf_factor * wl_factor
        self.recommended_multiplier = np.clip(self.recommended_multiplier, 0.0, 1.0)


def extract_kelly_results(
    pair: str, 
    timeframe: str, 
    results: Dict
) -> List[KellyResult]:
    """Extract KellyResult objects from validation results."""
    
    kelly_inputs = results.get('statistical', {}).get('kelly_inputs', {})
    
    kelly_results = []
    
    for regime in [1, -1]:
        if regime not in kelly_inputs:
            continue
        
        ki = kelly_inputs[regime]
        
        kr = KellyResult(
            pair=pair,
            timeframe=timeframe,
            regime=regime,
            n_samples=ki.get('n_samples', 0),
            n_wins=ki.get('n_wins', 0),
            n_losses=ki.get('n_losses', 0),
            hit_rate=ki.get('hit_rate', 0.5),
            avg_win=ki.get('avg_win', 0),
            avg_loss=ki.get('avg_loss', 0),
            win_loss_ratio=ki.get('win_loss_ratio', 1.0),
            raw_kelly=ki.get('raw_kelly', 0),
            confidence=ki.get('confidence', 0),
        )
        
        kelly_results.append(kr)
    
    return kelly_results


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 80)
    print("KELLY INPUTS CALCULATOR")
    print("=" * 80)
    print(f"\nPairs: {PAIRS}")
    print(f"Timeframes: {list(CONFIGS.keys())}")
    
    # Connect to database
    print("\nConnecting to database...")
    db = Database()
    
    # Collect all Kelly results
    all_results: List[KellyResult] = []
    
    # Process each pair
    for pair in PAIRS:
        print(f"\n{'='*80}")
        print(f"PAIR: {pair}")
        print("=" * 80)
        
        # Load 1h data
        try:
            df_1h = db.get_ohlcv(pair, start=START_DATE, end=END_DATE)
            print(f"  Loaded {len(df_1h):,} 1h bars")
        except Exception as e:
            print(f"  ERROR loading {pair}: {e}")
            continue
        
        # Process each timeframe
        for tf, config in CONFIGS.items():
            print(f"\n{'-'*60}")
            print(f"{pair} - {tf}")
            print("-" * 60)
            
            try:
                # Resample
                df_tf = resample_ohlcv(df_1h, tf)
                print(f"  Resampled to {len(df_tf):,} bars")
                
                # Label and validate
                labeler = RegimeLabeler(config)
                labels = labeler.label(df_tf)
                results = labeler.validate(df_tf, labels, forward_periods=[1, 5, 10])
                
                # Extract Kelly results
                kelly_results = extract_kelly_results(pair, tf, results)
                all_results.extend(kelly_results)
                
                # Print Kelly inputs
                print(f"\n  Kelly Inputs:")
                for kr in kelly_results:
                    regime_name = "Up" if kr.regime == 1 else "Down"
                    print(f"    {regime_name:5}: n={kr.n_samples:3}, hit={kr.hit_rate*100:.1f}%, "
                          f"W/L={kr.win_loss_ratio:.2f}, kelly={kr.raw_kelly*100:.1f}%, "
                          f"conf={kr.confidence*100:.0f}%, mult={kr.recommended_multiplier:.2f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # ==========================================================================
    # Summary Tables
    # ==========================================================================
    
    if not all_results:
        print("\nNo results to summarize.")
        return
    
    print("\n" + "=" * 80)
    print("KELLY INPUTS SUMMARY")
    print("=" * 80)
    
    # Convert to DataFrame
    df_results = pd.DataFrame([
        {
            'pair': kr.pair,
            'timeframe': kr.timeframe,
            'regime': 'Up' if kr.regime == 1 else 'Down',
            'n_samples': kr.n_samples,
            'hit_rate': kr.hit_rate,
            'avg_win': kr.avg_win,
            'avg_loss': kr.avg_loss,
            'win_loss_ratio': kr.win_loss_ratio,
            'raw_kelly': kr.raw_kelly,
            'confidence': kr.confidence,
            'edge': kr.edge,
            'adjusted_kelly': kr.adjusted_kelly,
            'recommended_mult': kr.recommended_multiplier,
        }
        for kr in all_results
    ])
    
    # --- Table 1: Core Metrics ---
    print("\n--- Core Kelly Metrics ---")
    print(f"{'Pair':<8} {'TF':>5} {'Regime':>6} {'N':>5} {'Hit%':>6} {'W/L':>6} {'Kelly%':>7} {'Conf%':>6} {'Mult':>6}")
    print("-" * 70)
    
    for _, row in df_results.iterrows():
        print(f"{row['pair']:<8} {row['timeframe']:>5} {row['regime']:>6} "
              f"{row['n_samples']:>5} {row['hit_rate']*100:>5.1f}% {row['win_loss_ratio']:>6.2f} "
              f"{row['raw_kelly']*100:>6.1f}% {row['confidence']*100:>5.0f}% {row['recommended_mult']:>6.2f}")
    
    # --- Table 2: Win/Loss Details ---
    print("\n--- Win/Loss Details ---")
    print(f"{'Pair':<8} {'TF':>5} {'Regime':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'TotalWin':>10} {'TotalLoss':>10}")
    print("-" * 70)
    
    for _, row in df_results.iterrows():
        print(f"{row['pair']:<8} {row['timeframe']:>5} {row['regime']:>6} "
              f"{row['avg_win']*100:>+7.2f}% {row['avg_loss']*100:>8.2f}%")
    
    # --- Aggregated by Timeframe ---
    print("\n--- Aggregated by Timeframe (Mean across pairs) ---")
    tf_agg = df_results.groupby(['timeframe', 'regime']).agg({
        'n_samples': 'sum',
        'hit_rate': 'mean',
        'win_loss_ratio': 'mean',
        'raw_kelly': 'mean',
        'confidence': 'mean',
        'recommended_mult': 'mean',
    }).round(3)
    
    print(tf_agg.to_string())
    
    # ==========================================================================
    # Recommended Multipliers
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("RECOMMENDED REGIME MULTIPLIERS")
    print("=" * 80)
    
    print("""
These multipliers are derived from empirical data:
  - Hit rate → win probability
  - Win/Loss ratio → payoff asymmetry
  - Sample size → statistical confidence
  - Combined into Kelly-optimal sizing

Usage in KellySizer:
  multiplier = regime_multipliers[regime] × vol_adjustment
""")
    
    # Average across pairs for each TF
    for tf in CONFIGS.keys():
        print(f"\n{tf} Timeframe:")
        tf_data = df_results[df_results['timeframe'] == tf]
        
        for regime_name in ['Up', 'Down']:
            regime_data = tf_data[tf_data['regime'] == regime_name]
            if len(regime_data) == 0:
                print(f"  {regime_name:5}: No data")
                continue
            
            avg_mult = regime_data['recommended_mult'].mean()
            avg_hit = regime_data['hit_rate'].mean()
            avg_n = regime_data['n_samples'].sum()
            avg_kelly = regime_data['raw_kelly'].mean()
            
            print(f"  {regime_name:5}: mult={avg_mult:.2f} (hit={avg_hit*100:.1f}%, kelly={avg_kelly*100:.1f}%, n={avg_n})")
        
        # Flat regime
        print(f"  {'Flat':5}: mult=0.00 (no directional edge)")
    
    # ==========================================================================
    # Code Output
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("COPY-PASTE CODE")
    print("=" * 80)
    
    print("""
# Empirically-derived regime multipliers
# Generated from regime validation on historical data

REGIME_MULTIPLIERS = {""")
    
    for tf in CONFIGS.keys():
        tf_data = df_results[df_results['timeframe'] == tf]
        
        up_data = tf_data[tf_data['regime'] == 'Up']
        down_data = tf_data[tf_data['regime'] == 'Down']
        
        up_mult = up_data['recommended_mult'].mean() if len(up_data) > 0 else 0.0
        down_mult = down_data['recommended_mult'].mean() if len(down_data) > 0 else 0.0
        
        print(f"    '{tf}': {{{1}: {up_mult:.2f}, 0: 0.00, -1: {down_mult:.2f}}},")
    
    print("}")
    
    # Kelly inputs for direct use
    print("""
# Kelly inputs per regime (for advanced sizing)
# Formula: kelly = (p * b - q) / b where p=hit_rate, b=win_loss_ratio

KELLY_INPUTS = {""")
    
    for tf in CONFIGS.keys():
        tf_data = df_results[df_results['timeframe'] == tf]
        print(f"    '{tf}': {{")
        
        for regime_name in ['Up', 'Down']:
            regime_val = 1 if regime_name == 'Up' else -1
            regime_data = tf_data[tf_data['regime'] == regime_name]
            
            if len(regime_data) > 0:
                hit = regime_data['hit_rate'].mean()
                wl = regime_data['win_loss_ratio'].mean()
                kelly = regime_data['raw_kelly'].mean()
                conf = regime_data['confidence'].mean()
                n = regime_data['n_samples'].sum()
                
                print(f"        {regime_val}: {{'hit_rate': {hit:.3f}, 'win_loss_ratio': {wl:.2f}, "
                      f"'raw_kelly': {kelly:.3f}, 'confidence': {conf:.2f}, 'n_samples': {n}}},")
        
        print(f"    }},")
    
    print("}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return df_results


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    results = main()