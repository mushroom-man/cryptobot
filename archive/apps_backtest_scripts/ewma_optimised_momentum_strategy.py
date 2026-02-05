#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EWMA Lambda Optimizer - TRUE Out-of-Sample
==========================================
Train (50%) → Validation (25%) → Test (25%)

Optuna optimizes on VALIDATION period.
Final results on TEST period (never seen during optimization).
"""

import sys
sys.path.insert(0, '/home/johnhenry/cryptobot')

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from cryptobot.data.database import Database
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEPLOY_PAIRS = ['XLMUSD', 'ZECUSD', 'ETCUSD', 'XMRUSD', 'ADAUSD']

LAMBDA_MIN = 0.80
LAMBDA_MAX = 0.9995
LAMBDA_MIN_SEP = 0.02

CONFIRMATION_HOURS = 3
KELLY_FRACTION = 0.25
COST_PER_SIDE = 0.0035
SLIPPAGE = 0.0005

N_TRIALS = 200

# Three-way split
TRAIN_PCT = 0.50      # First 50% - select bullish states
VALID_PCT = 0.25      # Next 25% - Optuna optimizes here
TEST_PCT = 0.25       # Last 25% - final evaluation (NEVER seen during optimization)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptResult:
    lambda_fast: float
    lambda_med: float
    lambda_slow: float
    lambda_vslow: float
    sharpe_valid: float
    sharpe_test: float
    annual_return_test: float
    max_drawdown_test: float
    n_trades_test: int
    exposure_test: float
    bullish_states: set


# =============================================================================
# EWMA BACKTESTER
# =============================================================================

class EWMABacktester:
    
    def __init__(self, hourly_data: Dict[str, pd.DataFrame]):
        self.hourly_data = hourly_data
        self.eval_count = 0
        self.best_sharpe = -999
        
        # Pre-split data
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self._split_data()
    
    def _split_data(self):
        """Split hourly data into train/valid/test."""
        for pair, df_1h in self.hourly_data.items():
            n = len(df_1h)
            train_end = int(n * TRAIN_PCT)
            valid_end = int(n * (TRAIN_PCT + VALID_PCT))
            
            self.train_data[pair] = df_1h.iloc[:train_end]
            self.valid_data[pair] = df_1h.iloc[train_end:valid_end]
            self.test_data[pair] = df_1h.iloc[valid_end:]
    
    def compute_states_for_pair(self, df_1h: pd.DataFrame, lambdas: Tuple) -> pd.DataFrame:
        if len(df_1h) < 100:
            return pd.DataFrame()
        
        df = df_1h.copy()
        
        for i, name in enumerate(['fast', 'med', 'slow', 'vslow']):
            alpha = 1 - lambdas[i]
            df[f'ewma_{name}'] = df['close'].ewm(alpha=alpha, adjust=False).mean()
        
        df = df.dropna()
        
        df['state'] = (
            (df['close'] > df['ewma_fast']).astype(int) * 8 +
            (df['close'] > df['ewma_slow']).astype(int) * 4 +
            (df['ewma_fast'] > df['ewma_med']).astype(int) * 2 +
            (df['ewma_slow'] > df['ewma_med']).astype(int) * 1
        )
        
        if CONFIRMATION_HOURS > 0:
            raw = df['state'].values
            confirmed = np.zeros_like(raw)
            current = int(raw[0])
            pending = None
            pending_count = 0
            
            for i in range(len(raw)):
                r = int(raw[i])
                if r == current:
                    pending = None
                    pending_count = 0
                elif r == pending:
                    pending_count += 1
                    if pending_count >= CONFIRMATION_HOURS:
                        current = pending
                        pending = None
                        pending_count = 0
                else:
                    pending = r
                    pending_count = 1
                confirmed[i] = current
            
            df['state'] = confirmed
        
        daily = df.resample('24h').agg({'close': 'last', 'state': 'last'}).dropna()
        daily['return'] = daily['close'].pct_change()
        
        return daily
    
    def find_bullish_states(self, daily_data: Dict[str, pd.DataFrame]) -> set:
        state_returns = {s: [] for s in range(16)}
        
        for pair, daily in daily_data.items():
            states = daily['state'].values
            returns = daily['return'].values
            
            for i in range(1, len(returns)):
                state = int(states[i])
                ret = returns[i]
                if not np.isnan(ret):
                    state_returns[state].append(ret)
        
        bullish = set()
        state_sharpes = {}
        for state, rets in state_returns.items():
            if len(rets) > 30:
                mean_r = np.mean(rets)
                std_r = np.std(rets)
                if std_r > 0:
                    sharpe = (mean_r / std_r) * np.sqrt(365)
                    state_sharpes[state] = sharpe
                    if sharpe > 0.3:
                        bullish.add(state)
        
        if len(bullish) < 2:
            sorted_states = sorted(state_sharpes.items(), key=lambda x: x[1], reverse=True)
            bullish = {s for s, _ in sorted_states[:4]}
        
        return bullish
    
    def backtest_period(self, daily_data: Dict[str, pd.DataFrame], 
                        bullish_states: set) -> Dict:
        """Run backtest on a specific period."""
        sharpes = []
        returns_list = []
        max_dds = []
        total_trades = 0
        exposures = []
        
        for pair, daily in daily_data.items():
            if len(daily) < 10:
                continue
            
            states = daily['state'].values
            returns = daily['return'].values
            n = len(returns)
            
            position = np.array([1.0 if int(states[i]) in bullish_states else 0.0 for i in range(n)])
            strat_returns = position[1:] * returns[1:] * KELLY_FRACTION
            
            n_trades = int(np.sum(np.abs(np.diff(position)) > 0.5))
            total_trades += n_trades
            
            cost_per_trade = (COST_PER_SIDE + SLIPPAGE) * KELLY_FRACTION
            total_cost = n_trades * cost_per_trade
            
            if len(strat_returns) > 0 and np.std(strat_returns) > 0:
                mean_r = np.mean(strat_returns)
                std_r = np.std(strat_returns)
                sharpe = (mean_r / std_r) * np.sqrt(365)
                
                years = n / 365
                annual_cost = total_cost / years if years > 0 else 0
                sharpe_adj = sharpe - (annual_cost / std_r / np.sqrt(365)) if std_r > 0 else sharpe
                sharpes.append(sharpe_adj)
                
                total_ret = np.sum(strat_returns) - total_cost
                ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
                returns_list.append(ann_ret)
                
                equity = np.cumprod(1 + strat_returns)
                rolling_max = np.maximum.accumulate(equity)
                dd = (equity - rolling_max) / rolling_max
                max_dds.append(np.min(dd))
                
                exposures.append(np.mean(position))
        
        return {
            'sharpe': np.mean(sharpes) if sharpes else -999,
            'annual_return': np.mean(returns_list) if returns_list else 0,
            'max_dd': np.mean(max_dds) if max_dds else -1,
            'n_trades': total_trades,
            'exposure': np.mean(exposures) if exposures else 0,
        }
    
    def evaluate_for_optuna(self, lambdas: Tuple[float, float, float, float]) -> float:
        """
        Optuna objective: 
        - Select bullish states on TRAIN
        - Evaluate on VALIDATION
        """
        self.eval_count += 1
        
        lf, lm, ls, lv = lambdas
        if not (lf + LAMBDA_MIN_SEP < lm < ls - LAMBDA_MIN_SEP and ls + LAMBDA_MIN_SEP < lv):
            return -999
        
        # Compute states on TRAIN data
        train_daily = {}
        for pair, df_1h in self.train_data.items():
            daily = self.compute_states_for_pair(df_1h, lambdas)
            if len(daily) > 30:
                train_daily[pair] = daily
        
        if not train_daily:
            return -999
        
        # Find bullish states on TRAIN only
        bullish_states = self.find_bullish_states(train_daily)
        
        # Compute states on VALIDATION data
        valid_daily = {}
        for pair, df_1h in self.valid_data.items():
            daily = self.compute_states_for_pair(df_1h, lambdas)
            if len(daily) > 10:
                valid_daily[pair] = daily
        
        if not valid_daily:
            return -999
        
        # Evaluate on VALIDATION
        result = self.backtest_period(valid_daily, bullish_states)
        sharpe = result['sharpe']
        
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            print(f"    Trial {self.eval_count:3d}: λ=({lf:.3f},{lm:.3f},{ls:.3f},{lv:.3f}) "
                  f"Valid Sharpe={sharpe:.3f} * (states: {sorted(bullish_states)})")
        
        return sharpe
    
    def final_evaluation(self, lambdas: Tuple) -> OptResult:
        """
        Final evaluation:
        - Select bullish states on TRAIN
        - Report VALIDATION sharpe (what Optuna saw)
        - Report TEST sharpe (truly out-of-sample)
        """
        # Compute on TRAIN
        train_daily = {}
        for pair, df_1h in self.train_data.items():
            daily = self.compute_states_for_pair(df_1h, lambdas)
            if len(daily) > 30:
                train_daily[pair] = daily
        
        bullish = self.find_bullish_states(train_daily)
        
        # Validation results
        valid_daily = {}
        for pair, df_1h in self.valid_data.items():
            daily = self.compute_states_for_pair(df_1h, lambdas)
            if len(daily) > 10:
                valid_daily[pair] = daily
        
        valid_result = self.backtest_period(valid_daily, bullish)
        
        # TEST results (never seen during optimization)
        test_daily = {}
        for pair, df_1h in self.test_data.items():
            daily = self.compute_states_for_pair(df_1h, lambdas)
            if len(daily) > 10:
                test_daily[pair] = daily
        
        test_result = self.backtest_period(test_daily, bullish)
        
        return OptResult(
            lambda_fast=lambdas[0],
            lambda_med=lambdas[1],
            lambda_slow=lambdas[2],
            lambda_vslow=lambdas[3],
            sharpe_valid=valid_result['sharpe'],
            sharpe_test=test_result['sharpe'],
            annual_return_test=test_result['annual_return'],
            max_drawdown_test=test_result['max_dd'],
            n_trades_test=test_result['n_trades'],
            exposure_test=test_result['exposure'],
            bullish_states=bullish,
        )


# =============================================================================
# MA BASELINE
# =============================================================================

def compute_ma_baseline_test(hourly_data: Dict[str, pd.DataFrame]) -> Dict:
    """MA system evaluated on TEST period only."""
    
    MA_PERIOD_24H, MA_PERIOD_72H, MA_PERIOD_168H = 16, 6, 2
    ENTRY_BUFFER, EXIT_BUFFER = 0.025, 0.005
    
    def resample(df, tf):
        return df.resample(tf).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
    
    def hysteresis(close, ma, entry_buf, exit_buf):
        trend = []
        current = 1
        for i in range(len(close)):
            if pd.isna(ma.iloc[i]):
                trend.append(current)
                continue
            p, m = close.iloc[i], ma.iloc[i]
            if current == 1 and p < m * (1 - exit_buf):
                current = 0
            elif current == 0 and p > m * (1 + entry_buf):
                current = 1
            trend.append(current)
        return trend
    
    sharpes, returns_list, max_dds, total_trades, exposures = [], [], [], 0, []
    
    for pair, df_1h in hourly_data.items():
        # Use only TEST period
        n = len(df_1h)
        test_start = int(n * (TRAIN_PCT + VALID_PCT))
        df_1h_test = df_1h.iloc[test_start:]
        
        if len(df_1h_test) < 500:
            continue
        
        df_24h = resample(df_1h_test, '24h')
        df_72h = resample(df_1h_test, '72h')
        df_168h = resample(df_1h_test, '168h')
        
        if len(df_24h) < MA_PERIOD_24H:
            continue
        
        ma_24h = df_24h['close'].rolling(MA_PERIOD_24H).mean()
        ma_72h = df_72h['close'].rolling(MA_PERIOD_72H).mean()
        ma_168h = df_168h['close'].rolling(MA_PERIOD_168H).mean()
        
        ma_24h_h = ma_24h.reindex(df_1h_test.index, method='ffill')
        ma_72h_h = ma_72h.reindex(df_1h_test.index, method='ffill')
        ma_168h_h = ma_168h.reindex(df_1h_test.index, method='ffill')
        
        trend_24h = hysteresis(df_1h_test['close'], ma_24h_h, ENTRY_BUFFER, EXIT_BUFFER)
        trend_168h = hysteresis(df_1h_test['close'], ma_168h_h, ENTRY_BUFFER, EXIT_BUFFER)
        
        states = pd.DataFrame(index=df_1h_test.index)
        states['close'] = df_1h_test['close']
        states['state'] = (
            pd.Series(trend_24h, index=df_1h_test.index) * 8 +
            pd.Series(trend_168h, index=df_1h_test.index) * 4 +
            (ma_72h_h > ma_24h_h).astype(int) * 2 +
            (ma_168h_h > ma_24h_h).astype(int) * 1
        )
        states = states.dropna()
        
        # Confirmation
        raw = states['state'].values
        confirmed = np.zeros_like(raw)
        current = int(raw[0])
        pending, pending_count = None, 0
        
        for i in range(len(raw)):
            r = int(raw[i])
            if r == current:
                pending, pending_count = None, 0
            elif r == pending:
                pending_count += 1
                if pending_count >= CONFIRMATION_HOURS:
                    current = pending
                    pending, pending_count = None, 0
            else:
                pending, pending_count = r, 1
            confirmed[i] = current
        
        states['state'] = confirmed
        
        daily = states.resample('24h').agg({'close': 'last', 'state': 'last'}).dropna()
        daily['return'] = daily['close'].pct_change()
        
        bullish = {8, 9, 10, 11, 13, 15}
        state_arr = daily['state'].values
        ret_arr = daily['return'].values
        n_days = len(ret_arr)
        
        position = np.array([1.0 if int(state_arr[i]) in bullish else 0.0 for i in range(n_days)])
        strat_returns = position[1:] * ret_arr[1:] * KELLY_FRACTION
        
        n_trades = int(np.sum(np.abs(np.diff(position)) > 0.5))
        total_trades += n_trades
        
        if len(strat_returns) > 0 and np.std(strat_returns) > 0:
            sharpes.append((np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(365))
            years = n_days / 365
            returns_list.append((1 + np.sum(strat_returns)) ** (1 / years) - 1 if years > 0 else 0)
            equity = np.cumprod(1 + strat_returns)
            max_dds.append(np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)))
            exposures.append(np.mean(position))
    
    return {
        'sharpe': np.mean(sharpes) if sharpes else 0,
        'annual_return': np.mean(returns_list) if returns_list else 0,
        'max_dd': np.mean(max_dds) if max_dds else -1,
        'n_trades': total_trades,
        'exposure': np.mean(exposures) if exposures else 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 95)
    print("  EWMA LAMBDA OPTIMIZER - TRUE OUT-OF-SAMPLE VALIDATION")
    print("=" * 95)
    
    print(f"""
    Data Split:
    ─────────────────────────────────────────────────────────
    TRAIN  ({TRAIN_PCT:.0%}):  Select bullish states
    VALID  ({VALID_PCT:.0%}):  Optuna optimizes lambda here
    TEST   ({TEST_PCT:.0%}):  Final evaluation (NEVER seen by optimizer)
    ─────────────────────────────────────────────────────────
    
    Configuration:
    ─────────────────────────────────────────────────────────
    Lambda range:       [{LAMBDA_MIN}, {LAMBDA_MAX}]
    Trials:             {N_TRIALS}
    Pairs:              {', '.join(DEPLOY_PAIRS)}
    ─────────────────────────────────────────────────────────
    """)
    
    db = Database()
    
    # Load data
    hourly_data = {}
    for pair in DEPLOY_PAIRS:
        print(f"  Loading {pair}...", end=" ", flush=True)
        df_1h = db.get_ohlcv(pair)
        if df_1h is None or len(df_1h) == 0:
            print("SKIP")
            continue
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h.index = pd.to_datetime(df_1h.index)
        hourly_data[pair] = df_1h.sort_index()
        
        n = len(df_1h)
        test_start = df_1h.index[int(n * (TRAIN_PCT + VALID_PCT))]
        print(f"{len(df_1h):,} hours (test starts: {test_start.date()})")
    
    # MA baseline on TEST
    print("\n  Computing MA baseline (test period only)...")
    ma_baseline = compute_ma_baseline_test(hourly_data)
    print(f"  MA Baseline (TEST): Sharpe={ma_baseline['sharpe']:.2f}, "
          f"Return={ma_baseline['annual_return']:+.1%}")
    
    # Create backtester
    backtester = EWMABacktester(hourly_data)
    
    # Optuna optimization (on VALIDATION, not TEST)
    print(f"\n  Running Optuna optimization ({N_TRIALS} trials on VALIDATION period)...")
    
    def objective(trial):
        lf = trial.suggest_float('lambda_fast', 0.80, 0.91)
        lm = trial.suggest_float('lambda_med', 0.85, 0.95)
        ls = trial.suggest_float('lambda_slow', 0.90, 0.98)
        lv = trial.suggest_float('lambda_vslow', 0.95, 0.999)
        
        if not (lf + LAMBDA_MIN_SEP < lm < ls - LAMBDA_MIN_SEP and ls + LAMBDA_MIN_SEP < lv):
            return -999
        
        return backtester.evaluate_for_optuna((lf, lm, ls, lv))
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    # Get best result and evaluate on TEST
    best = study.best_params
    lambdas = (best['lambda_fast'], best['lambda_med'], best['lambda_slow'], best['lambda_vslow'])
    result = backtester.final_evaluation(lambdas)
    
    # Display results
    print(f"\n{'=' * 95}")
    print("  RESULTS")
    print(f"{'=' * 95}")
    
    diff_valid = result.sharpe_valid - ma_baseline['sharpe']
    diff_test = result.sharpe_test - ma_baseline['sharpe']
    
    print(f"""
    ┌───────────────────────────────────────────────────────────────────────────────────────────┐
    │  OPTIMAL EWMA CONFIGURATION                                                               │
    ├───────────────────────────────────────────────────────────────────────────────────────────┤
    │  Lambdas:  fast={result.lambda_fast:.4f}, med={result.lambda_med:.4f}, slow={result.lambda_slow:.4f}, vslow={result.lambda_vslow:.4f}          │
    │  Bullish States: {sorted(result.bullish_states)}                                                     │
    ├───────────────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                           │
    │                         EWMA (Valid)     EWMA (Test)      MA (Test)       Δ (Test)        │
    │  ─────────────────────────────────────────────────────────────────────────────────────── │
    │  Sharpe Ratio:         {result.sharpe_valid:>8.2f}         {result.sharpe_test:>8.2f}         {ma_baseline['sharpe']:>8.2f}         {diff_test:>+8.2f}        │
    │  Annual Return:                         {result.annual_return_test:>+8.1%}         {ma_baseline['annual_return']:>+8.1%}                          │
    │  Max Drawdown:                          {result.max_drawdown_test:>8.1%}         {ma_baseline['max_dd']:>8.1%}                          │
    │  Trades:                                {result.n_trades_test:>8}         {ma_baseline['n_trades']:>8}                          │
    │  Exposure:                              {result.exposure_test:>8.1%}         {ma_baseline['exposure']:>8.1%}                          │
    │                                                                                           │
    ├───────────────────────────────────────────────────────────────────────────────────────────┤
    │  VALIDATION vs TEST Sharpe: {result.sharpe_valid:.2f} → {result.sharpe_test:.2f} (degradation: {result.sharpe_valid - result.sharpe_test:+.2f})                              │
    │                                                                                           │
    │  Conclusion (TEST period): {'EWMA WINS' if diff_test > 0.1 else 'MA WINS' if diff_test < -0.1 else 'SIMILAR'}                                                     │
    └───────────────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Sanity check
    if result.sharpe_test > 5:
        print("""
    ⚠️  WARNING: Test Sharpe > 5 is still suspiciously high.
        Consider: implementation bug, data error, or lucky period.
        Recommend: paper trade before trusting these results.
        """)
    
    print(f"  Evaluations: {backtester.eval_count}")
    print(f"\n{'=' * 95}")
    
    return result, ma_baseline, study


if __name__ == "__main__":
    result, ma_baseline, study = main()