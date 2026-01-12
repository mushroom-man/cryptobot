# CryptoBot Backtesting Platform Architecture

**Version:** 1.0  
**Date:** December 2024  
**Status:** Design Complete — Ready for Implementation

---

## Executive Summary

This document defines the architecture for a comprehensive backtesting platform for CryptoBot. The platform supports:

- Multi-timeframe analysis (hourly, daily, weekly)
- External data integration (VIX, SPY, DXY)
- Multiple model types (ML, GARCH, ARIMA) with ensembling
- Kelly criterion position sizing with signal combinations
- Walk-forward validation
- Statistical significance testing
- RAG-based institutional memory
- Claude AI integration for analysis and code fixes

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Layer Specifications](#3-layer-specifications)
4. [Multi-Timeframe Design](#4-multi-timeframe-design)
5. [Configuration System](#5-configuration-system)
6. [Module Breakdown](#6-module-breakdown)
7. [Implementation Priorities](#7-implementation-priorities)
8. [Key Findings from Analysis](#8-key-findings-from-analysis)
9. [Open Questions](#9-open-questions)

---

## 1. Platform Overview

### Core Requirements

| Requirement | Description |
|-------------|-------------|
| Data Management | Pull data, set train/val/test splits |
| Feature Engineering | Compute features across multiple timeframes |
| Feature Selection | Statistical and ML-based selection methods |
| Model Training | ML, GARCH, ARIMA, hybrid ensembles |
| Hyperparameter Optimization | Grid, random, Bayesian (Optuna) |
| Walk-Forward Validation | Rolling retrain for robustness |
| Signal Generation | Convert predictions to trading signals |
| Position Sizing | Kelly criterion with signal-derived weights |
| Backtesting | Realistic simulation with costs |
| Evaluation | Metrics, graphs, statistical tests |
| Storage | Save configs, results, models |
| AI Integration | Claude analysis, RAG memory, code fixes |

### Design Principles

1. **Modularity** — Each component is independent and testable
2. **Reproducibility** — Full config saved with every run
3. **No Look-Ahead Bias** — Strict data alignment across timeframes
4. **Realistic Costs** — Transaction costs, slippage, rebalancing
5. **Statistical Rigor** — Bootstrap, Monte Carlo, significance tests

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKTESTING PLATFORM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         1. DATA LAYER                                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │   Internal   │  │   External   │  │  Timeframe   │                 │ │
│  │  │    Crypto    │  │  VIX/SPY/DXY │  │  Resampler   │                 │ │
│  │  │   (Kraken)   │  │   (Yahoo)    │  │  1h→1d→1w    │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  │         │                 │                 │                         │ │
│  │         └─────────────────┼─────────────────┘                         │ │
│  │                           ▼                                           │ │
│  │                   ┌──────────────┐                                    │ │
│  │                   │    Data      │                                    │ │
│  │                   │   Aligner    │                                    │ │
│  │                   │ (no lookahd) │                                    │ │
│  │                   └──────────────┘                                    │ │
│  │                           │                                           │ │
│  │                           ▼                                           │ │
│  │                   ┌──────────────┐                                    │ │
│  │                   │ Train/Val/   │                                    │ │
│  │                   │ Test Split   │                                    │ │
│  │                   └──────────────┘                                    │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      2. FEATURE LAYER                                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                  MULTI-TIMEFRAME FEATURES                       │  │ │
│  │  ├─────────────────────────────────────────────────────────────────┤  │ │
│  │  │                                                                 │  │ │
│  │  │  HOURLY (1h)          DAILY (1d)          WEEKLY (1w)          │  │ │
│  │  │  ────────────         ──────────          ──────────           │  │ │
│  │  │  • vol_zscore_1h      • vol_zscore_1d     • vol_zscore_1w      │  │ │
│  │  │  • ma_ratio_24h       • ma_ratio_5d       • ma_ratio_4w        │  │ │
│  │  │  • ma_slope_24h       • ma_slope_5d       • ma_slope_4w        │  │ │
│  │  │  • rsi_14h            • rsi_14d           • trend_regime_1w    │  │ │
│  │  │  • macd_hist_1h       • macd_hist_1d      • macro_regime_1w    │  │ │
│  │  │  • stoch_k_1h         • stoch_k_1d                             │  │ │
│  │  │                                                                 │  │ │
│  │  │  EXTERNAL (aligned to hourly)                                  │  │ │
│  │  │  ────────────────────────────                                  │  │ │
│  │  │  • vix_level          • vix_change        • vix_regime         │  │ │
│  │  │  • spy_trend          • spy_vs_ma         • risk_on_off        │  │ │
│  │  │  • dxy_trend          • dxy_momentum                           │  │ │
│  │  │                                                                 │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                           │                                           │ │
│  │                           ▼                                           │ │
│  │                   ┌──────────────┐                                    │ │
│  │                   │   Feature    │                                    │ │
│  │                   │  Selection   │                                    │ │
│  │                   │ (MI, RF, L1) │                                    │ │
│  │                   └──────────────┘                                    │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       3. MODEL LAYER                                   │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │  Hyperparam  │  │    Model     │  │   Walk-Fwd   │                 │ │
│  │  │   Search     │→ │   Training   │→ │  Validation  │                 │ │
│  │  │(Optuna/Grid) │  │              │  │              │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  │                           │                                           │ │
│  │         ┌─────────────────┼─────────────────┐                         │ │
│  │         ▼                 ▼                 ▼                         │ │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐                   │ │
│  │  │     ML     │    │   GARCH    │    │   ARIMA    │                   │ │
│  │  │ (LR,GBM,RF)│    │ Volatility │    │   Trend    │                   │ │
│  │  └────────────┘    └────────────┘    └────────────┘                   │ │
│  │         │                 │                 │                         │ │
│  │         └─────────────────┼─────────────────┘                         │ │
│  │                           ▼                                           │ │
│  │                   ┌──────────────┐                                    │ │
│  │                   │   Ensemble   │                                    │ │
│  │                   │Avg/Stack/Vote│                                    │ │
│  │                   └──────────────┘                                    │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      4. SIGNAL LAYER                                   │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │   Signal     │  │    Kelly     │  │   Position   │                 │ │
│  │  │  Generator   │→ │  Calculator  │→ │   Sizing     │                 │ │
│  │  │              │  │  (per signal)│  │              │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  │                                             │                         │ │
│  │  ┌──────────────────────────────────────────┘                         │ │
│  │  │                                                                    │ │
│  │  │  MULTI-TIMEFRAME SIGNAL COMBINATION                               │ │
│  │  │  ──────────────────────────────────                               │ │
│  │  │                                                                    │ │
│  │  │  Option A: Hierarchical Filter                                    │ │
│  │  │  ┌────────┐    ┌────────┐    ┌────────┐                           │ │
│  │  │  │Weekly  │ →  │Daily   │ →  │Hourly  │ → Trade                   │ │
│  │  │  │Regime  │    │Bias    │    │Trigger │                           │ │
│  │  │  └────────┘    └────────┘    └────────┘                           │ │
│  │  │                                                                    │ │
│  │  │  Option B: Weighted Ensemble                                      │ │
│  │  │  Weekly(0.2) + Daily(0.3) + Hourly(0.5) = Combined Signal         │ │
│  │  │                                                                    │ │
│  │  │  Option C: Regime-Conditional                                     │ │
│  │  │  Weekly regime → selects which Daily/Hourly model to use          │ │
│  │  │                                                                    │ │
│  │  └────────────────────────────────────────────────────────────────── │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     5. BACKTEST ENGINE                                 │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │  Execution   │  │    Risk      │  │  Transaction │                 │ │
│  │  │  Simulator   │  │  Management  │  │    Costs     │                 │ │
│  │  │              │  │(Limits,Stops)│  │  (Slippage)  │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  │         │                 │                 │                         │ │
│  │         └─────────────────┼─────────────────┘                         │ │
│  │                           ▼                                           │ │
│  │                   ┌──────────────┐                                    │ │
│  │                   │   Backtest   │                                    │ │
│  │                   │     Loop     │                                    │ │
│  │                   │ (Vectorized) │                                    │ │
│  │                   └──────────────┘                                    │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     6. EVALUATION LAYER                                │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │   Metrics    │  │    Stat      │  │  Benchmark   │                 │ │
│  │  │ Sharpe,DD,   │  │   Tests      │  │  Comparison  │                 │ │
│  │  │ Return, etc  │  │ Bootstrap,MC │  │  B&H, EW     │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  │         │                 │                 │                         │ │
│  │         └─────────────────┼─────────────────┘                         │ │
│  │                           ▼                                           │ │
│  │  ┌──────────────┐  ┌──────────────┐                                   │ │
│  │  │    Graphs    │  │   Summary    │                                   │ │
│  │  │  MPL/Plotly  │  │   Report     │                                   │ │
│  │  └──────────────┘  └──────────────┘                                   │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      7. STORAGE LAYER                                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │   Config     │  │   Results    │  │   Models     │                 │ │
│  │  │   (YAML)     │  │ (TimescaleDB)│  │  (Pickle)    │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  │                           │                                           │ │
│  │                           ▼                                           │ │
│  │                   ┌──────────────┐                                    │ │
│  │                   │  Run History │                                    │ │
│  │                   │   & Versions │                                    │ │
│  │                   └──────────────┘                                    │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      8. AI LAYER                                       │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │     RAG      │  │   Claude     │  │    Code      │                 │ │
│  │  │   Memory     │  │   Analysis   │  │    Fixer     │                 │ │
│  │  │              │  │  (optional)  │  │  (optional)  │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  │         │                 │                 │                         │ │
│  │         └─────────────────┼─────────────────┘                         │ │
│  │                           ▼                                           │ │
│  │                   ┌──────────────┐                                    │ │
│  │                   │ Institutional│                                    │ │
│  │                   │    Memory    │                                    │ │
│  │                   │  (Learnings) │                                    │ │
│  │                   └──────────────┘                                    │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer Specifications

### 3.1 Data Layer

**Purpose:** Load, align, and split data across timeframes and sources.

| Component | Responsibility |
|-----------|----------------|
| Internal Loader | Crypto OHLCV from Kraken via Database class |
| External Loader | VIX, SPY, DXY from Yahoo Finance |
| Resampler | Convert hourly → daily → weekly |
| Aligner | Merge timeframes without look-ahead bias |
| Splitter | Train/validation/test splits |

**Look-Ahead Prevention:**
```
Hourly bar at Mon 03:00:
├── Uses LAST WEEK's weekly features (closed Sunday 23:59)
├── Uses YESTERDAY's daily features (closed Sunday 23:59)
└── Uses current hourly features
```

### 3.2 Feature Layer

**Purpose:** Compute and select features across all timeframes.

**New Features to Implement:**

| Category | Features |
|----------|----------|
| **Continuous MA** | `ma_ratio_24`, `ma_ratio_72`, `ma_ratio_168` |
| **MA Slope** | `ma_slope_24`, `ma_slope_72` |
| **MA Spread** | `ma_spread_24_72`, `ma_spread_24_168` |
| **Momentum** | `macd_line`, `macd_signal`, `macd_histogram` |
| **Oscillators** | `stochastic_k`, `stochastic_d` |
| **External** | `vix_level`, `vix_zscore`, `spy_trend`, `dxy_trend` |

**Selection Methods:**
- Mutual Information
- Random Forest Importance
- L1 Regularization (Lasso)
- Correlation with target

### 3.3 Model Layer

**Purpose:** Train, validate, and ensemble prediction models.

| Model Type | Use Case |
|------------|----------|
| Logistic Regression | Baseline, interpretable |
| Gradient Boosting | Non-linear relationships |
| Random Forest | Feature importance, robustness |
| GARCH | Volatility forecasting |
| ARIMA | Trend forecasting |

**Ensemble Methods:**
- Simple Average
- Weighted Average (by validation performance)
- Stacking (meta-model)
- Voting (classification)

**Validation Methods:**
- Single Split (fast iteration)
- Walk-Forward (production validation)
- Expanding Window (more training data over time)

### 3.4 Signal Layer

**Purpose:** Convert predictions to position sizes via Kelly criterion.

**Signal Definition:**
```python
@dataclass
class Signal:
    name: str
    condition: Callable[[pd.DataFrame], pd.Series]
    signal_type: str = "continuous"  # or "discrete", "binary"
```

**Kelly Calculation:**
```
kelly = (p * b - q) / b

where:
    p = win rate
    q = 1 - p
    b = avg_win / avg_loss
```

**Multi-Timeframe Combination:**

| Method | Description |
|--------|-------------|
| Hierarchical | Weekly filters → Daily filters → Hourly triggers |
| Weighted Ensemble | Weighted average of timeframe signals |
| Regime-Conditional | Weekly regime selects which model to use |

### 3.5 Backtest Engine

**Purpose:** Simulate trading with realistic execution.

| Component | Details |
|-----------|---------|
| Execution Simulator | Fill orders at specified prices |
| Transaction Costs | 31 bps (26 bps Kraken + 5 bps slippage) |
| Rebalancing | Configurable period (default 168h) |
| Risk Management | Position limits, drawdown limits, circuit breakers |

### 3.6 Evaluation Layer

**Purpose:** Measure and validate performance.

**Metrics:**
- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

**Statistical Tests:**
- Bootstrap confidence intervals
- Monte Carlo simulation
- t-test (return ≠ 0)
- Paired t-test (strategy A vs B)
- White's Reality Check (multiple testing correction)

**Benchmarks:**
- Buy and Hold
- Equal Weight
- Risk Parity

### 3.7 Storage Layer

**Purpose:** Persist configs, results, and models.

| Data | Format | Storage |
|------|--------|---------|
| Config | YAML | Filesystem |
| Results | Tables | TimescaleDB |
| Models | Pickle | Filesystem |
| Time Series | Parquet | TimescaleDB |

### 3.8 AI Layer

**Purpose:** Analysis, memory, and automation.

| Component | Function |
|-----------|----------|
| RAG Memory | Query past backtests, learnings, rationale |
| Claude Analysis | Interpret results, suggest improvements |
| Code Fixer | Auto-fix errors with human approval |

---

## 4. Multi-Timeframe Design

### Timeframe Roles

| Timeframe | Role | What it captures |
|-----------|------|------------------|
| Weekly | Strategic direction | Major trends, macro regime |
| Daily | Tactical bias | Intermediate trend, key levels |
| Hourly | Execution | Entry/exit timing, short-term signals |

### Data Alignment

```
Weekly bar (closes Sunday 23:59):
├── Mon 00:00 - 23:59 (24 hourly bars)
├── Tue 00:00 - 23:59 (24 hourly bars)
├── Wed 00:00 - 23:59 (24 hourly bars)
├── Thu 00:00 - 23:59 (24 hourly bars)
├── Fri 00:00 - 23:59 (24 hourly bars)
├── Sat 00:00 - 23:59 (24 hourly bars)
└── Sun 00:00 - 23:59 (24 hourly bars)
                       └─→ Weekly features available AFTER this
```

### External Data Challenge

| Market | Hours | Alignment Strategy |
|--------|-------|-------------------|
| Crypto | 24/7 | Primary timeframe |
| US Equities | Mon-Fri 9:30-16:00 ET | Use prior close |
| VIX | Same as equities | Use prior close |
| Forex (DXY) | Sun 17:00 - Fri 17:00 ET | Nearly 24h |

---

## 5. Configuration System

### BacktestConfig

```python
@dataclass
class BacktestConfig:
    """Complete configuration — saved with every backtest."""
    
    # ===== DATA =====
    assets: List[str]
    start_date: str
    end_date: str
    train_pct: float = 0.6
    val_pct: float = 0.2
    test_pct: float = 0.2
    
    # ===== TIMEFRAMES =====
    primary_timeframe: str = "1h"
    additional_timeframes: List[str] = ["1d", "1w"]
    timeframe_combination: str = "hierarchical"
    include_external: bool = True
    external_symbols: List[str] = ["^VIX", "SPY", "DX-Y.NYB"]
    
    # ===== FEATURES =====
    features_1h: List[str]
    features_1d: List[str]
    features_1w: List[str]
    features_external: List[str]
    selection_method: str = "mutual_info"
    n_features: int = 15
    
    # ===== MODEL =====
    model_type: str = "ensemble"
    ensemble_models: List[str] = ["logistic", "gbm", "garch"]
    ensemble_method: str = "weighted_average"
    hyperparam_search: str = "optuna"
    hyperparam_trials: int = 100
    validation_method: str = "walk_forward"
    walk_forward_windows: int = 5
    expanding_window: bool = True
    
    # ===== SIGNALS =====
    signal_threshold: float = 0.5
    
    # ===== POSITION SIZING =====
    sizing_method: str = "kelly"
    kelly_fraction: float = 0.25
    max_position: float = 0.30
    min_position: float = 0.01
    
    # ===== EXECUTION =====
    rebalance_hours: int = 168
    cost_bps: float = 31
    slippage_bps: float = 5
    
    # ===== RISK =====
    max_drawdown_limit: float = 0.25
    circuit_breaker_enabled: bool = True
    
    # ===== EVALUATION =====
    benchmark: str = "buy_and_hold"
    stat_tests: List[str] = ["bootstrap", "monte_carlo", "t_test"]
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # ===== OUTPUT =====
    generate_graphs: bool = True
    graph_backend: str = "both"
    save_results: bool = True
    storage_backend: str = "timescaledb"
    
    # ===== AI =====
    send_to_claude: bool = False
    use_rag: bool = True
    auto_fix_code: bool = False
    
    # ===== METADATA =====
    name: str = ""
    description: str = ""
    tags: List[str] = []
    random_seed: int = 42
    version: str = "1.0.0"
```

### BacktestResults

```python
@dataclass
class BacktestResults:
    """Complete results — saved with config for reproducibility."""
    
    # Identity
    config: BacktestConfig
    run_id: str
    
    # Performance
    metrics: Dict[str, float]
    metrics_by_asset: pd.DataFrame
    
    # Time series
    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    signals: pd.DataFrame
    
    # Model info
    feature_importance: pd.DataFrame
    selected_features: List[str]
    model_params: Dict
    
    # Diagnostics
    trade_log: pd.DataFrame
    costs_breakdown: Dict
    
    # Statistical tests
    bootstrap_ci: Dict
    monte_carlo_results: Dict
    
    # Methods
    def save(self, path: str): ...
    def load(cls, path: str): ...
    def summary(self) -> str: ...
    def to_claude_prompt(self) -> str: ...
```

---

## 6. Module Breakdown

### Implementation Modules

| Module | Priority | Depends On | Status |
|--------|----------|------------|--------|
| `data/loader.py` | 1 | — | Exists (Database) |
| `data/resampler.py` | 1 | loader | NEW |
| `data/aligner.py` | 1 | resampler | NEW |
| `data/external.py` | 2 | aligner | NEW |
| `features/technical.py` | 1 | data | Update (add MA, MACD, Stoch) |
| `features/multi_tf.py` | 2 | technical | NEW |
| `selection/methods.py` | 2 | features | Exists (features.py) |
| `models/base.py` | 2 | selection | NEW |
| `models/ml.py` | 2 | base | NEW |
| `models/garch.py` | 2 | base | Exists (volatility.py) |
| `models/ensemble.py` | 3 | ml, garch | NEW |
| `signals/generator.py` | 2 | models | NEW |
| `signals/kelly.py` | 2 | generator | Exists (kelly.py) |
| `signals/combiner.py` | 3 | kelly | NEW |
| `backtest/engine.py` | 2 | signals | Exists (runner.py) |
| `backtest/walk_forward.py` | 3 | engine | NEW |
| `evaluation/metrics.py` | 2 | backtest | Exists (metrics.py) |
| `evaluation/stats.py` | 3 | metrics | NEW |
| `evaluation/graphs.py` | 3 | metrics | NEW |
| `storage/timescale.py` | 2 | evaluation | Exists (database.py) |
| `ai/claude.py` | 4 | storage | NEW |
| `ai/rag.py` | 4 | storage | NEW |

---

## 7. Implementation Priorities

### Phase 1: Foundation (Immediate)

| Task | Description |
|------|-------------|
| Fix `ma_score` | Replace discrete 0-3 with continuous MA indicators |
| Add MACD | `macd_line`, `macd_signal`, `macd_histogram` |
| Add Stochastic | `stochastic_k`, `stochastic_d` |
| Data Resampler | Convert hourly → daily → weekly |

### Phase 2: Core Platform

| Task | Description |
|------|-------------|
| Multi-timeframe features | Namespace by timeframe |
| Signal framework | Signal class, Kelly calculator, combiner |
| Walk-forward validation | Rolling retrain |
| Config system | Full BacktestConfig dataclass |

### Phase 3: Advanced Features

| Task | Description |
|------|-------------|
| External data | VIX, SPY, DXY integration |
| Hyperparameter optimization | Optuna integration |
| Statistical tests | Bootstrap, Monte Carlo |
| Model ensembling | Stacking, weighted average |

### Phase 4: AI Integration

| Task | Description |
|------|-------------|
| RAG memory | Store and query learnings |
| Claude analysis | Auto-generate insights |
| Code fixer | Error detection and fixes |

---

## 8. Key Findings from Analysis

### What Works

| Finding | Evidence |
|---------|----------|
| Vol mean reversion | 90% reversion at 168h |
| Vol signal Kelly weight | +2.4% per unit |
| Regime detection | 73.8% return spread |
| 168h holding period | Optimal for vol timing |

### What Doesn't Work

| Finding | Evidence |
|---------|----------|
| Discrete ma_score | Never goes negative (no bearish signal) |
| Linear momentum | U-shaped relationship (both extremes good) |
| Trend signal (current) | +1.7% Kelly weight, but no bearish data |
| Direction prediction | Only 5% correlation |

### Critical Issues to Fix

1. **ma_score is broken** — Only values 0-3, never negative
2. **Momentum is U-shaped** — Linear Kelly weight captures nothing
3. **No external data** — Missing VIX, SPY, DXY signals

---

## 9. Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Walk-forward windows | 3, 5, 10 | TBD |
| Timeframe weights | Equal, Kelly-derived | TBD |
| External data frequency | Daily only, or hourly with fill | TBD |
| RAG implementation | Vector DB, or simpler | TBD |
| Claude API integration | Direct call, or manual | Direct call |

---

## Appendix A: Existing Code Assets

| File | Purpose | Reuse |
|------|---------|-------|
| `kelly.py` | Kelly sizer with regime/MA multipliers | Yes, extend |
| `runner.py` | Event-driven backtest | Yes, extend |
| `features.py` | Feature analysis | Yes |
| `technical.py` | MA, RSI, ATR | Yes, extend |
| `volatility.py` | Vol features, GARCH | Yes |
| `regime.py` | BinSeg, MSM, Hybrid | Yes |
| `database.py` | Data access | Yes |
| `metrics.py` | Performance metrics | Yes |

---

## Appendix B: External Data Sources

| Symbol | Source | Frequency | Notes |
|--------|--------|-----------|-------|
| ^VIX | Yahoo Finance | Daily | Fear index |
| SPY | Yahoo Finance | Daily | S&P 500 ETF |
| DX-Y.NYB | Yahoo Finance | Daily | Dollar index |
| GLD | Yahoo Finance | Daily | Gold ETF |
| ^TNX | Yahoo Finance | Daily | 10Y Treasury yield |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial architecture design |

---

*End of Document*
