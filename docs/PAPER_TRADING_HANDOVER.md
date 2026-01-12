# CryptoBot Paper Trading System - Handover Document

## Document Purpose

This document contains all information required to build a paper trading system for the validated 8-State Risk-Managed Risk Parity cryptocurrency trading strategy.

---

## 1. Project Overview

### Objective
Build a paper trading system that:
- Generates daily trading signals using the validated 8-state strategy
- Simulates trade execution with realistic costs
- Tracks portfolio performance in TimescaleDB
- Displays results in Grafana
- Uses identical logic to enable seamless transition to live trading

### Key Constraint
**Paper trading only** — Uses Kraken Public API for real price data, but no order placement. All trades are simulated locally.

---

## 2. Strategy Summary: 8-State Risk-Managed Risk Parity

### Validated Performance

| Metric | Value |
|--------|-------|
| Annual Return | ~40% |
| Sharpe Ratio | 1.0 - 1.3 |
| Max Drawdown | -31% to -39% |
| Bootstrap p-value | 0.001 (statistically significant) |
| Parameter Robustness | 97/100 |
| Trading Costs Modeled | 0.15% round-trip |

### Signal Generation Logic

#### Step 1: Calculate Moving Average Score (per pair)
```python
def calculate_ma_score(close_prices):
    """
    Score based on price vs 3 moving averages.
    Returns: -3 to +3
    """
    ma_24 = close_prices.rolling(24).mean()
    ma_72 = close_prices.rolling(72).mean()
    ma_168 = close_prices.rolling(168).mean()
    
    score = 0
    if close_prices > ma_24: score += 1
    else: score -= 1
    if close_prices > ma_72: score += 1
    else: score -= 1
    if close_prices > ma_168: score += 1
    else: score -= 1
    
    return score  # -3 to +3
```

#### Step 2: Determine 8-State (per pair)
```python
def get_8_state(ma_score, previous_state):
    """
    8 states based on MA score + hysteresis.
    States: -3, -2, -1, 0-, 0+, +1, +2, +3
    """
    entry_buffer = 0.02  # 2% buffer to enter
    exit_buffer = 0.005  # 0.5% buffer to exit
    
    # Map raw score to state with hysteresis
    if ma_score == 3:
        state = '+3'
    elif ma_score == 2:
        state = '+2'
    elif ma_score == 1:
        state = '+1'
    elif ma_score == 0 and previous_state in ['+1', '+2', '+3', '0+']:
        state = '0+'
    elif ma_score == 0:
        state = '0-'
    elif ma_score == -1:
        state = '-1'
    elif ma_score == -2:
        state = '-2'
    else:  # -3
        state = '-3'
    
    return state
```

#### Step 3: State to Signal Mapping
```python
STATE_HIT_RATES = {
    '+3': 0.57,  # Bullish
    '+2': 0.54,
    '+1': 0.52,
    '0+': 0.50,
    '0-': 0.48,
    '-1': 0.47,
    '-2': 0.45,
    '-3': 0.43,  # Bearish
}

HIT_RATE_THRESHOLD = 0.50

def state_to_signal(state):
    """
    Convert state to trading signal.
    Returns: 1 (long), 0 (flat), -1 (short - not used)
    """
    hit_rate = STATE_HIT_RATES[state]
    if hit_rate >= HIT_RATE_THRESHOLD:
        return 1  # Long
    else:
        return 0  # Flat (no shorting)
```

#### Step 4: Risk Parity Weights (across pairs)
```python
def calculate_risk_parity_weights(volatilities, signals):
    """
    Equal risk contribution across active pairs.
    
    Args:
        volatilities: dict of {pair: rolling_vol_168h}
        signals: dict of {pair: signal} (1 or 0)
    
    Returns:
        dict of {pair: weight}
    """
    active_pairs = [p for p, s in signals.items() if s == 1]
    
    if not active_pairs:
        return {p: 0.0 for p in signals}
    
    # Inverse volatility weighting
    inv_vols = {p: 1.0 / volatilities[p] for p in active_pairs}
    total_inv_vol = sum(inv_vols.values())
    
    weights = {}
    for pair in signals:
        if pair in active_pairs:
            weights[pair] = inv_vols[pair] / total_inv_vol
        else:
            weights[pair] = 0.0
    
    return weights
```

#### Step 5: Risk Management Overlay
```python
# Validated parameters from grid search
RISK_PARAMS = {
    'target_vol': 0.40,          # 40% target volatility
    'dd_start_reduce': -0.20,    # Start reducing at -20% drawdown
    'dd_full_reduce': -0.50,     # Minimum exposure at -50% drawdown
    'min_exposure_floor': 0.40,  # Never go below 40% exposure
}

def apply_risk_management(weights, portfolio_volatility, current_drawdown):
    """
    Apply volatility targeting and drawdown control.
    """
    # Volatility targeting
    if portfolio_volatility > 0:
        vol_scalar = RISK_PARAMS['target_vol'] / portfolio_volatility
        vol_scalar = min(vol_scalar, 2.0)  # Cap at 2x
    else:
        vol_scalar = 1.0
    
    # Drawdown control
    if current_drawdown <= RISK_PARAMS['dd_start_reduce']:
        # Linear reduction from dd_start to dd_full
        dd_range = RISK_PARAMS['dd_full_reduce'] - RISK_PARAMS['dd_start_reduce']
        dd_progress = (current_drawdown - RISK_PARAMS['dd_start_reduce']) / dd_range
        dd_progress = min(max(dd_progress, 0), 1)
        
        dd_scalar = 1.0 - dd_progress * (1.0 - RISK_PARAMS['min_exposure_floor'])
    else:
        dd_scalar = 1.0
    
    # Apply both scalars
    final_scalar = vol_scalar * dd_scalar
    
    adjusted_weights = {p: w * final_scalar for p, w in weights.items()}
    
    return adjusted_weights
```

### Trading Pairs (Validated)

| Pair | Status | Notes |
|------|--------|-------|
| XBTUSD | ✅ Primary | Best data coverage |
| ETHUSD | ✅ Active | Second largest |
| ADAUSD | ✅ Active | Good diversification |
| XLMUSD | ✅ Active | Lower correlation |
| XMRUSD | ✅ Active | Privacy coin exposure |
| ZECUSD | ✅ Active | Additional diversification |
| ETCUSD | ✅ Active | Ethereum classic |

---

## 3. Existing Platform Architecture

### Project Location
```
D:/cryptobot_docker/
```

### Key Existing Files

| File | Purpose | Use In Paper Trading |
|------|---------|---------------------|
| `database.py` | TimescaleDB connection | ✅ Use directly |
| `kraken_api.py` | Fetch OHLCV data (public API) | ✅ Use directly |
| `portfolio.py` | Position tracking | ✅ Extend or use |
| `executor.py` | Simulated execution | ✅ Adapt for paper trading |
| `base.py` | Feature registry | Reference only |
| `technical.py` | Technical indicators | Reference only |
| `volatility.py` | Volatility calculations | Reference only |
| `manager.py` | Risk management | Reference patterns |
| `kelly.py` | Position sizing | Reference only |

### Database Schema (TimescaleDB)

The database already has tables defined in `init_db.sql`:

```sql
-- OHLCV data (already populated)
CREATE TABLE ohlcv (
    timestamp TIMESTAMPTZ NOT NULL,
    pair VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    PRIMARY KEY (timestamp, pair)
);

-- Signals table (may need to create)
CREATE TABLE IF NOT EXISTS signals (
    timestamp TIMESTAMPTZ NOT NULL,
    pair VARCHAR(20) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    ma_score INTEGER,
    state VARCHAR(10),
    signal INTEGER,
    hit_rate DOUBLE PRECISION,
    raw_weight DOUBLE PRECISION,
    risk_adjusted_weight DOUBLE PRECISION,
    PRIMARY KEY (timestamp, pair, strategy)
);

-- Paper trades table (create new)
CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    pair VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'BUY' or 'SELL'
    quantity DOUBLE PRECISION,
    price DOUBLE PRECISION,
    cost DOUBLE PRECISION,  -- Transaction cost
    strategy VARCHAR(50) NOT NULL,
    PRIMARY KEY (id, timestamp)
);

-- Paper portfolio table (create new)
CREATE TABLE IF NOT EXISTS paper_portfolio (
    timestamp TIMESTAMPTZ NOT NULL,
    pair VARCHAR(20) NOT NULL,
    position DOUBLE PRECISION,
    avg_entry_price DOUBLE PRECISION,
    current_price DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    PRIMARY KEY (timestamp, pair)
);

-- Paper equity table (create new)
CREATE TABLE IF NOT EXISTS paper_equity (
    timestamp TIMESTAMPTZ NOT NULL,
    equity DOUBLE PRECISION,
    cash DOUBLE PRECISION,
    positions_value DOUBLE PRECISION,
    drawdown DOUBLE PRECISION,
    portfolio_volatility DOUBLE PRECISION,
    PRIMARY KEY (timestamp)
);
```

### Database Connection

```python
# From database.py
import os
from sqlalchemy import create_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://cryptobot:cryptobot_dev@localhost:5432/cryptobot"
)

engine = create_engine(DATABASE_URL)
```

### Docker Setup

```yaml
# docker-compose.yaml includes:
# - TimescaleDB on port 5432
# - Grafana on port 3000
```

---

## 4. Paper Trading System Requirements

### Components to Build

```
┌─────────────────────────────────────────────────────────────┐
│                  PAPER TRADING SYSTEM                        │
│                                                              │
│  ┌─────────────────┐                                        │
│  │  1. Data Fetcher │ ← Kraken Public API                   │
│  │     (hourly)     │   (already exists: kraken_api.py)     │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  2. Signal      │ ← 8-state logic                        │
│  │     Generator   │   (NEW: to build)                      │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  3. Position    │ ← Risk parity + risk mgmt              │
│  │     Calculator  │   (NEW: to build)                      │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  4. Paper       │ ← Simulate fills, apply costs          │
│  │     Executor    │   (NEW: to build)                      │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  5. Database    │ ← Store everything                     │
│  │     Logger      │   (NEW: to build)                      │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  6. Scheduler   │ ← Run daily                            │
│  │                 │   (NEW: to build)                      │
│  └─────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Proposed File Structure

```
D:/cryptobot_docker/
│
├── ... (existing files)
│
├── strategies/
│   ├── __init__.py
│   ├── base.py                    # BaseStrategy class
│   └── eight_state/
│       ├── __init__.py
│       ├── strategy.py            # 8-state signal generation
│       └── config.yaml            # Parameters
│
├── trading/
│   ├── __init__.py
│   ├── signal_manager.py          # Generate & store signals
│   ├── position_calculator.py     # Risk parity + risk mgmt
│   ├── paper_executor.py          # Simulate trades
│   ├── db_logger.py               # Write to TimescaleDB
│   └── runner.py                  # Main orchestrator
│
└── scripts/
    └── run_paper_trading.py       # Entry point
```

---

## 5. Configuration Parameters

### Strategy Parameters (Validated)

```yaml
# strategies/eight_state/config.yaml

strategy:
  name: "8-state-risk-managed-risk-parity"
  version: "1.0"
  
signals:
  ma_periods:
    short: 24
    medium: 72
    long: 168
  hysteresis:
    entry_buffer: 0.02
    exit_buffer: 0.005
  hit_rate_threshold: 0.50

risk_management:
  target_volatility: 0.40
  drawdown_start_reduce: -0.20
  drawdown_full_reduce: -0.50
  min_exposure_floor: 0.40

execution:
  trading_cost_pct: 0.0015  # 0.15% round-trip
  min_trade_threshold: 0.02  # Don't trade < 2% position change

pairs:
  - XBTUSD
  - ETHUSD
  - ADAUSD
  - XLMUSD
  - XMRUSD
  - ZECUSD
  - ETCUSD

initial_capital: 100000
```

### State Hit Rates (From Validation)

```yaml
# Empirically derived hit rates
hit_rates:
  "+3": 0.57
  "+2": 0.54
  "+1": 0.52
  "0+": 0.50
  "0-": 0.48
  "-1": 0.47
  "-2": 0.45
  "-3": 0.43
```

---

## 6. Data Requirements

### OHLCV Data
- **Source**: Kraken Public API (via existing `kraken_api.py`)
- **Frequency**: Hourly
- **History needed**: Minimum 168 hours (for MA calculations)
- **Storage**: TimescaleDB `ohlcv` table

### Lookback Requirements

| Calculation | Lookback Needed |
|-------------|-----------------|
| MA-24 | 24 hours |
| MA-72 | 72 hours |
| MA-168 | 168 hours |
| Rolling volatility | 168 hours |
| **Total minimum** | **168 hours** |

---

## 7. Execution Logic

### Paper Trade Execution

```python
def execute_paper_trade(
    pair: str,
    current_position: float,
    target_position: float,
    current_price: float,
    trading_cost_pct: float = 0.0015
) -> dict:
    """
    Simulate a trade with realistic costs.
    
    Returns:
        {
            'pair': str,
            'side': 'BUY' or 'SELL',
            'quantity': float,
            'price': float,
            'cost': float,
            'new_position': float
        }
    """
    position_change = target_position - current_position
    
    if abs(position_change) < MIN_TRADE_THRESHOLD:
        return None  # No trade
    
    trade = {
        'pair': pair,
        'side': 'BUY' if position_change > 0 else 'SELL',
        'quantity': abs(position_change),
        'price': current_price,
        'cost': abs(position_change) * current_price * trading_cost_pct,
        'new_position': target_position
    }
    
    return trade
```

---

## 8. Scheduling

### Run Frequency
- **Signal generation**: Daily (once per day)
- **Data update**: Hourly (ensure latest prices)
- **Recommended time**: After market close (if applicable) or fixed time daily

### Scheduler Options

| Option | Implementation |
|--------|----------------|
| Windows Task Scheduler | External, runs Python script |
| Cron (Linux) | External, runs Python script |
| APScheduler (Python) | Internal, runs in background |

### Simple Runner Script

```python
# scripts/run_paper_trading.py

from trading.runner import PaperTradingRunner

if __name__ == "__main__":
    runner = PaperTradingRunner(config_path="strategies/eight_state/config.yaml")
    
    # Run once (for cron/task scheduler)
    runner.run_daily()
    
    # OR run continuously with internal scheduler
    # runner.run_scheduled(hour=0, minute=0)  # Run at midnight
```

---

## 9. Grafana Dashboard Requirements

### Panels Needed

| Panel | Data Source | Query |
|-------|-------------|-------|
| Equity Curve | paper_equity | `SELECT timestamp, equity FROM paper_equity` |
| Drawdown | paper_equity | `SELECT timestamp, drawdown FROM paper_equity` |
| Current Positions | paper_portfolio | Latest positions by pair |
| Daily P&L | paper_equity | Day-over-day equity change |
| Trade Log | paper_trades | Recent trades |
| Signal States | signals | Current state per pair |

---

## 10. Testing Checklist

### Before Going Live with Paper Trading

| Test | Description | Status |
|------|-------------|--------|
| Signal accuracy | Compare with backtest signals | ☐ |
| Cost calculation | Verify 0.15% applied correctly | ☐ |
| Position sizing | Risk parity weights sum correctly | ☐ |
| Risk management | DD control triggers at -20% | ☐ |
| Database writes | All tables populated correctly | ☐ |
| Grafana display | All panels render | ☐ |
| Error handling | Graceful failure on API errors | ☐ |
| Idempotency | Running twice doesn't duplicate | ☐ |

---

## 11. Transition to Live Trading

### Changes Required for Live

| Component | Paper Trading | Live Trading |
|-----------|--------------|--------------|
| Price data | Kraken Public API | Same |
| Order execution | Simulated locally | Kraken Private API |
| API keys | None | Required |
| Tables | paper_* tables | live_* tables |
| Error handling | Log and continue | Alert and halt |

### Code Change

```python
# Paper trading
executor = PaperExecutor(db=database)

# Live trading (future)
executor = KrakenExecutor(
    api_key=os.getenv("KRAKEN_API_KEY"),
    api_secret=os.getenv("KRAKEN_API_SECRET")
)
```

---

## 12. Success Criteria

### Paper Trading Goals

| Metric | Target | Monitoring |
|--------|--------|------------|
| **Track record duration** | 3-6 months minimum | Grafana |
| **Match backtest** | Returns within 20% of expected | Compare monthly |
| **No bugs** | Zero calculation errors | Logging |
| **Uptime** | 99%+ daily runs | Alerts |

### What Proves Success

1. Paper equity curve tracks similar to backtest
2. Sharpe ratio in expected range (0.8 - 1.5)
3. Drawdowns within expected bounds (-40% max)
4. No look-ahead bias (signals use only past data)
5. Costs properly deducted

---

## 13. Important Notes

### No Look-Ahead Bias
- Signals must be generated AFTER the bar closes
- Use `close` price of previous bar for entry
- Never use future data

### Error Handling
- If Kraken API fails, retry 3x with backoff
- If database fails, log locally and alert
- Never skip a day without logging why

### Logging
- Log every signal generated
- Log every trade (or decision not to trade)
- Log daily portfolio state

---

## 14. Contact / Resources

### Existing Documentation
- `8STATE_RISK_MANAGED_STRATEGY.pdf` — Full strategy documentation
- `bootstrap_validation.py` — Statistical validation code
- `parameter_sensitivity_fast.py` — Robustness testing

### Key Validation Results
- Bootstrap p-value: 0.001 (alpha is real)
- 97/100 parameter robustness score
- Walk-forward: 73.3% of windows beat buy-and-hold

---

## Document Version

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 1.0 | 2025-01-10 | Claude | Initial handover document |

---

## Appendix A: Quick Start Commands

```bash
# Start Docker containers
cd D:/cryptobot_docker
docker-compose up -d

# Check database
docker exec -it cryptobot_timescaledb psql -U cryptobot -d cryptobot

# Run paper trading (once built)
python scripts/run_paper_trading.py
```

---

## Appendix B: Database Queries for Validation

```sql
-- Check latest OHLCV data
SELECT pair, MAX(timestamp) as latest
FROM ohlcv
GROUP BY pair
ORDER BY latest DESC;

-- Check signal generation
SELECT * FROM signals
WHERE timestamp > NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC;

-- Check paper equity
SELECT * FROM paper_equity
ORDER BY timestamp DESC
LIMIT 30;

-- Check paper trades
SELECT * FROM paper_trades
ORDER BY timestamp DESC
LIMIT 50;
```

---

*End of Handover Document*
