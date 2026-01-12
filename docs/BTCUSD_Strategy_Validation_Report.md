# CryptoBot Strategy Validation Report: BTCUSD

## Executive Summary

**Conclusion: No statistically significant trading edge was found.**

Despite initial backtested returns of +390% vs +197% buy-and-hold, rigorous validation revealed critical flaws that invalidated the results. After corrections, while some configurations still showed positive excess returns, none achieved statistical significance (p < 0.05).

---

## 1. Initial Strategy: "Avoid Danger"

### Concept
- Use regime detection (BinSeg + MSM) to identify market states
- Train ensemble model (LogReg + GBM) to predict 24h forward returns
- Exit positions when in "danger" regime AND prediction is bearish
- Regime 0 (Calm/Calm) with P < 0.50 was identified as optimal exit signal

### Initial Results (Appeared Promising)
| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Return | +390.3% | +197.1% |
| Sharpe | 1.20 | 0.89 |
| Max DD | -52.5% | -57.2% |

---

## 2. Validation Process & Findings

### 2.1 Look-Ahead Bias Discovery

**Problem:** The BinSeg regime detector used future data.

```python
# The ruptures library fits on ENTIRE series
model.fit(signal)  # Uses future data!
overall_median = vol_clean.median()  # Future data!
```

**Evidence:** When computing regime on train-only data vs full data:
- Match rate: **79.9%**
- Different values: **5,895 out of 29,271** (20.1%)

**Impact by Year:**
| Year | Misclassified Rows |
|------|-------------------|
| 2019 | 765 |
| 2020 | 2,085 |
| 2021 | 2,541 |
| 2022 | 504 |

**Fix:** Replaced with rolling regime detector using only past data:
```python
rolling_vol = log_ret.rolling(window=168).std()
rolling_median = rolling_vol.rolling(window=504).median()
regime = (rolling_vol > rolling_median).astype(int)
```

**Verification:** After fix, 100% match between train-only and full computation.

---

### 2.2 Transaction Cost Impact

**Assumptions:**
- Commission: 0.26% (Kraken taker fee)
- Slippage: 0.05%
- Total: 0.31% per trade

**Results by Holding Period:**

| Hold Period | Gross Return | Net Return | Trades |
|-------------|-------------|------------|--------|
| 1 hour | +243.9% | **-96.3%** | 1,461 |
| 24 hours | +243.9% | -23.5% | 445 |
| 72 hours | +243.9% | +30.8% | 233 |
| 168 hours | +243.9% | **+236.4%** | 133 |
| Buy & Hold | +197.1% | +195.2% | 2 |

**Finding:** Only 168-hour (1 week) minimum hold produced positive net returns exceeding buy-and-hold.

---

### 2.3 Statistical Significance Testing

**Method:** Block bootstrap (10,000 iterations, 168h blocks)

**Results for Avoid R0 @ P<0.50 (168h hold):**
```
Actual excess return: +39.3%
95% CI: [-901.7%, +587.9%]
P-value: 0.4095
Significance: NOT SIGNIFICANT
```

**Results for Avoid R3 @ P<0.58 (168h hold):**
```
Actual excess return: +4.7%
95% CI: [-1279.5%, +330.2%]
P-value: 0.5733
Significance: NOT SIGNIFICANT
```

**Why Confidence Intervals Are So Wide:**
1. Only ~133 independent decisions over 3 years (1/week with 168h hold)
2. Crypto returns are extremely volatile
3. Block bootstrap preserves return clustering

---

### 2.4 Out-of-Sample Testing

**Method:** Walk-forward testing across different years

| Period | Market | R0 Strategy | vs B&H |
|--------|--------|-------------|--------|
| 2021 | Bull (+62%) | +16.5% | ✗ -45% |
| 2022 | Bear (-65%) | -43.9% | ✓ +21% |
| 2023 | Bull (+156%) | +64.3% | ✗ -92% |
| 2024-25 | Bull (+155%) | +208.2% | ✓ +53% |

**Summary:**
- Win rate vs B&H: **2/4 (50%)**
- Average excess return: **-16.0%**

**Pattern:** Strategy protects in bear markets, loses in bull markets.

---

### 2.5 Long/Short Strategy Test

**Concept:** Instead of exiting, go SHORT when danger signal.

**Results:**
```
Long/Short: +184.5%
Buy & Hold: +197.1%
Excess: -12.6%
P-value: 0.5143
Significance: NOT SIGNIFICANT
```

---

## 3. Why the Strategy Failed

### 3.1 The "Edge" Was Illusory

| Validation Stage | Return | Valid? |
|-----------------|--------|--------|
| Initial (with look-ahead) | +390.3% | ❌ Biased |
| Without look-ahead | +243.9% | ❌ No costs |
| With costs (hourly) | -96.3% | ❌ Too many trades |
| With costs (168h hold) | +236.4% | ⚠️ p=0.41 |
| Out-of-sample | 50% win rate | ❌ Inconsistent |

### 3.2 Fundamental Issues

1. **Insufficient Independent Observations**
   - 168h hold = ~133 trades over 3 years
   - Not enough data to distinguish skill from luck

2. **High Variance Asset**
   - Crypto volatility drowns out signal
   - Any edge is swamped by noise

3. **Strategy is a Bear Market Hedge, Not Alpha**
   - Works when avoiding crashes (2022: -44% vs -65%)
   - Loses by sitting out during rallies (2023: +64% vs +156%)

4. **Parameter Sensitivity**
   - Results highly sensitive to holding period
   - Optimal regime changes with threshold
   - Signs of overfitting to test period

---

## 4. What Was Learned

### 4.1 Critical Validation Steps
1. **Check for look-ahead bias** - Compute features on train-only, compare to full
2. **Include transaction costs** - Can destroy any edge
3. **Test statistical significance** - Bootstrap with appropriate block size
4. **Out-of-sample testing** - Multiple time periods, different market conditions

### 4.2 Regime Detection Insights
- BinSeg (ruptures library) has inherent look-ahead bias
- Rolling volatility regime works without bias
- Regime alone doesn't predict returns (underperforms B&H)
- ML predictions add value only when combined with regime

### 4.3 Economic Insight
The ML model + regime combination may detect:
- Distribution periods disguised as consolidation
- Volatility compression before breakdowns

But this is more useful as **crash insurance** than as a return enhancer.

---

## 5. Recommendations

### 5.1 If Proceeding with Strategy
- Treat as **bear market hedge**, not alpha generator
- Accept underperformance during bull runs
- Size positions conservatively given uncertainty
- Use 168h minimum hold to manage costs

### 5.2 Alternative Directions
1. **Multi-asset testing** - Pool signals across 12 cryptocurrencies for more observations
2. **Simplify further** - Pure volatility timing without ML
3. **Different timeframes** - Test on daily/weekly data for longer history
4. **Accept B&H** - For long-term crypto exposure, buy-and-hold remains robust

### 5.3 What NOT To Do
- Don't trust backtest returns without validation
- Don't optimize parameters on test data
- Don't ignore transaction costs
- Don't assume stationarity of any edge found

---

## 6. Technical Details

### 6.1 Data
- Source: Kraken via TimescaleDB
- Period: 2019-01-01 to 2025-09-30
- Granularity: Hourly OHLCV
- Train/Test Split: 50/50

### 6.2 Features (Look-Ahead Free)
```python
feature_cols = [
    'ma_score',           # Count of MAs above price
    'price_vs_sma_6',     # Price / 6h SMA
    'price_vs_sma_24',    # Price / 24h SMA
    'price_vs_sma_72',    # Price / 72h SMA
    'rolling_vol_168h',   # 168h rolling volatility
    'garch_vol_simple',   # EWMA volatility proxy
    'regime_binseg',      # Rolling vol vs median
    'regime_msm'          # Short vol vs long vol
]
```

### 6.3 Model
- Ensemble: Logistic Regression + Gradient Boosting
- Target: Binary 24h forward return
- Prediction: Average of both model probabilities

### 6.4 Regime Definition
```
regime_hybrid = regime_binseg * 2 + regime_msm
  R0 (Calm/Calm):     binseg=0, msm=0 → vol low, not spiking
  R1 (Calm/Volatile): binseg=0, msm=1 → vol low but spiking
  R2 (Vol/Calm):      binseg=1, msm=0 → vol high but settling
  R3 (Vol/Volatile):  binseg=1, msm=1 → vol high and spiking
```

---

## 7. Files & Code

### Key Functions Created
- `compute_features_no_lookahead()` - Bias-free feature computation
- `compute_regime_binseg_rolling()` - Rolling regime detector
- `backtest_with_costs()` - Realistic cost modeling
- `backtest_with_hold_period()` - Minimum hold constraint

### Database
- Table: `ohlcv`
- Pairs loaded: XBTUSD, ETHUSD, LTCUSD, ETCUSD, XMRUSD, ZECUSD, XRPUSD, XLMUSD, ADAUSD, LINKUSD, SOLUSD, AVAXUSD

---

## 8. Conclusion

**The honest assessment:** After rigorous validation, no statistically significant trading edge was found for the BTCUSD regime-based strategy. The initial +390% return was an artifact of look-ahead bias. The corrected +236% return (with costs and 168h hold) has a p-value of 0.41, meaning we cannot reject the hypothesis that it occurred by chance.

**Next step:** Test across multiple cryptocurrencies to increase sample size and determine if any edge is real and generalizable.

---

*Report generated: December 2024*
*Test period: May 2022 - September 2025*
*Methodology: Look-ahead-free features, realistic costs, block bootstrap*
