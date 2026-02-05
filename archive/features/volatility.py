# -*- coding: utf-8 -*-
"""
CryptoBot - Volatility Features
================================
Rolling volatility, GARCH forecasts, and volatility regime indicators.

Features:
    - rolling_vol_24h: 24-hour realized volatility
    - rolling_vol_168h: 168-hour (weekly) realized volatility
    - garch_vol: GARCH(1,1) volatility forecast
    - vol_regime: High/low volatility classification
"""

import pandas as pd
import numpy as np
import warnings
from cryptobot.features.base import (
    Feature, register_feature, register_group, FeatureGroup,
    compute_log_returns, compute_rolling_std
)


# =============================================================================
# Rolling Volatility
# =============================================================================

@register_feature
class RollingVol24H(Feature):
    """24-hour rolling volatility (annualized std of log returns)."""
    name = "rolling_vol_24h"
    lookback = 25
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_returns = compute_log_returns(df, 'close', periods=1)
        rolling_std = log_returns.rolling(window=24, min_periods=24).std()
        # Annualize (24 hours * 365 days = 8760 hours/year)
        return rolling_std * np.sqrt(8760)


@register_feature
class RollingVol72H(Feature):
    """72-hour (3-day) rolling volatility (annualized)."""
    name = "rolling_vol_72h"
    lookback = 73
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_returns = compute_log_returns(df, 'close', periods=1)
        rolling_std = log_returns.rolling(window=72, min_periods=72).std()
        return rolling_std * np.sqrt(8760)


@register_feature
class RollingVol168H(Feature):
    """168-hour (weekly) rolling volatility (annualized)."""
    name = "rolling_vol_168h"
    lookback = 169
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_returns = compute_log_returns(df, 'close', periods=1)
        rolling_std = log_returns.rolling(window=168, min_periods=168).std()
        return rolling_std * np.sqrt(8760)


@register_feature
class RollingVolRaw168H(Feature):
    """168-hour rolling std of log returns (non-annualized, for regime detection)."""
    name = "rolling_vol_raw_168h"
    lookback = 169
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_returns = compute_log_returns(df, 'close', periods=1)
        return log_returns.rolling(window=168, min_periods=168).std()


# =============================================================================
# GARCH Volatility
# =============================================================================

@register_feature
class GARCHVol(Feature):
    """
    GARCH(1,1) volatility forecast.
    
    Uses arch library. Falls back to rolling vol if not available.
    """
    name = "garch_vol"
    lookback = 500  # Need sufficient data for GARCH estimation
    output_type = "continuous"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.refit_every = params.get('refit_every', 168)  # Refit weekly
        self._model = None
        self._last_fit = 0
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute rolling GARCH forecasts.
        
        For efficiency, refits model periodically rather than every row.
        """
        try:
            from arch import arch_model
        except ImportError:
            warnings.warn("arch library not installed. Using rolling volatility fallback.")
            return self._fallback_vol(df)
        
        log_returns = compute_log_returns(df, 'close', periods=1) * 100  # Scale for GARCH
        log_returns = log_returns.dropna()
        
        if len(log_returns) < self.lookback:
            return pd.Series(index=df.index, dtype=float)
        
        # Initialize output
        forecasts = pd.Series(index=df.index, dtype=float)
        
        # Rolling GARCH with periodic refitting
        window = 500
        
        for i in range(window, len(df)):
            if i % self.refit_every == 0 or self._model is None:
                # Refit model
                train_data = log_returns.iloc[i-window:i]
                try:
                    model = arch_model(train_data, vol='GARCH', p=1, q=1, rescale=False)
                    self._model = model.fit(disp='off', show_warning=False)
                except Exception:
                    continue
            
            if self._model is not None:
                try:
                    forecast = self._model.forecast(horizon=1)
                    var_forecast = forecast.variance.values[-1, 0]
                    # Convert back to annualized vol
                    forecasts.iloc[i] = np.sqrt(var_forecast) * np.sqrt(8760) / 100
                except Exception:
                    pass
        
        return forecasts
    
    def _fallback_vol(self, df: pd.DataFrame) -> pd.Series:
        """Fallback to EWMA volatility if GARCH fails."""
        log_returns = compute_log_returns(df, 'close', periods=1)
        ewma_var = log_returns.ewm(span=72, adjust=False).var()
        return np.sqrt(ewma_var) * np.sqrt(8760)


@register_feature
class GARCHVolSimple(Feature):
    """
    Simplified GARCH-like volatility using EWMA.
    
    Faster than full GARCH, captures similar dynamics.
    """
    name = "garch_vol_simple"
    lookback = 73
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_returns = compute_log_returns(df, 'close', periods=1)
        
        # EWMA variance (similar to RiskMetrics)
        ewma_var = log_returns.ewm(span=72, adjust=False).var()
        
        # Annualize
        return np.sqrt(ewma_var) * np.sqrt(8760)


# =============================================================================
# Volatility Regime Indicators
# =============================================================================

@register_feature
class VolRegime(Feature):
    """
    Volatility regime classification.
    
    0 = Low vol (below 25th percentile of rolling vol)
    1 = Normal vol
    2 = High vol (above 75th percentile)
    """
    name = "vol_regime"
    lookback = 504  # 3 weeks to establish percentiles
    output_type = "discrete"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_returns = compute_log_returns(df, 'close', periods=1)
        rolling_vol = log_returns.rolling(window=168).std()
        
        # Rolling percentiles
        p25 = rolling_vol.rolling(window=504).quantile(0.25)
        p75 = rolling_vol.rolling(window=504).quantile(0.75)
        
        regime = pd.Series(1, index=df.index)  # Default normal
        regime[rolling_vol < p25] = 0  # Low vol
        regime[rolling_vol > p75] = 2  # High vol
        
        return regime


@register_feature
class VolZScore(Feature):
    """
    Volatility z-score: How extreme is current vol vs history.
    
    High positive = unusually high vol
    Negative = unusually low vol
    """
    name = "vol_zscore"
    lookback = 504
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_returns = compute_log_returns(df, 'close', periods=1)
        rolling_vol = log_returns.rolling(window=168).std()
        
        mean_vol = rolling_vol.rolling(window=504).mean()
        std_vol = rolling_vol.rolling(window=504).std()
        
        return (rolling_vol - mean_vol) / std_vol


@register_feature
class VolOfVol(Feature):
    """
    Volatility of volatility - measures vol clustering/regime changes.
    """
    name = "vol_of_vol"
    lookback = 336  # 2 weeks
    output_type = "continuous"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_returns = compute_log_returns(df, 'close', periods=1)
        rolling_vol = log_returns.rolling(window=24).std()
        
        # Std of the rolling volatility
        return rolling_vol.rolling(window=168).std()


# =============================================================================
# Feature Groups
# =============================================================================

register_group(FeatureGroup(
    name="volatility",
    features=['rolling_vol_24h', 'rolling_vol_168h', 'garch_vol_simple', 'vol_zscore'],
    description="Volatility measures and forecasts"
))

register_group(FeatureGroup(
    name="volatility_full",
    features=['rolling_vol_24h', 'rolling_vol_72h', 'rolling_vol_168h', 
              'garch_vol_simple', 'vol_regime', 'vol_zscore', 'vol_of_vol'],
    description="Complete volatility feature set"
))
