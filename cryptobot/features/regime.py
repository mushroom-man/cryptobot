# -*- coding: utf-8 -*-
"""
CryptoBot - Regime Detection Features
======================================
Structural (LASSO) and Tactical (MSM) regime detection.

Architecture:
    - RegimeStructural: Abstract base class for structural regime detectors
    - RegimeLASSO: LASSO-based structural regime using MA alignment labels
    - RegimeMSM: Markov Switching Model for tactical/momentum regime
    - RegimeHybrid: Combines structural + tactical into 4-state regime

Regime States (Hybrid):
    0 = Bear/Calm (structural down, tactical calm)
    1 = Bear/Volatile (structural down, tactical active)
    2 = Bull/Calm (structural up, tactical calm)
    3 = Bull/Volatile (structural up, tactical active)

Features:
    - regime_structural: Binary (0=bear, 1=bull) from LASSO
    - regime_msm: Binary (0=calm, 1=volatile) from MSM
    - regime_hybrid: Combined 4-state regime (0-3)
    - regime_multiplier: Position size multiplier based on regime
"""

import pandas as pd
import numpy as np
import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Any
from cryptobot.features.base import (
    Feature, register_feature, register_group, FeatureGroup,
    compute_log_returns, compute_sma
)


# =============================================================================
# Abstract Base Class for Structural Regime
# =============================================================================

class RegimeStructural(Feature):
    """
    Abstract base class for structural regime detectors.
    
    Structural regimes are slow-moving (weeks/months) and detect
    the underlying market environment (bull/bear, high/low vol).
    
    Subclasses must implement:
        - compute(df) -> pd.Series with binary output (0/1)
    
    This allows swapping implementations:
        - RegimeLASSO (current)
        - Future: HMM, online change point, etc.
    """
    
    output_type = "binary"
    regime_type = "structural"
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute structural regime. Must return 0 (bear) or 1 (bull)."""
        pass
    
    def get_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Get training labels for supervised approaches.
        
        Default: MA alignment labels (can be overridden).
        """
        return self._compute_ma_alignment_labels(df)
    
    def _compute_ma_alignment_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute regime labels from MA alignment.
        
        Uses slope of MAs at multiple timeframes:
            - 24h MA rising + 72h MA rising + 168h MA rising = Bull (1)
            - Otherwise = Bear (0)
        
        No look-ahead bias: uses only past data for each point.
        """
        close = df['close']
        
        # Compute MAs
        ma_24 = compute_sma(close, 24)
        ma_72 = compute_sma(close, 72)
        ma_168 = compute_sma(close, 168)
        
        # Compute slopes (change over lookback period)
        # Using 24h lookback for slope calculation
        slope_24 = ma_24.diff(24) / ma_24.shift(24)
        slope_72 = ma_72.diff(24) / ma_72.shift(24)
        slope_168 = ma_168.diff(24) / ma_168.shift(24)
        
        # Bull = majority of MAs rising (2+ out of 3)
        rising_count = (
            (slope_24 > 0).astype(int) +
            (slope_72 > 0).astype(int) +
            (slope_168 > 0).astype(int)
        )
        
        # Label: 1 if 2+ MAs rising, else 0
        labels = (rising_count >= 2).astype(int)
        
        return labels


# =============================================================================
# LASSO Structural Regime
# =============================================================================

@register_feature
class RegimeLASSO(RegimeStructural):
    """
    LASSO-based structural regime classifier.
    
    Uses MA alignment to define regime labels, then trains LASSO
    to identify which features best predict the structural regime.
    
    Training approach:
        1. Define labels from MA alignment (bull/bear)
        2. Train LASSO logistic regression on feature set
        3. Use trained model for regime classification
    
    Output: 0 = bear, 1 = bull
    
    Parameters:
        feature_cols: List of features to use (None = auto-detect all)
        exclude_cols: List of features to exclude from auto-detect
        alpha: LASSO regularization strength (higher = more sparse)
        retrain_every: Bars between retraining (default 720 = ~30 days)
        train_window: Training window size (default 2160 = ~90 days)
        min_train_samples: Minimum samples required for training
    
    Usage:
        # Auto-detect features
        regime = RegimeLASSO()
        result = regime.compute(df_with_features)
        
        # Custom features
        regime = RegimeLASSO(feature_cols=['rolling_vol_168h', 'rsi_14', 'atr_percent'])
        result = regime.compute(df_with_features)
        
        # Exclude certain features
        regime = RegimeLASSO(exclude_cols=['regime_msm', 'regime_hybrid'])
        result = regime.compute(df_with_features)
    """
    name = "regime_structural"
    lookback = 504  # 3 weeks for MA calculations
    output_type = "binary"
    
    # Columns to always exclude from feature detection
    EXCLUDE_ALWAYS = [
        # Target/label columns
        'regime_structural', 'regime_msm', 'regime_hybrid', 'regime_multiplier',
        'ma_alignment', 'target', 'label',
        # OHLCV columns
        'open', 'high', 'low', 'close', 'volume', 'volume_quote',
        # Metadata
        'pair', 'source', 'timestamp',
    ]
    
    def __init__(self, **params):
        super().__init__(**params)
        self.feature_cols = params.get('feature_cols', None)  # None = auto-detect
        self.exclude_cols = params.get('exclude_cols', [])    # Additional exclusions
        self.alpha = params.get('alpha', 1.0)                 # LASSO regularization
        self.retrain_every = params.get('retrain_every', 720) # ~30 days
        self.train_window = params.get('train_window', 2160)  # ~90 days
        self.min_train_samples = params.get('min_train_samples', 500)
        
        # Model state
        self._model = None
        self._scaler = None
        self._trained_feature_cols = None
        self._last_train_idx = 0
        self._training_stats = {}
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute LASSO structural regime.
        
        Automatically trains on available features and predicts regime.
        Falls back to MA alignment if training fails or insufficient data.
        
        Args:
            df: DataFrame with OHLCV and feature columns
        
        Returns:
            Series with regime values (0=bear, 1=bull)
        """
        # Get MA alignment labels (always available as fallback)
        labels = self.get_labels(df)
        
        # Detect features to use
        features_to_use = self._get_feature_columns(df)
        
        if not features_to_use:
            # No features available, return raw labels
            return labels.fillna(0).astype(int)
        
        # Check if we need to train/retrain
        need_training = (
            self._model is None or
            self._trained_feature_cols is None or
            len(df) - self._last_train_idx >= self.retrain_every
        )
        
        if need_training:
            success = self._train_model(df, features_to_use, labels)
            if success:
                self._last_train_idx = len(df)
        
        # Predict using trained model (or fallback to labels)
        if self._model is not None and self._trained_feature_cols is not None:
            return self._predict(df)
        else:
            return labels.fillna(0).astype(int)
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature columns to use for training.
        
        If feature_cols specified, use those.
        Otherwise, auto-detect numeric columns excluding known non-features.
        """
        if self.feature_cols is not None:
            # Use specified features (validate they exist)
            valid_cols = [c for c in self.feature_cols if c in df.columns]
            if len(valid_cols) < len(self.feature_cols):
                missing = set(self.feature_cols) - set(valid_cols)
                warnings.warn(f"Some specified features not found: {missing}")
            return valid_cols
        
        # Auto-detect: all numeric columns except exclusions
        exclude_set = set(self.EXCLUDE_ALWAYS + list(self.exclude_cols))
        
        feature_cols = []
        for col in df.columns:
            if col in exclude_set:
                continue
            if col.startswith('regime_'):  # Exclude all regime columns
                continue
            if col.startswith('_'):  # Exclude private/internal columns
                continue
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                feature_cols.append(col)
        
        return feature_cols
    
    def _train_model(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str],
        labels: pd.Series
    ) -> bool:
        """
        Train LASSO logistic regression model.
        
        Args:
            df: DataFrame with features
            feature_cols: Columns to use as features
            labels: Target labels (0/1)
        
        Returns:
            True if training successful
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            warnings.warn("sklearn not available for LASSO training. Install with: pip install scikit-learn")
            return False
        
        # Prepare data
        X = df[feature_cols].copy()
        y = labels.copy()
        
        # Use only recent data (training window)
        if len(X) > self.train_window:
            X = X.iloc[-self.train_window:]
            y = y.iloc[-self.train_window:]
        
        # Drop rows with NaN
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < self.min_train_samples:
            warnings.warn(
                f"Insufficient data for LASSO training: {len(X_clean)} samples "
                f"(need {self.min_train_samples})"
            )
            return False
        
        # Check class balance
        class_counts = y_clean.value_counts()
        if len(class_counts) < 2:
            warnings.warn("Only one class in training data, cannot train")
            return False
        
        min_class_pct = class_counts.min() / len(y_clean)
        if min_class_pct < 0.1:
            warnings.warn(f"Severe class imbalance: minority class is {min_class_pct:.1%}")
        
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Train LASSO logistic regression
            model = LogisticRegression(
                penalty='l1',
                solver='saga',
                C=1.0 / max(self.alpha, 0.001),  # Prevent division by zero
                max_iter=2000,
                random_state=42,
                class_weight='balanced',  # Handle imbalance
            )
            model.fit(X_scaled, y_clean)
            
            # Store model and metadata
            self._model = model
            self._scaler = scaler
            self._trained_feature_cols = feature_cols
            
            # Store training stats
            n_nonzero = np.sum(model.coef_[0] != 0)
            self._training_stats = {
                'n_samples': len(X_clean),
                'n_features': len(feature_cols),
                'n_selected': n_nonzero,
                'class_balance': class_counts.to_dict(),
                'train_score': model.score(X_scaled, y_clean),
            }
            
            return True
            
        except Exception as e:
            warnings.warn(f"LASSO training failed: {e}")
            return False
    
    def _predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime using trained model.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Series with predictions
        """
        if self._model is None or self._trained_feature_cols is None:
            return self.get_labels(df)
        
        # Check all required features exist
        missing = [c for c in self._trained_feature_cols if c not in df.columns]
        if missing:
            warnings.warn(f"Missing features for prediction: {missing}")
            return self.get_labels(df)
        
        X = df[self._trained_feature_cols].copy()
        
        # Identify valid rows (no NaN)
        valid_mask = X.notna().all(axis=1)
        
        # Initialize result with fallback labels
        result = self.get_labels(df).copy()
        
        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            
            try:
                X_scaled = self._scaler.transform(X_valid)
                predictions = self._model.predict(X_scaled)
                result.loc[valid_mask] = predictions
            except Exception as e:
                warnings.warn(f"Prediction failed: {e}")
        
        return result.fillna(0).astype(int)
    
    def train(self, df: pd.DataFrame, feature_cols: List[str] = None) -> bool:
        """
        Manually trigger training.
        
        Args:
            df: DataFrame with features
            feature_cols: Features to use (None = auto-detect)
        
        Returns:
            True if training successful
        """
        if feature_cols is None:
            feature_cols = self._get_feature_columns(df)
        
        labels = self.get_labels(df)
        return self._train_model(df, feature_cols, labels)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from trained LASSO model.
        
        Returns DataFrame with:
            - coefficient: Raw LASSO coefficient
            - abs_coefficient: Absolute value
            - selected: Whether feature was selected (coef != 0)
        """
        if self._model is None or self._trained_feature_cols is None:
            return None
        
        coefs = self._model.coef_[0]
        
        importance = pd.DataFrame({
            'feature': self._trained_feature_cols,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs),
            'selected': coefs != 0,
        })
        
        return importance.sort_values('abs_coefficient', ascending=False).set_index('feature')
    
    def get_selected_features(self) -> Optional[List[str]]:
        """Get list of features selected by LASSO (non-zero coefficients)."""
        importance = self.get_feature_importance()
        if importance is None:
            return None
        
        selected = importance[importance['selected']].index.tolist()
        return selected
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics from last training run."""
        return self._training_stats.copy()
    
    def reset(self):
        """Reset model state (forces retraining on next compute)."""
        self._model = None
        self._scaler = None
        self._trained_feature_cols = None
        self._last_train_idx = 0
        self._training_stats = {}


# =============================================================================
# Markov Switching Model Regime (Tactical)
# =============================================================================

@register_feature
class RegimeMSM(Feature):
    """
    Markov Switching Model regime for tactical/momentum detection.
    
    Detects fast-switching volatility regimes (days, not weeks).
    Uses statsmodels MarkovRegression when available,
    falls back to volatility ratio otherwise.
    
    Output: 0 = calm, 1 = volatile
    Average duration: ~4.2 days
    """
    name = "regime_msm"
    lookback = 504  # 3 weeks minimum for estimation
    output_type = "binary"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.refit_every = params.get('refit_every', 168)  # Refit weekly
        self._model = None
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute MSM regime using Markov Switching Model."""
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
            return self._compute_with_msm(df)
        except ImportError:
            warnings.warn("statsmodels not available for MSM. Using volatility threshold fallback.")
            return self._compute_fallback(df)
    
    def _compute_with_msm(self, df: pd.DataFrame) -> pd.Series:
        """Use statsmodels MarkovRegression."""
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        
        # Use hourly returns
        log_returns = compute_log_returns(df, 'close', periods=1)
        returns_clean = log_returns.dropna() * 100  # Scale for numerical stability
        
        if len(returns_clean) < self.lookback:
            return pd.Series(0, index=df.index)
        
        regime = pd.Series(index=df.index, dtype=float)
        
        # Rolling MSM with periodic refitting
        window = 500
        
        for i in range(window, len(returns_clean), self.refit_every):
            end_idx = min(i + self.refit_every, len(returns_clean))
            
            # Fit on training window
            train_data = returns_clean.iloc[i-window:i]
            
            try:
                model = MarkovRegression(
                    train_data, 
                    k_regimes=2, 
                    switching_variance=True
                )
                fitted = model.fit(disp=False)
                
                # State with higher variance = volatile (1)
                if fitted.params['sigma2[0]'] < fitted.params['sigma2[1]']:
                    volatile_state = 1
                else:
                    volatile_state = 0
                
                # Use smoothed probabilities
                smoothed = fitted.smoothed_marginal_probabilities
                regime_values = (smoothed[volatile_state] > 0.5).astype(int)
                
                # Assign to original index
                regime.iloc[i-window:i] = regime_values.values
                
            except Exception:
                # Model failed, continue with previous values
                continue
        
        # Fill gaps
        regime = regime.ffill().bfill().fillna(0).astype(int)
        return regime
    
    def _compute_fallback(self, df: pd.DataFrame) -> pd.Series:
        """
        Volatility-based fallback for MSM.
        Uses short-term vol vs medium-term vol.
        """
        log_returns = compute_log_returns(df, 'close', periods=1)
        
        # Short-term volatility (24h)
        short_vol = log_returns.rolling(window=24, min_periods=24).std()
        
        # Medium-term volatility (72h)
        med_vol = log_returns.rolling(window=72, min_periods=72).std()
        
        # Volatile when short-term > medium-term (vol expansion)
        regime = (short_vol > med_vol).astype(int)
        
        return regime.fillna(0).astype(int)


# =============================================================================
# Hybrid Regime (Structural + Tactical)
# =============================================================================

@register_feature
class RegimeHybrid(Feature):
    """
    Combined Structural (LASSO) + Tactical (MSM) regime.
    
    Creates 4-state regime:
        0 = Bear/Calm: Structural bear, tactical calm
        1 = Bear/Volatile: Structural bear, tactical volatile
        2 = Bull/Calm: Structural bull, tactical calm
        3 = Bull/Volatile: Structural bull, tactical volatile
    
    Strategy implications:
        State 0: Reduce exposure, wait for transition
        State 1: Potential capitulation or bounce
        State 2: Steady accumulation zone
        State 3: Strong momentum, full position
    """
    name = "regime_hybrid"
    lookback = 504
    output_type = "discrete"
    dependencies = ["regime_structural", "regime_msm"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute hybrid 4-state regime."""
        # Check if components already in df
        if 'regime_structural' in df.columns:
            structural = df['regime_structural']
        else:
            structural_feature = RegimeLASSO()
            structural = structural_feature.compute(df)
        
        if 'regime_msm' in df.columns:
            msm = df['regime_msm']
        else:
            msm_feature = RegimeMSM()
            msm = msm_feature.compute(df)
        
        # Combine: state = structural * 2 + msm
        # structural=0 (bear), msm=0 (calm) -> 0
        # structural=0 (bear), msm=1 (volatile) -> 1
        # structural=1 (bull), msm=0 (calm) -> 2
        # structural=1 (bull), msm=1 (volatile) -> 3
        hybrid = structural * 2 + msm
        
        return hybrid.fillna(0).astype(int)


# =============================================================================
# Regime Multiplier
# =============================================================================

@register_feature
class RegimeMultiplier(Feature):
    """
    Position size multiplier based on hybrid regime.
    
    Default multipliers (configurable):
        State 0 (Bear/Calm): 0.3 - Minimal exposure
        State 1 (Bear/Volatile): 0.5 - Small position for bounces
        State 2 (Bull/Calm): 0.8 - Solid position
        State 3 (Bull/Volatile): 1.0 - Full position
    """
    name = "regime_multiplier"
    lookback = 504
    output_type = "continuous"
    dependencies = ["regime_hybrid"]
    
    def __init__(self, **params):
        super().__init__(**params)
        self.multipliers = params.get('multipliers', {
            0: 0.3,  # Bear/Calm
            1: 0.5,  # Bear/Volatile
            2: 0.8,  # Bull/Calm
            3: 1.0,  # Bull/Volatile
        })
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute position multiplier from regime."""
        if 'regime_hybrid' in df.columns:
            regime = df['regime_hybrid']
        else:
            hybrid_feature = RegimeHybrid()
            regime = hybrid_feature.compute(df)
        
        multiplier = regime.map(self.multipliers)
        return multiplier.fillna(0.5)


# =============================================================================
# MA Alignment Features (for regime labeling)
# =============================================================================

@register_feature
class MAAlignment(Feature):
    """
    MA Alignment score for regime labeling.
    
    Counts how many MAs are rising:
        0 = All falling (strong bear)
        1 = 1 rising (weak bear)
        2 = 2 rising (weak bull)
        3 = All rising (strong bull)
    """
    name = "ma_alignment"
    lookback = 192  # 168h MA + 24h for slope
    output_type = "discrete"
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        
        # Compute MAs
        ma_24 = compute_sma(close, 24)
        ma_72 = compute_sma(close, 72)
        ma_168 = compute_sma(close, 168)
        
        # Compute slopes (24h lookback)
        slope_24 = ma_24.diff(24) / ma_24.shift(24)
        slope_72 = ma_72.diff(24) / ma_72.shift(24)
        slope_168 = ma_168.diff(24) / ma_168.shift(24)
        
        # Count rising MAs
        alignment = (
            (slope_24 > 0).astype(int) +
            (slope_72 > 0).astype(int) +
            (slope_168 > 0).astype(int)
        )
        
        return alignment


# =============================================================================
# Feature Groups
# =============================================================================

register_group(FeatureGroup(
    name="regime",
    features=['regime_structural', 'regime_msm', 'regime_hybrid'],
    description="Core regime detection features"
))

register_group(FeatureGroup(
    name="regime_full",
    features=['regime_structural', 'regime_msm', 'regime_hybrid', 
              'regime_multiplier', 'ma_alignment'],
    description="Complete regime feature set with multipliers"
))