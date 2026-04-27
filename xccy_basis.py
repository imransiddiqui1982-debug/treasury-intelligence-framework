"""
tif.derivatives.fx_exposure_forecast
======================================
FX exposure forecasting using XGBoost + LSTM ensemble.

Predicts future net foreign currency cash flow exposure for
corporate and banking book hedging program design.

Reference
---------
Siddiqui, I. (2025). An Integrated Derivative Hedging Framework
for U.S. Corporates and Regional Banks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from xgboost import XGBRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_percentage_error
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    _TF_OK = True
except ImportError:
    _TF_OK = False


@dataclass
class ExposureConfig:
    """Configuration for FX exposure forecasting."""
    currency_pairs:       List[str]       # e.g. ["USD/EUR", "USD/GBP"]
    horizon_months:       int = 12
    lookback_months:      int = 36
    lstm_units:           int = 64
    lstm_seq_len:         int = 12
    ensemble_weight_xgb:  float = 0.55
    ensemble_weight_lstm: float = 0.45

    def __post_init__(self):
        total = self.ensemble_weight_xgb + self.ensemble_weight_lstm
        assert abs(total - 1.0) < 1e-6, "Ensemble weights must sum to 1.0"


def build_fx_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for FX exposure forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Columns required: date, net_fx_exposure, fx_rate,
        domestic_rate, foreign_rate, domestic_cpi, foreign_cpi,
        trade_balance, hy_spread, ig_spread, vix.

    Returns
    -------
    pd.DataFrame with engineered features.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Lagged exposure signals
    for lag in [1, 3, 6, 12]:
        df[f"exposure_lag_{lag}m"] = df["net_fx_exposure"].shift(lag)

    # FX rate dynamics
    df["fx_returns"]   = df["fx_rate"].pct_change()
    df["fx_vol_30d"]   = df["fx_returns"].rolling(30, min_periods=5).std() * np.sqrt(252)
    df["fx_vol_90d"]   = df["fx_returns"].rolling(90, min_periods=20).std() * np.sqrt(252)
    df["vol_ratio"]    = df["fx_vol_30d"] / df["fx_vol_90d"].replace(0, np.nan)
    df["fx_momentum"]  = df["fx_rate"].pct_change(3)

    # Macro fundamentals
    df["rate_differential"] = df["domestic_rate"] - df["foreign_rate"]
    df["cpi_differential"]  = df["domestic_cpi"]  - df["foreign_cpi"]
    df["trade_balance_yoy"] = df["trade_balance"].pct_change(12)
    df["carry"]             = df["rate_differential"] / df["fx_vol_30d"].replace(0, np.nan)

    # Risk sentiment
    df["credit_spread"]     = df["hy_spread"] - df["ig_spread"]
    df["spread_change_30"]  = df["credit_spread"].diff(30)
    df["vix_30d_avg"]       = df["vix"].rolling(30, min_periods=5).mean()
    df["vix_regime"]        = (df["vix"] > 25).astype(int)

    return df.dropna()


class FXExposureForecaster:
    """
    Ensemble FX exposure forecaster (XGBoost + LSTM).

    Predicts net FX exposure over a multi-month horizon for use in
    hedge notional sizing and hedge ratio optimization.

    Parameters
    ----------
    config : ExposureConfig

    Examples
    --------
    >>> cfg = ExposureConfig(currency_pairs=["USD/EUR"])
    >>> model = FXExposureForecaster(cfg)
    >>> model.fit(features_df, target_series)
    >>> forecast = model.predict(latest_features)
    """

    FEATURE_COLS = [
        "exposure_lag_1m", "exposure_lag_3m", "exposure_lag_6m",
        "fx_vol_30d", "fx_vol_90d", "vol_ratio", "fx_momentum",
        "rate_differential", "cpi_differential", "trade_balance_yoy",
        "carry", "credit_spread", "spread_change_30", "vix_30d_avg", "vix_regime",
    ]

    def __init__(self, config: ExposureConfig) -> None:
        if not _XGB_OK:
            raise ImportError("Install: xgboost scikit-learn")
        self.config    = config
        self._xgb: Optional[XGBRegressor] = None
        self._lstm     = None
        self._scaler_X = MinMaxScaler()
        self._scaler_y = MinMaxScaler()
        self._is_fitted = False

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        verbose: bool = False,
    ) -> "FXExposureForecaster":
        """Fit XGBoost and (optionally) LSTM models."""
        X = features[self.FEATURE_COLS].copy().values
        y = target.values.reshape(-1, 1)
        X_sc = self._scaler_X.fit_transform(X)
        y_sc = self._scaler_y.fit_transform(y).ravel()

        # XGBoost
        self._xgb = XGBRegressor(
            n_estimators=500, max_depth=4,
            learning_rate=0.025, subsample=0.8,
            colsample_bytree=0.75, reg_alpha=0.1,
            objective="reg:squarederror", random_state=42, verbosity=0,
        )
        self._xgb.fit(X_sc, y_sc)

        # LSTM (optional)
        if _TF_OK and len(X_sc) >= self.config.lstm_seq_len + 10:
            seq_len = self.config.lstm_seq_len
            Xs, ys  = [], []
            for i in range(seq_len, len(X_sc)):
                Xs.append(X_sc[i-seq_len:i])
                ys.append(y_sc[i])
            Xs, ys = np.array(Xs), np.array(ys)
            mdl = Sequential([
                LSTM(self.config.lstm_units, return_sequences=True,
                     input_shape=(seq_len, X_sc.shape[1])),
                Dropout(0.2),
                LSTM(self.config.lstm_units // 2),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ])
            mdl.compile(optimizer="adam", loss="huber")
            mdl.fit(
                Xs, ys, epochs=50, batch_size=16, verbose=0,
                callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
                validation_split=0.15,
            )
            self._lstm = mdl
            self._X_last_seq = X_sc[-seq_len:]

        self._is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Generate point forecast and confidence bounds.

        Returns
        -------
        dict with keys: xgb_forecast, lstm_forecast, ensemble_forecast,
                        ci_lower, ci_upper (all in original units).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        X   = features[self.FEATURE_COLS].iloc[[-1]].values
        Xsc = self._scaler_X.transform(X)

        xgb_sc = float(self._xgb.predict(Xsc)[0])
        xgb_val = float(self._scaler_y.inverse_transform([[xgb_sc]])[0, 0])

        lstm_val = xgb_val  # fallback
        if self._lstm is not None and _TF_OK:
            seq = self._X_last_seq.reshape(1, *self._X_last_seq.shape)
            lstm_sc  = float(self._lstm.predict(seq, verbose=0)[0, 0])
            lstm_val = float(self._scaler_y.inverse_transform([[lstm_sc]])[0, 0])

        ensemble = (
            self.config.ensemble_weight_xgb  * xgb_val +
            self.config.ensemble_weight_lstm * lstm_val
        )

        # Uncertainty: ±15% of ensemble as a simple proxy
        ci_lo = ensemble * 0.85
        ci_hi = ensemble * 1.15

        return {
            "xgb_forecast":      round(xgb_val,  2),
            "lstm_forecast":     round(lstm_val, 2),
            "ensemble_forecast": round(ensemble, 2),
            "ci_lower":          round(ci_lo,    2),
            "ci_upper":          round(ci_hi,    2),
        }
