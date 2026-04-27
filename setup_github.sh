"""
tif.liquidity.deposit_outflow_model
====================================
Behavioral deposit outflow modeling using gradient boosting (XGBoost).

Replaces static Basel III outflow rate tables with institution-specific,
data-derived behavioral assumptions calibrated to actual deposit history,
macroeconomic indicators, and market stress proxies.

Reference
---------
Siddiqui, I. (2025). Predictive LCR Modeling for Regional Banks:
A Framework for Forward-Looking Liquidity Governance.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_percentage_error
    import shap
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


# ── Basel III standard outflow rate table (fallback) ─────────────────────────
BASEL_OUTFLOW_RATES: Dict[str, float] = {
    "retail_stable":       0.05,
    "retail_less_stable":  0.10,
    "sme_stable":          0.05,
    "sme_less_stable":     0.10,
    "corporate_operating": 0.25,
    "corporate_other":     0.40,
    "financial_other":     1.00,
    "central_bank":        0.00,
}


@dataclass
class DepositSegment:
    """Single deposit segment definition."""
    name: str
    balance: float                # USD millions
    category: str                 # maps to BASEL_OUTFLOW_RATES key
    insured_fraction: float = 1.0 # fraction covered by deposit insurance
    tenor_days: int = 0           # 0 = on-demand


@dataclass
class OutflowForecast:
    """Output of the deposit outflow model."""
    segment: str
    balance: float
    predicted_outflow_rate: float   # decimal, e.g. 0.12 = 12%
    basel_outflow_rate: float       # static Basel III benchmark
    predicted_outflow_usd: float    # USD millions
    confidence_lower: float         # 10th percentile
    confidence_upper: float         # 90th percentile
    shap_top_features: Optional[Dict[str, float]] = None


def build_deposit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for deposit behavioral modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date, net_outflow, balance, top10_balance,
        fed_funds_rate, vix, hy_spread, ig_spread.

    Returns
    -------
    pd.DataFrame with engineered features ready for model training.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Rolling behavioral signals ────────────────────────────────
    df["outflow_7d_ma"]    = df["net_outflow"].rolling(7,  min_periods=1).mean()
    df["outflow_30d_ma"]   = df["net_outflow"].rolling(30, min_periods=1).mean()
    df["outflow_30d_std"]  = df["net_outflow"].rolling(30, min_periods=1).std().fillna(0)
    df["outflow_90d_std"]  = df["net_outflow"].rolling(90, min_periods=1).std().fillna(0)

    # ── Concentration metrics ─────────────────────────────────────
    df["concentration"]       = df["top10_balance"] / df["balance"].replace(0, np.nan)
    df["balance_growth_30d"]  = df["balance"].pct_change(30).fillna(0)
    df["balance_growth_90d"]  = df["balance"].pct_change(90).fillna(0)

    # ── Macro stress proxies ──────────────────────────────────────
    df["rate_shock_30d"]  = df["fed_funds_rate"].diff(30).fillna(0)
    df["rate_shock_90d"]  = df["fed_funds_rate"].diff(90).fillna(0)
    df["vix_30d_avg"]     = df["vix"].rolling(30, min_periods=1).mean()
    df["vix_spike"]       = (df["vix"] > 30).astype(int)
    df["credit_spread"]   = df["hy_spread"] - df["ig_spread"]
    df["spread_change_30"]= df["credit_spread"].diff(30).fillna(0)

    # ── Carry & momentum ─────────────────────────────────────────
    df["carry"] = df["rate_shock_30d"] / df["vix_30d_avg"].replace(0, np.nan)

    # ── Lag features ──────────────────────────────────────────────
    for lag in [1, 3, 7, 14, 30]:
        df[f"outflow_lag_{lag}d"] = df["net_outflow"].shift(lag).fillna(0)

    return df.dropna(subset=["outflow_30d_std"])


class DepositOutflowModel:
    """
    Machine learning model for predicting deposit outflow rates.

    Uses XGBoost gradient boosting trained on institution-specific
    deposit transaction history and macroeconomic features.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (default: 400).
    learning_rate : float
        XGBoost learning rate (default: 0.03).
    cv_folds : int
        Number of time-series cross-validation folds (default: 5).

    Examples
    --------
    >>> model = DepositOutflowModel()
    >>> model.fit(features_df, outflow_rate_series)
    >>> forecasts = model.predict(segments, new_features_df)
    """

    FEATURE_COLS = [
        "outflow_7d_ma", "outflow_30d_ma", "outflow_30d_std", "outflow_90d_std",
        "concentration", "balance_growth_30d", "balance_growth_90d",
        "rate_shock_30d", "rate_shock_90d", "vix_30d_avg", "vix_spike",
        "credit_spread", "spread_change_30", "carry",
        "outflow_lag_1d", "outflow_lag_3d", "outflow_lag_7d",
        "outflow_lag_14d", "outflow_lag_30d",
    ]

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.03,
        cv_folds: int = 5,
    ) -> None:
        if not _DEPS_OK:
            raise ImportError("Install: xgboost scikit-learn shap")
        self.n_estimators   = n_estimators
        self.learning_rate  = learning_rate
        self.cv_folds       = cv_folds
        self._model: Optional[XGBRegressor] = None
        self._scaler        = StandardScaler()
        self._is_fitted     = False
        self.cv_scores_: List[float] = []

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        verbose: bool = False,
    ) -> "DepositOutflowModel":
        """Fit the outflow model with time-series cross-validation."""
        X = features[self.FEATURE_COLS].copy()
        y = target.copy()
        X_scaled = self._scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        for fold, (tr, va) in enumerate(tscv.split(X_scaled), 1):
            m = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
            )
            m.fit(X_scaled[tr], y.iloc[tr])
            preds = m.predict(X_scaled[va])
            mape = mean_absolute_percentage_error(y.iloc[va], preds)
            self.cv_scores_.append(mape)
            if verbose:
                print(f"  Fold {fold}: MAPE = {mape:.3f}")

        # Final model on all data
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict(
        self,
        segments: List[DepositSegment],
        features: pd.DataFrame,
        n_bootstrap: int = 200,
    ) -> List[OutflowForecast]:
        """
        Generate outflow forecasts for each deposit segment.

        Parameters
        ----------
        segments : list of DepositSegment
        features : pd.DataFrame — latest feature row(s)
        n_bootstrap : int — bootstrap samples for confidence intervals

        Returns
        -------
        list of OutflowForecast
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X = features[self.FEATURE_COLS].iloc[[-1]]
        X_scaled = self._scaler.transform(X)

        # Bootstrap confidence intervals by perturbing inputs
        boot_preds = []
        rng = np.random.default_rng(0)
        for _ in range(n_bootstrap):
            noise = rng.normal(0, 0.01, X_scaled.shape)
            pred  = float(self._model.predict(X_scaled + noise)[0])
            boot_preds.append(np.clip(pred, 0, 1))
        base_pred = float(self._model.predict(X_scaled)[0])
        base_pred = np.clip(base_pred, 0, 1)
        ci_lo     = np.percentile(boot_preds, 10)
        ci_hi     = np.percentile(boot_preds, 90)

        # SHAP feature importance for top row
        explainer   = shap.TreeExplainer(self._model)
        shap_vals   = explainer.shap_values(X_scaled)
        shap_map    = dict(zip(self.FEATURE_COLS, shap_vals[0]))
        top_feats   = dict(sorted(shap_map.items(), key=lambda x: abs(x[1]),
                                  reverse=True)[:5])

        forecasts = []
        for seg in segments:
            # Adjust base rate by insurance status (insured deposits less likely to run)
            adj_rate    = base_pred * (1 - seg.insured_fraction * 0.6)
            adj_rate    = np.clip(adj_rate, 0, 1)
            basel_rate  = BASEL_OUTFLOW_RATES.get(seg.category, 0.40)
            forecasts.append(OutflowForecast(
                segment              = seg.name,
                balance              = seg.balance,
                predicted_outflow_rate = round(adj_rate, 4),
                basel_outflow_rate   = basel_rate,
                predicted_outflow_usd= round(seg.balance * adj_rate, 2),
                confidence_lower     = round(seg.balance * ci_lo, 2),
                confidence_upper     = round(seg.balance * ci_hi, 2),
                shap_top_features    = top_feats,
            ))
        return forecasts

    def summary(self) -> pd.DataFrame:
        """Return cross-validation performance summary."""
        return pd.DataFrame({
            "fold": range(1, len(self.cv_scores_) + 1),
            "mape": self.cv_scores_,
        })
