"""
tif.derivatives.hedge_ratio
=============================
Hedge ratio optimization: minimum variance, GARCH-dynamic, and ML-optimized.

Reference
---------
Siddiqui, I. (2025). An Integrated Derivative Hedging Framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import statsmodels.api as sm
    _SM_OK = True
except ImportError:
    _SM_OK = False

try:
    from arch import arch_model
    _ARCH_OK = True
except ImportError:
    _ARCH_OK = False

try:
    from xgboost import XGBRegressor
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

try:
    from scipy.optimize import minimize
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


class HedgeMethod(Enum):
    MINIMUM_VARIANCE = "minimum_variance"
    GARCH_DYNAMIC    = "garch_dynamic"
    ML_OPTIMIZED     = "ml_optimized"


@dataclass
class HedgeRatioResult:
    method:       HedgeMethod
    ratio:        float           # optimal hedge ratio (0–1)
    r_squared:    float           # regression R² (quality of hedge)
    residual_var: float           # residual variance after hedging
    series:       pd.Series       # rolling ratio time series


class HedgeRatioOptimizer:
    """
    Compute optimal hedge ratios using three methodologies.

    Parameters
    ----------
    window : int
        Rolling estimation window in observations (default: 60).

    Examples
    --------
    >>> opt = HedgeRatioOptimizer(window=60)
    >>> result = opt.optimize(spot_returns, hedge_returns, HedgeMethod.GARCH_DYNAMIC)
    """

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._ml_model: Optional[XGBRegressor] = None

    # ── Minimum Variance OLS ──────────────────────────────────────
    def minimum_variance(
        self,
        spot_returns: pd.Series,
        hedge_returns: pd.Series,
    ) -> HedgeRatioResult:
        """Johnson (1960) rolling minimum variance hedge ratio."""
        if not _SM_OK:
            raise ImportError("Install: statsmodels")
        ratios = []
        for i in range(self.window, len(spot_returns)):
            y = spot_returns.iloc[i - self.window:i]
            X = sm.add_constant(hedge_returns.iloc[i - self.window:i])
            ols = sm.OLS(y, X).fit()
            ratios.append(float(ols.params.iloc[1]))

        series = pd.Series(ratios, index=spot_returns.index[self.window:])
        ratio  = float(series.mean())

        # Full-sample R² and residual variance
        X_full = sm.add_constant(hedge_returns)
        ols_full = sm.OLS(spot_returns, X_full).fit()
        hedged   = spot_returns - ratio * hedge_returns
        return HedgeRatioResult(
            method       = HedgeMethod.MINIMUM_VARIANCE,
            ratio        = round(ratio, 4),
            r_squared    = round(float(ols_full.rsquared), 4),
            residual_var = round(float(hedged.var()), 6),
            series       = series,
        )

    # ── GARCH Dynamic ─────────────────────────────────────────────
    def garch_dynamic(
        self,
        spot_returns: pd.Series,
        hedge_returns: pd.Series,
    ) -> HedgeRatioResult:
        """Time-varying hedge ratio with GARCH(1,1) volatility weighting."""
        if not (_SM_OK and _ARCH_OK):
            raise ImportError("Install: statsmodels arch")
        ratios = []
        for i in range(self.window, len(spot_returns)):
            s = spot_returns.iloc[i - self.window:i]
            h = hedge_returns.iloc[i - self.window:i]
            try:
                gm  = arch_model(h * 100, vol="Garch", p=1, q=1, rescale=False)
                res = gm.fit(disp="off", show_warning=False)
                cond_vol = res.conditional_volatility / 100
                weights  = 1.0 / (cond_vol.clip(lower=1e-6))
                X        = sm.add_constant(h)
                wls      = sm.WLS(s, X, weights=weights).fit()
                ratios.append(float(wls.params.iloc[1]))
            except Exception:
                ratios.append(np.nan)

        series = pd.Series(ratios, index=spot_returns.index[self.window:]).fillna(method="ffill")
        ratio  = float(series.mean())
        hedged = spot_returns - ratio * hedge_returns
        return HedgeRatioResult(
            method       = HedgeMethod.GARCH_DYNAMIC,
            ratio        = round(ratio, 4),
            r_squared    = round(float(np.corrcoef(spot_returns, hedge_returns)[0,1]**2), 4),
            residual_var = round(float(hedged.var()), 6),
            series       = series,
        )

    # ── ML-Optimized ──────────────────────────────────────────────
    def fit_ml_model(
        self,
        features: pd.DataFrame,
        realized_ratios: pd.Series,
    ) -> "HedgeRatioOptimizer":
        """Train gradient boosting model on realized optimal hedge ratios."""
        if not _XGB_OK:
            raise ImportError("Install: xgboost")
        self._ml_model = XGBRegressor(
            n_estimators=300, max_depth=3,
            learning_rate=0.04, subsample=0.85,
            random_state=42, verbosity=0,
        )
        idx = features.index.intersection(realized_ratios.index)
        self._ml_model.fit(features.loc[idx], realized_ratios.loc[idx])
        return self

    def ml_predict(self, features: pd.DataFrame) -> float:
        """Predict optimal hedge ratio from current market features."""
        if self._ml_model is None:
            raise RuntimeError("Call fit_ml_model() first.")
        return float(np.clip(self._ml_model.predict(features.iloc[[-1]])[0], 0, 1))

    # ── Multi-instrument allocation ───────────────────────────────
    def multi_instrument_allocation(
        self,
        exposure: float,
        instruments: List[str],
        cov_matrix: np.ndarray,
        cost_bps:   np.ndarray,
        max_cost_bps: float = 30.0,
    ) -> Dict[str, float]:
        """
        Minimum variance allocation across multiple hedging instruments
        subject to a transaction cost budget.

        Parameters
        ----------
        exposure      : total notional to hedge (USD millions)
        instruments   : list of instrument names
        cov_matrix    : covariance matrix of instrument returns
        cost_bps      : per-instrument transaction cost in bps
        max_cost_bps  : total portfolio cost budget in bps

        Returns
        -------
        dict {instrument: allocated_notional}
        """
        if not _SCIPY_OK:
            raise ImportError("Install: scipy")
        n = len(instruments)

        def objective(w): return float(w @ cov_matrix @ w)

        constraints = [
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: max_cost_bps - float(w @ cost_bps)},
        ]
        res = minimize(
            objective, x0=np.ones(n) / n,
            method="SLSQP", bounds=[(0, 1)] * n, constraints=constraints,
        )
        weights = res.x if res.success else np.ones(n) / n
        return {inst: round(w * exposure, 2) for inst, w in zip(instruments, weights)}

    def optimize(
        self,
        spot_returns:  pd.Series,
        hedge_returns: pd.Series,
        method: HedgeMethod = HedgeMethod.MINIMUM_VARIANCE,
    ) -> HedgeRatioResult:
        """Dispatch to the selected optimization method."""
        if method == HedgeMethod.MINIMUM_VARIANCE:
            return self.minimum_variance(spot_returns, hedge_returns)
        elif method == HedgeMethod.GARCH_DYNAMIC:
            return self.garch_dynamic(spot_returns, hedge_returns)
        else:
            raise ValueError("For ML_OPTIMIZED use fit_ml_model() + ml_predict() directly.")
