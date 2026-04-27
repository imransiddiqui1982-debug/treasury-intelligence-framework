"""
tif.curves.sofr_curve
======================
SOFR OIS swap curve bootstrapping with cubic spline interpolation.

Constructs a smooth, arbitrage-free discount curve from traded SOFR
OIS swap rates, producing discount factors, zero rates, and forward
rates at arbitrary tenors.

Reference
---------
Siddiqui, I. (2025). Constructing IRS and Cross-Currency Basis Curves
for Middle East Financial Institutions.
Hagan & West (2006). Interpolation methods for curve construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


@dataclass
class SwapRate:
    """A single SOFR OIS swap rate input."""
    tenor_years: float
    rate:        float    # annualised decimal, e.g. 0.0525 = 5.25%
    frequency:   int = 2  # 1=annual, 2=semi-annual, 4=quarterly

    @classmethod
    def from_percent(cls, tenor: float, rate_pct: float, freq: int = 2) -> "SwapRate":
        return cls(tenor_years=tenor, rate=rate_pct / 100, frequency=freq)


# Indicative SOFR swap rates (April 2025 approximate market levels)
DEFAULT_SOFR_RATES: List[SwapRate] = [
    SwapRate(0.0833, 0.0530),   # 1M
    SwapRate(0.2500, 0.0528),   # 3M
    SwapRate(0.5000, 0.0522),   # 6M
    SwapRate(1.0000, 0.0505),   # 1Y
    SwapRate(2.0000, 0.0482),   # 2Y
    SwapRate(3.0000, 0.0468),   # 3Y
    SwapRate(5.0000, 0.0455),   # 5Y
    SwapRate(7.0000, 0.0448),   # 7Y
    SwapRate(10.000, 0.0442),   # 10Y
    SwapRate(15.000, 0.0438),   # 15Y
    SwapRate(20.000, 0.0435),   # 20Y
    SwapRate(30.000, 0.0430),   # 30Y
]


class SOFRCurve:
    """
    Bootstrapped SOFR OIS swap curve.

    Iteratively solves for discount factors at each swap tenor,
    ensuring the curve exactly reprices all input swap rates.
    Cubic spline interpolation provides smooth zero and forward
    rates at arbitrary intermediate tenors.

    Parameters
    ----------
    swap_rates : list of SwapRate
        Input SOFR OIS swap rates. If None, uses DEFAULT_SOFR_RATES.
    interpolation : str
        Interpolation method: "cubic_spline" or "pchip" (monotone).

    Examples
    --------
    >>> curve = SOFRCurve()
    >>> df_5y = curve.discount(5.0)          # 0.7998...
    >>> z_10y = curve.zero_rate(10.0)        # 0.0442...
    >>> fwd   = curve.forward_rate(5.0, 6.0) # 1-year forward in 5 years
    >>> par   = curve.par_swap_rate(7.0)     # par 7-year SOFR swap rate
    """

    def __init__(
        self,
        swap_rates:    Optional[List[SwapRate]] = None,
        interpolation: str = "cubic_spline",
    ) -> None:
        if not _SCIPY_OK:
            raise ImportError("Install: scipy")
        self._rates        = swap_rates or DEFAULT_SOFR_RATES
        self._interp_type  = interpolation
        self._df: Dict[float, float] = {0.0: 1.0}
        self._bootstrap()
        self._build_spline()

    # ── Bootstrapping ─────────────────────────────────────────────
    def _interpolate_known(self, t: float) -> float:
        """Interpolate log-discount factor using currently known pillars."""
        known_t  = sorted(self._df.keys())
        known_df = [self._df[tt] for tt in known_t]
        if len(known_t) < 2:
            return 1.0
        log_df = np.log(known_df)
        cs     = CubicSpline(known_t, log_df, extrapolate=True)
        return float(np.exp(cs(t)))

    def _bootstrap(self) -> None:
        """
        Iterative bootstrapping: for each swap tenor, solve for the
        terminal discount factor that prices the swap at par.
        """
        for sw in sorted(self._rates, key=lambda s: s.tenor_years):
            T   = sw.tenor_years
            freq = sw.frequency
            dt   = 1.0 / freq
            cpn  = sw.rate / freq

            # Sum of coupon PVs using already-bootstrapped + interpolated DFs
            times = np.arange(dt, T - dt / 2, dt)
            pv_coupons = sum(cpn * self._interpolate_known(t) for t in times)

            # Terminal DF: solve swap_PV = 0
            # fixed_leg = cpn * annuity + 1 * df(T) = 1 (par condition)
            df_T = (1.0 - pv_coupons) / (1.0 + cpn)
            self._df[T] = max(df_T, 1e-9)

    def _build_spline(self) -> None:
        """Fit log-linear spline over all bootstrapped discount factors."""
        t_arr  = np.array(sorted(self._df.keys()))
        df_arr = np.array([self._df[t] for t in t_arr])
        log_df = np.log(df_arr)
        if self._interp_type == "pchip":
            self._spline = PchipInterpolator(t_arr, log_df, extrapolate=True)
        else:
            self._spline = CubicSpline(t_arr, log_df, extrapolate=True)

    # ── Public API ────────────────────────────────────────────────
    def discount(self, t: float) -> float:
        """Discount factor P(0, t) at tenor t years."""
        if t <= 0:
            return 1.0
        if t in self._df:
            return self._df[t]
        return float(np.exp(self._spline(t)))

    def zero_rate(self, t: float, compounding: str = "continuous") -> float:
        """
        Zero rate at tenor t.

        Parameters
        ----------
        compounding : str
            "continuous" (default) or "annual" or "semi_annual".
        """
        if t <= 0:
            return 0.0
        df = self.discount(t)
        z  = -np.log(df) / t
        if compounding == "annual":
            return np.exp(z) - 1
        elif compounding == "semi_annual":
            return 2 * (np.exp(z / 2) - 1)
        return float(z)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between t1 and t2."""
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        df1 = self.discount(t1)
        df2 = self.discount(t2)
        return float(np.log(df1 / df2) / (t2 - t1))

    def par_swap_rate(self, tenor: float, freq: int = 2) -> float:
        """
        Par (fair) swap rate for given tenor.
        This is the fixed rate at which a new swap has zero fair value.
        """
        dt    = 1.0 / freq
        times = np.arange(dt, tenor + dt / 2, dt)
        if len(times) == 0:
            times = np.array([tenor])
        annuity = sum(dt * self.discount(t) for t in times)
        if annuity < 1e-12:
            return self.zero_rate(tenor)
        return float((1.0 - self.discount(tenor)) / annuity)

    def zero_curve_df(
        self,
        tenors: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Return full zero curve as a DataFrame."""
        tenors = tenors or [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        return pd.DataFrame({
            "Tenor (Y)":        tenors,
            "Discount Factor":  [round(self.discount(t), 6) for t in tenors],
            "Zero Rate (%)":    [round(self.zero_rate(t) * 100, 4) for t in tenors],
            "Par Swap Rate (%)": [round(self.par_swap_rate(t) * 100, 4) for t in tenors],
        })

    def bump(self, shift_bps: float) -> "SOFRCurve":
        """Return a new SOFRCurve with all rates shifted by shift_bps."""
        shift = shift_bps / 10_000
        bumped = [
            SwapRate(s.tenor_years, s.rate + shift, s.frequency)
            for s in self._rates
        ]
        return SOFRCurve(bumped, self._interp_type)

    def dv01(self, notional: float, tenor: float, freq: int = 2) -> float:
        """
        DV01 of a par swap (1bp parallel shift).
        Positive = receiver (long duration).
        """
        par_base   = self.par_swap_rate(tenor, freq)
        curve_up   = self.bump(+1)
        par_up     = curve_up.par_swap_rate(tenor, freq)
        dt         = 1.0 / freq
        times      = np.arange(dt, tenor + dt / 2, dt)
        annuity    = sum(dt * self.discount(t) for t in times)
        return float(-notional * annuity * (par_up - par_base))

    def __repr__(self) -> str:
        n = len(self._rates)
        r_min = min(s.rate for s in self._rates) * 100
        r_max = max(s.rate for s in self._rates) * 100
        return f"SOFRCurve(pillars={n}, range=[{r_min:.2f}%, {r_max:.2f}%])"
