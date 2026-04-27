"""
tif.curves.gcc_irs_curve
==========================
GCC institution-specific IRS curve construction and swap pricing.

Assembles SOFR base + sovereign spread + banking sector premium +
liquidity premium + XCCY basis into a full institution IRS curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .sofr_curve import SOFRCurve
from .sovereign_spreads import SovereignSpreadCurve, SOVEREIGN_DATA

try:
    from scipy.interpolate import CubicSpline
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


@dataclass
class GCCInstitution:
    """Profile of a GCC financial institution."""
    name:                  str
    country_iso3:          str     # "SAU", "ARE", "OMN", "BHR", etc.
    tier:                  int     # 1=national champion, 2=mid-tier, 3=smaller
    total_car:             float   # total capital adequacy ratio, e.g. 0.178
    npl_ratio:             float   # non-performing loan ratio, e.g. 0.025
    wholesale_funding_pct: float   # fraction of wholesale funding, e.g. 0.30
    loan_deposit_ratio:    float = 0.80

    def label(self) -> str:
        tier_labels = {1: "Tier 1 (National Champion)",
                       2: "Tier 2 (Mid-Tier)", 3: "Tier 3 (Smaller)"}
        return f"{self.name} — {self.country_iso3} — {tier_labels.get(self.tier, '')}"


# Tenor structure multipliers for BSP (BSP widens with tenor)
TENOR_MULT: Dict[float, float] = {
    0.5: 0.50, 1.0: 0.65, 2.0: 0.80, 3.0: 0.90,
    5.0: 1.00, 7.0: 1.20, 10.0: 1.45, 15.0: 1.68, 20.0: 1.85, 30.0: 2.00,
}

# Tier base BSP (bps) — calibrated to GCC subordinated bank bond spreads
TIER_BASE_BSP: Dict[int, float] = {1: 28.0, 2: 65.0, 3: 125.0}


@dataclass
class IRSPrice:
    """Output of GCC IRS pricing."""
    notional:        float
    tenor:           float
    pay_fixed:       bool
    par_swap_rate:   float   # all-in par rate (decimal)
    fair_value:      float   # USD millions
    dv01:            float   # USD per 1bp shift
    spread_breakdown: Dict[str, float]  # decomposition in bps


class GCCIRSCurve:
    """
    Institution-specific IRS curve for a GCC financial institution.

    Constructs the curve from SOFR base + sovereign + BSP + liquidity + basis,
    and prices vanilla interest rate swaps against the resulting curve.

    Parameters
    ----------
    institution : GCCInstitution
    sofr_curve  : SOFRCurve — pre-built SOFR base curve
    xccy_basis_bps : dict — {tenor: basis_bps} XCCY basis adjustment
    liquidity_premium_bps : float — illiquidity premium (default 15bps)

    Examples
    --------
    >>> soa = SOFRCurve()
    >>> inst = GCCInstitution("Bank Muscat", "OMN", tier=1, ...)
    >>> gcc_curve = GCCIRSCurve(inst, sofr_curve)
    >>> gcc_curve.build()
    >>> price = gcc_curve.price_irs(500e6, 5.0, pay_fixed=True)
    """

    STANDARD_TENORS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

    def __init__(
        self,
        institution:           GCCInstitution,
        sofr_curve:            SOFRCurve,
        xccy_basis_bps:        Optional[Dict[float, float]] = None,
        liquidity_premium_bps: float = 15.0,
    ) -> None:
        if not _SCIPY_OK:
            raise ImportError("Install: scipy")
        self.inst      = institution
        self.sofr      = sofr_curve
        self.xccy_bps  = xccy_basis_bps or {t: 0.0 for t in self.STANDARD_TENORS}
        self.liq_bps   = liquidity_premium_bps
        self._curve: Dict[float, float] = {}   # tenor -> all-in zero rate
        self._spline = None
        self._spread_breakdown: Dict[float, Dict[str, float]] = {}

    def _banking_sector_premium(self, tenor: float) -> float:
        """Institution-specific BSP in bps at a given tenor."""
        tier_base = TIER_BASE_BSP.get(self.inst.tier, 65.0)
        npl_adj   = max(0.0, (self.inst.npl_ratio  - 0.03) * 800)
        car_adj   = max(0.0, (0.14 - self.inst.total_car)  * 1500)
        whl_adj   = max(0.0, (self.inst.wholesale_funding_pct - 0.35) * 200)
        ldr_adj   = max(0.0, (self.inst.loan_deposit_ratio - 0.90)    * 300)
        base_bps  = tier_base + npl_adj + car_adj + whl_adj + ldr_adj
        mult      = np.interp(tenor,
                              sorted(TENOR_MULT.keys()),
                              [TENOR_MULT[t] for t in sorted(TENOR_MULT.keys())])
        return base_bps * mult

    def build(self) -> "GCCIRSCurve":
        """Construct the full GCC IRS curve."""
        sov = SOVEREIGN_DATA.get(self.inst.country_iso3)
        if sov is None:
            raise ValueError(f"No sovereign data for {self.inst.country_iso3}. "
                             f"Available: {list(SOVEREIGN_DATA.keys())}")
        sov_curve = SovereignSpreadCurve(sov)
        sov_curve.build()

        for t in self.STANDARD_TENORS:
            sofr_zero  = self.sofr.zero_rate(t) * 100          # bps
            sov_sp     = sov_curve.spread_at(t)
            bank_sp    = self._banking_sector_premium(t)
            basis      = self.xccy_bps.get(t, 0.0)
            liq        = self.liq_bps
            total_bps  = sofr_zero + sov_sp + bank_sp + basis + liq
            self._curve[t] = total_bps / 100   # decimal

            self._spread_breakdown[t] = {
                "SOFR_zero_bps":     round(sofr_zero, 2),
                "sovereign_bps":     round(sov_sp,    2),
                "banking_bsp_bps":   round(bank_sp,   2),
                "xccy_basis_bps":    round(basis,      2),
                "liquidity_prem_bps":round(liq,        2),
                "total_bps":         round(total_bps,  2),
            }

        t_arr = np.array(sorted(self._curve.keys()))
        z_arr = np.array([self._curve[t] for t in t_arr])
        self._spline = CubicSpline(t_arr, np.log(z_arr + 1e-8), extrapolate=True)
        return self

    def zero_rate(self, t: float) -> float:
        """Interpolated GCC IRS zero rate at tenor t (decimal)."""
        if not self._curve:
            self.build()
        if t in self._curve:
            return self._curve[t]
        return max(float(np.exp(self._spline(t))), 0.0)

    def discount(self, t: float) -> float:
        """GCC curve discount factor."""
        return np.exp(-self.zero_rate(t) * t)

    def par_swap_rate(self, tenor: float, freq: int = 2) -> float:
        """Par IRS rate at given tenor against GCC curve."""
        dt    = 1.0 / freq
        times = np.arange(dt, tenor + dt / 2, dt)
        annuity = sum(dt * self.discount(t) for t in times)
        return (1.0 - self.discount(tenor)) / annuity

    def price_irs(
        self,
        notional: float,
        tenor:    float,
        pay_fixed: bool = True,
        fixed_rate: Optional[float] = None,
        freq: int = 2,
    ) -> IRSPrice:
        """
        Price a vanilla IRS using the GCC institution curve.

        Parameters
        ----------
        notional   : USD (not millions)
        tenor      : years
        pay_fixed  : True = pay fixed / receive floating (classic hedge)
        fixed_rate : if None, uses par swap rate (new trade)
        freq       : payment frequency (default 2 = semi-annual)

        Returns
        -------
        IRSPrice
        """
        if not self._curve:
            self.build()
        par = self.par_swap_rate(tenor, freq)
        K   = fixed_rate if fixed_rate is not None else par
        dt  = 1.0 / freq
        times = np.arange(dt, tenor + dt / 2, dt)
        if len(times) == 0:
            times = np.array([tenor])

        # Fixed leg PV
        fixed_pv = notional * K * dt * sum(self.discount(t) for t in times)
        # Floating leg PV = notional * (1 - df(T))
        float_pv = notional * (1.0 - self.discount(tenor))

        sign = -1 if pay_fixed else +1
        fv   = sign * (float_pv - fixed_pv)

        # DV01: analytical approximation via annuity × 1bp
        annuity = dt * sum(self.discount(t) for t in times)
        dv01    = sign * notional * annuity * 0.0001

        return IRSPrice(
            notional         = notional,
            tenor            = tenor,
            pay_fixed        = pay_fixed,
            par_swap_rate    = round(par, 6),
            fair_value       = round(fv / 1e6, 4),
            dv01             = round(dv01 / 1e6, 6),
            spread_breakdown = self._spread_breakdown.get(tenor, {}),
        )

    def curve_table(self) -> pd.DataFrame:
        """Full GCC IRS curve decomposition table."""
        if not self._curve:
            self.build()
        rows = []
        for t in self.STANDARD_TENORS:
            bd = self._spread_breakdown.get(t, {})
            rows.append({
                "Tenor (Y)":         t,
                "SOFR Zero (bps)":   bd.get("SOFR_zero_bps", 0),
                "Sovereign (bps)":   bd.get("sovereign_bps", 0),
                "Bank BSP (bps)":    bd.get("banking_bsp_bps", 0),
                "XCCY Basis (bps)":  bd.get("xccy_basis_bps", 0),
                "Liquidity (bps)":   bd.get("liquidity_prem_bps", 0),
                "All-in Rate (bps)": bd.get("total_bps", 0),
                "Par Swap (%)":      round(self.par_swap_rate(t) * 100, 4),
            })
        return pd.DataFrame(rows)
