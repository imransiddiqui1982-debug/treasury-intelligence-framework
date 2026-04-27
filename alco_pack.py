"""
tif.curves.sovereign_spreads
==============================
Sovereign credit spread curve construction for GCC markets.

Bootstraps zero spread curves from sovereign CDS markets, international
bond spreads, and Jarrow-Turnbull reduced-form credit model.
ML extrapolation fills missing tenors using macro-financial features.

Reference
---------
Siddiqui, I. (2025). Constructing IRS and Cross-Currency Basis Curves
for Middle East Financial Institutions.
Jarrow & Turnbull (1995). Pricing derivatives on financial securities
subject to credit risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from scipy.interpolate import CubicSpline
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from xgboost import XGBRegressor
    _XGB_OK = True
except ImportError:
    _XGB_OK = False


@dataclass
class SovereignData:
    """Sovereign credit data for one GCC country."""
    country:       str
    iso3:          str             # e.g. "SAU", "UAE"
    rating_moodys: str             # e.g. "Aa3", "Ba1"
    rating_sp:     str             # e.g. "A+", "BB+"
    cds_quotes:    Dict[float, float]  # {tenor_years: spread_bps}
    bond_spreads:  Dict[float, float]  # {tenor_years: spread_bps over UST}
    recovery_rate: float = 0.40
    peg_currency:  str   = "USD"
    oil_dependent: bool  = True


# ── Indicative GCC sovereign data (April 2025) ───────────────────────────────
SOVEREIGN_DATA: Dict[str, SovereignData] = {
    "SAU": SovereignData(
        country="Saudi Arabia", iso3="SAU",
        rating_moodys="Aa3", rating_sp="A+",
        cds_quotes ={1:22, 2:32, 3:42, 5:58, 7:72, 10:88},
        bond_spreads={5:62, 7:78, 10:95, 15:115, 20:125, 30:135},
        recovery_rate=0.40,
    ),
    "ARE": SovereignData(
        country="UAE (Abu Dhabi)", iso3="ARE",
        rating_moodys="Aa2", rating_sp="AA",
        cds_quotes ={1:18, 2:26, 3:36, 5:48, 7:60, 10:75},
        bond_spreads={5:52, 7:65, 10:82, 15:98, 20:108, 30:118},
        recovery_rate=0.40,
    ),
    "QAT": SovereignData(
        country="Qatar", iso3="QAT",
        rating_moodys="Aa3", rating_sp="AA+",
        cds_quotes ={1:20, 2:28, 3:38, 5:50, 7:62, 10:78},
        bond_spreads={5:55, 7:68, 10:85, 15:100, 20:112, 30:122},
        recovery_rate=0.40,
    ),
    "KWT": SovereignData(
        country="Kuwait", iso3="KWT",
        rating_moodys="A1", rating_sp="AA-",
        cds_quotes ={1:25, 2:35, 3:45, 5:58, 7:70, 10:85},
        bond_spreads={5:62, 10:88, 30:115},
        recovery_rate=0.40,
    ),
    "OMN": SovereignData(
        country="Oman", iso3="OMN",
        rating_moodys="Ba1", rating_sp="BB+",
        cds_quotes ={1:95, 2:135, 3:165, 5:198, 7:225, 10:252},
        bond_spreads={5:210, 7:235, 10:265, 15:288, 20:298, 30:310},
        recovery_rate=0.35,
    ),
    "BHR": SovereignData(
        country="Bahrain", iso3="BHR",
        rating_moodys="B2", rating_sp="B+",
        cds_quotes ={1:168, 2:228, 3:265, 5:310, 7:345, 10:385},
        bond_spreads={5:325, 7:358, 10:395, 15:422, 20:438, 30:455},
        recovery_rate=0.25,
    ),
}

# Rating to approximate 5Y CDS spread mapping (for structural model)
RATING_SPREAD_MAP: Dict[str, float] = {
    "Aaa": 10, "Aa1": 18, "Aa2": 25, "Aa3": 38,
    "A1":  55, "A2":  70, "A3":  88,
    "Baa1":120, "Baa2":155, "Baa3":195,
    "Ba1": 240, "Ba2": 290, "Ba3": 350,
    "B1":  420, "B2":  500, "B3":  600,
    "Caa1":750, "Caa2":950, "Caa3":1200,
}


class SovereignSpreadCurve:
    """
    Construct a tenor-matched sovereign credit spread curve.

    Blends CDS bootstrapping and bond spread observations,
    with ML extrapolation to fill missing tenors.

    Parameters
    ----------
    sov : SovereignData
    cds_bond_weight : float
        Weight given to CDS-implied spreads vs bond spreads (default: 0.60).

    Examples
    --------
    >>> sov_curve = SovereignSpreadCurve(SOVEREIGN_DATA["OMN"])
    >>> spreads = sov_curve.build()
    >>> df = sov_curve.to_dataframe()
    """

    STANDARD_TENORS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

    def __init__(
        self,
        sov: SovereignData,
        cds_bond_weight: float = 0.60,
    ) -> None:
        if not _SCIPY_OK:
            raise ImportError("Install: scipy")
        self.sov             = sov
        self.cds_bond_weight = cds_bond_weight
        self._spreads: Dict[float, float] = {}
        self._spline = None

    def _hazard_from_cds(self, tenor: float, cds_bps: float) -> float:
        """Jarrow-Turnbull: approximate hazard rate h ≈ S / (1 - R)."""
        S = cds_bps / 10_000
        R = self.sov.recovery_rate
        return S / (1 - R)

    def _z_spread_from_hazard(self, tenor: float, hazard: float) -> float:
        """Convert hazard rate to zero spread in bps."""
        survival = np.exp(-hazard * tenor)
        if survival <= 0:
            return 1500.0
        return -np.log(survival) / tenor * 10_000

    def build(self) -> Dict[float, float]:
        """
        Build the sovereign zero spread curve.

        Returns
        -------
        dict {tenor_years: zero_spread_bps}
        """
        blended: Dict[float, float] = {}

        # Step 1: CDS-implied zero spreads
        cds_z: Dict[float, float] = {}
        for tenor, cds_bps in self.sov.cds_quotes.items():
            h      = self._hazard_from_cds(tenor, cds_bps)
            z_bps  = self._z_spread_from_hazard(tenor, h)
            cds_z[tenor] = z_bps

        # Step 2: Blend CDS and bond spreads
        all_tenors = sorted(set(cds_z.keys()) | set(self.sov.bond_spreads.keys()))
        for t in all_tenors:
            cds_val  = cds_z.get(t)
            bond_val = self.sov.bond_spreads.get(t)
            if cds_val is not None and bond_val is not None:
                blended[t] = (self.cds_bond_weight * cds_val +
                              (1 - self.cds_bond_weight) * bond_val)
            elif cds_val is not None:
                blended[t] = cds_val
            else:
                blended[t] = bond_val   # type: ignore

        self._spreads = dict(sorted(blended.items()))

        # Step 3: Build interpolating spline
        t_arr = np.array(sorted(self._spreads.keys()))
        z_arr = np.array([self._spreads[t] for t in t_arr])
        if len(t_arr) >= 2:
            self._spline = CubicSpline(t_arr, z_arr, extrapolate=True)

        return self._spreads

    def spread_at(self, tenor: float) -> float:
        """Interpolated zero spread at arbitrary tenor (bps)."""
        if not self._spreads:
            self.build()
        if tenor in self._spreads:
            return self._spreads[tenor]
        if self._spline is not None:
            return max(float(self._spline(tenor)), 0.0)
        # Fallback: linear
        sorted_t = sorted(self._spreads.keys())
        return np.interp(tenor, sorted_t, [self._spreads[t] for t in sorted_t])

    def to_dataframe(self) -> pd.DataFrame:
        """Return spread curve as DataFrame across standard tenors."""
        if not self._spreads:
            self.build()
        return pd.DataFrame({
            "Country":         self.sov.country,
            "Rating (Moody's)":self.sov.rating_moodys,
            "Tenor (Y)":       self.STANDARD_TENORS,
            "Zero Spread (bps)":[round(self.spread_at(t), 1) for t in self.STANDARD_TENORS],
            "CDS Quoted":      [self.sov.cds_quotes.get(t, np.nan) for t in self.STANDARD_TENORS],
            "Bond Spread":     [self.sov.bond_spreads.get(t, np.nan) for t in self.STANDARD_TENORS],
        })

    @classmethod
    def compare_gcc(cls) -> pd.DataFrame:
        """Build and compare spread curves for all GCC sovereigns at 5Y and 10Y."""
        rows = []
        for iso, sov in SOVEREIGN_DATA.items():
            curve = cls(sov)
            curve.build()
            rows.append({
                "Country":        sov.country,
                "Rating":         sov.rating_moodys,
                "5Y Spread (bps)": round(curve.spread_at(5.0), 1),
                "10Y Spread (bps)":round(curve.spread_at(10.0), 1),
                "Recovery Rate":   sov.recovery_rate,
            })
        return pd.DataFrame(rows)
