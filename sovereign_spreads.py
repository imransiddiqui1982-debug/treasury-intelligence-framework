"""
tif.derivatives.irrbb_sensitivity
===================================
BCBS 368 Interest Rate Risk in the Banking Book (IRRBB) engine.

Computes NII sensitivity and EVE impact across all six standardized
scenarios with derivative overlay support.

Reference
---------
BCBS 368 (2016). Standards: Interest Rate Risk in the Banking Book.
Siddiqui, I. (2025). An Integrated Derivative Hedging Framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

EVE_LIMIT_PCT_TIER1 = 0.20   # BCBS 368 outlier threshold: |ΔEVE| > 20% Tier 1


class BCBSScenario(Enum):
    """Six BCBS 368 standardized interest rate shock scenarios."""
    PARALLEL_UP      = (+200,   0,   "Parallel shift up +200bps")
    PARALLEL_DOWN    = (-200,   0,   "Parallel shift down -200bps")
    STEEPENER        = (-150, +150,  "Short rates down, long rates up")
    FLATTENER        = (+150, -150,  "Short rates up, long rates down")
    SHORT_RATE_UP    = (+250,   0,   "Short-end shock up +250bps")
    SHORT_RATE_DOWN  = (-100,   0,   "Short-end shock down -100bps")

    def __init__(self, short_shock: int, long_shock: int, description: str):
        self.short_shock = short_shock
        self.long_shock  = long_shock
        self.description = description


@dataclass
class Position:
    """A banking book or derivative position."""
    name:           str
    notional:       float          # USD millions
    duration:       float          # modified duration (years)
    convexity:      float = 0.0    # dollar convexity
    is_derivative:  bool  = False  # True = hedging instrument
    pay_fixed:      bool  = True   # for derivatives: pay_fixed IRS or receive_fixed
    is_asset:       bool  = True   # asset vs liability
    rate_type:      str   = "fixed"   # "fixed" or "floating"
    repricing_years:float = 1.0    # for floating: avg repricing horizon


@dataclass
class ScenarioResult:
    """Output of IRRBB sensitivity computation for one scenario."""
    scenario:              str
    description:           str
    banking_book_eve:      float   # USD millions
    derivative_eve:        float
    net_eve:               float
    banking_book_nii:      float
    derivative_nii:        float
    net_nii:               float
    eve_pct_tier1:         float
    nii_pct_tier1:         float
    eve_outlier:           bool    # True if |net_eve/T1| > 20%
    hedge_effectiveness_eve: float  # |derivative_eve / banking_book_eve|


class IRRBBEngine:
    """
    Full BCBS 368 IRRBB measurement engine.

    Computes both EVE and NII sensitivity under all six standardized
    rate shock scenarios, with derivative overlay decomposition.

    Parameters
    ----------
    tier1_capital : float
        Tier 1 capital in USD millions (for ratio computation).

    Examples
    --------
    >>> engine = IRRBBEngine(tier1_capital=2500.0)
    >>> engine.add_positions(positions)
    >>> report = engine.full_report()
    """

    def __init__(self, tier1_capital: float) -> None:
        self.tier1_capital = tier1_capital
        self._positions: List[Position] = []

    def add_positions(self, positions: List[Position]) -> "IRRBBEngine":
        self._positions.extend(positions)
        return self

    def _effective_shock(self, pos: Position, scenario: BCBSScenario) -> float:
        """Map scenario shocks to effective rate change for position duration."""
        if pos.duration < 1.0:
            shock = scenario.short_shock
        elif pos.duration > 5.0:
            shock = scenario.long_shock
        else:
            # Linear interpolation between short and long zones
            w = (pos.duration - 1.0) / 4.0
            shock = (1 - w) * scenario.short_shock + w * scenario.long_shock
        return shock / 10_000   # convert bps to decimal

    def _compute_eve(self, pos: Position, scenario: BCBSScenario) -> float:
        """
        Duration-convexity approximation of EVE change.
        ΔEVE ≈ -D * Δr * N + 0.5 * C * (Δr)² * N
        Sign flipped for pay-fixed derivatives and liabilities.
        """
        dr = self._effective_shock(pos, scenario)
        dv = (-pos.duration * dr + 0.5 * pos.convexity * dr**2) * pos.notional

        if pos.is_derivative:
            sign = -1 if pos.pay_fixed else +1
        else:
            sign = +1 if pos.is_asset else -1
        return sign * dv

    def _compute_nii(self, pos: Position, scenario: BCBSScenario) -> float:
        """
        12-month NII impact for floating-rate positions.
        Fixed-rate positions: no immediate NII impact.
        """
        if pos.rate_type == "fixed":
            return 0.0
        dr = self._effective_shock(pos, scenario)
        nii = pos.notional * dr * min(pos.repricing_years, 1.0)
        if pos.is_derivative:
            sign = -1 if pos.pay_fixed else +1
        else:
            sign = +1 if pos.is_asset else -1
        return sign * nii

    def compute_scenario(self, scenario: BCBSScenario) -> ScenarioResult:
        """Compute full IRRBB metrics for one scenario."""
        bb_eve = sum(
            self._compute_eve(p, scenario)
            for p in self._positions if not p.is_derivative
        )
        d_eve = sum(
            self._compute_eve(p, scenario)
            for p in self._positions if p.is_derivative
        )
        bb_nii = sum(
            self._compute_nii(p, scenario)
            for p in self._positions if not p.is_derivative
        )
        d_nii = sum(
            self._compute_nii(p, scenario)
            for p in self._positions if p.is_derivative
        )
        net_eve = bb_eve + d_eve
        net_nii = bb_nii + d_nii
        eve_pct = net_eve / self.tier1_capital * 100
        nii_pct = net_nii / self.tier1_capital * 100
        hedge_eff = abs(d_eve / bb_eve) if abs(bb_eve) > 1e-6 else 0.0

        return ScenarioResult(
            scenario                = scenario.name,
            description             = scenario.description,
            banking_book_eve        = round(bb_eve, 2),
            derivative_eve          = round(d_eve, 2),
            net_eve                 = round(net_eve, 2),
            banking_book_nii        = round(bb_nii, 2),
            derivative_nii          = round(d_nii, 2),
            net_nii                 = round(net_nii, 2),
            eve_pct_tier1           = round(eve_pct, 2),
            nii_pct_tier1           = round(nii_pct, 2),
            eve_outlier             = abs(eve_pct) > EVE_LIMIT_PCT_TIER1 * 100,
            hedge_effectiveness_eve = round(min(hedge_eff, 1.0), 4),
        )

    def full_report(self) -> pd.DataFrame:
        """Run all six BCBS 368 scenarios and return ALCO-ready DataFrame."""
        rows = []
        for sc in BCBSScenario:
            res = self.compute_scenario(sc)
            rows.append({
                "Scenario":               res.scenario,
                "Description":            res.description,
                "BB EVE ($M)":            res.banking_book_eve,
                "Deriv Offset ($M)":      res.derivative_eve,
                "Net EVE ($M)":           res.net_eve,
                "BB NII ($M)":            res.banking_book_nii,
                "Deriv NII ($M)":         res.derivative_nii,
                "Net NII ($M)":           res.net_nii,
                "EVE % T1":               f"{res.eve_pct_tier1:.1f}%",
                "NII % T1":               f"{res.nii_pct_tier1:.1f}%",
                "Outlier (>20% T1)":      "YES" if res.eve_outlier else "No",
                "Hedge Eff (EVE)":        f"{res.hedge_effectiveness_eve:.1%}",
            })
        return pd.DataFrame(rows)

    def dv01_report(self) -> pd.DataFrame:
        """Portfolio DV01 breakdown by position."""
        rows = []
        for pos in self._positions:
            dv01 = pos.duration * pos.notional * 0.0001
            sign = -1 if pos.pay_fixed and pos.is_derivative else 1
            rows.append({
                "Position":    pos.name,
                "Type":        "Derivative" if pos.is_derivative else "Banking Book",
                "Asset/Liab":  "Asset" if pos.is_asset else "Liability",
                "Notional $M": pos.notional,
                "Duration":    pos.duration,
                "DV01 $M":     round(sign * dv01, 4),
            })
        df = pd.DataFrame(rows)
        df.loc[len(df)] = ["TOTAL", "", "", df["Notional $M"].sum(), "", df["DV01 $M"].sum()]
        return df
