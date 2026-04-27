"""
tif.derivatives.effectiveness_testing
=======================================
Automated hedge effectiveness testing under ASC 815 (U.S. GAAP)
and IFRS 9 principles-based framework.

Generates audit-ready documentation with timestamp and classification.

Reference
---------
Siddiqui, I. (2025). An Integrated Derivative Hedging Framework.
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

try:
    from scipy.stats import pearsonr, linregress
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

Classification = Literal["OCI", "P&L", "Discontinue", "Pending"]
Standard       = Literal["ASC815", "IFRS9"]


@dataclass
class HedgeRelationship:
    """Definition of a hedge relationship for effectiveness testing."""
    hedge_id:            str
    start_date:          str           # ISO date string
    hedged_item:         str           # description
    hedging_instrument:  str           # description
    hedge_type:          str           # "fair_value", "cash_flow", "net_investment"
    notional_usd_m:      float
    standard:            Standard = "ASC815"


@dataclass
class EffectivenessResult:
    """Effectiveness test output for one assessment period."""
    hedge_id:               str
    assessment_date:        str
    standard:               Standard
    method:                 str
    dollar_offset_ratio:    float
    r_squared:              float
    is_highly_effective:    bool
    classification:         Classification
    ineffectiveness_usd_m:  float
    notes:                  str = ""

    def to_dict(self) -> dict:
        return {
            "Hedge ID":          self.hedge_id,
            "Assessment Date":   self.assessment_date,
            "Standard":          self.standard,
            "Method":            self.method,
            "Dollar Offset":     f"{self.dollar_offset_ratio:.4f}",
            "R-Squared":         f"{self.r_squared:.4f}",
            "Highly Effective":  self.is_highly_effective,
            "Classification":    self.classification,
            "Ineffectiveness $M":self.ineffectiveness_usd_m,
        }


class EffectivenessTest:
    """
    Automated hedge effectiveness testing engine.

    Supports:
    - ASC 815 Hypothetical Derivative Method (HDM)
    - ASC 815 Dollar Offset Method
    - IFRS 9 Principles-Based Test (economic relationship,
      credit risk dominance, hedge ratio alignment)

    Parameters
    ----------
    offset_lower : float
        Lower bound of acceptable dollar offset range (default: 0.80).
    offset_upper : float
        Upper bound of acceptable dollar offset range (default: 1.25).

    Examples
    --------
    >>> et = EffectivenessTest()
    >>> result = et.asc815_hdm(rel, actual_changes, hypo_changes)
    >>> df = et.report([result])
    """

    def __init__(
        self,
        offset_lower: float = 0.80,
        offset_upper: float = 1.25,
    ) -> None:
        if not _SCIPY_OK:
            raise ImportError("Install: scipy")
        self.offset_lower = offset_lower
        self.offset_upper = offset_upper

    def _classify_offset(self, ratio: float) -> Classification:
        if self.offset_lower <= ratio <= self.offset_upper:
            return "OCI"
        elif 0.60 <= ratio < self.offset_lower or self.offset_upper < ratio <= 1.50:
            return "P&L"
        return "Discontinue"

    # ── ASC 815 Hypothetical Derivative Method ────────────────────
    def asc815_hdm(
        self,
        relationship: HedgeRelationship,
        actual_changes: pd.Series,          # fair value changes of actual derivative
        hypothetical_changes: pd.Series,    # fair value changes of perfect hedge
    ) -> EffectivenessResult:
        """
        ASC 815-20 Hypothetical Derivative Method.

        Compares cumulative fair value change of actual derivative
        against theoretically perfect (hypothetical) derivative.
        Dollar offset ratio must be in [0.80, 1.25] for OCI treatment.
        """
        cum_actual = float(actual_changes.cumsum().iloc[-1])
        cum_hypo   = float(hypothetical_changes.cumsum().iloc[-1])

        if abs(cum_hypo) < 1e-9:
            return EffectivenessResult(
                hedge_id=relationship.hedge_id,
                assessment_date=str(datetime.date.today()),
                standard="ASC815", method="HDM",
                dollar_offset_ratio=0.0, r_squared=0.0,
                is_highly_effective=False, classification="Pending",
                ineffectiveness_usd_m=0.0,
                notes="Insufficient data: hypothetical changes near zero.",
            )

        ratio = abs(cum_actual / cum_hypo)

        # Regression R²
        slope, intercept, r, *_ = linregress(hypothetical_changes, actual_changes)
        r2 = r**2

        # Ineffectiveness = actual − hypothetical (cumulative)
        ineff = (actual_changes - hypothetical_changes).sum()
        ineff_usd = round(ineff * relationship.notional_usd_m / 1_000_000, 4)

        highly_eff = self.offset_lower <= ratio <= self.offset_upper
        return EffectivenessResult(
            hedge_id             = relationship.hedge_id,
            assessment_date      = str(datetime.date.today()),
            standard             = "ASC815",
            method               = "Hypothetical Derivative Method",
            dollar_offset_ratio  = round(ratio, 4),
            r_squared            = round(r2, 4),
            is_highly_effective  = highly_eff,
            classification       = self._classify_offset(ratio),
            ineffectiveness_usd_m= ineff_usd,
            notes=(
                "ASC 815-20-55: Dollar offset within [80%, 125%] — OCI treatment." if highly_eff
                else f"Dollar offset {ratio:.2%} outside [80%, 125%] — earnings impact required."
            ),
        )

    # ── ASC 815 Dollar Offset (retrospective) ────────────────────
    def asc815_dollar_offset(
        self,
        relationship: HedgeRelationship,
        hedged_item_changes: pd.Series,
        hedging_instr_changes: pd.Series,
    ) -> EffectivenessResult:
        """ASC 815 retrospective dollar offset ratio test."""
        cum_item  = float(hedged_item_changes.cumsum().iloc[-1])
        cum_instr = float(hedging_instr_changes.cumsum().iloc[-1])
        ratio = abs(cum_instr / cum_item) if abs(cum_item) > 1e-9 else 0.0
        ineff = (hedging_instr_changes + hedged_item_changes).sum()
        _, _, r, *_ = linregress(hedged_item_changes, -hedging_instr_changes)
        return EffectivenessResult(
            hedge_id             = relationship.hedge_id,
            assessment_date      = str(datetime.date.today()),
            standard             = "ASC815",
            method               = "Dollar Offset (Retrospective)",
            dollar_offset_ratio  = round(ratio, 4),
            r_squared            = round(r**2, 4),
            is_highly_effective  = self.offset_lower <= ratio <= self.offset_upper,
            classification       = self._classify_offset(ratio),
            ineffectiveness_usd_m= round(ineff * relationship.notional_usd_m / 1e6, 4),
        )

    # ── IFRS 9 Principles-Based ───────────────────────────────────
    def ifrs9_principles(
        self,
        relationship: HedgeRelationship,
        hedged_item_changes:  pd.Series,
        hedging_instr_changes: pd.Series,
        credit_risk_proxy:    pd.Series,
    ) -> EffectivenessResult:
        """
        IFRS 9 principles-based effectiveness test.

        Three conditions:
        1. Economic relationship between hedged item and instrument.
        2. Credit risk does not dominate the value changes.
        3. Hedge ratio aligned (dollar offset in [80%, 125%]).
        """
        # 1. Economic relationship: correlation > 0.80 and significant
        corr_val, pval = pearsonr(hedged_item_changes, -hedging_instr_changes)
        economic_rel = (corr_val > 0.80) and (pval < 0.05)

        # 2. Credit risk dominance check
        combined    = hedged_item_changes + hedging_instr_changes
        corr_credit, _ = pearsonr(combined, credit_risk_proxy) if len(credit_risk_proxy) > 3 else (0, 1)
        credit_dominates = abs(corr_credit) > 0.60

        # 3. Hedge ratio
        cum_item  = float(hedged_item_changes.cumsum().iloc[-1])
        cum_instr = float(hedging_instr_changes.cumsum().iloc[-1])
        ratio     = abs(cum_instr / cum_item) if abs(cum_item) > 1e-9 else 0.0
        ratio_ok  = self.offset_lower <= ratio <= self.offset_upper

        highly_eff = economic_rel and not credit_dominates and ratio_ok
        ineff      = combined.sum()

        notes_parts = []
        if not economic_rel:
            notes_parts.append(f"Economic relationship weak (corr={corr_val:.2f})")
        if credit_dominates:
            notes_parts.append(f"Credit risk dominant (corr={corr_credit:.2f})")
        if not ratio_ok:
            notes_parts.append(f"Hedge ratio misaligned ({ratio:.2%})")
        if highly_eff:
            notes_parts.append("All three IFRS 9 conditions satisfied.")

        return EffectivenessResult(
            hedge_id             = relationship.hedge_id,
            assessment_date      = str(datetime.date.today()),
            standard             = "IFRS9",
            method               = "Principles-Based (IFRS 9 B6.4)",
            dollar_offset_ratio  = round(ratio, 4),
            r_squared            = round(corr_val**2, 4),
            is_highly_effective  = highly_eff,
            classification       = "OCI" if highly_eff else "P&L",
            ineffectiveness_usd_m= round(ineff * relationship.notional_usd_m / 1e6, 4),
            notes                = " | ".join(notes_parts),
        )

    @staticmethod
    def report(results: List[EffectivenessResult]) -> pd.DataFrame:
        """Generate audit-ready effectiveness report."""
        return pd.DataFrame([r.to_dict() for r in results])

    @staticmethod
    def portfolio_summary(results: List[EffectivenessResult]) -> Dict[str, int]:
        """Count by classification across portfolio."""
        counts: Dict[str, int] = {"OCI": 0, "P&L": 0, "Discontinue": 0, "Pending": 0}
        for r in results:
            counts[r.classification] = counts.get(r.classification, 0) + 1
        return counts
