"""
tif.liquidity.early_warning
=============================
Early Warning Indicator (EWI) system for liquidity stress monitoring.

Computes 8 regulatory and market-based EWIs with traffic-light
status classification aligned to BCBS liquidity risk monitoring tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Literal

Status = Literal["GREEN", "AMBER", "RED"]


@dataclass
class EWIResult:
    """Single EWI reading."""
    indicator:   str
    value:       float
    unit:        str
    status:      Status
    threshold_amber: float
    threshold_red:   float
    description: str

    def to_dict(self) -> dict:
        return {
            "Indicator":    self.indicator,
            "Value":        self.value,
            "Unit":         self.unit,
            "Status":       self.status,
            "Amber Thresh": self.threshold_amber,
            "Red Thresh":   self.threshold_red,
        }


class EarlyWarningSystem:
    """
    Compute EWIs and assign traffic-light status.

    Parameters
    ----------
    thresholds : dict, optional
        Override default amber/red thresholds per indicator.

    Examples
    --------
    >>> ews = EarlyWarningSystem()
    >>> results = ews.compute(metrics_dict)
    >>> df = ews.to_dataframe(results)
    """

    # Default thresholds: (amber, red)
    DEFAULT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
        "wholesale_funding_concentration": (0.20, 0.35),
        "deposit_outflow_rate_7d":         (0.01, 0.03),
        "interbank_funding_ratio":         (0.20, 0.35),
        "unencumbered_assets_ratio":       (0.50, 0.35),  # inverted: lower is worse
        "fx_liquidity_gap_usd_m":         (500, 2_000),
        "intraday_liquidity_usage_pct":   (0.65, 0.85),
        "secured_funding_rollover_pct":    (0.80, 0.65),  # inverted
        "contingent_commitments_usd_m":   (2_000, 5_000),
    }

    INVERTED = {  # indicators where lower value = worse
        "unencumbered_assets_ratio",
        "secured_funding_rollover_pct",
    }

    def __init__(self, thresholds: Optional[Dict] = None) -> None:
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def _classify(self, key: str, value: float) -> Status:
        amber, red = self.thresholds[key]
        if key in self.INVERTED:
            if value >= amber:
                return "GREEN"
            elif value >= red:
                return "AMBER"
            return "RED"
        else:
            if value <= amber:
                return "GREEN"
            elif value <= red:
                return "AMBER"
            return "RED"

    def compute(self, metrics: Dict[str, float]) -> List[EWIResult]:
        """
        Compute all EWIs from a metrics dictionary.

        Parameters
        ----------
        metrics : dict
            Keys matching DEFAULT_THRESHOLDS. Values in natural units.

        Returns
        -------
        list of EWIResult
        """
        labels = {
            "wholesale_funding_concentration": ("Wholesale Funding Concentration", "%"),
            "deposit_outflow_rate_7d":          ("Deposit Outflow Rate (7d)",       "%"),
            "interbank_funding_ratio":           ("Interbank Funding Ratio",         "%"),
            "unencumbered_assets_ratio":         ("Unencumbered Assets Ratio",       "%"),
            "fx_liquidity_gap_usd_m":           ("FX Liquidity Gap",                "USD M"),
            "intraday_liquidity_usage_pct":     ("Intraday Liquidity Usage",         "%"),
            "secured_funding_rollover_pct":      ("Secured Funding Rollover",         "%"),
            "contingent_commitments_usd_m":     ("Contingent Commitments",           "USD M"),
        }
        results = []
        for key, (label, unit) in labels.items():
            val = metrics.get(key, 0.0)
            if unit == "%":
                disp_val = round(val * 100, 2)
            else:
                disp_val = round(val, 1)
            amber, red = self.thresholds[key]
            results.append(EWIResult(
                indicator       = label,
                value           = disp_val,
                unit            = unit,
                status          = self._classify(key, val),
                threshold_amber = amber * (100 if unit == "%" else 1),
                threshold_red   = red   * (100 if unit == "%" else 1),
                description     = f"BCBS liquidity risk monitoring tool: {label}",
            ))
        return results

    @staticmethod
    def to_dataframe(results: List[EWIResult]) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in results])

    @staticmethod
    def overall_status(results: List[EWIResult]) -> Status:
        if any(r.status == "RED" for r in results):
            return "RED"
        if any(r.status == "AMBER" for r in results):
            return "AMBER"
        return "GREEN"
