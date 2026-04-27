"""
tif.liquidity.lcr_simulator
=============================
Monte Carlo LCR / NSFR simulation engine.

Assembles HQLA projections and behavioral outflow forecasts into
a forward-looking LCR ratio with confidence intervals.

Reference
---------
Siddiqui, I. (2025). Predictive LCR Modeling for Regional Banks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .deposit_outflow_model import DepositOutflowModel, DepositSegment, OutflowForecast
from .hqla_projection import HQLAProjection, HQLAProjectionResult, RateScenario

LCR_MINIMUM  = 1.00   # 100% regulatory minimum
NSFR_MINIMUM = 1.00   # 100% regulatory minimum

# Basel III ASF and RSF factors for NSFR (simplified)
ASF_FACTORS: Dict[str, float] = {
    "tier1_capital":        1.00,
    "retail_stable":        0.95,
    "retail_less_stable":   0.90,
    "wholesale_op":         0.50,
    "wholesale_other":      0.00,
}

RSF_FACTORS: Dict[str, float] = {
    "hqla_l1":              0.00,
    "hqla_l2a":             0.15,
    "hqla_l2b":             0.50,
    "loans_less_1yr":       0.50,
    "loans_more_1yr":       1.00,
    "securities_hqla":      0.05,
}


@dataclass
class SimulationResult:
    """Monte Carlo simulation output for a single scenario."""
    scenario: str
    dates: List[str]
    lcr_mean: List[float]
    lcr_p10:  List[float]
    lcr_p25:  List[float]
    lcr_p75:  List[float]
    lcr_p90:  List[float]
    breach_probability: List[float]   # P(LCR < 100%) at each date
    nsfr_mean: Optional[List[float]] = None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date":         self.dates,
            "lcr_mean":     self.lcr_mean,
            "lcr_p10":      self.lcr_p10,
            "lcr_p25":      self.lcr_p25,
            "lcr_p75":      self.lcr_p75,
            "lcr_p90":      self.lcr_p90,
            "breach_prob":  self.breach_probability,
        })

    def first_breach_date(self) -> Optional[str]:
        """Return first date where median LCR < 100%, or None."""
        for d, v in zip(self.dates, self.lcr_mean):
            if v < LCR_MINIMUM:
                return d
        return None

    def days_to_breach(self) -> Optional[int]:
        bd = self.first_breach_date()
        if bd is None:
            return None
        return self.dates.index(bd)


class LCRSimulator:
    """
    Forward-looking LCR / NSFR simulator using Monte Carlo sampling.

    Combines behavioral outflow model predictions with HQLA projections
    to compute a daily LCR trajectory with uncertainty quantification.

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo draws per date (default: 1000).
    outflow_noise_std : float
        Standard deviation of Gaussian noise added to outflow rates
        to model parameter uncertainty (default: 0.02).

    Examples
    --------
    >>> sim = LCRSimulator(n_simulations=2000)
    >>> results = sim.run(forecasts, hqla_results, total_assets)
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        outflow_noise_std: float = 0.02,
    ) -> None:
        self.n_simulations    = n_simulations
        self.outflow_noise_std = outflow_noise_std

    def _compute_net_outflow(
        self,
        forecasts: List[OutflowForecast],
        noise_scale: float,
        rng: np.random.Generator,
    ) -> float:
        """Compute total net cash outflow for one Monte Carlo draw."""
        total = 0.0
        for fc in forecasts:
            noise = rng.normal(0, noise_scale)
            rate  = np.clip(fc.predicted_outflow_rate + noise, 0, 1)
            total += fc.balance * rate
        return max(total, 1e-6)

    def run(
        self,
        outflow_forecasts: List[OutflowForecast],
        hqla_result: HQLAProjectionResult,
        total_liabilities_usd_m: float = 10_000.0,
        asf_map: Optional[Dict[str, float]] = None,
        rsf_map: Optional[Dict[str, float]] = None,
    ) -> SimulationResult:
        """
        Run Monte Carlo LCR simulation.

        Parameters
        ----------
        outflow_forecasts : from DepositOutflowModel.predict()
        hqla_result       : from HQLAProjection.project() — one scenario
        total_liabilities_usd_m : total stable funding base (for NSFR)

        Returns
        -------
        SimulationResult
        """
        rng  = np.random.default_rng(42)
        n    = min(len(hqla_result.dates), 30)   # 30-day horizon for LCR
        lcr_traces: List[List[float]] = []

        for _ in range(self.n_simulations):
            trace = []
            for i in range(n):
                hqla = hqla_result.total_hqla[i]
                nco  = self._compute_net_outflow(
                    outflow_forecasts, self.outflow_noise_std, rng
                )
                # Scale outflow to 30-day stress period proportionally
                day_factor = (i + 1) / 30
                lcr = hqla / (nco * day_factor)
                trace.append(np.clip(lcr, 0, 5))
            lcr_traces.append(trace)

        arr = np.array(lcr_traces)   # shape: (n_sim, n_days)

        return SimulationResult(
            scenario          = hqla_result.scenario,
            dates             = hqla_result.dates[:n],
            lcr_mean          = np.mean(arr, axis=0).round(4).tolist(),
            lcr_p10           = np.percentile(arr, 10, axis=0).round(4).tolist(),
            lcr_p25           = np.percentile(arr, 25, axis=0).round(4).tolist(),
            lcr_p75           = np.percentile(arr, 75, axis=0).round(4).tolist(),
            lcr_p90           = np.percentile(arr, 90, axis=0).round(4).tolist(),
            breach_probability= (arr < LCR_MINIMUM).mean(axis=0).round(4).tolist(),
        )

    def run_all_scenarios(
        self,
        outflow_forecasts: List[OutflowForecast],
        hqla_results: List[HQLAProjectionResult],
    ) -> List[SimulationResult]:
        """Run simulation across all HQLA projection scenarios."""
        return [
            self.run(outflow_forecasts, hr)
            for hr in hqla_results
        ]

    @staticmethod
    def summary_table(results: List[SimulationResult]) -> pd.DataFrame:
        """Produce a scenario comparison summary at day 30."""
        rows = []
        for r in results:
            rows.append({
                "Scenario":          r.scenario,
                "LCR Day-1 (mean)":  r.lcr_mean[0],
                "LCR Day-30 (mean)": r.lcr_mean[-1],
                "LCR P10 Day-30":    r.lcr_p10[-1],
                "Breach Prob D30":   f"{r.breach_probability[-1]:.1%}",
                "First Breach Date": r.first_breach_date() or "No breach",
            })
        return pd.DataFrame(rows)
