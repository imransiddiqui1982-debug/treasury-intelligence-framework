"""tif.curves.xccy_basis — Cross-currency basis swap adjustment."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

@dataclass
class CurrencyPair:
    base: str; quote: str; peg_rate: float; typical_basis_bps: float

GCC_FX_PAIRS: Dict[str, CurrencyPair] = {
    "SAR/USD": CurrencyPair("SAR","USD",3.75,  2.0),
    "AED/USD": CurrencyPair("AED","USD",3.6725,1.5),
    "QAR/USD": CurrencyPair("QAR","USD",3.64,  2.5),
    "KWD/USD": CurrencyPair("KWD","USD",0.307, 3.0),
    "OMR/USD": CurrencyPair("OMR","USD",0.385, 8.0),
    "BHD/USD": CurrencyPair("BHD","USD",0.376,12.0),
}

class XCCYBasis:
    """Cross-currency basis adjustment for GCC/USD currency pairs."""

    TENOR_MULT: Dict[float, float] = {
        0.5:0.6, 1.0:0.8, 2.0:0.9, 3.0:1.0,
        5.0:1.1, 7.0:1.2, 10.0:1.35, 15.0:1.5, 20.0:1.65, 30.0:1.8
    }

    def __init__(self, pair: CurrencyPair, stress_multiplier: float = 1.0):
        self.pair   = pair
        self.stress = stress_multiplier

    def basis_at(self, tenor: float) -> float:
        mult = np.interp(tenor,
                         sorted(self.TENOR_MULT.keys()),
                         [self.TENOR_MULT[t] for t in sorted(self.TENOR_MULT.keys())])
        return self.pair.typical_basis_bps * mult * self.stress

    def basis_curve(self) -> Dict[float, float]:
        tenors = sorted(self.TENOR_MULT.keys())
        return {t: round(self.basis_at(t), 2) for t in tenors}

    def to_dataframe(self) -> pd.DataFrame:
        bc = self.basis_curve()
        return pd.DataFrame({
            "Currency Pair": self.pair.base + "/" + self.pair.quote,
            "Tenor (Y)":     list(bc.keys()),
            "Basis (bps)":   list(bc.values()),
        })
