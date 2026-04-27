"""tif.curves.country_data — GCC country reference data."""
from dataclasses import dataclass
from typing import Dict

@dataclass
class GCCCountry:
    name: str; iso3: str; currency: str; peg_rate: float
    rating_moodys: str; rating_sp: str
    gdp_usd_bn: float; oil_revenue_pct: float; car_system_pct: float

GCC_COUNTRIES: Dict[str, GCCCountry] = {
    "SAU": GCCCountry("Saudi Arabia","SAU","SAR",3.750,"Aa3","A+",   1067,65,19.2),
    "ARE": GCCCountry("UAE",         "ARE","AED",3.673,"Aa2","AA",    509,30,17.8),
    "QAT": GCCCountry("Qatar",       "QAT","QAR",3.640,"Aa3","AA+",  219,55,20.1),
    "KWT": GCCCountry("Kuwait",      "KWT","KWD",0.307,"A1","AA-",   184,60,18.6),
    "OMN": GCCCountry("Oman",        "OMN","OMR",0.385,"Ba1","BB+",   104,75,17.2),
    "BHR": GCCCountry("Bahrain",     "BHR","BHD",0.376,"B2","B+",     44, 68,19.8),
}
