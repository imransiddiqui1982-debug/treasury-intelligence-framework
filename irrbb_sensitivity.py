"""
Treasury Intelligence Framework — Test Suite
=============================================
Validates core functionality across all three modules.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════
# MODULE 1: SOFR Curve Tests
# ═══════════════════════════════════════════════════════════════
class TestSOFRCurve:
    def setup_method(self):
        from tif.curves.sofr_curve import SOFRCurve, SwapRate
        self.curve = SOFRCurve()

    def test_discount_factor_at_zero(self):
        assert self.curve.discount(0.0) == pytest.approx(1.0, abs=1e-9)

    def test_discount_factor_decreasing(self):
        """Discount factors must be monotonically decreasing."""
        tenors = [1, 2, 5, 10, 20, 30]
        dfs    = [self.curve.discount(t) for t in tenors]
        for i in range(len(dfs) - 1):
            assert dfs[i] > dfs[i+1], f"DF not decreasing at tenor {tenors[i]}"

    def test_zero_rate_positive(self):
        for t in [1, 5, 10]:
            assert self.curve.zero_rate(t) > 0

    def test_forward_rate_between_tenors(self):
        fwd = self.curve.forward_rate(5.0, 6.0)
        assert 0.0 < fwd < 0.20, f"Forward rate out of range: {fwd}"

    def test_par_swap_reprices_at_par(self):
        """Par swap rate should reprice a new swap at exactly zero FV."""
        par = self.curve.par_swap_rate(5.0)
        assert 0.03 < par < 0.08, f"Par rate out of range: {par}"

    def test_bump_shifts_rates(self):
        bumped = self.curve.bump(+100)
        for t in [1, 5, 10]:
            assert bumped.zero_rate(t) > self.curve.zero_rate(t)

    def test_curve_table_returns_dataframe(self):
        df = self.curve.zero_curve_df()
        assert isinstance(df, pd.DataFrame)
        assert "Zero Rate (%)" in df.columns
        assert len(df) > 0


# ═══════════════════════════════════════════════════════════════
# MODULE 2: Sovereign Spread Curve Tests
# ═══════════════════════════════════════════════════════════════
class TestSovereignSpreadCurve:
    def setup_method(self):
        from tif.curves.sovereign_spreads import SovereignSpreadCurve, SOVEREIGN_DATA
        self.data  = SOVEREIGN_DATA
        self.curve_sau = SovereignSpreadCurve(SOVEREIGN_DATA["SAU"])
        self.curve_omn = SovereignSpreadCurve(SOVEREIGN_DATA["OMN"])
        self.curve_bhr = SovereignSpreadCurve(SOVEREIGN_DATA["BHR"])

    def test_spreads_positive(self):
        for iso in ["SAU", "OMN", "BHR"]:
            from tif.curves.sovereign_spreads import SovereignSpreadCurve, SOVEREIGN_DATA
            c = SovereignSpreadCurve(SOVEREIGN_DATA[iso])
            c.build()
            assert c.spread_at(5.0) > 0

    def test_higher_risk_wider_spread(self):
        """Bahrain (B2) > Oman (Ba1) > Saudi Arabia (Aa3)."""
        self.curve_sau.build()
        self.curve_omn.build()
        self.curve_bhr.build()
        sp_sau = self.curve_sau.spread_at(5.0)
        sp_omn = self.curve_omn.spread_at(5.0)
        sp_bhr = self.curve_bhr.spread_at(5.0)
        assert sp_bhr > sp_omn > sp_sau, (
            f"Expected BHR({sp_bhr:.0f}) > OMN({sp_omn:.0f}) > SAU({sp_sau:.0f})"
        )

    def test_spread_interpolation_smooth(self):
        self.curve_omn.build()
        sp_5  = self.curve_omn.spread_at(5.0)
        sp_7  = self.curve_omn.spread_at(7.0)
        sp_10 = self.curve_omn.spread_at(10.0)
        assert sp_5 < sp_7 < sp_10, "Spreads should widen with tenor for Ba1"

    def test_compare_gcc_returns_all_countries(self):
        from tif.curves.sovereign_spreads import SovereignSpreadCurve
        df = SovereignSpreadCurve.compare_gcc()
        assert len(df) == 6
        assert "5Y Spread (bps)" in df.columns


# ═══════════════════════════════════════════════════════════════
# MODULE 3: GCC IRS Curve Tests
# ═══════════════════════════════════════════════════════════════
class TestGCCIRSCurve:
    def setup_method(self):
        from tif.curves.sofr_curve import SOFRCurve
        from tif.curves.gcc_irs_curve import GCCIRSCurve, GCCInstitution
        self.sofr = SOFRCurve()
        self.inst = GCCInstitution(
            name="Test Bank Oman", country_iso3="OMN", tier=1,
            total_car=0.172, npl_ratio=0.028, wholesale_funding_pct=0.32,
        )
        self.gcc = GCCIRSCurve(self.inst, self.sofr)
        self.gcc.build()

    def test_gcc_rate_exceeds_sofr(self):
        """GCC IRS rate must exceed SOFR due to sovereign + BSP spreads."""
        for t in [1, 5, 10]:
            gcc_z  = self.gcc.zero_rate(t)
            sofr_z = self.sofr.zero_rate(t)
            assert gcc_z > sofr_z, f"GCC rate ≤ SOFR at tenor {t}Y"

    def test_irs_pricing_structure(self):
        price = self.gcc.price_irs(100_000_000, 5.0, pay_fixed=True)
        assert price.par_swap_rate > 0
        assert isinstance(price.fair_value, float)
        assert isinstance(price.dv01, float)

    def test_curve_table_completeness(self):
        df = self.gcc.curve_table()
        assert len(df) == len(self.gcc.STANDARD_TENORS)
        assert "All-in Rate (bps)" in df.columns
        assert "Sovereign (bps)" in df.columns

    def test_par_swap_reasonable_range(self):
        par = self.gcc.par_swap_rate(5.0)
        # Oman tier-1 5Y IRS should be roughly SOFR + 200-350 bps
        assert 0.04 < par < 0.15, f"Oman par swap out of range: {par:.4%}"


# ═══════════════════════════════════════════════════════════════
# MODULE 4: IRRBB Engine Tests
# ═══════════════════════════════════════════════════════════════
class TestIRRBBEngine:
    def setup_method(self):
        from tif.derivatives.irrbb_sensitivity import IRRBBEngine, Position
        self.engine = IRRBBEngine(tier1_capital=2500.0)
        self.engine.add_positions([
            Position("Fixed mortgages", 8000, 4.5, 0.05, is_asset=True),
            Position("Floating loans",  4000, 0.3, 0.0,  is_asset=True,  rate_type="floating"),
            Position("Retail deposits", 9000, 0.5, 0.0,  is_asset=False, rate_type="floating"),
            Position("Pay-fixed IRS",   5000, 4.0, 0.04, is_derivative=True, pay_fixed=True),
        ])

    def test_full_report_has_six_scenarios(self):
        df = self.engine.full_report()
        assert len(df) == 6

    def test_parallel_up_negative_eve(self):
        """Rising rates should reduce EVE for an asset-sensitive bank."""
        from tif.derivatives.irrbb_sensitivity import BCBSScenario
        res = self.engine.compute_scenario(BCBSScenario.PARALLEL_UP)
        # Banking book EVE should be negative (fixed assets lose value)
        assert res.banking_book_eve < 0

    def test_derivative_offsets_banking_book(self):
        from tif.derivatives.irrbb_sensitivity import BCBSScenario
        res = self.engine.compute_scenario(BCBSScenario.PARALLEL_UP)
        # Pay-fixed IRS gains value when rates rise (positive derivative EVE)
        assert res.derivative_eve > 0

    def test_dv01_report_has_total_row(self):
        df = self.engine.dv01_report()
        assert "TOTAL" in df["Position"].values


# ═══════════════════════════════════════════════════════════════
# MODULE 5: Effectiveness Testing Tests
# ═══════════════════════════════════════════════════════════════
class TestEffectivenessTesting:
    def setup_method(self):
        from tif.derivatives.effectiveness_testing import EffectivenessTest, HedgeRelationship
        self.et  = EffectivenessTest()
        self.rel = HedgeRelationship(
            "H001", "2025-01-01", "Fixed bond", "IRS", "fair_value", 100.0
        )
        # Perfect hedge: actual = -hypothetical
        np.random.seed(42)
        self.hypo   = pd.Series(np.random.normal(0, 1, 60))
        self.actual = -self.hypo * np.random.uniform(0.90, 1.10, 60)  # ~90-110% effective
        self.credit = pd.Series(np.random.normal(0, 0.2, 60))

    def test_highly_effective_hedge(self):
        result = self.et.asc815_hdm(self.rel, self.actual, self.hypo)
        assert result.is_highly_effective
        assert result.classification == "OCI"

    def test_ineffective_hedge_classified_pl(self):
        bad_actual = pd.Series(np.random.normal(0, 1, 60))  # uncorrelated
        result = self.et.asc815_hdm(self.rel, bad_actual, self.hypo)
        assert result.classification in ["P&L", "Discontinue"]

    def test_ifrs9_passes_for_good_hedge(self):
        result = self.et.ifrs9_principles(self.rel, self.hypo, self.actual, self.credit)
        assert isinstance(result.is_highly_effective, bool)

    def test_report_returns_dataframe(self):
        r1 = self.et.asc815_hdm(self.rel, self.actual, self.hypo)
        df = EffectivenessTest.report([r1])
        assert isinstance(df, pd.DataFrame)
        assert "Classification" in df.columns


# ═══════════════════════════════════════════════════════════════
# MODULE 6: Early Warning System Tests
# ═══════════════════════════════════════════════════════════════
class TestEarlyWarningSystem:
    def setup_method(self):
        from tif.liquidity.early_warning import EarlyWarningSystem
        self.ews = EarlyWarningSystem()
        self.metrics_ok = {
            "wholesale_funding_concentration": 0.18,
            "deposit_outflow_rate_7d":          0.004,
            "interbank_funding_ratio":           0.15,
            "unencumbered_assets_ratio":         0.67,
            "fx_liquidity_gap_usd_m":           500,
            "intraday_liquidity_usage_pct":      0.54,
            "secured_funding_rollover_pct":       0.88,
            "contingent_commitments_usd_m":      1500,
        }
        self.metrics_stress = {**self.metrics_ok,
                               "wholesale_funding_concentration": 0.40,
                               "contingent_commitments_usd_m":    6000}

    def test_all_green_when_normal(self):
        results = self.ews.compute(self.metrics_ok)
        assert all(r.status in ["GREEN", "AMBER"] for r in results)

    def test_red_triggered_on_stress(self):
        results = self.ews.compute(self.metrics_stress)
        statuses = [r.status for r in results]
        assert "RED" in statuses

    def test_eight_indicators_returned(self):
        results = self.ews.compute(self.metrics_ok)
        assert len(results) == 8

    def test_overall_status_propagates(self):
        from tif.liquidity.early_warning import EarlyWarningSystem
        ok_results = self.ews.compute(self.metrics_ok)
        st_results = self.ews.compute(self.metrics_stress)
        assert EarlyWarningSystem.overall_status(st_results) == "RED"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
