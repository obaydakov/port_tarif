"""Unit tests for the generic tariff calculator engine.

These tests verify that the generic calculator reproduces known tariff
calculations when given the right schema values.  The rules used as fixtures
here are inline test data — they exercise the CALCULATOR and say nothing
about where rules come from in production (which is always LLM extraction).
"""

from __future__ import annotations

import math

import pytest

from calculation.calculator import (
    TariffLineItem,
    apply_rule,
    run_all_calculations,
)
from core.tariff_schema import ExtractedTariffRule, RateBracket
from core.vessel_profile import VesselProfile, load_vessel_profile

# ---------------------------------------------------------------------------
# Test fixture: SUDESTADA @ Durban (from the take-home test spec)
# ---------------------------------------------------------------------------

_SUDESTADA_DATA: dict = {
    "vessel_metadata": {
        "name": "SUDESTADA",
        "built_year": 2010,
        "flag": "MLT - Malta",
        "classification_society": "Registro Italiano Navale",
        "call_sign": None,
    },
    "technical_specs": {
        "imo_number": None,
        "type": "Bulk Carrier",
        "dwt": 93274,
        "gross_tonnage": 51300,
        "net_tonnage": 31192,
        "loa_meters": 229.2,
        "beam_meters": 38.0,
        "moulded_depth_meters": 20.7,
        "lbp_meters": 222.0,
        "draft_sw_s_w_t": [14.9, 0.0, 0.0],
        "suez_gt": None,
        "suez_nt": 49069,
    },
    "operational_data": {
        "cargo_quantity_mt": 40000,
        "days_alongside": 3.39,
        "arrival_time": "2024-11-15T10:12:00",
        "departure_time": "2024-11-22T13:00:00",
        "activity": "Exporting Iron Ore",
        "num_operations": 2,
        "num_holds": 7,
    },
}


@pytest.fixture
def vessel() -> VesselProfile:
    return load_vessel_profile(_SUDESTADA_DATA, port="Durban")


# ---------------------------------------------------------------------------
# Light Dues — simple per-100-GT rate with ceiling
# ---------------------------------------------------------------------------

class TestLightDues:
    def test_exact_match(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="light_dues",
            measurement_basis="GT",
            rate_per_unit=117.08,
            unit_divisor=100,
            use_ceiling=True,
            num_services=1,
            extraction_confidence="HIGH",
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(60_062.04, abs=0.01)

    def test_formula_contains_rate(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="light_dues",
            measurement_basis="GT",
            rate_per_unit=117.08,
            unit_divisor=100,
            use_ceiling=True,
            num_services=1,
        )
        item = apply_rule(vessel, rule)
        assert "117.08" in item.formula_used


# ---------------------------------------------------------------------------
# Per-GT rate with minimum floor
# ---------------------------------------------------------------------------

class TestVtsDues:
    def test_durban_rate(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="vts_dues",
            measurement_basis="GT",
            rate_per_unit=0.65,
            unit_divisor=1,
            use_ceiling=False,
            minimum_fee=235.52,
            num_services=1,
            extraction_confidence="HIGH",
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(33_345.00, abs=0.01)

    def test_minimum_fee_applied(self) -> None:
        """For a very small vessel, minimum fee should kick in."""
        small_vessel = load_vessel_profile({
            "vessel_metadata": {"name": "TINY"},
            "technical_specs": {"type": "Yacht", "dwt": 50, "gross_tonnage": 100,
                                "net_tonnage": 50, "loa_meters": 20.0},
            "operational_data": {},
        }, port="Durban")
        rule = ExtractedTariffRule(
            tariff_name="vts_dues",
            measurement_basis="GT",
            rate_per_unit=0.65,
            unit_divisor=1,
            use_ceiling=False,
            minimum_fee=235.52,
            num_services=1,
        )
        item = apply_rule(small_vessel, rule)
        assert item.amount == pytest.approx(235.52, abs=0.01)


# ---------------------------------------------------------------------------
# Basic fee + per-100-GT add-on (classic pilotage shape)
# ---------------------------------------------------------------------------

class TestPilotageDues:
    def test_exact_match(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="pilotage_dues",
            measurement_basis="GT",
            basic_fee=18_608.61,
            rate_per_unit=9.72,
            unit_divisor=100,
            use_ceiling=True,
            num_services=2,
            extraction_confidence="HIGH",
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(47_189.94, abs=0.01)


# ---------------------------------------------------------------------------
# Bracket schedule (classic towage shape)
# ---------------------------------------------------------------------------

_TOWAGE_BRACKETS = [
    RateBracket(min_value=0,       max_value=2000,    base_fee=8_140.00,  rate_above_min=0.0,    rate_divisor=100),
    RateBracket(min_value=2001,    max_value=10_000,  base_fee=12_633.99, rate_above_min=268.99, rate_divisor=100),
    RateBracket(min_value=10_001,  max_value=50_000,  base_fee=38_494.51, rate_above_min=84.95,  rate_divisor=100),
    RateBracket(min_value=50_001,  max_value=100_000, base_fee=73_118.07, rate_above_min=32.24,  rate_divisor=100),
    RateBracket(min_value=100_001, max_value=None,    base_fee=93_548.13, rate_above_min=23.65,  rate_divisor=100),
]


class TestTowageDues:
    def test_exact_match(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="towage_dues",
            measurement_basis="GT",
            brackets=_TOWAGE_BRACKETS,
            num_services=2,
            extraction_confidence="HIGH",
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(147_074.38, abs=0.01)

    def test_small_vessel_first_bracket(self) -> None:
        small = load_vessel_profile({
            "vessel_metadata": {"name": "SMALL"},
            "technical_specs": {"type": "Tug", "dwt": 500, "gross_tonnage": 1000,
                                "net_tonnage": 500, "loa_meters": 30.0},
            "operational_data": {},
        }, port="Durban")
        rule = ExtractedTariffRule(
            tariff_name="towage_dues",
            measurement_basis="GT",
            brackets=_TOWAGE_BRACKETS,
            num_services=2,
        )
        item = apply_rule(small, rule)
        assert item.amount == pytest.approx(8_140.00 * 2, abs=0.01)


# ---------------------------------------------------------------------------
# Flat per-service fee (e.g. mooring line service)
# ---------------------------------------------------------------------------

class TestRunningLines:
    def test_exact_match(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="running_lines",
            measurement_basis="NONE",
            basic_fee=1_654.56,
            rate_per_unit=0,
            use_ceiling=False,
            num_services=12,
            extraction_confidence="HIGH",
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(19_854.72, abs=0.01)

    def test_flat_fee_not_scaled_by_gt(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="running_lines",
            measurement_basis="NONE",
            basic_fee=1_654.56,
            rate_per_unit=0,
            use_ceiling=False,
            num_services=12,
        )
        item = apply_rule(vessel, rule)
        assert item.amount < 100_000


# ---------------------------------------------------------------------------
# Per-100-GT rate with a time component (classic port dues shape)
# ---------------------------------------------------------------------------

class TestPortDues:
    def test_close_to_ground_truth(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="port_dues",
            measurement_basis="GT",
            rate_per_unit=192.73,
            unit_divisor=100,
            use_ceiling=True,
            time_rate_per_unit_per_day=57.79,
            num_services=1,
            extraction_confidence="HIGH",
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(199_549.22, rel=0.002)

    def test_time_component_increases_with_days(self) -> None:
        short_stay = load_vessel_profile({
            "vessel_metadata": {"name": "TEST"},
            "technical_specs": {"type": "Bulk Carrier", "dwt": 50000,
                                "gross_tonnage": 30000, "net_tonnage": 20000, "loa_meters": 200.0},
            "operational_data": {"days_alongside": 1.0},
        }, port="Durban")
        long_stay = load_vessel_profile({
            "vessel_metadata": {"name": "TEST"},
            "technical_specs": {"type": "Bulk Carrier", "dwt": 50000,
                                "gross_tonnage": 30000, "net_tonnage": 20000, "loa_meters": 200.0},
            "operational_data": {"days_alongside": 5.0},
        }, port="Durban")
        rule = ExtractedTariffRule(
            tariff_name="port_dues",
            measurement_basis="GT",
            rate_per_unit=192.73,
            unit_divisor=100,
            use_ceiling=True,
            time_rate_per_unit_per_day=57.79,
            num_services=1,
        )
        short_item = apply_rule(short_stay, rule)
        long_item = apply_rule(long_stay, rule)
        assert long_item.amount > short_item.amount


# ---------------------------------------------------------------------------
# New generality: non-GT bases (LOA, DWT, CARGO_MT)
# ---------------------------------------------------------------------------

class TestAlternativeMeasurementBases:
    def test_loa_based_rate(self, vessel: VesselProfile) -> None:
        """Some ports charge by LOA (e.g. hypothetical quay rental)."""
        rule = ExtractedTariffRule(
            tariff_name="quay_rental",
            measurement_basis="LOA",
            rate_per_unit=10.0,
            unit_divisor=1,
            num_services=1,
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(229.2 * 10.0, abs=0.01)

    def test_cargo_mt_based_rate(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="wharfage",
            measurement_basis="CARGO_MT",
            rate_per_unit=2.5,
            unit_divisor=1,
            num_services=1,
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(40_000 * 2.5, abs=0.01)

    def test_flat_fee_no_measurement(self, vessel: VesselProfile) -> None:
        rule = ExtractedTariffRule(
            tariff_name="admin_fee",
            measurement_basis="NONE",
            basic_fee=500.0,
            num_services=1,
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(500.0, abs=0.01)


# ---------------------------------------------------------------------------
# Integration: the full Transnet rule set (inline fixture) reproduces total
# ---------------------------------------------------------------------------

def _reference_transnet_rules_for_durban() -> dict[str, ExtractedTariffRule]:
    """Test-only fixture. Reproduces the Transnet published rates inline so we
    can assert the calculator computes the right total — without any of this
    ever living in the production code path.
    """
    return {
        "light_dues": ExtractedTariffRule(
            tariff_name="light_dues", measurement_basis="GT",
            rate_per_unit=117.08, unit_divisor=100, use_ceiling=True, num_services=1,
        ),
        "vts_dues": ExtractedTariffRule(
            tariff_name="vts_dues", measurement_basis="GT",
            rate_per_unit=0.65, unit_divisor=1, minimum_fee=235.52, num_services=1,
        ),
        "pilotage_dues": ExtractedTariffRule(
            tariff_name="pilotage_dues", measurement_basis="GT",
            basic_fee=18_608.61, rate_per_unit=9.72,
            unit_divisor=100, use_ceiling=True, num_services=2,
        ),
        "towage_dues": ExtractedTariffRule(
            tariff_name="towage_dues", measurement_basis="GT",
            brackets=_TOWAGE_BRACKETS, num_services=2,
        ),
        "running_lines": ExtractedTariffRule(
            tariff_name="running_lines", measurement_basis="NONE",
            basic_fee=1_654.56, num_services=12,
        ),
        "port_dues": ExtractedTariffRule(
            tariff_name="port_dues", measurement_basis="GT",
            rate_per_unit=192.73, unit_divisor=100, use_ceiling=True,
            time_rate_per_unit_per_day=57.79, num_services=1,
        ),
    }


class TestRunAllCalculations:
    def test_total_close_to_ground_truth(self, vessel: VesselProfile) -> None:
        """Full calculation with inline reference rules should match the
        published total within 0.2% — proving the calculator is correct."""
        rules = _reference_transnet_rules_for_durban()
        result = run_all_calculations(vessel, rules)
        assert result.total == pytest.approx(506_830.83, rel=0.002)
        assert len(result.line_items) == 6

    def test_error_handling_unknown_basis(self, vessel: VesselProfile) -> None:
        """An invalid unit_divisor should surface as an error line item, not crash."""
        rules = {"bad": ExtractedTariffRule(
            tariff_name="bad",
            measurement_basis="GT",
            rate_per_unit=10.0,
            unit_divisor=0,  # invalid — should raise inside apply_rule
        )}
        result = run_all_calculations(vessel, rules)
        assert len(result.line_items) == 1
        assert result.line_items[0].error != ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_gt_vessel(self) -> None:
        zero_gt = load_vessel_profile({
            "vessel_metadata": {"name": "ZERO"},
            "technical_specs": {"type": "Barge", "dwt": 0, "gross_tonnage": 0,
                                "net_tonnage": 0, "loa_meters": 10.0},
            "operational_data": {"days_alongside": 1.0},
        }, port="Durban")
        rule = ExtractedTariffRule(
            tariff_name="light_dues",
            measurement_basis="GT",
            rate_per_unit=117.08,
            unit_divisor=100,
            use_ceiling=True,
            num_services=1,
        )
        item = apply_rule(zero_gt, rule)
        assert item.amount == 0.0

    def test_ceiling_rounding(self) -> None:
        vessel = load_vessel_profile({
            "vessel_metadata": {"name": "TEST"},
            "technical_specs": {"type": "Cargo", "dwt": 1000, "gross_tonnage": 51300,
                                "net_tonnage": 30000, "loa_meters": 200.0},
            "operational_data": {},
        }, port="Durban")
        rule = ExtractedTariffRule(
            tariff_name="test",
            measurement_basis="GT",
            rate_per_unit=1.0,
            unit_divisor=100,
            use_ceiling=True,
            num_services=1,
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(math.ceil(51300 / 100) * 1.0)

    def test_no_ceiling_rounding(self) -> None:
        vessel = load_vessel_profile({
            "vessel_metadata": {"name": "TEST"},
            "technical_specs": {"type": "Cargo", "dwt": 1000, "gross_tonnage": 51301,
                                "net_tonnage": 30000, "loa_meters": 200.0},
            "operational_data": {},
        }, port="Durban")
        rule = ExtractedTariffRule(
            tariff_name="test",
            measurement_basis="GT",
            rate_per_unit=1.0,
            unit_divisor=100,
            use_ceiling=False,
            num_services=1,
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(513.01)

    def test_maximum_fee_cap_applied(self) -> None:
        vessel = load_vessel_profile({
            "vessel_metadata": {"name": "BIG"},
            "technical_specs": {"type": "Bulk Carrier", "dwt": 200000,
                                "gross_tonnage": 200000, "net_tonnage": 150000, "loa_meters": 350.0},
            "operational_data": {},
        }, port="Durban")
        rule = ExtractedTariffRule(
            tariff_name="capped",
            measurement_basis="GT",
            rate_per_unit=10.0,
            unit_divisor=1,
            maximum_fee=50_000.0,
            num_services=1,
        )
        item = apply_rule(vessel, rule)
        assert item.amount == pytest.approx(50_000.0, abs=0.01)
