"""Unit tests for the tariff schema models."""

from __future__ import annotations

import json

import pytest

from core.tariff_schema import (
    DiscoveredTariff,
    DiscoveredTariffSet,
    ExtractedTariffRule,
    RateBracket,
)


class TestRateBracket:
    def test_basic_construction(self) -> None:
        b = RateBracket(
            min_value=0, max_value=2000, base_fee=8140.0, rate_above_min=0.0, rate_divisor=100
        )
        assert b.min_value == 0
        assert b.max_value == 2000
        assert b.base_fee == 8140.0
        assert b.rate_divisor == 100

    def test_open_ended_bracket(self) -> None:
        b = RateBracket(
            min_value=100_001, max_value=None, base_fee=93_548.13, rate_above_min=23.65
        )
        assert b.max_value is None

    def test_serialization_round_trip(self) -> None:
        b = RateBracket(
            min_value=2001, max_value=10_000, base_fee=12_633.99, rate_above_min=268.99
        )
        data = json.loads(b.model_dump_json())
        b2 = RateBracket.model_validate(data)
        assert b == b2


class TestExtractedTariffRule:
    def test_defaults(self) -> None:
        rule = ExtractedTariffRule(tariff_name="test")
        assert rule.measurement_basis == "NONE"
        assert rule.basic_fee == 0.0
        assert rule.rate_per_unit == 0.0
        assert rule.unit_divisor == 1.0
        assert rule.use_ceiling is False
        assert rule.minimum_fee == 0.0
        assert rule.maximum_fee is None
        assert rule.time_rate_per_unit_per_day == 0.0
        assert rule.num_services == 1.0
        assert rule.brackets == []
        assert rule.extraction_confidence == "MEDIUM"
        assert rule.currency == ""
        assert rule.applicability_notes == ""

    def test_light_dues_rule(self) -> None:
        rule = ExtractedTariffRule(
            tariff_name="light_dues",
            section_reference="Section 1.1",
            measurement_basis="GT",
            rate_per_unit=117.08,
            unit_divisor=100,
            use_ceiling=True,
            num_services=1,
            extraction_confidence="HIGH",
        )
        assert rule.tariff_name == "light_dues"
        assert rule.extraction_confidence == "HIGH"

    def test_bracket_tariff(self) -> None:
        brackets = [
            RateBracket(min_value=0, max_value=2000, base_fee=8140.0),
            RateBracket(min_value=2001, max_value=None, base_fee=12000.0, rate_above_min=100.0),
        ]
        rule = ExtractedTariffRule(
            tariff_name="towage_dues",
            measurement_basis="GT",
            brackets=brackets,
            num_services=2,
        )
        assert len(rule.brackets) == 2
        assert rule.brackets[1].max_value is None

    def test_confidence_literal(self) -> None:
        with pytest.raises(Exception):
            ExtractedTariffRule(tariff_name="test", extraction_confidence="INVALID")  # type: ignore[arg-type]

    def test_measurement_basis_literal(self) -> None:
        with pytest.raises(Exception):
            ExtractedTariffRule(tariff_name="test", measurement_basis="INVALID")  # type: ignore[arg-type]

    def test_all_new_measurement_bases_valid(self) -> None:
        for basis in ("GT", "NT", "DWT", "LOA", "BEAM", "DRAFT", "CARGO_MT", "NONE"):
            rule = ExtractedTariffRule(tariff_name="test", measurement_basis=basis)  # type: ignore[arg-type]
            assert rule.measurement_basis == basis

    def test_tariff_name_is_freeform(self) -> None:
        """tariff_name must accept arbitrary strings — it is NOT a fixed enum."""
        for name in ("quay_rental", "wharfage", "conservancy_charge", "some_new_fee"):
            rule = ExtractedTariffRule(tariff_name=name)
            assert rule.tariff_name == name

    def test_serialization_round_trip(self) -> None:
        rule = ExtractedTariffRule(
            tariff_name="pilotage_dues",
            measurement_basis="GT",
            basic_fee=18_608.61,
            rate_per_unit=9.72,
            unit_divisor=100,
            brackets=[RateBracket(min_value=0, max_value=5000, base_fee=100.0)],
            extraction_confidence="HIGH",
            currency="ZAR",
            applicability_notes="Non-coastal vessels",
            notes="Test note",
        )
        data = json.loads(rule.model_dump_json())
        rule2 = ExtractedTariffRule.model_validate(data)
        assert rule == rule2

    def test_model_copy_update(self) -> None:
        rule = ExtractedTariffRule(tariff_name="test", extraction_confidence="HIGH")
        downgraded = rule.model_copy(update={"extraction_confidence": "LOW"})
        assert downgraded.extraction_confidence == "LOW"
        assert rule.extraction_confidence == "HIGH"


class TestDiscoveredTariff:
    def test_defaults(self) -> None:
        t = DiscoveredTariff(name="light_dues")
        assert t.name == "light_dues"
        assert t.section_reference == ""
        assert t.retrieval_keywords == []

    def test_full_construction(self) -> None:
        t = DiscoveredTariff(
            name="wharfage",
            section_reference="Section 7.1",
            description="Per-tonne charge on cargo handled",
            applies_to="Cargo vessels only",
            retrieval_keywords=["wharfage", "cargo", "per tonne", "7.1"],
        )
        assert len(t.retrieval_keywords) == 4

    def test_set_wrapper(self) -> None:
        s = DiscoveredTariffSet(
            currency="USD",
            tariffs=[DiscoveredTariff(name="a"), DiscoveredTariff(name="b")],
        )
        assert len(s.tariffs) == 2
        assert s.currency == "USD"
