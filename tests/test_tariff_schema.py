"""Unit tests for the tariff schema models."""

from __future__ import annotations

import json

import pytest

from core.tariff_schema import ExtractedTariffRule, RateBracket


class TestRateBracket:
    def test_basic_construction(self) -> None:
        b = RateBracket(min_gt=0, max_gt=2000, base_fee=8140.0, per_100_gt_above_min=0.0)
        assert b.min_gt == 0
        assert b.max_gt == 2000
        assert b.base_fee == 8140.0

    def test_open_ended_bracket(self) -> None:
        b = RateBracket(min_gt=100_001, max_gt=None, base_fee=93_548.13, per_100_gt_above_min=23.65)
        assert b.max_gt is None

    def test_serialization_round_trip(self) -> None:
        b = RateBracket(min_gt=2001, max_gt=10_000, base_fee=12_633.99, per_100_gt_above_min=268.99)
        data = json.loads(b.model_dump_json())
        b2 = RateBracket.model_validate(data)
        assert b == b2


class TestExtractedTariffRule:
    def test_defaults(self) -> None:
        rule = ExtractedTariffRule(tariff_type="test")
        assert rule.measurement_basis == "GT"
        assert rule.basic_fee == 0.0
        assert rule.rate_per_unit == 0.0
        assert rule.unit_divisor == 1
        assert rule.use_ceiling is True
        assert rule.minimum_fee == 0.0
        assert rule.time_rate_per_unit_per_day == 0.0
        assert rule.num_services == 1
        assert rule.brackets == []
        assert rule.extraction_confidence == "MEDIUM"

    def test_light_dues_rule(self) -> None:
        rule = ExtractedTariffRule(
            tariff_type="light_dues",
            section_reference="Section 1.1",
            rate_per_unit=117.08,
            unit_divisor=100,
            use_ceiling=True,
            num_services=1,
            extraction_confidence="HIGH",
        )
        assert rule.tariff_type == "light_dues"
        assert rule.extraction_confidence == "HIGH"

    def test_bracket_tariff(self) -> None:
        brackets = [
            RateBracket(min_gt=0, max_gt=2000, base_fee=8140.0),
            RateBracket(min_gt=2001, max_gt=None, base_fee=12000.0, per_100_gt_above_min=100.0),
        ]
        rule = ExtractedTariffRule(
            tariff_type="towage_dues",
            brackets=brackets,
            num_services=2,
        )
        assert len(rule.brackets) == 2
        assert rule.brackets[1].max_gt is None

    def test_confidence_literal(self) -> None:
        """Only HIGH/MEDIUM/LOW are valid."""
        with pytest.raises(Exception):
            ExtractedTariffRule(tariff_type="test", extraction_confidence="INVALID")  # type: ignore[arg-type]

    def test_measurement_basis_literal(self) -> None:
        """Only GT/NT/LOA/DWT are valid."""
        with pytest.raises(Exception):
            ExtractedTariffRule(tariff_type="test", measurement_basis="INVALID")  # type: ignore[arg-type]

    def test_serialization_round_trip(self) -> None:
        rule = ExtractedTariffRule(
            tariff_type="pilotage_dues",
            basic_fee=18_608.61,
            rate_per_unit=9.72,
            unit_divisor=100,
            brackets=[RateBracket(min_gt=0, max_gt=5000, base_fee=100.0)],
            extraction_confidence="HIGH",
            notes="Test note",
        )
        data = json.loads(rule.model_dump_json())
        rule2 = ExtractedTariffRule.model_validate(data)
        assert rule == rule2

    def test_model_copy_update(self) -> None:
        """model_copy(update=...) should produce a new rule with changed fields."""
        rule = ExtractedTariffRule(
            tariff_type="test",
            extraction_confidence="HIGH",
        )
        downgraded = rule.model_copy(update={"extraction_confidence": "LOW"})
        assert downgraded.extraction_confidence == "LOW"
        assert rule.extraction_confidence == "HIGH"  # original unchanged
