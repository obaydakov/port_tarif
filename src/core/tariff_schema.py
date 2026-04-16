from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RateBracket(BaseModel):
    """One row of a step-rate bracket schedule (e.g. towage by GT range)."""

    min_gt: float
    max_gt: float | None = None
    base_fee: float
    per_100_gt_above_min: float = 0.0


class ExtractedTariffRule(BaseModel):
    """
    Portable, schema-driven representation of a single tariff calculation rule.

    The generic calculator (calculator.py) operates entirely from this schema —
    no tariff-specific code required.  Adapting to a new port or document update
    only requires fresh LLM extraction; zero code changes.

    Computation model
    -----------------
    If brackets is non-empty (step-rate tariff, e.g. towage):
        fee_per_service = bracket_fee(basis)

    Otherwise:
        units = ceil(basis / unit_divisor)   if use_ceiling
              | basis / unit_divisor          otherwise

        fee_per_service = max(
            basic_fee + units * rate_per_unit,
            minimum_fee
        )
        fee_per_service += units * time_rate_per_unit_per_day * days_alongside

    total = fee_per_service * num_services
    """

    tariff_type: str
    section_reference: str = ""

    # Vessel measurement that drives this tariff
    measurement_basis: Literal["GT", "NT", "LOA", "DWT"] = "GT"

    # Rate components
    basic_fee: float = 0.0                   # Fixed base added per service (e.g. pilotage basic)
    rate_per_unit: float = 0.0               # Per-unit rate
    unit_divisor: int = 1                    # e.g. 100 → "per 100 GT"
    use_ceiling: bool = True                 # ceil(basis/divisor) vs basis/divisor
    minimum_fee: float = 0.0                 # Floor applied after basic + per-unit

    # Time component — incremental rate per unit per 24 h (e.g. port dues)
    time_rate_per_unit_per_day: float = 0.0

    # How many times the per-service fee applies for a complete port call
    # (e.g. 2 for arrival + departure; higher for running lines)
    num_services: int = 1

    # Step-rate schedule — if populated, overrides all simple-rate fields above
    brackets: list[RateBracket] = Field(default_factory=list)

    # Extraction metadata
    extraction_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    notes: str = ""
