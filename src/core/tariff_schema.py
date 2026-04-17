from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# The schema below is intentionally shape-neutral.  It does NOT encode any
# particular port's tariff patterns (pilotage = basic + per-100-GT, running
# lines = flat-fee-times-six, etc.).  It provides a set of additive building
# blocks and lets the LLM choose which ones the document actually uses.
#
# A tariff rule is always computed as:
#
#     units = basis / unit_divisor           (optionally ceiled)
#
#     if brackets is non-empty:
#         fee_per_service = bracket_lookup(basis)
#     else:
#         fee_per_service = basic_fee + units * rate_per_unit
#         fee_per_service = clamp(fee_per_service, minimum_fee, maximum_fee)
#         fee_per_service += units * time_rate_per_unit_per_day * days_alongside
#
#     total = fee_per_service * num_services
#
# Measurement basis, divisor, ceiling, minimum/maximum, time component,
# brackets, and services multiplier are all independent knobs — every tariff
# we have seen across South African, Dutch, and US port tariff books fits
# this shape by turning knobs on or off.  If a document describes a rule
# that cannot be expressed this way, the LLM is instructed to return
# extraction_confidence = "LOW" with an explanation in notes.
# ---------------------------------------------------------------------------


MeasurementBasis = Literal[
    "GT",        # Gross tonnage
    "NT",        # Net tonnage
    "DWT",       # Deadweight tonnage
    "LOA",       # Length overall (metres)
    "BEAM",      # Beam (metres)
    "DRAFT",     # Draft (metres)
    "CARGO_MT",  # Cargo quantity (metric tonnes)
    "NONE",      # Flat fee — no measurement dependency
]

ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


class RateBracket(BaseModel):
    """One row of a step-rate bracket schedule.

    Example (towage by GT): ``RateBracket(min_value=10001, max_value=50000,
    base_fee=38494.51, rate_above_min=84.95, rate_divisor=100)`` means
    "vessels between 10 001 and 50 000 GT pay R38 494.51 + R84.95 per 100 GT
    above 10 001 GT".

    The bracket fields are deliberately named generically so the schedule can
    step on any measurement — GT, LOA, DWT, cargo tonnes, etc.  The enclosing
    ``ExtractedTariffRule.measurement_basis`` says which one.
    """

    min_value: float = Field(
        description="Lower bound of the measurement range for this bracket (inclusive).",
    )
    max_value: float | None = Field(
        default=None,
        description="Upper bound of the measurement range (inclusive). Leave null for the open-ended top bracket.",
    )
    base_fee: float = Field(
        default=0.0,
        description="Flat fee charged when the vessel's measurement falls in this bracket.",
    )
    rate_above_min: float = Field(
        default=0.0,
        description="Optional per-unit rate applied to the measurement above min_value.",
    )
    rate_divisor: float = Field(
        default=1.0,
        description="Divisor for rate_above_min (e.g. 100 for 'per 100 GT above'). 1 = per unit.",
    )


class ExtractedTariffRule(BaseModel):
    """Portable, schema-driven representation of a single tariff calculation rule.

    The generic calculator operates entirely from this schema.  No tariff
    taxonomy, section numbers, or port-specific rates live in code.  Adapting
    the system to a new port tariff document requires only re-running the
    discovery + extraction stages against the new PDF.
    """

    # ------------------------------------------------------------------
    # Identification (all free-form strings — not from a fixed taxonomy)
    # ------------------------------------------------------------------
    tariff_name: str = Field(
        description=(
            "Canonical name of this tariff, extracted verbatim from the "
            "document (e.g. 'light_dues', 'wharfage', 'conservancy_charge'). "
            "No fixed list — whatever the document calls it."
        ),
    )
    section_reference: str = Field(
        default="",
        description="Section or clause reference from the source document (e.g. '3.3.2', 'Schedule B-II').",
    )
    description: str = Field(
        default="",
        description="One-sentence description of what this tariff charges for.",
    )
    applicability_notes: str = Field(
        default="",
        description=(
            "Conditions that narrow when this rule applies "
            "(e.g. 'non-coastal vessels', 'vessels handling cargo', 'Durban and Saldanha only')."
        ),
    )
    currency: str = Field(
        default="",
        description="Currency code or symbol found in the document (e.g. 'ZAR', 'USD', 'EUR').",
    )

    # ------------------------------------------------------------------
    # Calculation model
    # ------------------------------------------------------------------
    measurement_basis: MeasurementBasis = Field(
        default="NONE",
        description=(
            "Which vessel measurement drives the per-unit charge. "
            "Use 'NONE' for flat fees that do not depend on a vessel measurement."
        ),
    )

    # Simple rate components (combine additively per service)
    basic_fee: float = Field(default=0.0, description="Fixed amount added per service.")
    rate_per_unit: float = Field(default=0.0, description="Per-unit rate applied to measurement_basis.")
    unit_divisor: float = Field(default=1.0, description="Divisor for the per-unit rate (1 = per unit, 100 = per 100 units).")
    use_ceiling: bool = Field(
        default=False,
        description="True if the document says 'per N units or part thereof'. False for strictly proportional rates.",
    )
    minimum_fee: float = Field(default=0.0, description="Lower bound on (basic_fee + rate_per_unit * units).")
    maximum_fee: float | None = Field(
        default=None, description="Upper bound on (basic_fee + rate_per_unit * units). Null = no cap."
    )

    # Time component — incremental rate per unit per 24 h (e.g. port dues)
    time_rate_per_unit_per_day: float = Field(
        default=0.0,
        description="Per-unit rate added per day alongside. 0 for one-off charges.",
    )

    # Step-rate schedule (if non-empty, overrides basic_fee + rate_per_unit)
    brackets: list[RateBracket] = Field(
        default_factory=list,
        description="Step-rate schedule. If populated, overrides basic_fee + rate_per_unit.",
    )

    # ------------------------------------------------------------------
    # Multiplier
    # ------------------------------------------------------------------
    num_services: float = Field(
        default=1.0,
        description=(
            "How many times the per-service fee is charged for a single port "
            "call. Not a pre-set constant — derived from the document and the "
            "vessel's operational context (arrival + departure, lines per "
            "operation, etc.)."
        ),
    )
    services_basis: str = Field(
        default="",
        description="Short free-text explaining what num_services counts and why.",
    )

    # ------------------------------------------------------------------
    # Extraction metadata
    # ------------------------------------------------------------------
    extraction_confidence: ConfidenceLevel = Field(
        default="MEDIUM",
        description=(
            "HIGH   = every rate in the output is quoted directly from the source passage. "
            "MEDIUM = rates are present but some applicability condition is ambiguous. "
            "LOW    = the passage does not contain rates sufficient to compute this tariff."
        ),
    )
    notes: str = Field(
        default="",
        description="Free-text notes, caveats, or extraction warnings.",
    )


class DiscoveredTariff(BaseModel):
    """Output of the discovery stage — one entry per chargeable tariff the
    document defines.  Used to drive the targeted per-tariff extraction pass.
    """

    name: str = Field(description="Canonical name in snake_case (e.g. 'light_dues', 'wharfage').")
    section_reference: str = Field(default="", description="Section number or heading from the document.")
    description: str = Field(default="", description="One-sentence description.")
    applies_to: str = Field(default="", description="Applicability conditions, if any.")
    retrieval_keywords: list[str] = Field(
        default_factory=list,
        description="3-5 keywords/phrases useful for vector-searching the rule's exact rates.",
    )


class DiscoveredTariffSet(BaseModel):
    """Wrapper used for structured LLM output in the discovery stage."""

    currency: str = Field(default="", description="Primary currency of the document (code or symbol).")
    tariffs: list[DiscoveredTariff] = Field(default_factory=list)
