from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from beartype import beartype

from core.tariff_schema import ExtractedTariffRule, MeasurementBasis, RateBracket
from core.vessel_profile import VesselProfile


class CalculationError(Exception):
    """Raised when a tariff rule cannot be applied to a vessel profile."""

    def __init__(self, tariff_name: str, message: str) -> None:
        self.tariff_name = tariff_name
        super().__init__(f"{tariff_name}: {message}")


@dataclass
class TariffLineItem:
    tariff_name: str
    description: str
    formula_used: str
    amount: float
    currency: str = ""
    confidence: str = ""  # HIGH / MEDIUM / LOW
    breakdown: list[str] = field(default_factory=list)
    notes: str = ""
    error: str = ""  # non-empty when calculation failed


@dataclass
class CalculationResult:
    vessel_name: str
    port: str
    currency: str = ""
    line_items: list[TariffLineItem] = field(default_factory=list)
    total: float = 0.0

    def add(self, item: TariffLineItem) -> None:
        self.line_items.append(item)
        self.total = sum(i.amount for i in self.line_items)
        if not self.currency and item.currency:
            self.currency = item.currency

    def to_dict(self) -> dict[str, Any]:
        return {
            "vessel": self.vessel_name,
            "port": self.port,
            "currency": self.currency,
            "total": round(self.total, 2),
            "line_items": [
                {
                    "tariff": item.tariff_name,
                    "description": item.description,
                    "formula": item.formula_used,
                    "amount": round(item.amount, 2),
                    "currency": item.currency,
                    "confidence": item.confidence,
                    "breakdown": item.breakdown,
                    "notes": item.notes,
                    **({"error": item.error} if item.error else {}),
                }
                for item in self.line_items
            ],
        }


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _get_basis(vessel: VesselProfile, measurement: MeasurementBasis) -> float:
    """Read the requested measurement from the vessel profile.

    Returns 0.0 when the measurement is NONE (flat-fee tariffs) or when the
    underlying value is absent on the profile.
    """
    match measurement:
        case "GT":
            return float(vessel.gt)
        case "NT":
            return float(vessel.nt)
        case "DWT":
            return float(vessel.technical_specs.dwt)
        case "LOA":
            return float(vessel.loa)
        case "BEAM":
            beam = vessel.technical_specs.beam_meters
            return float(beam) if beam is not None else 0.0
        case "DRAFT":
            drafts = vessel.technical_specs.draft_sw_s_w_t or []
            return float(drafts[0]) if drafts else 0.0
        case "CARGO_MT":
            cargo = vessel.operational_data.cargo_quantity_mt
            return float(cargo) if cargo is not None else 0.0
        case "NONE":
            return 0.0
    # Exhaustive match above — defensive fallback keeps mypy/pyright happy
    raise CalculationError("_get_basis", f"Unknown measurement basis: {measurement!r}")


def _compute_units(basis: float, unit_divisor: float, use_ceiling: bool) -> float:
    if unit_divisor <= 0:
        raise CalculationError("units", f"unit_divisor must be > 0, got {unit_divisor}")
    ratio = basis / unit_divisor
    return float(math.ceil(ratio)) if use_ceiling else ratio


def _apply_brackets(basis: float, brackets: list[RateBracket]) -> tuple[float, str]:
    """Look up fee from a step-rate bracket schedule.

    Returns (fee, human-readable explanation).
    """
    for bracket in brackets:
        if bracket.max_value is None or basis <= bracket.max_value:
            above = max(0.0, basis - bracket.min_value)
            units_above = (
                math.ceil(above / bracket.rate_divisor) if bracket.rate_divisor > 0 else 0.0
            )
            fee = bracket.base_fee + units_above * bracket.rate_above_min
            expl = (
                f"Bracket [{bracket.min_value:,.0f}..{bracket.max_value if bracket.max_value is not None else '∞'}]"
                f": {bracket.base_fee:,.2f} + ceil({above:,.2f}/{bracket.rate_divisor}) × {bracket.rate_above_min:,.4f}"
                f" = {fee:,.2f}"
            )
            return fee, expl

    # basis exceeds all defined brackets — use the last one (open-ended fallback)
    last = brackets[-1]
    above = max(0.0, basis - last.min_value)
    units_above = (
        math.ceil(above / last.rate_divisor) if last.rate_divisor > 0 else 0.0
    )
    fee = last.base_fee + units_above * last.rate_above_min
    return fee, (
        f"Above all brackets → using last row: "
        f"{last.base_fee:,.2f} + ceil({above:,.2f}/{last.rate_divisor}) × {last.rate_above_min:,.4f}"
        f" = {fee:,.2f}"
    )


def _apply_bounds(fee: float, minimum_fee: float, maximum_fee: float | None) -> tuple[float, str]:
    """Apply optional min/max bounds, returning the new fee and a short note."""
    note = ""
    if minimum_fee and fee < minimum_fee:
        note = f"Minimum applied ({minimum_fee:,.2f})"
        fee = minimum_fee
    if maximum_fee is not None and fee > maximum_fee:
        note = f"Maximum applied ({maximum_fee:,.2f})"
        fee = maximum_fee
    return fee, note


def _build_formula(rule: ExtractedTariffRule, units: float, days: float) -> str:
    """Construct a human-readable formula string from a rule."""
    if rule.brackets:
        return f"Bracket({rule.measurement_basis}) × {rule.num_services:g} services"

    parts: list[str] = []
    if rule.basic_fee:
        parts.append(f"{rule.basic_fee:,.2f}")
    if rule.rate_per_unit:
        parts.append(f"{units:g} × {rule.rate_per_unit}")
    base_expr = " + ".join(parts) if parts else "0"
    if rule.minimum_fee:
        base_expr = f"max({base_expr}, {rule.minimum_fee:,.2f})"
    if rule.maximum_fee is not None:
        base_expr = f"min({base_expr}, {rule.maximum_fee:,.2f})"
    if rule.time_rate_per_unit_per_day:
        base_expr += (
            f" + {units:g} × {rule.time_rate_per_unit_per_day}/day × {days:.4f}d"
        )
    return f"({base_expr}) × {rule.num_services:g}"


# ---------------------------------------------------------------------------
# Generic tariff application engine
# ---------------------------------------------------------------------------

@beartype
def apply_rule(vessel: VesselProfile, rule: ExtractedTariffRule) -> TariffLineItem:
    """Apply an ExtractedTariffRule to a vessel profile and return a TariffLineItem.

    This is the sole calculation engine.  All rates and formulas come from
    the rule schema — no tariff-specific logic lives here.  The same function
    handles simple per-unit tariffs, bracket schedules, flat fees, and
    time-accruing charges identically.
    """
    basis = _get_basis(vessel, rule.measurement_basis)
    breakdown: list[str] = [
        f"{rule.measurement_basis} = {basis:,.2f}"
        if rule.measurement_basis != "NONE"
        else "No measurement basis (flat fee)"
    ]
    bound_note = ""

    if rule.brackets:
        fee_per_service, explanation = _apply_brackets(basis, rule.brackets)
        breakdown.append(explanation)
        # Bracket schedules may still be bounded by an explicit minimum/maximum
        fee_per_service, bound_note = _apply_bounds(
            fee_per_service, rule.minimum_fee, rule.maximum_fee
        )
        if bound_note:
            breakdown.append(bound_note)
    else:
        units = (
            _compute_units(basis, rule.unit_divisor, rule.use_ceiling)
            if rule.measurement_basis != "NONE"
            else 0.0
        )
        if rule.measurement_basis != "NONE":
            ceil_label = (
                f"ceil({basis:,.2f}/{rule.unit_divisor})"
                if rule.use_ceiling
                else f"{basis:,.2f}/{rule.unit_divisor}"
            )
            breakdown.append(f"Units = {ceil_label} = {units:,.4f}")

        per_unit_component = units * rule.rate_per_unit
        fee_before_bounds = rule.basic_fee + per_unit_component
        fee_per_service, bound_note = _apply_bounds(
            fee_before_bounds, rule.minimum_fee, rule.maximum_fee
        )

        if rule.basic_fee:
            breakdown.append(f"Basic fee = {rule.basic_fee:,.2f}")
        if rule.rate_per_unit:
            breakdown.append(
                f"Per-unit = {units:,.4f} × {rule.rate_per_unit} = {per_unit_component:,.2f}"
            )
        if bound_note:
            breakdown.append(bound_note)
        breakdown.append(f"Fee/service (before time component) = {fee_per_service:,.2f}")

        if rule.time_rate_per_unit_per_day:
            days = vessel.days_alongside_value
            time_component = units * rule.time_rate_per_unit_per_day * days
            breakdown.append(
                f"Time = {units:,.4f} × {rule.time_rate_per_unit_per_day}/day × {days:.4f}d "
                f"= {time_component:,.2f}"
            )
            fee_per_service += time_component

    total = round(fee_per_service * rule.num_services, 2)
    breakdown.append(
        f"Total = {fee_per_service:,.2f} × {rule.num_services:g} = {total:,.2f}"
    )

    formula = _build_formula(rule, _get_basis(vessel, rule.measurement_basis) / max(rule.unit_divisor, 1e-9), vessel.days_alongside_value)

    section = f" ({rule.section_reference})" if rule.section_reference else ""
    label = rule.tariff_name.replace("_", " ").title()

    return TariffLineItem(
        tariff_name=rule.tariff_name,
        description=f"{label}{section}",
        formula_used=formula,
        amount=total,
        currency=rule.currency,
        confidence=rule.extraction_confidence,
        breakdown=breakdown,
        notes=rule.notes,
    )


@beartype
def run_all_calculations(
    vessel: VesselProfile,
    rules: dict[str, ExtractedTariffRule],
) -> CalculationResult:
    """Apply a set of extracted tariff rules to a vessel profile.

    Individual tariff failures are recorded as error line items (amount=0)
    rather than silently dropped, so the caller can see exactly what failed.
    """
    result = CalculationResult(
        vessel_name=vessel.vessel_metadata.name,
        port=vessel.port,
    )
    for name, rule in rules.items():
        try:
            result.add(apply_rule(vessel, rule))
        except Exception as e:
            print(f"[calculator] ERROR calculating {name}: {e}")
            label = name.replace("_", " ").title()
            result.add(TariffLineItem(
                tariff_name=name,
                description=f"{label} (FAILED)",
                formula_used="N/A",
                amount=0.0,
                currency=rule.currency,
                confidence=rule.extraction_confidence,
                breakdown=[f"Error: {e}"],
                notes=rule.notes,
                error=str(e),
            ))
    return result
