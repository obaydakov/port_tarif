from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from beartype import beartype

from core.tariff_schema import ExtractedTariffRule, RateBracket
from core.vessel_profile import VesselProfile


class CalculationError(Exception):
    """Raised when a tariff rule cannot be applied to a vessel profile."""

    def __init__(self, tariff_type: str, message: str) -> None:
        self.tariff_type = tariff_type
        super().__init__(f"{tariff_type}: {message}")


@dataclass
class TariffLineItem:
    tariff_type: str
    description: str
    formula_used: str
    amount_zar: float
    breakdown: list[str] = field(default_factory=list)
    notes: str = ""
    error: str = ""  # non-empty when calculation failed


@dataclass
class CalculationResult:
    vessel_name: str
    port: str
    line_items: list[TariffLineItem] = field(default_factory=list)
    total_zar: float = 0.0
    currency: str = "ZAR"

    def add(self, item: TariffLineItem) -> None:
        self.line_items.append(item)
        self.total_zar = sum(i.amount_zar for i in self.line_items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vessel": self.vessel_name,
            "port": self.port,
            "currency": self.currency,
            "total": round(self.total_zar, 2),
            "line_items": [
                {
                    "tariff": item.tariff_type,
                    "description": item.description,
                    "formula": item.formula_used,
                    "amount": round(item.amount_zar, 2),
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

def _get_basis(vessel: VesselProfile, measurement: str) -> float:
    match measurement:
        case "GT":
            return vessel.gt
        case "NT":
            return vessel.nt
        case "LOA":
            return vessel.loa
        case "DWT":
            return vessel.technical_specs.dwt
        case _:
            raise ValueError(f"Unknown measurement_basis: {measurement!r}")


def _compute_units(basis: float, unit_divisor: int, use_ceiling: bool) -> float:
    ratio = basis / unit_divisor
    return float(math.ceil(ratio)) if use_ceiling else ratio


def _apply_brackets(basis: float, brackets: list[RateBracket]) -> float:
    """Look up fee from a step-rate bracket schedule."""
    for bracket in brackets:
        if bracket.max_gt is None or basis <= bracket.max_gt:
            above = max(0.0, basis - bracket.min_gt)
            units_above = math.ceil(above / 100)
            return bracket.base_fee + units_above * bracket.per_100_gt_above_min
    # basis exceeds all defined brackets — use the last one (open-ended fallback)
    last = brackets[-1]
    above = max(0.0, basis - last.min_gt)
    return last.base_fee + math.ceil(above / 100) * last.per_100_gt_above_min


def _build_formula(rule: ExtractedTariffRule, units: float, days: float) -> str:
    """Construct a human-readable formula string from a rule."""
    parts: list[str] = []
    if rule.basic_fee:
        parts.append(f"R{rule.basic_fee:,.2f}")
    if rule.rate_per_unit:
        parts.append(f"{units:,.0f} × R{rule.rate_per_unit}")
    base_expr = " + ".join(parts) if parts else "0"
    if rule.minimum_fee:
        base_expr = f"max({base_expr}, R{rule.minimum_fee:,.2f})"
    if rule.time_rate_per_unit_per_day:
        base_expr += f" + {units:,.0f} × R{rule.time_rate_per_unit_per_day}/day × {days:.4f}d"
    return f"({base_expr}) × {rule.num_services}"


# ---------------------------------------------------------------------------
# Generic tariff application engine
# ---------------------------------------------------------------------------

@beartype
def apply_rule(vessel: VesselProfile, rule: ExtractedTariffRule) -> TariffLineItem:
    """
    Apply an ExtractedTariffRule to a vessel profile and return a TariffLineItem.

    This is the sole calculation engine.  All rates and formulas come from the
    rule schema — no tariff-specific logic lives here.  The same function handles
    LLM-extracted rules and fallback Transnet rules identically.
    """
    basis = _get_basis(vessel, rule.measurement_basis)
    breakdown: list[str] = [f"{rule.measurement_basis} = {basis:,.2f}"]

    if rule.brackets:
        fee_per_service = _apply_brackets(basis, rule.brackets)
        breakdown.append(
            f"Bracket lookup ({basis:,.0f} {rule.measurement_basis}) "
            f"-> R{fee_per_service:,.2f}/service"
        )
        formula = f"Bracket({rule.measurement_basis}) × {rule.num_services} services"
    else:
        units = _compute_units(basis, rule.unit_divisor, rule.use_ceiling)
        ceil_label = (
            f"ceil({basis:,.0f}/{rule.unit_divisor})"
            if rule.use_ceiling
            else f"{basis:,.0f}/{rule.unit_divisor}"
        )
        breakdown.append(f"Units = {ceil_label} = {units:,.2f}")

        per_unit_component = units * rule.rate_per_unit
        fee_before_floor = rule.basic_fee + per_unit_component
        fee_per_service = max(fee_before_floor, rule.minimum_fee)

        if rule.basic_fee:
            breakdown.append(f"Basic fee = R{rule.basic_fee:,.2f}")
        if rule.rate_per_unit:
            breakdown.append(
                f"Per-unit = {units:,.2f} × R{rule.rate_per_unit} = R{per_unit_component:,.2f}"
            )
        if rule.minimum_fee and fee_before_floor < rule.minimum_fee:
            breakdown.append(f"Minimum applied: R{rule.minimum_fee:,.2f}")
        breakdown.append(f"Fee/service (before time) = R{fee_per_service:,.2f}")

        if rule.time_rate_per_unit_per_day:
            days = vessel.days_alongside_value
            time_component = units * rule.time_rate_per_unit_per_day * days
            breakdown.append(
                f"Time component = {units:,.2f} × R{rule.time_rate_per_unit_per_day}/day"
                f" × {days:.4f}d = R{time_component:,.2f}"
            )
            fee_per_service += time_component

        formula = _build_formula(rule, units, vessel.days_alongside_value)

    total = round(fee_per_service * rule.num_services, 2)
    breakdown.append(
        f"Total = R{fee_per_service:,.2f} × {rule.num_services} = R{total:,.2f}"
    )

    section = f" ({rule.section_reference})" if rule.section_reference else ""
    label = rule.tariff_type.replace("_", " ").title()

    return TariffLineItem(
        tariff_type=rule.tariff_type,
        description=f"{label}{section}",
        formula_used=formula,
        amount_zar=total,
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
    result = CalculationResult(vessel_name=vessel.vessel_metadata.name, port=vessel.port)
    for tariff_type, rule in rules.items():
        try:
            result.add(apply_rule(vessel, rule))
        except Exception as e:
            print(f"[calculator] ERROR calculating {tariff_type}: {e}")
            label = tariff_type.replace("_", " ").title()
            result.add(TariffLineItem(
                tariff_type=tariff_type,
                description=f"{label} (FAILED)",
                formula_used="N/A",
                amount_zar=0.0,
                breakdown=[f"Error: {e}"],
                error=str(e),
            ))
    return result
