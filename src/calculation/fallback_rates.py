from __future__ import annotations

from core.tariff_schema import ExtractedTariffRule, RateBracket

# ---------------------------------------------------------------------------
# Transnet South African Port Tariffs — April 2024 to March 2025
#
# These rates are intentionally isolated here so the rest of the codebase
# remains document-agnostic.  They are used only when:
#   (a) --no-llm mode is requested, or
#   (b) LLM extraction returns LOW confidence for a specific tariff.
# ---------------------------------------------------------------------------

# Bracket schedule shared across all Transnet ports (Section 3.6)
_TOWAGE_BRACKETS: list[RateBracket] = [
    RateBracket(min_gt=0,      max_gt=2000,   base_fee=8_140.00,  per_100_gt_above_min=0.0),
    RateBracket(min_gt=2001,   max_gt=10_000, base_fee=12_633.99, per_100_gt_above_min=268.99),
    RateBracket(min_gt=10_001, max_gt=50_000, base_fee=38_494.51, per_100_gt_above_min=84.95),
    RateBracket(min_gt=50_001, max_gt=100_000, base_fee=73_118.07, per_100_gt_above_min=32.24),
    RateBracket(min_gt=100_001, max_gt=None,  base_fee=93_548.13, per_100_gt_above_min=23.65),
]


def get_fallback_rules(port: str, num_operations: int = 2) -> dict[str, ExtractedTariffRule]:
    """
    Return Transnet-specific fallback rules for all six tariff types.

    All rates are encoded as ExtractedTariffRule objects so the same generic
    calculator (apply_rule) handles both LLM-extracted and fallback rules —
    no separate code path exists.
    """
    p = port.lower()

    # ------------------------------------------------------------------ #
    # Light Dues  (Section 1.1)                                           #
    # R117.08 per 100 GT or part thereof — non-coastal vessels            #
    # ------------------------------------------------------------------ #
    light_dues = ExtractedTariffRule(
        tariff_type="light_dues",
        section_reference="Section 1.1",
        rate_per_unit=117.08,
        unit_divisor=100,
        use_ceiling=True,
        num_services=1,
        extraction_confidence="HIGH",
        notes="R117.08 per 100 GT (or part thereof). Non-coastal vessel rate.",
    )

    # ------------------------------------------------------------------ #
    # VTS Dues  (Section 2.1)                                             #
    # R0.65/GT at Durban & Saldanha Bay; R0.54/GT elsewhere. Min R235.52  #
    # ------------------------------------------------------------------ #
    vts_rate = 0.65 if ("durban" in p or "saldanha" in p) else 0.54
    vts_dues = ExtractedTariffRule(
        tariff_type="vts_dues",
        section_reference="Section 2.1",
        rate_per_unit=vts_rate,
        unit_divisor=1,
        use_ceiling=False,
        minimum_fee=235.52,
        num_services=1,
        extraction_confidence="HIGH",
        notes=(
            f"R{vts_rate}/GT ({'Durban/Saldanha' if vts_rate == 0.65 else 'Other Ports'}). "
            "Minimum R235.52 per port call."
        ),
    )

    # ------------------------------------------------------------------ #
    # Pilotage Dues  (Section 3.3)                                        #
    # Basic fee (port-specific) + R/100GT per service                     #
    # ------------------------------------------------------------------ #
    _pilotage_rates: dict[str, tuple[float, float]] = {
        "richards bay":   (30_960.46, 10.93),
        "durban":         (18_608.61,  9.72),
        "port elizabeth": ( 8_970.00, 14.33),
        "ngqura":         ( 8_970.00, 14.33),
        "cape town":      ( 6_342.39, 10.20),
        "saldanha":       ( 9_673.57, 13.66),
    }
    pilot_basic, pilot_per_100gt = next(
        (v for k, v in _pilotage_rates.items() if k in p),
        (6_547.45, 10.49),  # default
    )
    pilotage_dues = ExtractedTariffRule(
        tariff_type="pilotage_dues",
        section_reference="Section 3.3",
        basic_fee=pilot_basic,
        rate_per_unit=pilot_per_100gt,
        unit_divisor=100,
        use_ceiling=True,
        num_services=num_operations,
        extraction_confidence="HIGH",
        notes=(
            f"Basic R{pilot_basic:,.2f} + R{pilot_per_100gt}/100GT per service. "
            f"{num_operations} services (arrival + departure)."
        ),
    )

    # ------------------------------------------------------------------ #
    # Towage Dues  (Section 3.6)                                          #
    # GT bracket schedule; per service (arrival + departure)              #
    # ------------------------------------------------------------------ #
    towage_dues = ExtractedTariffRule(
        tariff_type="towage_dues",
        section_reference="Section 3.6",
        brackets=_TOWAGE_BRACKETS,
        num_services=num_operations,
        extraction_confidence="HIGH",
        notes=(
            f"GT bracket schedule. {num_operations} services (arrival + departure). "
            "Published fee covers full tug allocation per service."
        ),
    )

    # ------------------------------------------------------------------ #
    # Running Lines  (Section 3.9)                                        #
    # Port-specific rate per line service; 6 services per operation       #
    # ------------------------------------------------------------------ #
    _lines_rates: dict[str, float] = {
        "port elizabeth": 2_266.73,
        "ngqura":         2_266.73,
        "cape town":      2_370.84,
        "saldanha":       2_085.59,
    }
    lines_rate = next((v for k, v in _lines_rates.items() if k in p), 1_654.56)
    running_lines = ExtractedTariffRule(
        tariff_type="running_lines",
        section_reference="Section 3.9",
        basic_fee=lines_rate,        # flat fee per service — NOT a per-GT rate
        rate_per_unit=0,
        unit_divisor=1,
        use_ceiling=False,
        # 6 line services per operation (3 head + 3 stern for standard bulk carrier)
        num_services=num_operations * 6,
        extraction_confidence="HIGH",
        notes=(
            f"R{lines_rate:.2f}/service. "
            f"6 line services × {num_operations} operations = {num_operations * 6} total services."
        ),
    )

    # ------------------------------------------------------------------ #
    # Port Dues  (Section 4.1)                                            #
    # R192.73/100GT basic + R57.79/100GT/day incremental (pro-rata)       #
    # ------------------------------------------------------------------ #
    port_dues = ExtractedTariffRule(
        tariff_type="port_dues",
        section_reference="Section 4.1",
        rate_per_unit=192.73,
        unit_divisor=100,
        use_ceiling=True,
        time_rate_per_unit_per_day=57.79,
        num_services=1,
        extraction_confidence="HIGH",
        notes="R192.73/100GT basic + R57.79/100GT/day incremental (pro-rata days alongside).",
    )

    return {
        "light_dues":    light_dues,
        "vts_dues":      vts_dues,
        "pilotage_dues": pilotage_dues,
        "towage_dues":   towage_dues,
        "running_lines": running_lines,
        "port_dues":     port_dues,
    }
