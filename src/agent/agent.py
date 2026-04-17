from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from calculation.calculator import CalculationResult, run_all_calculations
from core.config import OPENAI_API_KEY
from core.tariff_schema import DiscoveredTariff, DiscoveredTariffSet, ExtractedTariffRule
from core.vessel_profile import VesselProfile
from ingestion.tariff_extractor import (
    _pdf_hash,
    discover_tariffs,
    extract_tariff_rule,
    load_cached_rule,
    make_extraction_llm,
    retrieve_tariff_context,
    save_cached_rule,
)


# ---------------------------------------------------------------------------
# Single-tariff extraction (unit of work for the thread pool)
# ---------------------------------------------------------------------------

def _extract_one(
    llm: ChatOpenAI,
    vectorstore: Chroma,
    vessel: VesselProfile,
    tariff: DiscoveredTariff,
    pdf_hash: str,
    document_currency: str,
) -> tuple[str, ExtractedTariffRule]:
    """Extract a single tariff rule — cache-first, then LLM.  No hardcoded fallback."""
    # 1. Disk cache
    if pdf_hash:
        cached = load_cached_rule(pdf_hash, tariff.name, vessel.port)
        if cached is not None and cached.extraction_confidence != "LOW":
            print(
                f"[agent] {tariff.name!r}: loaded from cache "
                f"(confidence={cached.extraction_confidence})"
            )
            return tariff.name, cached

    # 2. Retrieve context and run LLM extraction
    context = retrieve_tariff_context(vectorstore, tariff, vessel.port)
    rule = extract_tariff_rule(
        llm=llm,
        tariff=tariff,
        context=context,
        port=vessel.port,
        vessel_type=vessel.vessel_type,
        num_operations=vessel.num_operations,
        days_alongside=vessel.days_alongside_value,
        document_currency=document_currency,
    )

    # 3. Cache HIGH/MEDIUM-confidence extractions; surface LOW ones in output
    if pdf_hash and rule.extraction_confidence != "LOW":
        save_cached_rule(pdf_hash, tariff.name, vessel.port, rule)

    return tariff.name, rule


# ---------------------------------------------------------------------------
# Rule extraction — parallel across all discovered tariff categories
# ---------------------------------------------------------------------------

def _extract_all_rules(
    llm: ChatOpenAI,
    vectorstore: Chroma,
    vessel: VesselProfile,
    discovery: DiscoveredTariffSet,
    pdf_hash: str,
) -> dict[str, ExtractedTariffRule]:
    """Extract one ExtractedTariffRule per discovered tariff category, in parallel.

    No fallback rates: LOW-confidence extractions are returned as-is with an
    explanatory ``notes`` field and confidence="LOW" so the caller / report
    can surface the failure instead of silently substituting hardcoded rates.
    """
    rules: dict[str, ExtractedTariffRule] = {}
    if not discovery.tariffs:
        return rules

    print(f"[agent] Extracting {len(discovery.tariffs)} tariff rules in parallel...")

    with ThreadPoolExecutor(max_workers=min(len(discovery.tariffs), 8)) as pool:
        futures = {
            pool.submit(
                _extract_one,
                llm,
                vectorstore,
                vessel,
                tariff,
                pdf_hash,
                discovery.currency,
            ): tariff
            for tariff in discovery.tariffs
        }
        for future in as_completed(futures):
            tariff = futures[future]
            try:
                key, rule = future.result()
                rules[key] = rule
            except Exception as e:
                print(f"[agent] {tariff.name!r}: extraction crashed ({e})")
                rules[tariff.name] = ExtractedTariffRule(
                    tariff_name=tariff.name,
                    section_reference=tariff.section_reference,
                    description=tariff.description,
                    applicability_notes=tariff.applies_to,
                    extraction_confidence="LOW",
                    notes=f"Extraction crashed: {e}",
                )

    return rules


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_agentic_calculation(
    vessel: VesselProfile,
    vectorstore: Chroma,
    pdf_path: str | None = None,
) -> CalculationResult:
    """Full agentic pipeline.

    Three stages:

    1. **Discovery.**  Ask the LLM to list every chargeable tariff category
       in the document, along with section references and retrieval
       keywords.  No tariff names are hardcoded; this is the sole source
       of truth for the document's taxonomy.
    2. **Extraction.**  For each discovered tariff, retrieve targeted
       context and ask the LLM to populate an ``ExtractedTariffRule``
       using the schema's generic building blocks (per-unit, bracket,
       min/max, time component, multiplier).  The extraction prompt does
       NOT prescribe a formula shape per tariff.
    3. **Calculation.**  A single generic ``apply_rule`` function computes
       the fee from the schema.  Identical code path for every tariff
       category — no per-tariff if-branches anywhere.

    No hardcoded rate fallback exists.  LOW-confidence extractions are
    surfaced in the output with their confidence level and notes, so the
    caller can see exactly which tariffs the LLM could not interpret.

    Requires ``OPENAI_API_KEY`` to be set and a populated ``vectorstore``.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is required — the pipeline is document-driven and "
            "does not ship any hardcoded rate fallback."
        )

    print(
        f"\n[agent] Starting calculation — {vessel.vessel_metadata.name} @ {vessel.port}"
        f" (GT={vessel.gt:,.0f}, days={vessel.days_alongside_value:.4f})"
    )

    llm = make_extraction_llm()
    pdf_hash = _pdf_hash(pdf_path) if pdf_path else ""

    print("[agent] Phase 1: Discovering tariff categories from the document...")
    discovery = discover_tariffs(llm, vectorstore, pdf_hash=pdf_hash)

    if not discovery.tariffs:
        print("[agent] WARNING: discovery returned no tariff categories.")
        result = CalculationResult(
            vessel_name=vessel.vessel_metadata.name,
            port=vessel.port,
            currency=discovery.currency,
        )
        return result

    print("[agent] Phase 2: Extracting rules via RAG + OpenAI (parallel)...")
    rules = _extract_all_rules(llm, vectorstore, vessel, discovery, pdf_hash=pdf_hash)

    print("\n[agent] Phase 3: Applying rules via generic calculator...")
    result = run_all_calculations(vessel, rules)
    if not result.currency and discovery.currency:
        result.currency = discovery.currency

    for item in result.line_items:
        status = " (FAILED)" if item.error else f" [{item.confidence}]"
        print(f"  {item.tariff_name}: {item.amount:,.2f}{status}")
    print(f"\n[agent] Total ({result.currency or 'currency?'}): {result.total:,.2f}")
    return result


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_results(result: CalculationResult) -> str:
    """Format a CalculationResult as a human-readable text report."""
    currency = result.currency or ""
    lines: list[str] = [
        "=" * 70,
        "PORT DUES CALCULATION REPORT",
        f"Vessel  : {result.vessel_name}",
        f"Port    : {result.port}",
        f"Currency: {currency or '(unknown)'}",
        "=" * 70,
    ]

    if not result.line_items:
        lines.append("\n(No tariffs calculated — discovery returned no categories.)")
        lines += ["=" * 70]
        return "\n".join(lines)

    for item in result.line_items:
        lines.append("\n" + "-" * 70)
        lines.append(f"  {item.description}  [{item.confidence}]")
        if item.error:
            lines.append(f"  ERROR  : {item.error}")
        else:
            lines.append(f"  Formula: {item.formula_used}")
            for step in item.breakdown:
                lines.append(f"    * {step}")
            if item.notes:
                lines.append(f"  Notes  : {item.notes}")
        lines.append(f"  AMOUNT : {currency} {item.amount:>14,.2f}")

    lines += [
        "\n" + "=" * 70,
        f"  TOTAL ESTIMATED PORT DUES : {currency} {result.total:>14,.2f}",
        "=" * 70,
    ]
    return "\n".join(lines)
