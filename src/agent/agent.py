from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from calculation.calculator import CalculationResult, run_all_calculations
from calculation.fallback_rates import get_fallback_rules
from core.config import OPENAI_API_KEY, TARIFF_TYPES
from core.tariff_schema import ExtractedTariffRule
from core.vessel_profile import VesselProfile
from ingestion.tariff_extractor import (
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
    tariff_type: str,
    pdf_hash: str | None,
    fallback: dict[str, ExtractedTariffRule],
) -> tuple[str, ExtractedTariffRule]:
    """Extract a single tariff rule — cache-first, then LLM, then fallback."""
    # 1. Try disk cache
    if pdf_hash:
        cached = load_cached_rule(pdf_hash, tariff_type, vessel.port)
        if cached and cached.extraction_confidence != "LOW":
            print(f"[agent] {tariff_type}: loaded from cache (confidence={cached.extraction_confidence})")
            return tariff_type, cached

    # 2. Retrieve context and run LLM extraction
    context = retrieve_tariff_context(vectorstore, tariff_type, vessel.port)
    rule = extract_tariff_rule(
        llm=llm,
        tariff_type=tariff_type,
        context=context,
        port=vessel.port,
        vessel_type=vessel.vessel_type,
        num_services=vessel.num_operations,
    )

    # 3. Cache successful extractions; fall back on LOW confidence
    if rule.extraction_confidence == "LOW":
        print(f"[agent] {tariff_type}: low confidence — substituting fallback rates")
        return tariff_type, fallback[tariff_type]

    if pdf_hash:
        save_cached_rule(pdf_hash, tariff_type, vessel.port, rule)

    return tariff_type, rule


# ---------------------------------------------------------------------------
# Rule extraction — parallel across all tariff types
# ---------------------------------------------------------------------------

def _extract_all_rules(
    llm: ChatOpenAI,
    vectorstore: Chroma,
    vessel: VesselProfile,
    pdf_hash: str | None = None,
) -> dict[str, ExtractedTariffRule]:
    """
    For each tariff type: retrieve relevant document context, then ask the LLM
    to extract a structured ExtractedTariffRule.

    Extractions run in parallel via ThreadPoolExecutor (one thread per tariff).
    If extraction confidence is LOW for any tariff, the hardcoded Transnet
    fallback rule is transparently substituted for that tariff only.
    """
    fallback = get_fallback_rules(vessel.port, vessel.num_operations)
    rules: dict[str, ExtractedTariffRule] = {}

    print(f"[agent] Extracting {len(TARIFF_TYPES)} tariff rules in parallel...")

    with ThreadPoolExecutor(max_workers=len(TARIFF_TYPES)) as pool:
        futures = {
            pool.submit(
                _extract_one, llm, vectorstore, vessel, tariff_type, pdf_hash, fallback
            ): tariff_type
            for tariff_type in TARIFF_TYPES
        }
        for future in as_completed(futures):
            tariff_type = futures[future]
            try:
                key, rule = future.result()
                rules[key] = rule
            except Exception as e:
                print(f"[agent] {tariff_type}: extraction failed ({e}) — using fallback")
                rules[tariff_type] = fallback[tariff_type]

    return rules


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_agentic_calculation(
    vessel: VesselProfile,
    vectorstore: Chroma | None = None,
    use_llm: bool = True,
    pdf_path: str | None = None,
) -> CalculationResult:
    """
    Full agentic pipeline:

    LLM mode  (use_llm=True, OPENAI_API_KEY set, vectorstore provided):
        1. Retrieve tariff context per tariff type from the PDF vector store.
        2. LLM extracts a structured ExtractedTariffRule from each context block
           (all 6 tariff types run in parallel).
           LOW-confidence extractions fall back to hardcoded Transnet rates.
        3. Generic calculator applies every rule to the vessel profile.

    Fallback mode (--no-llm, no key, or no vector store):
        Hardcoded Transnet rules are passed directly to the same generic
        calculator — identical code path, different rule source.

    This means adding support for a new tariff document requires only indexing
    the PDF and running LLM extraction.  No code changes are needed.
    """
    print(
        f"\n[agent] Starting calculation — {vessel.vessel_metadata.name} @ {vessel.port}"
        f" (GT={vessel.gt:,.0f}, days={vessel.days_alongside_value:.4f})"
    )

    if use_llm and OPENAI_API_KEY and vectorstore is not None:
        print("[agent] Phase 1: Extracting rules via RAG + OpenAI (parallel)...")
        llm = make_extraction_llm()

        # Compute PDF hash for rule caching (if path was provided)
        from ingestion.tariff_extractor import _pdf_hash
        pdf_hash = _pdf_hash(pdf_path) if pdf_path else None

        rules = _extract_all_rules(llm, vectorstore, vessel, pdf_hash=pdf_hash)
    else:
        reason = (
            "--no-llm requested" if not use_llm
            else "no API key" if not OPENAI_API_KEY
            else "no vector store provided"
        )
        print(f"[agent] Using fallback Transnet rates ({reason})")
        rules = get_fallback_rules(vessel.port, vessel.num_operations)

    print("\n[agent] Phase 2: Applying rules via generic calculator...")
    result = run_all_calculations(vessel, rules)
    for item in result.line_items:
        status = " (FAILED)" if item.error else ""
        print(f"  {item.tariff_type}: R{item.amount_zar:,.2f}{status}")
    print(f"\n[agent] Total: R{result.total_zar:,.2f}")
    return result


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_results(result: CalculationResult) -> str:
    """Format a CalculationResult as a human-readable text report."""
    lines: list[str] = [
        "=" * 65,
        "PORT DUES CALCULATION REPORT",
        f"Vessel  : {result.vessel_name}",
        f"Port    : {result.port}",
        f"Currency: {result.currency}",
        "=" * 65,
    ]

    for item in result.line_items:
        lines.append("\n" + "-" * 65)
        lines.append(f"  {item.description}")
        if item.error:
            lines.append(f"  ERROR : {item.error}")
        else:
            lines.append(f"  Formula : {item.formula_used}")
            for step in item.breakdown:
                lines.append(f"    * {step}")
            if item.notes:
                lines.append(f"  Notes   : {item.notes}")
        lines.append(f"  AMOUNT  : R {item.amount_zar:>12,.2f}")

    lines += [
        "\n" + "=" * 65,
        f"  TOTAL ESTIMATED PORT DUES : R {result.total_zar:>12,.2f}",
        "=" * 65,
    ]
    return "\n".join(lines)
