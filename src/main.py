from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from beartype import beartype

from agent.agent import format_results, run_agentic_calculation
from agent.query_parser import parse_vessel_query
from core.config import OPENAI_API_KEY
from core.vessel_profile import VesselProfile, load_vessel_profile
from ingestion.document_processor import process_tariff_pdf
from ingestion.tariff_extractor import build_vector_store

# ---------------------------------------------------------------------------
# Reference vessel for validation (from the Take Home Test spec)
# ---------------------------------------------------------------------------
REFERENCE_VESSEL: dict = {
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

GROUND_TRUTH: dict[str, float] = {
    "light_dues":    60_062.04,
    "port_dues":    199_549.22,
    "towage_dues":  147_074.38,
    "vts_dues":      33_315.75,
    "pilotage_dues": 47_189.94,
    "running_lines": 19_639.50,
}

DEFAULT_TARIFF_PDF = str(Path(__file__).parent.parent / "data" / "Port Tariff.pdf")


@beartype
def print_validation_table(result_items: list, ground_truth: dict[str, float]) -> None:
    """Print a side-by-side comparison of calculated vs ground truth values."""
    print("\n" + "=" * 75)
    print(f"  {'TARIFF':<20} {'CALCULATED':>14} {'GROUND TRUTH':>14} {'DIFF':>10} {'%ERR':>8}")
    print("=" * 75)

    total_calc = 0.0
    total_gt = 0.0
    for item in result_items:
        gt = ground_truth.get(item.tariff_type, 0.0)
        diff = item.amount_zar - gt
        pct = (diff / gt * 100) if gt else 0.0
        flag = "OK" if abs(pct) < 2 else ("~" if abs(pct) < 10 else "FAIL")
        print(
            f"  [{flag}] {item.tariff_type:<16} "
            f"R{item.amount_zar:>12,.2f} "
            f"R{gt:>12,.2f} "
            f"{diff:>+10.2f} "
            f"{pct:>+7.2f}%"
        )
        total_calc += item.amount_zar
        total_gt += gt

    total_diff = total_calc - total_gt
    total_pct = (total_diff / total_gt * 100) if total_gt else 0.0
    print("=" * 75)
    print(
        f"  {'TOTAL':<20} "
        f"R{total_calc:>12,.2f} "
        f"R{total_gt:>12,.2f} "
        f"{total_diff:>+10.2f} "
        f"{total_pct:>+7.2f}%"
    )
    print("=" * 75)


@beartype
def run_validation(
    tariff_pdf: str = DEFAULT_TARIFF_PDF,
    use_llm: bool = True,
    force_rebuild: bool = False,
) -> None:
    """Run the full pipeline against the reference vessel (SUDESTADA @ Durban)."""
    if not os.path.exists(tariff_pdf):
        print(f"[main] ERROR: Tariff PDF not found at '{tariff_pdf}'")
        sys.exit(1)

    if use_llm and not OPENAI_API_KEY:
        print("[main] WARNING: OPENAI_API_KEY not set — falling back to deterministic mode.")
        use_llm = False

    print("[main] Loading vessel profile: SUDESTADA @ Durban")
    vessel = load_vessel_profile(REFERENCE_VESSEL, port="Durban")

    vectorstore = None
    if use_llm:
        print(f"[main] Processing tariff document: {tariff_pdf}")
        chunks = process_tariff_pdf(tariff_pdf)
        print("[main] Building / loading vector store...")
        vectorstore = build_vector_store(chunks, force_rebuild=force_rebuild)

    result = run_agentic_calculation(
        vessel, vectorstore=vectorstore, use_llm=use_llm, pdf_path=tariff_pdf,
    )
    print("\n" + format_results(result))
    print_validation_table(result.line_items, GROUND_TRUTH)


@beartype
def run_custom(
    tariff_pdf: str,
    vessel_json_path: str,
    port: str,
    use_llm: bool = True,
) -> None:
    """Run against a custom vessel JSON file and tariff PDF."""
    with open(vessel_json_path, encoding="utf-8") as f:
        vessel_data: dict = json.load(f)

    vessel = load_vessel_profile(vessel_data, port=port)

    vectorstore = None
    if use_llm and OPENAI_API_KEY:
        if not os.path.exists(tariff_pdf):
            print(f"[main] ERROR: Tariff PDF not found at '{tariff_pdf}'")
            sys.exit(1)
        chunks = process_tariff_pdf(tariff_pdf)
        vectorstore = build_vector_store(chunks)

    result = run_agentic_calculation(
        vessel, vectorstore=vectorstore, use_llm=use_llm, pdf_path=tariff_pdf,
    )
    print(format_results(result))


@beartype
def run_query(
    query: str,
    tariff_pdf: str = DEFAULT_TARIFF_PDF,
    force_rebuild: bool = False,
) -> None:
    """Parse a natural language vessel query and calculate port dues."""
    print(f"[main] Parsing natural language query: {query!r}")
    vessel, port = parse_vessel_query(query)

    if not os.path.exists(tariff_pdf):
        print(f"[main] ERROR: Tariff PDF not found at '{tariff_pdf}'")
        sys.exit(1)

    print(f"[main] Processing tariff document: {tariff_pdf}")
    chunks = process_tariff_pdf(tariff_pdf)
    vectorstore = build_vector_store(chunks, force_rebuild=force_rebuild)

    result = run_agentic_calculation(
        vessel, vectorstore=vectorstore, use_llm=True, pdf_path=tariff_pdf,
    )
    print("\n" + format_results(result))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Port Tariff Calculator — Agentic RAG System")
    subparsers = parser.add_subparsers(dest="command")

    val = subparsers.add_parser("validate", help="Run against reference vessel (SUDESTADA @ Durban)")
    val.add_argument("--pdf", default=DEFAULT_TARIFF_PDF, help="Path to tariff PDF")
    val.add_argument("--no-llm", action="store_true", help="Use fallback rates, skip LLM extraction")
    val.add_argument("--rebuild", action="store_true", help="Force vector store rebuild")

    calc = subparsers.add_parser("calculate", help="Calculate dues for a custom vessel")
    calc.add_argument("--pdf", required=True, help="Path to tariff PDF")
    calc.add_argument("--vessel", required=True, help="Path to vessel JSON file")
    calc.add_argument("--port", required=True, help="Port name (e.g. Durban)")
    calc.add_argument("--no-llm", action="store_true", help="Use fallback rates, skip LLM extraction")

    q = subparsers.add_parser("query", help="Natural language vessel query")
    q.add_argument("query", help="Vessel description in natural language")
    q.add_argument("--pdf", default=DEFAULT_TARIFF_PDF, help="Path to tariff PDF")
    q.add_argument("--rebuild", action="store_true", help="Force vector store rebuild")

    args = parser.parse_args()

    if args.command == "validate":
        run_validation(tariff_pdf=args.pdf, use_llm=not args.no_llm, force_rebuild=args.rebuild)
    elif args.command == "calculate":
        run_custom(tariff_pdf=args.pdf, vessel_json_path=args.vessel, port=args.port, use_llm=not args.no_llm)
    elif args.command == "query":
        run_query(query=args.query, tariff_pdf=args.pdf, force_rebuild=args.rebuild)
    else:
        run_validation()


if __name__ == "__main__":
    main()
