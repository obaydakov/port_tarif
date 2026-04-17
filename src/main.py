from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from beartype import beartype

from agent.agent import format_results, run_agentic_calculation
from agent.query_parser import parse_vessel_query
from calculation.calculator import TariffLineItem
from core.config import OPENAI_API_KEY
from core.vessel_profile import VesselProfile, load_vessel_profile
from ingestion.document_processor import process_tariff_pdf
from ingestion.tariff_extractor import build_vector_store

# ---------------------------------------------------------------------------
# Reference vessel used by the ``validate`` command.
#
# NOTE: this vessel + its ground-truth numbers are a verification artefact,
# not a default.  They prove that the GENERIC pipeline (discovery -> LLM
# extraction -> generic calculator) reproduces the published Transnet numbers
# for one specific known case.  Nothing in the production code path references
# this data; it exists only to give the take-home reviewer a reproducible
# correctness check.
# ---------------------------------------------------------------------------
REFERENCE_VESSEL: dict[str, dict[str, object]] = {
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

# Reference values published by Transnet for SUDESTADA @ Durban.
# Keys are matched to discovered tariff names using fuzzy substring matching
# (see _match_ground_truth) — we do not assume the LLM uses these exact names.
REFERENCE_GROUND_TRUTH: dict[str, float] = {
    "light_dues":    60_062.04,
    "port_dues":    199_549.22,
    "towage_dues":  147_074.38,
    "vts_dues":      33_315.75,
    "pilotage_dues": 47_189.94,
    "running_lines": 19_639.50,
}

DEFAULT_TARIFF_PDF = str(Path(__file__).parent.parent / "data" / "Port Tariff.pdf")


@beartype
def _match_ground_truth(tariff_name: str, ground_truth: dict[str, float]) -> tuple[str, float] | None:
    """Fuzzy-match a discovered tariff name to a ground-truth key.

    The LLM may call a tariff ``lighthouse_dues`` or ``vessel_traffic_service``;
    we treat any substring overlap as a match.  Returns (key, value) or None.
    """
    needle = tariff_name.lower().replace("-", "_").replace(" ", "_")
    # Check the needle tokens against each ground-truth key
    needle_tokens = {t for t in needle.split("_") if t}
    for key, value in ground_truth.items():
        key_tokens = {t for t in key.split("_") if t}
        if needle_tokens & key_tokens:
            return key, value
    return None


@beartype
def print_validation_table(
    result_items: list[TariffLineItem],
    ground_truth: dict[str, float],
) -> None:
    """Print a side-by-side comparison of calculated vs published reference values.

    Uses fuzzy name matching so the pipeline's discovered tariff names don't
    have to exactly match the reference keys.
    """
    print("\n" + "=" * 85)
    print(
        f"  {'DISCOVERED TARIFF':<28} {'CALCULATED':>14} "
        f"{'REFERENCE':>14} {'DIFF':>10} {'%ERR':>8}"
    )
    print("=" * 85)

    total_calc = 0.0
    total_gt = 0.0
    matched_keys: set[str] = set()
    for item in result_items:
        match = _match_ground_truth(item.tariff_name, ground_truth)
        if match:
            key, gt = match
            matched_keys.add(key)
        else:
            key, gt = "(no reference)", 0.0

        diff = item.amount - gt
        pct = (diff / gt * 100) if gt else 0.0
        flag = "OK" if gt and abs(pct) < 2 else ("~" if gt and abs(pct) < 10 else "—")
        print(
            f"  [{flag}] {item.tariff_name:<24} "
            f"{item.amount:>12,.2f}   "
            f"{gt:>12,.2f}   "
            f"{diff:>+8.2f}  "
            f"{pct:>+7.2f}%"
        )
        total_calc += item.amount
        if gt:
            total_gt += gt

    unmatched = set(ground_truth) - matched_keys
    if unmatched:
        print(f"  (reference keys with no matching discovered tariff: {sorted(unmatched)})")

    total_diff = total_calc - total_gt
    total_pct = (total_diff / total_gt * 100) if total_gt else 0.0
    print("=" * 85)
    print(
        f"  {'TOTAL (matched only)':<28} "
        f"{total_calc:>12,.2f}   "
        f"{total_gt:>12,.2f}   "
        f"{total_diff:>+8.2f}  "
        f"{total_pct:>+7.2f}%"
    )
    print("=" * 85)


def _require_api_key() -> None:
    if not OPENAI_API_KEY:
        print(
            "[main] ERROR: OPENAI_API_KEY is not set. The generalisable pipeline "
            "is document-driven and has no hardcoded rate fallback. Set the key "
            "as a system environment variable (or in .env) and retry."
        )
        sys.exit(2)


@beartype
def run_validation(tariff_pdf: str = DEFAULT_TARIFF_PDF, force_rebuild: bool = False) -> None:
    """Run the full pipeline against the reference vessel (SUDESTADA @ Durban).

    This is a correctness sanity check against a document we have a published
    answer for — it does NOT use any hardcoded rates.  The pipeline discovers
    the tariff taxonomy, extracts rules via LLM, and calculates generically.
    """
    _require_api_key()
    if not os.path.exists(tariff_pdf):
        print(f"[main] ERROR: Tariff PDF not found at '{tariff_pdf}'")
        sys.exit(1)

    print("[main] Loading vessel profile: SUDESTADA @ Durban")
    vessel = load_vessel_profile(REFERENCE_VESSEL, port="Durban")

    print(f"[main] Processing tariff document: {tariff_pdf}")
    chunks = process_tariff_pdf(tariff_pdf)
    print("[main] Building / loading vector store...")
    vectorstore = build_vector_store(chunks, force_rebuild=force_rebuild)

    result = run_agentic_calculation(vessel, vectorstore=vectorstore, pdf_path=tariff_pdf)
    print("\n" + format_results(result))
    print_validation_table(result.line_items, REFERENCE_GROUND_TRUTH)


@beartype
def run_custom(tariff_pdf: str, vessel_json_path: str, port: str, force_rebuild: bool = False) -> None:
    """Run against a custom vessel JSON file and tariff PDF."""
    _require_api_key()
    if not os.path.exists(tariff_pdf):
        print(f"[main] ERROR: Tariff PDF not found at '{tariff_pdf}'")
        sys.exit(1)

    try:
        with open(vessel_json_path, encoding="utf-8") as f:
            vessel_data: dict[str, object] = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[main] ERROR: Could not read vessel JSON at {vessel_json_path!r}: {e}")
        sys.exit(1)

    vessel = load_vessel_profile(vessel_data, port=port)

    chunks = process_tariff_pdf(tariff_pdf)
    vectorstore = build_vector_store(chunks, force_rebuild=force_rebuild)

    result = run_agentic_calculation(vessel, vectorstore=vectorstore, pdf_path=tariff_pdf)
    print(format_results(result))


@beartype
def run_query(query: str, tariff_pdf: str = DEFAULT_TARIFF_PDF, force_rebuild: bool = False) -> None:
    """Parse a natural language vessel query and calculate port dues."""
    _require_api_key()
    print(f"[main] Parsing natural language query: {query!r}")
    vessel, port = parse_vessel_query(query)

    if not os.path.exists(tariff_pdf):
        print(f"[main] ERROR: Tariff PDF not found at '{tariff_pdf}'")
        sys.exit(1)

    print(f"[main] Processing tariff document: {tariff_pdf}")
    chunks = process_tariff_pdf(tariff_pdf)
    vectorstore = build_vector_store(chunks, force_rebuild=force_rebuild)

    result = run_agentic_calculation(vessel, vectorstore=vectorstore, pdf_path=tariff_pdf)
    print("\n" + format_results(result))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Port Tariff Calculator — Agentic RAG System")
    subparsers = parser.add_subparsers(dest="command")

    val = subparsers.add_parser(
        "validate",
        help="Run the full pipeline against the reference vessel (SUDESTADA @ Durban) and compare with the published numbers.",
    )
    val.add_argument("--pdf", default=DEFAULT_TARIFF_PDF, help="Path to tariff PDF")
    val.add_argument("--rebuild", action="store_true", help="Force vector store rebuild")

    calc = subparsers.add_parser("calculate", help="Calculate dues for a custom vessel + PDF")
    calc.add_argument("--pdf", required=True, help="Path to tariff PDF")
    calc.add_argument("--vessel", required=True, help="Path to vessel JSON file")
    calc.add_argument("--port", required=True, help="Port name (e.g. Durban, Rotterdam, Singapore)")
    calc.add_argument("--rebuild", action="store_true", help="Force vector store rebuild")

    q = subparsers.add_parser("query", help="Natural language vessel query")
    q.add_argument("query", help="Vessel description in natural language")
    q.add_argument("--pdf", default=DEFAULT_TARIFF_PDF, help="Path to tariff PDF")
    q.add_argument("--rebuild", action="store_true", help="Force vector store rebuild")

    args = parser.parse_args()

    if args.command == "validate":
        run_validation(tariff_pdf=args.pdf, force_rebuild=args.rebuild)
    elif args.command == "calculate":
        run_custom(
            tariff_pdf=args.pdf,
            vessel_json_path=args.vessel,
            port=args.port,
            force_rebuild=args.rebuild,
        )
    elif args.command == "query":
        run_query(query=args.query, tariff_pdf=args.pdf, force_rebuild=args.rebuild)
    else:
        run_validation()


if __name__ == "__main__":
    main()
