from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import chromadb
from beartype import beartype
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from core.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    TARIFF_SECTION_KEYWORDS,
    TOP_K_RETRIEVAL,
)
from core.tariff_schema import ExtractedTariffRule

# ---------------------------------------------------------------------------
# LLM extraction prompt (schema is enforced by structured output — not in prompt)
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are an expert maritime port tariff analyst extracting calculation rules.

Extract the rule for **{tariff_label}** from the document context below.
Port: {port}
Vessel type: {vessel_type}

TARIFF DOCUMENT CONTEXT:
{context}

VESSEL OPERATIONAL DATA:
- Number of operations (arrival + departure): {num_services}

Extraction guidelines:
- ONLY use numbers found in the document context.  Do not invent rates.
- CRITICAL: You are extracting **{tariff_label}** only.  If the retrieved context is
  primarily about a DIFFERENT tariff type, set extraction_confidence to LOW and stop.
  Do NOT reinterpret another tariff's rates as {tariff_label}.
- brackets: populate ONLY for step-rate (bracket) tariffs such as towage.  Set to [] for all others.

num_services rules — THIS IS CRITICAL, do not leave num_services as 1 unless the tariff
is genuinely charged once per port call (e.g. light dues, VTS, port dues):
- Pilotage: num_services = {num_services} (one pilotage service per operation:
  1 for arrival, 1 for departure = {num_services} total).
- Towage: num_services = {num_services} (one tug allocation per operation).
- Running lines: num_services = {num_services} operations × 6 lines per operation
  = {num_services_x6} total.  A standard port call has 6 line services per operation
  (3 head lines + 3 stern lines for mooring/unmooring).
- Light dues / VTS / Port dues: num_services = 1 (charged once per port call).

Tariff-specific encoding patterns:
- Pilotage pattern    -> basic_fee = fixed base per service; rate_per_unit = per-100-GT add-on;
                        unit_divisor = 100; use_ceiling = true;
                        num_services = {num_services}.
- VTS pattern         -> basic_fee = 0; rate_per_unit = per-GT rate;
                        unit_divisor = 1; use_ceiling = false; num_services = 1.
- Port dues pattern   -> rate_per_unit = basic rate per 100 GT; unit_divisor = 100; use_ceiling = true;
                        time_rate_per_unit_per_day = incremental rate per 100 GT per 24 h;
                        num_services = 1.
- Running lines / flat per-service charge -> basic_fee = the flat fee per service;
                        rate_per_unit = 0; unit_divisor = 1; use_ceiling = false;
                        num_services = {num_services_x6}.
- Towage / bracket tariff -> populate brackets array with GT ranges and fees;
                        set rate_per_unit = 0; num_services = {num_services}.
- extraction_confidence: HIGH = exact rates found in context for THIS tariff;
                         MEDIUM = rates found but applicability conditions unclear;
                         LOW = context is about a different tariff or rates are missing.
"""

# Expected section keyword hints per tariff type — used to validate LLM output
_SECTION_HINTS: dict[str, list[str]] = {
    "light_dues":    ["1.1"],
    "vts_dues":      ["2.1", "2."],
    "pilotage_dues": ["3.3"],
    "towage_dues":   ["3.6"],
    "running_lines": ["3.9"],
    "port_dues":     ["4.1"],
}

# ---------------------------------------------------------------------------
# Rule cache — avoids re-running LLM extraction for the same PDF + tariff + port
# ---------------------------------------------------------------------------

_RULE_CACHE_DIR = str(Path(__file__).parent.parent.parent / ".rule_cache")


def _pdf_hash(pdf_path: str) -> str:
    """Return a short SHA-256 hex digest of a PDF file for cache keying."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_key(pdf_hash: str, tariff_type: str, port: str) -> str:
    port_slug = port.lower().replace(" ", "_")
    return f"{pdf_hash}_{tariff_type}_{port_slug}"


def load_cached_rule(pdf_hash: str, tariff_type: str, port: str) -> ExtractedTariffRule | None:
    """Load a previously extracted rule from disk cache, or return None."""
    path = Path(_RULE_CACHE_DIR) / f"{_cache_key(pdf_hash, tariff_type, port)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExtractedTariffRule.model_validate(data)
    except Exception:
        return None


def save_cached_rule(pdf_hash: str, tariff_type: str, port: str, rule: ExtractedTariffRule) -> None:
    """Persist an extracted rule to disk cache."""
    Path(_RULE_CACHE_DIR).mkdir(exist_ok=True)
    path = Path(_RULE_CACHE_DIR) / f"{_cache_key(pdf_hash, tariff_type, port)}.json"
    path.write_text(rule.model_dump_json(indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------

def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )


@beartype
def build_vector_store(chunks: list[Document], force_rebuild: bool = False) -> Chroma:
    """Index document chunks into ChromaDB, loading from disk cache when available.

    Note: if you switch embedding models, pass force_rebuild=True to re-index —
    the stored vectors are incompatible across different embedding models.
    """
    embeddings = _get_embeddings()

    if force_rebuild and os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"[tariff_extractor] Deleted existing vector store at {CHROMA_PERSIST_DIR}")

    if not force_rebuild and os.path.exists(CHROMA_PERSIST_DIR):
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        if CHROMA_COLLECTION_NAME in [c.name for c in client.list_collections()]:
            print(f"[tariff_extractor] Loading existing vector store from {CHROMA_PERSIST_DIR}")
            return Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
            )

    print(f"[tariff_extractor] Building new vector store with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[tariff_extractor] Vector store ready at {CHROMA_PERSIST_DIR}")
    return vectorstore


@beartype
def retrieve_tariff_context(
    vectorstore: Chroma,
    tariff_type: str,
    port: str,
    extra_query: str = "",
) -> str:
    """Retrieve and deduplicate the most relevant tariff chunks for a given type and port.

    Uses ALL configured keywords (not just the first) so section numbers like
    '3.6' and '4.1' are included in the query and steer results away from
    neighbouring sections.
    """
    keywords = TARIFF_SECTION_KEYWORDS.get(tariff_type, [tariff_type])
    query = f"{' '.join(keywords)} tariff rules {port} port calculation formula"
    if extra_query:
        query = f"{query} {extra_query}"

    try:
        results: list[Document] = vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
    except Exception as e:
        print(f"[tariff_extractor] Retrieval error for {tariff_type}: {e}")
        return ""

    seen: set[str] = set()
    parts: list[str] = []
    for doc in results:
        snippet = doc.page_content.strip()
        key = snippet[:100]
        if key not in seen:
            seen.add(key)
            page = doc.metadata.get("page", "?")
            section = doc.metadata.get("section", "")
            parts.append(f"[Page {page} | {section}]\n{snippet}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM extraction into typed schema (OpenAI structured output)
# ---------------------------------------------------------------------------

def _section_matches(tariff_type: str, section_reference: str) -> bool:
    """Return True if the extracted section reference is plausible for the tariff type."""
    hints = _SECTION_HINTS.get(tariff_type, [])
    ref = section_reference.lower()
    return any(hint in ref for hint in hints)


@beartype
def extract_tariff_rule(
    llm: ChatOpenAI,
    tariff_type: str,
    context: str,
    port: str,
    vessel_type: str,
    num_services: int,
) -> ExtractedTariffRule:
    """
    Ask the LLM to extract a structured ExtractedTariffRule from retrieved context.

    Uses OpenAI structured output (with_structured_output) so the response is
    guaranteed to match the Pydantic schema — no manual JSON parsing needed.

    After extraction, validates that the section reference matches the expected
    tariff type — mismatches are downgraded to LOW confidence so the fallback kicks in.

    Returns a LOW-confidence placeholder on any API error.
    """
    prompt = _EXTRACTION_PROMPT.format(
        tariff_label=tariff_type.replace("_", " ").upper(),
        tariff_type=tariff_type,
        port=port,
        vessel_type=vessel_type,
        context=context,
        num_services=num_services,
        num_services_x6=num_services * 6,
    )

    try:
        structured_llm = llm.with_structured_output(ExtractedTariffRule)
        rule: ExtractedTariffRule = structured_llm.invoke(prompt)  # type: ignore[assignment]

        # Downgrade confidence if the section reference belongs to a different tariff
        if (
            rule.extraction_confidence != "LOW"
            and rule.section_reference
            and not _section_matches(tariff_type, rule.section_reference)
        ):
            print(
                f"[tariff_extractor] Section mismatch for {tariff_type}: "
                f"got {rule.section_reference!r} — downgrading to LOW"
            )
            rule = rule.model_copy(update={"extraction_confidence": "LOW"})

        print(
            f"[tariff_extractor] {tariff_type}: "
            f"confidence={rule.extraction_confidence}  section={rule.section_reference!r}"
        )
        return rule

    except Exception as e:
        print(f"[tariff_extractor] LLM error for {tariff_type}: {e}")

    return ExtractedTariffRule(
        tariff_type=tariff_type,
        extraction_confidence="LOW",
        notes="Extraction failed — fallback rates applied.",
    )


def make_extraction_llm() -> ChatOpenAI:
    """Instantiate the LLM client used for rule extraction (called once per pipeline run)."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
