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
    DISCOVERY_TOP_K,
    MAX_DISCOVERED_TARIFFS,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    TOP_K_RETRIEVAL,
)
from core.tariff_schema import (
    DiscoveredTariff,
    DiscoveredTariffSet,
    ExtractedTariffRule,
)

# ---------------------------------------------------------------------------
# Discovery prompt
# ---------------------------------------------------------------------------
# Identifies every chargeable tariff category in the document.  No tariff
# names or section numbers are hardcoded anywhere — discovery is the single
# source of truth for the taxonomy, and it runs per document.
# ---------------------------------------------------------------------------

_DISCOVERY_PROMPT = """\
You are a maritime tariff analyst.  You will be given raw passages sampled
from a single port tariff document.  Your job is to identify EVERY distinct
chargeable tariff category this document defines.

For each category, return:
- name: a short canonical name in snake_case, taken from the terminology the
  document itself uses (e.g. "light_dues", "pilotage", "quay_rental",
  "wharfage", "vts_dues", "conservancy_charge").  Use the document's own
  wording — do NOT map to any external taxonomy.
- section_reference: the section number or heading where the rule lives
  (e.g. "Section 1.1", "3.3.2", "Schedule A"). Leave blank if not stated.
- description: one sentence describing what is being charged.
- applies_to: any applicability conditions (vessel type, port, cargo type,
  coastal vs international, etc.).  Leave blank if unconditional.
- retrieval_keywords: 3-5 keywords or phrases useful for finding the rule's
  exact rates in a subsequent vector search.  Include the section number if
  known, the tariff name, and any distinctive terminology.

Also return:
- currency: the primary currency code or symbol used in the document
  (e.g. "ZAR", "USD", "EUR").  Leave blank if ambiguous.

Rules of discovery:
- Do NOT invent tariffs.  Only list categories for which the document has a
  calculation rule or rate table.
- Do NOT merge categories that are calculated differently.  Pilotage and
  towage are separate even if both live under "marine services".
- Do NOT split a single rule into multiple categories.  If the document
  defines one combined rule, return it as one entry.
- If the document defines tariffs for multiple ports, return ONE category
  per tariff type (not per port) — the per-port rates will be handled in
  the downstream extraction pass.

Aim for completeness: every chargeable item the document defines should
appear in the list.  Return AT MOST {max_tariffs} entries.

DOCUMENT PASSAGES:
{passages}
"""

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------
# Intentionally free of tariff-specific formula hints.  The prompt describes
# the schema's expressive capacity and asks the LLM to choose whichever
# fields the document actually uses.  It does NOT tell the LLM which fields
# pilotage, port dues, or any other tariff category "should" populate.
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a careful maritime tariff interpreter.  Your job is to read an
extracted passage from a port tariff document and encode a single tariff
rule into a structured schema.

You are extracting the rule for the tariff category:
    {tariff_name}
Document description: {tariff_description}
Applicability (from discovery): {applicability}
Section reference (if known): {section_reference}

USE ONLY the numbers, formulas, and conditions that appear in the passage
below.  If the passage does not contain the rates you need, set
extraction_confidence to "LOW" and leave numeric fields at their defaults.

PASSAGE (retrieved from the source document):
{context}

VESSEL CONTEXT (used ONLY to resolve conditional rates that depend on the
port or vessel type — do NOT invent numbers):
- Port:                 {port}
- Vessel type:          {vessel_type}
- Operations this call: {num_operations}
- Days alongside:       {days_alongside}

Schema fields you may populate (every tariff fits this schema — your job is
to choose the fields that match what the document actually says, not the
other way around):

1. measurement_basis — which vessel measurement drives the charge?
   Choose exactly one of: GT, NT, DWT, LOA, BEAM, DRAFT, CARGO_MT, NONE.
   Use NONE for pure flat fees that do not depend on a vessel measurement.

2. Simple rate components (combine additively per service):
   - basic_fee: a fixed amount added per service.
   - rate_per_unit: a per-unit rate on measurement_basis.
   - unit_divisor: the "per-N" divisor.
       Examples: 1 for "per GT", 100 for "per 100 GT", 1000 for "per 1000 GT".
   - use_ceiling: True if the document says "per N units or part thereof",
                  False if the rate is strictly proportional.
   - minimum_fee: floor applied to (basic_fee + rate_per_unit * units).
   - maximum_fee: cap applied to the same expression; null = no cap.
   Set each of these to its default (0 / 1 / false / null) if the document
   does not define it.

3. Time component — for tariffs that accrue per day alongside:
   - time_rate_per_unit_per_day: per-unit amount added per day.  If the
     document specifies "per hour", divide by 24 and put the result here.
     Leave 0 for one-off charges.

4. Bracket schedule — for step-rate tariffs:
   - brackets: list of RateBracket rows
       (min_value, max_value, base_fee, rate_above_min, rate_divisor).
     If brackets is non-empty, the simple-rate fields above are IGNORED
     by the calculator.  Use brackets when the document gives a stepped
     rate table (e.g. GT ranges with different base fees).

5. Multiplier — how many times the per-service fee applies per port call:
   - num_services: e.g. 1 (charged once per call), {num_operations}
     (one per arrival + one per departure), or a documented multiple
     such as "6 lines × operations".  Read the document carefully:
     running/mooring lines are almost always charged multiple times per
     operation, pilotage and towage typically once per operation, dues
     like light/VTS/port dues typically once per call.
   - services_basis: one short sentence explaining what you counted and why.

6. Currency: the currency code or symbol used in the passage.

7. applicability_notes: any conditions that narrow when this rule applies
   (e.g. "non-coastal vessels", "vessels handling cargo",
   "Durban and Saldanha only").

8. extraction_confidence:
   HIGH   = every rate you output is quoted directly from the passage.
   MEDIUM = rates are present but some applicability condition is ambiguous.
   LOW    = the passage does not contain the rates you need; fields are
            left at defaults.

Do NOT assume a formula shape.  A tariff may legitimately be:
- a flat fee with no per-unit component (basic_fee only),
- a pure per-unit rate (rate_per_unit only),
- a bracket schedule (brackets only),
- a per-unit rate plus a per-day time component,
- a minimum/maximum-bounded per-unit rate,
- or some combination of the above.

If the document describes a calculation that cannot be expressed by the
schema above, set extraction_confidence to LOW and explain in ``notes``.
"""

# ---------------------------------------------------------------------------
# Rule cache — avoids re-running LLM extraction for the same PDF + tariff + port
# ---------------------------------------------------------------------------

_RULE_CACHE_DIR = str(Path(__file__).parent.parent.parent / ".rule_cache")
_DISCOVERY_CACHE_DIR = str(Path(__file__).parent.parent.parent / ".discovery_cache")


def _pdf_hash(pdf_path: str) -> str:
    """Return a short SHA-256 hex digest of a PDF file for cache keying."""
    try:
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception as e:
        print(f"[tariff_extractor] Could not hash PDF {pdf_path!r}: {e}")
        return ""


def _cache_key(pdf_hash: str, tariff_name: str, port: str) -> str:
    port_slug = port.lower().replace(" ", "_") or "any"
    name_slug = tariff_name.lower().replace(" ", "_")
    return f"{pdf_hash}_{name_slug}_{port_slug}"


def load_cached_rule(pdf_hash: str, tariff_name: str, port: str) -> ExtractedTariffRule | None:
    """Load a previously extracted rule from disk cache, or return None."""
    path = Path(_RULE_CACHE_DIR) / f"{_cache_key(pdf_hash, tariff_name, port)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExtractedTariffRule.model_validate(data)
    except Exception as e:
        print(f"[tariff_extractor] Rule cache read failed for {tariff_name}: {e}")
        return None


def save_cached_rule(pdf_hash: str, tariff_name: str, port: str, rule: ExtractedTariffRule) -> None:
    """Persist an extracted rule to disk cache."""
    try:
        Path(_RULE_CACHE_DIR).mkdir(exist_ok=True)
        path = Path(_RULE_CACHE_DIR) / f"{_cache_key(pdf_hash, tariff_name, port)}.json"
        path.write_text(rule.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[tariff_extractor] Rule cache write failed for {tariff_name}: {e}")


def load_cached_discovery(pdf_hash: str) -> DiscoveredTariffSet | None:
    """Load a previously discovered tariff taxonomy from disk cache."""
    if not pdf_hash:
        return None
    path = Path(_DISCOVERY_CACHE_DIR) / f"{pdf_hash}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DiscoveredTariffSet.model_validate(data)
    except Exception as e:
        print(f"[tariff_extractor] Discovery cache read failed: {e}")
        return None


def save_cached_discovery(pdf_hash: str, discovery: DiscoveredTariffSet) -> None:
    """Persist the discovered tariff taxonomy to disk cache."""
    if not pdf_hash:
        return
    try:
        Path(_DISCOVERY_CACHE_DIR).mkdir(exist_ok=True)
        path = Path(_DISCOVERY_CACHE_DIR) / f"{pdf_hash}.json"
        path.write_text(discovery.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[tariff_extractor] Discovery cache write failed: {e}")


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
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print(f"[tariff_extractor] Deleted existing vector store at {CHROMA_PERSIST_DIR}")
        except OSError as e:
            print(f"[tariff_extractor] Could not delete vector store: {e}")

    if not force_rebuild and os.path.exists(CHROMA_PERSIST_DIR):
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            if CHROMA_COLLECTION_NAME in [c.name for c in client.list_collections()]:
                print(f"[tariff_extractor] Loading existing vector store from {CHROMA_PERSIST_DIR}")
                return Chroma(
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=embeddings,
                    persist_directory=CHROMA_PERSIST_DIR,
                )
        except Exception as e:
            print(f"[tariff_extractor] Could not reuse existing vector store ({e}) — rebuilding")

    print(f"[tariff_extractor] Building new vector store with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[tariff_extractor] Vector store ready at {CHROMA_PERSIST_DIR}")
    return vectorstore


def _dedupe_chunks(results: list[Document]) -> str:
    """Concatenate a list of Documents into a single context string, de-duplicating
    identical snippets (which are common with overlapping chunks).
    """
    seen: set[str] = set()
    parts: list[str] = []
    for doc in results:
        snippet = doc.page_content.strip()
        key = snippet[:100]
        if key in seen:
            continue
        seen.add(key)
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")
        parts.append(f"[Page {page} | {section}]\n{snippet}")
    return "\n\n---\n\n".join(parts)


@beartype
def retrieve_tariff_context(
    vectorstore: Chroma,
    tariff: DiscoveredTariff,
    port: str,
    extra_query: str = "",
) -> str:
    """Retrieve the most relevant passages for a discovered tariff.

    The retrieval query is built from the tariff's own name, description,
    applicability, section reference, and discovery-time keywords — none of
    which are hardcoded in this repository.
    """
    query_parts: list[str] = [tariff.name.replace("_", " ")]
    if tariff.section_reference:
        query_parts.append(tariff.section_reference)
    if tariff.description:
        query_parts.append(tariff.description)
    if tariff.applies_to:
        query_parts.append(tariff.applies_to)
    query_parts.extend(tariff.retrieval_keywords)
    query_parts.extend(["tariff rate fee calculation", port])
    if extra_query:
        query_parts.append(extra_query)

    query = " ".join(p for p in query_parts if p)

    try:
        results: list[Document] = vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
    except Exception as e:
        print(f"[tariff_extractor] Retrieval error for {tariff.name!r}: {e}")
        return ""

    return _dedupe_chunks(results)


@beartype
def _retrieve_discovery_context(vectorstore: Chroma) -> str:
    """Retrieve a broad sample of the document for tariff taxonomy discovery.

    Uses generic tariff terminology as the query — no port or tariff names
    are hardcoded.  Returns the concatenated, deduplicated chunk text.
    """
    query = (
        "tariff fee rate charge dues schedule per gross tonnage vessel "
        "section calculation table rands basic additional minimum per day"
    )
    try:
        results: list[Document] = vectorstore.similarity_search(query, k=DISCOVERY_TOP_K)
    except Exception as e:
        print(f"[tariff_extractor] Discovery retrieval error: {e}")
        return ""
    return _dedupe_chunks(results)


# ---------------------------------------------------------------------------
# Discovery stage — identify the document's tariff taxonomy at runtime
# ---------------------------------------------------------------------------

@beartype
def discover_tariffs(
    llm: ChatOpenAI,
    vectorstore: Chroma,
    pdf_hash: str = "",
) -> DiscoveredTariffSet:
    """Ask the LLM to list every chargeable tariff category in the document.

    The result is cached on disk by PDF hash so repeated runs against the
    same document skip the LLM call.
    """
    cached = load_cached_discovery(pdf_hash) if pdf_hash else None
    if cached is not None and cached.tariffs:
        print(
            f"[tariff_extractor] Discovery: loaded {len(cached.tariffs)} tariffs from cache"
        )
        return cached

    passages = _retrieve_discovery_context(vectorstore)
    if not passages:
        print("[tariff_extractor] Discovery: no passages retrieved — returning empty set")
        return DiscoveredTariffSet()

    prompt = _DISCOVERY_PROMPT.format(
        passages=passages,
        max_tariffs=MAX_DISCOVERED_TARIFFS,
    )

    try:
        structured_llm = llm.with_structured_output(DiscoveredTariffSet)
        discovery = structured_llm.invoke(prompt)
        assert isinstance(discovery, DiscoveredTariffSet)
    except Exception as e:
        print(f"[tariff_extractor] Discovery LLM error: {e}")
        return DiscoveredTariffSet()

    # Cap and dedupe by name
    seen_names: set[str] = set()
    uniq: list[DiscoveredTariff] = []
    for t in discovery.tariffs[:MAX_DISCOVERED_TARIFFS]:
        key = t.name.lower().strip()
        if key and key not in seen_names:
            seen_names.add(key)
            uniq.append(t)
    discovery = DiscoveredTariffSet(currency=discovery.currency, tariffs=uniq)

    print(
        f"[tariff_extractor] Discovery: found {len(discovery.tariffs)} tariff categories "
        f"(currency={discovery.currency or 'unknown'})"
    )
    for t in discovery.tariffs:
        print(f"    - {t.name!r:32} section={t.section_reference!r}")

    if pdf_hash:
        save_cached_discovery(pdf_hash, discovery)

    return discovery


# ---------------------------------------------------------------------------
# Per-tariff extraction into typed schema (OpenAI structured output)
# ---------------------------------------------------------------------------

@beartype
def extract_tariff_rule(
    llm: ChatOpenAI,
    tariff: DiscoveredTariff,
    context: str,
    port: str,
    vessel_type: str,
    num_operations: int,
    days_alongside: float,
    document_currency: str = "",
) -> ExtractedTariffRule:
    """Ask the LLM to extract a structured ExtractedTariffRule from context.

    Uses OpenAI structured output (``with_structured_output``) so the response
    is guaranteed to match the Pydantic schema — no manual JSON parsing.

    Returns a LOW-confidence placeholder on any API error.  Does NOT fall
    back to any hardcoded rate table.
    """
    prompt = _EXTRACTION_PROMPT.format(
        tariff_name=tariff.name,
        tariff_description=tariff.description or "(none)",
        applicability=tariff.applies_to or "(none)",
        section_reference=tariff.section_reference or "(unknown)",
        context=context or "(no passages retrieved)",
        port=port or "(any)",
        vessel_type=vessel_type or "(unspecified)",
        num_operations=num_operations,
        days_alongside=f"{days_alongside:.4f}",
    )

    try:
        structured_llm = llm.with_structured_output(ExtractedTariffRule)
        rule = structured_llm.invoke(prompt)
        assert isinstance(rule, ExtractedTariffRule)
    except Exception as e:
        print(f"[tariff_extractor] LLM error for {tariff.name!r}: {e}")
        return ExtractedTariffRule(
            tariff_name=tariff.name,
            section_reference=tariff.section_reference,
            description=tariff.description,
            applicability_notes=tariff.applies_to,
            extraction_confidence="LOW",
            notes=f"Extraction failed: {e}",
        )

    # Backfill identification fields from discovery when the LLM left them blank
    if not rule.tariff_name:
        rule = rule.model_copy(update={"tariff_name": tariff.name})
    if not rule.section_reference and tariff.section_reference:
        rule = rule.model_copy(update={"section_reference": tariff.section_reference})
    if not rule.description and tariff.description:
        rule = rule.model_copy(update={"description": tariff.description})
    if not rule.applicability_notes and tariff.applies_to:
        rule = rule.model_copy(update={"applicability_notes": tariff.applies_to})
    if not rule.currency and document_currency:
        rule = rule.model_copy(update={"currency": document_currency})

    print(
        f"[tariff_extractor] {tariff.name!r}: confidence={rule.extraction_confidence}  "
        f"basis={rule.measurement_basis}  brackets={len(rule.brackets)}  "
        f"num_services={rule.num_services}"
    )
    return rule


def make_extraction_llm() -> ChatOpenAI:
    """Instantiate the LLM client used for discovery and rule extraction."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
