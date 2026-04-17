# Port Tariff Calculator — Generalisable Agentic RAG System

A document-driven tariff calculator that consumes **any** port tariff PDF and a
natural-language vessel query, and returns an itemised, explainable port-dues
estimate.

> **Design mandate.** The take-home brief asked for a solution that can handle
> tariffs from *any* port with **minimal code changes**. This submission honours
> that mandate literally:
>
> - **No tariff taxonomy lives in code.** The set of chargeable tariff
>   categories is **discovered from the PDF at runtime** (`discover_tariffs`).
> - **No section numbers live in code.** Retrieval uses the tariff names,
>   descriptions, and keywords returned by discovery.
> - **No formula shapes live in code.** The extraction prompt describes a
>   *generic additive schema* (measurement basis + per-unit rate + brackets +
>   time component + multiplier) and asks the LLM which fields the document
>   actually populates. It does **not** prescribe formula patterns per tariff.
> - **No port-specific rate fallback exists.** If extraction fails the pipeline
>   surfaces `extraction_confidence="LOW"` with an explanatory note. It never
>   substitutes hardcoded numbers.
>
> Adapting the system to the Port of Rotterdam, Singapore MPA, or Port Houston
> requires **zero code changes** — point it at the new PDF.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUTS                                       │
│  • Any port tariff PDF                                               │
│  • VesselProfile (structured JSON)  OR  natural-language query       │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 0 — Ingestion                                                 │
│  document_processor.py: pdfplumber → pages → LangChain chunks        │
│  tariff_extractor.py:   chunks → OpenAI embeddings → ChromaDB index  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 1 — DISCOVERY                                                 │
│  discover_tariffs() retrieves a broad sample of the document and     │
│  asks the LLM to list EVERY chargeable tariff category the document  │
│  defines, returning DiscoveredTariffSet(currency, tariffs=[...]).    │
│                                                                      │
│  Each DiscoveredTariff carries: name, section_reference,             │
│  description, applies_to, retrieval_keywords.                        │
│                                                                      │
│  No tariff names live anywhere in code — this is the sole source of  │
│  truth for the document's taxonomy.                                  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 2 — EXTRACTION (parallel across discovered tariffs)           │
│  For each DiscoveredTariff:                                          │
│    • retrieve_tariff_context(): vector search on the tariff's own    │
│      name/description/keywords (not hardcoded section numbers)       │
│    • extract_tariff_rule(): OpenAI structured output → a fully       │
│      populated ExtractedTariffRule.                                  │
│                                                                      │
│  The extraction prompt describes the SCHEMA's building blocks        │
│  (measurement basis, per-unit rate, brackets, min/max, time          │
│  component, multiplier) and asks the LLM to pick the fields the      │
│  document actually uses.  It never says "pilotage = basic +          │
│  per-100-GT" or any other tariff-specific pattern.                   │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 3 — CALCULATION                                               │
│  calculator.apply_rule() is a single generic function.  It reads     │
│  basis (GT/NT/DWT/LOA/BEAM/DRAFT/CARGO_MT/NONE), applies brackets    │
│  OR simple per-unit rate with optional min/max bounds, adds any      │
│  per-day time component, multiplies by num_services.                 │
│                                                                      │
│  There are ZERO tariff-specific branches.  The same code path        │
│  handles light dues, wharfage, quay rental, conservancy charges,     │
│  or anything else the schema can express.                            │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                      CalculationResult
             (per-tariff line items + confidence
              + breakdown + currency + total)
```

---

## Project structure

```
Marcura/
├── src/
│   ├── core/
│   │   ├── config.py           # LLM models, chunking, retrieval knobs
│   │   │                       # (NO tariff taxonomy, NO section numbers)
│   │   ├── tariff_schema.py    # ExtractedTariffRule, RateBracket,
│   │   │                       # DiscoveredTariff, DiscoveredTariffSet
│   │   └── vessel_profile.py   # VesselProfile + typed sub-models
│   ├── ingestion/
│   │   ├── document_processor.py  # PDF → LangChain Document chunks
│   │   └── tariff_extractor.py    # discover_tariffs + extract_tariff_rule
│   │                              # + vector store + disk cache
│   ├── calculation/
│   │   └── calculator.py       # apply_rule (generic) + run_all_calculations
│   ├── agent/
│   │   ├── agent.py            # Discovery → extraction → calculation
│   │   └── query_parser.py     # NL vessel description → VesselProfile
│   ├── api/
│   │   └── api.py              # FastAPI: /calculate, /calculate/upload, /query
│   └── main.py                 # CLI: validate / calculate / query
├── tests/
│   ├── test_calculator.py      # 20 generic calculator tests
│   └── test_tariff_schema.py   # 15 schema tests
├── data/Port Tariff.pdf        # Sample document (SA Transnet — example only)
├── pyproject.toml              # uv, mypy-strict, pyright-strict
└── README.md
```

Removed vs the previous iteration:

| File / constant | Status | Reason |
|---|---|---|
| `fallback_rates.py` | **deleted** | Hardcoded Transnet rates. Replaced by LOW-confidence signalling. |
| `config.TARIFF_TYPES` | **removed** | Tariff taxonomy is now discovered per document. |
| `config.TARIFF_SECTION_KEYWORDS` | **removed** | Retrieval keywords come from discovery. |
| `_SECTION_HINTS` + `_section_matches` gate | **removed** | Forced LOW confidence if docs didn't match Transnet section numbers. |
| Tariff-specific patterns in the extraction prompt | **removed** | Replaced by a schema-describing prompt. |

---

## Setup

```bash
cd Marcura
uv sync
```

`OPENAI_API_KEY` is required (the pipeline is document-driven and ships no
rate fallback). Set it as a system environment variable or in `.env`.

---

## CLI usage

### 1. Validate against the published reference numbers

Runs the full pipeline (discovery → extraction → calculation) against the
bundled Transnet PDF and compares the result to Transnet's published numbers
for **SUDESTADA @ Durban**. Nothing hardcoded — this is a pure correctness
check that the generic pipeline reproduces a known answer.

```bash
uv run python src/main.py validate
uv run python src/main.py validate --rebuild   # force vector store rebuild
```

### 2. Custom vessel + custom tariff PDF

```bash
uv run python src/main.py calculate \
    --pdf /path/to/rotterdam_tariff.pdf \
    --vessel /path/to/vessel.json \
    --port "Rotterdam"
```

### 3. Natural-language query

```bash
uv run python src/main.py query \
    "SUDESTADA, 51300 GT bulk carrier at Durban, DWT 93274, 3.39 days"
```

### 4. FastAPI server

```bash
uv run python src/api/api.py
# http://localhost:8000/docs
```

Endpoints:
- `POST /calculate` — structured vessel data + PDF path
- `POST /calculate/upload` — multipart upload of the PDF
- `POST /query` — natural-language vessel description + PDF path
- `GET  /health`

---

## The schema, in one screen

```python
class ExtractedTariffRule(BaseModel):
    tariff_name: str                         # free-form, extracted verbatim
    section_reference: str = ""
    description: str = ""
    applicability_notes: str = ""
    currency: str = ""                       # e.g. "ZAR", "USD", "EUR"

    measurement_basis: Literal[              # pick whichever the document uses
        "GT", "NT", "DWT", "LOA", "BEAM", "DRAFT", "CARGO_MT", "NONE"
    ] = "NONE"

    # Simple per-unit rate (used when brackets is empty)
    basic_fee: float = 0.0
    rate_per_unit: float = 0.0
    unit_divisor: float = 1.0
    use_ceiling: bool = False
    minimum_fee: float = 0.0
    maximum_fee: float | None = None

    # Per-day accrual (e.g. port dues)
    time_rate_per_unit_per_day: float = 0.0

    # Step-rate schedule (overrides basic_fee + rate_per_unit when non-empty)
    brackets: list[RateBracket] = []

    # Multiplier
    num_services: float = 1.0
    services_basis: str = ""

    extraction_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    notes: str = ""
```

Every port tariff we tested against (South African Transnet, Rotterdam,
Singapore, US-style) fits this shape by turning knobs on or off:

| Tariff shape in the real world | Schema fields used |
|---|---|
| Flat fee per service | `basic_fee` only, `measurement_basis="NONE"` |
| Pure per-GT charge | `rate_per_unit`, `unit_divisor`, `measurement_basis="GT"` |
| Per-100-GT with ceiling (_"or part thereof"_) | add `use_ceiling=True` |
| Minimum/maximum bounded charge | `minimum_fee` / `maximum_fee` |
| Step-rate table (e.g. towage by GT) | `brackets` |
| Per-GT with per-day incremental (e.g. port dues) | `rate_per_unit` + `time_rate_per_unit_per_day` |
| Multi-service charges (lines × operations) | `num_services` + `services_basis` |
| Applicability filters (coastal/international, cargo vessels) | `applicability_notes` |

---

## OpenAI models

| Purpose | Model |
|---|---|
| Embeddings | `text-embedding-3-small` |
| Discovery + rule extraction + NL query parsing | `gpt-5.4` |

Both are configured in `src/core/config.py`.

---

## Caches

- **Vector store** (`.chroma_db/`) — indexed chunks of the PDF. Rebuild with
  `--rebuild` if you change the PDF or chunking parameters.
- **Discovery cache** (`.discovery_cache/`) — per-PDF tariff taxonomy. Keyed
  by PDF content hash.
- **Rule cache** (`.rule_cache/`) — per-PDF per-tariff per-port extracted
  rules. Keyed by PDF hash + tariff name + port. Only HIGH/MEDIUM confidence
  extractions are cached; LOW ones are always re-extracted.

Delete any of these directories to force re-processing.

---

## Type safety

- Strict `mypy` and `pyright` modes configured in `pyproject.toml`
- Runtime type-checking via `beartype` (`@beartype` on public functions)
- All schemas are Pydantic models — no `dict[str, Any]` on the data path

---

## Tests

```bash
uv run pytest tests/ -v
```

35 unit tests covering:
- The generic calculator (per-unit, brackets, time, min/max, services)
- Schema serialisation / validation
- Alternative measurement bases (LOA, CARGO_MT, flat fees)
- Edge cases (zero GT, ceiling rounding, maximum-fee cap)

> The Transnet reference rates used in the integration test are **inline test
> fixtures** inside `tests/test_calculator.py`. They exercise the calculator;
> they are never imported by production code.

---

## What the system does *not* do

- It does not ship any port's rates. Rates come from the PDF, via the LLM,
  every time.
- It does not assume any particular country or currency.
- It does not know what `pilotage` or `light dues` are. The LLM tells it.
- It does not need section numbers to be `1.1`, `3.3`, or any specific
  scheme — section references come from discovery and flow through to
  retrieval as keywords.
- It does not silently substitute fallback numbers on extraction failure.
  Failures surface as `confidence="LOW"` line items with explanatory notes.

---

## Change log

### v2.0 — Full generalisation pass

Goal of this release: close every path through the code that assumed a
specific port, a specific tariff taxonomy, or a specific formula shape. The
previous version was too tightly coupled to the South African Transnet
document that was only ever meant to be an *example* input.

**Removed (hardcoded coupling to the example document)**

- Deleted `src/calculation/fallback_rates.py` in full. The file contained
  Transnet ZAR rates encoded as `ExtractedTariffRule` objects and was
  silently substituted on LOW-confidence extractions, regardless of which
  port's PDF had been supplied. Removing it means the pipeline now succeeds
  or fails on the document itself — it never invents numbers.
- Removed `TARIFF_TYPES` from `src/core/config.py`. The pipeline no longer
  ships a fixed taxonomy of six tariff categories (`light_dues`, `vts_dues`,
  `pilotage_dues`, `towage_dues`, `running_lines`, `port_dues`). The set
  of chargeable tariffs is discovered from each document at runtime.
- Removed `TARIFF_SECTION_KEYWORDS` from `src/core/config.py`. Retrieval
  no longer steers towards Transnet-specific section numbers (`1.1`, `2.1`,
  `3.3`, `3.6`, `3.9`, `4.1`).
- Removed the `_SECTION_HINTS` dictionary and `_section_matches()` guard
  from `src/ingestion/tariff_extractor.py`. This guard used to downgrade
  any extraction whose section reference didn't match the Transnet layout
  to LOW confidence, which actively prevented the system from generalising.
- Removed the `--no-llm` CLI flag from `src/main.py` and the `use_llm`
  parameter from every `src/api/api.py` endpoint and request model. Both
  were switches into the deleted fallback rates path.
- Removed `get_fallback_rules()` and all `use_llm=False` branches from
  `src/agent/agent.py`.
- Rewrote the LLM extraction prompt in `src/ingestion/tariff_extractor.py`.
  The old prompt prescribed formula shapes per tariff category
  (*"Pilotage pattern → basic_fee = fixed base per service; rate_per_unit
  = per-100-GT add-on; unit_divisor = 100; use_ceiling = true;"*, and five
  similar blocks). The new prompt describes the schema's generic building
  blocks and asks the LLM to pick which fields the document actually
  populates. No tariff category is associated with any particular formula.

**Added (to make true generalisation work)**

- **Discovery stage.** `discover_tariffs()` in
  `src/ingestion/tariff_extractor.py` runs an LLM pass that returns a
  `DiscoveredTariffSet` — the document's own tariff taxonomy with
  per-tariff section references, applicability notes, and retrieval
  keywords. Cached on disk by PDF content hash in `.discovery_cache/`.
- **Dynamic retrieval.** `retrieve_tariff_context()` builds its vector
  search query from the `DiscoveredTariff`'s own name, description,
  applicability, section reference, and retrieval keywords. No keyword
  list lives in the repository.
- **Richer schema.** `src/core/tariff_schema.py` now exposes:
  - `measurement_basis` expanded to `GT | NT | DWT | LOA | BEAM | DRAFT | CARGO_MT | NONE`
  - `maximum_fee` (caps the per-service amount)
  - `currency` (extracted from the document, not assumed to be ZAR)
  - `applicability_notes` (coastal/international, vessel-type filters, etc.)
  - `services_basis` (short explanation of what `num_services` counts)
  - `RateBracket.rate_divisor` (brackets can step on any per-N unit, not
    just per-100)
  - `num_services` as `float` (supports fractional multipliers)
  - `DiscoveredTariff` / `DiscoveredTariffSet` models for the discovery stage
- **Generic calculator.** `src/calculation/calculator.py` now handles every
  new measurement basis, both min and max bounds, and bracket schedules
  with arbitrary per-unit divisors, in a single `apply_rule()` with zero
  tariff-specific branches.
- **Graceful failure surface.** LOW-confidence extractions are no longer
  swapped for hardcoded numbers. They appear in the result as line items
  with `confidence="LOW"` and an explanatory `notes` field so the caller
  can see exactly which tariff the system could not interpret.
- **New cache tier.** `.discovery_cache/` stores the per-document tariff
  taxonomy, alongside the existing per-rule `.rule_cache/`.

**Changed (callers and reporting)**

- `src/agent/agent.py` reorganised into three explicit phases
  (*discovery → extraction → calculation*) with a single public entry
  point, `run_agentic_calculation(vessel, vectorstore, pdf_path)`.
  The old `use_llm` switch is gone.
- `src/main.py`'s `validate` command now relies on the LLM pipeline
  end-to-end (there is no other mode) and uses fuzzy substring matching
  to line up discovered tariff names against the published reference
  numbers for SUDESTADA @ Durban, so differently-named discovered
  tariffs still produce a meaningful comparison.
- `src/api/api.py` version bumped to `2.0.0`; request/response shapes no
  longer carry `use_llm`. A 503 is returned if `OPENAI_API_KEY` is absent.
- `TariffLineItem` now carries `currency` and `confidence` fields; the
  textual report formatter prints the confidence next to every line item.
- Reference Transnet rates that were needed to keep the calculator
  integration test meaningful were moved **inline** into
  `tests/test_calculator.py` as a fixture. Nothing Transnet-specific
  exists in production code.

**Migration notes**

- `ExtractedTariffRule.tariff_type` was renamed to `tariff_name` to
  reflect that it is a free-form string extracted from the document,
  not a member of a fixed enum.
- `RateBracket` field names changed (`min_gt` → `min_value`,
  `max_gt` → `max_value`, `per_100_gt_above_min` → `rate_above_min`,
  plus new `rate_divisor`) so the bracket schedule can step on any
  measurement.
- `use_ceiling` now defaults to `False` instead of `True` — the previous
  default encoded a Transnet-ism (*"per 100 GT or part thereof"*) that
  does not hold in general.
- `measurement_basis` now defaults to `"NONE"` instead of `"GT"`, so
  tariffs with no vessel-measurement dependency serialise correctly.
- Delete `.chroma_db/`, `.rule_cache/`, and `.discovery_cache/` before
  the first run on this version — the cached artefacts from v1 use the
  old schema shape.
