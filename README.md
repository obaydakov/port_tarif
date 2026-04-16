# Port Tariff Calculator — Agentic RAG System

An AI-powered system that ingests any Port Tariff PDF and **a natural language vessel query** to automatically calculate all applicable port dues — **with no hardcoded rates in the calculation engine**.

Built as a take-home assessment for **Marcura**, demonstrating generalizable agentic RAG architecture for maritime document understanding.

---

## Architecture Overview

```
Natural Language Query          Structured JSON
"SUDESTADA, 51300 GT            vessel_profile.json
 bulk carrier at Durban..."
        |                              |
        v                              v
 ┌──────────────┐             ┌──────────────────────┐
 │ query_parser │ (LLM NLP)  │  vessel_profile.py   │
 │ -> VesselProfile           │  (Pydantic model)    │
 └──────┬───────┘             └──────────┬───────────┘
        │                                │
        └──────────┬─────────────────────┘
                   v
Port Tariff PDF
      │
      v
┌─────────────────────┐
│  document_processor │  pdfplumber -> text + tables -> LangChain Document chunks
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│  tariff_extractor   │  ChromaDB vector store (OpenAI embeddings)
│  (RAG layer)        │  -> semantic retrieval per tariff type
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│  tariff_extractor   │  gpt-5.4 extracts structured ExtractedTariffRule
│  (structured output)│  via OpenAI structured output (guaranteed schema)
└─────────┬───────────┘
          │              ┌─────────────────────────────┐
          │              │  .rule_cache/ (disk cache)   │
          │<------------>│  PDF hash + tariff + port    │
          │              └─────────────────────────────┘
          v
┌─────────────────────┐
│  agent.py           │  Parallel extraction (ThreadPoolExecutor)
│  (orchestrator)     │  6 tariff types extracted concurrently
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│  calculator.py      │  Generic apply_rule() engine
│  (no tariff-specific│  — operates entirely from ExtractedTariffRule schema
│   logic)            │
└─────────┬───────────┘
          │
          v
   CalculationResult
   (JSON + formatted report)
          │
    ┌─────┴──────┐
    │            │
  CLI          FastAPI
 main.py       api.py
```

### Design principles

**Natural language input** — users can describe a vessel in plain English. The LLM parses the query into structured data via OpenAI structured output, then the standard pipeline calculates all applicable dues.

**Schema-driven calculation** — the LLM reads the tariff PDF and populates an `ExtractedTariffRule` Pydantic schema (rates, brackets, formulas) via OpenAI structured output (guaranteed valid schema). A single generic `apply_rule()` function in `calculator.py` applies any rule to any vessel. No tariff-specific code exists in the calculator.

**Parallel extraction** — all 6 tariff types are extracted concurrently via `ThreadPoolExecutor`, cutting LLM extraction time by ~5x compared to sequential processing.

**Rule caching** — extracted rules are cached to disk by `(PDF hash, tariff type, port)`. Re-runs with the same PDF skip LLM extraction entirely.

**Graceful fallback** — if LLM extraction returns LOW confidence for a tariff (e.g. ambiguous PDF context), the system transparently substitutes hardcoded Transnet rates from `fallback_rates.py`. The calculator code path is identical for both.

**Document-agnostic** — adapting to a new port tariff document requires only re-indexing the PDF. No code changes are needed.

---

## Project Structure

```
Marcura/
├── tariff_schema.py       # ExtractedTariffRule + RateBracket Pydantic models (the schema contract)
├── fallback_rates.py      # Transnet-specific hardcoded rates as ExtractedTariffRule objects
├── config.py              # Model names, API keys, constants, section keywords
├── vessel_profile.py      # Pydantic vessel data model + derived properties
├── query_parser.py        # Natural language vessel query -> VesselProfile (LLM-powered)
├── document_processor.py  # PDF -> pdfplumber -> LangChain Document chunks
├── tariff_extractor.py    # ChromaDB vector store + LLM rule extraction (structured output)
├── calculator.py          # Generic apply_rule() engine — no tariff-specific logic
├── agent.py               # Pipeline orchestration (parallel) + report formatter
├── main.py                # CLI entry point (validate / calculate / query commands)
├── api.py                 # FastAPI REST endpoints (including /query for NL input)
├── tests/                 # Unit tests (pytest)
│   ├── test_calculator.py # 16 tests covering all tariff patterns + edge cases
│   └── test_tariff_schema.py # 10 tests for schema validation + serialization
├── requirements.txt       # Dependencies without version pins
├── pyproject.toml         # uv project config, mypy/pyright settings
├── .env                   # OPENAI_API_KEY (not committed)
└── Port Tariff.pdf        # Reference: Transnet SA tariff book (Apr 2024-Mar 2025)
```

---

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
cd Marcura

# Option 1: uv (recommended)
uv sync

# Option 2: pip (if uv is not installed)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### API Key

Create a `.env` file (or set a system environment variable):

```
OPENAI_API_KEY=your_key_here
```

> **Note:** The system runs fully without an API key using `--no-llm`. In this mode it uses the hardcoded Transnet fallback rates and produces identical numerical results. The `query` command always requires an API key (natural language parsing needs LLM).

---

## Usage

### CLI — Natural language vessel query

Describe a vessel in plain English and the system calculates all applicable dues:

```bash
uv run python main.py query \
  "Calculate port dues for SUDESTADA, a 51300 GT bulk carrier at Durban, \
   DWT 93274, NT 31192, LOA 229.2m, 3.39 days alongside, 2 operations"
```

The LLM parses the natural language into structured vessel data, then extracts tariff rules from the PDF and calculates all dues.

### CLI — Validate against reference vessel

Runs SUDESTADA (Bulk Carrier, GT 51,300) at Durban and compares against ground truth:

```bash
# Fallback mode — no API key needed, instant
uv run python main.py validate --no-llm

# Full agentic mode — LLM extracts rules from PDF (requires OPENAI_API_KEY)
uv run python main.py validate

# Force rebuild of the vector store (required after switching embedding models)
uv run python main.py validate --rebuild
```

### CLI — Custom vessel and port

```bash
uv run python main.py calculate \
  --pdf "Port Tariff.pdf" \
  --vessel my_vessel.json \
  --port "Cape Town" \
  --no-llm
```

**Vessel JSON format:**

```json
{
  "vessel_metadata": {
    "name": "MY VESSEL",
    "built_year": 2018,
    "flag": "PAN - Panama",
    "classification_society": "DNV",
    "call_sign": null
  },
  "technical_specs": {
    "imo_number": null,
    "type": "Container Ship",
    "dwt": 50000,
    "gross_tonnage": 35000,
    "net_tonnage": 20000,
    "loa_meters": 200.0,
    "beam_meters": 32.0,
    "moulded_depth_meters": 18.0,
    "lbp_meters": 195.0,
    "draft_sw_s_w_t": [12.0, 0.0, 0.0],
    "suez_gt": null,
    "suez_nt": null
  },
  "operational_data": {
    "cargo_quantity_mt": 20000,
    "days_alongside": 2.5,
    "arrival_time": "2024-11-15T08:00:00",
    "departure_time": "2024-11-17T14:00:00",
    "activity": "Loading Containers",
    "num_operations": 2,
    "num_holds": 5
  }
}
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Quick run
uv run pytest tests/ -q
```

### API Server

```bash
# Start server (port 8000)
uv run python api.py
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/query` | **Natural language vessel query** (requires OPENAI_API_KEY) |
| `POST` | `/calculate` | Calculate dues (structured JSON + tariff PDF path) |
| `POST` | `/calculate/upload` | Calculate dues (upload tariff PDF directly) |

**Interactive docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

**Example: Natural language query:**

```json
{
  "query": "Calculate port dues for SUDESTADA, a 51300 GT bulk carrier at Durban, DWT 93274, NT 31192, LOA 229.2m, 3.39 days alongside, 2 operations",
  "tariff_pdf_path": "D:\\CHI\\Marcura\\Port Tariff.pdf"
}
```

**Example: Structured JSON request:**

```json
{
  "tariff_pdf_path": "D:\\CHI\\Marcura\\Port Tariff.pdf",
  "vessel_data": {
    "vessel_metadata": { "name": "SUDESTADA", "built_year": 2010, "flag": "MLT - Malta" },
    "technical_specs": { "type": "Bulk Carrier", "dwt": 93274, "gross_tonnage": 51300, "net_tonnage": 31192, "loa_meters": 229.2, "beam_meters": 38.0, "draft_sw_s_w_t": [14.9, 0.0, 0.0] },
    "operational_data": { "days_alongside": 3.39, "num_operations": 2, "activity": "Exporting Iron Ore" }
  },
  "port": "Durban",
  "use_llm": false
}
```

**Example response:**

```json
{
  "vessel": "SUDESTADA",
  "port": "Durban",
  "currency": "ZAR",
  "total": 506897.43,
  "line_items": [
    { "tariff": "light_dues",    "amount": 60062.04,  "formula": "(513 x R117.08) x 1" },
    { "tariff": "vts_dues",      "amount": 33345.00,  "formula": "(max(51300 x R0.65, R235.52)) x 1" },
    { "tariff": "pilotage_dues", "amount": 47189.94,  "formula": "(R18608.61 + 513 x R9.72) x 2" },
    { "tariff": "towage_dues",   "amount": 147074.38, "formula": "Bracket(GT) x 2 services" },
    { "tariff": "running_lines", "amount": 19854.72,  "formula": "(R1654.56) x 12" },
    { "tariff": "port_dues",     "amount": 199371.35, "formula": "(513 x R192.73 + 513 x R57.79/day x 3.39d) x 1" }
  ]
}
```

---

## Validation Results

Tested against the reference vessel (SUDESTADA, Bulk Carrier at Durban) from the assessment spec:

| Tariff | Calculated (ZAR) | Ground Truth (ZAR) | Error |
|---|---|---|---|
| Light Dues | 60,062.04 | 60,062.04 | **0.00%** |
| VTS Dues | 33,345.00 | 33,315.75 | +0.09% |
| Pilotage Dues | 47,189.94 | 47,189.94 | **0.00%** |
| Towage Dues | 147,074.38 | 147,074.38 | **0.00%** |
| Running Lines | 19,854.72 | 19,639.50 | +1.10% |
| Port Dues | 199,371.35 | 199,549.22 | -0.09% |
| **TOTAL** | **506,897.43** | **506,830.83** | **+0.01%** |

3 tariffs are exact, all 6 are within +/-1.1% of ground truth (which the spec labels as "Approx.").

---

## Supported Tariff Types

| Tariff | Tariff Book Section | Calculation Basis |
|---|---|---|
| Light Dues | Section 1.1 | R117.08 per 100 GT (non-coastal vessels) |
| VTS Dues | Section 2.1 | R0.65/GT at Durban/Saldanha, R0.54/GT elsewhere |
| Pilotage Dues | Section 3.3 | Port-specific basic fee + R/100GT x services |
| Towage Dues | Section 3.6 | GT-bracket base fee x services |
| Running Lines | Section 3.9 | R1,654.56/service (Other Ports), 6 services/call |
| Port Dues | Section 4.1 | Basic R192.73/100GT + R57.79/100GT/day (pro-rata) |

---

## OpenAI Models Used

| Purpose | Model |
|---|---|
| Embeddings (vector store) | `text-embedding-3-small` |
| Tariff rule extraction + NL query parsing | `gpt-5.4` |

> The system uses **OpenAI structured output** (`with_structured_output`) for both tariff rule extraction and natural language query parsing. This guarantees valid Pydantic schema output — no manual JSON parsing needed.

> The system runs fully without an API key using `--no-llm`. Hardcoded Transnet fallback rates in `fallback_rates.py` are applied through the same generic calculator — identical code path.

---

## Testing

26 unit tests covering all tariff calculation patterns, edge cases, and schema validation:

```bash
uv run pytest tests/ -v
```

| Test Suite | Tests | Coverage |
|---|---|---|
| `test_calculator.py` | 16 | All 6 tariff patterns, minimum fees, brackets, ceiling/no-ceiling, zero GT, time components, error handling |
| `test_tariff_schema.py` | 10 | Defaults, Literal validation, serialization round-trips, model_copy |

---

## Future Roadmap

### Accuracy improvements

- **Better running lines formula** — integrate a lookup table keyed on vessel LOA and cargo type to replace the fixed 6-services heuristic; this would bring the +1.10% error to near-zero
- **Surcharge engine** — currently surcharges (outside working hours, pilot not ready within 30min, vessel late by >30min) are not applied; a time-of-day + event-based surcharge module would handle these
- **Cargo dues** — Section 7 of the tariff book covers cargo dues (dry bulk, breakbulk, containers) by cargo tonnage; adding this would complete the full port cost estimate
- **Berth dues** — Section 4.1.2 berth dues apply to vessels not handling cargo or undergoing repairs; currently excluded

### Architecture improvements

- **PDF table parser hardening** — the current pdfplumber extraction occasionally misaligns multi-column tables; a dedicated table-to-JSON normaliser (e.g. using Camelot or a vision LLM) would improve LLM extraction confidence and reduce fallback rate
- **Multi-document support** — allow multiple tariff PDFs (e.g. per-port supplements) to be indexed into the same vector store, with port-based routing at query time
- **Tariff version management** — track tariff book version dates so the system automatically applies the correct rate schedule based on vessel arrival date

### Operational improvements

- **Async API** — for large fleets, accept a batch of vessel profiles and return results asynchronously via a webhook or polling endpoint
- **Database persistence** — store calculation history in PostgreSQL so results can be audited, compared across tariff versions, and exported to CSV/Excel
- **Authentication** — add API key or OAuth2 authentication to the FastAPI endpoints for production deployment

### CI/CD & Deployment

- **CI pipeline** — GitHub Actions workflow running `pytest`, `mypy`, and `pyright` on every PR; gate merges on test pass + type-check clean
- **Docker** — single-stage `Dockerfile` with `uv sync --frozen` for reproducible builds; expose port 8000 for the FastAPI server
- **Cloud deployment** — containerised API deployable to AWS ECS/Fargate, GCP Cloud Run, or Azure Container Apps; stateless design (ChromaDB + rule cache can be mounted as a volume or replaced with managed vector DB)
- **Secrets management** — `OPENAI_API_KEY` injected via environment variable (AWS Secrets Manager, GCP Secret Manager, or Kubernetes secrets) — never baked into the image
- **Monitoring** — structured JSON logging + health check endpoint (`GET /health`) ready for load balancer probes; add OpenTelemetry tracing for LLM call latency visibility

### Integration possibilities

- **Vessel data enrichment** — auto-fetch GT/NT/LOA from Marine Traffic or Equasis using IMO number, removing the need to supply full technical specs manually
- **Multi-currency output** — convert ZAR results to USD/EUR/GBP using live FX rates
- **Port comparison** — given a vessel and cargo, rank ports by total cost to help route planning
