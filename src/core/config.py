from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (development convenience — production uses system env vars)
# __file__ is src/core/config.py → parent.parent.parent is the project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)

# ---------------------------------------------------------------------------
# LLM / API
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = "gpt-5.4"
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = str(Path(__file__).parent.parent.parent / ".chroma_db")
CHROMA_COLLECTION_NAME: str = "port_tariffs"

# ---------------------------------------------------------------------------
# Chunking + retrieval
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1500
CHUNK_OVERLAP: int = 200

# Per-tariff retrieval depth (used when extracting one rule)
TOP_K_RETRIEVAL: int = 8

# Retrieval depth for the discovery stage (scanning the whole document
# for the set of chargeable tariff categories it defines)
DISCOVERY_TOP_K: int = 40

# Safety cap on how many distinct tariff categories discovery can return
# (prevents runaway extraction against pathological PDFs)
MAX_DISCOVERED_TARIFFS: int = 30

# ---------------------------------------------------------------------------
# NOTE: This file used to contain TARIFF_TYPES and TARIFF_SECTION_KEYWORDS
# constants that hardcoded the six Transnet tariffs and their section numbers.
# Both were removed: the pipeline now discovers tariff categories from the
# document at runtime via ``discover_tariffs`` in ingestion/tariff_extractor.py.
# No tariff names, section numbers, or port-specific keywords live here.
# ---------------------------------------------------------------------------
