from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (development convenience — production uses system env vars)
_env_path = Path(__file__).parent / ".env"
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
CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), ".chroma_db")
CHROMA_COLLECTION_NAME: str = "port_tariffs"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1500
CHUNK_OVERLAP: int = 200
TOP_K_RETRIEVAL: int = 8

# ---------------------------------------------------------------------------
# Tariff types the system must calculate
# ---------------------------------------------------------------------------
TARIFF_TYPES: list[str] = [
    "light_dues",
    "vts_dues",
    "pilotage_dues",
    "towage_dues",
    "running_lines",
    "port_dues",
]

# ---------------------------------------------------------------------------
# Section keywords for smarter retrieval
# ---------------------------------------------------------------------------
TARIFF_SECTION_KEYWORDS: dict[str, list[str]] = {
    "light_dues":    ["light dues", "section 1", "1.1"],
    "vts_dues":      ["vessel traffic", "VTS", "section 2", "2.1"],
    "pilotage_dues": ["pilotage", "pilot", "section 3.3", "3.3"],
    "towage_dues":   ["tug", "towage", "craft assistance", "section 3.6", "3.6"],
    "running_lines": ["running of vessel lines", "mooring", "section 3.9", "3.9"],
    "port_dues":     ["port dues", "port fees", "section 4.1", "4.1"],
}
