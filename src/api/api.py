from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure src/ is on sys.path when this file is run directly (python src/api/api.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.agent import run_agentic_calculation
from agent.query_parser import parse_vessel_query
from core.config import OPENAI_API_KEY
from core.vessel_profile import load_vessel_profile
from ingestion.document_processor import process_tariff_pdf
from ingestion.tariff_extractor import build_vector_store

app = FastAPI(
    title="Port Tariff Calculator API",
    description=(
        "Generalisable agentic RAG system for calculating port dues from any "
        "port tariff document. No tariff names or rates are hardcoded: the "
        "pipeline discovers the document's taxonomy at runtime and extracts "
        "each rule into a generic schema."
    ),
    version="2.0.0",
)

# In-memory vector store cache keyed by PDF path.
# Fine for a single-process demo; replace with a proper cache layer for production.
_VECTORSTORE_CACHE: dict[str, Any] = {}


class CalculateFromPathRequest(BaseModel):
    tariff_pdf_path: str
    vessel_data: dict[str, Any]
    port: str


class NaturalLanguageQueryRequest(BaseModel):
    query: str
    tariff_pdf_path: str


def _require_api_key() -> None:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "OPENAI_API_KEY is not configured. This service is document-driven "
                "and ships no hardcoded rate fallback."
            ),
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "port-tariff-calculator"}


@app.post("/calculate")
def calculate_from_path(request: CalculateFromPathRequest) -> JSONResponse:
    """Calculate port dues given a tariff PDF path and structured vessel data."""
    _require_api_key()
    pdf_path = request.tariff_pdf_path

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")

    try:
        if pdf_path not in _VECTORSTORE_CACHE:
            chunks = process_tariff_pdf(pdf_path)
            _VECTORSTORE_CACHE[pdf_path] = build_vector_store(chunks)
        vectorstore = _VECTORSTORE_CACHE[pdf_path]

        vessel = load_vessel_profile(request.vessel_data, port=request.port)
        result = run_agentic_calculation(vessel, vectorstore, pdf_path=pdf_path)
        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        print(f"[api] /calculate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/upload")
async def calculate_from_upload(
    tariff_pdf: UploadFile = File(...),
    vessel_json: str = Form(...),
    port: str = Form(...),
) -> JSONResponse:
    """Calculate port dues by uploading a tariff PDF directly (multipart/form-data)."""
    _require_api_key()
    try:
        vessel_data: dict[str, Any] = json.loads(vessel_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid vessel JSON: {e}")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await tariff_pdf.read())
            tmp_path = tmp.name

        chunks = process_tariff_pdf(tmp_path)
        vectorstore = build_vector_store(chunks, force_rebuild=True)
        vessel = load_vessel_profile(vessel_data, port=port)
        result = run_agentic_calculation(vessel, vectorstore, pdf_path=tmp_path)
        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        print(f"[api] /calculate/upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                print(f"[api] Could not delete temp file {tmp_path}: {e}")


@app.post("/query")
def query_natural_language(request: NaturalLanguageQueryRequest) -> JSONResponse:
    """Accept a natural language vessel description and calculate port dues.

    The LLM parses the free-text query into structured vessel data, then the
    standard agentic RAG pipeline discovers tariffs, extracts rules, and
    calculates dues.
    """
    _require_api_key()
    pdf_path = request.tariff_pdf_path

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")

    try:
        vessel, _port = parse_vessel_query(request.query)

        if pdf_path not in _VECTORSTORE_CACHE:
            chunks = process_tariff_pdf(pdf_path)
            _VECTORSTORE_CACHE[pdf_path] = build_vector_store(chunks)
        vectorstore = _VECTORSTORE_CACHE[pdf_path]

        result = run_agentic_calculation(vessel, vectorstore, pdf_path=pdf_path)
        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[api] /query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
