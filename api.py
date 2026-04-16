from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent import format_results, run_agentic_calculation
from document_processor import process_tariff_pdf
from query_parser import parse_vessel_query
from tariff_extractor import build_vector_store
from vessel_profile import load_vessel_profile

app = FastAPI(
    title="Port Tariff Calculator API",
    description="Agentic RAG system for calculating port dues from tariff documents.",
    version="1.0.0",
)

# In-memory vector store cache keyed by PDF path.
# Fine for a single-process demo; replace with a proper cache layer for production.
_VECTORSTORE_CACHE: dict[str, Any] = {}


class CalculateFromPathRequest(BaseModel):
    tariff_pdf_path: str
    vessel_data: dict[str, Any]
    port: str
    use_llm: bool = True


class NaturalLanguageQueryRequest(BaseModel):
    query: str
    tariff_pdf_path: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "port-tariff-calculator"}


@app.post("/calculate")
def calculate_from_path(request: CalculateFromPathRequest) -> JSONResponse:
    """
    Calculate port dues given a tariff PDF path and vessel data.
    The PDF is indexed on the first call and cached for subsequent requests.
    """
    pdf_path = request.tariff_pdf_path

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")

    try:
        if pdf_path not in _VECTORSTORE_CACHE:
            chunks = process_tariff_pdf(pdf_path)
            _VECTORSTORE_CACHE[pdf_path] = build_vector_store(chunks)
        vectorstore = _VECTORSTORE_CACHE[pdf_path]

        vessel = load_vessel_profile(request.vessel_data, port=request.port)
        result = run_agentic_calculation(
            vessel, vectorstore, use_llm=request.use_llm, pdf_path=pdf_path,
        )
        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/upload")
async def calculate_from_upload(
    tariff_pdf: UploadFile = File(...),
    vessel_json: str = Form(...),
    port: str = Form(...),
    use_llm: bool = Form(True),
) -> JSONResponse:
    """
    Calculate port dues by uploading a tariff PDF directly.
    Accepts multipart/form-data: the PDF file plus vessel JSON as a form field.
    """
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
        result = run_agentic_calculation(
            vessel, vectorstore, use_llm=use_llm, pdf_path=tmp_path,
        )
        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/query")
def query_natural_language(request: NaturalLanguageQueryRequest) -> JSONResponse:
    """
    Accept a natural language vessel description and calculate port dues.

    The LLM parses the free-text query into structured vessel data, then the
    standard agentic RAG pipeline extracts tariff rules and calculates dues.

    Example query:
      "Calculate port dues for SUDESTADA, a 51300 GT bulk carrier at Durban,
       DWT 93274, NT 31192, LOA 229.2m, 3.39 days alongside, 2 operations"
    """
    pdf_path = request.tariff_pdf_path

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")

    try:
        vessel, port = parse_vessel_query(request.query)

        if pdf_path not in _VECTORSTORE_CACHE:
            chunks = process_tariff_pdf(pdf_path)
            _VECTORSTORE_CACHE[pdf_path] = build_vector_store(chunks)
        vectorstore = _VECTORSTORE_CACHE[pdf_path]

        result = run_agentic_calculation(
            vessel, vectorstore, use_llm=True, pdf_path=pdf_path,
        )
        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
