from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from beartype import beartype
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass(frozen=True, slots=True)
class PageContent:
    """Typed representation of a single extracted PDF page."""

    page: str
    text: str

# ---------------------------------------------------------------------------
# Generic numbered-section header pattern.
#
# Matches either ``SECTION <n>`` or ``<n>.<m>[.<k>] <Capital>`` — the shape
# used by virtually every regulatory / tariff document we have encountered.
# This is only used to tag page metadata so discovery can reference sections
# by name; it does NOT gate retrieval, and the pipeline works even when no
# section match is found (``current_section`` falls back to "unknown").
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r"(SECTION\s+\d+|^\d+\.\d+(?:\.\d+)?\s+[A-Z])", re.MULTILINE | re.IGNORECASE
)


@beartype
def extract_text_from_pdf(pdf_path: str) -> list[PageContent]:
    """
    Extract text and table content from a tariff PDF using pdfplumber.
    Returns a list of PageContent objects, one per PDF page.
    """
    pages: list[PageContent] = []
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

                # Extract tables and convert them to readable markdown-style text
                table_text = ""
                tables = page.extract_tables()
                for table in tables:
                    rows: list[str] = []
                    for row in table:
                        cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    table_text += "\n" + "\n".join(rows) + "\n"

                combined = page_text + ("\n\nTABLE DATA:\n" + table_text if table_text.strip() else "")
                pages.append(PageContent(page=str(i), text=combined))
    except Exception as e:
        print(f"[document_processor] Error extracting PDF: {e}")
        raise

    print(f"[document_processor] Extracted {len(pages)} pages from {path.name}")
    return pages


@beartype
def build_documents(pages: list[PageContent], source: str = "") -> list[Document]:
    """
    Convert PageContent objects into LangChain Documents with metadata.
    Detects section headers and tags each chunk accordingly.
    """
    docs: list[Document] = []
    current_section = "unknown"

    for page_data in pages:
        text = page_data.text
        page_num = page_data.page

        # Detect the dominant section on this page
        section_match = _SECTION_RE.search(text)
        if section_match:
            current_section = section_match.group(0).strip()[:80]

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page_num,
                    "section": current_section,
                },
            )
        )

    return docs


@beartype
def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for vector indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[document_processor] Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks


@beartype
def process_tariff_pdf(pdf_path: str) -> list[Document]:
    """Full pipeline: PDF → text → LangChain Document chunks."""
    pages = extract_text_from_pdf(pdf_path)
    docs = build_documents(pages, source=pdf_path)
    return chunk_documents(docs)
