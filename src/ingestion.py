"""
PDF ingestion module.

Responsibilities:
  - Discover PDF files in a directory
  - Extract text from each PDF (with page-level metadata)
  - Split text into overlapping chunks ready for embedding
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIRECTORY

logger = logging.getLogger(__name__)


def discover_pdfs(directory: str | Path | None = None) -> List[Path]:
    """Return a sorted list of PDF file paths found in *directory*."""
    directory = Path(directory or PDF_DIRECTORY)
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory not found: {directory}")
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {directory}")
    logger.info("Found %d PDF(s) in %s", len(pdfs), directory)
    return pdfs


def load_pdfs(directory: str | Path | None = None) -> List[Document]:
    """Load all PDFs from *directory* and return a flat list of Documents."""
    pdf_paths = discover_pdfs(directory)
    documents: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = path.name
        documents.extend(docs)
        logger.info("  Loaded %s (%d pages)", path.name, len(docs))
    logger.info("Total pages loaded: %d", len(documents))
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d pages into %d chunks", len(documents), len(chunks))
    return chunks


def ingest(directory: str | Path | None = None) -> List[Document]:
    """Full pipeline: discover -> load -> chunk.  Returns list of chunks."""
    docs = load_pdfs(directory)
    return chunk_documents(docs)
