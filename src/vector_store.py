"""
Vector store module (ChromaDB + sentence-transformers).

Responsibilities:
  - Create / load a persisted ChromaDB collection
  - Index document chunks with their embeddings
  - Perform similarity search for a query
"""

from __future__ import annotations

import logging
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding function (cached at module level to avoid reloading the model)
# ---------------------------------------------------------------------------
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return (and cache) the embedding model."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model '%s' ...", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


# ---------------------------------------------------------------------------
# ChromaDB wrapper
# ---------------------------------------------------------------------------

def get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.Client(
        Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False,
            is_persistent=True,
        )
    )


def build_index(chunks: List[Document]) -> Chroma:
    """Embed *chunks* and persist them into a ChromaDB collection.

    If the collection already exists it is **deleted and rebuilt**.
    """
    embeddings = get_embeddings()
    logger.info(
        "Indexing %d chunks into collection '%s' ...", len(chunks), COLLECTION_NAME
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info("Indexing complete. Persisted to %s", CHROMA_PERSIST_DIR)
    return vector_store


def load_index() -> Chroma:
    """Load an existing ChromaDB collection (must have been built first)."""
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    count = vector_store._collection.count()
    if count == 0:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' is empty. Run indexing first "
            f"(python main.py --reindex)."
        )
    logger.info("Loaded collection '%s' with %d vectors.", COLLECTION_NAME, count)
    return vector_store


def retrieve(query: str, vector_store: Chroma | None = None, top_k: int = TOP_K) -> List[Document]:
    """Return the *top_k* most relevant chunks for *query*."""
    if vector_store is None:
        vector_store = load_index()
    results = vector_store.similarity_search(query, k=top_k)
    return results
