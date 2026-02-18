"""
Configuration module for the PDF RAG agent.

To adapt to a different document base:
  1. Change PDF_DIRECTORY to point to your new folder
  2. Change COLLECTION_NAME to a unique name for the new collection
  3. (Optional) Adjust CHUNK_SIZE / CHUNK_OVERLAP for your documents
  4. Re-run indexing via `python main.py --reindex`
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Document source -- CHANGE THESE to switch document bases
# ---------------------------------------------------------------------------
PDF_DIRECTORY: str = r"C:\Users\binya\Desktop\CME295"
COLLECTION_NAME: str = "cme295_docs"

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1000        # characters per chunk
CHUNK_OVERLAP: int = 200      # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = str(Path(__file__).resolve().parent.parent / "chroma_db")

# ---------------------------------------------------------------------------
# Embedding model (runs locally via sentence-transformers)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# LLM -- uses OpenAI-compatible API (works with OpenAI, Azure, local, etc.)
# ---------------------------------------------------------------------------
LLM_MODEL: str = "gpt-4o-mini"   # or "gpt-4o", "gpt-3.5-turbo", etc.
LLM_TEMPERATURE: float = 0.2

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = 5                # number of chunks to retrieve per query

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS: int = 10   # max Q/A pairs kept in memory
