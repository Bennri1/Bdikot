"""
Main entry point for the PDF Conversational RAG Agent.

Usage:
    python main.py              # Start chatting (index must exist)
    python main.py --reindex    # (Re)build the vector index then chat
    python main.py --reindex --pdf-dir /path/to/pdfs  # Index a custom folder
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import COLLECTION_NAME, PDF_DIRECTORY


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_indexing(pdf_dir: str) -> None:
    """Run the full ingestion + indexing pipeline."""
    from src.ingestion import ingest
    from src.vector_store import build_index

    print(f"\n=== Indexing PDFs from: {pdf_dir} ===")
    print(f"    Collection name  : {COLLECTION_NAME}")
    chunks = ingest(pdf_dir)
    build_index(chunks)
    print(f"=== Indexing complete ({len(chunks)} chunks) ===\n")


def run_chat() -> None:
    """Interactive chat loop."""
    from src.agent import ConversationalRAGAgent

    print("\n=== PDF Conversational Agent ===")
    print("Type your questions below. Commands:")
    print("  /reset   - clear conversation history")
    print("  /quit    - exit\n")

    agent = ConversationalRAGAgent()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break
        if question.lower() == "/reset":
            agent.reset()
            print("(conversation history cleared)\n")
            continue

        try:
            answer = agent.ask(question)
            print(f"\nAssistant: {answer}\n")
        except Exception as exc:
            logging.getLogger(__name__).error("Error: %s", exc, exc_info=True)
            print(f"\n[Error] {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Conversational RAG Agent")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="(Re)build the vector index before chatting.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=PDF_DIRECTORY,
        help=f"Path to the folder containing PDF files (default: {PDF_DIRECTORY}).",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Build the index and exit (no chat).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.reindex or args.index_only:
        run_indexing(args.pdf_dir)
        if args.index_only:
            sys.exit(0)

    run_chat()


if __name__ == "__main__":
    main()
