"""
Conversational RAG agent.

Responsibilities:
  - Maintain conversation history (sliding window)
  - Reformulate the user question using history for better retrieval
  - Retrieve relevant chunks via the vector store
  - Generate an answer grounded in the retrieved context
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, MAX_HISTORY_TURNS, TOP_K
from src.vector_store import load_index, retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CONDENSE_PROMPT = """\
Given the following conversation history and a new user question, \
reformulate the question so that it is self-contained (i.e., \
understandable without the conversation history). \
If the question is already self-contained, return it as-is. \
Return ONLY the reformulated question, nothing else.

Conversation history:
{history}

New question: {question}

Reformulated question:"""

QA_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on the provided \
document excerpts. Follow these rules:
1. Base your answer ONLY on the provided context. If the context does not \
   contain enough information, say so explicitly.
2. Cite the source file and page number when possible (e.g., [source.pdf, p.3]).
3. Be concise but thorough.
4. Answer in the same language the user used for their question."""

QA_USER_PROMPT = """\
Context (retrieved from documents):
---
{context}
---

Question: {question}"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

@dataclass
class ConversationalRAGAgent:
    """Stateful conversational agent backed by a vector store."""

    vector_store: Chroma | None = None
    llm: ChatOpenAI | None = None
    history: List[Tuple[str, str]] = field(default_factory=list)
    top_k: int = TOP_K
    max_history: int = MAX_HISTORY_TURNS

    def __post_init__(self) -> None:
        if self.vector_store is None:
            self.vector_store = load_index()
        if self.llm is None:
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
            )

    # ----- helpers --------------------------------------------------------

    def _format_history(self) -> str:
        if not self.history:
            return "(no prior conversation)"
        lines: list[str] = []
        for user_msg, ai_msg in self.history:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {ai_msg}")
        return "\n".join(lines)

    def _condense_question(self, question: str) -> str:
        """Use the LLM to make *question* standalone given the history."""
        if not self.history:
            return question
        prompt = CONDENSE_PROMPT.format(
            history=self._format_history(),
            question=question,
        )
        response = self.llm.invoke(prompt)
        condensed = response.content.strip()
        logger.debug("Condensed question: %s", condensed)
        return condensed

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
        return "\n\n".join(parts)

    # ----- public API -----------------------------------------------------

    def ask(self, question: str) -> str:
        """Process a user question and return the agent's answer."""
        # 1. Condense the question using conversation history
        standalone_q = self._condense_question(question)

        # 2. Retrieve relevant chunks
        docs = retrieve(standalone_q, vector_store=self.vector_store, top_k=self.top_k)
        context = self._format_context(docs)

        # 3. Build messages for the LLM
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": QA_USER_PROMPT.format(
                context=context, question=standalone_q
            )},
        ]

        # 4. Generate answer
        response = self.llm.invoke(messages)
        answer = response.content.strip()

        # 5. Update history (sliding window)
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return answer

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared.")
