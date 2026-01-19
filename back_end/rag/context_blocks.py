"""Prompt context formatting helpers for the backend RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .retrieval import RetrievalResult


@dataclass(frozen=True)
class PromptContext:
    """Structured data describing the context passed to the LLM."""

    text_block: str
    used_chunks: List[dict[str, object]]


def build_context_block(
    contexts: Sequence[RetrievalResult],
    *,
    max_chars: int,
) -> PromptContext:
    """Construct the context block string respecting a max character budget."""
    budget = max(500, max_chars)
    buffer: List[str] = []
    used: List[dict[str, object]] = []
    consumed = 0
    for idx, result in enumerate(contexts, start=1):
        source_name = result.metadata.get("source_name") or result.metadata.get(
            "original_name", "source"
        )
        chunk_index = result.metadata.get("chunk_index", idx)
        header = f"[Source {idx} | Score {result.score:.3f} | {source_name}]\n"
        body = result.text.strip()
        entry = header + body
        entry_len = len(entry)
        if consumed + entry_len > budget and buffer:
            break
        buffer.append(entry)
        consumed += entry_len
        used.append(
            {
                "chunk_id": result.metadata.get("chunk_id"),
                "document_id": result.metadata.get("document_id"),
                "source_name": source_name,
                "chunk_index": chunk_index,
                "score": result.score,
                "char_start": result.metadata.get("char_start"),
                "char_end": result.metadata.get("char_end"),
                "text": body,
            }
        )
    if not buffer:
        return PromptContext(text_block="[no matching context retrieved]", used_chunks=[])
    return PromptContext(text_block="\n\n".join(buffer), used_chunks=used)


def build_user_message(
    *,
    question: str,
    context_block: str,
    task_prompt: str,
    guardrails: str | None = None,
) -> str:
    """Format the user message combining question, context, and guardrails."""
    guardrail_text = guardrails.strip() if guardrails else ""
    user_message = (
        f"User question:\n{question.strip()}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Task:\n{task_prompt.strip()}"
    )
    if guardrail_text:
        user_message += f"\n\nSafety rules:\n{guardrail_text}"
    return user_message


__all__ = [
    "PromptContext",
    "build_context_block",
    "build_user_message",
]
