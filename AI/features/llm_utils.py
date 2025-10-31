"""Helpers to interact with the LLM client."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Sequence

from ..llm import LLMClient, LLMPrompt
from ..memory import Session


def request_json_response(
    llm: LLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    default_title: str,
    history: Sequence[dict] | None = None,
) -> Dict[str, Any]:
    """Ask the LLM for a JSON object, falling back to raw text if decoding fails."""
    prompts: list[LLMPrompt] = [LLMPrompt(role="system", content=system_prompt)]
    if history:
        prompts.extend(
            LLMPrompt(role=entry.get("role", "assistant"), content=entry.get("content", ""))
            for entry in history
        )
    prompts.append(LLMPrompt(role="user", content=user_prompt))
    raw = llm.generate(
        prompts,
        extra={"response_format": {"type": "json_object"}},
    )
    if not raw:
        return {"title": default_title, "summary": ""}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = {"title": default_title, "summary": raw}
    if "title" not in data:
        data["title"] = default_title
    return data


def build_attachment_context(
    session: Session,
    *,
    char_limit: int = 2000,
) -> str:
    """Return a concise textual digest of the session's attachments."""
    digest = session.attachment_digest(char_limit=char_limit)
    if not digest:
        return "No supporting documents attached."
    return (
        f"Chunking strategy: {session.chunking_strategy}\n"
        f"Indexing strategy: {session.indexing_strategy}\n"
        f"Attached documents:\n{digest}"
    )
