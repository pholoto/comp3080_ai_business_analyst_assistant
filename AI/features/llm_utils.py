"""Helpers to interact with the LLM client."""
from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any, Dict, Sequence

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
    cleaned = _sanitize_model_output(raw)
    data = _load_json_loose(cleaned)
    if data is None:
        # Fallback to whatever the model returned after removing thinking traces
        return {"title": default_title, "summary": cleaned}
    if "title" not in data:
        data["title"] = default_title
    return data


def _sanitize_model_output(text: str) -> str:
    if not text:
        return ""
    # Remove thinking blocks the provider may emit.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip markdown fences and conversational prefixes.
    cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()
    return cleaned


def _load_json_loose(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except (JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    # Try parsing from the first JSON object occurrence in the text.
    brace_index = text.find("{")
    if brace_index == -1:
        return None
    snippet = text[brace_index:]
    try:
        obj, _ = decoder.raw_decode(snippet)
        if isinstance(obj, dict):
            return obj
    except JSONDecodeError:
        pass
    # Attempt to locate JSON wrapped inside larger text via regex.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        candidate = match.group(0)
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except JSONDecodeError:
        return None
    return None


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
