"""LLM-powered generator that converts retrieved context into answers."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .retrieval import RetrievalResult


@dataclass
class StubPrompt:
    """Minimal prompt container used when the main LLM client is unavailable."""

    role: str
    content: str


class StubLLMClient:
    """Fallback client that returns deterministic placeholder text."""

    def generate(self, messages: Iterable[StubPrompt], **_: object) -> str:
        last_user = ""
        for message in messages:
            if message.role == "user":
                last_user = message.content
        return (
            "[stub-llm] Unable to reach the primary language model. "
            "Context snapshot: "
            f"{last_user[:400]}"
        )


class ResponseGenerator:
    """Generate responses from the LLM using retrieved context."""

    def __init__(self, llm_client=None) -> None:
        prompt_cls, default_client_fn = self._load_llm_dependencies()
        self._prompt_cls = prompt_cls
        self._default_client_fn = default_client_fn
        self.llm_client = llm_client or self._default_client_fn()

    def generate(
        self,
        *,
        query: str,
        contexts: Sequence[RetrievalResult],
        system_prompt: str,
        task_prompt: str,
        guardrails: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """Generate a response using the provided context."""
        context_block = self._format_context(contexts)
        guardrail_text = guardrails.strip() if guardrails else ""
        user_message = (
            f"User question:\n{query}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            f"Task:\n{task_prompt.strip()}"
        )
        if guardrail_text:
            user_message += f"\n\nSafety rules:\n{guardrail_text}"
        messages = [
            self._prompt_cls(
                role="system",
                content=system_prompt.strip(),
            ),
            self._prompt_cls(
                role="user",
                content=user_message,
            ),
        ]
        return self.llm_client.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def call_llm(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """Directly invoke the underlying LLM with explicit prompt strings."""
        messages = [
            self._prompt_cls(role="system", content=system_prompt.strip()),
            self._prompt_cls(role="user", content=user_message.strip()),
        ]
        return self.llm_client.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _format_context(self, contexts: Sequence[RetrievalResult]) -> str:
        if not contexts:
            return "[no matching context retrieved]"
        blocks = []
        for index, result in enumerate(contexts, start=1):
            source_name = result.metadata.get("source_name") or result.metadata.get(
                "original_name"
            ) or result.metadata.get("document_id", "unknown")
            block = (
                f"[Source {index} | Score {result.score:.3f} | {source_name}]\n"
                f"{result.text}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    @staticmethod
    def _load_llm_dependencies():
        try:
            module = importlib.import_module("AI.llm.client")
            prompt_cls = getattr(module, "LLMPrompt")
            get_default_client = getattr(module, "get_default_client")
            return prompt_cls, get_default_client
        except Exception:  # pragma: no cover - fallback path
            return StubPrompt, (lambda: StubLLMClient())


__all__ = ["ResponseGenerator"]
