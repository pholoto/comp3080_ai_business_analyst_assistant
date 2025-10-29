"""LLM client abstraction with optional Ollama support."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

import requests

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1"


@dataclass
class LLMPrompt:
    """Container for a prompt block sent to the LLM."""

    role: str
    content: str


class LLMClient:
    """Abstract base class for LLM providers."""

    def generate(
        self,
        messages: Iterable[LLMPrompt],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        extra: Optional[Mapping[str, object]] = None,
    ) -> str:
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """Fallback LLM client that returns deterministic placeholder text."""

    def generate(
        self,
        messages: Iterable[LLMPrompt],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        extra: Optional[Mapping[str, object]] = None,
    ) -> str:
        last = ""
        for message in messages:
            if message.role == "user":
                last = message.content
        return (
            "[stub-model] Unable to contact external LLM. Input summary: "
            f"{last[:200]}"
        )


class OllamaLLMClient(LLMClient):
    """Client that talks to a local Ollama server."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self._check_server()

    def _check_server(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - environment check
            raise RuntimeError(
                "Ollama server not reachable. Ensure Ollama is running locally."
            ) from exc

    def generate(
        self,
        messages: Iterable[LLMPrompt],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        extra: Optional[Mapping[str, object]] = None,
    ) -> str:
        payload_messages = [
            {"role": prompt.role, "content": prompt.content} for prompt in messages
        ]
        options = {"temperature": temperature, "num_predict": max_tokens}
        if extra:
            # Merge supported options from extra, if provided.
            if "options" in extra and isinstance(extra["options"], Mapping):
                options.update(extra["options"])  # type: ignore[arg-type]
            if "model" in extra:
                self.model = str(extra["model"])
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": options,
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        return message.get("content", "")


def get_default_client() -> LLMClient:
    """Return the default LLM client, falling back to the stub client."""
    try:
        return OllamaLLMClient()
    except Exception:
        return StubLLMClient()
