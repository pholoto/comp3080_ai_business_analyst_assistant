"""LLM client abstraction with optional MLVoca support."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import requests

DEFAULT_MLVOCA_BASE_URL = "https://mlvoca.com/api"
DEFAULT_MLVOCA_MODEL = "deepseek-r1:1.5b"


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


class FailoverLLMClient(LLMClient):
    """LLM client that wraps a primary backend with a resilient stub fallback."""

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self.primary = primary
        self.fallback = fallback
        self._logger = logging.getLogger(__name__)

    def generate(
        self,
        messages: Iterable[LLMPrompt],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        extra: Optional[Mapping[str, object]] = None,
    ) -> str:
        try:
            return self.primary.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )
        except Exception as exc:
            self._logger.warning("Primary LLM failed (%s); falling back to stub.", exc)
            return self.fallback.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )


class MLVocaLLMClient(LLMClient):
    """Client that talks to the public MLVoca text generation API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        request_timeout: int = 120,
    ) -> None:
        self.base_url = base_url or os.getenv("MLVOCA_BASE_URL", DEFAULT_MLVOCA_BASE_URL)
        self.model = model or os.getenv("MLVOCA_MODEL", DEFAULT_MLVOCA_MODEL)
        self.request_timeout = request_timeout

    def _format_messages(self, messages: Iterable[LLMPrompt]) -> str:
        parts = []
        for prompt in messages:
            role = prompt.role.lower()
            if role == "system":
                prefix = "System"
            elif role == "assistant":
                prefix = "Assistant"
            else:
                prefix = "User"
            cleaned = prompt.content.strip()
            if cleaned:
                parts.append(f"{prefix}: {cleaned}")
        return "\n\n".join(parts)

    def generate(
        self,
        messages: Iterable[LLMPrompt],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,  # Unused but kept for signature compatibility.
        extra: Optional[Mapping[str, object]] = None,
    ) -> str:
        prompt_text = self._format_messages(messages)
        if not prompt_text:
            prompt_text = "Assistant:"

        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": False,
        }

        options: dict[str, object] = {"temperature": temperature}
        if extra:
            if "options" in extra and isinstance(extra["options"], Mapping):
                options.update(extra["options"])  # type: ignore[arg-type]
            if "model" in extra:
                payload["model"] = str(extra["model"])
            if "response_format" in extra and isinstance(extra["response_format"], Mapping):
                fmt = extra["response_format"].get("type")
                if fmt == "json_object":
                    payload["format"] = "json"
            for key in ("suffix", "format", "system", "template", "raw", "stream"):
                if key in extra:
                    payload[key] = extra[key]
        if options:
            payload["options"] = options

        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", ""))


def get_default_client() -> LLMClient:
    """Return the default LLM client, falling back to the stub client."""
    fallback = StubLLMClient()
    try:
        primary = MLVocaLLMClient()
    except Exception:
        return fallback
    return FailoverLLMClient(primary, fallback)
