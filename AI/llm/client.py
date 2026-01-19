"""LLM client abstraction with a generic HTTP chat backend."""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import requests

# DEFAULT_CHAT_ENDPOINT = "https://apifreellm.com/api/chat"
# DEFAULT_CHAT_ENDPOINT = "https://apifreellm.com/api/v1/chat"
MODEL = "gpt-3.5-turbo"
DEFAULT_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
# Put your API key here

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


class HTTPChatLLMClient(LLMClient):
    """Generic HTTP chat client compatible with simple JSON APIs."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        request_timeout: int = 120,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.endpoint = endpoint or os.getenv("LLM_CHAT_URL", "https://apifreellm.com/api/chat")
        self.request_timeout = request_timeout
        base_headers = {"Content-Type": "application/json"}
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            base_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            for key, value in headers.items():
                base_headers[str(key)] = str(value)
        self.headers = base_headers
        # Use a requests.Session for connection reuse and better performance.
        self._session = requests.Session()
        self._session.headers.update(self.headers)

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

    def _apply_format_hint(
        self,
        prompt_text: str,
        extra: Optional[Mapping[str, object]],
    ) -> str:
        if not extra:
            return prompt_text
        format_spec = extra.get("response_format")
        if isinstance(format_spec, Mapping) and format_spec.get("type") == "json_object":
            hint = (
                "Please respond with a single valid JSON object only, without additional prose,"
                " ensuring it is parseable."
            )
            return f"{prompt_text}\n\n{hint}"
        return prompt_text

    def _build_headers(self, extra: Optional[Mapping[str, object]]) -> dict[str, str]:
        headers = dict(self.headers)
        if extra and "headers" in extra and isinstance(extra["headers"], Mapping):
            for key, value in extra["headers"].items():
                headers[str(key)] = str(value)
        return headers

    def _build_payload(
        self,
        message: str,
        extra: Optional[Mapping[str, object]],
    ) -> dict[str, object]:
        payload: dict[str, object] = {"message": message}
        # Temperature is only forwarded when explicitly overridden to avoid
        # relying on provider-specific parameters.
        if extra and "temperature" in extra:
            payload["temperature"] = extra["temperature"]
        if extra and "payload" in extra and isinstance(extra["payload"], Mapping):
            payload.update(extra["payload"])
        return payload

    def generate(
        self,
        messages: Iterable[LLMPrompt],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        extra: Optional[Mapping[str, object]] = None,
    ) -> str:
        prompt_text = self._format_messages(messages)
        if not prompt_text:
            prompt_text = "User:"
        prompt_text = self._apply_format_hint(prompt_text, extra)
        payload = self._build_payload(prompt_text, extra)
        headers = self._build_headers(extra)

        # Retry/backoff loop for transient provider errors or rate limits.
        max_attempts = 3
        backoff_base = 2.0
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                if resp.status_code == 429:
                    time.sleep(backoff_base ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "response" in data:
                    return str(data["response"])
                return str(data)
            except Exception as exc:
                last_exc = exc
                time.sleep(backoff_base ** attempt)
                continue

        raise RuntimeError(f"LLM provider error after retries: {last_exc}")


class ChatGPTClient(LLMClient):
    """LLM client for OpenAI's ChatGPT API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-3.5-turbo",
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        request_timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.endpoint = endpoint
        self.request_timeout = request_timeout
        self._session = requests.Session()

    def generate(
        self,
        messages: Iterable[LLMPrompt],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        extra: Optional[Mapping[str, object]] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY environment variable.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        openai_messages = []
        for m in messages:
            openai_messages.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if extra and "response_format" in extra:
            payload["response_format"] = extra["response_format"]

        max_attempts = 3
        backoff_base = 2.0
        last_exc: Exception | None = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                if resp.status_code == 429:
                    time.sleep(backoff_base ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                time.sleep(backoff_base ** attempt)
                continue

        raise RuntimeError(f"ChatGPT API error after retries: {last_exc}")


def get_default_client() -> LLMClient:
    """Return the default LLM client, falling back to the stub client."""
    fallback = StubLLMClient()
    
    # Check for ChatGPT/OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY") or API_KEY
    if openai_key:
        primary = ChatGPTClient(api_key=openai_key)
        return FailoverLLMClient(primary, fallback)
        
    # Fallback to generic HTTP client if configured
    if os.getenv("LLM_CHAT_URL"):
        primary = HTTPChatLLMClient()
        return FailoverLLMClient(primary, fallback)
        
    return fallback

