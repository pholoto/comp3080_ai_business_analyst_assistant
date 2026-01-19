"""Lightweight LLM client abstraction."""

from .client import LLMClient, LLMPrompt, get_default_client

__all__ = ["LLMClient", "LLMPrompt", "get_default_client"]
