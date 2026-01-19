"""Feature base classes and registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Protocol

from ..llm import LLMClient
from ..memory import Session


@dataclass
class FeatureContext:
    """Runtime context passed to features when invoked."""

    session: Session
    llm: LLMClient


@dataclass
class FeatureResult:
    """Standardised response from a feature invocation."""

    title: str
    summary: str
    data: Dict[str, Any]


class Feature(Protocol):
    """Protocol for each assistant feature."""

    name: str
    description: str

    def run(
        self,
        user_input: str,
        *,
        context: FeatureContext | None = None,
    ) -> FeatureResult:
        ...


class FeatureRegistry:
    """Simple registry that keeps feature factories by name."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[FeatureContext], Feature]] = {}

    def register(self, key: str, factory: Callable[[FeatureContext], Feature]) -> None:
        self._factories[key] = factory

    def create(self, key: str, context: FeatureContext) -> Feature:
        if key not in self._factories:
            raise KeyError(f"Unknown feature '{key}'")
        return self._factories[key](context)

    def keys(self) -> Iterable[str]:  # pragma: no cover - trivial container helper
        return self._factories.keys()
