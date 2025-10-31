"""Base interfaces for chunking strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


class ChunkingStrategy(Protocol):
    """Protocol for chunking implementations."""

    name: str
    description: str

    def chunk(self, text: str) -> List[str]:
        ...


@dataclass
class ChunkingResult:
    """Container for chunked output."""

    strategy: str
    chunks: List[str]