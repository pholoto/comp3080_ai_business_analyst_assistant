"""Chunking strategy registry for document ingestion."""
from __future__ import annotations

from typing import Dict, Iterable, List

from .base import ChunkingStrategy
from .strategies import AllInOneChunker, FixedSizeChunker, SemanticChunker

_CHUNKERS: Dict[str, ChunkingStrategy] = {
    "all_in_one": AllInOneChunker(),
    "fixed": FixedSizeChunker(),
    "semantic": SemanticChunker(),
}


def get_chunker(key: str) -> ChunkingStrategy:
    try:
        return _CHUNKERS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown chunking strategy '{key}'") from exc


def available_chunkers() -> Iterable[str]:
    return _CHUNKERS.keys()


def describe_chunkers() -> List[dict]:
    return [
        {
            "key": name,
            "description": chunker.description,
            "name": getattr(chunker, "name", name),
        }
        for name, chunker in _CHUNKERS.items()
    ]