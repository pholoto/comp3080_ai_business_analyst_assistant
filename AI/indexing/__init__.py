"""Indexing strategy registry and helpers."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List

from .base import IndexingStrategy, NullIndex
from .faiss_like import FaissLikeIndex
from .llama_index_stub import LlamaIndexStub

_INDEXER_FACTORIES: Dict[str, Callable[[], IndexingStrategy]] = {
    "none": lambda: NullIndex(),
    "faiss": lambda: FaissLikeIndex(),
    "llama_index": lambda: LlamaIndexStub(),
}


def get_indexer(key: str) -> IndexingStrategy:
    try:
        factory = _INDEXER_FACTORIES[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown indexing strategy '{key}'") from exc
    return factory()


def available_indexers() -> Iterable[str]:
    return _INDEXER_FACTORIES.keys()


def describe_indexers() -> List[dict]:
    descriptions: List[dict] = []
    for key, factory in _INDEXER_FACTORIES.items():
        instance = factory()
        descriptions.append(
            {
                "key": key,
                "description": instance.description,
                "name": getattr(instance, "name", key),
            }
        )
    return descriptions
