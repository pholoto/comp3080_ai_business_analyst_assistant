"""Base interfaces for indexing strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


@dataclass
class IndexResult:
    """Single search hit from an index."""

    chunk: str
    score: float
    metadata: Optional[dict] = None


class IndexingStrategy(Protocol):
    """Protocol for pluggable indexing implementations."""

    name: str
    description: str

    def reset(self) -> None:
        ...

    def add_documents(
        self,
        documents: Sequence[str],
        *,
        metadata: Optional[Sequence[Optional[dict]]] = None,
    ) -> None:
        ...

    def search(self, query: str, *, top_k: int = 5) -> List[IndexResult]:
        ...


@dataclass
class NullIndex:
    """No-op indexing strategy for baseline comparisons."""

    name: str = "none"
    description: str = "No indexing; linear scan performed on demand."
    _documents: List[Tuple[str, Optional[dict]]] = field(default_factory=list)

    def reset(self) -> None:
        self._documents = []

    def add_documents(
        self,
        documents: Sequence[str],
        *,
        metadata: Optional[Sequence[Optional[dict]]] = None,
    ) -> None:
        meta_seq: Sequence[Optional[dict]]
        if metadata is None:
            meta_seq = [None] * len(documents)
        else:
            meta_seq = metadata
        for idx, chunk in enumerate(documents):
            meta = meta_seq[idx] if idx < len(meta_seq) else None
            self._documents.append((chunk, meta))

    def search(self, query: str, *, top_k: int = 5) -> List[IndexResult]:
        if not query or not self._documents:
            return []
        matches: List[IndexResult] = []
        section_scores: Dict[str, float] = {}
        for chunk, meta in self._documents:
            if query.lower() in chunk.lower():
                meta_copy = dict(meta or {})
                matches.append(IndexResult(chunk=chunk, score=1.0, metadata=meta_copy))
                rank = _derive_section_rank(meta_copy)
                section_scores[rank] = max(section_scores.get(rank, 0.0), 1.0)
        for match in matches:
            meta = match.metadata or {}
            rank = _derive_section_rank(meta)
            meta["section_score"] = section_scores.get(rank, match.score)
        return matches[:top_k]


def _derive_section_rank(meta: dict | None) -> str:
    if not meta:
        return "General"
    rank = meta.get("section_rank")
    if rank:
        return str(rank)
    path = meta.get("section_path")
    if path:
        rank = " > ".join(path)
    else:
        rank = meta.get("section_heading") or meta.get("section") or "General"
    meta["section_rank"] = rank
    return str(rank)