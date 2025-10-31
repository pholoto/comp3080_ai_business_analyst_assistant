"""Lightweight FAISS-inspired vector index."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .base import IndexingStrategy, IndexResult
from .embedding import cosine_similarity, embed


@dataclass
class FaissLikeIndex(IndexingStrategy):
    """Minimal in-memory cosine index that mimics FAISS APIs."""

    name: str = "faiss"
    description: str = "Cosine similarity search with bag-of-words embeddings."
    _vectors: List[tuple[dict, str, Optional[dict]]] = field(default_factory=list)

    def reset(self) -> None:
        self._vectors.clear()

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
            vector = embed(chunk)
            self._vectors.append((vector, chunk, meta))

    def search(self, query: str, *, top_k: int = 5) -> List[IndexResult]:
        if not query or not self._vectors:
            return []
        query_vec = embed(query)
        results: List[IndexResult] = []
        for vector, chunk, meta in self._vectors:
            score = cosine_similarity(query_vec, vector)
            if score > 0:
                results.append(IndexResult(chunk=chunk, score=float(score), metadata=meta))
        if not results:
            return []
        results.sort(key=lambda item: item.score, reverse=True)
        section_scores: Dict[str, float] = {}
        for item in results:
            meta = dict(item.metadata or {})
            item.metadata = meta
            section_rank = _ensure_section_rank(meta)
            current = section_scores.get(section_rank)
            if current is None or item.score > current:
                section_scores[section_rank] = item.score
        for item in results:
            meta = item.metadata or {}
            section_rank = _ensure_section_rank(meta)
            meta["section_score"] = section_scores.get(section_rank, item.score)
        return results[:top_k]


def _ensure_section_rank(meta: dict) -> str:
    if not meta:
        return "General"
    rank = meta.get("section_rank")
    if rank:
        return str(rank)
    path = meta.get("section_path")
    if path:
        rank = " > ".join(path)
    else:
        heading = meta.get("section_heading") or meta.get("section") or "General"
        rank = str(heading)
    meta["section_rank"] = rank
    return rank
