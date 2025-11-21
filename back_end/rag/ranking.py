"""Ranking utilities for presenting retrieval results."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence

from .retrieval import RetrievalResult


@dataclass
class RankingEntry:
    """Structured representation of a ranked retrieval result."""

    rank: int
    chunk_id: str
    document_id: str
    source_name: str
    score: float
    chunk_index: int
    preview: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SimilarityRanker:
    """Prepare ranked metadata for downstream consumption."""

    def __init__(self, preview_chars: int = 160) -> None:
        self.preview_chars = preview_chars

    def rank(self, results: Sequence[RetrievalResult], top_n: int = 5) -> List[RankingEntry]:
        entries: List[RankingEntry] = []
        for idx, result in enumerate(results[:top_n], start=1):
            chunk_id = str(result.metadata.get("chunk_id", ""))
            document_id = str(result.metadata.get("document_id", ""))
            source_name = result.metadata.get("source_name") or result.metadata.get(
                "original_name", "unknown"
            )
            preview = result.text.strip().replace("\n", " ")
            if len(preview) > self.preview_chars:
                preview = preview[: self.preview_chars].rstrip() + "..."
            entries.append(
                RankingEntry(
                    rank=idx,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_name=str(source_name),
                    score=float(result.score),
                    chunk_index=int(result.metadata.get("chunk_index", idx - 1)),
                    preview=preview,
                )
            )
        return entries


__all__ = ["SimilarityRanker", "RankingEntry"]
