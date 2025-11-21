"""Retriever that orchestrates the embedding generator and vector store."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence

from .config import DEFAULT_CONFIG, RagConfig
from .embedding_generator import EmbeddingGenerator
from .vector_store import FaissVectorStore, InMemoryVectorStore, VectorHit


@dataclass
class RetrievalResult:
    """Container for retrieved chunk data."""

    text: str
    score: float
    metadata: dict[str, Any]


class ContextRetriever:
    """Find relevant chunks for a user query."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: FaissVectorStore | InMemoryVectorStore | None = None,
        config: RagConfig | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.embedding_generator = embedding_generator or EmbeddingGenerator(self.config)
        self.vector_store = vector_store or FaissVectorStore(self.config)

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int | None = None,
        tags: Optional[Sequence[str]] = None,
        fetch_multiplier: int = 3,
        score_threshold: float | None = None,
        recency_boost_days: int | None = None,
        recency_boost_weight: float | None = None,
    ) -> List[RetrievalResult]:
        """Retrieve the most relevant chunks for a user query."""
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id is required")
        if not query.strip():
            raise ValueError("query is required")
        top_k = top_k or self.config.top_k
        query_embedding = self.embedding_generator.embed_query(query)
        fetch_k = max(top_k * fetch_multiplier, top_k)
        vector_hits = self.vector_store.search(user_id, query_embedding, top_k=fetch_k)
        results: List[RetrievalResult] = []
        normalized_tags = {
            tag.strip().lower() for tag in (tags or []) if tag and tag.strip()
        }
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        recency_days = recency_boost_days if recency_boost_days is not None else self.config.recency_boost_days
        recency_weight = (
            recency_boost_weight if recency_boost_weight is not None else self.config.recency_boost_weight
        )
        seen_chunks: set[str] = set()
        for hit in vector_hits:
            if normalized_tags and not self._metadata_matches_tags(hit, normalized_tags):
                continue
            adjusted_score = float(hit.score)
            if recency_days and recency_days > 0:
                adjusted_score += self._recency_bonus(hit, recency_days, recency_weight)
            if threshold is not None and adjusted_score < threshold:
                continue
            chunk_id = str(hit.metadata.get("chunk_id", ""))
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            results.append(
                RetrievalResult(
                    text=str(hit.metadata.get("text", "")),
                    score=adjusted_score,
                    metadata=dict(hit.metadata),
                )
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _metadata_matches_tags(hit: VectorHit, tags: set[str]) -> bool:
        chunk_tags = hit.metadata.get("tags", [])
        chunk_tag_set = {
            str(tag).strip().lower() for tag in chunk_tags if str(tag).strip()
        }
        return bool(tags.intersection(chunk_tag_set))

    @staticmethod
    def _recency_bonus(hit: VectorHit, recency_days: int, weight: float) -> float:
        created_at = hit.metadata.get("created_at")
        if not created_at:
            return 0.0
        try:
            created_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        age_days = (datetime.now(timezone.utc) - created_dt.astimezone(timezone.utc)).days
        if age_days < 0:
            age_days = 0
        freshness = max(0.0, 1 - (age_days / recency_days))
        return round(freshness * max(weight, 0.0), 6)


__all__ = ["ContextRetriever", "RetrievalResult"]
