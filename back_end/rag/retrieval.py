"""Retriever that orchestrates the embedding generator and vector store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from .config import DEFAULT_CONFIG, RagConfig
from .embedding_generator import EmbeddingGenerator
from .vector_store import FaissVectorStore, VectorHit


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
        vector_store: FaissVectorStore | None = None,
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
        for hit in vector_hits:
            if normalized_tags and not self._metadata_matches_tags(hit, normalized_tags):
                continue
            results.append(
                RetrievalResult(
                    text=str(hit.metadata.get("text", "")),
                    score=hit.score,
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


__all__ = ["ContextRetriever", "RetrievalResult"]
