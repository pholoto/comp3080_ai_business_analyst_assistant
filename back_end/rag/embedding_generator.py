"""Utility for turning text chunks into dense embeddings."""
from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .config import DEFAULT_CONFIG, RagConfig
from .text_splitter import DocumentChunk


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol describing an object capable of embedding text inputs."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        ...

    def embed_query(self, query: str) -> np.ndarray:  # pragma: no cover - interface
        ...


class EmbeddingGenerator:
    """Wrapper around a concrete embedding provider with batching helpers."""

    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        batch_size: int = 32,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.batch_size = batch_size
        self._provider = provider
        self._lock = threading.Lock()

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> np.ndarray:
        """Generate embeddings for a sequence of chunks."""
        texts = [chunk.text for chunk in chunks]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._encode(texts)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Generate embeddings for raw text inputs."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single query string."""
        provider = self._get_provider()
        return provider.embed_query(query)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        provider = self._get_provider()
        embeddings = provider.embed_documents(list(texts))
        return np.asarray(embeddings, dtype=np.float32)

    def _get_provider(self) -> EmbeddingProvider:
        if self._provider is None or not isinstance(self._provider, EmbeddingProvider):
            with self._lock:
                if self._provider is None or not isinstance(self._provider, EmbeddingProvider):
                    self._provider = SentenceTransformerProvider(
                        model_name=self.config.embedding_model_name,
                        batch_size=self.batch_size,
                    )
        return self._provider

@dataclass
class SentenceTransformerProvider(EmbeddingProvider):
    """Default provider backed by sentence-transformers."""

    model_name: str
    batch_size: int = 32

    def __post_init__(self) -> None:
        self._model = self._load_model(self.model_name)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]

    @staticmethod
    def _load_model(model_name: str):
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "sentence-transformers must be installed to use EmbeddingGenerator"
            ) from exc
        sentence_transformer = getattr(module, "SentenceTransformer", None)
        if sentence_transformer is None:
            raise ImportError(
                "sentence-transformers installation missing SentenceTransformer"
            )
        return sentence_transformer(model_name)


__all__ = ["EmbeddingGenerator", "EmbeddingProvider", "SentenceTransformerProvider"]
