"""Utility for turning text chunks into dense embeddings."""
from __future__ import annotations

import importlib
import threading
from typing import Sequence

import numpy as np

from .config import DEFAULT_CONFIG, RagConfig
from .text_splitter import DocumentChunk


class EmbeddingGenerator:
    """Wrapper around a sentence-transformer embedding model."""

    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        batch_size: int = 32,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.batch_size = batch_size
        self._model = None
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
        embedding = self._encode([query])
        return embedding[0]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model(self.config.embedding_model_name)
        return self._model

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


__all__ = ["EmbeddingGenerator"]
