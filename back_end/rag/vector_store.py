"""FAISS-backed vector store for chunk retrieval."""
from __future__ import annotations

import importlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import numpy as np

from .config import DEFAULT_CONFIG, RagConfig
from .text_splitter import DocumentChunk


@dataclass
class VectorHit:
    """Raw output from the vector store before ranking."""

    score: float
    metadata: Dict[str, Any]


class FaissVectorStore:
    """Lightweight FAISS wrapper that persists indices per user."""

    def __init__(self, config: RagConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        self._faiss = self._load_faiss_module()

    def add_chunks(
        self,
        user_id: str,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Add chunk embeddings to the FAISS index for a user."""
        if embeddings.size == 0 or len(chunks) == 0:
            return
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings row count must match chunk count")

        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        index_path = user_dir / "index.faiss"
        metadata_path = user_dir / "metadata.json"

        metadata_blob = self._load_metadata(metadata_path)
        metadata_chunks: List[Dict[str, Any]] = metadata_blob.get("chunks", [])
        existing_dim = metadata_blob.get("dimension")
        embedding_dim = int(embeddings.shape[1])

        index = self._load_index(index_path, embedding_dim)
        if existing_dim is not None and existing_dim != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: existing={existing_dim}, new={embedding_dim}"
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.size == 0:
            return
        # IndexFlatIP expects normalized vectors; embeddings are pre-normalized by the generator.
        index.add(embeddings)

        for chunk in chunks:
            metadata_entry = chunk.metadata.to_dict()
            metadata_entry["text"] = chunk.text
            metadata_chunks.append(metadata_entry)

        if index.ntotal != len(metadata_chunks):
            raise RuntimeError(
                "Metadata count mismatch after indexing operation. Aborting save to avoid corruption."
            )

        self._faiss.write_index(index, str(index_path))
        metadata_blob["dimension"] = embedding_dim
        metadata_blob["chunks"] = metadata_chunks
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_blob, handle, indent=2, ensure_ascii=True)

    def search(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
    ) -> List[VectorHit]:
        """Search the FAISS index for similar chunks."""
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a 1D array")
        user_dir = self._user_dir(user_id)
        index_path = user_dir / "index.faiss"
        metadata_path = user_dir / "metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            return []

        metadata_blob = self._load_metadata(metadata_path)
        metadata_chunks = metadata_blob.get("chunks", [])
        if not metadata_chunks:
            return []

        index = self._faiss.read_index(str(index_path))
        dim = metadata_blob.get("dimension")
        if dim is None:
            raise RuntimeError("Stored metadata missing embedding dimension")
        if index.d != int(dim):
            raise RuntimeError(
                f"Stored index dimension {index.d} does not match metadata dimension {dim}"
            )

        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query_vector.shape[1] != int(dim):
            raise ValueError(
                f"Query embedding dimension {query_vector.shape[1]} does not match index dimension {dim}"
            )

        k = min(top_k, index.ntotal)
        if k <= 0:
            return []
        scores, indices = index.search(query_vector, k)
        hits: List[VectorHit] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            if idx >= len(metadata_chunks):
                continue
            metadata_entry = metadata_chunks[idx]
            hits.append(VectorHit(score=float(score), metadata=metadata_entry))
        return hits

    def _user_dir(self, user_id: str) -> Path:
        return self.config.vector_dir / user_id

    @staticmethod
    def _load_metadata(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"dimension": None, "chunks": []}
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return {"dimension": None, "chunks": []}
        data = json.loads(content)
        data["chunks"] = list(data.get("chunks", []))
        return data

    def _load_index(self, path: Path, dimension: int):
        if path.exists():
            index = self._faiss.read_index(str(path))
            if index.d != dimension:
                raise ValueError(
                    f"Existing FAISS index dimension {index.d} does not match {dimension}"
                )
            return index
        return self._faiss.IndexFlatIP(dimension)

    @staticmethod
    def _load_faiss_module():
        try:
            return importlib.import_module("faiss")
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("faiss-cpu must be installed to use FaissVectorStore") from exc


class InMemoryVectorStore:
    """Simple numpy-based vector store used as a lightweight fallback."""

    def __init__(self) -> None:
        self._entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_chunks(
        self,
        user_id: str,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        for chunk, vector in zip(chunks, embeddings):
            metadata = dict(chunk.metadata.to_dict())
            metadata["text"] = chunk.text
            self._entries[user_id].append(
                {
                    "vector": np.asarray(vector, dtype=np.float32),
                    "metadata": metadata,
                }
            )

    def search(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
    ) -> List[VectorHit]:
        records = self._entries.get(user_id, [])
        if not records:
            return []
        query_vec = np.asarray(query_embedding, dtype=np.float32).ravel()
        hits: List[VectorHit] = []
        for record in records:
            vector = cast(np.ndarray, record["vector"])
            metadata = cast(Dict[str, Any], record["metadata"])
            score = float(np.dot(vector, query_vec))
            hits.append(VectorHit(score=score, metadata=dict(metadata)))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


__all__ = ["FaissVectorStore", "VectorHit", "InMemoryVectorStore"]
