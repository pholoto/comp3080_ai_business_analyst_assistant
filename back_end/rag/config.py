"""Configuration helpers for the backend RAG subsystem."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagConfig:
    """Container for RAG configuration values."""

    data_dir: Path
    vector_dir: Path
    index_path: Path
    chunk_size: int = 512
    chunk_overlap: int = 128
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 8

    def ensure_directories(self) -> None:
        """Ensure all filesystem locations exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)


def get_default_config() -> RagConfig:
    """Return the default configuration rooted in the backend package."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    vector_dir = base_dir / "vector_store"
    index_path = base_dir / "doc_index.json"
    config = RagConfig(data_dir=data_dir, vector_dir=vector_dir, index_path=index_path)
    config.ensure_directories()
    return config


DEFAULT_CONFIG = get_default_config()

__all__ = ["RagConfig", "DEFAULT_CONFIG", "get_default_config"]
