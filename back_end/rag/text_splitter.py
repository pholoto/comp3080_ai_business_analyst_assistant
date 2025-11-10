"""Chunking utilities built on top of LangChain's text splitter."""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence

from .config import DEFAULT_CONFIG, RagConfig
from .document_store import DocumentMetadata, DocumentRecord


@dataclass
class ChunkMetadata:
    """Metadata that describes a specific text chunk."""

    chunk_id: str
    document_id: str
    user_id: str
    source_name: str
    chunk_index: int
    tags: List[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DocumentChunk:
    """A chunk of text with associated metadata."""

    text: str
    metadata: ChunkMetadata


class TextSplitter:
    """Wrapper around LangChain's recursive character splitter."""

    def __init__(self, config: RagConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        splitter_cls = self._load_splitter_class()
        self._splitter = splitter_cls.from_tiktoken_encoder(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            disallowed_special=(),
        )

    @staticmethod
    def _load_splitter_class():
        try:
            module = importlib.import_module("langchain.text_splitter")
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "langchain must be installed to use the TextSplitter component"
            ) from exc
        splitter_cls = getattr(module, "RecursiveCharacterTextSplitter", None)
        if splitter_cls is None:
            raise ImportError(
                "LangChain installation missing RecursiveCharacterTextSplitter"
            )
        return splitter_cls

    def split_record(self, record: DocumentRecord) -> List[DocumentChunk]:
        """Split a single document record into chunks."""
        raw_chunks = self._splitter.split_text(record.text)
        chunks: List[DocumentChunk] = []
        for index, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{record.metadata.document_id}_{index:04d}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=record.metadata.document_id,
                user_id=record.metadata.user_id,
                source_name=record.metadata.original_name,
                chunk_index=index,
                tags=record.metadata.tags,
            )
            chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
        return chunks

    def split_records(self, records: Sequence[DocumentRecord]) -> List[DocumentChunk]:
        """Split multiple records into chunks."""
        all_chunks: List[DocumentChunk] = []
        for record in records:
            all_chunks.extend(self.split_record(record))
        return all_chunks

    def split_texts(
        self,
        *,
        text: str,
        metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        """Split an arbitrary text payload using existing metadata."""
        record = DocumentRecord(metadata=metadata, text=text)
        return self.split_record(record)


__all__ = ["TextSplitter", "DocumentChunk", "ChunkMetadata"]
