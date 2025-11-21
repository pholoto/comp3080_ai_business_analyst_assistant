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
    char_start: int
    char_end: int
    source_path: str
    checksum: str
    created_at: str

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
            return _FallbackRecursiveSplitter
        splitter_cls = getattr(module, "RecursiveCharacterTextSplitter", None)
        if splitter_cls is None:
            return _FallbackRecursiveSplitter
        return splitter_cls

    def split_record(self, record: DocumentRecord) -> List[DocumentChunk]:
        """Split a single document record into chunks."""
        raw_chunks = self._splitter.split_text(record.text)
        positions = self._compute_positions(record.text, raw_chunks)
        chunks: List[DocumentChunk] = []
        for index, (chunk_text, (start, end)) in enumerate(zip(raw_chunks, positions)):
            chunk_id = f"{record.metadata.document_id}_{index:04d}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=record.metadata.document_id,
                user_id=record.metadata.user_id,
                source_name=record.metadata.original_name,
                chunk_index=index,
                tags=record.metadata.tags,
                char_start=start,
                char_end=end,
                source_path=record.metadata.stored_path,
                checksum=record.metadata.checksum,
                created_at=record.metadata.created_at,
            )
            chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
        return chunks

    @staticmethod
    def _compute_positions(text: str, chunks: Sequence[str]) -> List[tuple[int, int]]:
        positions: List[tuple[int, int]] = []
        cursor = 0
        for chunk in chunks:
            if not chunk:
                positions.append((cursor, cursor))
                continue
            idx = text.find(chunk, cursor)
            if idx == -1:
                idx = cursor
            start = idx
            end = start + len(chunk)
            cursor = end
            positions.append((start, end))
        return positions

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


class _FallbackRecursiveSplitter:
    """Minimal splitter used when LangChain is unavailable."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

    @classmethod
    def from_tiktoken_encoder(
        cls,
        *,
        chunk_size: int,
        chunk_overlap: int,
        disallowed_special: Sequence[str] | tuple[str, ...],
    ):
        # disallowed_special parameter kept for interface compatibility
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(length, start + self.chunk_size)
            chunks.append(text[start:end])
            if end == length:
                break
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            if start >= length:
                break
        return chunks
