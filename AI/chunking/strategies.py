"""Concrete chunking strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import ChunkingStrategy
from .sections import match_heading_line


@dataclass
class AllInOneChunker(ChunkingStrategy):
    name: str = "all_in_one"
    description: str = "Single chunk containing the entire document."

    def chunk(self, text: str) -> List[str]:
        cleaned = text.strip()
        return [cleaned] if cleaned else []


@dataclass
class FixedSizeChunker(ChunkingStrategy):
    name: str = "fixed"
    description: str = "Fixed-length windows with configurable overlap."
    chunk_size: int = 1200
    overlap: int = 200

    def chunk(self, text: str) -> List[str]:
        cleaned = _normalise_text(text)
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [cleaned]
        chunks: List[str] = []
        start = 0
        while start < len(cleaned):
            end = start + self.chunk_size
            chunks.append(cleaned[start:end])
            if end >= len(cleaned):
                break
            start = max(end - self.overlap, start + 1)
        return chunks


@dataclass
class SemanticChunker(ChunkingStrategy):
    name: str = "semantic"
    description: str = "Heuristic semantic segmentation based on headings and paragraphs."
    min_chunk_size: int = 400

    def chunk(self, text: str) -> List[str]:
        cleaned_lines = [line.strip() for line in text.splitlines()]
        segments: List[str] = []
        buffer: List[str] = []

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            combined = "\n".join(buffer).strip()
            if combined:
                segments.append(combined)
            buffer = []

        for idx, line in enumerate(cleaned_lines):
            if not line:
                continue
            if match_heading_line(line) and buffer:
                flush()
                buffer.append(line)
            else:
                buffer.append(line)
            if len("\n".join(buffer)) >= self.min_chunk_size:
                flush()

        flush()

        if not segments:
            return [text.strip()] if text.strip() else []

        merged: List[str] = []
        for chunk in segments:
            if merged and len(merged[-1]) + len(chunk) < self.min_chunk_size:
                merged[-1] = merged[-1] + "\n" + chunk
            else:
                merged.append(chunk)
        return merged


def _normalise_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
