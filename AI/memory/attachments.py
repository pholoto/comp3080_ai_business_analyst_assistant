"""Attachment utilities for session-scoped document management."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from docx import Document

try:  # PyPDF2 is optional but recommended for PDF ingestion
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    PdfReader = None  # type: ignore

@dataclass
class Attachment:
    """Representation of a user-supplied reference document."""

    attachment_id: str
    filename: str
    content_type: str
    size: int
    text: str
    chunks: List[str]
    added_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks_by_strategy: Dict[str, List[str]] = field(default_factory=dict)

    def preview(self, char_limit: int = 240) -> str:
        snippet = self.text.strip().replace("\r\n", "\n").replace("\r", "\n")
        if len(snippet) <= char_limit:
            return snippet
        return snippet[: char_limit - 1].rstrip() + "…"

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def as_dict(
        self,
        *,
        include_text: bool = False,
        include_chunks: bool = False,
        preview_chars: int = 240,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "attachment_id": self.attachment_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "word_count": self.word_count,
            "added_at": self.added_at.isoformat(),
            "preview": self.preview(preview_chars),
            "metadata": dict(self.metadata),
            "chunk_counts": {key: len(value) for key, value in self.chunks_by_strategy.items()},
        }
        if include_text:
            data["text"] = self.text
        if include_chunks:
            data["chunks"] = list(self.chunks)
            data["chunks_by_strategy"] = {
                key: list(chunks) for key, chunks in self.chunks_by_strategy.items()
            }
        return data

    def get_chunks(self, strategy: str | None = None) -> List[str]:
        if strategy and strategy in self.chunks_by_strategy:
            return self.chunks_by_strategy[strategy]
        return self.chunks


def extract_text_from_attachment(
    filename: str,
    content_type: str,
    blob: bytes,
) -> Tuple[str, Dict[str, Any]]:
    """Return plain text content and metadata extracted from an uploaded file."""
    suffix = Path(filename).suffix.lower()
    extractor = _select_extractor(suffix, content_type)
    text = extractor(blob)
    metadata = {
        "source_extension": suffix.lstrip("."),
        "content_type": content_type,
    }
    return text, metadata


def _select_extractor(suffix: str, content_type: str):
    if suffix in {".txt", ".md", ".markdown"} or content_type.startswith("text/"):
        return _extract_text_plain
    if suffix == ".docx":
        return _extract_text_docx
    if suffix == ".pdf":
        return _extract_text_pdf
    return _extract_text_fallback


def _extract_text_plain(blob: bytes) -> str:
    return blob.decode("utf-8", errors="replace")


def _extract_text_docx(blob: bytes) -> str:
    document = Document(BytesIO(blob))
    paragraphs = [para.text for para in document.paragraphs if para.text]
    return "\n".join(paragraphs)


def _extract_text_pdf(blob: bytes) -> str:
    if PdfReader is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "PyPDF2 is required to read PDF attachments. Install it via `pip install PyPDF2`."
        )
    reader = PdfReader(BytesIO(blob))
    fragments: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            fragments.append(text)
    return "\n".join(fragments)


def _extract_text_fallback(blob: bytes) -> str:
    try:
        return blob.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive fallback
        return ""