"""Filesystem-backed document store with per-user segregation."""
from __future__ import annotations

import json
import mimetypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import (IO, Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    cast)
from uuid import uuid4

from docx import Document as DocxDocument
from PyPDF2 import PdfReader

from .config import DEFAULT_CONFIG, RagConfig


@dataclass
class DocumentMetadata:
    """Metadata that describes a stored source document."""

    document_id: str
    user_id: str
    original_name: str
    stored_name: str
    stored_path: str
    mime_type: str
    checksum: str
    tags: List[str]
    created_at: str

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DocumentMetadata":
        raw_tags = data.get("tags", [])
        if isinstance(raw_tags, (list, tuple, set)):
            tags_iterable = list(raw_tags)
        elif raw_tags:
            tags_iterable = [raw_tags]
        else:
            tags_iterable = []
        tags = [str(tag) for tag in tags_iterable]
        return cls(
            document_id=str(data["document_id"]),
            user_id=str(data["user_id"]),
            original_name=str(data["original_name"]),
            stored_name=str(data["stored_name"]),
            stored_path=str(data["stored_path"]),
            mime_type=str(data.get("mime_type", "application/octet-stream")),
            checksum=str(data["checksum"]),
            tags=tags,
            created_at=str(data["created_at"]),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DocumentRecord:
    """A document alongside its metadata."""

    metadata: DocumentMetadata
    text: str


class UnsupportedDocumentError(ValueError):
    """Raised when a document type is not supported for ingestion."""


class DuplicateDocumentError(ValueError):
    """Raised when attempting to ingest a document that already exists."""


class DocumentStore:
    """Simple filesystem-backed store that tracks documents per user."""

    def __init__(self, config: RagConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        self._index = self._load_index()

    def ingest_file(
        self,
        user_id: str,
        filename: str,
    data: bytes | IO[bytes] | Path,
        *,
        tags: Optional[Sequence[str]] = None,
    ) -> DocumentRecord:
        """Persist a file for a user and return the parsed record."""
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id is required")
        dest_dir = self.config.data_dir / user_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        original_name = Path(filename).name
        tags_list = sorted({tag.strip().lower() for tag in tags or [] if tag.strip()})
        document_id = uuid4().hex
        suffix = Path(original_name).suffix
        stored_name = f"{document_id}{suffix}" if suffix else document_id
        dest_path = dest_dir / stored_name

        if isinstance(data, Path):
            with data.open("rb") as source, dest_path.open("wb") as target:
                target.write(source.read())
            file_bytes = dest_path.read_bytes()
        elif isinstance(data, bytes):
            dest_path.write_bytes(data)
            file_bytes = data
        elif isinstance(data, (bytearray, memoryview)):
            payload = bytes(data)
            dest_path.write_bytes(payload)
            file_bytes = payload
        else:
            # Binary IO object
            stream = cast(IO[bytes], data)
            payload = stream.read()
            dest_path.write_bytes(payload)
            file_bytes = payload

        checksum = sha256(file_bytes).hexdigest()
        existing = self._find_by_checksum(user_id, checksum)
        if existing:
            raise DuplicateDocumentError(
                f"Document already ingested for user {user_id} (document_id={existing.document_id})."
            )

        mime_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        metadata = DocumentMetadata(
            document_id=document_id,
            user_id=user_id,
            original_name=original_name,
            stored_name=stored_name,
            stored_path=str(dest_path),
            mime_type=mime_type,
            checksum=checksum,
            tags=tags_list,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        text = self._extract_text(dest_path, mime_type)
        record = DocumentRecord(metadata=metadata, text=text)
        self._add_to_index(record.metadata)
        return record

    def list_documents(self, user_id: str) -> List[DocumentMetadata]:
        """Return metadata for all documents belonging to a user."""
        docs = self._index.get(user_id, [])
        return [DocumentMetadata.from_dict(doc) for doc in docs]

    def load_document(self, user_id: str, document_id: str) -> Optional[DocumentRecord]:
        """Load a specific document record if it exists."""
        metadata = self._find_by_id(user_id, document_id)
        if not metadata:
            return None
        path = Path(metadata.stored_path)
        if not path.exists():
            return None
        text = self._extract_text(path, metadata.mime_type)
        return DocumentRecord(metadata=metadata, text=text)

    def iter_user_documents(self, user_id: str) -> Iterable[DocumentRecord]:
        """Yield all documents for a user with their parsed text."""
        for metadata_dict in self._index.get(user_id, []):
            metadata = DocumentMetadata.from_dict(metadata_dict)
            path = Path(metadata.stored_path)
            if not path.exists():
                continue
            yield DocumentRecord(metadata=metadata, text=self._extract_text(path, metadata.mime_type))

    def get_document_by_checksum(self, user_id: str, checksum: str) -> Optional[DocumentMetadata]:
        """Return previously ingested document metadata that matches a checksum."""
        return self._find_by_checksum(user_id, checksum)

    def _extract_text(self, path: Path, mime_type: str) -> str:
        """Extract raw text from a stored document."""
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
        if suffix == ".docx":
            doc = DocxDocument(str(path))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        if mime_type.startswith("text/"):
            return path.read_text(encoding="utf-8", errors="ignore")
        raise UnsupportedDocumentError(
            f"Unsupported document type: {suffix or mime_type}. Provide txt, md, pdf, or docx files."
        )

    def _find_by_id(self, user_id: str, document_id: str) -> Optional[DocumentMetadata]:
        for data in self._index.get(user_id, []):
            if data.get("document_id") == document_id:
                return DocumentMetadata.from_dict(data)
        return None

    def _find_by_checksum(self, user_id: str, checksum: str) -> Optional[DocumentMetadata]:
        for data in self._index.get(user_id, []):
            if data.get("checksum") == checksum:
                return DocumentMetadata.from_dict(data)
        return None

    def _add_to_index(self, metadata: DocumentMetadata) -> None:
        self._index.setdefault(metadata.user_id, []).append(metadata.to_dict())
        # Maintain deterministic order by created_at
        self._index[metadata.user_id].sort(
            key=lambda item: str(item.get("created_at", ""))
        )
        self._flush_index()

    def _load_index(self) -> Dict[str, List[Dict[str, object]]]:
        if not self.config.index_path.exists():
            return {}
        content = self.config.index_path.read_text(encoding="utf-8")
        if not content.strip():
            return {}
        data = json.loads(content)
        return {str(user): list(entries) for user, entries in data.items()}

    def _flush_index(self) -> None:
        with self.config.index_path.open("w", encoding="utf-8") as handle:
            json.dump(self._index, handle, indent=2, ensure_ascii=True)


__all__ = [
    "DocumentStore",
    "DocumentMetadata",
    "DocumentRecord",
    "UnsupportedDocumentError",
    "DuplicateDocumentError",
]
