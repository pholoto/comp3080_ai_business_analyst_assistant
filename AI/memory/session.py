"""Session objects and manager for the assistant."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..chunking import available_chunkers, get_chunker
from ..chunking.sections import detect_section_heading
from ..evaluation import (mean_reciprocal_rank, ndcg_at_k, precision_at_k,
                          recall_at_k, summarise_latency)
from ..indexing import available_indexers, get_indexer
from ..indexing.base import IndexingStrategy, IndexResult
from .attachments import Attachment, extract_text_from_attachment
from .memory import ConversationMemory


@dataclass
class Session:
    """Per-user conversational session."""

    session_id: str
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    state: Dict[str, Any] = field(default_factory=dict)
    attachments: Dict[str, Attachment] = field(default_factory=dict)
    chunking_strategy: str = "fixed"
    indexing_strategy: str = "none"
    _index: IndexingStrategy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = get_indexer(self.indexing_strategy)
        self._index.reset()
        self.state["index_size"] = 0
        self._refresh_attachment_state()

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any | None = None) -> Any:
        return self.state.get(key, default)

    def add_attachment(
        self,
        *,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> Attachment:
        attachment_id = secrets.token_hex(8)
        text, metadata = extract_text_from_attachment(filename, content_type, data)
        chunk_map: Dict[str, List[str]] = {}
        for strategy_name in available_chunkers():
            chunker = get_chunker(strategy_name)
            chunk_map[strategy_name] = chunker.chunk(text)
        selected_chunks = chunk_map.get(self.chunking_strategy)
        if selected_chunks is None:
            selected_chunks = chunk_map.get("fixed", [])
        attachment = Attachment(
            attachment_id=attachment_id,
            filename=filename,
            content_type=content_type or "application/octet-stream",
            size=len(data),
            text=text,
            chunks=selected_chunks,
            added_at=datetime.now(timezone.utc),
            metadata=metadata,
            chunks_by_strategy=chunk_map,
        )
        self.attachments[attachment_id] = attachment
        self._refresh_attachment_state()
        self._rebuild_index()
        return attachment

    def remove_attachment(self, attachment_id: str) -> bool:
        removed = self.attachments.pop(attachment_id, None)
        if removed is None:
            return False
        self._refresh_attachment_state()
        self._rebuild_index()
        return True

    def list_attachments(self) -> List[Attachment]:
        return list(self.attachments.values())

    def get_attachment(self, attachment_id: str) -> Attachment:
        if attachment_id not in self.attachments:
            raise KeyError(f"Attachment {attachment_id} not found")
        return self.attachments[attachment_id]

    def iter_attachment_chunks(self, max_chunks: int | None = None) -> Iterable[str]:
        produced = 0
        for attachment in self.attachments.values():
            for chunk in attachment.get_chunks(self.chunking_strategy):
                yield chunk
                produced += 1
                if max_chunks is not None and produced >= max_chunks:
                    return

    def attachment_digest(self, *, char_limit: int = 2000) -> str:
        if not self.attachments:
            return ""
        pieces: List[str] = []
        remaining = char_limit
        for attachment in self.attachments.values():
            if remaining <= 0:
                break
            preview = attachment.preview(min(remaining, 320))
            chunk_count = len(attachment.get_chunks(self.chunking_strategy))
            entry = (
                f"{attachment.filename} (chunks: {chunk_count}, added {attachment.added_at.date()}):\n"
                f"{preview}"
            )
            pieces.append(entry)
            remaining -= len(entry)
        return "\n\n".join(pieces)[:char_limit]

    def _refresh_attachment_state(self) -> None:
        summary = [attachment.as_dict(preview_chars=160) for attachment in self.attachments.values()]
        self.state["attachments"] = summary
        self.state["chunking_strategy"] = self.chunking_strategy
        self.state["indexing_strategy"] = self.indexing_strategy

    def _rebuild_index(self) -> None:
        self._index = get_indexer(self.indexing_strategy)
        self._index.reset()
        documents: List[str] = []
        metadata: List[dict] = []
        for attachment in self.attachments.values():
            chunks = attachment.get_chunks(self.chunking_strategy)
            section_context: Optional[dict] = None
            for idx, chunk in enumerate(chunks):
                documents.append(chunk)
                section_context = _derive_section_context(chunk, section_context)
                metadata.append(
                    _build_chunk_metadata(
                        attachment=attachment,
                        chunk_index=idx,
                        context=section_context,
                    )
                )
        if documents:
            self._index.add_documents(documents, metadata=metadata)
        self.state["index_size"] = len(documents)

    def set_chunking_strategy(self, strategy: str) -> None:
        if strategy == self.chunking_strategy:
            return
        if strategy not in available_chunkers():
            raise ValueError(f"Unsupported chunking strategy '{strategy}'")
        self.chunking_strategy = strategy
        for attachment in self.attachments.values():
            if strategy not in attachment.chunks_by_strategy:
                chunker = get_chunker(strategy)
                attachment.chunks_by_strategy[strategy] = chunker.chunk(attachment.text)
            attachment.chunks = attachment.get_chunks(strategy)
        self._refresh_attachment_state()
        self._rebuild_index()

    def set_indexing_strategy(self, strategy: str) -> None:
        if strategy == self.indexing_strategy:
            return
        if strategy not in available_indexers():
            raise ValueError(f"Unsupported indexing strategy '{strategy}'")
        self.indexing_strategy = strategy
        self._rebuild_index()

    def search(self, query: str, *, top_k: int = 5) -> List[IndexResult]:
        return self._index.search(query, top_k=top_k)

    def section_ranking(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        raw_results = self.search(query, top_k=max(top_k * 3, top_k))
        if not raw_results:
            return []
        buckets: Dict[str, Dict[str, Any]] = {}
        for result in raw_results:
            meta = result.metadata or {}
            rank = meta.get("section_rank") or meta.get("section") or "General"
            bucket = buckets.get(rank)
            section_heading = meta.get("section_heading") or meta.get("section") or "General"
            if bucket is None:
                bucket = {
                    "section_rank": rank,
                    "section_heading": section_heading,
                    "section_title": meta.get("section_title") or section_heading,
                    "section_identifier": meta.get("section_identifier"),
                    "section_path": list(meta.get("section_path") or []),
                    "document_id": meta.get("document_id"),
                    "document_label": meta.get("document_label") or meta.get("filename"),
                    "best_chunk": result.chunk,
                    "best_chunk_score": result.score,
                    "score": meta.get("section_score", result.score),
                    "chunk_count": meta.get("chunk_count"),
                    "matches": 0,
                }
                buckets[rank] = bucket
            bucket["matches"] += 1
            bucket["score"] = max(bucket["score"], meta.get("section_score", result.score))
            if result.score > bucket["best_chunk_score"]:
                bucket["best_chunk_score"] = result.score
                bucket["best_chunk"] = result.chunk
        ordered = sorted(buckets.values(), key=lambda item: item["score"], reverse=True)
        return ordered[:top_k]

    def evaluate_retrieval(
        self,
        queries: Sequence[dict],
        *,
        default_top_k: int = 5,
        latency_samples_ms: Optional[Sequence[float]] = None,
        index_build_ms: Optional[float] = None,
        throughput_qps: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not queries:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "ndcg_at_k": 0.0,
                "per_query": [],
                "efficiency": None,
            }
        aggregate_precision: List[float] = []
        aggregate_recall: List[float] = []
        aggregate_mrr: List[float] = []
        aggregate_ndcg: List[float] = []
        per_query_results: List[Dict[str, Any]] = []

        for entry in queries:
            query_text = entry.get("query", "").strip()
            if not query_text:
                continue
            relevant_chunks: List[str] = entry.get("relevant_chunks", [])
            top_k = int(entry.get("top_k") or default_top_k)
            results = self.search(query_text, top_k=top_k)
            retrieved = [result.chunk for result in results]
            relevance_flags = _compute_relevance_flags(retrieved, relevant_chunks)
            precision = precision_at_k(relevance_flags, top_k)
            recall = recall_at_k(relevance_flags, len(relevant_chunks), top_k)
            mrr = mean_reciprocal_rank(relevance_flags)
            ndcg = ndcg_at_k(relevance_flags, top_k)
            aggregate_precision.append(precision)
            aggregate_recall.append(recall)
            aggregate_mrr.append(mrr)
            aggregate_ndcg.append(ndcg)
            per_query_results.append(
                {
                    "query": query_text,
                    "top_k": top_k,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "mrr": mrr,
                    "ndcg_at_k": ndcg,
                }
            )

        efficiency: Optional[Dict[str, float]] = None
        if latency_samples_ms or index_build_ms or throughput_qps:
            median, p95 = summarise_latency(latency_samples_ms or [])
            efficiency = {
                "median_latency_ms": median,
                "p95_latency_ms": p95,
            }
            if index_build_ms is not None:
                efficiency["index_build_ms"] = index_build_ms
            if throughput_qps is not None:
                efficiency["throughput_qps"] = throughput_qps

        count = len(per_query_results) or 1
        return {
            "precision_at_k": sum(aggregate_precision) / count,
            "recall_at_k": sum(aggregate_recall) / count,
            "mrr": sum(aggregate_mrr) / count,
            "ndcg_at_k": sum(aggregate_ndcg) / count,
            "per_query": per_query_results,
            "efficiency": efficiency,
        }


class SessionManager:
    """Simple in-memory session manager."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> Session:
        session_id = secrets.token_hex(16)
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear(self) -> None:
        self._sessions.clear()


def _derive_section_context(chunk: str, previous: Optional[dict]) -> dict:
    heading = detect_section_heading(chunk or "")
    if heading:
        heading_text = heading.get("heading") or heading.get("title") or "General"
        title = heading.get("title") or heading_text
        identifier = heading.get("identifier")
        path = list(heading.get("path") or [])
        if not path:
            path = [title]
        rank = " > ".join(path) if path else heading_text
        return {
            "heading": heading_text,
            "title": title,
            "identifier": identifier,
            "path": path,
            "rank": rank,
            "level": len(path) if path else 1,
        }
    if previous:
        clone = dict(previous)
        clone["path"] = list(previous.get("path", []))
        return clone
    return _default_section_context()


def _default_section_context() -> dict:
    return {
        "heading": "General",
        "title": "General",
        "identifier": None,
        "path": ["General"],
        "rank": "General",
        "level": 1,
    }


def _build_chunk_metadata(*, attachment: Attachment, chunk_index: int, context: dict) -> dict:
    identifier = context.get("identifier")
    title = context.get("title") or context.get("heading") or "General"
    label_parts = []
    if identifier:
        label_parts.append(str(identifier))
    if title:
        label_parts.append(str(title))
    label = " ".join(label_parts).strip() or context.get("heading", "General")
    metadata = {
        "attachment_id": attachment.attachment_id,
        "chunk_index": chunk_index,
        "chunk_id": f"{attachment.attachment_id}:{chunk_index}",
        "filename": attachment.filename,
        "document_id": attachment.attachment_id,
        "document_label": attachment.filename,
        "section": context.get("heading", "General"),
        "section_title": title,
        "section_heading": context.get("heading", "General"),
        "section_label": label,
        "section_identifier": identifier,
        "section_path": list(context.get("path", [])),
        "section_rank": context.get("rank", label),
        "section_level": context.get("level", 1),
    }
    # Remove identifier field if not present to keep metadata tidy.
    if identifier is None:
        metadata.pop("section_identifier")
    return metadata


def _compute_relevance_flags(retrieved: Sequence[str], relevant: Sequence[str]) -> List[int]:
    if not retrieved:
        return []
    if not relevant:
        return [0 for _ in retrieved]
    flags: List[int] = []
    for chunk in retrieved:
        chunk_lower = chunk.lower()
        match = any(rel.lower() in chunk_lower for rel in relevant if rel)
        flags.append(1 if match else 0)
    return flags
