"""Hierarchical index inspired by LlamaIndex."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .base import IndexingStrategy, IndexResult
from .embedding import cosine_similarity, embed


@dataclass
class _SectionNode:
    heading: str
    chunks: List[str]
    metadata: Optional[dict]
    path: List[str]

    def all_chunks(self) -> List[str]:
        return list(self.chunks)


@dataclass
class _DocumentNode:
    doc_id: str
    sections: List[_SectionNode] = field(default_factory=list)

    def all_chunks(self) -> List[str]:
        collected: List[str] = []
        for section in self.sections:
            collected.extend(section.all_chunks())
        return collected


@dataclass
class LlamaIndexStub(IndexingStrategy):
    """Simplified multi-level index for semantic navigation."""

    name: str = "llama_index"
    description: str = "Hierarchy: document → section → chunk with cosine scoring."
    _documents: Dict[str, _DocumentNode] = field(default_factory=dict)

    def reset(self) -> None:
        self._documents.clear()

    def add_documents(
        self,
        documents: Sequence[str],
        *,
        metadata: Optional[Sequence[Optional[dict]]] = None,
    ) -> None:
        meta_seq: Sequence[Optional[dict]]
        if metadata is None:
            meta_seq = [None] * len(documents)
        else:
            meta_seq = metadata
        for idx, chunk in enumerate(documents):
            meta = meta_seq[idx] if idx < len(meta_seq) else None
            base_meta = dict(meta or {})
            doc_key = base_meta.get("document_id", f"doc_{idx}")
            section_heading = (
                base_meta.get("section_heading")
                or base_meta.get("section")
                or "General"
            )
            section_path = list(base_meta.get("section_path") or [])
            if not section_path:
                rank = base_meta.get("section_rank")
                if rank:
                    section_path = [part.strip() for part in str(rank).split(">") if part.strip()]
            if not section_path:
                section_path = [section_heading]
            node = self._documents.setdefault(doc_key, _DocumentNode(doc_id=doc_key))
            section = next((s for s in node.sections if s.path == section_path), None)
            if section is None:
                section_meta = {
                    **base_meta,
                    "section_heading": section_heading,
                    "section_path": section_path,
                    "section_rank": " > ".join(section_path),
                }
                section = _SectionNode(
                    heading=section_heading,
                    chunks=[],
                    metadata=section_meta,
                    path=section_path,
                )
                node.sections.append(section)
            section.chunks.append(chunk)

    def search(self, query: str, *, top_k: int = 5) -> List[IndexResult]:
        if not query or not self._documents:
            return []
        query_vec = embed(query)
        section_scores: List[IndexResult] = []
        for node in self._documents.values():
            for section in node.sections:
                combined_text = "\n".join(section.chunks)
                score = cosine_similarity(query_vec, embed(combined_text))
                if score <= 0:
                    continue
                section_meta = dict(section.metadata or {})
                section_meta.update(
                    {
                        "document_id": node.doc_id,
                        "chunk_count": len(section.chunks),
                    }
                )
                section_scores.append(
                    IndexResult(
                        chunk=combined_text,
                        score=float(score),
                        metadata=section_meta,
                    )
                )
        section_scores.sort(key=lambda item: item.score, reverse=True)
        top_sections = section_scores[:top_k]
        # Expand to individual chunks while preserving hierarchy scores.
        expanded: List[IndexResult] = []
        for section in top_sections:
            meta = dict(section.metadata or {})
            meta["section_score"] = section.score
            doc_id = meta.get("document_id", "unknown")
            section_path = meta.get("section_path") or []
            node = self._documents.get(doc_id)
            if node is None:
                continue
            section_node = next((s for s in node.sections if s.path == section_path), None)
            if section_node is None:
                continue
            for chunk in section_node.chunks:
                chunk_score = cosine_similarity(query_vec, embed(chunk))
                if chunk_score <= 0:
                    continue
                expanded.append(
                    IndexResult(
                        chunk=chunk,
                        score=float(chunk_score),
                        metadata={
                            **meta,
                            "section_heading": meta.get("section_heading", section_node.heading),
                            "section_path": section_path,
                        },
                    )
                )
        expanded.sort(key=lambda item: item.score, reverse=True)
        return expanded[:top_k]
