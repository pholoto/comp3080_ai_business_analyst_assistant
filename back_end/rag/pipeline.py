"""End-to-end RAG orchestration helpers and CLI."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

from common.prompts import (DEFAULT_GUARDRAILS, DEFAULT_SYSTEM_PROMPT,
                            DEFAULT_TASK_PROMPT)

from .config import DEFAULT_CONFIG, RagConfig
from .document_store import DocumentRecord, DocumentStore
from .embedding_generator import EmbeddingGenerator
from .generation import ResponseGenerator
from .prompts import PromptContext, build_context_block, build_user_message
from .ranking import SimilarityRanker
from .retrieval import ContextRetriever
from .text_splitter import TextSplitter
from .vector_store import FaissVectorStore, InMemoryVectorStore


@dataclass
class RagQuery:
    """Input parameters for the unified rag_pipeline call."""

    user_id: str
    question: str
    top_k: int | None = None
    tags: Sequence[str] | None = None
    score_threshold: float | None = None
    recency_boost_days: int | None = None
    recency_boost_weight: float | None = None
    task_prompt: str | None = None
    system_prompt: str | None = None
    guardrails: str | None = None


@dataclass
class RagCitation:
    """Structured citation entry returned alongside the answer."""

    source: str
    chunk_id: str
    chunk_index: int
    score: float


@dataclass
class RagAnswer:
    """Result payload from rag_pipeline."""

    answer: str
    citations: List[RagCitation]
    used_context: List[dict[str, object]]
    prompt_template: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "answer": self.answer,
                "citations": [citation.__dict__ for citation in self.citations],
                "used_context": self.used_context,
                "prompt_template": self.prompt_template,
                "timings_ms": self.timings_ms,
            },
            ensure_ascii=True,
            indent=2,
        )


@dataclass
class DocumentIngestResult:
    metadata: dict[str, object]
    chunks_indexed: int


class RagPipeline:
    """High-level orchestrator for ingestion and question answering."""

    def __init__(
        self,
        *,
        document_store: DocumentStore | None = None,
        text_splitter: TextSplitter | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: FaissVectorStore | InMemoryVectorStore | None = None,
        retriever: ContextRetriever | None = None,
        generator: ResponseGenerator | None = None,
        ranker: SimilarityRanker | None = None,
        config: RagConfig | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.document_store = document_store or DocumentStore(self.config)
        self.text_splitter = text_splitter or TextSplitter(self.config)
        self.embedding_generator = embedding_generator or EmbeddingGenerator(self.config)
        self.vector_store = vector_store or FaissVectorStore(self.config)
        self.retriever = retriever or ContextRetriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            config=self.config,
        )
        self.generator = generator or ResponseGenerator()
        self.ranker = ranker or SimilarityRanker()

    # ------------------------------------------------------------------
    # Indexing / ingestion
    # ------------------------------------------------------------------
    def ingest_document(
        self,
        *,
        user_id: str,
        filename: str,
        data: bytes,
        tags: Sequence[str] | None = None,
    ) -> DocumentIngestResult:
        record = self.document_store.ingest_file(user_id, filename, data, tags=tags)
        chunks = self.text_splitter.split_record(record)
        embeddings = self.embedding_generator.embed_chunks(chunks)
        self.vector_store.add_chunks(user_id, chunks, embeddings)
        return DocumentIngestResult(
            metadata=_metadata_to_dict(record),
            chunks_indexed=len(chunks),
        )

    # ------------------------------------------------------------------
    # Retrieval + generation
    # ------------------------------------------------------------------
    def run(self, query: RagQuery) -> RagAnswer:
        start = perf_counter()
        retrieval_start = perf_counter()
        results = self.retriever.retrieve(
            query.user_id,
            query.question,
            top_k=query.top_k,
            tags=query.tags,
            score_threshold=query.score_threshold,
            recency_boost_days=query.recency_boost_days,
            recency_boost_weight=query.recency_boost_weight,
        )
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        context: PromptContext = build_context_block(
            results,
            max_chars=self.config.max_context_chars,
        )
        system_prompt = (query.system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        task_prompt = (query.task_prompt or DEFAULT_TASK_PROMPT).strip()
        guardrails = query.guardrails or DEFAULT_GUARDRAILS
        user_message = build_user_message(
            question=query.question,
            context_block=context.text_block,
            task_prompt=task_prompt,
            guardrails=guardrails,
        )
        answer = self.generator.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.2,
            max_tokens=700,
        )
        ranked = self.ranker.rank(results, top_n=query.top_k or self.config.top_k)
        citations = [
            RagCitation(
                source=entry.source_name,
                chunk_id=entry.chunk_id,
                chunk_index=entry.chunk_index,
                score=entry.score,
            )
            for entry in ranked
        ]
        total_ms = (perf_counter() - start) * 1000
        return RagAnswer(
            answer=answer,
            citations=citations,
            used_context=context.used_chunks,
            prompt_template=user_message,
            timings_ms={
                "retrieval": round(retrieval_ms, 2),
                "total": round(total_ms, 2),
            },
        )


def rag_pipeline(query: RagQuery, *, pipeline: RagPipeline | None = None) -> RagAnswer:
    """Functional wrapper for callers that prefer a single function."""
    engine = pipeline or RagPipeline()
    return engine.run(query)


# ----------------------------------------------------------------------
# CLI / demo helpers
# ----------------------------------------------------------------------


def _ingest_folder(pipeline: RagPipeline, user_id: str, folder: Path) -> int:
    count = 0
    for path in folder.glob("*"):
        if path.is_dir():
            continue
        data = path.read_bytes()
        pipeline.ingest_document(user_id=user_id, filename=path.name, data=data)
        count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline demo")
    parser.add_argument("question", help="User question to answer")
    parser.add_argument("--user-id", default="demo", help="User namespace")
    parser.add_argument(
        "--ingest-folder",
        type=Path,
        help="Optional folder of documents to ingest before answering",
    )
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    pipeline = RagPipeline()
    if args.ingest_folder and args.ingest_folder.exists():
        ingested = _ingest_folder(pipeline, args.user_id, args.ingest_folder)
        print(f"Ingested {ingested} documents from {args.ingest_folder}.")

    response = pipeline.run(
        RagQuery(
            user_id=args.user_id,
            question=args.question,
            top_k=args.top_k,
        )
    )
    print(response.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


def _metadata_to_dict(record: DocumentRecord) -> dict[str, object]:
    metadata = record.metadata
    return {
        "document_id": metadata.document_id,
        "user_id": metadata.user_id,
        "original_name": metadata.original_name,
        "stored_name": metadata.stored_name,
        "stored_path": metadata.stored_path,
        "mime_type": metadata.mime_type,
        "checksum": metadata.checksum,
        "tags": metadata.tags,
        "created_at": metadata.created_at,
    }
