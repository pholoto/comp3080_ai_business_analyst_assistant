"""FastAPI application exposing the backend RAG capabilities."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag import (ContextRetriever, DocumentStore, EmbeddingGenerator,
                  FaissVectorStore, ResponseGenerator, SimilarityRanker,
                  TextSplitter)
from .rag.agents import (IdeationAgent, IdeationRequest, ProgressAgent,
                         ProgressRequest)
from .rag.config import DEFAULT_CONFIG
from .rag.document_store import (DuplicateDocumentError,
                                 UnsupportedDocumentError)

app = FastAPI(title="AI Business Analyst Assistant Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DocumentMetadataModel(BaseModel):
    document_id: str
    original_name: str
    stored_name: str
    stored_path: str
    mime_type: str
    checksum: str
    tags: List[str]
    created_at: str


class DocumentIngestResponse(BaseModel):
    metadata: DocumentMetadataModel
    chunks_indexed: int = Field(..., description="Number of chunks added to the vector store")


class RankingEntryModel(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    source_name: str
    score: float
    preview: str


class IdeationPayload(BaseModel):
    topic: str
    desired_ideas: Optional[int] = Field(6, ge=1, le=10)
    top_k: Optional[int] = Field(8, ge=1, le=50)
    tags: Optional[List[str]] = None


class IdeationResponseModel(BaseModel):
    answer: str
    sources: List[RankingEntryModel]
    context_items: List[dict]


class ProgressPayload(BaseModel):
    reference_date: Optional[str] = Field(
        None, description="ISO date (YYYY-MM-DD) representing the reporting date"
    )
    top_k: Optional[int] = Field(8, ge=1, le=50)
    tags: Optional[List[str]] = None


class ProgressResponseModel(BaseModel):
    answer: str
    sources: List[RankingEntryModel]
    context_items: List[dict]
    reference_date: Optional[str]


class RagDependencies:
    """Container that wires together the reusable RAG components."""

    def __init__(self) -> None:
        self.config = DEFAULT_CONFIG
        # Reuse single instances to avoid duplicate model loads.
        self.document_store = DocumentStore(self.config)
        self.text_splitter = TextSplitter(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.vector_store = FaissVectorStore(self.config)
        self.retriever = ContextRetriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            config=self.config,
        )
        self.generator = ResponseGenerator()
        self.ranker = SimilarityRanker()
        self.ideation_agent = IdeationAgent(
            retriever=self.retriever,
            generator=self.generator,
            ranker=self.ranker,
            config=self.config,
        )
        self.progress_agent = ProgressAgent(
            retriever=self.retriever,
            generator=self.generator,
            ranker=self.ranker,
            config=self.config,
        )


_DEPENDENCIES = RagDependencies()


def get_dependencies() -> RagDependencies:
    return _DEPENDENCIES


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/users/{user_id}/documents", response_model=List[DocumentMetadataModel])
def list_documents(user_id: str, deps: RagDependencies = Depends(get_dependencies)):
    metadata = deps.document_store.list_documents(user_id)
    return [
        DocumentMetadataModel(
            document_id=item.document_id,
            original_name=item.original_name,
            stored_name=item.stored_name,
            stored_path=item.stored_path,
            mime_type=item.mime_type,
            checksum=item.checksum,
            tags=list(item.tags),
            created_at=item.created_at,
        )
        for item in metadata
    ]


@app.post("/users/{user_id}/documents", response_model=DocumentIngestResponse)
async def ingest_document(
    user_id: str,
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None, description="Comma separated tag list"),
    deps: RagDependencies = Depends(get_dependencies),
):
    raw_bytes = await file.read()
    tag_list = _parse_tags(tags)
    filename = file.filename or "uploaded_file"
    try:
        record = deps.document_store.ingest_file(
            user_id,
            filename,
            raw_bytes,
            tags=tag_list,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = deps.text_splitter.split_record(record)
    embeddings = deps.embedding_generator.embed_chunks(chunks)
    deps.vector_store.add_chunks(user_id, chunks, embeddings)

    return DocumentIngestResponse(
        metadata=DocumentMetadataModel(
            document_id=record.metadata.document_id,
            original_name=record.metadata.original_name,
            stored_name=record.metadata.stored_name,
            stored_path=record.metadata.stored_path,
            mime_type=record.metadata.mime_type,
            checksum=record.metadata.checksum,
            tags=list(record.metadata.tags),
            created_at=record.metadata.created_at,
        ),
        chunks_indexed=len(chunks),
    )


@app.post("/users/{user_id}/ideation", response_model=IdeationResponseModel)
def generate_ideas(
    user_id: str,
    payload: IdeationPayload,
    deps: RagDependencies = Depends(get_dependencies),
):
    request = IdeationRequest(
        topic=payload.topic,
        desired_ideas=payload.desired_ideas or 6,
        top_k=payload.top_k or deps.config.top_k,
        tags=tuple(payload.tags) if payload.tags else None,
    )
    result = deps.ideation_agent.generate_ideas(user_id, request)
    raw_sources = cast(List[Dict[str, Any]], result.get("sources", []))
    sources = [RankingEntryModel(**entry) for entry in raw_sources]
    context_items = cast(List[Dict[str, Any]], result.get("context_items", []))
    answer_text = str(result.get("answer", ""))
    return IdeationResponseModel(
        answer=answer_text,
        sources=sources,
        context_items=context_items,
    )


@app.post("/users/{user_id}/progress", response_model=ProgressResponseModel)
def analyze_progress(
    user_id: str,
    payload: ProgressPayload,
    deps: RagDependencies = Depends(get_dependencies),
):
    request = ProgressRequest(
        reference_date=payload.reference_date,
        top_k=payload.top_k or deps.config.top_k,
        tags=tuple(payload.tags) if payload.tags else None,
    )
    result = deps.progress_agent.analyze_progress(user_id, request)
    raw_sources = cast(List[Dict[str, Any]], result.get("sources", []))
    sources = [RankingEntryModel(**entry) for entry in raw_sources]
    context_items = cast(List[Dict[str, Any]], result.get("context_items", []))
    answer_text = str(result.get("answer", ""))
    reference_date = cast(Optional[str], result.get("reference_date"))
    return ProgressResponseModel(
        answer=answer_text,
        sources=sources,
        context_items=context_items,
        reference_date=reference_date,
    )


def _parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tag.strip().lower() for tag in raw.split(",") if tag.strip()]


__all__ = ["app"]
