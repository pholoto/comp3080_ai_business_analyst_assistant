"""FastAPI application exposing the backend RAG capabilities."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from AI.features import FeatureContext, build_default_registry
from AI.llm import LLMClient
from AI.memory import Session, SessionManager
from AI.schemas import FeatureName

# Import the new phase routes
from .phase_routes import router as phase_router
from .rag import (ContextRetriever, DocumentStore, EmbeddingGenerator,
                  FaissVectorStore, RagPipeline, RagQuery, ResponseGenerator,
                  SimilarityRanker, TextSplitter)
from .rag.agents import (IdeationAgent, IdeationRequest, ProgressAgent,
                         ProgressRequest)
from .rag.config import DEFAULT_CONFIG
from .rag.context_blocks import build_context_block
from .rag.document_store import (DuplicateDocumentError,
                                 UnsupportedDocumentError)

app = FastAPI(title="AI Business Analyst Assistant Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the new phase routes
from .phase_routes import router as phase_router
from fastapi.staticfiles import StaticFiles
import os

# Ensure data directory exists
os.makedirs("back_end/data", exist_ok=True)

app.include_router(phase_router)

# Mount static files for exported documents
app.mount("/export", StaticFiles(directory="back_end/data"), name="export")

LOGGER = logging.getLogger(__name__)


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
    chunk_index: int
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


class CitationModel(BaseModel):
    source: str
    chunk_id: str
    chunk_index: int
    score: float


class RagQueryPayload(BaseModel):
    question: str
    top_k: Optional[int] = Field(None, ge=1, le=50)
    tags: Optional[List[str]] = None
    score_threshold: Optional[float] = Field(None, ge=0, le=1)
    recency_boost_days: Optional[int] = Field(None, ge=1)
    recency_boost_weight: Optional[float] = Field(None, ge=0, le=1)
    system_prompt: Optional[str] = None
    task_prompt: Optional[str] = None
    guardrails: Optional[str] = None


class RagPipelineResponseModel(BaseModel):
    answer: str
    citations: List[CitationModel]
    used_context: List[dict]
    prompt_template: str
    timings_ms: Dict[str, float]


class ChatPayload(BaseModel):
    feature: FeatureName = Field(..., description="AI feature to engage")
    message: str = Field(..., min_length=1, description="User question or instruction")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata persisted to the session state")
    use_rag: bool = Field(True, description="Whether to run document retrieval before answering")
    rag_top_k: Optional[int] = Field(None, ge=1, le=50)
    rag_tags: Optional[List[str]] = Field(None, description="Optional tag filters for retrieval")
    rag_score_threshold: Optional[float] = Field(None, ge=0, le=1)


class CombinedChatResponseModel(BaseModel):
    feature: FeatureName
    title: str
    summary: str
    data: Dict[str, Any]
    session_state: Dict[str, Any]
    citations: List[CitationModel]
    context_snippets: List[Dict[str, Any]]


class RagDependencies:
    """Container that wires together the reusable RAG components."""

    def __init__(self) -> None:
        self.config = DEFAULT_CONFIG
        # Reuse single instances to avoid duplicate model loads.
        self.document_store = DocumentStore(self.config)
        self.text_splitter = TextSplitter(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        try:
            self.vector_store = FaissVectorStore(self.config)
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is required for document attachments. Install it before starting the backend."
            ) from exc
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
        self.pipeline = RagPipeline(
            document_store=self.document_store,
            text_splitter=self.text_splitter,
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            retriever=self.retriever,
            generator=self.generator,
            ranker=self.ranker,
            config=self.config,
        )
        self.session_manager = SessionManager()
        self.feature_registry = build_default_registry()
        self._user_sessions: Dict[str, str] = {}

    def get_or_create_session(self, user_id: str) -> Session:
        session_id = self._user_sessions.get(user_id)
        if session_id:
            try:
                return self.session_manager.get_session(session_id)
            except KeyError:
                pass
        session = self.session_manager.create_session()
        self._user_sessions[user_id] = session.session_id
        session.set_state("user_id", user_id)
        return session

    def attach_document_to_session(
        self,
        *,
        user_id: str,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> str:
        session = self.get_or_create_session(user_id)
        attachment = session.add_attachment(
            filename=filename,
            content_type=content_type or "application/octet-stream",
            data=data,
        )
        return attachment.attachment_id


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
        ingest_result = deps.pipeline.ingest_document(
            user_id=user_id,
            filename=filename,
            data=raw_bytes,
            tags=tag_list,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    metadata = dict(ingest_result.metadata)
    metadata.pop("user_id", None)
    typed_metadata = cast(Dict[str, Any], metadata)
    try:
        attachment_id = deps.attach_document_to_session(
            user_id=user_id,
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
            data=raw_bytes,
        )
        typed_metadata["session_attachment_id"] = attachment_id
    except Exception as exc:  # pragma: no cover - attachment failures should not block ingestion
        LOGGER.warning("Unable to attach document to conversational session: %s", exc)
    return DocumentIngestResponse(
        metadata=DocumentMetadataModel(**typed_metadata),
        chunks_indexed=ingest_result.chunks_indexed,
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


@app.post("/users/{user_id}/rag", response_model=RagPipelineResponseModel)
def run_rag_pipeline(
    user_id: str,
    payload: RagQueryPayload,
    deps: RagDependencies = Depends(get_dependencies),
):
    query = RagQuery(
        user_id=user_id,
        question=payload.question,
        top_k=payload.top_k,
        tags=tuple(payload.tags) if payload.tags else None,
        score_threshold=payload.score_threshold,
        recency_boost_days=payload.recency_boost_days,
        recency_boost_weight=payload.recency_boost_weight,
        system_prompt=payload.system_prompt,
        task_prompt=payload.task_prompt,
        guardrails=payload.guardrails,
    )
    result = deps.pipeline.run(query)
    citation_models = [
        CitationModel(
            source=citation.source,
            chunk_id=citation.chunk_id,
            chunk_index=citation.chunk_index,
            score=citation.score,
        )
        for citation in result.citations
    ]
    return RagPipelineResponseModel(
        answer=result.answer,
        citations=citation_models,
        used_context=result.used_context,
        prompt_template=result.prompt_template,
        timings_ms=result.timings_ms,
    )


@app.post("/users/{user_id}/chat", response_model=CombinedChatResponseModel)
def run_conversational_feature(
    user_id: str,
    payload: ChatPayload,
    deps: RagDependencies = Depends(get_dependencies),
):
    session = deps.get_or_create_session(user_id)
    citations: List[CitationModel] = []
    context_snippets: List[Dict[str, Any]] = []
    enriched_message = payload.message.strip()

    if payload.use_rag:
        tag_filter = _normalize_tag_list(payload.rag_tags)
        results = deps.retriever.retrieve(
            user_id,
            payload.message,
            top_k=payload.rag_top_k or deps.config.top_k,
            tags=tuple(tag_filter) if tag_filter else None,
            score_threshold=payload.rag_score_threshold,
        )
        prompt_context = build_context_block(
            results,
            max_chars=deps.config.max_context_chars,
        )
        context_snippets = prompt_context.used_chunks
        ranked = deps.ranker.rank(
            results, top_n=payload.rag_top_k or deps.config.top_k
        )
        citations = [
            CitationModel(
                source=entry.source_name,
                chunk_id=entry.chunk_id,
                chunk_index=entry.chunk_index,
                score=entry.score,
            )
            for entry in ranked
        ]
        if prompt_context.text_block and prompt_context.text_block != "[no matching context retrieved]":
            enriched_message = (
                f"{payload.message.strip()}\n\n"
                "Document evidence retrieved from the knowledge base:\n"
                f"{prompt_context.text_block}\n\n"
                "Combine this evidence with your broader analysis expertise."
            )
        session.set_state("last_rag_query", payload.message)
        session.set_state("last_rag_context", context_snippets)
        session.set_state(
            "last_rag_citations",
            [citation.dict() for citation in citations],
        )

    llm_client = cast(LLMClient, deps.generator.llm_client)
    ctx = FeatureContext(session=session, llm=llm_client)
    try:
        feature = deps.feature_registry.create(payload.feature.value, ctx)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session.memory.append("user", payload.message, feature=payload.feature.value)
    if payload.metadata:
        session.set_state("last_metadata", payload.metadata)
    result = feature.run(enriched_message, context=ctx)
    session.memory.append("assistant", result.summary, feature=payload.feature.value)

    return CombinedChatResponseModel(
        feature=payload.feature,
        title=result.title,
        summary=result.summary,
        data=result.data,
        session_state=dict(session.state),
        citations=citations,
        context_snippets=context_snippets,
    )


def _parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tag.strip().lower() for tag in raw.split(",") if tag.strip()]


def _normalize_tag_list(raw: Optional[List[str]]) -> List[str]:
    if not raw:
        return []
    return [tag.strip().lower() for tag in raw if isinstance(tag, str) and tag.strip()]


__all__ = ["app"]
