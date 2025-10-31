"""FastAPI application exposing the AI Business Analyst features."""
from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .chunking import describe_chunkers
from .features import FeatureContext, build_default_registry
from .indexing import describe_indexers
from .llm import get_default_client
from .memory import Session, SessionManager
from .schemas import (AttachmentListResponse, AttachmentMetadata,
                      AttachmentUploadResponse, ChatRequest, ChatResponse,
                      ChunkingUpdateRequest, ErrorResponse,
                      EvaluationPerQueryResult, EvaluationRequest,
                      EvaluationResponse, FeatureDescriptor,
                      FeatureListResponse, FeatureName, IndexingUpdateRequest,
                      SearchRequest, SearchResponse, SearchResult,
                      SessionCreateResponse, StrategyCatalogResponse,
                      StrategyDescriptor, TranscriptResponse)

app = FastAPI(
    title="AIBA – AI Business Analyst Assistant",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_manager = SessionManager()
_llm_client = get_default_client()
_feature_registry = build_default_registry()


def get_session(session_id: str = Path(..., description="Session identifier")) -> Session:
    try:
        return _session_manager.get_session(session_id)
    except KeyError as exc:  # pragma: no cover - simple mapping error
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/features", response_model=FeatureListResponse)
def list_features() -> FeatureListResponse:
    """Enumerate available AI features."""
    temp_session = _session_manager.create_session()
    try:
        ctx = FeatureContext(session=temp_session, llm=_llm_client)
        features: list[FeatureDescriptor] = []
        for key in _feature_registry.keys():
            feature_instance = _feature_registry.create(key, ctx)
            features.append(
                FeatureDescriptor(
                    key=FeatureName(key), description=feature_instance.description
                )
            )
        return FeatureListResponse(features=features)
    finally:
        _session_manager.delete_session(temp_session.session_id)


@app.post("/sessions", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    session = _session_manager.create_session()
    return SessionCreateResponse(session_id=session.session_id)


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    _session_manager.delete_session(session_id)


@app.get(
    "/sessions/{session_id}/transcript",
    response_model=TranscriptResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_transcript(session: Session = Depends(get_session)) -> TranscriptResponse:
    return TranscriptResponse(messages=session.memory.as_list())


@app.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponse,
    responses={404: {"model": ErrorResponse}},
)
def chat_with_feature(
    request: ChatRequest,
    session: Session = Depends(get_session),
) -> ChatResponse:
    feature_key = request.feature.value
    ctx = FeatureContext(session=session, llm=_llm_client)
    try:
        feature = _feature_registry.create(feature_key, ctx)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.memory.append("user", request.message, feature=feature_key)
    if request.metadata:
        session.set_state("last_metadata", request.metadata)
    result = feature.run(request.message, context=ctx)
    response = ChatResponse(
        feature=request.feature,
        title=result.title,
        summary=result.summary,
        data=result.data,
        session_state=dict(session.state),
    )
    return response


@app.get("/strategies", response_model=StrategyCatalogResponse)
def list_strategies() -> StrategyCatalogResponse:
    chunking = [StrategyDescriptor(**item) for item in describe_chunkers()]
    indexing = [StrategyDescriptor(**item) for item in describe_indexers()]
    return StrategyCatalogResponse(chunking=chunking, indexing=indexing)


@app.get(
    "/sessions/{session_id}/attachments",
    response_model=AttachmentListResponse,
    responses={404: {"model": ErrorResponse}},
)
def list_attachments(session: Session = Depends(get_session)) -> AttachmentListResponse:
    attachments = [
        AttachmentMetadata(**attachment.as_dict(preview_chars=320))
        for attachment in session.list_attachments()
    ]
    return AttachmentListResponse(attachments=attachments)


@app.post(
    "/sessions/{session_id}/attachments",
    response_model=AttachmentUploadResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def upload_attachments(
    session: Session = Depends(get_session),
    files: List[UploadFile] = File(..., description="Documents to attach to the session"),
) -> AttachmentUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files supplied")
    processed: List[AttachmentMetadata] = []
    for file in files:
        content = await file.read()
        await file.close()
        if not content:
            continue
        try:
            attachment = session.add_attachment(
                filename=file.filename or "attachment",
                content_type=file.content_type or "application/octet-stream",
                data=content,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        processed.append(AttachmentMetadata(**attachment.as_dict(preview_chars=320)))
    if not processed:
        raise HTTPException(status_code=400, detail="Unable to process provided files")
    return AttachmentUploadResponse(attachments=processed)


@app.delete(
    "/sessions/{session_id}/attachments/{attachment_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
)
def delete_attachment(
    attachment_id: str,
    session: Session = Depends(get_session),
) -> None:
    if not session.remove_attachment(attachment_id):
        raise HTTPException(status_code=404, detail="Attachment not found")


@app.post(
    "/sessions/{session_id}/chunking",
    response_model=dict,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
def set_chunking_strategy(
    request: ChunkingUpdateRequest,
    session: Session = Depends(get_session),
) -> dict:
    try:
        session.set_chunking_strategy(request.strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dict(session.state)


@app.post(
    "/sessions/{session_id}/indexing",
    response_model=dict,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
def set_indexing_strategy(
    request: IndexingUpdateRequest,
    session: Session = Depends(get_session),
) -> dict:
    try:
        session.set_indexing_strategy(request.strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dict(session.state)


@app.post(
    "/sessions/{session_id}/search",
    response_model=SearchResponse,
    responses={404: {"model": ErrorResponse}},
)
def search_index(
    request: SearchRequest,
    session: Session = Depends(get_session),
) -> SearchResponse:
    results = session.search(request.query, top_k=request.top_k)
    payload = [
        SearchResult(chunk=result.chunk, score=result.score, metadata=result.metadata)
        for result in results
    ]
    return SearchResponse(results=payload)


@app.post(
    "/sessions/{session_id}/evaluation",
    response_model=EvaluationResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
def evaluate_index(
    request: EvaluationRequest,
    session: Session = Depends(get_session),
) -> EvaluationResponse:
    if not request.queries:
        raise HTTPException(status_code=400, detail="No queries provided for evaluation")
    evaluation = session.evaluate_retrieval(
        [query.dict() for query in request.queries],
        latency_samples_ms=request.latency_samples_ms,
        index_build_ms=request.index_build_ms,
        throughput_qps=request.throughput_qps,
    )
    per_query = [
        EvaluationPerQueryResult(
            query=item["query"],
            top_k=item["top_k"],
            precision_at_k=item["precision_at_k"],
            recall_at_k=item["recall_at_k"],
            mrr=item["mrr"],
            ndcg_at_k=item["ndcg_at_k"],
        )
        for item in evaluation["per_query"]
    ]
    return EvaluationResponse(
        precision_at_k=evaluation["precision_at_k"],
        recall_at_k=evaluation["recall_at_k"],
        mrr=evaluation["mrr"],
        ndcg_at_k=evaluation["ndcg_at_k"],
        efficiency=evaluation["efficiency"],
        per_query=per_query,
    )


@app.get(
    "/sessions/{session_id}/state",
    response_model=dict,
    responses={404: {"model": ErrorResponse}},
)
def get_session_state(session: Session = Depends(get_session)) -> dict:
    return dict(session.state)
