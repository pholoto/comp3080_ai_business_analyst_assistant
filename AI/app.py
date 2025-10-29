"""FastAPI application exposing the AI Business Analyst features."""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware

from .features import FeatureContext, build_default_registry
from .llm import get_default_client
from .memory import Session, SessionManager
from .schemas import (ChatRequest, ChatResponse, ErrorResponse,
                      FeatureDescriptor, FeatureListResponse, FeatureName,
                      SessionCreateResponse, TranscriptResponse)

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


@app.get(
    "/sessions/{session_id}/state",
    response_model=dict,
    responses={404: {"model": ErrorResponse}},
)
def get_session_state(session: Session = Depends(get_session)) -> dict:
    return dict(session.state)
