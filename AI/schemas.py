"""Pydantic schemas for the AI BA assistant API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureName(str, Enum):
    requirement_clarifier = "requirement_clarifier"
    use_case_generator = "use_case_generator"
    feature_prioritization = "feature_prioritization"
    market_fit_analyzer = "market_fit_analyzer"
    stakeholder_insights = "stakeholder_insights"
    ba_report_export = "ba_report_export"


class SessionCreateResponse(BaseModel):
    session_id: str = Field(..., description="Identifier for the newly created session")


class ChatRequest(BaseModel):
    feature: FeatureName = Field(..., description="Feature to engage")
    message: str = Field(..., description="User message or task prompt")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class FeatureDescriptor(BaseModel):
    key: FeatureName
    description: str


class FeatureListResponse(BaseModel):
    features: List[FeatureDescriptor]


class ChatResponse(BaseModel):
    feature: FeatureName
    title: str
    summary: str
    data: Dict[str, Any]
    session_state: Dict[str, Any]


class TranscriptResponse(BaseModel):
    messages: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str


class AttachmentMetadata(BaseModel):
    attachment_id: str
    filename: str
    content_type: str
    size: int
    word_count: int
    added_at: datetime
    preview: str
    metadata: Dict[str, Any]


class AttachmentListResponse(BaseModel):
    attachments: List[AttachmentMetadata]


class AttachmentUploadResponse(BaseModel):
    attachments: List[AttachmentMetadata]


class StrategyDescriptor(BaseModel):
    key: str
    name: str
    description: str


class StrategyCatalogResponse(BaseModel):
    chunking: List[StrategyDescriptor]
    indexing: List[StrategyDescriptor]


class ChunkingUpdateRequest(BaseModel):
    strategy: str = Field(..., description="Chunking strategy key to activate")


class IndexingUpdateRequest(BaseModel):
    strategy: str = Field(..., description="Indexing strategy key to activate")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=50)


class SearchResult(BaseModel):
    chunk: str
    score: float
    metadata: Optional[Dict[str, Any]]


class SearchResponse(BaseModel):
    results: List[SearchResult]


class EvaluationQuery(BaseModel):
    query: str
    relevant_chunks: List[str]
    top_k: Optional[int] = None


class EvaluationRequest(BaseModel):
    queries: List[EvaluationQuery]
    latency_samples_ms: Optional[List[float]] = None
    index_build_ms: Optional[float] = None
    throughput_qps: Optional[float] = None


class EvaluationPerQueryResult(BaseModel):
    query: str
    top_k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


class EvaluationResponse(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    efficiency: Optional[Dict[str, float]]
    per_query: List[EvaluationPerQueryResult]
