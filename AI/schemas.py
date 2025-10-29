"""Pydantic schemas for the AI BA assistant API."""
from __future__ import annotations

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
