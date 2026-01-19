"""API Router for Phase-based Analysis Endpoints."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from AI.llm import LLMClient, get_default_client

from .phases import (DocumentationInput,  # Phase 1; Phase 2; Phase 3; Phase 7
                     DocumentationOutput, DocumentationPhase,
                     DocumentationType, FeatureAnalyzerInput,
                     FeatureAnalyzerOutput, MarketAnalysisInput,
                     MarketAnalysisOutput, MarketAnalysisPhase,
                     CompetitorResearchInput, CompetitorResearchOutput,
                     ProblemDefinitionInput, ProblemDefinitionOutput,
                     ProblemDefinitionPhase, RequirementsAnalysisPhase,
                     UserJourneyInput, UserJourneyOutput)

router = APIRouter(prefix="/api/v2/phases", tags=["phases"])


# ============================================================================
# Shared State Management
# ============================================================================

class PhaseStateManager:
    """Manages state across phases for a user session."""
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._llm_client: Optional[LLMClient] = None
    
    def get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = get_default_client()
        return self._llm_client
    
    def get_session(self, user_id: str) -> Dict[str, Any]:
        """Get or create session for user."""
        if user_id not in self._sessions:
            self._sessions[user_id] = {
                "phase1_output": None,
                "phase2_output": None,
                "phase3_output": None,
            }
        return self._sessions[user_id]
    
    def set_phase_output(self, user_id: str, phase: str, output: Any) -> None:
        """Store phase output in session."""
        session = self.get_session(user_id)
        session[f"{phase}_output"] = output
    
    def get_phase_output(self, user_id: str, phase: str) -> Optional[Any]:
        """Retrieve phase output from session."""
        session = self.get_session(user_id)
        return session.get(f"{phase}_output")


_state_manager = PhaseStateManager()


def get_state_manager() -> PhaseStateManager:
    """Dependency injection for state manager."""
    return _state_manager


# ============================================================================
# Phase 1: Problem Definition Endpoints
# ============================================================================

class Phase1Request(BaseModel):
    """Request body for Phase 1."""
    problem_description: str = Field(
        ...,
        description="Describe the problem in your own words. What challenge or pain point are you addressing?",
        min_length=10,
    )
    target_users: str = Field(
        ...,
        description="Who experiences this problem? Be specific about the user group.",
        min_length=5,
    )
    why_it_matters: str = Field(
        ...,
        description="What's the impact? Why should this problem be solved?",
        min_length=10,
    )
    pain_points: list[str] = Field(
        default_factory=list,
        description="Specific pain points your users experience.",
    )
    has_existing_solutions: bool = Field(
        default=False,
        description="Are there existing solutions in the market?",
    )
    current_solutions: Optional[str] = Field(
        default=None,
        description="If existing solutions exist, what are they and what are potential issues?",
    )


@router.post(
    "/users/{user_id}/phase1/problem-definition",
    response_model=ProblemDefinitionOutput,
    summary="Phase 1: Problem Definition",
    description="Analyze problem definition and generate comprehensive analysis including quality scoring, normalized problem summary, user personas, pain moments, root cause analysis, and more.",
)
async def run_phase1_problem_definition(
    user_id: str,
    request: Phase1Request,
    state: PhaseStateManager = Depends(get_state_manager),
) -> ProblemDefinitionOutput:
    """Execute Phase 1: Problem Definition analysis."""
    
    # Convert request to input model
    input_data = ProblemDefinitionInput(
        problem_description=request.problem_description,
        target_users=request.target_users,
        why_it_matters=request.why_it_matters,
        pain_points=request.pain_points,
        has_existing_solutions=request.has_existing_solutions,
        current_solutions=request.current_solutions,
    )
    
    # Create and run phase
    phase = ProblemDefinitionPhase(llm_client=state.get_llm_client())
    
    try:
        output = phase.run(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phase 1 analysis failed: {str(e)}")
    
    # Store output for subsequent phases
    state.set_phase_output(user_id, "phase1", output.model_dump())
    
    return output


# ============================================================================
# Phase 2: Requirements Analysis Endpoints
# ============================================================================

class Phase2FeatureAnalyzerRequest(BaseModel):
    """Request body for Phase 2 Feature Analyzer."""
    desired_features: list[str] = Field(
        ...,
        description="List of desired features",
        min_length=1,
    )
    mvp_goal: str = Field(
        ...,
        description="Very specific MVP goal using SMART model",
        min_length=10,
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Project deadline",
    )
    team_skill_level: Optional[str] = Field(
        default=None,
        description="Team skill level: solo-developer, junior-team, senior-team, mixed-experience",
    )
    additional_constraints: Optional[str] = Field(
        default=None,
        description="Any additional constraints or limitations",
    )


class Phase2UserJourneyRequest(BaseModel):
    """Request body for Phase 2 User Journey Generator."""
    selected_feature: str = Field(
        ...,
        description="One feature from the List of Desired Features to generate user journey for",
    )


@router.post(
    "/users/{user_id}/phase2/feature-analyzer",
    response_model=FeatureAnalyzerOutput,
    summary="Phase 2: Feature Analyzer",
    description="Analyze features and generate normalized feature list, functional/non-functional requirements, MVP scope, and scope warnings.",
)
async def run_phase2_feature_analyzer(
    user_id: str,
    request: Phase2FeatureAnalyzerRequest,
    state: PhaseStateManager = Depends(get_state_manager),
) -> FeatureAnalyzerOutput:
    """Execute Phase 2: Feature Analyzer."""
    
    # Get Phase 1 output for context
    phase1_output = state.get_phase_output(user_id, "phase1")
    
    # Extract primary persona from Phase 1 if available
    primary_persona = None
    if phase1_output and phase1_output.get("personas"):
        personas = phase1_output["personas"]
        if personas.get("primary_user"):
            primary_persona = personas["primary_user"].get("role", "")
    
    # Convert team skill level string to enum if provided
    from .phases.phase2_requirements_analysis import TeamSkillLevel
    team_skill = None
    if request.team_skill_level:
        try:
            team_skill = TeamSkillLevel(request.team_skill_level)
        except ValueError:
            pass
    
    # Convert request to input model
    input_data = FeatureAnalyzerInput(
        desired_features=request.desired_features,
        mvp_goal=request.mvp_goal,
        primary_user_persona=primary_persona,
        deadline=request.deadline,
        team_skill_level=team_skill,
        additional_constraints=request.additional_constraints,
    )
    
    # Create and run phase
    phase = RequirementsAnalysisPhase(
        llm_client=state.get_llm_client(),
        phase1_context=phase1_output,
    )
    
    try:
        output = phase.analyze_features(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature analysis failed: {str(e)}")
    
    # Store output for subsequent phases
    state.set_phase_output(user_id, "phase2", output.model_dump())
    
    return output


@router.post(
    "/users/{user_id}/phase2/user-journey",
    response_model=UserJourneyOutput,
    summary="Phase 2: User Journey Generator",
    description="Generate step-by-step user journey for a selected feature.",
)
async def run_phase2_user_journey(
    user_id: str,
    request: Phase2UserJourneyRequest,
    state: PhaseStateManager = Depends(get_state_manager),
) -> UserJourneyOutput:
    """Execute Phase 2: User Journey Generator."""
    
    # Get Phase 1 and 2 outputs for context
    phase1_output = state.get_phase_output(user_id, "phase1")
    phase2_output = state.get_phase_output(user_id, "phase2")
    
    # Find feature context from Phase 2 output
    feature_context = None
    if phase2_output and phase2_output.get("normalized_features"):
        for feature in phase2_output["normalized_features"]:
            if (feature.get("original_name") == request.selected_feature or
                feature.get("normalized_name") == request.selected_feature):
                feature_context = feature
                break
    
    # Convert request to input model
    input_data = UserJourneyInput(
        selected_feature=request.selected_feature,
        feature_context=feature_context,
    )
    
    # Create and run phase
    phase = RequirementsAnalysisPhase(
        llm_client=state.get_llm_client(),
        phase1_context=phase1_output,
    )
    
    try:
        output = phase.generate_user_journey(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User journey generation failed: {str(e)}")
    
    return output


# ============================================================================
# Phase 3: Market Analysis Endpoints
# ============================================================================

class Phase3Request(BaseModel):
    """Request body for Phase 3."""
    geographic_scope: Optional[str] = Field(
        default=None,
        description="Geographic scope: local, regional, multi-regional, national, international, global",
    )
    industry_context: Optional[str] = Field(
        default=None,
        description="Industry context (e.g., 'EdTech', 'FinTech', 'Healthcare')",
    )
    competitors: Optional[list[str]] = Field(
        default=None,
        description="Known competitors to analyze",
    )


@router.post(
    "/users/{user_id}/phase3/market-analysis",
    response_model=MarketAnalysisOutput,
    summary="Phase 3: Market Analysis",
    description="Generate market research summary, Porter's Five Forces analysis, competitor analysis, and unique selling points.",
)
async def run_phase3_market_analysis(
    user_id: str,
    request: Phase3Request,
    state: PhaseStateManager = Depends(get_state_manager),
) -> MarketAnalysisOutput:
    """Execute Phase 3: Market Analysis."""
    
    # Get previous phase outputs for context
    phase1_output = state.get_phase_output(user_id, "phase1")
    phase2_output = state.get_phase_output(user_id, "phase2")
    
    # Extract context from Phase 1
    problem_summary = None
    target_users = None
    if phase1_output:
        if phase1_output.get("normalized_summary"):
            problem_summary = phase1_output["normalized_summary"].get("summary", "")
        if phase1_output.get("personas", {}).get("primary_user"):
            target_users = phase1_output["personas"]["primary_user"].get("role", "")
    
    # Extract MVP goal from Phase 2
    mvp_goal = None
    if phase2_output and phase2_output.get("mvp_scope"):
        mvp_goal = phase2_output["mvp_scope"].get("mvp_rationale", "")
    
    # Convert geographic scope string to enum if provided
    from .phases.phase3_market_analysis import GeographicScope
    geo_scope = None
    if request.geographic_scope:
        try:
            geo_scope = GeographicScope(request.geographic_scope)
        except ValueError:
            pass
    
    # Convert request to input model
    input_data = MarketAnalysisInput(
        geographic_scope=geo_scope,
        industry_context=request.industry_context,
        competitors=request.competitors,
        problem_summary=problem_summary,
        target_users=target_users,
        mvp_goal=mvp_goal,
    )
    
    # Create and run phase
    phase = MarketAnalysisPhase(
        llm_client=state.get_llm_client(),
        phase1_context=phase1_output,
        phase2_context=phase2_output,
    )
    
    try:
        output = phase.run(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market analysis failed: {str(e)}")
    
    # Store output for subsequent phases
    state.set_phase_output(user_id, "phase3", output.model_dump())
    
    return output


@router.post(
    "/users/{user_id}/phase3/research-competitors",
    response_model=CompetitorResearchOutput,
    summary="Quick Competitor Research",
    description="Research competitors based on industry and known items.",
)
async def research_competitors(
    user_id: str,
    request: CompetitorResearchInput,
    state: PhaseStateManager = Depends(get_state_manager),
) -> CompetitorResearchOutput:
    """Execute quick competitor research."""
    
    # Create phase instance
    phase = MarketAnalysisPhase(llm_client=state.get_llm_client())
    
    try:
        output = phase.research_competitors(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Competitor research failed: {str(e)}")
    
    return output


# ============================================================================
# Phase 7: Documentation Endpoints
# ============================================================================

class Phase7Request(BaseModel):
    """Request body for Phase 7."""
    document_type: str = Field(
        ...,
        description="Type of documentation: academic-report, software-engineering, business-proposal, startup-pitch",
    )
    project_title: Optional[str] = Field(
        default=None,
        description="Project title for the document",
    )
    author_name: Optional[str] = Field(
        default=None,
        description="Author or team name",
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Any additional context or specific requirements",
    )


@router.post(
    "/users/{user_id}/phase7/documentation",
    response_model=DocumentationOutput,
    summary="Phase 7: Documentation",
    description="Generate documentation in the selected format (Academic Report, SRS, Business Proposal, or Startup Pitch).",
)
async def run_phase7_documentation(
    user_id: str,
    request: Phase7Request,
    state: PhaseStateManager = Depends(get_state_manager),
) -> DocumentationOutput:
    """Execute Phase 7: Documentation."""
    
    # Get all previous phase outputs for context
    phase1_output = state.get_phase_output(user_id, "phase1")
    phase2_output = state.get_phase_output(user_id, "phase2")
    phase3_output = state.get_phase_output(user_id, "phase3")
    
    # Convert document type string to enum
    try:
        doc_type = DocumentationType(request.document_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type. Must be one of: {[t.value for t in DocumentationType]}"
        )
    
    # Convert request to input model
    input_data = DocumentationInput(
        document_type=doc_type,
        project_title=request.project_title,
        author_name=request.author_name,
        additional_context=request.additional_context,
        phase1_output=phase1_output,
        phase2_output=phase2_output,
        phase3_output=phase3_output,
    )
    
    # Create and run phase
    phase = DocumentationPhase(llm_client=state.get_llm_client())
    
    try:
        output = phase.run(input_data)
        
        # Export the document to DOCX
        exported_files = phase.export_document(output, user_id)
        
        # Set the download URL if a DOCX was generated
        if "docx" in exported_files:
            # Construct URL relative to the mounted static directory
            # path is like "back_end/data/user_id/filename.docx"
            # mounted at /export
            # URL should be /export/user_id/filename.docx
            import os
            relative_path = os.path.relpath(exported_files["docx"], "back_end/data")
            output.download_url = f"/export/{relative_path}"
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Documentation generation failed: {str(e)}")
    
    return output


# ============================================================================
# Session Management Endpoints
# ============================================================================

class SessionSummaryResponse(BaseModel):
    """Response model for session summary."""
    user_id: str
    has_phase1: bool
    has_phase2: bool
    has_phase3: bool
    phase1_summary: Optional[Dict[str, Any]] = None
    phase2_summary: Optional[Dict[str, Any]] = None
    phase3_summary: Optional[Dict[str, Any]] = None


@router.get(
    "/users/{user_id}/session",
    response_model=SessionSummaryResponse,
    summary="Get Session Summary",
    description="Get summary of completed phases for a user session.",
)
async def get_session_summary(
    user_id: str,
    state: PhaseStateManager = Depends(get_state_manager),
) -> SessionSummaryResponse:
    """Get session summary showing completed phases."""
    
    phase1 = state.get_phase_output(user_id, "phase1")
    phase2 = state.get_phase_output(user_id, "phase2")
    phase3 = state.get_phase_output(user_id, "phase3")
    
    # Create brief summaries
    phase1_summary = None
    if phase1:
        phase1_summary = {
            "quality_score": phase1.get("quality_score", {}).get("overall_score"),
            "problem_summary": phase1.get("normalized_summary", {}).get("summary", "")[:200],
        }
    
    phase2_summary = None
    if phase2:
        phase2_summary = {
            "feature_count": len(phase2.get("normalized_features", [])),
            "functional_req_count": len(phase2.get("functional_requirements", [])),
            "mvp_features": [f.get("feature_name") for f in phase2.get("mvp_scope", {}).get("included_features", [])],
        }
    
    phase3_summary = None
    if phase3:
        phase3_summary = {
            "competitor_count": len(phase3.get("competitor_analysis", {}).get("competitors", [])),
            "primary_usp": phase3.get("usp_generation", {}).get("primary_usp", {}).get("usp", ""),
        }
    
    return SessionSummaryResponse(
        user_id=user_id,
        has_phase1=phase1 is not None,
        has_phase2=phase2 is not None,
        has_phase3=phase3 is not None,
        phase1_summary=phase1_summary,
        phase2_summary=phase2_summary,
        phase3_summary=phase3_summary,
    )


@router.delete(
    "/users/{user_id}/session",
    summary="Clear Session",
    description="Clear all phase outputs for a user session.",
)
async def clear_session(
    user_id: str,
    state: PhaseStateManager = Depends(get_state_manager),
) -> Dict[str, str]:
    """Clear user session data."""
    
    if user_id in state._sessions:
        del state._sessions[user_id]
    
    return {"status": "success", "message": f"Session cleared for user {user_id}"}
