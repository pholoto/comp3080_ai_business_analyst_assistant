"""
Phase 1: Problem Definition

This phase captures structured user input about their problem and generates
comprehensive analysis including quality scoring, normalized problem summary,
user personas, pain moments, root cause analysis, and more.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from AI.llm import LLMPrompt
from common.prompts import (PHASE1_EXISTING_SOLUTIONS_PROMPT,
                            PHASE1_IMPACT_STAKES_PROMPT,
                            PHASE1_NORMALIZE_PROBLEM_PROMPT,
                            PHASE1_PAIN_MOMENTS_PROMPT,
                            PHASE1_QUALITY_SCORING_PROMPT,
                            PHASE1_ROOT_CAUSE_PROMPT,
                            PHASE1_SCOPE_BOUNDARY_PROMPT,
                            PHASE1_TRANSITION_QUESTIONS_PROMPT,
                            PHASE1_USER_PERSONA_PROMPT)

# ============================================================================
# Input Models
# ============================================================================

class ProblemDefinitionInput(BaseModel):
    """Input schema for Phase 1: Problem Definition."""
    
    # Required fields (*)
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
    
    # Optional fields
    pain_points: List[str] = Field(
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


# ============================================================================
# Output Models
# ============================================================================

class QualityDimension(BaseModel):
    """Individual quality dimension score."""
    name: str
    score: float = Field(..., ge=0, le=100)
    feedback: str


class QualityScore(BaseModel):
    """Overall quality assessment of the problem definition."""
    overall_score: float = Field(..., ge=0, le=100)
    dimensions: List[QualityDimension]
    summary: str


class NormalizedProblemSummary(BaseModel):
    """Rewritten problem statement without solution bias."""
    summary: str
    key_elements: Dict[str, str] = Field(
        default_factory=dict,
        description="Extracted elements: who, what, why_hard",
    )


class UserPersona(BaseModel):
    """Primary or secondary user persona."""
    role: str
    context: str
    goal: str
    urgency: str
    failure_consequence: str


class UserPersonaClarification(BaseModel):
    """User persona analysis output."""
    primary_user: UserPersona
    secondary_users: List[UserPersona] = Field(default_factory=list)
    mvp_focus_rationale: str


class PainMoment(BaseModel):
    """Concrete pain moment (not abstract pain point)."""
    moment: str
    trigger: str
    current_behavior: str
    why_it_hurts: str


class RootCause(BaseModel):
    """Root cause analysis item."""
    cause: str
    category: str = Field(
        ...,
        description="Category: Knowledge / Process / Access / Psychology / Technical / Other",
    )
    explanation: str


class ExistingSolution(BaseModel):
    """Analysis of an existing solution."""
    name: str
    description: str
    strengths: List[str]
    weaknesses: List[str]
    gap: str = Field(..., description="Why it fails to fully solve the problem")


class ImpactStake(BaseModel):
    """Problem impact and stakes."""
    category: str = Field(
        ...,
        description="Category: Academic / Emotional / Time / Opportunity / Financial / Other",
    )
    description: str
    quantification: str = Field(
        ...,
        description="Quantified or bounded estimate of impact",
    )


class ScopeBoundary(BaseModel):
    """Problem scope exclusions."""
    exclusions: List[str]
    rationale: str


class ProblemDefinitionOutput(BaseModel):
    """Complete output from Phase 1: Problem Definition."""
    
    # 1. Quality Assessment
    quality_score: QualityScore
    
    # 2. Normalized Problem Summary
    normalized_summary: NormalizedProblemSummary
    
    # 3. User Persona Clarification
    personas: UserPersonaClarification
    
    # 4. Concrete Pain Moments
    pain_moments: List[PainMoment]
    
    # 5. Root Cause Analysis
    root_causes: List[RootCause]
    
    # 6. Existing Solutions & Why They Fail
    existing_solutions_analysis: List[ExistingSolution]
    
    # 7. Problem Impact & Stakes
    impact_stakes: List[ImpactStake]
    
    # 8. Problem Scope Boundary
    scope_boundary: ScopeBoundary
    
    # 9. Transition Questions to Phase 2
    transition_questions: List[str]


# ============================================================================
# Phase Implementation
# ============================================================================

@dataclass
class ProblemDefinitionPhase:
    """
    Phase 1: Problem Definition
    
    Processes structured user input and generates comprehensive problem analysis
    through multiple LLM calls, each with a dedicated prompt.
    """
    
    llm_client: Any  # LLMClient
    
    def run(self, input_data: ProblemDefinitionInput) -> ProblemDefinitionOutput:
        """Execute all analysis steps for Phase 1."""
        
        # Build context from input
        input_context = self._build_input_context(input_data)
        
        # Step 1: Quality Scoring
        quality_score = self._assess_quality(input_context)
        
        # Step 2: Normalize Problem Summary
        normalized_summary = self._normalize_problem(input_context)
        
        # Step 3: User Persona Clarification
        personas = self._clarify_personas(input_context)
        
        # Step 4: Pain Moments
        pain_moments = self._identify_pain_moments(input_context)
        
        # Step 5: Root Cause Analysis
        root_causes = self._analyze_root_causes(input_context)
        
        # Step 6: Existing Solutions Analysis
        existing_solutions = self._analyze_existing_solutions(input_context)
        
        # Step 7: Impact & Stakes
        impact_stakes = self._assess_impact(input_context)
        
        # Step 8: Scope Boundary
        scope_boundary = self._define_scope_boundary(input_context)
        
        # Step 9: Transition Questions
        transition_questions = self._generate_transition_questions(input_context)
        
        return ProblemDefinitionOutput(
            quality_score=quality_score,
            normalized_summary=normalized_summary,
            personas=personas,
            pain_moments=pain_moments,
            root_causes=root_causes,
            existing_solutions_analysis=existing_solutions,
            impact_stakes=impact_stakes,
            scope_boundary=scope_boundary,
            transition_questions=transition_questions,
        )
    
    def _build_input_context(self, input_data: ProblemDefinitionInput) -> str:
        """Build a context string from user input."""
        pain_points_str = "\n".join(f"- {p}" for p in input_data.pain_points) if input_data.pain_points else "None provided"
        
        context = f"""
Problem Description:
{input_data.problem_description}

Target Users:
{input_data.target_users}

Why It Matters:
{input_data.why_it_matters}

Pain Points:
{pain_points_str}

Has Existing Solutions: {input_data.has_existing_solutions}
"""
        if input_data.has_existing_solutions and input_data.current_solutions:
            context += f"\nCurrent Solutions Used:\n{input_data.current_solutions}"
        
        return context.strip()
    
    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call LLM and parse JSON response."""
        messages = [
            LLMPrompt(role="system", content=system_prompt),
            LLMPrompt(role="user", content=user_prompt),
        ]
        response = self.llm_client.generate(
            messages,
            extra={"response_format": {"type": "json_object"}},
        )
        content = str(response) if response else "{}"
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response", "raw": content}
    
    def _assess_quality(self, input_context: str) -> QualityScore:
        """Step 1: Assess overall quality of the problem definition."""
        result = self._call_llm_json(
            PHASE1_QUALITY_SCORING_PROMPT,
            input_context,
        )
        
        dimensions = [
            QualityDimension(
                name=d.get("name", "Unknown"),
                score=float(d.get("score", 0)),
                feedback=d.get("feedback", ""),
            )
            for d in result.get("dimensions", [])
        ]
        
        return QualityScore(
            overall_score=float(result.get("overall_score", 0)),
            dimensions=dimensions,
            summary=result.get("summary", "Quality assessment completed."),
        )
    
    def _normalize_problem(self, input_context: str) -> NormalizedProblemSummary:
        """Step 2: Rewrite problem statement neutrally."""
        result = self._call_llm_json(
            PHASE1_NORMALIZE_PROBLEM_PROMPT,
            input_context,
        )
        
        return NormalizedProblemSummary(
            summary=result.get("summary", ""),
            key_elements=result.get("key_elements", {}),
        )
    
    def _clarify_personas(self, input_context: str) -> UserPersonaClarification:
        """Step 3: Clarify user personas."""
        result = self._call_llm_json(
            PHASE1_USER_PERSONA_PROMPT,
            input_context,
        )
        
        primary = result.get("primary_user", {})
        primary_persona = UserPersona(
            role=primary.get("role", ""),
            context=primary.get("context", ""),
            goal=primary.get("goal", ""),
            urgency=primary.get("urgency", ""),
            failure_consequence=primary.get("failure_consequence", ""),
        )
        
        secondary_users = [
            UserPersona(
                role=s.get("role", ""),
                context=s.get("context", ""),
                goal=s.get("goal", ""),
                urgency=s.get("urgency", ""),
                failure_consequence=s.get("failure_consequence", ""),
            )
            for s in result.get("secondary_users", [])
        ]
        
        return UserPersonaClarification(
            primary_user=primary_persona,
            secondary_users=secondary_users,
            mvp_focus_rationale=result.get("mvp_focus_rationale", ""),
        )
    
    def _identify_pain_moments(self, input_context: str) -> List[PainMoment]:
        """Step 4: Identify concrete pain moments."""
        result = self._call_llm_json(
            PHASE1_PAIN_MOMENTS_PROMPT,
            input_context,
        )
        
        return [
            PainMoment(
                moment=m.get("moment", ""),
                trigger=m.get("trigger", ""),
                current_behavior=m.get("current_behavior", ""),
                why_it_hurts=m.get("why_it_hurts", ""),
            )
            for m in result.get("pain_moments", [])
        ]
    
    def _analyze_root_causes(self, input_context: str) -> List[RootCause]:
        """Step 5: Analyze root causes."""
        result = self._call_llm_json(
            PHASE1_ROOT_CAUSE_PROMPT,
            input_context,
        )
        
        return [
            RootCause(
                cause=r.get("cause", ""),
                category=r.get("category", "Other"),
                explanation=r.get("explanation", ""),
            )
            for r in result.get("root_causes", [])
        ]
    
    def _analyze_existing_solutions(self, input_context: str) -> List[ExistingSolution]:
        """Step 6: Analyze existing solutions."""
        result = self._call_llm_json(
            PHASE1_EXISTING_SOLUTIONS_PROMPT,
            input_context,
        )
        
        return [
            ExistingSolution(
                name=s.get("name", ""),
                description=s.get("description", ""),
                strengths=s.get("strengths", []),
                weaknesses=s.get("weaknesses", []),
                gap=s.get("gap", ""),
            )
            for s in result.get("existing_solutions", [])
        ]
    
    def _assess_impact(self, input_context: str) -> List[ImpactStake]:
        """Step 7: Assess problem impact and stakes."""
        result = self._call_llm_json(
            PHASE1_IMPACT_STAKES_PROMPT,
            input_context,
        )
        
        return [
            ImpactStake(
                category=i.get("category", "Other"),
                description=i.get("description", ""),
                quantification=i.get("quantification", ""),
            )
            for i in result.get("impact_stakes", [])
        ]
    
    def _define_scope_boundary(self, input_context: str) -> ScopeBoundary:
        """Step 8: Define scope boundary."""
        result = self._call_llm_json(
            PHASE1_SCOPE_BOUNDARY_PROMPT,
            input_context,
        )
        
        return ScopeBoundary(
            exclusions=result.get("exclusions", []),
            rationale=result.get("rationale", ""),
        )
    
    def _generate_transition_questions(self, input_context: str) -> List[str]:
        """Step 9: Generate transition questions for Phase 2."""
        result = self._call_llm_json(
            PHASE1_TRANSITION_QUESTIONS_PROMPT,
            input_context,
        )
        
        return result.get("transition_questions", [])
