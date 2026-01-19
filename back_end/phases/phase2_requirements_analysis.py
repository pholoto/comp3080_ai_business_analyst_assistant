"""
Phase 2: Requirement Analysis

This phase contains two sections:
1. Feature Analyzer - Analyzes and categorizes features
2. User Journey Generator - Creates step-by-step user journeys for selected features
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from AI.llm import LLMPrompt
from common.prompts import (PHASE2_FUNCTIONAL_REQUIREMENTS_PROMPT,
                            PHASE2_MVP_SCOPE_PROMPT,
                            PHASE2_NON_FUNCTIONAL_REQUIREMENTS_PROMPT,
                            PHASE2_NORMALIZE_FEATURES_PROMPT,
                            PHASE2_SCOPE_WARNINGS_PROMPT,
                            PHASE2_USER_JOURNEY_PROMPT)

# ============================================================================
# Helper Functions
# ============================================================================

# Type variable for enum parsing
E = TypeVar('E', bound=Enum)


def safe_enum_parse(enum_class: Type[E], value: str, default: E) -> E:
    """Parse enum value case-insensitively with fallback to default.
    
    Handles common LLM response variations:
    - "Core" -> "core"
    - "Must-Have" -> "must-have"
    - "AI/ML" -> "AI/ML" (exact match first)
    """
    if not value:
        return default
    
    # First try exact match
    for member in enum_class:
        if member.value == value:
            return member
    
    # Try case-insensitive match
    value_lower = value.lower().strip()
    for member in enum_class:
        if member.value.lower() == value_lower:
            return member
    
    # Try matching with normalized separators (spaces, underscores, hyphens)
    value_normalized = value_lower.replace(" ", "-").replace("_", "-")
    for member in enum_class:
        member_normalized = member.value.lower().replace(" ", "-").replace("_", "-")
        if member_normalized == value_normalized:
            return member
    
    # Return default if no match found
    return default


# ============================================================================
# Enums
# ============================================================================

class FeatureCategory(str, Enum):
    CORE = "core"
    ENHANCEMENT = "enhancement"
    NICE_TO_HAVE = "nice-to-have"
    FUTURE = "future"


class MoscowPriority(str, Enum):
    MUST_HAVE = "must-have"
    SHOULD_HAVE = "should-have"
    COULD_HAVE = "could-have"
    WONT_HAVE = "wont-have"


class RequirementCategory(str, Enum):
    AI_ML = "AI/ML"
    AUTHENTICATION = "Authentication/Security"
    DATABASE = "Database"
    USER_INTERFACE = "User Interface"
    PERFORMANCE = "Performance"
    INTEGRATION = "Integration"
    INFRASTRUCTURE = "Infrastructure"
    OTHER = "Other"


class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TeamSkillLevel(str, Enum):
    SOLO_DEVELOPER = "solo-developer"
    JUNIOR_TEAM = "junior-team"
    SENIOR_TEAM = "senior-team"
    MIXED_EXPERIENCE = "mixed-experience"


# ============================================================================
# Input Models
# ============================================================================

class FeatureAnalyzerInput(BaseModel):
    """Input schema for Feature Analyzer section."""
    
    # Required fields
    desired_features: List[str] = Field(
        ...,
        description="List of desired features",
        min_length=1,
    )
    mvp_goal: str = Field(
        ...,
        description="Very specific MVP goal using SMART model (e.g., 'Help students submit their polished personal statement essay within 48 hours')",
        min_length=10,
    )
    
    # From Phase 1 (auto-populated)
    primary_user_persona: Optional[str] = Field(
        default=None,
        description="Primary user persona generated from Phase 1",
    )
    
    # Optional constraints
    deadline: Optional[str] = Field(
        default=None,
        description="Project deadline (e.g., '3 months', 'December 2025')",
    )
    team_skill_level: Optional[TeamSkillLevel] = Field(
        default=None,
        description="Team skill level",
    )
    additional_constraints: Optional[str] = Field(
        default=None,
        description="Any additional constraints or limitations",
    )


class UserJourneyInput(BaseModel):
    """Input schema for User Journey Generator section."""
    
    selected_feature: str = Field(
        ...,
        description="One feature from the List of Desired Features to generate user journey for",
    )
    # Context from Feature Analyzer
    feature_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Context about the feature from Feature Analyzer output",
    )


# ============================================================================
# Output Models
# ============================================================================

class NormalizedFeature(BaseModel):
    """Normalized feature with category."""
    original_name: str
    normalized_name: str = Field(
        ...,
        description="Renamed in user-outcome language",
    )
    category: FeatureCategory
    description: str


class FunctionalRequirement(BaseModel):
    """Functional requirement specification."""
    id: str
    name: str
    description: str
    category: RequirementCategory
    moscow_priority: MoscowPriority
    complexity: Complexity
    rationale: str
    acceptance_criteria: List[str] = Field(default_factory=list)


class NonFunctionalRequirement(BaseModel):
    """Non-functional requirement specification."""
    id: str
    attribute: str = Field(
        ...,
        description="NFR attribute (e.g., Performance, Security, Scalability)",
    )
    requirement: str
    category: RequirementCategory
    moscow_priority: MoscowPriority
    complexity: Complexity
    metric: Optional[str] = Field(
        default=None,
        description="Measurable metric for this requirement",
    )
    rationale: str


class MvpFeature(BaseModel):
    """Feature included in MVP scope."""
    feature_name: str
    justification: str
    estimated_effort: str


class MvpScopeSummary(BaseModel):
    """MVP scope summary."""
    included_features: List[MvpFeature]
    excluded_features: List[str]
    mvp_rationale: str
    estimated_timeline: Optional[str] = None


class ScopeWarning(BaseModel):
    """Potential risk or concern about scope."""
    warning_type: str = Field(
        ...,
        description="Type: Technical Risk, Resource Risk, Timeline Risk, Dependency Risk, etc.",
    )
    description: str
    severity: str = Field(
        ...,
        description="Low, Medium, High, Critical",
    )
    mitigation: str


class FeatureAnalyzerOutput(BaseModel):
    """Complete output from Feature Analyzer section."""
    
    # 1. Normalized Feature List
    normalized_features: List[NormalizedFeature]
    
    # 2. Functional Requirements
    functional_requirements: List[FunctionalRequirement]
    
    # 3. Non-Functional Requirements
    non_functional_requirements: List[NonFunctionalRequirement]
    
    # 4. MVP Scope Summary
    mvp_scope: MvpScopeSummary
    
    # 5. Scope Warnings
    scope_warnings: List[ScopeWarning]


class UserJourneyStep(BaseModel):
    """Single step in user journey."""
    step_number: int
    title: str
    goal: str = Field(
        ...,
        description="What the user is trying to achieve in this step",
    )
    user_action: str = Field(
        ...,
        description="What the user does",
    )
    system_response: str = Field(
        ...,
        description="How the system responds",
    )
    success_criteria: str
    potential_issues: List[str] = Field(default_factory=list)


class UserJourneyOutput(BaseModel):
    """Output from User Journey Generator."""
    
    feature_name: str
    journey_title: str
    overview: str
    preconditions: List[str]
    steps: List[UserJourneyStep]
    postconditions: List[str]
    alternative_flows: List[str] = Field(default_factory=list)
    error_scenarios: List[str] = Field(default_factory=list)


# ============================================================================
# Phase Implementation
# ============================================================================

@dataclass
class RequirementsAnalysisPhase:
    """
    Phase 2: Requirements Analysis
    
    Contains two main sections:
    1. Feature Analyzer - Processes feature list and generates requirements
    2. User Journey Generator - Creates detailed user journeys for features
    """
    
    llm_client: Any  # LLMClient
    
    # Store Phase 1 output for context
    phase1_context: Optional[Dict[str, Any]] = None
    
    def analyze_features(self, input_data: FeatureAnalyzerInput) -> FeatureAnalyzerOutput:
        """Execute Feature Analyzer section."""
        
        # Build context from input
        input_context = self._build_feature_context(input_data)
        
        # Step 1: Normalize Features
        normalized_features = self._normalize_features(input_context, input_data.desired_features)
        
        # Step 2: Generate Functional Requirements
        functional_reqs = self._generate_functional_requirements(input_context)
        
        # Step 3: Generate Non-Functional Requirements
        non_functional_reqs = self._generate_non_functional_requirements(input_context)
        
        # Step 4: Define MVP Scope
        mvp_scope = self._define_mvp_scope(input_context, normalized_features)
        
        # Step 5: Identify Scope Warnings
        scope_warnings = self._identify_scope_warnings(input_context, functional_reqs, non_functional_reqs)
        
        return FeatureAnalyzerOutput(
            normalized_features=normalized_features,
            functional_requirements=functional_reqs,
            non_functional_requirements=non_functional_reqs,
            mvp_scope=mvp_scope,
            scope_warnings=scope_warnings,
        )
    
    def generate_user_journey(self, input_data: UserJourneyInput) -> UserJourneyOutput:
        """Execute User Journey Generator section."""
        
        input_context = self._build_journey_context(input_data)
        
        result = self._call_llm_json(
            PHASE2_USER_JOURNEY_PROMPT,
            input_context,
        )
        
        steps = [
            UserJourneyStep(
                step_number=s.get("step_number", i + 1),
                title=s.get("title", ""),
                goal=s.get("goal", ""),
                user_action=s.get("user_action", ""),
                system_response=s.get("system_response", ""),
                success_criteria=s.get("success_criteria", ""),
                potential_issues=s.get("potential_issues", []),
            )
            for i, s in enumerate(result.get("steps", []))
        ]
        
        return UserJourneyOutput(
            feature_name=input_data.selected_feature,
            journey_title=result.get("journey_title", f"User Journey: {input_data.selected_feature}"),
            overview=result.get("overview", ""),
            preconditions=result.get("preconditions", []),
            steps=steps,
            postconditions=result.get("postconditions", []),
            alternative_flows=result.get("alternative_flows", []),
            error_scenarios=result.get("error_scenarios", []),
        )
    
    def _build_feature_context(self, input_data: FeatureAnalyzerInput) -> str:
        """Build context string for feature analysis."""
        features_str = "\n".join(f"- {f}" for f in input_data.desired_features)
        
        context = f"""
Desired Features:
{features_str}

MVP Goal (SMART):
{input_data.mvp_goal}

Primary User Persona:
{input_data.primary_user_persona or 'Not specified'}
"""
        
        if input_data.deadline:
            context += f"\nDeadline: {input_data.deadline}"
        if input_data.team_skill_level:
            context += f"\nTeam Skill Level: {input_data.team_skill_level.value}"
        if input_data.additional_constraints:
            context += f"\nAdditional Constraints:\n{input_data.additional_constraints}"
        
        # Add Phase 1 context if available
        if self.phase1_context:
            context += f"\n\nPhase 1 Problem Context:\n{json.dumps(self.phase1_context, indent=2)}"
        
        return context.strip()
    
    def _build_journey_context(self, input_data: UserJourneyInput) -> str:
        """Build context string for user journey generation."""
        context = f"""
Selected Feature:
{input_data.selected_feature}
"""
        if input_data.feature_context:
            context += f"\nFeature Context:\n{json.dumps(input_data.feature_context, indent=2)}"
        
        if self.phase1_context:
            context += f"\n\nPrimary User Persona:\n{json.dumps(self.phase1_context.get('personas', {}), indent=2)}"
        
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
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response", "raw": content}
    
    def _normalize_features(self, input_context: str, features: List[str]) -> List[NormalizedFeature]:
        """Step 1: Normalize and categorize features."""
        result = self._call_llm_json(
            PHASE2_NORMALIZE_FEATURES_PROMPT,
            input_context,
        )
        
        return [
            NormalizedFeature(
                original_name=f.get("original_name", ""),
                normalized_name=f.get("normalized_name", ""),
                category=safe_enum_parse(FeatureCategory, f.get("category", ""), FeatureCategory.NICE_TO_HAVE),
                description=f.get("description", ""),
            )
            for f in result.get("normalized_features", [])
        ]
    
    def _generate_functional_requirements(self, input_context: str) -> List[FunctionalRequirement]:
        """Step 2: Generate functional requirements."""
        result = self._call_llm_json(
            PHASE2_FUNCTIONAL_REQUIREMENTS_PROMPT,
            input_context,
        )
        
        return [
            FunctionalRequirement(
                id=r.get("id", f"FR-{i+1}"),
                name=r.get("name", ""),
                description=r.get("description", ""),
                category=safe_enum_parse(RequirementCategory, r.get("category", ""), RequirementCategory.OTHER),
                moscow_priority=safe_enum_parse(MoscowPriority, r.get("moscow_priority", ""), MoscowPriority.COULD_HAVE),
                complexity=safe_enum_parse(Complexity, r.get("complexity", ""), Complexity.MEDIUM),
                rationale=r.get("rationale", ""),
                acceptance_criteria=r.get("acceptance_criteria", []),
            )
            for i, r in enumerate(result.get("functional_requirements", []))
        ]
    
    def _generate_non_functional_requirements(self, input_context: str) -> List[NonFunctionalRequirement]:
        """Step 3: Generate non-functional requirements."""
        result = self._call_llm_json(
            PHASE2_NON_FUNCTIONAL_REQUIREMENTS_PROMPT,
            input_context,
        )
        
        return [
            NonFunctionalRequirement(
                id=r.get("id", f"NFR-{i+1}"),
                attribute=r.get("attribute", ""),
                requirement=r.get("requirement", ""),
                category=safe_enum_parse(RequirementCategory, r.get("category", ""), RequirementCategory.OTHER),
                moscow_priority=safe_enum_parse(MoscowPriority, r.get("moscow_priority", ""), MoscowPriority.COULD_HAVE),
                complexity=safe_enum_parse(Complexity, r.get("complexity", ""), Complexity.MEDIUM),
                metric=r.get("metric"),
                rationale=r.get("rationale", ""),
            )
            for i, r in enumerate(result.get("non_functional_requirements", []))
        ]
    
    def _define_mvp_scope(
        self,
        input_context: str,
        normalized_features: List[NormalizedFeature],
    ) -> MvpScopeSummary:
        """Step 4: Define MVP scope."""
        features_json = json.dumps([f.model_dump() for f in normalized_features], indent=2)
        context_with_features = f"{input_context}\n\nNormalized Features:\n{features_json}"
        
        result = self._call_llm_json(
            PHASE2_MVP_SCOPE_PROMPT,
            context_with_features,
        )
        
        included = [
            MvpFeature(
                feature_name=f.get("feature_name", ""),
                justification=f.get("justification", ""),
                estimated_effort=f.get("estimated_effort", ""),
            )
            for f in result.get("included_features", [])
        ]
        
        return MvpScopeSummary(
            included_features=included,
            excluded_features=result.get("excluded_features", []),
            mvp_rationale=result.get("mvp_rationale", ""),
            estimated_timeline=result.get("estimated_timeline"),
        )
    
    def _identify_scope_warnings(
        self,
        input_context: str,
        functional_reqs: List[FunctionalRequirement],
        non_functional_reqs: List[NonFunctionalRequirement],
    ) -> List[ScopeWarning]:
        """Step 5: Identify potential scope warnings."""
        reqs_json = json.dumps({
            "functional": [r.model_dump() for r in functional_reqs],
            "non_functional": [r.model_dump() for r in non_functional_reqs],
        }, indent=2)
        context_with_reqs = f"{input_context}\n\nRequirements:\n{reqs_json}"
        
        result = self._call_llm_json(
            PHASE2_SCOPE_WARNINGS_PROMPT,
            context_with_reqs,
        )
        
        return [
            ScopeWarning(
                warning_type=w.get("warning_type", "Other"),
                description=w.get("description", ""),
                severity=w.get("severity", "Medium"),
                mitigation=w.get("mitigation", ""),
            )
            for w in result.get("scope_warnings", [])
        ]
