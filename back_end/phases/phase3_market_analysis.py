"""
Phase 3: Market Analysis

This phase provides market research, competitive analysis, and strategic positioning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, field_validator

from AI.llm import LLMPrompt
from common.prompts import (PHASE3_COMPETITOR_ANALYSIS_PROMPT,
                            PHASE3_MARKET_RESEARCH_PROMPT,
                            PHASE3_PORTERS_FIVE_FORCES_PROMPT,
                            PHASE3_COMPETITOR_RESEARCH_PROMPT,
                            PHASE3_USP_GENERATION_PROMPT)

# ============================================================================
# Helper Functions
# ============================================================================

# Type variable for enum parsing
E = TypeVar('E', bound=Enum)


def safe_enum_parse(enum_class: Type[E], value: str, default: E) -> E:
    """Parse enum value case-insensitively with fallback to default."""
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
    
    # Try matching with normalized separators
    value_normalized = value_lower.replace(" ", "-").replace("_", "-")
    for member in enum_class:
        member_normalized = member.value.lower().replace(" ", "-").replace("_", "-")
        if member_normalized == value_normalized:
            return member
    
    return default

# ============================================================================
# Enums
# ============================================================================

class GeographicScope(str, Enum):
    LOCAL = "local"
    REGIONAL = "regional"
    MULTI_REGIONAL = "multi-regional"
    NATIONAL = "national"
    INTERNATIONAL = "international"
    GLOBAL = "global"


class ForceStrength(str, Enum):
    VERY_LOW = "very-low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very-high"


# ============================================================================
# Input Models
# ============================================================================

class MarketAnalysisInput(BaseModel):
    """Input schema for Phase 3: Market Analysis."""
    
    # All optional
    geographic_scope: Optional[GeographicScope] = Field(
        default=None,
        description="Geographic scope of the market analysis",
    )
    industry_context: Optional[str] = Field(
        default=None,
        description="Industry context (e.g., 'EdTech', 'FinTech', 'Healthcare')",
    )
    competitors: Optional[List[str]] = Field(
        default=None,
        description="Known competitors to analyze",
    )
    
    # Context from previous phases (auto-populated)
    problem_summary: Optional[str] = Field(
        default=None,
        description="Normalized problem summary from Phase 1",
    )
    target_users: Optional[str] = Field(
        default=None,
        description="Target users from Phase 1",
    )
    mvp_goal: Optional[str] = Field(
        default=None,
        description="MVP goal from Phase 2",
    )


class CompetitorResearchInput(BaseModel):
    """Input for quick competitor research."""
    industry: str
    geographic_scope: Optional[str] = "Global"
    known_competitors: Optional[str] = ""


# ============================================================================
# Output Models
# ============================================================================

class MarketStatistic(BaseModel):
    """A market statistic with source."""
    metric: str
    value: str
    source: str = Field(..., description="Source URL or reference")
    year: Optional[str] = None
    
    @field_validator('year', mode='before')
    @classmethod
    def coerce_year_to_string(cls, v):
        """Convert year to string if it's an integer."""
        if v is None:
            return None
        return str(v)


class ResearchedCompetitor(BaseModel):
    """Simplified competitor profile for quick research results."""
    name: str
    description: str
    business_model: str
    target_customers: str


class CompetitorResearchOutput(BaseModel):
    """Output for competitor research."""
    competitors: List[ResearchedCompetitor]


class MarketResearchSummary(BaseModel):
    """Market research summary with statistics and links."""
    overview: str
    market_size: Optional[MarketStatistic] = None
    growth_rate: Optional[MarketStatistic] = None
    key_statistics: List[MarketStatistic]
    key_trends: List[str]
    market_drivers: List[str]
    market_challenges: List[str]
    sources: List[str] = Field(
        ...,
        description="List of URLs to real websites with market data",
    )


class PorterForce(BaseModel):
    """Single Porter's Five Forces analysis."""
    force: str
    strength: ForceStrength
    analysis: str
    key_factors: List[str]


class PortersFiveForcesAnalysis(BaseModel):
    """Complete Porter's Five Forces analysis."""
    supplier_power: PorterForce
    buyer_power: PorterForce
    competitive_rivalry: PorterForce
    threat_of_substitution: PorterForce
    threat_of_new_entry: PorterForce
    overall_assessment: str
    strategic_implications: List[str]


class CompetitorProfile(BaseModel):
    """Detailed competitor analysis."""
    name: str
    description: str
    business_model: str
    target_customer: str
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    pricing_model: Optional[str] = None
    market_share: Optional[str] = None
    key_differentiators: List[str] = Field(default_factory=list)


class CompetitorAnalysis(BaseModel):
    """Complete competitor analysis output."""
    competitors: List[CompetitorProfile]
    market_gaps: List[str]
    competitive_landscape_summary: str


class UniqueSellingPoint(BaseModel):
    """A unique selling point."""
    usp: str
    target_audience: str
    supporting_evidence: str
    differentiation_level: str = Field(
        ...,
        description="Low, Medium, High, or Very High",
    )


class UspGeneration(BaseModel):
    """Generated unique selling points."""
    primary_usp: UniqueSellingPoint
    secondary_usps: List[UniqueSellingPoint]
    positioning_statement: str
    value_proposition_canvas: Dict[str, Any] = Field(
        default_factory=dict,
        description="Customer jobs, pains, gains mapped to product features",
    )


class MarketAnalysisOutput(BaseModel):
    """Complete output from Phase 3: Market Analysis."""
    
    # 1. Market Research Summary
    market_research: MarketResearchSummary
    
    # 2. Porter's Five Forces Analysis
    porters_analysis: PortersFiveForcesAnalysis
    
    # 3. Competitor Analysis
    competitor_analysis: CompetitorAnalysis
    
    # 4. Generated Unique Selling Points
    usp_generation: UspGeneration


# ============================================================================
# Phase Implementation
# ============================================================================

@dataclass
class MarketAnalysisPhase:
    """
    Phase 3: Market Analysis
    
    Provides comprehensive market research, competitive analysis,
    and strategic positioning insights.
    """
    
    llm_client: Any  # LLMClient
    
    # Store previous phase outputs for context
    phase1_context: Optional[Dict[str, Any]] = None
    phase2_context: Optional[Dict[str, Any]] = None
    
    def run(self, input_data: MarketAnalysisInput) -> MarketAnalysisOutput:
        """Execute all analysis steps for Phase 3."""
        
        # Build context from input and previous phases
        input_context = self._build_input_context(input_data)
        
        # Step 1: Market Research Summary
        market_research = self._research_market(input_context)
        
        # Step 2: Porter's Five Forces Analysis
        porters_analysis = self._analyze_porters_forces(input_context)
        
        # Step 3: Competitor Analysis
        competitor_analysis = self._analyze_competitors(input_context, input_data.competitors)
        
        # Step 4: Generate USPs
        usp_generation = self._generate_usps(input_context, market_research, competitor_analysis)
        
        return MarketAnalysisOutput(
            market_research=market_research,
            porters_analysis=porters_analysis,
            competitor_analysis=competitor_analysis,
            usp_generation=usp_generation,
        )
    
    def _build_input_context(self, input_data: MarketAnalysisInput) -> str:
        """Build context string from input and previous phases."""
        context_parts = []
        
        if input_data.industry_context:
            context_parts.append(f"Industry: {input_data.industry_context}")
        
        if input_data.geographic_scope:
            context_parts.append(f"Geographic Scope: {input_data.geographic_scope.value}")
        
        if input_data.problem_summary:
            context_parts.append(f"Problem Summary:\n{input_data.problem_summary}")
        
        if input_data.target_users:
            context_parts.append(f"Target Users:\n{input_data.target_users}")
        
        if input_data.mvp_goal:
            context_parts.append(f"MVP Goal:\n{input_data.mvp_goal}")
        
        if input_data.competitors:
            context_parts.append(f"Known Competitors:\n" + "\n".join(f"- {c}" for c in input_data.competitors))
        
        # Add Phase 1 context
        if self.phase1_context:
            context_parts.append(f"Phase 1 Context:\n{json.dumps(self.phase1_context, indent=2)}")
        
        # Add Phase 2 context
        if self.phase2_context:
            context_parts.append(f"Phase 2 Context:\n{json.dumps(self.phase2_context, indent=2)}")
        
        return "\n\n".join(context_parts) if context_parts else "No specific context provided."
    
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
    
    def _research_market(self, input_context: str) -> MarketResearchSummary:
        """Step 1: Generate market research summary."""
        result = self._call_llm_json(
            PHASE3_MARKET_RESEARCH_PROMPT,
            input_context,
        )
        
        market_size = None
        if result.get("market_size"):
            ms = result["market_size"]
            market_size = MarketStatistic(
                metric=ms.get("metric", "Market Size"),
                value=ms.get("value", ""),
                source=ms.get("source", ""),
                year=ms.get("year"),
            )
        
        growth_rate = None
        if result.get("growth_rate"):
            gr = result["growth_rate"]
            growth_rate = MarketStatistic(
                metric=gr.get("metric", "Growth Rate"),
                value=gr.get("value", ""),
                source=gr.get("source", ""),
                year=gr.get("year"),
            )
        
        key_statistics = [
            MarketStatistic(
                metric=s.get("metric", ""),
                value=s.get("value", ""),
                source=s.get("source", ""),
                year=s.get("year"),
            )
            for s in result.get("key_statistics", [])
        ]
        
        return MarketResearchSummary(
            overview=result.get("overview", ""),
            market_size=market_size,
            growth_rate=growth_rate,
            key_statistics=key_statistics,
            key_trends=result.get("key_trends", []),
            market_drivers=result.get("market_drivers", []),
            market_challenges=result.get("market_challenges", []),
            sources=result.get("sources", []),
        )
    
    def _analyze_porters_forces(self, input_context: str) -> PortersFiveForcesAnalysis:
        """Step 2: Analyze Porter's Five Forces."""
        result = self._call_llm_json(
            PHASE3_PORTERS_FIVE_FORCES_PROMPT,
            input_context,
        )
        
        def parse_force(data: Dict[str, Any], force_name: str) -> PorterForce:
            return PorterForce(
                force=force_name,
                strength=safe_enum_parse(ForceStrength, data.get("strength", ""), ForceStrength.MODERATE),
                analysis=data.get("analysis", ""),
                key_factors=data.get("key_factors", []),
            )
        
        return PortersFiveForcesAnalysis(
            supplier_power=parse_force(result.get("supplier_power", {}), "Supplier Power"),
            buyer_power=parse_force(result.get("buyer_power", {}), "Buyer Power"),
            competitive_rivalry=parse_force(result.get("competitive_rivalry", {}), "Competitive Rivalry"),
            threat_of_substitution=parse_force(result.get("threat_of_substitution", {}), "Threat of Substitution"),
            threat_of_new_entry=parse_force(result.get("threat_of_new_entry", {}), "Threat of New Entry"),
            overall_assessment=result.get("overall_assessment", ""),
            strategic_implications=result.get("strategic_implications", []),
        )
    
    def _analyze_competitors(
        self,
        input_context: str,
        known_competitors: Optional[List[str]],
    ) -> CompetitorAnalysis:
        """Step 3: Analyze competitors."""
        context = input_context
        if known_competitors:
            context += f"\n\nSpecific competitors to analyze:\n" + "\n".join(f"- {c}" for c in known_competitors)
        
        result = self._call_llm_json(
            PHASE3_COMPETITOR_ANALYSIS_PROMPT,
            context,
        )
        
        competitors = [
            CompetitorProfile(
                name=c.get("name", ""),
                description=c.get("description", ""),
                business_model=c.get("business_model", ""),
                target_customer=c.get("target_customer", ""),
                strengths=c.get("strengths", []),
                weaknesses=c.get("weaknesses", []),
                opportunities=c.get("opportunities", []),
                threats=c.get("threats", []),
                pricing_model=c.get("pricing_model"),
                market_share=c.get("market_share"),
                key_differentiators=c.get("key_differentiators", []),
            )
            for c in result.get("competitors", [])
        ]
        
        return CompetitorAnalysis(
            competitors=competitors,
            market_gaps=result.get("market_gaps", []),
            competitive_landscape_summary=result.get("competitive_landscape_summary", ""),
        )
    
    def _generate_usps(
        self,
        input_context: str,
        market_research: MarketResearchSummary,
        competitor_analysis: CompetitorAnalysis,
    ) -> UspGeneration:
        """Step 4: Generate unique selling points."""
        context = f"""{input_context}

Market Research Summary:
{market_research.overview}

Key Trends:
{json.dumps(market_research.key_trends, indent=2)}

Competitor Market Gaps:
{json.dumps(competitor_analysis.market_gaps, indent=2)}

Competitive Landscape:
{competitor_analysis.competitive_landscape_summary}
"""
        
        result = self._call_llm_json(
            PHASE3_USP_GENERATION_PROMPT,
            context,
        )
        
        primary = result.get("primary_usp", {})
        primary_usp = UniqueSellingPoint(
            usp=primary.get("usp", ""),
            target_audience=primary.get("target_audience", ""),
            supporting_evidence=primary.get("supporting_evidence", ""),
            differentiation_level=primary.get("differentiation_level", "Medium"),
        )
        
        secondary_usps = [
            UniqueSellingPoint(
                usp=u.get("usp", ""),
                target_audience=u.get("target_audience", ""),
                supporting_evidence=u.get("supporting_evidence", ""),
                differentiation_level=u.get("differentiation_level", "Medium"),
            )
            for u in result.get("secondary_usps", [])
        ]
        
        return UspGeneration(
            primary_usp=primary_usp,
            secondary_usps=secondary_usps,
            positioning_statement=result.get("positioning_statement", ""),
            value_proposition_canvas=result.get("value_proposition_canvas", {}),
        )

    def research_competitors(self, input_data: CompetitorResearchInput) -> CompetitorResearchOutput:
        """Quick research of competitors based on industry and known items."""
        context = f"Industry: {input_data.industry}\n"
        context += f"Geographic Scope: {input_data.geographic_scope}\n"
        if input_data.known_competitors:
            context += f"Known Competitors: {input_data.known_competitors}\n"
            
        result = self._call_llm_json(
            PHASE3_COMPETITOR_RESEARCH_PROMPT,
            context
        )
        
        competitors = [
            ResearchedCompetitor(
                name=c.get("name", "Unknown"),
                description=c.get("description", ""),
                business_model=c.get("business_model", "N/A"),
                target_customers=c.get("target_customers", "N/A")
            )
            for c in result.get("competitors", [])
        ]
        
        return CompetitorResearchOutput(competitors=competitors)
