"""Phase-based analysis modules for the AI Business Analyst Assistant."""
from __future__ import annotations

from .phase1_problem_definition import (ProblemDefinitionInput,
                                        ProblemDefinitionOutput,
                                        ProblemDefinitionPhase)
from .phase2_requirements_analysis import (FeatureAnalyzerInput,
                                           FeatureAnalyzerOutput,
                                           RequirementsAnalysisPhase,
                                           UserJourneyInput, UserJourneyOutput)
from .phase3_market_analysis import (MarketAnalysisInput, MarketAnalysisOutput,
                                     MarketAnalysisPhase,
                                     CompetitorResearchInput, CompetitorResearchOutput)
from .phase7_documentation import (DocumentationInput, DocumentationOutput,
                                   DocumentationPhase, DocumentationType)

__all__ = [
    # Phase 1
    "ProblemDefinitionInput",
    "ProblemDefinitionOutput",
    "ProblemDefinitionPhase",
    # Phase 2
    "FeatureAnalyzerInput",
    "FeatureAnalyzerOutput",
    "UserJourneyInput",
    "UserJourneyOutput",
    "RequirementsAnalysisPhase",
    # Phase 3
    "MarketAnalysisInput",
    "MarketAnalysisOutput",
    "MarketAnalysisPhase",
    "CompetitorResearchInput",
    "CompetitorResearchOutput",
    # Phase 7
    "DocumentationInput",
    "DocumentationOutput",
    "DocumentationType",
    "DocumentationPhase",
]
