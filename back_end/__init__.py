"""Backend package for the AI Business Analyst Assistant."""

from .phases import (DocumentationInput,  # Phase 1; Phase 2; Phase 3; Phase 7
                     DocumentationOutput, DocumentationPhase,
                     DocumentationType, FeatureAnalyzerInput,
                     FeatureAnalyzerOutput, MarketAnalysisInput,
                     MarketAnalysisOutput, MarketAnalysisPhase,
                     ProblemDefinitionInput, ProblemDefinitionOutput,
                     ProblemDefinitionPhase, RequirementsAnalysisPhase,
                     UserJourneyInput, UserJourneyOutput)

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
    # Phase 7
    "DocumentationInput",
    "DocumentationOutput",
    "DocumentationType",
    "DocumentationPhase",
]
