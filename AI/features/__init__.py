"""Feature registry for the AIBA assistant."""

from .base import Feature, FeatureContext, FeatureRegistry, FeatureResult
from .documentation import DocumentationFeature
from .market_analysis import MarketAnalysisFeature
from .problem_definition import ProblemDefinitionFeature
from .prototype_development import PrototypeDevelopmentFeature
from .requirements_analysis import RequirementsAnalysisFeature
from .solution_design import SolutionDesignFeature
from .testing_validation import TestingValidationFeature

__all__ = [
    "Feature",
    "FeatureContext",
    "FeatureRegistry",
    "FeatureResult",
    "ProblemDefinitionFeature",
    "RequirementsAnalysisFeature",
    "SolutionDesignFeature",
    "PrototypeDevelopmentFeature",
    "TestingValidationFeature",
    "DocumentationFeature",
    "MarketAnalysisFeature",
    "build_default_registry",
]


def build_default_registry() -> FeatureRegistry:
    """Wire up the default set of AI BA features."""

    def _factory(feature_cls):
        def _create(context: FeatureContext):
            return feature_cls(context)

        return _create

    registry = FeatureRegistry()
    registry.register("problem_definition", _factory(ProblemDefinitionFeature))
    registry.register("requirements_analysis", _factory(RequirementsAnalysisFeature))
    registry.register("solution_design", _factory(SolutionDesignFeature))
    registry.register("prototype_development", _factory(PrototypeDevelopmentFeature))
    registry.register("testing_validation", _factory(TestingValidationFeature))
    registry.register("documentation", _factory(DocumentationFeature))
    registry.register("market_analysis", _factory(MarketAnalysisFeature))
    return registry
