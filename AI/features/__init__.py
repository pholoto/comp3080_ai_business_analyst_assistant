"""Feature registry for the AIBA assistant."""

from .base import Feature, FeatureContext, FeatureRegistry, FeatureResult
from .feature_prioritization import FeaturePrioritizationFeature
from .market_fit_analyzer import MarketFitAnalyzerFeature
from .report_exporter import BAReportExporterFeature
from .requirement_clarifier import RequirementClarifierFeature
from .stakeholder_insights import StakeholderInsightsFeature
from .use_case_generator import UseCaseGeneratorFeature

__all__ = [
    "Feature",
    "FeatureContext",
    "FeatureRegistry",
    "FeatureResult",
    "RequirementClarifierFeature",
    "UseCaseGeneratorFeature",
    "FeaturePrioritizationFeature",
    "MarketFitAnalyzerFeature",
    "StakeholderInsightsFeature",
    "BAReportExporterFeature",
    "build_default_registry",
]


def build_default_registry() -> FeatureRegistry:
    """Wire up the default set of AI BA features."""

    def _factory(feature_cls):
        def _create(context: FeatureContext):
            return feature_cls(context)

        return _create

    registry = FeatureRegistry()
    registry.register("requirement_clarifier", _factory(RequirementClarifierFeature))
    registry.register("use_case_generator", _factory(UseCaseGeneratorFeature))
    registry.register("feature_prioritization", _factory(FeaturePrioritizationFeature))
    registry.register("market_fit_analyzer", _factory(MarketFitAnalyzerFeature))
    registry.register("stakeholder_insights", _factory(StakeholderInsightsFeature))
    registry.register("ba_report_export", _factory(BAReportExporterFeature))
    return registry
