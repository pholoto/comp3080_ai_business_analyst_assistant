"""Market Fit Analyzer feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class MarketFitAnalyzerFeature:
    """Compare the concept with the market landscape and stakeholders."""

    context: FeatureContext

    name: str = "market_fit_analyzer"
    description: str = "Analyse competitors, value proposition, and differentiation."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        template = (
            "You are an AI strategist. Produce a JSON object with keys: title, summary, "
            "competitive_landscape (list of objects with name, positioning, strengths, gaps), "
            "unique_value_proposition (string), target_segments (list of objects with segment, needs, "
            "fit_score), and go_to_market_ideas (list of strings). Reference prior artefacts to maintain alignment."
        )
        prompt = (
            "Project overview: "
            f"{ctx.session.get_state('project_overview', 'Unknown project')}\n"
            "Prioritised features: "
            f"{ctx.session.get_state('prioritised_features', 'Not available')}\n"
            "Supporting attachments:\n"
            f"{build_attachment_context(ctx.session)}\n"
            "Additional research prompt: "
            f"{user_input}"
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=template,
            user_prompt=prompt,
            default_title="Market Fit Analysis",
            history=history,
        )
        if data.get("competitive_landscape"):
            ctx.session.set_state("competitive_landscape", data["competitive_landscape"])
        if data.get("unique_value_proposition"):
            ctx.session.set_state("uvp", data["unique_value_proposition"])
        summary = data.get("summary") or "Market analysis drafted."
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Market Fit Analysis"),
            summary=summary,
            data=data,
        )
