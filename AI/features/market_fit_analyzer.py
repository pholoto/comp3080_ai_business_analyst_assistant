"""Market Fit Analyzer feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from common.prompts import (MARKET_FIT_SYSTEM_PROMPT,
                            build_market_fit_user_prompt)

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
        project_overview = ctx.session.get_state("project_overview", "Unknown project")
        prioritised_features = ctx.session.get_state("prioritised_features", "Not available")
        template = MARKET_FIT_SYSTEM_PROMPT
        prompt = build_market_fit_user_prompt(
            project_overview=project_overview,
            prioritised_features=prioritised_features,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
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
