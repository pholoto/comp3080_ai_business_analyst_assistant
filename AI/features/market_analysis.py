"""Market Analysis feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from common.prompts import (MARKET_ANALYSIS_SYSTEM_PROMPT,
                            build_market_analysis_user_prompt)

from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class MarketAnalysisFeature:
    """Evaluate market fit, competition, and impact."""

    context: FeatureContext

    name: str = "market_analysis"
    description: str = "Understand market fit, competitors, and potential impact."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        overview = ctx.session.get_state("project_overview", "Project overview not captured yet.")
        differentiators = ctx.session.get_state(
            "differentiators", ctx.session.get_state("uvp", "Differentiators not defined yet.")
        )
        prompt = build_market_analysis_user_prompt(
            project_overview=overview,
            differentiators=differentiators,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=MARKET_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=prompt,
            default_title="Market Analysis",
            history=history,
        )
        summary = data.get("summary") or "Market analysis drafted."
        for key, state_key in (
            ("competitive_landscape", "competitive_landscape"),
            ("unique_value_proposition", "uvp"),
            ("target_segments", "target_segments"),
            ("go_to_market_ideas", "go_to_market_ideas"),
            ("impact_considerations", "impact_considerations"),
        ):
            if data.get(key):
                ctx.session.set_state(state_key, data[key])
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Market Analysis"),
            summary=summary,
            data=data,
        )
