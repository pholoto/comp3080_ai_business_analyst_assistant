"""Stakeholder Insights feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..prompts import (STAKEHOLDER_INSIGHTS_SYSTEM_PROMPT,
                       build_stakeholder_insights_user_prompt)
from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class StakeholderInsightsFeature:
    """Help students map and engage with stakeholders."""

    context: FeatureContext

    name: str = "stakeholder_insights"
    description: str = "Identify stakeholders, their needs, and engagement tactics."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        project_overview = ctx.session.get_state("project_overview", "N/A")
        stakeholder_map = ctx.session.get_state("stakeholder_map", "None yet")
        template = STAKEHOLDER_INSIGHTS_SYSTEM_PROMPT
        prompt = build_stakeholder_insights_user_prompt(
            project_overview=project_overview,
            stakeholder_map=stakeholder_map,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=template,
            user_prompt=prompt,
            default_title="Stakeholder Insights",
            history=history,
        )
        stakeholder_map = data.get("stakeholder_map")
        if stakeholder_map:
            ctx.session.set_state("stakeholder_map", stakeholder_map)
        ctx.session.set_state("engagement_plan", data.get("engagement_plan"))
        summary = data.get("summary") or "Stakeholder insights updated."
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Stakeholder Insights"),
            summary=summary,
            data=data,
        )
