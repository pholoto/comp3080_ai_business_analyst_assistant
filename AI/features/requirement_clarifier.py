"""Requirement Clarifier feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from common.prompts import (REQUIREMENT_CLARIFIER_SYSTEM_PROMPT,
                            build_requirement_clarifier_user_prompt)

from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class RequirementClarifierFeature:
    """Guide teams to refine their problem statements and requirements."""

    context: FeatureContext

    name: str = "requirement_clarifier"
    description: str = "Clarify problem statements and capture actionable requirements."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        template = REQUIREMENT_CLARIFIER_SYSTEM_PROMPT
        prompt = build_requirement_clarifier_user_prompt(
            project_overview=ctx.session.get_state("project_overview", "N/A"),
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=template,
            user_prompt=prompt,
            default_title="Requirement Clarification",
            history=history,
        )
        summary = data.get("summary") or user_input
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        backlog = data.get("requirement_backlog")
        if backlog:
            ctx.session.set_state("requirements", backlog)
        ctx.session.set_state("project_overview", summary)
        return FeatureResult(
            title=data.get("title", "Requirement Clarification"),
            summary=summary,
            data=data,
        )
