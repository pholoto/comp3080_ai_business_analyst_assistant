"""Requirement Clarifier feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
        template = (
            "You are an AI Business Analyst helping student teams refine their project idea. "
            "Generate a JSON object with keys: title, summary, clarifying_questions (list of "
            "strings), assumptions (list of strings), and requirement_backlog (list of objects "
            "with fields id, requirement, rationale). Use the project context provided."
        )
        prompt = (
            "Current project context and notes:\n"
            f"{ctx.session.get_state('project_overview', 'N/A')}\n\n"
            "Supporting documents summary:\n"
            f"{build_attachment_context(ctx.session)}\n\n"
            "New user input: "
            f"{user_input}\n\n"
            "If previous decisions exist, ensure you respect them."
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
