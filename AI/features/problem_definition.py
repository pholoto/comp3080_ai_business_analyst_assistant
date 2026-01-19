"""Problem Definition feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from common.prompts import (PROBLEM_DEFINITION_SYSTEM_PROMPT,
                            build_problem_definition_user_prompt)

from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class ProblemDefinitionFeature:
    """Guide teams to articulate a sharp problem statement."""

    context: FeatureContext

    name: str = "problem_definition"
    description: str = "Articulate and refine the project problem statement with precision."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        project_overview = ctx.session.get_state(
            "project_overview", "Project overview not captured yet."
        )
        constraints = ctx.session.get_state("constraints", "Constraints not documented yet.")
        prompt = build_problem_definition_user_prompt(
            project_overview=project_overview,
            constraints=constraints,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=PROBLEM_DEFINITION_SYSTEM_PROMPT,
            user_prompt=prompt,
            default_title="Problem Definition",
            history=history,
        )
        summary = data.get("summary") or "Problem statement refined."
        refined = data.get("refined_problem_statement")
        if refined:
            ctx.session.set_state("problem_statement", refined)
            ctx.session.set_state("project_overview", refined)
        for key in ("pain_points", "success_metrics", "clarifying_questions", "assumptions"):
            if data.get(key):
                ctx.session.set_state(key, data[key])
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Problem Definition"),
            summary=summary,
            data=data,
        )
