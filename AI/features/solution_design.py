"""Solution Design feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from common.prompts import (SOLUTION_DESIGN_SYSTEM_PROMPT,
                            build_solution_design_user_prompt)

from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class SolutionDesignFeature:
    """Generate architecture options aligned to the backlog."""

    context: FeatureContext

    name: str = "solution_design"
    description: str = "Generate architecture diagrams and design patterns tailored to the project."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        requirements = ctx.session.get_state("requirements") or "Requirements not captured."
        constraints = ctx.session.get_state("constraints", "No constraints captured yet.")
        prompt = build_solution_design_user_prompt(
            requirements_digest=requirements,
            constraints=constraints,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=SOLUTION_DESIGN_SYSTEM_PROMPT,
            user_prompt=prompt,
            default_title="Solution Design",
            history=history,
        )
        summary = data.get("summary") or "Solution design drafted."
        if data.get("architecture_overview"):
            ctx.session.set_state("solution_blueprint", data["architecture_overview"])
        for key, state_key in (
            ("component_breakdown", "architecture_components"),
            ("design_patterns", "design_patterns"),
            ("technology_choices", "technology_choices"),
            ("diagrams", "diagram_ideas"),
            ("open_questions", "solution_open_questions"),
        ):
            if data.get(key):
                ctx.session.set_state(state_key, data[key])
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Solution Design"),
            summary=summary,
            data=data,
        )
