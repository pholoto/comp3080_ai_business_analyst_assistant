"""Use Case Generator feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..prompts import USE_CASE_SYSTEM_PROMPT, build_use_case_user_prompt
from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class UseCaseGeneratorFeature:
    """Generate structured use cases and acceptance criteria."""

    context: FeatureContext

    name: str = "use_case_generator"
    description: str = "Produce user stories, diagrams summary, and acceptance criteria."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        requirements = ctx.session.get_state("requirements")
        requirements_context = requirements if requirements else "No formal requirements yet."
        template = USE_CASE_SYSTEM_PROMPT
        prompt = build_use_case_user_prompt(
            requirements=requirements_context,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=template,
            user_prompt=prompt,
            default_title="Use Case Package",
            history=history,
        )
        if data.get("user_stories"):
            ctx.session.set_state("user_stories", data["user_stories"])
        if data.get("use_case_flows"):
            ctx.session.set_state("use_case_flows", data["use_case_flows"])
        if data.get("acceptance_criteria"):
            ctx.session.set_state("acceptance_criteria", data["acceptance_criteria"])
        summary = data.get("summary") or "Use case package drafted."
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Use Case Package"),
            summary=summary,
            data=data,
        )
