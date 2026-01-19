"""Requirements Analysis feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from common.prompts import (REQUIREMENTS_ANALYSIS_SYSTEM_PROMPT,
                            build_requirements_analysis_user_prompt)

from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class RequirementsAnalysisFeature:
    """Capture functional and non-functional requirements comprehensively."""

    context: FeatureContext

    name: str = "requirements_analysis"
    description: str = "Systematically gather and document requirements."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        problem_statement = ctx.session.get_state(
            "problem_statement", "Problem statement not captured yet."
        )
        backlog = ctx.session.get_state("requirements") or []
        prompt = build_requirements_analysis_user_prompt(
            problem_statement=problem_statement,
            existing_requirements=backlog,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=REQUIREMENTS_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=prompt,
            default_title="Requirements Analysis",
            history=history,
        )
        summary = data.get("summary") or "Requirements backlog updated."
        if data.get("functional_requirements"):
            ctx.session.set_state("requirements", data["functional_requirements"])
        if data.get("non_functional_requirements"):
            ctx.session.set_state("non_functional_requirements", data["non_functional_requirements"])
        for key in ("dependencies", "risks", "open_questions"):
            if data.get(key):
                ctx.session.set_state(key, data[key])
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Requirements Analysis"),
            summary=summary,
            data=data,
        )
