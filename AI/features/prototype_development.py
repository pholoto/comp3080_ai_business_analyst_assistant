"""Prototype Development feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from common.prompts import (PROTOTYPE_DEVELOPMENT_SYSTEM_PROMPT,
                            build_prototype_development_user_prompt)

from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class PrototypeDevelopmentFeature:
    """Coach teams through building an MVP."""

    context: FeatureContext

    name: str = "prototype_development"
    description: str = "Guide MVP build with best practices and code suggestions."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        solution_blueprint = ctx.session.get_state(
            "solution_blueprint", "Solution blueprint not documented yet."
        )
        team_capabilities = ctx.session.get_state(
            "team_capabilities", "Team capability profile not provided."
        )
        prompt = build_prototype_development_user_prompt(
            solution_blueprint=solution_blueprint,
            team_capabilities=team_capabilities,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=PROTOTYPE_DEVELOPMENT_SYSTEM_PROMPT,
            user_prompt=prompt,
            default_title="Prototype Development",
            history=history,
        )
        summary = data.get("summary") or "Prototype plan drafted."
        for key, state_key in (
            ("mvp_scope", "mvp_scope"),
            ("implementation_steps", "prototype_plan"),
            ("code_suggestions", "prototype_code_suggestions"),
            ("tooling_recommendations", "tooling_recommendations"),
            ("risks", "prototype_risks"),
            ("success_metrics", "prototype_success_metrics"),
        ):
            if data.get(key):
                ctx.session.set_state(state_key, data[key])
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Prototype Development"),
            summary=summary,
            data=data,
        )
