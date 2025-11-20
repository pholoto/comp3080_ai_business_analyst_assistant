"""Feature Prioritization feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..prompts import (FEATURE_PRIORITISATION_SYSTEM_PROMPT,
                       build_feature_prioritisation_user_prompt)
from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class FeaturePrioritizationFeature:
    """Classify features using MoSCoW (Must/Should/Could)."""

    context: FeatureContext

    name: str = "feature_prioritization"
    description: str = "Organise features into priority bands with rationale."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        requirements = ctx.session.get_state("requirements") or []
        user_stories = ctx.session.get_state("user_stories") or []
        template = FEATURE_PRIORITISATION_SYSTEM_PROMPT
        prompt = build_feature_prioritisation_user_prompt(
            requirements=requirements,
            user_stories=user_stories,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=template,
            user_prompt=prompt,
            default_title="Feature Prioritisation",
            history=history,
        )
        prioritised = data.get("prioritised_features")
        if prioritised:
            ctx.session.set_state("prioritised_features", prioritised)
        if data.get("release_plan"):
            ctx.session.set_state("release_plan", data["release_plan"])
        summary = data.get("summary") or "Feature prioritisation snapshot recorded."
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Feature Prioritisation"),
            summary=summary,
            data=data,
        )
