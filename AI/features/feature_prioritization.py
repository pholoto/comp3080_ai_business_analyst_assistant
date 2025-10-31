"""Feature Prioritization feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
        template = (
            "Act as an AI Business Analyst performing MoSCoW prioritisation. Return a JSON "
            "object with keys: title, summary, prioritised_features (object with keys must, should, "
            "could, wont; each value is list of objects with fields name, rationale, dependencies), "
            "and release_plan (list of strings). Follow existing constraints from the conversation."
        )
        prompt = (
            "Consolidated artefacts:\n- Requirements: "
            f"{requirements}\n- User stories: {user_stories}\n\n"
            "Referenced attachments:\n"
            f"{build_attachment_context(ctx.session)}\n\n"
            "New considerations: "
            f"{user_input}"
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
