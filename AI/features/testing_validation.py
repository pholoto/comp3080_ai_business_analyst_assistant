"""Testing & Validation feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from common.prompts import (TESTING_VALIDATION_SYSTEM_PROMPT,
                            build_testing_validation_user_prompt)

from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class TestingValidationFeature:
    """Produce a comprehensive QA plan."""

    context: FeatureContext

    name: str = "testing_validation"
    description: str = "Create comprehensive test plans and validation strategies."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        requirements = ctx.session.get_state("requirements") or "Requirements not recorded."
        prototype_summary = ctx.session.get_state(
            "prototype_plan", "Prototype plan not established yet."
        )
        prompt = build_testing_validation_user_prompt(
            requirements_digest=requirements,
            prototype_summary=prototype_summary,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=TESTING_VALIDATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            default_title="Testing & Validation",
            history=history,
        )
        summary = data.get("summary") or "Testing strategy drafted."
        for key, state_key in (
            ("test_matrix", "test_matrix"),
            ("validation_plan", "validation_plan"),
            ("quality_gates", "quality_gates"),
            ("tooling", "qa_tooling"),
            ("risks", "qa_risks"),
        ):
            if data.get(key):
                ctx.session.set_state(state_key, data[key])
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Testing & Validation"),
            summary=summary,
            data=data,
        )
