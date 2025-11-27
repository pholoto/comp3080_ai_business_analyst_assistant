"""Documentation feature."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from common.prompts import (DOCUMENTATION_SYSTEM_PROMPT,
                            build_documentation_user_prompt)

from ..report import BAReportGenerator
from .base import FeatureContext, FeatureResult
from .llm_utils import build_attachment_context, request_json_response


@dataclass
class DocumentationFeature:
    """Auto-generate professional documentation and exports."""

    context: FeatureContext

    name: str = "documentation"
    description: str = "Auto-generate professional documentation from project work."

    def run(self, user_input: str, *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        history = ctx.session.memory.as_context()
        overview = ctx.session.get_state("project_overview", "Project overview not captured yet.")
        decisions = ctx.session.get_state("decision_log") or []
        prompt = build_documentation_user_prompt(
            project_overview=overview,
            decision_log=decisions,
            attachments=build_attachment_context(ctx.session),
            user_input=user_input,
        )
        data: Dict[str, Any] = request_json_response(
            ctx.llm,
            system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            default_title="Documentation",
            history=history,
        )
        summary = data.get("summary") or "Documentation outline generated."
        for key, state_key in (
            ("document_outline", "documentation_outline"),
            ("decision_log", "decision_log"),
            ("action_items", "documentation_actions"),
            ("publishing_assets", "publishing_assets"),
        ):
            if data.get(key):
                ctx.session.set_state(state_key, data[key])
        export_note = None
        export_dir = self._resolve_export_dir(ctx.session)
        ctx.session.set_state("report_output_dir", str(export_dir))
        try:
            report_path = BAReportGenerator().generate(ctx.session)
            export_note = f"BA report exported to {report_path}" if report_path else None
        except Exception as exc:  # pragma: no cover - best effort export
            export_note = f"BA report export failed: {exc}"
        if export_note:
            assets = data.get("publishing_assets") or []
            if isinstance(assets, list):
                assets = assets + [export_note]
            else:
                assets = [assets, export_note]
            data["publishing_assets"] = assets
            ctx.session.set_state("publishing_assets", assets)
        ctx.session.memory.append(
            "feature",
            summary,
            feature=self.name,
        )
        return FeatureResult(
            title=data.get("title", "Documentation"),
            summary=summary,
            data=data,
        )

    @staticmethod
    def _resolve_export_dir(session) -> Path:
        user_id = session.get_state("user_id") or "shared"
        project_root = Path(__file__).resolve().parents[2]
        export_dir = project_root / "back_end" / "data" / user_id
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir
