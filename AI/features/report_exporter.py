"""BA Report Export feature."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..report import BAReportGenerator
from .base import FeatureContext, FeatureResult


@dataclass
class BAReportExporterFeature:
    """Compile all artefacts into the official report template."""

    context: FeatureContext

    name: str = "ba_report_export"
    description: str = "Generate a structured BA report aligned with the template."

    def run(self, user_input: str = "", *, context: FeatureContext | None = None) -> FeatureResult:
        ctx = context or self.context
        generator = BAReportGenerator()
        if user_input:
            ctx.session.set_state("report_notes", user_input)
        output_path = generator.generate(ctx.session)
        ctx.session.memory.append(
            "feature",
            f"Report generated at {output_path}",
            feature=self.name,
        )
        data: Dict[str, Any] = {
            "title": "BA Report Export",
            "summary": "Latest artefacts exported to the VinUni CECS template.",
            "path": str(output_path),
        }
        return FeatureResult(
            title=data["title"],
            summary=data["summary"],
            data=data,
        )
