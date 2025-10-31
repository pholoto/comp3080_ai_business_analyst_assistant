"""Generate BA reports using the provided DOCX template."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from ..config import get_template_path
from ..memory import Session


class BAReportGenerator:
    """Render a structured report aligned with the VinUni CECS capstone template."""

    def __init__(self, template_path: Path | None = None) -> None:
        self.template_path = template_path or get_template_path()

    def generate(self, session: Session, output_path: Path | None = None) -> Path:
        """Create the report populated with the session artefacts."""
        if output_path is None:
            output_dir = session.state.get("report_output_dir")
            if output_dir is None:
                output_dir = self.template_path.parent / ".." / ".." / "reports"
                output_dir = Path(output_dir).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                session.set_state("report_output_dir", str(output_dir))
            else:
                output_dir = Path(output_dir)
            output_path = output_dir / f"BA_Report_{session.session_id}.docx"
        document = Document(str(self.template_path))
        self._apply_heading_style(document)
        self._populate_document(document, session)
        document.save(str(output_path))
        session.set_state("report_path", str(output_path))
        return output_path

    def _apply_heading_style(self, document: Document) -> None:
        """Ensure heading style is consistent even if template is missing definitions."""
        style = document.styles.get("Heading 1")
        if style is not None and style.font:
            style.font.name = "Calibri"
            style.font.size = Pt(16)

    def _populate_document(self, document: Document, session: Session) -> None:
        add_heading = document.add_heading
        add_paragraph = document.add_paragraph

        add_heading("Project Overview", level=1)
        overview = session.get_state("project_overview") or "Project overview not captured yet."
        add_paragraph(overview)

        add_heading("Requirement Backlog", level=1)
        requirements = session.get_state("requirements") or []
        if requirements:
            for item in requirements:
                requirement = item if isinstance(item, Mapping) else {"requirement": str(item)}
                req_text = requirement.get("requirement") or requirement.get("name") or str(item)
                rationale = requirement.get("rationale", "")
                para = add_paragraph(f"• {req_text}")
                if rationale:
                    add_paragraph(f"  Rationale: {rationale}")
        else:
            add_paragraph("No requirements have been documented.")

        add_heading("User Stories", level=1)
        user_stories = session.get_state("user_stories") or []
        if user_stories:
            for story in user_stories:
                story = story if isinstance(story, Mapping) else {}
                identifier = story.get("id", "Story")
                role = story.get("role", "[role]")
                goal = story.get("goal", "[goal]")
                benefit = story.get("benefit", "[benefit]")
                add_paragraph(f"{identifier}: As a {role}, I want {goal} so that {benefit}.")
        else:
            add_paragraph("No user stories captured yet.")

        add_heading("Use Case Flows", level=1)
        flows = session.get_state("use_case_flows") or []
        if flows:
            for flow in flows:
                flow = flow if isinstance(flow, Mapping) else {}
                name = flow.get("name", "Use Case")
                primary = flow.get("primary_path", [])
                alternates = flow.get("alternate_paths", [])
                add_paragraph(f"Use Case: {name}")
                self._add_numbered_list(document, primary, prefix="Primary")
                if alternates:
                    self._add_numbered_list(document, alternates, prefix="Alternate")
        else:
            add_paragraph("Use case flows not yet defined.")

        add_heading("Acceptance Criteria", level=1)
        criteria = session.get_state("acceptance_criteria") or []
        if criteria:
            for item in criteria:
                add_paragraph(f"• {item}")
        else:
            add_paragraph("Acceptance criteria pending.")

        add_heading("Feature Prioritisation", level=1)
        prioritised = session.get_state("prioritised_features") or {}
        if prioritised:
            for bucket in ("must", "should", "could", "wont"):
                features = prioritised.get(bucket) or []
                add_paragraph(bucket.upper())
                if features:
                    for feature in features:
                        feature = feature if isinstance(feature, Mapping) else {}
                        name = feature.get("name", "Unnamed feature")
                        rationale = feature.get("rationale", "")
                        add_paragraph(f"• {name}")
                        if rationale:
                            add_paragraph(f"  Rationale: {rationale}")
                else:
                    add_paragraph("  No items in this bucket yet.")
        else:
            add_paragraph("Feature priorities have not been established.")

        add_heading("Market Fit Analysis", level=1)
        uvp = session.get_state("uvp")
        competitive = session.get_state("competitive_landscape") or []
        if uvp:
            add_paragraph(f"Unique Value Proposition: {uvp}")
        if competitive:
            for competitor in competitive:
                competitor = competitor if isinstance(competitor, Mapping) else {}
                name = competitor.get("name", "Competitor")
                positioning = competitor.get("positioning", "")
                strengths = competitor.get("strengths", "")
                gaps = competitor.get("gaps", "")
                add_paragraph(f"• {name} — Positioning: {positioning}")
                if strengths:
                    add_paragraph(f"  Strengths: {strengths}")
                if gaps:
                    add_paragraph(f"  Gaps: {gaps}")
        else:
            add_paragraph("Competitive landscape assessment pending.")

        add_heading("Stakeholder Matrix", level=1)
        stakeholder_map = session.get_state("stakeholder_map") or []
        if stakeholder_map:
            for stakeholder in stakeholder_map:
                stakeholder = stakeholder if isinstance(stakeholder, Mapping) else {}
                name = stakeholder.get("stakeholder", "Stakeholder")
                influence = stakeholder.get("influence", "")
                interest = stakeholder.get("interest", "")
                needs = stakeholder.get("needs", "")
                success = stakeholder.get("success_metrics", "")
                para = add_paragraph(f"• {name} (Influence: {influence}, Interest: {interest})")
                if needs:
                    add_paragraph(f"  Needs: {needs}")
                if success:
                    add_paragraph(f"  Success Metrics: {success}")
        else:
            add_paragraph("Stakeholder analysis is not yet available.")

        add_heading("Engagement Plan", level=1)
        engagement_plan = session.get_state("engagement_plan") or []
        if engagement_plan:
            if isinstance(engagement_plan, list):
                for line in engagement_plan:
                    add_paragraph(f"• {line}")
            else:
                add_paragraph(str(engagement_plan))
        else:
            add_paragraph("Engagement plan to be defined.")

        add_heading("Attached Documents", level=1)
        attachments = session.list_attachments()
        if attachments:
            add_paragraph(
                f"Current strategies — Chunking: {session.chunking_strategy}, Indexing: {session.indexing_strategy}"
            )
            for attachment in attachments:
                add_paragraph(
                    f"• {attachment.filename} — {attachment.word_count} words, {attachment.size} bytes"
                )
        else:
            add_paragraph("No supporting documents attached.")

        add_heading("Conversation Log", level=1)
        messages = session.memory.as_list()
        for message in messages:
            role = message.get("feature") or message.get("role")
            content = message.get("content", "")
            para = add_paragraph(f"[{role}] {content}")
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def _add_numbered_list(
        self,
    document,
        items: Iterable[str],
        *,
        prefix: str,
    ) -> None:
        for idx, item in enumerate(items, start=1):
            document.add_paragraph(f"{prefix} {idx}: {item}")
