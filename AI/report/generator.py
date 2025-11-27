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

        add_heading("Problem Definition", level=1)
        problem_statement = session.get_state("problem_statement") or "Problem statement not captured."
        add_paragraph(problem_statement)
        self._write_simple_list(document, session.get_state("pain_points"), "Pain Points")
        self._write_simple_list(document, session.get_state("success_metrics"), "Success Metrics")
        self._write_simple_list(document, session.get_state("clarifying_questions"), "Clarifying Questions")
        self._write_simple_list(document, session.get_state("assumptions"), "Assumptions")

        add_heading("Requirements Analysis", level=1)
        requirements = session.get_state("requirements") or []
        if requirements:
            add_paragraph("Functional Requirements:")
            for item in requirements:
                requirement = item if isinstance(item, Mapping) else {"requirement": str(item)}
                req_id = requirement.get("id")
                req_text = requirement.get("requirement") or requirement.get("name") or str(item)
                priority = requirement.get("priority")
                rationale = requirement.get("rationale")
                label = f"• {req_id}: {req_text}" if req_id else f"• {req_text}"
                if priority:
                    label += f" (Priority: {priority})"
                add_paragraph(label)
                if rationale:
                    add_paragraph(f"  Rationale: {rationale}")
        else:
            add_paragraph("Functional requirements have not been documented.")
        non_functional = session.get_state("non_functional_requirements") or []
        if non_functional:
            add_paragraph("Non-functional Requirements:")
            for item in non_functional:
                requirement = item if isinstance(item, Mapping) else {"requirement": str(item)}
                attribute = requirement.get("attribute", "Attribute")
                req_text = requirement.get("requirement") or str(item)
                rationale = requirement.get("rationale")
                add_paragraph(f"• {attribute}: {req_text}")
                if rationale:
                    add_paragraph(f"  Rationale: {rationale}")
        self._write_simple_list(document, session.get_state("dependencies"), "Dependencies")
        self._write_simple_list(document, session.get_state("risks"), "Risks")

        add_heading("Solution Design", level=1)
        blueprint = session.get_state("solution_blueprint") or "Solution blueprint not documented."
        add_paragraph(blueprint)
        components = session.get_state("architecture_components") or []
        if components:
            add_paragraph("Component Breakdown:")
            for component in components:
                component = component if isinstance(component, Mapping) else {}
                name = component.get("name", "Component")
                responsibility = component.get("responsibility", "")
                tech_stack = component.get("tech_stack", "")
                integrations = component.get("integrations", "")
                add_paragraph(f"• {name} — {responsibility}")
                if tech_stack:
                    add_paragraph(f"  Tech: {tech_stack}")
                if integrations:
                    add_paragraph(f"  Integrations: {integrations}")
        self._write_simple_list(document, session.get_state("design_patterns"), "Design Patterns")
        technology_choices = session.get_state("technology_choices") or []
        if technology_choices:
            add_paragraph("Technology Choices:")
            for choice in technology_choices:
                choice = choice if isinstance(choice, Mapping) else {}
                area = choice.get("area", "Area")
                option = choice.get("option", "Option")
                rationale = choice.get("rationale", "")
                add_paragraph(f"• {area}: {option}")
                if rationale:
                    add_paragraph(f"  Rationale: {rationale}")
        self._write_simple_list(document, session.get_state("diagram_ideas"), "Diagram Ideas")

        add_heading("Prototype Development", level=1)
        self._write_simple_list(document, session.get_state("mvp_scope"), "MVP Scope")
        plan = session.get_state("prototype_plan") or []
        if plan:
            add_paragraph("Implementation Steps:")
            self._add_numbered_list(document, plan, prefix="Step")
        code_suggestions = session.get_state("prototype_code_suggestions") or []
        if code_suggestions:
            add_paragraph("Code Suggestions:")
            for suggestion in code_suggestions:
                suggestion = suggestion if isinstance(suggestion, Mapping) else {}
                desc = suggestion.get("description", "Suggestion")
                snippet = suggestion.get("snippet", "")
                add_paragraph(f"• {desc}")
                if snippet:
                    add_paragraph(snippet)
        self._write_simple_list(document, session.get_state("tooling_recommendations"), "Tooling Recommendations")
        self._write_simple_list(document, session.get_state("prototype_risks"), "Prototype Risks")
        self._write_simple_list(document, session.get_state("prototype_success_metrics"), "Prototype Success Metrics")

        add_heading("Testing & Validation", level=1)
        test_matrix = session.get_state("test_matrix") or []
        if test_matrix:
            add_paragraph("Test Matrix:")
            for entry in test_matrix:
                entry = entry if isinstance(entry, Mapping) else {}
                area = entry.get("area", "Area")
                objective = entry.get("objective", "Objective")
                test_cases = entry.get("test_cases", [])
                owner = entry.get("owner", "Owner")
                add_paragraph(f"• {area} — {objective} (Owner: {owner})")
                if test_cases:
                    self._add_numbered_list(document, test_cases, prefix="Test")
        self._write_simple_list(document, session.get_state("validation_plan"), "Validation Plan")
        self._write_simple_list(document, session.get_state("quality_gates"), "Quality Gates")
        self._write_simple_list(document, session.get_state("qa_tooling"), "QA Tooling")
        self._write_simple_list(document, session.get_state("qa_risks"), "QA Risks")

        add_heading("Documentation Summary", level=1)
        outline = session.get_state("documentation_outline") or []
        if outline:
            for section in outline:
                section = section if isinstance(section, Mapping) else {}
                name = section.get("section", "Section")
                intent = section.get("intent", "")
                content_summary = section.get("content_summary", section.get("summary", ""))
                add_paragraph(f"• {name} — {intent}")
                if content_summary:
                    add_paragraph(f"  Summary: {content_summary}")
        self._write_simple_list(document, session.get_state("decision_log"), "Decision Log")
        self._write_simple_list(document, session.get_state("documentation_actions"), "Action Items")
        self._write_simple_list(document, session.get_state("publishing_assets"), "Publishing Assets")

        add_heading("Market Analysis", level=1)
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
        target_segments = session.get_state("target_segments") or []
        if target_segments:
            add_paragraph("Target Segments:")
            for segment in target_segments:
                segment = segment if isinstance(segment, Mapping) else {}
                name = segment.get("segment", "Segment")
                needs = segment.get("needs", "")
                fit_score = segment.get("fit_score")
                add_paragraph(f"• {name} — Needs: {needs} (Fit: {fit_score})")
        self._write_simple_list(document, session.get_state("go_to_market_ideas"), "Go-to-Market Ideas")
        self._write_simple_list(document, session.get_state("impact_considerations"), "Impact Considerations")

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

    def _write_simple_list(self, document, items, heading: str) -> None:
        if not items:
            return
        add_paragraph = document.add_paragraph
        add_paragraph(f"{heading}:")
        if isinstance(items, (list, tuple, set)):
            for entry in items:
                add_paragraph(f"• {entry}")
        else:
            add_paragraph(f"• {items}")
