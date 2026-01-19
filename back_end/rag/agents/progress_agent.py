"""Progress agent that evaluates project execution against plan."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional, Sequence, Tuple

from ..config import DEFAULT_CONFIG, RagConfig
from ..generation import ResponseGenerator
from ..ranking import SimilarityRanker
from ..retrieval import ContextRetriever
from .ideation_agent import INSUFFICIENT_CONTEXT_RESPONSE

DEFAULT_PROGRESS_TAGS: Tuple[str, ...] = (
    "progress",
    "timeline",
    "milestone",
    "task",
    "deliverable",
    "status",
)


@dataclass
class ProgressRequest:
    """Input payload for the progress agent."""

    reference_date: Optional[str] = None
    top_k: int = 8
    tags: Optional[Sequence[str]] = None


class ProgressAgent:
    """Analyze project documents to surface status insights."""

    def __init__(
        self,
        retriever: ContextRetriever | None = None,
        generator: ResponseGenerator | None = None,
        ranker: SimilarityRanker | None = None,
        config: RagConfig | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.retriever = retriever or ContextRetriever(config=self.config)
        self.generator = generator or ResponseGenerator()
        self.ranker = ranker or SimilarityRanker()
        self.default_tags = DEFAULT_PROGRESS_TAGS
        self.system_prompt = (
            "You are the Progress Agent for the AI Business Analyst Assistant. "
            "Use only the retrieved context to assess the project's status. "
            f"If the context is insufficient, respond exactly with '{INSUFFICIENT_CONTEXT_RESPONSE}'."
        )

    def analyze_progress(self, user_id: str, request: ProgressRequest) -> dict[str, object]:
        """Evaluate project progress with respect to the provided date."""
        reference_date = self._resolve_date(request.reference_date)
        query = f"Project status update as of {reference_date}" if reference_date else "Project status update"
        active_tags = tuple(request.tags) if request.tags else self.default_tags
        results = self.retriever.retrieve(
            user_id,
            query,
            top_k=request.top_k,
            tags=active_tags,
        )
        task_prompt = (
            "Review the retrieved project documents to evaluate execution progress as of {ref_date}. "
            "Produce a Markdown table with columns: Task/Deliverable, Planned Due, Current Status, Completion %, Deadline Risk. "
            "Derive values strictly from the context; if a value is missing, write 'Unknown'. "
            "After the table, provide: (1) a brief summary progress score or pulse, (2) bullet notifications for late or at-risk deadlines, "
            "and (3) targeted suggestions for recovering late work."
        ).format(ref_date=reference_date or "the latest available date")
        response = self.generator.generate(
            query=query,
            contexts=results,
            system_prompt=self.system_prompt,
            task_prompt=task_prompt,
            guardrails=(
                "Do not invent dates or percentages. "
                "If a deadline or progress value is not present, mark it as 'Unknown'. "
                f"If the context lacks project status data, respond with '{INSUFFICIENT_CONTEXT_RESPONSE}'."
            ),
        )
        ranked = [entry.to_dict() for entry in self.ranker.rank(results)]
        return {
            "answer": response,
            "sources": ranked,
            "context_items": [result.metadata for result in results],
            "reference_date": reference_date,
        }

    @staticmethod
    def _resolve_date(value: Optional[str]) -> Optional[str]:
        if not value:
            # Default to today's date in UTC for consistent reasoning.
            return datetime.now(timezone.utc).date().isoformat()
        trimmed = value.strip()
        if not trimmed:
            return datetime.now(timezone.utc).date().isoformat()
        try:
            if len(trimmed) == 10:
                return date.fromisoformat(trimmed).isoformat()
        except ValueError:
            pass
        try:
            parsed_dt = datetime.fromisoformat(trimmed)
        except ValueError as exc:
            raise ValueError(
                "reference_date must be an ISO date (YYYY-MM-DD) or datetime string"
            ) from exc
        return parsed_dt.date().isoformat()


__all__ = ["ProgressAgent", "ProgressRequest"]
