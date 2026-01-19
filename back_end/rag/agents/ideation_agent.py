"""Ideation agent that proposes new ideas based on project context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..config import DEFAULT_CONFIG, RagConfig
from ..generation import ResponseGenerator
from ..ranking import SimilarityRanker
from ..retrieval import ContextRetriever

INSUFFICIENT_CONTEXT_RESPONSE = "I could not find enough information in the documents to answer."
DEFAULT_IDEATION_TAGS: Tuple[str, ...] = (
    "ideation",
    "brainstorm",
    "meeting",
    "brief",
    "report",
)


@dataclass
class IdeationRequest:
    """Input for the ideation agent."""

    topic: str
    desired_ideas: int = 6
    top_k: int = 8
    tags: Optional[Sequence[str]] = None


class IdeationAgent:
    """Generate new solution ideas grounded in project documents."""

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
        self.min_ideas = 5
        self.max_ideas = 10
        self.default_tags = DEFAULT_IDEATION_TAGS
        self.system_prompt = (
            "You are the Ideation Agent for the AI Business Analyst Assistant. "
            "Use only the retrieved context to craft your answer. "
            f"If the context is insufficient, respond exactly with '{INSUFFICIENT_CONTEXT_RESPONSE}'."
        )

    def generate_ideas(self, user_id: str, request: IdeationRequest) -> dict[str, object]:
        """Return ideation ideas grounded in the user's documents."""
        topic = request.topic.strip()
        if not topic:
            raise ValueError("topic is required")
        active_tags = tuple(request.tags) if request.tags else self.default_tags
        results = self.retriever.retrieve(
            user_id,
            topic,
            top_k=request.top_k,
            tags=active_tags,
        )
        bounded_ideas = max(
            self.min_ideas,
            min(request.desired_ideas, self.max_ideas),
        )
        task_prompt = (
            "Create between {min_count} and {max_count} actionable ideas that address the user's topic. "
            "Each idea must include: a concise title, a short rationale referencing the context, and how it aligns with the user's goal. "
            "Use numbered Markdown list format. "
            "Do not invent details not grounded in the context."
        ).format(min_count=self.min_ideas, max_count=bounded_ideas)
        response = self.generator.generate(
            query=topic,
            contexts=results,
            system_prompt=self.system_prompt,
            task_prompt=task_prompt,
            guardrails=(
                "Only use facts included in the context. "
                f"If the context does not mention the topic, respond with '{INSUFFICIENT_CONTEXT_RESPONSE}'."
            ),
        )
        ranked = [entry.to_dict() for entry in self.ranker.rank(results)]
        return {
            "answer": response,
            "sources": ranked,
            "context_items": [result.metadata for result in results],
        }


__all__ = ["IdeationAgent", "IdeationRequest"]
