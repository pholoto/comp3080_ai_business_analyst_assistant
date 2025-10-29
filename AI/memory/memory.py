"""Conversation memory store for the assistant."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Sequence

Role = Literal["system", "user", "assistant", "feature"]


@dataclass
class Message:
    """Single conversational message."""

    role: Role
    content: str
    feature: str | None = None

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "feature": self.feature}


@dataclass
class ConversationMemory:
    """In-memory rolling log of the conversation for a session."""

    messages: List[Message] = field(default_factory=list)

    def append(self, role: Role, content: str, *, feature: str | None = None) -> None:
        self.messages.append(Message(role=role, content=content, feature=feature))

    def last(self) -> Message | None:
        if not self.messages:
            return None
        return self.messages[-1]

    def extend(self, entries: Iterable[Message]) -> None:
        self.messages.extend(entries)

    def as_list(self) -> List[dict]:
        return [msg.as_dict() for msg in self.messages]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.messages)

    def truncate(self, max_messages: int) -> None:
        """Trim the log to the newest *max_messages* messages."""
        if max_messages <= 0:
            self.messages.clear()
            return
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def as_context(self) -> Sequence[dict]:
        """Return an OpenAI-style list of messages."""
        return [
            {"role": msg.role if msg.role != "feature" else "assistant", "content": msg.content}
            for msg in self.messages
        ]
