"""Session objects and manager for the assistant."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any, Dict

from .memory import ConversationMemory


@dataclass
class Session:
    """Per-user conversational session."""

    session_id: str
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    state: Dict[str, Any] = field(default_factory=dict)

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any | None = None) -> Any:
        return self.state.get(key, default)


class SessionManager:
    """Simple in-memory session manager."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> Session:
        session_id = secrets.token_hex(16)
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear(self) -> None:
        self._sessions.clear()
