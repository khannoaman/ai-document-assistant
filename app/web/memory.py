import threading
import uuid
from dataclasses import dataclass, field

from fastapi import Request

from app.config import settings

SESSION_COOKIE_NAME = "session_id"


@dataclass
class Turn:
    question: str
    standalone_question: str
    answer: str


@dataclass
class ChatMemory:
    turns: list[Turn] = field(default_factory=list)


# No TTL/eviction: sessions accumulate for the life of the process. Fine for
# a personal-use app; would need cleanup for a long-lived multi-user deployment.
_sessions: dict[str, ChatMemory] = {}
_lock = threading.Lock()


def get_session_id(request: Request) -> str:
    return request.cookies.get(SESSION_COOKIE_NAME) or uuid.uuid4().hex


def get_history(session_id: str) -> list[tuple[str, str]]:
    with _lock:
        memory = _sessions.setdefault(session_id, ChatMemory())
        return [(turn.question, turn.answer) for turn in memory.turns]


def add_turn(session_id: str, question: str, standalone_question: str, answer: str) -> None:
    with _lock:
        memory = _sessions.setdefault(session_id, ChatMemory())
        memory.turns.append(Turn(question, standalone_question, answer))
        if settings.max_memory_turns > 0 and len(memory.turns) > settings.max_memory_turns:
            memory.turns = memory.turns[-settings.max_memory_turns :]
