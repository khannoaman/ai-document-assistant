import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import redis
from fastapi import Request

from app.config import settings

SESSION_COOKIE_NAME = "session_id"


@dataclass
class Turn:
    question: str
    standalone_question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)


_redis_client: redis.Redis | None = None
_lock = threading.Lock()


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    with _lock:
        if _redis_client is None:
            _redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        return _redis_client


def _session_key(session_id: str) -> str:
    return f"chat_session:{session_id}"


def get_session_id(request: Request) -> str:
    return request.cookies.get(SESSION_COOKIE_NAME) or uuid.uuid4().hex


def _load_turns(session_id: str) -> list[Turn]:
    raw = _get_redis().get(_session_key(session_id))
    if not raw:
        return []
    return [Turn(**item) for item in json.loads(raw)]


def _save_turns(session_id: str, turns: list[Turn]) -> None:
    payload = json.dumps([asdict(turn) for turn in turns])
    # `ex` both stores the value and (re)sets the TTL, giving a sliding
    # expiry window that extends with each turn instead of a fixed lifetime.
    _get_redis().set(_session_key(session_id), payload, ex=settings.session_ttl_seconds)


def get_history(session_id: str) -> list[tuple[str, str]]:
    return [(turn.question, turn.answer) for turn in _load_turns(session_id)]


def get_turns(session_id: str) -> list[Turn]:
    """Full turn records (including sources), for restoring the chat log on
    page load — as opposed to get_history()'s lightweight (question, answer)
    tuples, which are shaped for the condenser prompt."""
    return _load_turns(session_id)


def add_turn(
    session_id: str,
    question: str,
    standalone_question: str,
    answer: str,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    turns = _load_turns(session_id)
    turns.append(Turn(question, standalone_question, answer, sources or []))
    if settings.max_memory_turns > 0 and len(turns) > settings.max_memory_turns:
        turns = turns[-settings.max_memory_turns :]
    _save_turns(session_id, turns)
