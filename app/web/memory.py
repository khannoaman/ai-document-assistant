from dataclasses import dataclass, field

from app.config import settings


@dataclass
class Turn:
    question: str
    standalone_question: str
    answer: str


@dataclass
class ChatMemory:
    turns: list[Turn] = field(default_factory=list)


chat_memory = ChatMemory()


def get_history() -> list[tuple[str, str]]:
    return [(turn.question, turn.answer) for turn in chat_memory.turns]


def add_turn(question: str, standalone_question: str, answer: str) -> None:
    chat_memory.turns.append(Turn(question, standalone_question, answer))
    if settings.max_memory_turns > 0 and len(chat_memory.turns) > settings.max_memory_turns:
        chat_memory.turns = chat_memory.turns[-settings.max_memory_turns :]
