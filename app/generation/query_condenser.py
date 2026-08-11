import logfire
from langchain_core.prompts import ChatPromptTemplate

from app.models import get_chat_model

logfire.configure(send_to_logfire="if-token-present")

_CONDENSE_SYSTEM_PROMPT = (
    "Given a conversation history and a follow-up question, rewrite the "
    "follow-up question into a standalone question that can be understood "
    "without the conversation history. Resolve pronouns and implicit "
    "references (e.g. 'it', 'that', 'the second one') using the history. "
    "If the follow-up question is already standalone, return it unchanged. "
    "Respond with only the rewritten question, no preamble or explanation."
)


def condense_question(chat_history: list[tuple[str, str]], question: str) -> str:
    """Rewrite `question` into a standalone search query using prior turns.
    Skips the LLM call entirely when there's no history to condense against."""
    if not chat_history:
        return question

    history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in chat_history)

    with logfire.span("generation.condense_question", question=question):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _CONDENSE_SYSTEM_PROMPT),
                ("human", "Conversation history:\n{history}\n\nFollow-up question: {question}"),
            ]
        )
        messages = prompt.format_messages(history=history_text, question=question)
        response = get_chat_model().invoke(messages)
        standalone = response.content.strip()
        logfire.info("condensed question", original=question, standalone=standalone)
        return standalone
