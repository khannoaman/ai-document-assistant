import logfire
import markdown
from fastapi import APIRouter, Form, Request

from app.generation import generate_answer
from app.web.memory import SESSION_COOKIE_NAME, add_turn, get_history, get_session_id
from app.web.templates import templates

logfire.configure(send_to_logfire="if-token-present")

router = APIRouter()

_ERROR_ANSWER_HTML = "<p>Something went wrong while answering that. Please try again in a moment.</p>"


@router.post("/chat")
def chat(request: Request, question: str = Form(...)):
    session_id = get_session_id(request)

    try:
        history = get_history(session_id)
        result = generate_answer(question, chat_history=history)
        add_turn(
            session_id,
            question,
            result.get("standalone_question", question),
            result["answer"],
            result["sources"],
        )
        # the LLM answers in markdown (bold, lists); render it to HTML here
        # rather than showing raw "**text**" syntax in the UI
        answer_html = markdown.markdown(result["answer"])
        sources = result["sources"]
    except Exception as e:
        # a Groq/Qdrant/Redis hiccup shouldn't surface a raw 500 to the user —
        # render a normal-looking assistant bubble instead. Deliberately not
        # calling add_turn: this isn't a real answer and shouldn't be fed back
        # into the condenser as if it were part of the conversation.
        logfire.error("chat request failed", question=question, error=str(e))
        answer_html = _ERROR_ANSWER_HTML
        sources = []

    response = templates.TemplateResponse(
        request,
        "partials/chat_message.html",
        {
            "question": question,
            "answer_html": answer_html,
            "sources": sources,
        },
    )
    # re-set on every response (idempotent) rather than tracking whether the
    # cookie was newly generated this request
    response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True, samesite="lax")
    return response
