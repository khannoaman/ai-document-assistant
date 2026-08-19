from fastapi import APIRouter, Form, Request

from app.generation import generate_answer
from app.web.memory import SESSION_COOKIE_NAME, add_turn, get_history, get_session_id
from app.web.templates import templates

router = APIRouter()


@router.post("/chat")
async def chat(request: Request, question: str = Form(...)):
    session_id = get_session_id(request)
    history = get_history(session_id)
    result = generate_answer(question, chat_history=history)
    add_turn(session_id, question, result.get("standalone_question", question), result["answer"])
    response = templates.TemplateResponse(
        request,
        "partials/chat_message.html",
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        },
    )
    # re-set on every response (idempotent) rather than tracking whether the
    # cookie was newly generated this request
    response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True, samesite="lax")
    return response
