from fastapi import APIRouter, Form, Request

from app.generation import generate_answer
from app.web.memory import add_turn, get_history
from app.web.templates import templates

router = APIRouter()


@router.post("/chat")
async def chat(request: Request, question: str = Form(...)):
    history = get_history()
    result = generate_answer(question, chat_history=history)
    add_turn(question, result.get("standalone_question", question), result["answer"])
    return templates.TemplateResponse(
        request,
        "partials/chat_message.html",
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        },
    )
