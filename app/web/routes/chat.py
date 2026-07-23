from fastapi import APIRouter, Form, Request

from app.generation import generate_answer
from app.web.templates import templates

router = APIRouter()


@router.post("/chat")
async def chat(request: Request, question: str = Form(...)):
    result = generate_answer(question)
    return templates.TemplateResponse(
        request,
        "partials/chat_message.html",
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        },
    )
