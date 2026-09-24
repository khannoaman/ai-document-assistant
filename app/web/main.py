from pathlib import Path

import logfire
import markdown
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.web.memory import get_session_id, get_turns
from app.web.routes import chat, upload
from app.web.templates import templates

logfire.configure(send_to_logfire="if-token-present")

_WEB_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI Document Assistant")
app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

app.include_router(upload.router)
app.include_router(chat.router)


@app.get("/")
async def index(request: Request):
    session_id = get_session_id(request)
    try:
        chat_history = [
            {
                "question": turn.question,
                "answer_html": markdown.markdown(turn.answer),
                "sources": turn.sources,
            }
            for turn in get_turns(session_id)
        ]
    except Exception as e:
        # a Redis hiccup here shouldn't take down the whole homepage — fall
        # back to an empty chat log rather than failing the page load
        logfire.error("failed to load chat history", error=str(e))
        chat_history = []
    return templates.TemplateResponse(request, "index.html", {"chat_history": chat_history})
