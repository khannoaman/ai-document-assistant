from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.web.routes import chat, upload
from app.web.templates import templates

_WEB_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI Document Assistant")
app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

app.include_router(upload.router)
app.include_router(chat.router)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
