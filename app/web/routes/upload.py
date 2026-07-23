from pathlib import Path

import logfire
from fastapi import APIRouter, BackgroundTasks, Request, UploadFile

from app.config import settings
from app.indexing.pipeline import run_indexing_pipeline
from app.web.state import indexing_state, set_status
from app.web.templates import templates

router = APIRouter()


def _run_indexing() -> None:
    try:
        run_indexing_pipeline()
        set_status("done", "Indexing complete.")
    except Exception as e:
        logfire.error("background indexing failed", error=str(e))
        set_status("error", str(e))


@router.post("/upload")
async def upload_files(request: Request, background_tasks: BackgroundTasks, files: list[UploadFile]):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        filename = Path(file.filename).name  # strip any path components — no path traversal
        (settings.data_dir / filename).write_bytes(await file.read())

    set_status("running", "Indexing documents...")
    background_tasks.add_task(_run_indexing)
    return templates.TemplateResponse(request, "partials/status.html", {"state": indexing_state})


@router.get("/status")
async def status(request: Request):
    return templates.TemplateResponse(request, "partials/status.html", {"state": indexing_state})
