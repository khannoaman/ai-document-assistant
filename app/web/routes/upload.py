from pathlib import Path

import logfire
from fastapi import APIRouter, BackgroundTasks, Request, UploadFile

from app.config import settings
from app.indexing.pipeline import run_indexing_pipeline_for_paths
from app.retrieval import invalidate_bm25_index
from app.web.memory import SESSION_COOKIE_NAME, get_session_id
from app.web.state import indexing_state, set_status
from app.web.templates import templates

router = APIRouter()


def _run_indexing(file_paths: list[Path], session_id: str) -> None:
    try:
        run_indexing_pipeline_for_paths(file_paths, session_id=session_id)
        invalidate_bm25_index()
        set_status("done", "Indexing complete.")
    except Exception as e:
        logfire.error("background indexing failed", error=str(e))
        set_status("error", str(e))


@router.post("/upload")
async def upload_files(request: Request, background_tasks: BackgroundTasks, files: list[UploadFile]):
    # read (or mint) the session cookie here, before the background task
    # starts, so uploads are scoped to the uploader even if this is their
    # first request of the visit (chat.py is otherwise the only route that
    # sets this cookie)
    session_id = get_session_id(request)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for file in files:
        filename = Path(file.filename).name  # strip any path components — no path traversal
        path = settings.data_dir / filename
        path.write_bytes(await file.read())
        saved_paths.append(path)

    set_status("running", "Indexing documents...")
    background_tasks.add_task(_run_indexing, saved_paths, session_id)
    response = templates.TemplateResponse(request, "partials/status.html", {"state": indexing_state})
    response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True, samesite="lax")
    return response


@router.get("/status")
async def status(request: Request):
    return templates.TemplateResponse(request, "partials/status.html", {"state": indexing_state})
