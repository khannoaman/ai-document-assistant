from pathlib import Path

from fastapi.templating import Jinja2Templates

_WEB_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
