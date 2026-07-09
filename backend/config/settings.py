import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parents[2]
_ENV_FILE = _BASE_DIR / ".env"

load_dotenv(_ENV_FILE)


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "google_genai")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.1-pro")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "google_genai")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or None
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None

    base_dir: Path = _BASE_DIR
    data_dir: Path = _BASE_DIR / "data"
    vector_db_dir: Path = _BASE_DIR / "vector_db"
    collection_name: str = "vector_db"


settings = Settings()
