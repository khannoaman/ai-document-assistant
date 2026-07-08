import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(_ENV_FILE)


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "google_genai")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.1-pro")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or None
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None


settings = Settings()


