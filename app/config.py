from pathlib import Path
from typing import Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):

    google_api_key: SecretStr | None = None
    huggingfacehub_api_token: SecretStr | None = None

    groq_api_key: SecretStr | None = None
    groq_fallback_api_key: SecretStr | None = None

    qdrant_api_key: SecretStr | None = None
    qdrant_cluster_endpoint: str | None = None
    collection_name: str = "ai_document_assistant_vector_store"

    base_dir: Path = _BASE_DIR
    data_dir: Path = _BASE_DIR / "data"

    @field_validator("google_api_key","huggingfacehub_api_token","groq_api_key","groq_fallback_api_key", "qdrant_api_key", "qdrant_cluster_endpoint", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

