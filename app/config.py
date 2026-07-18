from pathlib import Path
from typing import Any, Literal
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):

    google_api_key: SecretStr | None = None
    huggingfacehub_api_token: SecretStr | None = None

    groq_api_key: SecretStr | None = None
    groq_fallback_api_key: SecretStr | None = None

    qdrant_api_key: SecretStr | None = None
    qdrant_cluster_endpoint: str | None = None
    qdrant_port: int = 443
    collection_name: str = "ai_document_assistant_vector_store"

    embedding_provider: Literal["huggingface", "gemini"] = "huggingface"
    huggingface_embedding_model: str = "BAAI/bge-small-en-v1.5"
    gemini_embedding_model: str = "models/embedding-001"

    chunk_size: int = 800
    chunk_overlap: int = 100
    min_content_length: int = 20

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

