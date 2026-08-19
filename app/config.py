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
    groq_model: str = "openai/gpt-oss-120b"

    qdrant_api_key: SecretStr | None = None
    qdrant_cluster_endpoint: str | None = None
    qdrant_port: int = 443
    collection_name: str = "ai_document_assistant_vector_store"

    embedding_provider: Literal["huggingface", "gemini"] = "huggingface"
    huggingface_embedding_model: str = "BAAI/bge-small-en-v1.5"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    chunk_size: int = 800
    chunk_overlap: int = 100
    min_content_length: int = 20

    retrieval_top_k: int = 5

    # --- Hybrid retrieval / re-ranking ---
    retrieval_fetch_k: int = 20        # candidate pool size per sub-method (dense, BM25) before fusion
    rrf_k: int = 60                    # RRF damping constant (standard value from the original paper)
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10             # candidates kept after cross-encoder rerank, before MMR (keep >= retrieval_top_k)
    mmr_lambda: float = 0.5            # 0 = max diversity, 1 = max relevance
    bm25_scroll_batch_size: int = 256  # page size when scrolling Qdrant to build the BM25 corpus
    # cross-encoder logit threshold, not cosine similarity. -4.0 separates
    # observed off-topic queries (~-10 to -11) from on-topic ones, including
    # generic/summary-style questions that don't score strongly positive.
    min_rerank_score: float = -4.0

    # --- Conversational memory ---
    max_memory_turns: int = 5          # prior turns retained and fed to the condenser prompt

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

