from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings


def _build_embedding_model():
    if settings.embedding_provider == "gemini":
        if settings.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY is required when embedding_provider is 'gemini'")
        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
        )
    return HuggingFaceEmbeddings(
        model_name=settings.huggingface_embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


class RetryingEmbeddings(Embeddings):
    """Wraps an embeddings backend with retry/backoff so a transient API
    error (rate limit, timeout) doesn't fail an entire indexing batch."""

    def __init__(self, model):
        self._model = model

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


embedding_model = RetryingEmbeddings(_build_embedding_model())
