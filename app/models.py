import logfire
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import SecretStr
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


def _build_chat_client(api_key: SecretStr) -> ChatGroq:
    return ChatGroq(model=settings.groq_model, api_key=api_key.get_secret_value(), temperature=0)


class FailoverChatModel:
    """Wraps a primary ChatGroq client with retry/backoff; if the primary
    key's retries are exhausted, fails over to the secondary key when one is
    configured, instead of failing the whole request."""

    def __init__(self, primary_key: SecretStr, fallback_key: SecretStr | None):
        self._primary = _build_chat_client(primary_key)
        self._fallback = _build_chat_client(fallback_key) if fallback_key else None

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
    def _invoke_primary(self, messages):
        return self._primary.invoke(messages)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
    def _invoke_fallback(self, messages):
        return self._fallback.invoke(messages)

    def invoke(self, messages):
        try:
            return self._invoke_primary(messages)
        except Exception:
            if self._fallback is None:
                raise
            logfire.warn("groq primary key exhausted, failing over to fallback key")
            return self._invoke_fallback(messages)


def get_chat_model() -> FailoverChatModel:
    """Built lazily (not a module-level singleton like `embedding_model`) so
    importing this module doesn't require GROQ_API_KEY for callers that only
    need embeddings, e.g. app.retrieval."""
    if settings.groq_api_key is None:
        raise ValueError("GROQ_API_KEY is required to use generation")
    return FailoverChatModel(settings.groq_api_key, settings.groq_fallback_api_key)
