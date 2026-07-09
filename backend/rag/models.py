from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.config.settings import settings

_PROVIDER_API_KEYS = {
    "google_genai": settings.google_api_key,
    "anthropic": settings.anthropic_api_key,
    "openai": settings.openai_api_key,
}


def get_llm():
    return init_chat_model(
        model=f"{settings.llm_provider}:{settings.llm_model}",
        api_key=_PROVIDER_API_KEYS.get(settings.llm_provider),
    )


def get_embeddings():
    if settings.embedding_provider != "google_genai":
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider!r}")
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


if __name__ == "__main__":
    print(get_llm())
    print(get_embeddings())
