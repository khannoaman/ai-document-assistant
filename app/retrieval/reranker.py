import threading

import logfire
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings

_model: CrossEncoder | None = None
_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is None:
            with logfire.span("retrieval.rerank.load_model", model=settings.rerank_model):
                _model = CrossEncoder(settings.rerank_model)
        return _model


def rerank(query: str, candidates: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    if not candidates:
        return []

    pairs = [[query, doc.page_content] for doc, _ in candidates]
    scores = _get_model().predict(pairs)

    reranked = sorted(zip((doc for doc, _ in candidates), scores), key=lambda pair: pair[1], reverse=True)
    return [(doc, float(score)) for doc, score in reranked]
