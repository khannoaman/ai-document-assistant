import re
import threading
from dataclasses import dataclass, field
from typing import Any

import logfire
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from app.config import settings

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Bm25Cache:
    built: bool = False
    index: BM25Okapi | None = None
    documents: list[Document] = field(default_factory=list)


_cache = _Bm25Cache()
_lock = threading.Lock()


def _scroll_all_points(client: QdrantClient) -> list[Document]:
    documents: list[Document] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=settings.collection_name,
            limit=settings.bm25_scroll_batch_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for record in records:
            payload = record.payload or {}
            metadata = dict(payload.get("metadata") or {})
            metadata["_id"] = record.id
            documents.append(Document(page_content=payload.get("page_content", ""), metadata=metadata))
        if offset is None:
            break
    return documents


def get_bm25_index(client: QdrantClient) -> tuple[BM25Okapi | None, list[Document]]:
    if _cache.built:
        return _cache.index, _cache.documents

    with _lock:
        if _cache.built:
            return _cache.index, _cache.documents

        with logfire.span("retrieval.bm25.build_index"):
            documents = _scroll_all_points(client)
            index = BM25Okapi([_tokenize(doc.page_content) for doc in documents]) if documents else None
            _cache.index = index
            _cache.documents = documents
            _cache.built = True
            logfire.info("bm25 index built", document_count=len(documents))

        return _cache.index, _cache.documents


def invalidate_bm25_index() -> None:
    global _cache
    with _lock:
        _cache = _Bm25Cache()
        logfire.info("bm25 index invalidated")


def _matches_filters(doc: Document, filters: dict[str, Any] | None, session_id: str | None) -> bool:
    if filters and not all(doc.metadata.get(key) == value for key, value in filters.items()):
        return False
    if session_id is not None:
        # visible if it's this session's own upload, or "public" content
        # with no session_id at all (CLI/bulk-indexed) — mirrors the Qdrant
        # side's should-clause in app/retrieval/retriever.py
        doc_session = doc.metadata.get("session_id")
        if doc_session is not None and doc_session != session_id:
            return False
    return True


def search_bm25(
    client: QdrantClient,
    query: str,
    k: int,
    filters: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> list[tuple[Document, float]]:
    index, documents = get_bm25_index(client)
    if index is None:
        return []

    scores = index.get_scores(_tokenize(query))
    ranked = sorted(
        (
            (doc, float(score))
            for doc, score in zip(documents, scores)
            if _matches_filters(doc, filters, session_id)
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:k]
