import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Any

import logfire
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition, MatchValue, PayloadField, PayloadSchemaType

from app.config import settings
from app.models import embedding_model
from app.retrieval.bm25_index import search_bm25
from app.retrieval.fusion import mmr_select, reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.vectorstore.client import get_qdrant_client

logfire.configure(send_to_logfire="if-token-present")


class NoIndexedDocumentsError(RuntimeError):
    """Raised when a query runs before anything has ever been indexed.

    A distinct type (not a bare RuntimeError) so callers — chat.py in
    particular — can show a specific "upload a document first" message
    instead of lumping this in with real failures (a dead Qdrant/Groq
    connection, etc.) under one generic error message."""


# page_number/slide_number are ints in the normalized schema (see
# app/indexing/document_loader/normalize.py); everything else filterable is a
# keyword (exact-match string).
_INTEGER_FIELDS = {"page_number", "slide_number"}

# QdrantVectorStore stores Document metadata nested under a "metadata" key in
# the point payload (alongside "page_content"), not flat — filters and payload
# indexes both have to address it via that path.
_METADATA_PREFIX = "metadata."


def _ensure_payload_indexes(client: QdrantClient, fields: set[str]) -> None:
    # Qdrant requires an explicit payload index before a field can be used in
    # a filter; creating one that already exists is a cheap no-op.
    for field in fields:
        schema = PayloadSchemaType.INTEGER if field in _INTEGER_FIELDS else PayloadSchemaType.KEYWORD
        client.create_payload_index(
            collection_name=settings.collection_name,
            field_name=f"{_METADATA_PREFIX}{field}",
            field_schema=schema,
        )


def _session_condition(session_id: str) -> Filter:
    """Visible to `session_id`: chunks it uploaded itself, or "public" chunks
    with no session_id at all (CLI/bulk-indexed content, or anything indexed
    before this field existed) — not an exact match, since a strict match
    would also hide that public/legacy content from every session."""
    field = f"{_METADATA_PREFIX}session_id"
    return Filter(
        should=[
            FieldCondition(key=field, match=MatchValue(value=session_id)),
            IsEmptyCondition(is_empty=PayloadField(key=field)),
        ]
    )


def _build_filter(filters: dict[str, Any] | None, session_id: str | None) -> Filter | None:
    conditions: list[Any] = [
        FieldCondition(key=f"{_METADATA_PREFIX}{key}", match=MatchValue(value=value))
        for key, value in (filters or {}).items()
    ]
    if session_id:
        conditions.append(_session_condition(session_id))
    if not conditions:
        return None
    return Filter(must=conditions)


def retrieve(
    query: str,
    k: int | None = None,
    filters: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> list[tuple[Document, float]]:
    """Hybrid retrieval: fuse dense (Qdrant) + keyword (BM25) candidates via
    Reciprocal Rank Fusion, cross-encoder rerank the fused pool, then select
    the final top-k with MMR for relevance/diversity balance.

    session_id, when given, scopes results to that session's own uploads
    plus public (session_id-less) content — pass the web session's cookie
    value here; leave it None to search everything unscoped (e.g. the CLI)."""
    k = k if k is not None else settings.retrieval_top_k
    fetch_k = settings.retrieval_fetch_k

    with logfire.span("retrieval.search", query=query, k=k):
        client = get_qdrant_client()
        collections = {c.name for c in client.get_collections().collections}
        if settings.collection_name not in collections:
            raise NoIndexedDocumentsError(
                f"Collection '{settings.collection_name}' does not exist yet — "
                "run the indexing pipeline first."
            )

        index_fields = set(filters.keys()) if filters else set()
        if session_id:
            index_fields.add("session_id")
        if index_fields:
            _ensure_payload_indexes(client, index_fields)

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embedding_model,
        )

        qdrant_filter = _build_filter(filters, session_id)
        dense = vectorstore.similarity_search_with_score(query, k=fetch_k, filter=qdrant_filter)
        bm25 = search_bm25(client, query, fetch_k, filters=filters, session_id=session_id)
        logfire.info("hybrid candidates", dense_count=len(dense), bm25_count=len(bm25))

        fused = reciprocal_rank_fusion([dense, bm25], k=settings.rrf_k)
        reranked = rerank(query, fused)
        logfire.info("reranked", candidate_count=len(reranked))

        final = mmr_select(query, reranked[: settings.rerank_top_n], k=k, lambda_mult=settings.mmr_lambda)
        # MMR selects for relevance + diversity, not a pure score sort — reorder
        # the final set by score so "best match first" is always a safe assumption
        # for callers, instead of only patching the one call site that assumed it.
        final.sort(key=lambda pair: pair[1], reverse=True)
        logfire.info("retrieval complete", result_count=len(final))
        return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a similarity search against the indexed documents")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    for doc, score in retrieve(args.query, k=args.k):
        # score is a cross-encoder rerank logit here, not raw cosine similarity
        print(f"[{score:.4f}] {doc.metadata.get('file_name')} (page {doc.metadata.get('page_number')})")
        print(doc.page_content[:300], "\n")
