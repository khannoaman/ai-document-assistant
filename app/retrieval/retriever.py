import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Any

import logfire
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType

from app.config import settings
from app.models import embedding_model
from app.retrieval.bm25_index import search_bm25
from app.retrieval.fusion import mmr_select, reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.vectorstore.client import get_qdrant_client

logfire.configure(send_to_logfire="if-token-present")

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


def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
    if not filters:
        return None
    return Filter(
        must=[
            FieldCondition(key=f"{_METADATA_PREFIX}{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]
    )


def retrieve(
    query: str,
    k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Hybrid retrieval: fuse dense (Qdrant) + keyword (BM25) candidates via
    Reciprocal Rank Fusion, cross-encoder rerank the fused pool, then select
    the final top-k with MMR for relevance/diversity balance."""
    k = k if k is not None else settings.retrieval_top_k
    fetch_k = settings.retrieval_fetch_k

    with logfire.span("retrieval.search", query=query, k=k):
        client = get_qdrant_client()
        collections = {c.name for c in client.get_collections().collections}
        if settings.collection_name not in collections:
            raise RuntimeError(
                f"Collection '{settings.collection_name}' does not exist yet — "
                "run the indexing pipeline first."
            )

        if filters:
            _ensure_payload_indexes(client, set(filters.keys()))

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embedding_model,
        )

        dense = vectorstore.similarity_search_with_score(query, k=fetch_k, filter=_build_filter(filters))
        bm25 = search_bm25(client, query, fetch_k, filters=filters)
        logfire.info("hybrid candidates", dense_count=len(dense), bm25_count=len(bm25))

        fused = reciprocal_rank_fusion([dense, bm25], k=settings.rrf_k)
        reranked = rerank(query, fused)
        logfire.info("reranked", candidate_count=len(reranked))

        final = mmr_select(query, reranked[: settings.rerank_top_n], k=k, lambda_mult=settings.mmr_lambda)
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
