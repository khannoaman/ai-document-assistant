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
    """Embed `query` and return the top-k (chunk, similarity_score) pairs
    from the indexed collection, ranked best-first."""
    k = k if k is not None else settings.retrieval_top_k

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

        results = vectorstore.similarity_search_with_score(query, k=k, filter=_build_filter(filters))
        logfire.info("retrieval complete", result_count=len(results))
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a similarity search against the indexed documents")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    for doc, score in retrieve(args.query, k=args.k):
        print(f"[{score:.4f}] {doc.metadata.get('source_file')} (page {doc.metadata.get('page_number')})")
        print(doc.page_content[:300], "\n")
