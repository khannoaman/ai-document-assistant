from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from app.models import embedding_model


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int,
) -> list[tuple[Document, float]]:
    """Fuse multiple ranked result lists into one via Reciprocal Rank Fusion.

    Per-list scores (cosine similarity, BM25) aren't on comparable scales, so
    they're discarded in favor of a rank-based fused score.
    """
    fused: dict[Any, float] = {}
    docs_by_id: dict[Any, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _) in enumerate(ranked_list):
            doc_id = doc.metadata.get("_id")
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            docs_by_id.setdefault(doc_id, doc)

    ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
    return [(docs_by_id[doc_id], score) for doc_id, score in ranked]


def mmr_select(
    query: str,
    candidates: list[tuple[Document, float]],
    k: int,
    lambda_mult: float,
) -> list[tuple[Document, float]]:
    """Select k candidates balancing relevance to `query` and diversity among
    each other, preserving each candidate's incoming (rerank) score."""
    if len(candidates) <= k:
        return candidates

    query_embedding = np.array(embedding_model.embed_query(query))
    doc_embeddings = embedding_model.embed_documents([doc.page_content for doc, _ in candidates])

    selected_idxs = maximal_marginal_relevance(query_embedding, doc_embeddings, lambda_mult=lambda_mult, k=k)
    return [candidates[i] for i in selected_idxs]
