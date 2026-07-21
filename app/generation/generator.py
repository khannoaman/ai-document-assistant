import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Any

import logfire
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.models import get_chat_model
from app.retrieval import retrieve

logfire.configure(send_to_logfire="if-token-present")

_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Cite sources using the [n] markers shown in the context. "
    "If the context doesn't contain the answer, say so plainly."
)

_NO_CONTEXT_ANSWER = "I don't have relevant information in the indexed documents to answer that."


def _format_context(results: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, _) in enumerate(results, start=1):
        source = doc.metadata.get("source_file")
        position = doc.metadata.get("page_number") or doc.metadata.get("slide_number")
        location = f" (page {position})" if position else ""
        blocks.append(f"[{i}] Source: {source}{location}\n{doc.page_content}")
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve context via app.retrieval.retrieve() and synthesize a
    grounded answer. Returns {"answer": str, "sources": list[dict], "context_used": bool}."""
    with logfire.span("generation.answer", question=question):
        results = retrieve(question, k=k, filters=filters)

        if not results or results[0][1] < settings.min_relevance_score:
            logfire.info("no sufficiently relevant context found")
            return {"answer": _NO_CONTEXT_ANSWER, "sources": [], "context_used": False}

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )
        messages = prompt.format_messages(context=_format_context(results), question=question)

        response = get_chat_model().invoke(messages)
        sources = [
            {
                "source_file": doc.metadata.get("source_file"),
                "page_number": doc.metadata.get("page_number"),
                "slide_number": doc.metadata.get("slide_number"),
                "score": score,
            }
            for doc, score in results
        ]
        logfire.info("generated answer", source_count=len(sources))
        return {"answer": response.content, "sources": sources, "context_used": True}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer a question using retrieved context (RAG)")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    result = generate_answer(args.question, k=args.k)
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"- {s['source_file']} (page {s.get('page_number')}, score {s['score']:.4f})")
