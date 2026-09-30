import re
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Any

import logfire
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.generation.query_condenser import condense_question
from app.models import get_chat_model
from app.retrieval import retrieve

logfire.configure(send_to_logfire="if-token-present")

_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Cite sources using bracketed numeric markers in the exact form [n] (e.g. [1], [2]), "
    "matching the [n] markers shown in the context. "
    "Do not use any other citation style — for example, never use 【n†...】 or footnote-style citations. "
    "If the context doesn't contain the answer, say so plainly."
)

_NO_CONTEXT_ANSWER = "I don't have relevant information in the indexed documents to answer that."

_CITATION_RE = re.compile(r"\[(\d+)\]")

# Some models (e.g. gpt-oss, which is trained on OpenAI's own product
# conventions) default to OpenAI's file/browsing citation markup — like
# `【2†L1-L4】` — instead of the `[n]` format requested above. Rewrite it to
# `[n]` so both the displayed answer and _build_sources() below treat it the
# same as a well-behaved response, rather than silently falling back to
# "show every retrieved source" whenever a model ignores the instruction.
_OPENAI_STYLE_CITATION_RE = re.compile(r"【\s*(\d+)[^\】]*】")


def _normalize_citations(answer: str) -> str:
    return _OPENAI_STYLE_CITATION_RE.sub(lambda m: f"[{m.group(1)}]", answer)


def _format_context(results: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, _) in enumerate(results, start=1):
        source = doc.metadata.get("file_name")
        position = doc.metadata.get("page_number") or doc.metadata.get("slide_number")
        location = f" (page {position})" if position else ""
        blocks.append(f"[{i}] Source: {source}{location}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _build_sources(results: list[tuple[Document, float]], answer: str) -> list[dict[str, Any]]:
    """Number each source to match the [n] marker _format_context gave it, then
    keep only the ones the answer actually cites — retrieval always returns the
    full top-k regardless of how many the model ends up using."""
    all_sources = [
        {
            "citation_number": i,
            "file_name": doc.metadata.get("file_name"),
            "page_number": doc.metadata.get("page_number"),
            "slide_number": doc.metadata.get("slide_number"),
            "score": score,
        }
        for i, (doc, score) in enumerate(results, start=1)
    ]

    cited_numbers = {int(n) for n in _CITATION_RE.findall(answer)}
    if not cited_numbers:
        # model didn't use [n] markers at all — show everything retrieved
        # rather than silently hiding sources for an otherwise grounded answer
        return all_sources
    return [s for s in all_sources if s["citation_number"] in cited_numbers]


def generate_answer(
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
    k: int | None = None,
    filters: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Condense `question` against `chat_history` (if any), retrieve context
    via app.retrieval.retrieve() and synthesize a grounded answer. Returns
    {"answer": str, "sources": list[dict], "context_used": bool, "standalone_question": str}.

    session_id scopes retrieval to that session's own uploads plus public
    content — see app.retrieval.retriever.retrieve()."""
    with logfire.span("generation.answer", question=question):
        standalone_question = condense_question(chat_history or [], question)

        results = retrieve(standalone_question, k=k, filters=filters, session_id=session_id)

        # results are MMR-ordered (relevance + diversity), not sorted by
        # score, so the gate must check the best score present, not results[0]
        if not results or max(score for _, score in results) < settings.min_rerank_score:
            logfire.info("no sufficiently relevant context found")
            return {
                "answer": _NO_CONTEXT_ANSWER,
                "sources": [],
                "context_used": False,
                "standalone_question": standalone_question,
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )
        messages = prompt.format_messages(context=_format_context(results), question=standalone_question)

        response = get_chat_model().invoke(messages)
        answer = _normalize_citations(response.content)
        sources = _build_sources(results, answer)
        logfire.info("generated answer", source_count=len(sources), retrieved_count=len(results))
        return {
            "answer": answer,
            "sources": sources,
            "context_used": True,
            "standalone_question": standalone_question,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer a question using retrieved context (RAG)")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="Prior turn as 'question::answer'; repeatable, oldest first",
    )
    args = parser.parse_args()

    chat_history = [tuple(h.split("::", 1)) for h in args.history]
    result = generate_answer(args.question, chat_history=chat_history, k=args.k)
    if result.get("standalone_question") != args.question:
        print(f"(standalone query: {result['standalone_question']})")
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"- [{s['citation_number']}] {s['file_name']} (page {s.get('page_number')}, score {s['score']:.4f})")
