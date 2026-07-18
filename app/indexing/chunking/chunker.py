import logfire
from langchain_core.documents import Document

from app.indexing.chunking.splitters import generic_splitter, md_splitter

logfire.configure(send_to_logfire="if-token-present")


def chunk_document(docs: list[Document]) -> list[Document]:
    final_chunks: list[Document] = []
    with logfire.span("chunking.chunk_document", doc_count=len(docs)):
        for doc in docs:
            splitter = md_splitter if doc.metadata.get("file_type") == "md" else generic_splitter
            final_chunks.extend(splitter.split_documents([doc]))

        logfire.info("chunking complete", chunk_count=len(final_chunks))

    return final_chunks
