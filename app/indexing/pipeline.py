from pathlib import Path

import logfire

from app.config import settings
from app.indexing.chunking import chunk_document
from app.indexing.document_loader import load_documents
from app.vectorstore.indexer import index_documents

logfire.configure(send_to_logfire="if-token-present")


def run_indexing_pipeline(directory: str | Path | None = None) -> None:
    directory = str(directory) if directory else str(settings.data_dir)

    with logfire.span("indexing.pipeline.run", directory=directory):
        docs = load_documents(directory)
        logfire.info("loaded documents", doc_count=len(docs))

        chunks = chunk_document(docs)
        logfire.info("chunked documents", chunk_count=len(chunks))

        index_documents(chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the document indexing pipeline")
    parser.add_argument("--directory", default=None, help="Directory to index (defaults to settings.data_dir)")
    args = parser.parse_args()

    run_indexing_pipeline(args.directory)
