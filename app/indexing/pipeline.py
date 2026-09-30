import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import logfire

from app.config import settings
from app.indexing.chunking import chunk_document
from app.indexing.document_loader import load_document, load_documents
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


def run_indexing_pipeline_for_paths(file_paths: list[str | Path], session_id: str | None = None) -> None:
    """Index only the given files, instead of rescanning the whole data
    directory — used after an upload so re-embedding cost scales with the
    new files, not the full historical corpus. Chunk IDs are deterministic
    from file path + position + content hash, so this is a pure subset of
    what a full run_indexing_pipeline() call would produce, not a divergent
    path — the same chunk gets the same Qdrant point id either way.

    session_id scopes the resulting chunks to the uploading session (see
    app/retrieval/retriever.py) — pass it for web uploads; leave it None
    for CLI/bulk indexing, which is treated as public content."""
    paths = [str(p) for p in file_paths]

    with logfire.span("indexing.pipeline.run_for_paths", file_count=len(paths)):
        docs = [doc for path in paths for doc in load_document(path, session_id=session_id)]
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
