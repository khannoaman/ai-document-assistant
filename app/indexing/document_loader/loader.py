from pathlib import Path

import logfire
from langchain_core.documents import Document

from app.indexing.document_loader.detection import detect_mime_type
from app.indexing.document_loader.exceptions import EmptyContentError, UnsupportedFileType
from app.indexing.document_loader.loaders import LOADER_REGISTRY
from app.indexing.document_loader.normalize import normalize_metadata
from app.indexing.document_loader.validators import passes_content_check

logfire.configure(send_to_logfire="if-token-present")


def load_document(file_path: str) -> list[Document]:
    with logfire.span("ingestion.load_document", file_path=file_path):
        try:
            mime_type = detect_mime_type(file_path)
            logfire.info("detected mime type", mime_type=mime_type)

            loader_fn = LOADER_REGISTRY.get(mime_type)
            if loader_fn is None:
                raise UnsupportedFileType(file_path, mime_type)

            raw_docs = loader_fn(file_path)
            if not raw_docs:
                raise EmptyContentError(file_path)
        except UnsupportedFileType as e:
            logfire.warn("unsupported file type", file_path=file_path, mime_type=e.mime_type)
            return []
        except EmptyContentError:
            logfire.warn("loader returned no content", file_path=file_path)
            return []
        except Exception as e:
            logfire.error("loader failed", file_path=file_path, error=str(e))
            return []

        validated_docs = []
        for doc in raw_docs:
            if not passes_content_check(doc):
                logfire.warn(
                    "skipped low-content chunk",
                    file_path=file_path,
                    page=doc.metadata.get("page"),
                    content_length=len(doc.page_content.strip()),
                )
                continue
            validated_docs.append(doc)

        if not validated_docs:
            logfire.warn("file yielded no usable content after filtering", file_path=file_path)

        return [normalize_metadata(doc, file_path, mime_type) for doc in validated_docs]


def load_documents(directory: str) -> list[Document]:
    """Walks a directory and loads every file it finds. A single bad or
    unsupported file is logged and skipped, not fatal to the batch.
    """
    documents: list[Document] = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            documents.extend(load_document(str(path)))
    return documents
