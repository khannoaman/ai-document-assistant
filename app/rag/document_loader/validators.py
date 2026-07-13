from langchain_core.documents import Document

DEFAULT_MIN_CONTENT_LENGTH = 20


def passes_content_check(doc: Document, min_length: int = DEFAULT_MIN_CONTENT_LENGTH) -> bool:
    return len(doc.page_content.strip()) >= min_length
