from langchain_core.documents import Document

from app.config import settings


def passes_content_check(doc: Document, min_length: int = settings.min_content_length) -> bool:
    return len(doc.page_content.strip()) >= min_length
