from pathlib import Path

from langchain_core.documents import Document

_HEADER_KEYS = ("Header 1", "Header 2", "Header 3")


def normalize_metadata(doc: Document, file_path: str, loader_used: str) -> Document:
    raw = doc.metadata
    page = raw.get("page")
    headers = [raw[key] for key in _HEADER_KEYS if key in raw]
    # Markdown's "Header 1"/"Header 2" path takes precedence; other callers
    # (e.g. docx section extraction) can set "section_header" directly instead.
    section_header = " > ".join(headers) if headers else raw.get("section_header")

    doc.metadata = {
        "source_file": file_path,
        "file_type": Path(file_path).suffix.lower().lstrip("."),
        "page_number": page + 1 if isinstance(page, int) else None,
        "slide_number": raw.get("slide_number"),
        "section_header": section_header,
        "loader_used": raw.get("loader_used", loader_used),
    }
    return doc
