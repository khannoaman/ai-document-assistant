from pathlib import Path

from langchain_core.documents import Document



def normalize_metadata(doc: Document, file_path: str, loader_used: str) -> Document:
    raw = doc.metadata
    page = raw.get("page")

    doc.metadata = {
        "file_name": Path(file_path).name,
        "file_path": file_path,
        "file_type": Path(file_path).suffix.lower().lstrip("."),
        "page_number": page + 1 if isinstance(page, int) else None,
        "slide_number": raw.get("slide_number"),
        "loader_used": raw.get("loader_used", loader_used),
    }
    return doc
