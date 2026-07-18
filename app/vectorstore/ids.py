import hashlib
import uuid

from langchain_core.documents import Document

# Fixed namespace so the same (source, position, content) always maps to the
# same UUID across runs and machines. Qdrant point IDs must be an unsigned
# int or a UUID string, so a raw hex digest won't do.
_ID_NAMESPACE = uuid.UUID("d6e2b1c0-9a3e-4b8f-8f7a-1e6c2a9d4b0e")


def generate_chunk_id(chunk: Document) -> str:
    """Deterministic point ID from source + position + content hash.

    Re-ingesting unchanged content overwrites the same point instead of
    duplicating it; changed content at the same position gets a new ID
    rather than silently keeping stale data.
    """
    source = chunk.metadata.get("source_file", "")
    position = (
        chunk.metadata.get("page_number")
        or chunk.metadata.get("slide_number")
        or chunk.metadata.get("section_header")
        or ""
    )
    content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
    raw_id = f"{source}:{position}:{content_hash}"
    return str(uuid.uuid5(_ID_NAMESPACE, raw_id))
