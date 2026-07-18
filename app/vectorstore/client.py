from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings


def get_qdrant_client() -> QdrantClient:
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    # Qdrant Cloud also serves the REST API on 443; pin to it instead of the
    # client's 6333 default, since some networks block non-standard ports.
    return QdrantClient(url=settings.qdrant_cluster_endpoint, api_key=api_key, port=settings.qdrant_port)


def ensure_collection(client: QdrantClient, embedding_model) -> None:
    vector_size = len(embedding_model.embed_query("dimension probe"))
    collections = {c.name for c in client.get_collections().collections}

    if settings.collection_name not in collections:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    existing_size = client.get_collection(settings.collection_name).config.params.vectors.size
    if existing_size != vector_size:
        raise ValueError(
            f"Collection '{settings.collection_name}' was built with {existing_size}-dim vectors, "
            f"but the current embedding provider produces {vector_size}-dim vectors. "
            f"Re-index into a new collection or switch providers back."
        )
