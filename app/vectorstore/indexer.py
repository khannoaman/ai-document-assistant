import logfire
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from app.config import settings
from app.models import embedding_model
from app.vectorstore.client import ensure_collection, get_qdrant_client
from app.vectorstore.ids import generate_chunk_id

logfire.configure(send_to_logfire="if-token-present")


def index_documents(chunks: list[Document]) -> None:
    with logfire.span("indexing.index_documents", chunk_count=len(chunks)):
        if not chunks:
            logfire.warn("index_documents called with no chunks")
            return

        client = get_qdrant_client()
        ensure_collection(client, embedding_model)

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embedding_model,
        )
        ids = [generate_chunk_id(chunk) for chunk in chunks]

        try:
            vectorstore.add_documents(chunks, ids=ids)
            logfire.info("indexed chunks", count=len(chunks))
        except Exception as e:
            logfire.error("indexing failed", error=str(e))
            raise
