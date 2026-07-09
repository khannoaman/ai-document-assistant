from langchain_chroma import Chroma

from backend.rag.models import get_embeddings


class VectorStore:
    def __init__(self, directory: str, collection_name: str):
        self.persist_directory = directory
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)

    def reset_db(self):
        self.vector_store.reset_collection()

    def similarity_search(self, query, k=5):
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self):
        return self.vector_store.as_retriever()
