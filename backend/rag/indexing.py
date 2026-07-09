from backend.config.settings import settings
from backend.rag.document_loader import DocumentLoader
from backend.rag.text_splitter import TextSplitter
from backend.rag.vector_store import VectorStore


def index_documents():
    documents = DocumentLoader(str(settings.data_dir)).load_pdf()
    chunks = TextSplitter().split(documents)
    VectorStore(str(settings.vector_db_dir), settings.collection_name).add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} document(s).")


if __name__ == "__main__":
    index_documents()
