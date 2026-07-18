from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, autodetect_encoding=True).load()
    for doc in docs:
        doc.metadata["loader_used"] = "text"
    return docs
