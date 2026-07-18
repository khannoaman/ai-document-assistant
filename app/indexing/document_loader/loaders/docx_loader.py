from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document


def load(file_path: str) -> list[Document]:
    docs = Docx2txtLoader(file_path).load()
    for doc in docs:
        doc.metadata["loader_used"] = "docx2txt"
    return docs
