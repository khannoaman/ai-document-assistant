from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document


def load(file_path: str) -> list[Document]:

    docs = PDFPlumberLoader(file_path).load()
    for doc in docs:
        doc.metadata["loader_used"] = "pdfplumber"
    return docs

