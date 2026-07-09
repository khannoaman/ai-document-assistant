from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


class DocumentLoader:
    def __init__(self, directory: str):
        self.pdf_loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)

    def load_pdf(self):
        return self.pdf_loader.load()

    def lazy_load_pdf(self):
        return self.pdf_loader.lazy_load()


if __name__ == "__main__":
    from backend.config.settings import settings

    loader = DocumentLoader(str(settings.data_dir))
    documents = loader.load_pdf()
    print(len(documents))
    print(documents[0])
    print(documents[-1])
