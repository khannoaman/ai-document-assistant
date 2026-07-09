from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def split(self, document):
        return self.text_splitter.split_documents(document)


if __name__ == "__main__":
    from backend.config.settings import settings
    from backend.rag.document_loader import DocumentLoader

    documents = DocumentLoader(str(settings.data_dir)).load_pdf()
    chunks = TextSplitter().split(documents)
    print(len(chunks))
    print(chunks[0])
    print(chunks[-1])
