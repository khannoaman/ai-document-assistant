from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.config import settings

# Order = preference: paragraph break, then line break, then sentence break,
# then word break, then character break as a last resort.
generic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)

md_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.MARKDOWN,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
