from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Order = preference: paragraph break, then line break, then sentence break,
# then word break, then character break as a last resort.
generic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

md_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.MARKDOWN,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
