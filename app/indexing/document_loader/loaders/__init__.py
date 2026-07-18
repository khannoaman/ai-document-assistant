from app.indexing.document_loader.loaders import docx_loader, pdf_loader, pptx_loader, text_loader

# Keyed by MIME type (via python-magic content sniffing), not file extension.
LOADER_REGISTRY = {
    "application/pdf": pdf_loader.load,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": docx_loader.load,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": pptx_loader.load,
    # .md sniffs identically to .txt; both load flat here, structure-aware
    # splitting (if any) is chunking's concern, not the loader's.
    "text/plain": text_loader.load,
}
