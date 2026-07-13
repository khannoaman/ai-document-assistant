class DocumentLoaderError(Exception):
    """Base exception for the document_loader package."""


class UnsupportedFileType(DocumentLoaderError):
    def __init__(self, file_path: str, mime_type: str):
        super().__init__(f"Unsupported file type {mime_type!r} for {file_path}")
        self.file_path = file_path
        self.mime_type = mime_type


class EmptyContentError(DocumentLoaderError):
    def __init__(self, file_path: str):
        super().__init__(f"No content extracted from {file_path}")
        self.file_path = file_path
