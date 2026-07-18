import magic


def detect_mime_type(file_path: str) -> str:
    """
    Detect the MIME (Multipurpose Internet Mail Extensions) type of a file based on its contents.

    Unlike checking a file's extension, this function inspects the file
    itself to determine its actual format (e.g., "application/pdf",
    "image/jpeg", or "text/plain").

    Args:
        file_path: Path to the file whose MIME type should be detected.

    Returns:
        The detected MIME type as a string.
    """
    return magic.from_file(file_path, mime=True)

