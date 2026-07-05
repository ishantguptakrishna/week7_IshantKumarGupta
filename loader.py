"""
Stage 1: Document Ingestion
----------------------------
Loads PDFs or text files and converts them into raw text.
"""

import os
from pypdf import PdfReader


def load_document(file_path: str) -> str:
    """Load a single document (.pdf or .txt) and return its raw text."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext in (".txt", ".md"):
        return _load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .txt, or .md")


def _load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def _load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents_from_folder(folder_path: str) -> dict:
    """Load every supported document in a folder. Returns {filename: text}."""
    docs = {}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".pdf", ".txt", ".md")):
            full_path = os.path.join(folder_path, fname)
            docs[fname] = load_document(full_path)
    return docs
