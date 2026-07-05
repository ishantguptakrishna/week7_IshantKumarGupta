"""
Stage 2: Text Chunking
----------------------
Splits long text into smaller overlapping chunks so retrieval
can find precise, relevant pieces of context instead of whole documents.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def chunk_text(text: str, source: str = "document", chunk_size: int = 500,
               overlap: int = 100) -> list[Chunk]:
    """
    Split text into overlapping word-based chunks.

    chunk_size: approximate number of words per chunk
    overlap: number of words shared between consecutive chunks
             (keeps context from being cut off mid-idea)
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()

        if chunk_str:
            chunks.append(Chunk(text=chunk_str, source=source, chunk_id=chunk_id))
            chunk_id += 1

        if end >= len(words):
            break
        start = end - overlap  # step back to create overlap

    return chunks


def chunk_documents(docs: dict, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Chunk multiple documents. docs = {filename: text}"""
    all_chunks = []
    for source, text in docs.items():
        all_chunks.extend(chunk_text(text, source=source, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
