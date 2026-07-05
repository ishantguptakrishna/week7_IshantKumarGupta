"""
The full RAG pipeline, tying together every stage from the project spec:

1. Document Ingestion   -> loader.py
2. Text Chunking        -> chunker.py
3. Embedding Creation   -> vector_store.py
4. Vector Database      -> vector_store.py
5. Query Processing     -> vector_store.py (search)
6. Context Retrieval    -> vector_store.py (search)
7. Answer Generation    -> generator.py
"""

from loader import load_document, load_documents_from_folder
from chunker import chunk_documents
from vector_store import TfidfVectorStore
from generator import generate_answer


class RAGPipeline:
    def __init__(self, chunk_size: int = 500, overlap: int = 100, top_k: int = 3):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.vector_store = TfidfVectorStore()
        self._built = False

    def ingest_folder(self, folder_path: str) -> None:
        """Load all documents in a folder, chunk them, and build the vector store."""
        docs = load_documents_from_folder(folder_path)
        if not docs:
            raise ValueError(f"No .pdf/.txt/.md files found in {folder_path}")
        self._ingest(docs)

    def ingest_file(self, file_path: str) -> None:
        """Load a single document, chunk it, and build the vector store."""
        text = load_document(file_path)
        docs = {file_path.split("/")[-1]: text}
        self._ingest(docs)

    def _ingest(self, docs: dict) -> None:
        chunks = chunk_documents(docs, chunk_size=self.chunk_size, overlap=self.overlap)
        if not chunks:
            raise ValueError("No text could be extracted from the given document(s).")
        self.vector_store.build(chunks)
        self._built = True
        print(f"Ingested {len(docs)} document(s) into {len(chunks)} chunks.")

    def ask(self, question: str) -> str:
        """Run the full retrieve -> augment -> generate flow for a question."""
        if not self._built:
            raise RuntimeError("No documents ingested yet. Call ingest_file() or ingest_folder() first.")

        retrieved = self.vector_store.search(question, top_k=self.top_k)
        answer = generate_answer(question, retrieved)
        return answer

    def ask_with_sources(self, question: str) -> dict:
        """Like ask(), but also returns the retrieved chunks for transparency/debugging."""
        retrieved = self.vector_store.search(question, top_k=self.top_k)
        answer = generate_answer(question, retrieved)
        return {
            "answer": answer,
            "sources": [
                {"source": c.source, "chunk_id": c.chunk_id, "score": score, "text": c.text}
                for c, score in retrieved
            ],
        }
