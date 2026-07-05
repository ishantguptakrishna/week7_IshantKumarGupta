"""
Stage 3: Embedding Creation
Stage 4: Vector Database
---------------------------
Converts text chunks into vector representations and stores them
for similarity search.

This uses TF-IDF vectors (scikit-learn) so the whole pipeline runs
fully offline with no external model downloads required. This keeps
the "beginner" project simple and dependency-light while still
teaching the real retrieval concept: turn text into vectors, then
find the closest ones with cosine similarity.

Swap `TfidfVectorStore` for a sentence-transformers based version later
(see README) if you want true semantic embeddings instead of keyword-based ones.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = None
        self.chunks = []  # list of Chunk objects, aligned with chunk_vectors rows

    def build(self, chunks: list) -> None:
        """Fit the vectorizer and embed all chunks."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> list[tuple]:
        """
        Embed the query and return the top_k most similar chunks.
        Returns list of (Chunk, similarity_score), best first.
        """
        if self.chunk_vectors is None:
            raise RuntimeError("Vector store is empty. Call build() first.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_vectors).flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
