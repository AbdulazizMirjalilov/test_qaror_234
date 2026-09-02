"""
retriever.py

Semantic search over the persisted Chroma collection built by embedder.py.

Includes a similarity score threshold: if the best match for a question is
below the threshold, we don't even bother calling the LLM -- we already know
the document likely doesn't contain the answer. This is more reliable than
trusting the LLM's prompt instructions alone to say "not found".

Queries are passed through the same normalization as the indexed text
(apostrophe variants, whitespace, Cyrillic transliteration) -- see
app/core/text.py for why this matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.utils.text import normalize_query


class IndexNotReadyError(RuntimeError):
    """Raised when the Chroma collection hasn't been built yet."""


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float  # cosine similarity, higher = more relevant


class Retriever:
    def __init__(self):
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
        try:
            self._collection = client.get_collection(settings.COLLECTION_NAME)
        except Exception as exc:
            raise IndexNotReadyError(
                f"Chroma kolleksiyasi '{settings.COLLECTION_NAME}' topilmadi "
                f"({settings.CHROMA_DIR}). Avval ingestion pipeline'ni ishga tushiring:\n"
                "  python -m app.ingestion.loader\n"
                "  python -m app.ingestion.chunker\n"
                "  python -m app.ingestion.embedder"
            ) from exc

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query = normalize_query(query)
        query_embedding = self._model.encode([query], normalize_embeddings=True)[0].tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k or settings.TOP_K,
        )

        chunks = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        ):
            # Chroma with hnsw:space=cosine returns distance = 1 - cosine_similarity
            score = 1 - distance
            chunks.append(RetrievedChunk(text=text, metadata=meta, score=score))
        return chunks

    def search_with_threshold(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Returns [] if even the best match is below threshold."""
        if threshold is None:
            threshold = settings.SCORE_THRESHOLD
        chunks = self.search(query, top_k=top_k)
        if not chunks or chunks[0].score < threshold:
            return []
        return [c for c in chunks if c.score >= threshold]


if __name__ == "__main__":
    import sys

    retriever = Retriever()

    test_questions = [
        "Davlat ekologik ekspertizasi qanday maqsadlarda oʻtkaziladi?",
        "Ekolog-ekspert boʻlish uchun qanday talablar bor?",
        "Aeroportlar uchun ekspertiza oʻtkazish muddati qancha?",
        "Malaka sertifikatini bekor qilish qanday hollarda amalga oshiriladi?",
        # a deliberately unrelated / out-of-scope question
        "Oʻzbekistonda soliq stavkalari qancha?",
    ]

    if len(sys.argv) > 1:
        test_questions = [" ".join(sys.argv[1:])]

    for q in test_questions:
        print("=" * 80)
        print("Q:", q)
        results = retriever.search(q, top_k=3)
        for r in results:
            flag = "OK" if r.score >= settings.SCORE_THRESHOLD else "below threshold"
            print(
                f"  [{r.score:.3f} | {flag}] {r.metadata.get('bob_num')} "
                f"punkt {r.metadata.get('punkt_num')}"
            )
            print("   ", r.text[:150].replace("\n", " "))
