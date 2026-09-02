"""
Tests for app/services/retriever.py

Focus: the score threshold logic -- the actual embedding model and Chroma
DB are mocked out, so this test doesn't need Ollama/Chroma running and
stays fast and deterministic.

Importing the retriever module still pulls in chromadb/sentence-transformers,
which the lightweight dev environment (requirements-dev.txt) doesn't install,
so the whole module is skipped there and runs only on a full setup.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from app.services.retriever import RetrievedChunk, Retriever


def make_retriever_with_mocked_search(scores):
    """Builds a Retriever instance and monkeypatches its .search() to
    return fake RetrievedChunks with the given scores, without touching
    the real embedding model or ChromaDB.
    """
    with patch.object(Retriever, "__init__", lambda self: None):
        retriever = Retriever()

    def fake_search(query, top_k=None):
        return [
            RetrievedChunk(text=f"chunk {i}", metadata={}, score=s) for i, s in enumerate(scores)
        ]

    retriever.search = fake_search
    return retriever


def test_below_threshold_returns_empty():
    retriever = make_retriever_with_mocked_search(scores=[0.40, 0.35, 0.30])
    result = retriever.search_with_threshold("irrelevant question", threshold=0.55)
    assert result == []


def test_above_threshold_returns_chunks():
    retriever = make_retriever_with_mocked_search(scores=[0.70, 0.60, 0.50])
    result = retriever.search_with_threshold("relevant question", threshold=0.55)
    # only chunks >= threshold should be kept
    assert len(result) == 2
    assert all(c.score >= 0.55 for c in result)


def test_top_score_exactly_at_threshold_is_included():
    retriever = make_retriever_with_mocked_search(scores=[0.55])
    result = retriever.search_with_threshold("boundary question", threshold=0.55)
    assert len(result) == 1
