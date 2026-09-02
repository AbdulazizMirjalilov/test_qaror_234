"""Shared fixtures. The API tests run against the real FastAPI app but with
a fake retriever injected via app.main._build_retriever, so no embedding
model, torch, or Chroma index is needed to run the suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient


@dataclass
class FakeChunk:
    text: str
    metadata: dict
    score: float


class FakeRetriever:
    """Returns a canned hit for questions containing 'aeroport', nothing
    otherwise -- enough to exercise both branches of /ask."""

    def __init__(self):
        self.last_query = None

    def count(self) -> int:
        return 42

    def search_with_threshold(self, query, top_k=None, threshold=None):
        self.last_query = query
        if "aeroport" in query.lower():
            return [
                FakeChunk(
                    text="| T/r | Faoliyat | Muddat | \n| 2. | Aeroportlar. | 25 |",
                    metadata={"ilova_num": 1, "bob_num": "", "punkt_num": ""},
                    score=0.71,
                )
            ]
        return []


@pytest.fixture
def fake_retriever():
    return FakeRetriever()


@pytest.fixture
def client(monkeypatch, fake_retriever):
    from app import main

    monkeypatch.setattr(main, "_build_retriever", lambda: fake_retriever)
    app = main.create_app()
    with TestClient(app) as test_client:
        yield test_client
