"""
embedder.py

Embeds chunks (produced by chunker.py) using BGE-M3 (multilingual, strong
on low-resource languages incl. Uzbek) and stores them in a persistent
ChromaDB collection.

This is an OFFLINE step -- run once (or whenever the source document
changes), not on every API request. The FastAPI app will just open the
persisted Chroma collection and query it.
"""

from __future__ import annotations

import json

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings


def load_chunks() -> list[dict]:
    return json.loads(settings.CHUNKS_PATH.read_text(encoding="utf-8"))


def flatten_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool/None -- no nested dicts.
    None values also aren't reliably queryable, so we coerce them to safe defaults.
    """
    out = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        else:
            out[k] = v
    return out


def build_index():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {settings.EMBEDDING_MODEL} (first run downloads it)...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )

    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))

    # start clean each time this script runs, to avoid duplicate/stale entries
    try:
        client.delete_collection(settings.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[flatten_metadata(c["metadata"]) for c in chunks],
    )

    print(
        f"Indexed {collection.count()} chunks into '{settings.COLLECTION_NAME}' "
        f"at {settings.CHROMA_DIR}"
    )


if __name__ == "__main__":
    build_index()
