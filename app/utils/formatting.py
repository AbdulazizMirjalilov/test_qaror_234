"""
app/core/formatting.py

Citation formatting shared by the API and scripts. The source citation is
built programmatically from retrieval metadata -- we deliberately do NOT
trust the LLM to name its own source (it can hallucinate one).
"""

from __future__ import annotations


def format_source(metadata: dict) -> str:
    """Formats chunk metadata into a human-readable citation like
    '7-ilova, 9-bob, 43-band'."""
    # The identity chunk describes the decree as a whole, so it has no
    # ilova/bob/band to cite -- without this it would render as "".
    if metadata.get("source_type") == "document_identity":
        return "Qaror rekvizitlari"

    parts = []
    if metadata.get("ilova_num"):
        parts.append(f"{metadata['ilova_num']}-ilova")
    if metadata.get("bob_num"):
        parts.append(metadata["bob_num"])
    if metadata.get("punkt_num"):
        parts.append(f"{metadata['punkt_num']}-band")
    return ", ".join(parts)
