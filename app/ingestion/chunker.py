"""
chunker.py

Turns the flat list of blocks from loader.py into final retrieval-ready
chunks.

Logic:
  1. Text blocks: a numbered punkt ("N. ...") starts a new chunk. Any
     following unnumbered ACT_TEXT blocks (sub-items, definitions with "—",
     etc.) are appended to that same chunk until the next numbered punkt
     or a change of bob/ilova.
  2. Table blocks: kept as their own chunk (never merged with text), but
     prefixed with the surrounding ilova/bob context so an isolated table
     chunk still makes sense on its own when retrieved.
  3. Oversized chunks get split further with overlap, so no chunk blows
     past the embedding model's comfortable context.
  4. Every chunk carries metadata for citation and filtering:
     {ilova_num, ilova_title, bob_num, bob_title, punkt_num, source_type}
  5. A synthetic "document identity" chunk is prepended, carrying the
     decree's own requisites (number, date, title, signatory, entry into
     force). Those facts exist in the source only as a bare signature block
     ("Toshkent sh." / "2026-yil 11-may," / "234-son"), which has almost no
     semantic overlap with a natural-language question -- so questions
     *about the document itself* never retrieved it. Everything here is
     extracted from the blocks, never hardcoded, so it follows the document
     when it is replaced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

MAX_CHUNK_CHARS = settings.CHUNK_MAX_CHARS
OVERLAP_CHARS = settings.CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    text: str
    ilova_num: int | None
    ilova_title: str | None
    bob_num: str | None
    bob_title: str | None
    punkt_num: str | None
    source_type: str  # "text" | "table" | "document_identity"
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": {
                "ilova_num": self.ilova_num,
                "ilova_title": self.ilova_title,
                "bob_num": self.bob_num,
                "bob_title": self.bob_title,
                "punkt_num": self.punkt_num,
                "source_type": self.source_type,
            },
        }


# The signature block prints these on their own lines. Matching the whole
# line (not a substring) is deliberate: punkt 6 cites the decree this one
# repeals -- "2020-yil 7-sentabrdagi 541-son qarori" -- and a substring
# search would pick up that number and date instead of this decree's own.
_RE_DECREE_NO = re.compile(r"^(\d+)-son$")
_RE_DECREE_DATE = re.compile(r"^(\d{4}-yil \d{1,2}-[a-z]+)")
_RE_CITY = re.compile(r"^(.+? sh\.)")


def _extract_document_identity(blocks: list[dict]) -> Chunk | None:
    """Builds the document-identity chunk from the decree's own preamble and
    signature block. Returns None if the document doesn't have the expected
    shape, so a differently-formatted source degrades to "no identity chunk"
    rather than breaking ingestion.
    """
    # Requisites live outside any ilova, and the signature lines carry no
    # punkt number -- that pair of conditions isolates them cleanly.
    head = [b for b in blocks if b["ilova_num"] is None]
    plain = [b["text"].strip().rstrip(",") for b in head if b["punkt_num"] is None]

    number = date = city = title = signatory = None
    for text in plain:
        if number is None and (m := _RE_DECREE_NO.match(text)):
            number = m.group(1)
        if date is None and (m := _RE_DECREE_DATE.match(text)):
            date = m.group(1)
        if city is None and (m := _RE_CITY.match(text)):
            city = m.group(1)
        if signatory is None and "Bosh vaziri " in text:
            signatory = text

    # Title is the line right after the standalone "qarori" heading.
    for i, text in enumerate(plain[:6]):
        if text.lower() == "qarori" and i + 1 < len(plain):
            title = plain[i + 1]
            break

    # Entry into force is a numbered punkt, so it is not in `plain`. Its "7. "
    # prefix is dropped -- the number is noise once the sentence is lifted out
    # of the numbered list it belongs to.
    in_force = next(
        (
            b["text"].strip().removeprefix(f"{b['punkt_num']}. ")
            for b in head
            if b["punkt_num"] is not None and "kuchga kiradi" in b["text"]
        ),
        None,
    )
    ilova_count = max((b["ilova_num"] for b in blocks if b["ilova_num"]), default=0)

    if not (number and title):
        return None

    # Phrased as labelled statements ("Qaror raqami: ...") because questions
    # arrive using those same words ("Qaror raqami nechchi?"), while the bare
    # signature block they come from shares none of them.
    lines = [
        "[Hujjat rekvizitlari]",
        "Hujjat turi: Oʻzbekiston Respublikasi Vazirlar Mahkamasining qarori.",
        f"Qaror raqami: {number}-son.",
    ]
    if date:
        lines.append(f"Qaror sanasi: {date}.")
    if city:
        lines.append(f"Qabul qilingan joy: {city}")
    lines.append(f"Qaror nomi: {title}.")
    if signatory:
        lines.append(f"Qarorni imzolagan: {signatory}.")
    if in_force:
        lines.append(f"Kuchga kirishi: {in_force}")
    if ilova_count:
        lines.append(f"Qaror tarkibi: {ilova_count} ta ilova.")
    lines.append("Ushbu maʼlumotlar qarorning oʻzi haqida.")

    return Chunk(
        text="\n".join(lines),
        ilova_num=None,
        ilova_title=None,
        bob_num=None,
        bob_title=None,
        punkt_num=None,
        source_type="document_identity",
    )


def _context_header(ilova_num, ilova_title, bob_num, bob_title) -> str:
    """Builds a short prefix so a chunk is self-contained even out of order."""
    parts = []
    if ilova_num:
        title = f" — {ilova_title}" if ilova_title else ""
        parts.append(f"{ilova_num}-ilova{title}")
    if bob_num:
        title = f" — {bob_title}" if bob_title else ""
        parts.append(f"{bob_num}{title}")
    return " | ".join(parts)


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Splits oversized text on sentence-ish boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            # try to break at the last period/newline before the hard cutoff
            break_point = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if break_point > start:
                end = break_point + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


def _table_rows_to_chunks(markdown_table: str, header: str) -> list[str]:
    """Splits a markdown table into one chunk per data row, each carrying
    the column headers + section context, so a single row (e.g. one
    activity type + its deadline/fee) is independently searchable instead
    of being buried inside a giant multi-row chunk.
    """
    lines = markdown_table.strip().split("\n")
    if len(lines) < 3:
        return [f"[{header}]\n{markdown_table}" if header else markdown_table]

    header_row = lines[0]
    data_rows = lines[2:]  # skip header + separator line

    chunks = []
    for row in data_rows:
        if not row.strip("| ").strip():
            continue
        body = f"{header_row}\n{row}"
        prefix = f"[{header}]\n" if header else ""
        chunks.append(f"{prefix}{body}")
    return chunks if chunks else [f"[{header}]\n{markdown_table}" if header else markdown_table]


def build_chunks(blocks: list[dict]) -> list[dict]:
    chunks: list[Chunk] = []

    # First, so it reads as the document header it describes.
    identity = _extract_document_identity(blocks)
    if identity is not None:
        chunks.append(identity)

    # buffer for the text chunk currently being assembled
    buf_text: list[str] = []
    buf_meta: dict | None = None

    def flush():
        nonlocal buf_text, buf_meta
        if buf_text and buf_meta is not None:
            full_text = "\n".join(buf_text).strip()
            header = _context_header(
                buf_meta["ilova_num"],
                buf_meta["ilova_title"],
                buf_meta["bob_num"],
                buf_meta["bob_title"],
            )
            for piece in _split_long_text(full_text, MAX_CHUNK_CHARS, OVERLAP_CHARS):
                body = f"[{header}]\n{piece}" if header else piece
                chunks.append(
                    Chunk(
                        text=body,
                        ilova_num=buf_meta["ilova_num"],
                        ilova_title=buf_meta["ilova_title"],
                        bob_num=buf_meta["bob_num"],
                        bob_title=buf_meta["bob_title"],
                        punkt_num=buf_meta["punkt_num"],
                        source_type="text",
                    )
                )
        buf_text = []
        buf_meta = None

    for block in blocks:
        if block["type"] == "table":
            flush()  # tables never merge with surrounding text
            header = _context_header(
                block["ilova_num"],
                block["ilova_title"],
                block["bob_num"],
                block["bob_title"],
            )
            for piece in _table_rows_to_chunks(block["text"], header):
                chunks.append(
                    Chunk(
                        text=piece,
                        ilova_num=block["ilova_num"],
                        ilova_title=block["ilova_title"],
                        bob_num=block["bob_num"],
                        bob_title=block["bob_title"],
                        punkt_num=None,
                        source_type="table",
                    )
                )
            continue

        # text block
        starts_new_punkt = block["punkt_num"] is not None
        same_section = (
            buf_meta is not None
            and buf_meta["ilova_num"] == block["ilova_num"]
            and buf_meta["bob_num"] == block["bob_num"]
        )

        if starts_new_punkt or not same_section:
            flush()
            buf_meta = {
                "ilova_num": block["ilova_num"],
                "ilova_title": block["ilova_title"],
                "bob_num": block["bob_num"],
                "bob_title": block["bob_title"],
                "punkt_num": block["punkt_num"],
            }
            buf_text = [block["text"]]
        else:
            buf_text.append(block["text"])

    flush()

    for i, c in enumerate(chunks):
        c.chunk_id = f"chunk_{i:04d}"

    return [c.to_dict() for c in chunks]


if __name__ == "__main__":
    import sys

    # Same reason as loader.py: the preview below prints document text, which
    # a cp1252 console cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    src = sys.argv[1] if len(sys.argv) > 1 else settings.PARSED_BLOCKS_PATH
    blocks = json.loads(Path(src).read_text(encoding="utf-8"))
    chunks = build_chunks(blocks)

    out_path = settings.CHUNKS_PATH
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built {len(chunks)} chunks from {len(blocks)} blocks")
    print(f"Saved to {out_path}")

    lengths = [len(c["text"]) for c in chunks]
    print(f"Avg chunk length: {sum(lengths) // len(lengths)} chars, max: {max(lengths)}")

    for c in chunks[:3]:
        print("---")
        print(c["metadata"])
        print(c["text"][:200])
