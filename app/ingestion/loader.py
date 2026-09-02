"""
loader.py

Parses a lex.uz-style HTML export (saved with .doc extension) of a resolution
document into a flat, ordered list of structured blocks.

The source HTML uses semantic CSS classes we rely on instead of guessing via
raw regex on plain text:

  ACT_TITLE / ACT_FORM / ACCEPTING_BODY / SIGNATURE  -> document preamble
  ACT_TEXT                                            -> a band ("1."-numbered or sub-item)
  TEXT_HEADER_DEFAULT                                 -> "N-bob. <title>" chapter header
  APPL_BANNER_LANDSCAPE_TITLE                         -> "... N-ILOVA" appendix banner
  ACT_TITLE_APPL                                      -> title of the appendix / nizom
  TABLE_STD2                                          -> a <table> element

Each yielded block is a dict:
  {
      "type": "text" | "table",
      "ilova_num": int | None,       # current appendix number (1..9)
      "ilova_title": str | None,     # title of current appendix/nizom
      "bob_num": str | None,         # e.g. "1-bob"
      "bob_title": str | None,       # e.g. "Umumiy qoidalar"
      "punkt_num": str | None,       # e.g. "3" (only set when a block starts a numbered punkt)
      "text": str,                   # normalized text (or markdown for tables)
  }

Consecutive unnumbered ACT_TEXT blocks are NOT merged here -- that's the
chunker's job (Step 2). This module's only responsibility is faithful,
ordered extraction with correct hierarchy metadata attached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

# Normalization is shared with the query path (app/core/text.py) so the
# index and incoming questions always agree on apostrophe variants etc.
from app.utils.text import normalize_uzbek_text

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Block:
    type: str  # "text" | "table"
    text: str
    ilova_num: int | None = None
    ilova_title: str | None = None
    bob_num: str | None = None
    bob_title: str | None = None
    punkt_num: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "text": self.text,
            "ilova_num": self.ilova_num,
            "ilova_title": self.ilova_title,
            "bob_num": self.bob_num,
            "bob_title": self.bob_title,
            "punkt_num": self.punkt_num,
        }


# ---------------------------------------------------------------------------
# Table -> Markdown
# ---------------------------------------------------------------------------


def table_to_markdown(table_tag: Tag) -> str:
    """Convert a simple <table><tr><td> structure into a GitHub-flavored markdown table."""
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        row = [normalize_uzbek_text(c.get_text(" ", strip=True)) or " " for c in cells]
        if any(cell.strip() for cell in row):
            rows.append(row)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |"]
    lines.append("|" + "|".join(["---"] * width) + "|")
    for row in rows[1:]:
        lines.append("| " + " | ".join(esc(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

_ILOVA_NUM_RE = re.compile(r"(\d+)-ILOVA", re.IGNORECASE)
_BOB_RE = re.compile(r"^(\d+-bob)\.\s*(.*)$", re.IGNORECASE)
_PUNKT_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def _find_content_root(soup: BeautifulSoup) -> Tag:
    """Descend through wrapper <div>s down to the level holding the actual content blocks."""
    node = soup.body
    while True:
        children = [c for c in node.children if getattr(c, "name", None)]
        if len(children) == 1 and children[0].name == "div":
            node = children[0]
            continue
        return node


def parse_doc(path: str | Path) -> list[dict]:
    """Parse the lex.uz-exported .doc file into an ordered list of block dicts."""
    path = Path(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    root = _find_content_root(soup)

    blocks: list[Block] = []

    # Hierarchy state, updated as we walk through the document in order.
    current_ilova_num: int | None = None
    current_ilova_title: str | None = None
    current_bob_num: str | None = None
    current_bob_title: str | None = None

    for node in root.children:
        if not getattr(node, "name", None):
            continue

        css_class = (node.get("class") or [None])[0]
        raw_text = node.get_text(" ", strip=True)
        text = normalize_uzbek_text(raw_text)
        if not text and css_class != "TABLE_STD2":
            continue

        if css_class == "APPL_BANNER_LANDSCAPE_TITLE":
            m = _ILOVA_NUM_RE.search(text)
            if m:
                current_ilova_num = int(m.group(1))
                current_ilova_title = None  # title comes next, in ACT_TITLE_APPL
                current_bob_num = None
                current_bob_title = None
            continue

        if css_class == "ACT_TITLE_APPL":
            # Only take the first title seen right after an ilova banner --
            # nested appendices-within-a-nizom repeat this class too, but by
            # then current_ilova_title is already set for the outer nizom,
            # so we only overwrite while it's still empty.
            if current_ilova_title is None:
                current_ilova_title = text
            continue

        if css_class == "TEXT_HEADER_DEFAULT":
            m = _BOB_RE.match(text)
            if m:
                current_bob_num, current_bob_title = m.group(1), m.group(2)
            else:
                current_bob_num, current_bob_title = None, text
            continue

        if css_class == "TABLE_STD2":
            table_tag = node.find("table")
            if table_tag is None:
                continue
            md = table_to_markdown(table_tag)
            if not md:
                continue
            blocks.append(
                Block(
                    type="table",
                    text=md,
                    ilova_num=current_ilova_num,
                    ilova_title=current_ilova_title,
                    bob_num=current_bob_num,
                    bob_title=current_bob_title,
                )
            )
            continue

        if css_class == "ACT_TEXT":
            m = _PUNKT_RE.match(text)
            punkt_num = m.group(1) if m else None
            blocks.append(
                Block(
                    type="text",
                    text=text,
                    ilova_num=current_ilova_num,
                    ilova_title=current_ilova_title,
                    bob_num=current_bob_num,
                    bob_title=current_bob_title,
                    punkt_num=punkt_num,
                )
            )
            continue

        # Preamble / metadata classes (ACT_TITLE, ACT_FORM, ACCEPTING_BODY,
        # SIGNATURE, PUBLICATION_ORIGIN, FOOTNOTE, TEXT_CENTER, BY_DEFAULT, ...)
        # are captured as plain "text" blocks with no hierarchy, so nothing
        # from the source is silently dropped.
        if text:
            blocks.append(Block(type="text", text=text))

    return [b.to_dict() for b in blocks]


if __name__ == "__main__":
    import json
    import sys

    from app.core.config import settings

    # Windows consoles default to a legacy codepage (cp1252) that can't encode
    # Uzbek characters like U+02BB -- force UTF-8 output. Without this the
    # sanity preview below crashes *after* the parse succeeded, which aborts
    # the `make ingest` chain before the chunker and embedder ever run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    src = sys.argv[1] if len(sys.argv) > 1 else settings.SOURCE_DOC_PATH
    result = parse_doc(src)
    print(f"Parsed {len(result)} blocks")
    out_path = settings.PARSED_BLOCKS_PATH
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")

    # quick sanity preview
    for b in result[:5]:
        print(b)
