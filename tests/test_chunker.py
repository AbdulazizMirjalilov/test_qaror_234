from app.ingestion.chunker import _split_long_text, _table_rows_to_chunks, build_chunks


def text_block(text, punkt_num=None, ilova_num=1, bob_num="1-bob"):
    return {
        "type": "text",
        "text": text,
        "ilova_num": ilova_num,
        "ilova_title": "Test nizom",
        "bob_num": bob_num,
        "bob_title": "Umumiy qoidalar",
        "punkt_num": punkt_num,
    }


def test_numbered_punkt_starts_new_chunk():
    blocks = [
        text_block("1. Birinchi band.", punkt_num="1"),
        text_block("qoʻshimcha izoh;"),
        text_block("2. Ikkinchi band.", punkt_num="2"),
    ]
    chunks = build_chunks(blocks)
    assert len(chunks) == 2
    assert "Birinchi band" in chunks[0]["text"]
    assert "qoʻshimcha izoh" in chunks[0]["text"]
    assert chunks[0]["metadata"]["punkt_num"] == "1"
    assert chunks[1]["metadata"]["punkt_num"] == "2"


def test_section_change_flushes_buffer():
    blocks = [
        text_block("1. Birinchi bob matni.", punkt_num="1", bob_num="1-bob"),
        text_block("davomi boshqa bobda", bob_num="2-bob"),
    ]
    chunks = build_chunks(blocks)
    assert len(chunks) == 2
    assert chunks[1]["metadata"]["bob_num"] == "2-bob"


def test_ilova_change_forces_new_chunk_even_without_punkt_number():
    blocks = [
        text_block("1. Birinchi band.", punkt_num="1", ilova_num=2, bob_num="1-bob"),
        text_block("Ilova B matni.", ilova_num=3, bob_num=None),
    ]
    chunks = build_chunks(blocks)
    assert len(chunks) == 2
    assert chunks[1]["metadata"]["ilova_num"] == 3


def test_context_header_prefixed():
    chunks = build_chunks([text_block("1. Matn.", punkt_num="1")])
    assert chunks[0]["text"].startswith("[1-ilova")


def test_table_rows_become_individual_chunks():
    table = (
        "| T/r | Obyekt | Muddat |\n|---|---|---|\n| 1. | Yoʻllar | 10 |\n| 2. | Aeroportlar | 25 |"
    )
    blocks = [
        {
            "type": "table",
            "text": table,
            "ilova_num": 1,
            "ilova_title": "Roʻyxat",
            "bob_num": None,
            "bob_title": None,
            "punkt_num": None,
        }
    ]
    chunks = build_chunks(blocks)
    assert len(chunks) == 2
    # each row chunk repeats the column headers so it's self-contained
    for c in chunks:
        assert "T/r" in c["text"]
        assert c["metadata"]["source_type"] == "table"
    assert "Aeroportlar" in chunks[1]["text"]


def test_table_never_merges_with_surrounding_text():
    blocks = [
        text_block("1. Kirish matni.", punkt_num="1", bob_num=None),
        {
            "type": "table",
            "text": "| Nomi | Muddat |\n|---|---|\n| Obyekt A | 25 kun |",
            "ilova_num": 1,
            "ilova_title": "Test nizom",
            "bob_num": None,
            "bob_title": None,
        },
        text_block("2. Davomi.", punkt_num="2", bob_num=None),
    ]
    chunks = build_chunks(blocks)
    table_chunks = [c for c in chunks if c["metadata"]["source_type"] == "table"]
    assert len(table_chunks) == 1
    assert "Kirish matni" not in table_chunks[0]["text"]
    assert "Davomi" not in table_chunks[0]["text"]


def test_table_rows_to_chunks_small_table_kept_whole():
    md = "| faqat | bitta | qator |"
    result = _table_rows_to_chunks(md, header="")
    assert result == [md]


def test_split_long_text_respects_limit_and_overlaps():
    sentence = "Bu yetarlicha uzun gap boʻlib sinov uchun yozildi. "
    text = sentence * 60  # ~3000 chars
    parts = _split_long_text(text, max_chars=1000, overlap=100)
    assert len(parts) > 1
    assert all(len(p) <= 1000 for p in parts)
    # overlap: consecutive parts share content
    assert parts[0][-50:].strip() != ""


def test_chunk_ids_sequential():
    blocks = [text_block(f"{i}. Matn.", punkt_num=str(i)) for i in range(1, 4)]
    chunks = build_chunks(blocks)
    assert [c["chunk_id"] for c in chunks] == ["chunk_0000", "chunk_0001", "chunk_0002"]
