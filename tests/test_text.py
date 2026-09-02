from app.utils.text import (
    has_cyrillic,
    is_grounded,
    normalize_query,
    normalize_uzbek_text,
    transliterate_cyrillic_to_latin,
)

# Canonical characters used by the source document / index.
TURNED_COMMA = "ʻ"  # oʻ / gʻ
TUTUQ = "ʼ"  # tutuq belgisi


def test_ascii_apostrophe_normalized_to_turned_comma():
    assert normalize_uzbek_text("o'tkazish") == f"o{TURNED_COMMA}tkazish"


def test_typographic_quotes_normalized():
    assert normalize_uzbek_text("o’tkazish") == f"o{TURNED_COMMA}tkazish"
    assert normalize_uzbek_text("g‘oya") == f"g{TURNED_COMMA}oya"


def test_canonical_form_unchanged():
    text = f"o{TURNED_COMMA}tkazish ma{TUTUQ}lumot"
    assert normalize_uzbek_text(text) == text


def test_whitespace_collapsed():
    assert normalize_uzbek_text("bir   ikki\t uch") == "bir ikki uch"


def test_has_cyrillic():
    assert has_cyrillic("экспертиза")
    assert not has_cyrillic("ekspertiza")


def test_transliteration_basic_words():
    assert transliterate_cyrillic_to_latin("экспертиза") == "ekspertiza"
    assert transliterate_cyrillic_to_latin("аэропорт") == "aeroport"


def test_transliteration_special_letters():
    assert transliterate_cyrillic_to_latin("ўтказиш") == f"o{TURNED_COMMA}tkazish"
    assert transliterate_cyrillic_to_latin("ғоя") == f"g{TURNED_COMMA}oya"
    assert transliterate_cyrillic_to_latin("қарор") == "qaror"
    assert transliterate_cyrillic_to_latin("ҳужжат") == "hujjat"
    assert transliterate_cyrillic_to_latin("шартнома") == "shartnoma"
    assert transliterate_cyrillic_to_latin("чегара") == "chegara"


def test_transliteration_context_dependent_ye():
    # word-initial and post-vowel "е" -> "ye"; after a consonant -> "e"
    assert transliterate_cyrillic_to_latin("ер") == "yer"
    assert transliterate_cyrillic_to_latin("бекор") == "bekor"


def test_transliteration_soft_sign_dropped():
    assert transliterate_cyrillic_to_latin("сентябрь") == "sentyabr"


def test_transliteration_preserves_case_and_punctuation():
    assert transliterate_cyrillic_to_latin("Ўзбекистон?") == f"O{TURNED_COMMA}zbekiston?"


def test_normalize_query_handles_both_alphabets():
    # Cyrillic query lands in the same canonical form as its Latin twin
    assert normalize_query("ўтказиш") == normalize_query("o'tkazish")


def test_grounded_answer_detected():
    assert is_grounded("Ekolog-ekspert boʻlish uchun oliy maʼlumot talab qilinadi.") is True


def test_declined_answer_detected():
    assert is_grounded("Hujjatda bu haqida ma'lumot yo'q.") is False


def test_case_insensitive_marker_detection():
    assert is_grounded("HUJJATDA BU HAQIDA MA'LUMOT YO'Q.") is False
