"""
app/core/text.py

Text normalization shared by BOTH the ingestion pipeline and the query
path. The index and the incoming questions must go through the exact same
normalization, otherwise the same word can produce different tokens
depending on which apostrophe variant (or alphabet) the writer used --
silently hurting retrieval quality.

Uzbek Latin text in the wild mixes several apostrophe-like characters for
the same sounds (oʻ/gʻ and tutuq belgisi). Canonical choice: U+02BB
(MODIFIER LETTER TURNED COMMA) for oʻ/gʻ, U+02BC (MODIFIER LETTER
APOSTROPHE) for tutuq belgisi. These are what the source document already
uses natively, so we normalize *variants* down to these rather than to
plain ASCII apostrophes (which would make oʻ/gʻ indistinguishable from
words that never had a special letter at all).

Queries may also arrive in Uzbek Cyrillic while the indexed document is
Latin, so the query path additionally transliterates Cyrillic to Latin
before normalizing.
"""

from __future__ import annotations

import re
import unicodedata

_APOSTROPHE_VARIANTS = {
    "‘": "ʻ",  # left single quote
    "’": "ʻ",  # right single quote
    "`": "ʻ",  # grave accent
    "´": "ʻ",  # acute accent
    "'": "ʻ",  # ascii apostrophe
}


def normalize_uzbek_text(text: str) -> str:
    """Normalize apostrophe variants and whitespace for consistent tokenization."""
    text = unicodedata.normalize("NFC", text)
    for variant, canonical in _APOSTROPHE_VARIANTS.items():
        text = text.replace(variant, canonical)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Uzbek Cyrillic -> Latin transliteration (2021 official alphabet rules)
# ---------------------------------------------------------------------------

_CYR_MAP = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ш": "sh",
    "ч": "ch",
    "ц": "ts",
    "ё": "yo",
    "ю": "yu",
    "я": "ya",
    "э": "e",
    "ў": "oʻ",
    "ғ": "gʻ",
    "қ": "q",
    "ҳ": "h",
    "ъ": "ʼ",  # tutuq belgisi
    "ь": "",  # soft sign has no Latin counterpart
    # Russian-borrowed letters that appear in loanword spellings
    "щ": "sh",
    "ы": "i",
}

# Vowels (plus ъ/ь) after which Cyrillic "е" reads as "ye" rather than "e".
_YE_TRIGGERS = set("аеёиоуэюяўъь")

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def transliterate_cyrillic_to_latin(text: str) -> str:
    """Transliterates Uzbek Cyrillic to the Latin alphabet used in the index.

    Handles the context-dependent "е": word-initially and after a vowel
    (or ъ/ь) it is "ye", elsewhere plain "e".
    """
    out: list[str] = []
    for i, ch in enumerate(text):
        lower = ch.lower()
        if lower == "е":
            prev = text[i - 1].lower() if i > 0 else ""
            if not prev.isalpha() or prev in _YE_TRIGGERS:
                rep = "ye"
            else:
                rep = "e"
        else:
            rep = _CYR_MAP.get(lower)
            if rep is None:
                out.append(ch)
                continue
        if ch.isupper() and rep:
            rep = rep[0].upper() + rep[1:]
        out.append(rep)
    return "".join(out)


def normalize_query(text: str) -> str:
    """Full normalization for user questions: transliterate Cyrillic input
    to Latin (the alphabet of the indexed document), then apply the same
    apostrophe/whitespace normalization used at indexing time.
    """
    if has_cyrillic(text):
        text = transliterate_cyrillic_to_latin(text)
    return normalize_uzbek_text(text)


# ---------------------------------------------------------------------------
# Answer groundedness detection
# ---------------------------------------------------------------------------

NO_ANSWER_MARKER = "ma'lumot yo'q"
# The prompt tells the model to decline in Cyrillic when the question was
# Cyrillic, and that wording shares no characters with the Latin marker --
# without it a Cyrillic refusal is scored as a grounded answer.
NO_ANSWER_MARKER_CYRILLIC = "маълумот йўқ"


def is_grounded(answer: str) -> bool:
    """True if the LLM gave a substantive answer rather than declining."""
    lowered = answer.lower()
    return (
        NO_ANSWER_MARKER not in lowered and NO_ANSWER_MARKER_CYRILLIC not in lowered
    )


# ---------------------------------------------------------------------------
# Answer shape: did the model restate the question?
# ---------------------------------------------------------------------------

# Question words, conjunctions and determiners carry no topic information --
# an answer that "echoes" only these has not actually restated anything.
_ECHO_STOPWORDS = {
    "qanday", "qancha", "qanchalik", "nima", "nimalar", "necha", "nechta",
    "nechchi", "kim", "kimlar", "qaysi", "qayerda", "qachon", "yoki",
    "bilan", "uchun", "ham", "bu", "shu", "ushbu", "mazkur", "boʻyicha",
    "haqida", "kerak", "mumkin", "boʻladi", "oʻzi", "hamda",
}

_WORD_RE = re.compile(r"[a-zʻʼ]+")

# Uzbek is agglutinative, so the same root turns up with different endings
# ("muddat" / "muddati", "ekspertiza" / "ekspertizasini"). Comparing a fixed
# prefix matches those without dragging in a stemmer.
_STEM_LEN = 6
_MIN_WORD_LEN = 4


def _content_stems(text: str) -> set[str]:
    """Topic-bearing word stems, normalized so Latin and Cyrillic compare."""
    return {
        word[:_STEM_LEN]
        for word in _WORD_RE.findall(normalize_query(text).lower())
        if len(word) >= _MIN_WORD_LEN and word not in _ECHO_STOPWORDS
    }


def echoes_question(question: str, answer: str) -> bool:
    """True if the answer restates at least one substantive word from the
    question, which the prompt requires ("savoldagi asosiy soʻzlarni
    takrorlab", rule 5).

    This catches the fragment case that a bare length check cannot: an
    answer like "Yigirma besh ish kuni ichida" is comfortably longer than
    LLM_MIN_ANSWER_CHARS, yet it answers without restating anything and so
    is not the full sentence the prompt asked for.
    """
    question_stems = _content_stems(question)
    if not question_stems:
        # Nothing substantive to echo -- don't force a pointless rewrite.
        return True
    return bool(question_stems & _content_stems(answer))


# ---------------------------------------------------------------------------
# Capability ("what can you do?") detection
# ---------------------------------------------------------------------------

# Questions about the assistant rather than about the decree. The document
# says nothing about the bot, so these can never clear the retrieval score
# threshold and would otherwise be answered "Hujjatda bu haqida ma'lumot
# yo'q." -- which reads as a malfunction rather than an answer.
#
# Matched against the normalized (Cyrillic-transliterated) form, so the
# Cyrillic spellings of the same questions are covered by the same patterns.
_CAPABILITY_PATTERNS = [
    re.compile(p)
    for p in (
        r"\bnima(lar)?\s+(ish\s+)?qila\s+ol",  # nima qila olasan / olasiz
        r"\bnima\s+ish\s+qil",
        r"\bqanday\s+yordam",  # qanday yordam bera olasiz
        r"\byordam\s+ber\w*\s+(ol|mi)",  # yordam bera olasanmi
        r"\b(sen|siz)\s+kims\w+",  # sen kimsan / siz kimsiz
        r"\bvazifa(ng|ngiz)\b",
        r"\bimkoniyat\w*\s+(nima|qanday)",
        r"\bnima\w*\s+haqida\s+javob\s+ber",
    )
]

# A bare greeting with nothing else in it -- "salom" on its own is not a
# question about the decree, but "salomatlik" or a greeting followed by a
# real question must still go to retrieval.
_GREETING_RE = re.compile(r"^(assalomu\s+alaykum|salom|hayrli\s+(tong|kun|kech))[!.?]*$")


def is_capability_question(text: str) -> bool:
    """True for questions about the assistant itself ("nima qila olasan?")
    or a bare greeting, rather than questions about the decree.
    """
    normalized = normalize_query(text).lower()
    if _GREETING_RE.match(normalized):
        return True
    return any(pattern.search(normalized) for pattern in _CAPABILITY_PATTERNS)
