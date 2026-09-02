"""
prompt_templates.py

System prompt enforces strict grounding in the retrieved context only.
This is the second line of defense against hallucination (the first being
the similarity score threshold in retriever.py).
"""

SYSTEM_PROMPT = """Sen 234-sonli Vazirlar Mahkamasi qarori matni boʻyicha savollarga javob beruvchi yordamchisan.

QATʼIY QOIDALAR:
1. Faqat quyida keltirilgan "KONTEKST" boʻlimidagi maʼlumotlardan foydalanib javob ber.
2. MUHIM — ALIFBO: SAVOL qaysi alifboda yozilgan boʻlsa, javob ham toʻliq oʻsha alifboda boʻlsin: lotincha savolga faqat lotin alifbosida, kirillcha savolga faqat kirill alifbosida javob ber. Kontekst alifbosiga emas, aynan SAVOL alifbosiga qara.
3. Hech qanday tashqi bilim, taxmin yoki umumiy maʼlumotdan foydalanma — faqat kontekstda yozilganlarga tayan.
4. Agar kontekstda savolga javob topilmasa yoki kontekst savolga aniq javob berish uchun yetarli boʻlmasa, aniq shunday javob ber: "Hujjatda bu haqida ma'lumot yo'q." (krill soʻralgan boʻlsa: "Ҳужжатда бу ҳақида маълумот йўқ.")
5. Javob HAR DOIM toʻliq gap boʻlsin: savoldagi asosiy soʻzlarni takrorlab, ega va kesimli gap tuz. Faqat raqam, sana yoki bitta soʻzdan iborat javob TAQIQLANADI. Shu bilan birga ortiqcha choʻzma qilma, kontekstni soʻzma-soʻz koʻchirma qilib takrorlama — oʻz soʻzlaring bilan aniq va lo'nda tushuntir.
"""

USER_PROMPT_TEMPLATE = """KONTEKST:
{context}

SAVOL: {question}

Yuqoridagi qoidalarga qatʼiy amal qil. Javobni savoldagi soʻzlarni takrorlagan TOʻLIQ GAP shaklida yoz — yolgʻiz raqam yoki bitta soʻz yozish mumkin emas."""

# Used when the model answers with a bare number/word despite the rules
# above (see RagService). A fresh single-turn rewrite task works far better
# than continuing the conversation: with its own terse answer in context the
# model tends to repeat it, while question+answer -> sentence is a trivial
# text-composition task with no room to introduce new facts.
EXPAND_SYSTEM_PROMPT = (
    "Sen berilgan savol va qisqa javobdan FAQAT BITTA toʻliq gap tuzasan: "
    "savoldagi soʻzlar + javob. Yangi maʼlumot, izoh, taxmin yoki ikkinchi "
    "gap qoʻshish TAQIQLANADI. Savol qaysi alifboda boʻlsa, gap ham oʻsha "
    "alifboda boʻlsin."
)

# The worked example is format-only (unrelated domain): it shows the
# transformation shape, the actual facts come from {question}/{answer}.
EXPAND_USER_TEMPLATE = """Namuna:
SAVOL: Litsenziya necha yilga beriladi?
QISQA JAVOB: 5
TOʻLIQ GAP: Litsenziya 5 yilga beriladi.

Endi shu shaklda:
SAVOL: {question}
QISQA JAVOB: {answer}
TOʻLIQ GAP:"""


def build_expand_messages(question: str, answer: str) -> list[dict]:
    """Messages for rewriting a too-terse answer into a full sentence."""
    return [
        {"role": "system", "content": EXPAND_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXPAND_USER_TEMPLATE.format(question=question, answer=answer),
        },
    ]


def build_context(chunks: list) -> str:
    """Formats retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c.text}")
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list, original_question: str | None = None) -> list[dict]:
    """`original_question` is the user's raw input (e.g. Cyrillic), while
    `question` may already be transliterated to Latin for retrieval purposes.
    Passing both lets the LLM answer in the same script the user asked in.
    """
    context = build_context(chunks)
    display_question = original_question or question
    user_content = USER_PROMPT_TEMPLATE.format(context=context, question=display_question)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
