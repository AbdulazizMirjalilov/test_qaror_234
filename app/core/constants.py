"""
Application constants and user-facing messages.

Centralizes string literals to avoid duplication and ensure consistency.
User-facing messages are in Uzbek, matching the domain and audience.
"""


class Messages:
    """Standard answer strings returned by the RAG pipeline."""

    NOT_FOUND_ANSWER = "Hujjatda bu haqida ma'lumot yo'q."
    # The LLM is told to use this wording for Cyrillic questions (see
    # prompts.py rule 4); the threshold short-circuit needs it too, so a
    # Cyrillic question doesn't get a Latin refusal.
    NOT_FOUND_ANSWER_CYRILLIC = "Ҳужжатда бу ҳақида маълумот йўқ."

    # Answer for questions about the assistant rather than about the decree
    # ("nima qila olasan?"). Retrieval can never satisfy these -- the
    # document says nothing about the bot -- so they would otherwise fall
    # through the score threshold and get NOT_FOUND_ANSWER, which reads as a
    # malfunction rather than an answer.
    HELP_ANSWER = (
        "Men Vazirlar Mahkamasining 234-son qarori — ekologik ekspertiza, "
        "atrof-muhitga taʼsirni baholash va strategik ekologik baholash "
        "toʻgʻrisidagi hujjat — matni boʻyicha savollarga javob beraman.\n"
        "Masalan, soʻrashingiz mumkin:\n"
        "• ekspertiza muddatlari va toʻlov miqdori (BXM);\n"
        "• davlat va jamoat ekologik ekspertizasini oʻtkazish tartibi;\n"
        "• jamoatchilik eshituvlarini oʻtkazish tartibi;\n"
        "• malaka sertifikatini berish va bekor qilish;\n"
        "• strategik ekologik baholash va ekologik maʼruza.\n"
        "Savolni lotin yoki kirill alifbosida yozing — javobni manbasi "
        "(ilova/bob/band) bilan qaytaraman."
    )
    HELP_ANSWER_CYRILLIC = (
        "Мен Вазирлар Маҳкамасининг 234-сон қарори — экологик экспертиза, "
        "атроф-муҳитга таъсирни баҳолаш ва стратегик экологик баҳолаш "
        "тўғрисидаги ҳужжат — матни бўйича саволларга жавоб бераман.\n"
        "Масалан, сўрашингиз мумкин:\n"
        "• экспертиза муддатлари ва тўлов миқдори (БҲМ);\n"
        "• давлат ва жамоат экологик экспертизасини ўтказиш тартиби;\n"
        "• жамоатчилик эшитувларини ўтказиш тартиби;\n"
        "• малака сертификатини бериш ва бекор қилиш;\n"
        "• стратегик экологик баҳолаш ва экологик маъруза.\n"
        "Саволни лотин ёки кирилл алифбосида ёзинг — жавобни манбаси "
        "(илова/боб/банд) билан қайтараман."
    )


class ErrorMessages:
    """Standard error messages used across the application."""

    LLM_UNAVAILABLE = (
        "LLM xizmati (Ollama) bilan bogʻlanib boʻlmadi. "
        "Ollama ishlab turganini tekshiring (`ollama list`)."
    )
    LLM_BAD_RESPONSE = "LLM xizmati xato qaytardi (HTTP {status_code})."
    LLM_TIMEOUT = (
        "LLM belgilangan vaqt ichida javob bermadi. Model hali yuklanayotgan "
        "boʻlishi mumkin — birozdan soʻng qayta urinib koʻring, yoki "
        "QAROR_LLM_TIMEOUT_SECONDS qiymatini oshiring."
    )
    INTERNAL_ERROR = "Ichki server xatosi"
    VALIDATION_ERROR = "Soʻrov validatsiyadan oʻtmadi"
