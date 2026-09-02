"""Manual end-to-end smoke test: retrieval + generation for one question.

Run from the project root:
    python -m scripts.test_generation "Savolingiz shu yerda"
"""

import asyncio
import sys

from app.services.llm import generate_answer
from app.services.prompts import build_messages
from app.services.retriever import Retriever
from app.utils.formatting import format_source

# Windows consoles default to a legacy codepage (cp1252) that can't encode
# Uzbek characters like U+02BB -- force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


async def main():
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Aeroportlar uchun ekspertiza oʻtkazish muddati qancha?"
    )

    retriever = Retriever()
    chunks = retriever.search_with_threshold(question)

    if not chunks:
        print("Hujjatda bu haqida ma'lumot yo'q.")
        return

    messages = build_messages(question, chunks)
    answer = await generate_answer(messages)

    top = chunks[0]
    print("Q:", question)
    print("A:", answer)
    print(f"Manba (top-1 natija, score={top.score:.3f}): {format_source(top.metadata)}")


if __name__ == "__main__":
    asyncio.run(main())
