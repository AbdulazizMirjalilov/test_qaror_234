"""
Schemas for the ask (question answering) domain.
"""

from pydantic import Field

from app.schemas.base import BaseSchema


class AskRequest(BaseSchema):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Hujjat boʻyicha savol (lotin yoki kirill alifbosida)",
    )


class AskResponse(BaseSchema):
    answer: str
    source: str | None = Field(
        description="Manba (ilova/bob/band), retrieval metadatasidan dasturiy aniqlanadi"
    )
    score: float | None = Field(description="Top-1 natijaning cosine similarity balli")
    found_in_document: bool = Field(
        description="Retrieval bosqichida tegishli boʻlak topildimi (LLM'dan oldin)"
    )
    answer_grounded: bool = Field(
        description="LLM mazmunli javob berdimi, yoki 'ma'lumot yo'q' deb rad etdimi"
    )
