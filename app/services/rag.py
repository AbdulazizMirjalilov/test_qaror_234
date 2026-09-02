"""
RAG service: business logic for answering questions about the document.

Acts as the abstraction layer between API endpoints and the infrastructure
services (retriever, Ollama client). Endpoints stay thin; this service owns:
- the score-threshold "not found" short-circuit
- prompt assembly and LLM invocation
- mapping infrastructure failures to API exceptions
- programmatic source citation (never trusted to the LLM)
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from app.core.config import settings
from app.core.constants import ErrorMessages, Messages
from app.core.exceptions import (
    BadGatewayException,
    GatewayTimeoutException,
    ServiceUnavailableException,
)
from app.schemas.ask import AskResponse
from app.services.llm import LLMUnavailableError, generate_answer, stream_answer
from app.services.prompts import build_expand_messages, build_messages
from app.utils.formatting import format_source
from app.utils.text import (
    echoes_question,
    has_cyrillic,
    is_capability_question,
    is_grounded,
)


def _needs_full_sentence(question: str, answer: str) -> bool:
    """True for answers that break the prompt's full-sentence rule (rule 5)
    and should get one corrective rewrite.

    Two distinct failures, and length only catches the first:
      - a bare value ("25");
      - a fragment that clears the length bar but restates nothing, e.g.
        "Yigirma besh ish kuni ichida" -- 28 characters, no subject, and
        none of the question's words.
    """
    stripped = answer.strip()
    # A refusal is a valid answer shape; rewriting it would only pad it out.
    if not is_grounded(stripped):
        return False
    return len(stripped) < settings.LLM_MIN_ANSWER_CHARS or not echoes_question(
        question, stripped
    )


def _in_question_script(question: str, latin: str, cyrillic: str) -> str:
    """Picks the canned answer matching the question's alphabet, the same
    rule the prompt imposes on the LLM (see prompts.py rule 2). Without this
    a Cyrillic question that never reaches the LLM -- a capability question,
    or one cut by the score threshold -- gets a Latin reply."""
    return cyrillic if has_cyrillic(question) else latin


class RagService:
    """Service orchestrating retrieval + generation for one question."""

    def __init__(self, retriever):
        self._retriever = retriever

    async def _retrieve(self, question: str):
        # Embedding the query is CPU-bound and blocking -- keep it off the
        # event loop.
        return await run_in_threadpool(self._retriever.search_with_threshold, question)

    async def ask(self, question: str) -> AskResponse:
        started = time.perf_counter()

        # Answered before retrieval: the decree says nothing about the bot,
        # so these would fall through the threshold into "not found".
        if is_capability_question(question):
            logger.info(f"ask: capability question | q={question!r}")
            return AskResponse(
                answer=_in_question_script(
                    question, Messages.HELP_ANSWER, Messages.HELP_ANSWER_CYRILLIC
                ),
                source=None,
                score=None,
                found_in_document=False,
                answer_grounded=False,
            )

        chunks = await self._retrieve(question)

        if not chunks:
            logger.info(f"ask: below threshold, skipping LLM | q={question!r}")
            return AskResponse(
                answer=_in_question_script(
                    question,
                    Messages.NOT_FOUND_ANSWER,
                    Messages.NOT_FOUND_ANSWER_CYRILLIC,
                ),
                source=None,
                score=None,
                found_in_document=False,
                answer_grounded=False,
            )

        # The retriever transliterates internally for matching; the LLM gets
        # the user's raw question so it can mirror the original script in its
        # answer.
        messages = build_messages(question, chunks, original_question=question)
        try:
            answer = await generate_answer(messages)
            if _needs_full_sentence(question, answer):
                # One corrective round-trip: a fresh rewrite task combining
                # the question and the terse answer into a full sentence.
                logger.info(f"ask: terse answer {answer!r}, requesting expansion")
                # temperature 0 + tight token cap: a pure format rewrite,
                # with no room to append hallucinated extra sentences.
                expanded = await generate_answer(
                    build_expand_messages(question, answer),
                    temperature=0.0,
                    max_tokens=60,
                )
                expanded = expanded.strip()
                if len(expanded) > len(answer.strip()):
                    answer = expanded
        except LLMUnavailableError as exc:
            logger.error(f"ask: LLM unavailable: {exc}")
            raise ServiceUnavailableException() from exc
        except httpx.TimeoutException as exc:
            logger.error(f"ask: LLM timed out after {settings.LLM_TIMEOUT_SECONDS}s")
            raise GatewayTimeoutException() from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"ask: LLM returned HTTP {exc.response.status_code}")
            raise BadGatewayException(
                message=ErrorMessages.LLM_BAD_RESPONSE.format(status_code=exc.response.status_code)
            ) from exc

        top = chunks[0]
        source = format_source(top.metadata)
        logger.info(
            f"ask: answered | top_score={top.score:.3f} source={source!r} "
            f"elapsed={time.perf_counter() - started:.2f}s q={question!r}"
        )
        return AskResponse(
            answer=answer,
            source=source,
            score=round(top.score, 3),
            found_in_document=True,
            answer_grounded=is_grounded(answer),
        )

    async def ask_stream(self, question: str) -> AsyncIterator[str]:
        """NDJSON event stream: {"type": "meta", ...} first (source is known
        before generation starts, since it comes from retrieval metadata, not
        the LLM), then a series of {"type": "token", "text": ...}, then
        {"type": "done", "answer_grounded": ...}.
        """
        def event(obj: dict) -> str:
            return json.dumps(obj, ensure_ascii=False) + "\n"

        # Same short-circuit as /ask, before retrieval.
        if is_capability_question(question):
            logger.info(f"ask/stream: capability question | q={question!r}")
            yield event({"type": "meta", "found_in_document": False, "source": None, "score": None})
            yield event(
                {
                    "type": "token",
                    "text": _in_question_script(
                        question, Messages.HELP_ANSWER, Messages.HELP_ANSWER_CYRILLIC
                    ),
                }
            )
            yield event({"type": "done", "answer_grounded": False})
            return

        chunks = await self._retrieve(question)

        if not chunks:
            yield event({"type": "meta", "found_in_document": False, "source": None, "score": None})
            yield event(
                {
                    "type": "token",
                    "text": _in_question_script(
                        question,
                        Messages.NOT_FOUND_ANSWER,
                        Messages.NOT_FOUND_ANSWER_CYRILLIC,
                    ),
                }
            )
            yield event({"type": "done", "answer_grounded": False})
            return

        top = chunks[0]
        yield event(
            {
                "type": "meta",
                "found_in_document": True,
                "source": format_source(top.metadata),
                "score": round(top.score, 3),
            }
        )

        messages = build_messages(question, chunks, original_question=question)
        collected: list[str] = []
        try:
            async for piece in stream_answer(messages):
                collected.append(piece)
                yield event({"type": "token", "text": piece})
        except LLMUnavailableError:
            logger.error("ask/stream: LLM unavailable")
            yield event({"type": "error", "detail": ErrorMessages.LLM_UNAVAILABLE})
            return
        except httpx.TimeoutException:
            logger.error(f"ask/stream: LLM timed out after {settings.LLM_TIMEOUT_SECONDS}s")
            yield event({"type": "error", "detail": ErrorMessages.LLM_TIMEOUT})
            return
        except httpx.HTTPStatusError as exc:
            # Without this the error escapes mid-generator. The meta event has
            # already been sent, so the response has started and the exception
            # handlers can no longer replace it -- the client would just see
            # the connection drop, indistinguishable from a truncated answer.
            # Mirrors the /ask handler above.
            logger.error(f"ask/stream: LLM returned HTTP {exc.response.status_code}")
            yield event(
                {
                    "type": "error",
                    "detail": ErrorMessages.LLM_BAD_RESPONSE.format(
                        status_code=exc.response.status_code
                    ),
                }
            )
            return
        yield event({"type": "done", "answer_grounded": is_grounded("".join(collected))})
