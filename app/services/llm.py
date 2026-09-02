"""
Ollama client service.

Thin async wrapper around Ollama's local HTTP API, with:
  - retry on transient connection failures (tenacity)
  - a typed LLMUnavailableError so the API layer can return a clean 503
    instead of an opaque 500 when Ollama isn't running
  - a streaming variant for the /ask/stream endpoint
  - a cheap health probe used by /health
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class LLMUnavailableError(RuntimeError):
    """Raised when Ollama can't be reached after retries."""


# Only connection-level failures are worth retrying; a slow generation
# hitting the read timeout should fail fast rather than run 3x.
_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
async def _post_chat(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
        response = await client.post(settings.OLLAMA_CHAT_URL, json=payload)
        response.raise_for_status()
        return response.json()


def _build_payload(
    messages: list[dict],
    temperature: float | None,
    stream: bool,
    max_tokens: int | None = None,
) -> dict:
    options: dict = {
        "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature
    }
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload: dict = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": options,
    }
    # Omitted entirely when unset, so a shared Ollama server keeps applying
    # its own residency policy instead of having ours imposed per request.
    if settings.OLLAMA_KEEP_ALIVE is not None:
        payload["keep_alive"] = settings.OLLAMA_KEEP_ALIVE
    return payload


async def generate_answer(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Calls Ollama's chat endpoint and returns the assistant's text response."""
    payload = _build_payload(messages, temperature, stream=False, max_tokens=max_tokens)
    try:
        data = await _post_chat(payload)
    except _RETRYABLE as exc:
        raise LLMUnavailableError(
            f"Ollama bilan bogʻlanib boʻlmadi ({settings.OLLAMA_CHAT_URL})"
        ) from exc
    return data["message"]["content"]


async def stream_answer(
    messages: list[dict], temperature: float | None = None
) -> AsyncIterator[str]:
    """Yields the assistant's response token-by-token via Ollama's streaming API."""
    payload = _build_payload(messages, temperature, stream=True)
    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            async with client.stream("POST", settings.OLLAMA_CHAT_URL, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        return
    except _RETRYABLE as exc:
        raise LLMUnavailableError(
            f"Ollama bilan bogʻlanib boʻlmadi ({settings.OLLAMA_CHAT_URL})"
        ) from exc


async def check_ollama() -> bool:
    """Cheap liveness probe for /health -- true if Ollama answers at all."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(settings.OLLAMA_TAGS_URL)
            return response.status_code == 200
    except httpx.HTTPError:
        return False
