"""
Application configuration settings.

Uses Pydantic Settings for environment variable management with:
- Automatic .env file loading (project root)
- Type validation and coercion
- Computed properties for derived values

Every value can be overridden via environment variables with the QAROR_
prefix or a .env file (see .env.example), e.g.:

    QAROR_SCORE_THRESHOLD=0.5
    QAROR_OLLAMA_BASE_URL=http://ollama:11434

Paths are anchored to the project root so the app and scripts work
regardless of the current working directory.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ollama parses a keep_alive string with Go's time.ParseDuration: an optional
# sign, then either a bare "0" or one or more number+unit pairs ("30m",
# "1h30m", "-1s"). Values like "30min", "30 m" or a bare "-1" are refused by
# the server on *every* request, so they are caught here at startup instead.
_GO_DURATION_RE = re.compile(
    r"^[+-]?(?:0|(?:(?:\d+(?:\.\d*)?|\.\d+)(?:ns|us|µs|μs|ms|s|m|h))+)$"
)


class EnvironmentOption(str, Enum):
    """Application environment options."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Combined application settings.

    Reads from QAROR_-prefixed environment variables and the project .env file.
    All defaults target local development.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="QAROR_",
        extra="ignore",
    )

    # -- Application ------------------------------------------------------
    APP_NAME: str = "Qaror 234 RAG API"
    APP_DESCRIPTION: str | None = (
        "RAG tizimi: Vazirlar Mahkamasining 234-son qarori boʻyicha savol-javob"
    )
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: EnvironmentOption = EnvironmentOption.LOCAL

    # -- CORS -------------------------------------------------------------
    CORS_ORIGINS: list[str] = ["*"]

    # -- Data / index -----------------------------------------------------
    DATA_DIR: Path = PROJECT_ROOT / "data"
    SOURCE_DOC_PATH: Path = PROJECT_ROOT / "data" / "234_11.05.2026_ozb.doc"
    PARSED_BLOCKS_PATH: Path = PROJECT_ROOT / "data" / "parsed_blocks.json"
    CHUNKS_PATH: Path = PROJECT_ROOT / "data" / "chunks.json"
    CHROMA_DIR: Path = PROJECT_ROOT / "data" / "chroma_db"
    COLLECTION_NAME: str = "qaror_234"

    # -- Chunking ---------------------------------------------------------
    # Rough token estimate: ~4 chars per token works fine for uz/ru/en mixed
    # text without pulling in a real tokenizer.
    CHUNK_MAX_CHARS: int = 1800  # ~450 tokens
    CHUNK_OVERLAP_CHARS: int = 250  # ~60 tokens

    # -- Retrieval --------------------------------------------------------
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    TOP_K: int = 5
    # Cosine similarity threshold below which we treat retrieval as "no
    # answer in the document". Tuned on the eval set in data/eval_questions.json
    # (see scripts/evaluate_retrieval.py); 0.45-0.55 is a reasonable range
    # for bge-m3 on this kind of text.
    SCORE_THRESHOLD: float = 0.55

    # -- LLM (Ollama) -----------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
    # How long Ollama keeps the model in RAM after a request. The default
    # 5m eviction costs ~90-100s reloading the model from disk on the next
    # question; on CPU-only machines that dwarfs the answer time itself.
    # Leave empty to send no keep_alive at all and defer to the Ollama
    # server's own policy -- the right choice when that server is shared
    # with other tools on the machine rather than owned by this app.
    OLLAMA_KEEP_ALIVE: str | None = "30m"
    # Low temperature is intentional: for a grounded QA task we want
    # deterministic, literal answers, not creative variation.
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT_SECONDS: float = 120.0
    # Answers shorter than this (e.g. a bare "25") trigger one follow-up
    # request asking the model to restate as a full sentence.
    LLM_MIN_ANSWER_CHARS: int = 20

    @field_validator("OLLAMA_KEEP_ALIVE")
    @classmethod
    def _validate_keep_alive(cls, value: str | None) -> str | None:
        """Reject a malformed duration at startup rather than on every
        question: Ollama refuses the entire request when it cannot parse
        this field, so a typo here would mean a 100% failure rate.

        An empty value normalises to None, meaning "omit keep_alive".
        """
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not _GO_DURATION_RE.match(value):
            raise ValueError(
                f"QAROR_OLLAMA_KEEP_ALIVE={value!r} is not a duration Ollama "
                "accepts. Use a number with a unit ('30m', '1h30m', '90s'), "
                "'0' to unload immediately, or '-1s' to keep the model loaded "
                "indefinitely. Leave it empty to defer to the Ollama server's "
                "own setting."
            )
        return value

    @computed_field
    @property
    def OLLAMA_CHAT_URL(self) -> str:
        """Ollama chat completions endpoint."""
        return f"{self.OLLAMA_BASE_URL}/api/chat"

    @computed_field
    @property
    def OLLAMA_TAGS_URL(self) -> str:
        """Ollama model listing endpoint (used as a liveness probe)."""
        return f"{self.OLLAMA_BASE_URL}/api/tags"


settings = Settings()
