# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI RAG service that answers questions, in Uzbek Latin or Uzbek Cyrillic, about Cabinet of Ministers Decree No. 234 (environmental expertise). Retrieval is BGE-M3 embeddings in a persistent Chroma collection; generation is a local Ollama model (`qwen2.5:7b-instruct` by default). The README is in Uzbek and is the user-facing reference; user-facing strings in code are Uzbek too, and code comments are English.

## Commands

Poetry manages dependencies and the venv lives in `./.venv` (`poetry.toml`). The ML stack (chromadb, sentence-transformers/torch) is the optional Poetry group `ml`; the test suite and lint do not need it.

```bash
make install            # poetry install (full, incl. ML stack)
make install-light      # poetry install --without ml  (what CI uses; enough for tests + lint)
make dev                # uvicorn app.main:app --reload  (needs a built index + running Ollama)
make lint               # ruff check app tests scripts
make format             # ruff format + ruff check --fix
make ingest             # loader -> chunker -> embedder; rebuilds data/chroma_db
make eval               # python -m scripts.evaluate_retrieval  (full install + index required)
poetry run python -m scripts.test_generation "Savol matni"   # one-shot retrieval + LLM smoke test
make lock && make export-reqs   # after editing deps in pyproject.toml; requirements.txt is GENERATED, never hand-edit
```

Tests:

```bash
make test                                              # poetry run pytest -q  (full suite, ~5 s)
poetry run pytest tests/test_api.py -q                 # one file
poetry run pytest tests/test_api.py::test_ask_found -q # one test
```

The project is `package-mode = false`, so `app` is never installed into the venv; `pythonpath = ["."]` under `[tool.pytest.ini_options]` is what makes it importable from tests. Keep that setting if the pytest config is ever moved.

Docker: `make docker-up` runs API + Ollama in containers; `make docker-up-host` runs only the API and reuses the host's Ollama and HuggingFace cache. The image `COPY`s `data/` and the Chroma index is gitignored, so run `make ingest` on the host before `docker build`.

## Architecture

Layered: `app/api` (thin routers) -> `app/core/dependencies.py` (DI) -> `app/services/rag.py` (orchestration) -> `app/services/retriever.py` (Chroma + embeddings) and `app/services/llm.py` (Ollama HTTP client). `app/ingestion/` is an offline pipeline and is never touched at request time.

### Request path (`POST /v1/ask`, `POST /v1/ask/stream`)

`RagService.ask` in `app/services/rag.py` is the whole decision tree, in this order:

1. `is_capability_question()` (`app/utils/text.py`): bare greetings and "what can you do" questions get `Messages.HELP_ANSWER` before retrieval, with `found_in_document: false`.
2. `retriever.search_with_threshold()` runs in a threadpool and returns `[]` when the top-1 cosine score is below `SCORE_THRESHOLD` (0.55). Empty result -> canned `NOT_FOUND_ANSWER`, the LLM is never called.
3. `build_messages()` (`prompts.py`) + `generate_answer()`. The LLM receives the user's raw question so it answers in the same script; retrieval used the transliterated form.
4. `_needs_full_sentence()`: if the answer is a bare value, or a fragment that echoes no content word of the question (`echoes_question`), one corrective rewrite call is made via `build_expand_messages` at temperature 0 with a 60-token cap. Refusals are exempt.
5. `source` is `format_source(top_chunk.metadata)`, always programmatic and never taken from the LLM. `answer_grounded = is_grounded(answer)`, which looks for the Latin and Cyrillic "ma'lumot yo'q" markers.

`found_in_document` (the retrieval decision) and `answer_grounded` (whether the LLM declined) are deliberately separate response fields. Every canned reply has a Latin and a Cyrillic twin in `app/core/constants.py`, chosen by `has_cyrillic(question)`.

The stream variant emits NDJSON events `meta` (source and score are known before generation) -> `token`* -> `done`. LLM failures mid-stream become an `error` event because headers are already sent.

Ollama failures map to the shared error envelope `{"success": false, "message", "data"}` via `app/core/exceptions.py`: `LLMUnavailableError` -> 503, `httpx.TimeoutException` -> 504, `httpx.HTTPStatusError` -> 502. Only connection-level errors are retried (tenacity, 3 attempts); read timeouts fail fast.

### Ingestion pipeline (`make ingest`)

- `loader.py`: parses the lex.uz HTML export (`data/234_11.05.2026_ozb.doc` has a `.doc` extension but is HTML) by CSS class (`ACT_TEXT`, `TEXT_HEADER_DEFAULT`, `APPL_BANNER_LANDSCAPE_TITLE`, `TABLE_STD2`, ...) into ordered blocks carrying `ilova_num` / `bob_num` / `punkt_num` -> `data/parsed_blocks.json`.
- `chunker.py`: one chunk per numbered punkt (unnumbered follow-on blocks merged in), tables split one row per chunk with the header row repeated, every chunk prefixed with a `[N-ilova — title | N-bob — title]` context header, oversized chunks split with overlap. It also prepends a synthetic `document_identity` chunk (decree number, date, signatory, extracted from the signature block, never hardcoded) so questions about the decree itself retrieve something; it renders as source `"Qaror rekvizitlari"` -> `data/chunks.json`.
- `embedder.py`: BGE-M3 with normalized embeddings; drops and recreates the Chroma collection (`hnsw:space=cosine`) -> `data/chroma_db/`.

All three outputs are gitignored build artifacts. Chunking settings (`CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_CHARS`, `COLLECTION_NAME`) are baked in at index time, so changing them means re-running `make ingest`.

### Normalization contract

`app/utils/text.py` is shared by ingestion (`loader.py`) and the query path (`retriever.py`). Index text and queries must pass through the same `normalize_uzbek_text`: apostrophe variants are canonicalized to U+02BB `ʻ` for oʻ/gʻ and U+02BC `ʼ` for tutuq belgisi, not ASCII `'`. Queries additionally get Cyrillic -> Latin transliteration (`normalize_query`). Any change to normalization requires re-ingesting.

### Configuration and startup

`app/core/config.py` uses pydantic-settings with the `QAROR_` env prefix and `.env` at the project root. All paths are anchored to `PROJECT_ROOT`, so scripts work from any CWD. `OLLAMA_KEEP_ALIVE` is validated at startup as a Go duration string. `.env.example` lists the commonly changed values.

The `app/main.py` lifespan builds the `Retriever` once (loads BGE-M3, opens Chroma) and stores it on `app.state`; the import is deferred inside `_build_retriever()` so importing `app.main` does not pull in torch/chromadb. A missing index raises `IndexNotReadyError` at startup. `/health` returns 503 with per-component status when the index is empty or Ollama is unreachable.

## Testing conventions

- `tests/conftest.py` monkeypatches `app.main._build_retriever` with a `FakeRetriever` (hit for questions containing "aeroport", miss otherwise) and builds the app with `create_app()`. No torch, Chroma, or Ollama is needed.
- Patch LLM calls at their import site: `monkeypatch.setattr("app.services.rag.generate_answer", ...)` or `"app.services.rag.stream_answer"`. Patching `app.services.llm` has no effect on the service.
- `tests/test_retriever.py` calls `pytest.importorskip("chromadb")`, so it is skipped on a light install; it bypasses `Retriever.__init__` and stubs `.search` to test threshold logic only.
- `data/eval_questions.json` (cases with `expect_found`, `expected_ilova_num`, `expected_punkt_num`) drives `make eval`. Extend it when retuning `SCORE_THRESHOLD`; the script prints the score gap between in-scope and out-of-scope questions.

## Conventions

- Ruff: line length 100, rules E/F/I/UP/B, target py310, `data/` excluded. `app/services/prompts.py` is exempt from E501 because prompt text must stay verbatim; do not rewrap it. `tests/test_retriever.py` is exempt from E402 (imports follow `importorskip`).
- Pre-commit (`make precommit`) runs ruff plus hygiene hooks; `data/` is excluded from all formatters.
- CLI entry points (`__main__` blocks and `scripts/`) call `sys.stdout.reconfigure(encoding="utf-8")` because Windows consoles cannot print U+02BB. Keep this in any new script that prints document text.
- Heavy ML imports (`chromadb`, `sentence_transformers`) belong only in `retriever.py`, `embedder.py`, and `scripts/`, never at module top level in anything `app.main` imports eagerly.
