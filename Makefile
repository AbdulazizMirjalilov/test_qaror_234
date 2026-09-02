# Development workflow shortcuts. Requires poetry (https://python-poetry.org).

.PHONY: install install-light dev test lint format precommit ingest eval lock docker-up docker-down clean

## Install all dependencies (incl. the ML stack) into ./.venv
install:
	poetry install

## Light install without torch/chromadb -- enough to run the test suite
install-light:
	poetry install --without ml

## Run the API with auto-reload
dev:
	poetry run uvicorn app.main:app --reload

## Run the test suite
test:
	poetry run pytest -q

## Lint (no changes)
lint:
	poetry run ruff check app tests scripts

## Auto-format and fix lint issues
format:
	poetry run ruff format app tests scripts
	poetry run ruff check --fix app tests scripts

## Install the git pre-commit hooks
precommit:
	poetry run pre-commit install

## Rebuild the index from the source document (loader -> chunker -> embedder)
ingest:
	poetry run python -m app.ingestion.loader
	poetry run python -m app.ingestion.chunker
	poetry run python -m app.ingestion.embedder

## Evaluate retrieval quality against data/eval_questions.json
eval:
	poetry run python -m scripts.evaluate_retrieval

## Re-resolve and write poetry.lock
lock:
	poetry lock

## Regenerate requirements.txt (pip fallback) from poetry.lock
export-reqs:
	poetry export --with ml --without dev -f requirements.txt -o requirements.txt --without-hashes

## Start the full stack (API + Ollama) in Docker
docker-up:
	docker compose up --build

## Start only the API in Docker, reusing the host's Ollama + model caches
docker-up-host:
	docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build

docker-down:
	docker compose down

## Remove caches
clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
