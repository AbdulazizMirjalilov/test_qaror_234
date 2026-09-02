FROM python:3.14-slim

WORKDIR /srv/qaror

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    # keep the BGE-M3 download in a mountable location (see docker-compose)
    HF_HOME=/srv/qaror/.cache/huggingface

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root

COPY app ./app
COPY scripts ./scripts
# The Chroma index is a build artifact and is NOT in git, so run
#   make ingest
# on the host before `docker build` -- otherwise data/ arrives without an
# index and the app aborts at startup with IndexNotReadyError.
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
