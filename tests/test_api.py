import json

import pytest

from app.services.llm import LLMUnavailableError


async def fake_answer(messages, temperature=None, max_tokens=None):
    return "Aeroportlar uchun muddat 25 ish kuni."


async def failing_answer(messages, temperature=None, max_tokens=None):
    raise LLMUnavailableError("connection refused")


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_ask_found(client, monkeypatch):
    monkeypatch.setattr("app.services.rag.generate_answer", fake_answer)
    resp = client.post("/v1/ask", json={"question": "Aeroportlar uchun muddat qancha?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found_in_document"] is True
    assert body["answer"] == "Aeroportlar uchun muddat 25 ish kuni."
    # source comes from retrieval metadata, not the LLM
    assert body["source"] == "1-ilova"
    assert body["score"] == 0.71
    assert body["answer_grounded"] is True


def test_ask_terse_answer_expanded(client, monkeypatch):
    calls = []

    async def terse_then_full(messages, temperature=None, max_tokens=None):
        calls.append(messages)
        if len(calls) == 1:
            return "25"
        # the retry is a fresh rewrite task carrying question + terse answer
        assert "QISQA JAVOB: 25" in messages[-1]["content"]
        return "Aeroportlar uchun ekspertiza muddati 25 ish kuni."

    monkeypatch.setattr("app.services.rag.generate_answer", terse_then_full)
    resp = client.post("/v1/ask", json={"question": "Aeroportlar uchun muddat qancha?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Aeroportlar uchun ekspertiza muddati 25 ish kuni."
    assert len(calls) == 2


def test_ask_sentence_fragment_expanded(client, monkeypatch):
    """A fragment can clear LLM_MIN_ANSWER_CHARS and still break the prompt's
    full-sentence rule -- "Yigirma besh ish kuni ichida" is 28 characters but
    restates nothing from the question. Length alone cannot catch it."""
    calls = []

    async def fragment_then_sentence(messages, temperature=None, max_tokens=None):
        calls.append(messages)
        if len(calls) == 1:
            return "Yigirma besh ish kuni ichida"
        return "Aeroportlar uchun ekspertiza muddati 25 ish kuni."

    monkeypatch.setattr("app.services.rag.generate_answer", fragment_then_sentence)
    resp = client.post("/v1/ask", json={"question": "Aeroportlar uchun muddat qancha?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Aeroportlar uchun ekspertiza muddati 25 ish kuni."
    assert len(calls) == 2


def test_ask_refusal_not_reprompted(client, monkeypatch):
    """A refusal restates nothing either, but rewriting it would just pad it
    out -- the groundedness check must exempt it."""
    calls = []

    async def declining(messages, temperature=None, max_tokens=None):
        calls.append(messages)
        return "Hujjatda bu haqida ma'lumot yo'q."

    monkeypatch.setattr("app.services.rag.generate_answer", declining)
    resp = client.post("/v1/ask", json={"question": "Aeroportlar uchun muddat qancha?"})
    assert resp.status_code == 200
    assert len(calls) == 1


def test_ask_full_answer_not_reprompted(client, monkeypatch):
    calls = []

    async def full_answer(messages, temperature=None, max_tokens=None):
        calls.append(messages)
        return "Aeroportlar uchun muddat 25 ish kuni."

    monkeypatch.setattr("app.services.rag.generate_answer", full_answer)
    resp = client.post("/v1/ask", json={"question": "Aeroportlar uchun muddat qancha?"})
    assert resp.status_code == 200
    assert len(calls) == 1


def test_ask_llm_decline_flagged_ungrounded(client, monkeypatch):
    async def declining_answer(messages, temperature=None, max_tokens=None):
        return "Hujjatda bu haqida ma'lumot yo'q."

    monkeypatch.setattr("app.services.rag.generate_answer", declining_answer)
    resp = client.post("/v1/ask", json={"question": "Aeroportlar uchun muddat qancha?"})
    body = resp.json()
    # retrieval found something, but the LLM itself declined
    assert body["found_in_document"] is True
    assert body["answer_grounded"] is False


def test_ask_not_found_skips_llm(client, monkeypatch):
    called = False

    async def spy(messages, temperature=None, max_tokens=None):
        nonlocal called
        called = True
        return "should not happen"

    monkeypatch.setattr("app.services.rag.generate_answer", spy)
    resp = client.post("/v1/ask", json={"question": "Soliq stavkalari qancha?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found_in_document"] is False
    assert body["source"] is None
    assert body["answer_grounded"] is False
    assert called is False


def test_ask_llm_timeout_returns_504(client, monkeypatch):
    import httpx

    async def timing_out(messages, temperature=None, max_tokens=None):
        raise httpx.ReadTimeout("read timeout")

    monkeypatch.setattr("app.services.rag.generate_answer", timing_out)
    resp = client.post("/v1/ask", json={"question": "Aeroportlar uchun muddat qancha?"})
    assert resp.status_code == 504
    body = resp.json()
    assert body["success"] is False
    assert body["data"]["name"] == "GatewayTimeoutException"


def test_ask_llm_down_returns_503_envelope(client, monkeypatch):
    monkeypatch.setattr("app.services.rag.generate_answer", failing_answer)
    resp = client.post("/v1/ask", json={"question": "Aeroportlar uchun muddat qancha?"})
    assert resp.status_code == 503
    body = resp.json()
    # consistent error envelope from the registered exception handlers
    assert body["success"] is False
    assert "Ollama" in body["message"]
    assert body["data"]["name"] == "ServiceUnavailableException"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 3000},
        {},
    ],
)
def test_ask_invalid_question_rejected(client, payload):
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 422
    assert resp.json()["success"] is False


def test_health_reports_components(client, monkeypatch):
    async def ollama_up():
        return True

    monkeypatch.setattr("app.main.check_ollama", ollama_up)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index"] == {"ok": True, "chunks": 42}
    assert body["ollama"] == {"ok": True}


def test_health_degraded_when_ollama_down(client, monkeypatch):
    async def ollama_down():
        return False

    monkeypatch.setattr("app.main.check_ollama", ollama_down)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"


def test_ask_stream_emits_meta_tokens_done(client, monkeypatch):
    async def fake_stream(messages, temperature=None, max_tokens=None):
        for piece in ["25 ", "ish ", "kuni."]:
            yield piece

    monkeypatch.setattr("app.services.rag.stream_answer", fake_stream)
    resp = client.post("/v1/ask/stream", json={"question": "Aeroportlar uchun muddat qancha?"})
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.strip().split("\n")]
    assert events[0]["type"] == "meta"
    assert events[0]["source"] == "1-ilova"
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "25 ish kuni."
    assert events[-1]["type"] == "done"
    assert events[-1]["answer_grounded"] is True
