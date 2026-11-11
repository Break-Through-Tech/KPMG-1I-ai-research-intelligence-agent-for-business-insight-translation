"""Contract tests for the Stage 1 FastAPI bridge. The agent is faked."""

import threading
import time

from fastapi.testclient import TestClient

from api.main import app
from api.routes import research as research_route
from src.agent import GenerationError
from src.retrieval import RetrievalError


client = TestClient(app)

PAPER = {"arxiv_id": "2610.13767v1", "title": "DualQuant", "abstract": "We quantize LLMs to 4 bits.",
         "score": 0.8, "rank": 1}
CHUNK = {"chunk_id": "2610.29544v1_chunk_3", "arxiv_id": "2610.29544v1", "title": "RISK-KV",
         "content": "Selective KV reconstruction.", "breadcrumb": ["3 Method", "3.2 Selection"], "chunk_index": 3,
         "content_type": "text", "score": 0.03, "rank": 1}
STATE = {"query": "q", "paper_hits": [PAPER, {**PAPER, "rank": 2}], "chunk_hits": [CHUNK, {**CHUNK, "rank": 2}],
         "context": "ctx", "answer": "DualQuant reaches 4 bits [DualQuant](https://arxiv.org/abs/2610.13767v1)."}


def fake_agent(monkeypatch, run):
    monkeypatch.setattr(research_route, "run_agent", run)
    # the LLM was given only the top paper and passage
    monkeypatch.setattr(research_route, "hits_in_context", lambda state: (state["paper_hits"][:1], state["chunk_hits"][:1]))


def test_health() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_returns_the_answer_and_what_the_llm_was_given(monkeypatch) -> None:
    queries = []
    fake_agent(monkeypatch, lambda query: queries.append(query) or STATE)

    response = client.post(
        "/api/research",
        json={"query": "  How does 4-bit quantization work?  ", "thread_id": "t1", "focus_areas": ["AI Governance"]},
    )

    assert response.status_code == 200
    assert queries == ["How does 4-bit quantization work?"]
    assert response.json() == {
        "answer": STATE["answer"],
        "papers": [{"arxiv_id": "2610.13767v1", "title": "DualQuant", "abstract": "We quantize LLMs to 4 bits.",
                    "url": "https://arxiv.org/abs/2610.13767v1", "rank": 1}],
        "retrieved_chunks": [{"paper_id": "2610.29544v1", "title": "RISK-KV", "section_title": "3 Method > 3.2 Selection",
                              "chunk_text": "Selective KV reconstruction.", "chunk_index": 3, "score": 0.03,
                              "url": "https://arxiv.org/abs/2610.29544v1"}],
    }


def test_research_rejects_blank_query() -> None:
    response = client.post("/api/research", json={"query": "   "})

    assert response.status_code == 422


def test_search_failure_is_a_503(monkeypatch) -> None:
    def fail(_query):
        raise RetrievalError("Couldn't reach Qdrant")

    fake_agent(monkeypatch, fail)

    response = client.post("/api/research", json={"query": "What is AI governance?"})

    assert response.status_code == 503
    assert "paper search failed" in response.json()["detail"]


def test_llm_failure_is_a_502(monkeypatch) -> None:
    def fail(_query):
        raise GenerationError("connection refused")

    fake_agent(monkeypatch, fail)

    response = client.post("/api/research", json={"query": "What is AI governance?"})

    assert response.status_code == 502
    assert "Ollama" in response.json()["detail"]


def test_questions_are_answered_one_at_a_time(monkeypatch) -> None:
    from src import agent

    running, overlaps = [], []

    def slow_agent(query):
        running.append(query)
        overlaps.append(len(running))
        time.sleep(0.05)
        running.remove(query)
        return STATE

    monkeypatch.setattr(agent, "run_agent", slow_agent)
    threads = [threading.Thread(target=research_route.run_agent, args=(f"q{i}",)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert overlaps == [1, 1, 1]
