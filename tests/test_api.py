"""Contract tests for the Stage 1 FastAPI bridge."""

from fastapi.testclient import TestClient

from api.main import app
from api.routes import research as research_route


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_adapts_existing_result_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        research_route,
        "run_agent",
        lambda query, thread_id=None: {
            "answer": f"Answer for {query}",
            "confidence": "high",
            "top_score": 0.78,
            "retrieved_chunks": [
                {
                    "score": 0.78,
                    "paper_id": "arXiv:2403.01234",
                    "title": "A real backend title",
                    "section_title": "Abstract",
                    "chunk_text": "A retrieved passage.",
                    "chunk_index": 0,
                }
            ],
            "citations": [
                {
                    "paper_id": "arXiv:2403.01234",
                    "supporting_quote": "A supporting quote.",
                }
            ],
            "timings": {"total_execution_seconds": 1.2},
        },
    )

    response = client.post(
        "/api/research",
        json={"query": "What improves AI research reliability?", "focus_areas": ["AI Governance"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer for What improves AI research reliability?",
        "confidence": "high",
        "top_score": 0.78,
        "retrieved_chunks": [
            {
                "score": 0.78,
                "paper_id": "arXiv:2403.01234",
                "title": "A real backend title",
                "section_title": "Abstract",
                "chunk_text": "A retrieved passage.",
                "chunk_index": 0,
            }
        ],
        "citations": [
            {
                "paper_id": "arXiv:2403.01234",
                "supporting_quote": "A supporting quote.",
            }
        ],
    }


def test_research_rejects_blank_query() -> None:
    response = client.post("/api/research", json={"query": "   "})

    assert response.status_code == 422


def test_research_returns_gateway_error_when_backend_fails(monkeypatch) -> None:
    def fail(_query):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(research_route, "run_agent", fail)

    response = client.post("/api/research", json={"query": "What is AI governance?"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The research agent could not complete the request."
    }
