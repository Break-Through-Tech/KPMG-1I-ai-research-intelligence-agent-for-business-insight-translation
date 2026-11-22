"""Research endpoint backed by the LangGraph agent in src/agent.py."""

import threading
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_kpmg_user
from api.schemas import ResearchRequest, ResearchResponse

router = APIRouter()

# Local Qdrant allows one open client per storage folder, so questions are answered one at a time.
_agent_lock = threading.Lock()


def run_agent(query: str, thread_id: Optional[str] = None) -> dict[str, Any]:
    """Lazy import so API health/tests do not load the agent's models at import time."""

    from src.agent import run_agent as backend_run_agent

    with _agent_lock:
        return backend_run_agent(query, thread_id=thread_id)


def hits_in_context(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from src.agent import hits_in_context as backend_hits_in_context

    return backend_hits_in_context(state)


def _arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _metadata(hit: dict[str, Any]) -> dict[str, Any]:
    return {"authors": hit["authors"], "publish_date": hit["publish_date"], "categories": hit["categories"]}


@router.post("/api/research", response_model=ResearchResponse, response_model_exclude_none=True)
def research(
    request: ResearchRequest, user: dict[str, Any] = Depends(require_kpmg_user)
) -> ResearchResponse:
    # thread_id keeps conversation history across questions; focus_areas are accepted but unused.
    from src.agent import GenerationError
    from src.retrieval import RetrievalError

    try:
        # Prefix with the user's id so one user can never read or continue another user's conversation.
        thread_id = f"{user['uid']}:{request.thread_id}" if request.thread_id else None
        state = run_agent(request.query, thread_id=thread_id)
    except RetrievalError as error:
        raise HTTPException(status_code=503, detail="The paper search failed. Is the vector store loaded?") from error
    except GenerationError as error:
        raise HTTPException(
            status_code=502, detail="The LLM couldn't answer. Is Ollama running with qwen2.5:7b?"
        ) from error

    papers, chunks = hits_in_context(state)
    return ResearchResponse(
        answer=state["answer"],
        papers=[
            {"arxiv_id": hit["arxiv_id"], "title": hit["title"], "abstract": hit["abstract"],
             "url": _arxiv_url(hit["arxiv_id"]), "rank": hit["rank"], **_metadata(hit)}
            for hit in papers
        ],
        retrieved_chunks=[
            {"paper_id": hit["arxiv_id"], "title": hit["title"], "section_title": " > ".join(hit["breadcrumb"]),
             "chunk_text": hit["content"], "chunk_index": hit["chunk_index"], "score": hit["score"],
             "url": _arxiv_url(hit["arxiv_id"]), **_metadata(hit)}
            for hit in chunks
        ],
    )
