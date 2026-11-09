"""Research endpoint backed by the existing LangGraph agent."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from api.schemas import ResearchRequest, ResearchResponse

router = APIRouter()


def run_agent(query: str, thread_id: Optional[str] = None) -> dict[str, Any]:
    """Lazy adapter so API health/tests do not initialize Ollama at import time."""

    from src.agent import run_agent as backend_run_agent

    return backend_run_agent(query, thread_id=thread_id)

@router.post("/api/research", response_model=ResearchResponse, response_model_exclude_none=True)
def research(request: ResearchRequest) -> ResearchResponse:
    # focus_areas are intentionally accepted for frontend compatibility but are
    # not applied to the query or retrieval logic until a later migration stage.
    try:
        result = run_agent(
            request.query,
            thread_id=request.thread_id,
            )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="The research agent could not complete the request.",
        ) from error

    supported_fields = {
        field: result[field]
        for field in (
            "answer",
            "confidence",
            "top_score",
            "retrieved_chunks",
            "citations",
        )
        if field in result
    }
    return ResearchResponse.model_validate(supported_fields)
