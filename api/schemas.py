"""Request and response models for the Stage 1 API bridge."""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    """Input accepted by the React research composer."""

    query: str = Field(..., min_length=1, description="AI research question")
    thread_id: Optional[str] = None
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Prototype-only UI selections; not used for retrieval in Stage 1",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class ResearchResponse(BaseModel):
    """Subset of the existing run_agent result exposed to the frontend."""

    answer: Optional[str] = None
    confidence: Optional[str] = None
    top_score: Optional[float] = None
    retrieved_chunks: Optional[list[dict[str, Any]]] = None
    citations: Optional[list[dict[str, Any]]] = None
