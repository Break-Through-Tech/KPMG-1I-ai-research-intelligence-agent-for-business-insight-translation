"""Request and response models for the Stage 1 API bridge."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    """Input accepted by the React research composer."""

    query: str = Field(..., min_length=1, description="AI research question")
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

    answer: str | None = None
    confidence: str | None = None
    top_score: float | None = None
    retrieved_chunks: list[dict[str, Any]] | None = None
    citations: list[dict[str, Any]] | None = None
