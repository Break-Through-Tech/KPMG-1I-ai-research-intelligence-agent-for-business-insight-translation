"""
Metadata schema for our project 
- `Paper` describes one research paper's metadata (one row/object per paper).

Anyone on the team can import these classes to validate data before it's
passed into the ingestion pipeline or returned from the search function.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

class Paper(BaseModel):
    """Metadata for a single research paper."""

    # Unique identifier for the paper. We use the arXiv ID itself
    # every part of the pipeline can refer to the same paper the same way.
    paper_id: str = Field(..., description="arXiv ID, e.g. '2609.24974'")

    title: str = Field(..., description="Paper title")

    authors: list[str] = Field(
        default_factory=list,
        description="List of author names, e.g. ['Haoran Ye', 'Yuxing Lu']",
    )

    abstract: str = Field(..., description="Full abstract text")

    categories: list[str] = Field(
        default_factory=list,
        description="Subject categories, e.g. ['cs.AI', 'cs.CL', 'cs.NE']",
    )

    publish_date: Optional[date] = Field(
        default=None, description="Date the paper was published/listed"
    )

    pdf_url: Optional[HttpUrl] = Field(
        default=None, description="Direct link to the PDF, e.g. arXiv /pdf/ link"
    )

    
    comments: Optional[str] = Field(
        default=None, description="Optional extra notes, e.g. '20 pages, 10 tables'"
    )

    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this paper's metadata was ingested into our system",
    )

    @field_validator("paper_id")
    @classmethod
    def paper_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("paper_id must not be empty")
        return v.strip()

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, v):
        # Accept a single string ("cs.AI") or a list, always store as a list.
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v


if __name__ == "__main__":
    sample_paper = Paper(
        paper_id="2609.24974",
        title="Harness-Zero: Harness Distillation via Agent-as-Harness",
        authors=["Haoran Ye", "Yuxing Lu", "Haonan Dong", "Zhaochen Su", "Guojie Song"],
        abstract="This paper introduces...",
        categories=["cs.AI", "cs.CL", "cs.NE"],
        publish_date="2026-09-22",
        pdf_url="https://arxiv.org/pdf/2609.24974",
    )
    print(sample_paper.model_dump_json(indent=2))
