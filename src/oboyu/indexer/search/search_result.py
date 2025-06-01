"""Unified search result format for all search types."""

from typing import Dict, Union

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Result of a search operation."""

    chunk_id: str = Field(description="ID of the matching chunk")
    path: str = Field(description="Path to the source document")
    title: str = Field(description="Title of the document or chunk")
    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(ge=0, description="Position of this chunk in the original document")
    language: str = Field(description="Language code of the content")
    metadata: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk"
    )
    score: float = Field(ge=0.0, le=1.0, description="Similarity score (0-1, where 1 is perfect match)")
