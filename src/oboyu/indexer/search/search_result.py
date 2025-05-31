"""Unified search result format for all search types."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SearchResult:
    """Result of a search operation."""

    chunk_id: str
    """ID of the matching chunk."""

    path: str
    """Path to the source document."""

    title: str
    """Title of the document or chunk."""

    content: str
    """Chunk text content."""

    chunk_index: int
    """Position of this chunk in the original document."""

    language: str
    """Language code of the content."""

    metadata: Dict[str, object]
    """Additional metadata about the chunk."""

    score: float
    """Similarity score (0-1, where 1 is perfect match)."""
