"""Document chunk model for indexing and retrieval."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Chunk:
    """Document chunk for indexing."""

    id: str
    """Unique identifier for the chunk."""

    path: Path
    """Path to the source document."""

    title: str
    """Title of the document or chunk."""

    content: str
    """Chunk text content."""

    chunk_index: int
    """Position of this chunk in the original document."""

    language: str
    """Language code of the content."""

    created_at: datetime
    """Timestamp when the chunk was created."""

    modified_at: datetime
    """Timestamp when the chunk was last modified."""

    metadata: Dict[str, object]
    """Additional metadata about the chunk."""

    prefix_content: Optional[str] = None
    """Prefix content for embedding (e.g., document title or topic)."""

    prefix_type: Optional[str] = None
    """Type of prefix used (e.g., 'document', 'query', 'topic')."""
