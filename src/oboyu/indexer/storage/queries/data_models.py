"""Type-safe data models for database operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

# Import DateTimeEncoder from utils


@dataclass
class ChunkData:
    """Type-safe data structure for chunk database operations."""

    id: str
    path: str
    title: str
    content: str
    chunk_index: int
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingData:
    """Type-safe data structure for embedding database operations."""

    id: str
    chunk_id: str
    model: str
    vector: Union[List[float], NDArray[np.float32]]
    created_at: Optional[datetime] = None


@dataclass
class BM25Data:
    """Type-safe data structure for BM25 index operations."""

    term: str
    chunk_id: str
    term_frequency: int
    positions: Optional[List[int]] = None


@dataclass
class VocabularyData:
    """Type-safe data structure for vocabulary operations."""

    term: str
    document_frequency: int
    collection_frequency: int


@dataclass
class DocumentStatsData:
    """Type-safe data structure for document statistics."""

    chunk_id: str
    total_terms: int
    unique_terms: int
    avg_term_frequency: float


@dataclass
class CollectionStatsData:
    """Type-safe data structure for collection statistics."""

    total_documents: int
    total_terms: int
    avg_document_length: float
    last_updated: Optional[datetime] = None
