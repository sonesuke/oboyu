"""Data model components."""

# Import data models
from oboyu.indexer.models.database import ChunkRecord, EmbeddingRecord, VocabularyRecord
from oboyu.indexer.models.search import (
    EmbeddingModel,
    EmbeddingVector,
    IndexingStatus,
)

# Legacy imports for backward compatibility
try:
    from oboyu.indexer.services.embedding import EmbeddingCache, EmbeddingService
    # Legacy aliases for tests
    EmbeddingGenerator = EmbeddingService
except ImportError:
    EmbeddingCache = None
    EmbeddingService = None
    EmbeddingGenerator = None

__all__ = [
    # Data models
    "ChunkRecord",
    "EmbeddingRecord",
    "VocabularyRecord",
    "EmbeddingModel",
    "IndexingStatus",
    "EmbeddingVector",
    # Legacy compatibility
    "EmbeddingService",
    "EmbeddingCache",
    "EmbeddingGenerator",
]
