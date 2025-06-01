"""Legacy import compatibility for embedding module."""

# Re-export from the new location
from oboyu.indexer.models.embedding_service import EmbeddingCache, EmbeddingService

# Legacy alias
EmbeddingGenerator = EmbeddingService

__all__ = ["EmbeddingCache", "EmbeddingGenerator", "EmbeddingService"]
