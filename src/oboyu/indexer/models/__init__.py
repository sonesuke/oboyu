"""Model service components."""

from oboyu.indexer.models.embedding_service import EmbeddingCache, EmbeddingService

# Legacy aliases for tests
EmbeddingGenerator = EmbeddingService

__all__ = ["EmbeddingService", "EmbeddingCache", "EmbeddingGenerator"]
