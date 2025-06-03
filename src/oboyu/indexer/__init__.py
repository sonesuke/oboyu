"""Document indexer module for Oboyu.

This module is responsible for processing and indexing document chunks,
generating embeddings, and managing the vector database for semantic search.
It provides specialized handling for Japanese content with the Ruri v3 model.
"""

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.indexer import Indexer

# Legacy alias for tests
LegacyIndexer = Indexer

__all__ = [
    "Indexer",
    "IndexerConfig",
]
