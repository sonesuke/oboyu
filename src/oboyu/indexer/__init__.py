"""Document indexer module for Oboyu.

This module is responsible for processing and indexing document chunks,
generating embeddings, and managing the vector database for semantic search.
It provides specialized handling for Japanese content with the Ruri v3 model.
"""

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.core.document_processor import Chunk, DocumentProcessor
from oboyu.indexer.indexer import Indexer  # New modular indexer (default)
from oboyu.indexer.search.search_result import SearchResult
from oboyu.indexer.storage.change_detector import ChangeResult, FileChangeDetector
from oboyu.indexer.storage.database_service import DatabaseService

# Legacy alias for tests
LegacyIndexer = Indexer

__all__ = [
    # Primary components (New architecture)
    "Indexer",
    "LegacyIndexer",  # Legacy alias
    "IndexerConfig",
    "DocumentProcessor",
    "Chunk",
    "SearchResult",
    # Storage and utilities
    "DatabaseService",
    "FileChangeDetector",
    "ChangeResult",
]
