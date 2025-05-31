"""Document indexer module for Oboyu.

This module is responsible for processing and indexing document chunks,
generating embeddings, and managing the vector database for semantic search.
It provides specialized handling for Japanese content with the Ruri v3 model.
"""

from oboyu.indexer.change_detector import ChangeResult, FileChangeDetector
from oboyu.indexer.config import IndexerConfig, load_config_from_file, load_default_config
from oboyu.indexer.config.indexer_config import IndexerConfig as NewIndexerConfig
from oboyu.indexer.core.document_processor import Chunk, DocumentProcessor as NewDocumentProcessor
from oboyu.indexer.database import Database
from oboyu.indexer.embedding import EmbeddingGenerator
from oboyu.indexer.indexer import Indexer
from oboyu.indexer.new_indexer import NewIndexer
from oboyu.indexer.processor import DocumentProcessor
from oboyu.indexer.search.search_result import SearchResult

__all__ = [
    # Legacy components (for backward compatibility)
    "Indexer",
    "IndexerConfig",
    "Database",
    "DocumentProcessor",
    "EmbeddingGenerator",
    "FileChangeDetector",
    "ChangeResult",
    "load_default_config",
    "load_config_from_file",
    # New modular components
    "NewIndexer",
    "NewIndexerConfig",
    "NewDocumentProcessor",
    "Chunk",
    "SearchResult",
]
