"""Document indexer module for Oboyu.

This module is responsible for processing and indexing document chunks,
generating embeddings, and managing the vector database for semantic search.
It provides specialized handling for Japanese content with the Ruri v3 model.
"""

from oboyu.indexer.config import IndexerConfig, load_config_from_file, load_default_config
from oboyu.indexer.database import Database
from oboyu.indexer.embedding import EmbeddingGenerator
from oboyu.indexer.indexer import Indexer
from oboyu.indexer.processor import DocumentProcessor

__all__ = [
    "Indexer",
    "IndexerConfig",
    "Database",
    "DocumentProcessor",
    "EmbeddingGenerator",
    "load_default_config",
    "load_config_from_file",
]
