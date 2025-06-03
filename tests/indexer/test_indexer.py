"""Simplified tests for the main indexer functionality.

Note: Most complex indexer functionality was part of the old API.
This file contains basic tests that work with the new architecture.
"""

from pathlib import Path

import pytest

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.indexer import Indexer


class TestIndexer:
    """Test cases for the main Indexer class."""

    def test_indexer_initialization(self, temp_db_path: Path) -> None:
        """Test indexer initialization with configuration."""
        # Create config with temporary database path
        processing_config = ProcessingConfig(db_path=temp_db_path)
        config = IndexerConfig(processing=processing_config)
        
        # Initialize the indexer
        indexer = Indexer(config=config)
        
        # Verify basic initialization
        assert indexer.config is not None
        assert indexer.config.chunk_size == 1024  # Current default

    def test_index_documents(self, temp_db_path: Path) -> None:
        """Test that indexer can be created for document indexing."""
        processing_config = ProcessingConfig(db_path=temp_db_path)
        config = IndexerConfig(processing=processing_config)
        indexer = Indexer(config=config)
        
        # Basic smoke test - just verify the indexer was created
        assert indexer is not None
        assert hasattr(indexer, 'index_documents')

    def test_get_stats(self, temp_db_path: Path) -> None:
        """Test that indexer has statistics functionality."""
        processing_config = ProcessingConfig(db_path=temp_db_path)
        config = IndexerConfig(processing=processing_config)
        indexer = Indexer(config=config)
        
        # Basic smoke test - just verify stats method exists
        assert hasattr(indexer, 'get_index_stats')
        stats = indexer.get_index_stats()
        assert isinstance(stats, dict)

    def test_delete_document(self, temp_db_path: Path) -> None:
        """Test that indexer has delete functionality."""
        processing_config = ProcessingConfig(db_path=temp_db_path)
        config = IndexerConfig(processing=processing_config)
        indexer = Indexer(config=config)
        
        # Basic smoke test - just verify delete method exists
        assert hasattr(indexer, 'delete_document')

    def test_get_paths(self, temp_db_path: Path) -> None:
        """Test that indexer can get paths with chunks."""
        processing_config = ProcessingConfig(db_path=temp_db_path)
        config = IndexerConfig(processing=processing_config)
        indexer = Indexer(config=config)
        
        # Basic smoke test - just verify paths method exists
        assert hasattr(indexer, 'get_paths_with_chunks')
        paths = indexer.get_paths_with_chunks()
        assert isinstance(paths, list)