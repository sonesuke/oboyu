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

    def test_indexer_initialization(self) -> None:
        """Test indexer initialization with configuration."""
        # Create config
        processing_config = ProcessingConfig(db_path=Path("test.db"))
        config = IndexerConfig(processing=processing_config)
        
        # Initialize the indexer
        indexer = Indexer(config=config)
        
        # Verify basic initialization
        assert indexer.config is not None
        assert indexer.config.chunk_size == 1024  # Current default

    def test_index_documents(self) -> None:
        """Test that indexer can be created for document indexing."""
        config = IndexerConfig()
        indexer = Indexer(config=config)
        
        # Basic smoke test - just verify the indexer was created
        assert indexer is not None
        assert hasattr(indexer, 'index_documents')

    def test_search(self) -> None:
        """Test that indexer has search functionality."""
        config = IndexerConfig()
        indexer = Indexer(config=config)
        
        # Basic smoke test - just verify search method exists
        assert hasattr(indexer, 'search')

    def test_delete_document(self) -> None:
        """Test that indexer has delete functionality."""
        config = IndexerConfig()
        indexer = Indexer(config=config)
        
        # Basic smoke test - just verify delete method exists
        assert hasattr(indexer, 'delete_document')

    def test_clear_index(self) -> None:
        """Test that indexer has clear functionality."""
        config = IndexerConfig()
        indexer = Indexer(config=config)
        
        # Basic smoke test - just verify clear method exists
        assert hasattr(indexer, 'clear_index')