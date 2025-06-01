"""Tests for incremental indexing functionality.

This module tests the enhanced incremental indexing features including
file metadata tracking and differential updates.

Note: Most incremental indexing logic is tested in test_change_detector.py
This file contains integration tests that work with the new architecture.
"""

import pytest
from pathlib import Path

from oboyu.indexer.storage.change_detector import ChangeResult, FileChangeDetector
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig


class TestIncrementalIndexing:
    """Test incremental indexing functionality."""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return str(tmp_path / "test_index.db")
    
    @pytest.fixture
    def indexer_config(self, temp_db_path):
        """Create test indexer configuration."""
        processing_config = ProcessingConfig(
            db_path=Path(temp_db_path),
            chunk_size=512,
            chunk_overlap=128,
        )
        
        return IndexerConfig(
            model=ModelConfig(),
            processing=processing_config,
            search=SearchConfig(),
        )
    
    def test_change_detector_initialization(self, indexer_config):
        """Test that FileChangeDetector can be initialized."""
        from oboyu.indexer.storage.database_service import DatabaseService
        
        db_path = indexer_config.processing.db_path
        database = DatabaseService(db_path=db_path)
        database.initialize()
        
        change_detector = FileChangeDetector(database=database)
        
        assert change_detector.db == database
        assert change_detector.batch_size == 1000  # default value
        
        database.close()
    
    def test_change_result_structure(self):
        """Test ChangeResult data structure."""
        result = ChangeResult(
            new_files=[Path("new.txt")],
            modified_files=[Path("modified.txt")],
            deleted_files=[Path("deleted.txt")]
        )
        
        assert len(result.new_files) == 1
        assert len(result.modified_files) == 1
        assert len(result.deleted_files) == 1
        assert result.total_changes == 3
        assert result.has_changes() is True
        
        empty_result = ChangeResult(new_files=[], modified_files=[], deleted_files=[])
        assert empty_result.has_changes() is False