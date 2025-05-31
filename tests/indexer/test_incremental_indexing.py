"""Tests for incremental indexing functionality.

This module tests the enhanced incremental indexing features including
file metadata tracking and differential updates.
"""

import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from oboyu.indexer.change_detector import ChangeResult, FileChangeDetector
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer


class TestIncrementalIndexing:
    """Test incremental indexing functionality."""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return str(tmp_path / "test_index.db")
    
    @pytest.fixture
    def indexer_config(self, temp_db_path):
        """Create test indexer configuration."""
        return IndexerConfig(config_dict={
            "indexer": {
                "db_path": temp_db_path,
                "chunk_size": 512,
                "chunk_overlap": 128,
                "incremental": {
                    "enabled": True,
                    "change_detection_strategy": "smart",
                    "cleanup_deleted_files": True,
                }
            }
        })
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        mock_processor = MagicMock()
        mock_embedding_generator = MagicMock()
        mock_database = MagicMock()
        mock_reranker = MagicMock()
        mock_bm25_indexer = MagicMock()
        
        # Configure mock embedding generator
        mock_embedding_generator.dimensions = 256
        mock_embedding_generator.generate_embeddings.return_value = []
        
        # Configure mock database
        mock_database.execute.return_value.fetchall.return_value = []
        mock_database.store_chunks.return_value = None
        mock_database.store_embeddings.return_value = None
        
        return {
            "processor": mock_processor,
            "embedding_generator": mock_embedding_generator,
            "database": mock_database,
            "reranker": mock_reranker,
            "bm25_indexer": mock_bm25_indexer,
        }
    
    def test_file_metadata_storage(self, indexer_config, mock_components, tmp_path):
        """Test that file metadata is stored after indexing."""
        # Create test files
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for indexing")
        
        # Create mock crawler results
        mock_crawler_result = MagicMock()
        mock_crawler_result.path = test_file
        mock_crawler_result.content = "Test content"
        mock_crawler_result.metadata = {}
        
        # Create mock chunks
        mock_chunk = MagicMock()
        mock_chunk.path = test_file
        mock_chunk.content = "Test chunk"
        mock_components["processor"].process_documents.return_value = [mock_chunk]
        
        # Create indexer
        indexer = Indexer(
            config=indexer_config,
            **mock_components
        )
        
        # Index documents
        indexer.index_documents([mock_crawler_result])
        
        # Verify file metadata was stored
        db_calls = mock_components["database"].execute.call_args_list
        
        # Find the file metadata INSERT/REPLACE call
        metadata_call_found = False
        for call_item in db_calls:
            sql = call_item[0][0]
            if "INSERT OR REPLACE INTO file_metadata" in sql:
                metadata_call_found = True
                params = call_item[0][1]
                
                # Verify parameters
                assert params[0] == str(test_file)  # path
                assert params[5] == 1  # chunk_count
                assert params[6] == 'completed'  # processing_status
                assert params[7] is None  # error_message
                break
        
        assert metadata_call_found, "File metadata was not stored"
    
    def test_incremental_indexing_skips_unchanged_files(self, indexer_config, mock_components, tmp_path):
        """Test that incremental indexing skips unchanged files."""
        # Create test directory
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        
        test_file1 = test_dir / "file1.txt"
        test_file2 = test_dir / "file2.txt"
        test_file1.write_text("Content 1")
        test_file2.write_text("Content 2")
        
        # Mock change detector to return only file1 as new
        mock_change_result = ChangeResult(
            new_files=[test_file1],
            modified_files=[],
            deleted_files=[]
        )
        
        with patch.object(FileChangeDetector, 'detect_changes', return_value=mock_change_result):
            # Create indexer
            indexer = Indexer(
                config=indexer_config,
                **mock_components
            )
            
            # Mock crawler to filter files properly
            with patch('oboyu.indexer.indexer.Crawler') as MockCrawler:
                mock_crawler_instance = MockCrawler.return_value
                
                # Track which files the crawler actually processes
                processed_files = []
                
                def crawl_side_effect(directory, progress_callback=None):
                    # The crawler should only process file1
                    result = MagicMock()
                    result.path = test_file1
                    result.content = "Content 1"
                    result.metadata = {}
                    processed_files.append(test_file1)
                    return [result]
                
                mock_crawler_instance.crawl.side_effect = crawl_side_effect
                
                # Index directory incrementally
                chunks_indexed, files_processed = indexer.index_directory(
                    test_dir,
                    incremental=True
                )
                
                # Verify only file1 was processed
                assert files_processed == 1
                assert test_file1 in processed_files
                assert test_file2 not in processed_files
    
    def test_deleted_file_cleanup(self, indexer_config, mock_components, tmp_path):
        """Test that deleted files are cleaned up during incremental indexing."""
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        
        # Mock change detector to return deleted files
        deleted_file = Path("/old/deleted/file.txt")
        mock_change_result = ChangeResult(
            new_files=[],
            modified_files=[],
            deleted_files=[deleted_file]
        )
        
        with patch.object(FileChangeDetector, 'detect_changes', return_value=mock_change_result):
            # Create indexer
            indexer = Indexer(
                config=indexer_config,
                **mock_components
            )
            
            # Mock database delete response
            mock_components["database"].delete_chunks_by_path.return_value = 5  # 5 chunks deleted
            
            with patch('oboyu.indexer.indexer.Crawler') as MockCrawler:
                mock_crawler_instance = MockCrawler.return_value
                mock_crawler_instance.crawl.return_value = []  # No new files
                
                # Index directory with cleanup enabled
                chunks_indexed, files_processed = indexer.index_directory(
                    test_dir,
                    incremental=True,
                    cleanup_deleted=True
                )
                
                # Verify delete_document was called for the deleted file
                mock_components["database"].delete_chunks_by_path.assert_called_with(deleted_file)
                
                # Verify file metadata was deleted
                db_calls = mock_components["database"].execute.call_args_list
                metadata_delete_found = False
                for call_item in db_calls:
                    sql = call_item[0][0]
                    if "DELETE FROM file_metadata" in sql and "WHERE path = ?" in sql:
                        metadata_delete_found = True
                        params = call_item[0][1]
                        assert params[0] == str(deleted_file)
                        break
                
                assert metadata_delete_found, "File metadata was not deleted"
    
    def test_change_detection_strategies(self, indexer_config, mock_components, tmp_path):
        """Test different change detection strategies."""
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        
        strategies_tested = []
        
        def mock_detect_changes(file_paths, strategy="smart"):
            strategies_tested.append(strategy)
            return ChangeResult(new_files=file_paths, modified_files=[], deleted_files=[])
        
        with patch.object(FileChangeDetector, 'detect_changes', side_effect=mock_detect_changes):
            # Create indexer
            indexer = Indexer(
                config=indexer_config,
                **mock_components
            )
            
            with patch('oboyu.indexer.indexer.Crawler') as MockCrawler:
                mock_crawler_instance = MockCrawler.return_value
                mock_crawler_instance.crawl.return_value = []
                
                # Test timestamp strategy
                indexer.index_directory(test_dir, incremental=True, change_detection_strategy="timestamp")
                assert "timestamp" in strategies_tested
                
                # Test hash strategy
                indexer.index_directory(test_dir, incremental=True, change_detection_strategy="hash")
                assert "hash" in strategies_tested
                
                # Test smart strategy (default)
                indexer.index_directory(test_dir, incremental=True)
                assert "smart" in strategies_tested
    
    def test_clear_index_clears_file_metadata(self, indexer_config, mock_components):
        """Test that clear_index also clears file metadata."""
        # Create indexer
        indexer = Indexer(
            config=indexer_config,
            **mock_components
        )
        
        # Clear index
        indexer.clear_index()
        
        # Verify file metadata was cleared
        db_calls = mock_components["database"].execute.call_args_list
        metadata_clear_found = False
        for call_item in db_calls:
            sql = call_item[0][0]
            if sql == "DELETE FROM file_metadata":
                metadata_clear_found = True
                break
        
        assert metadata_clear_found, "File metadata was not cleared"
        
        # Also verify other components were cleared
        mock_components["database"].clear.assert_called_once()
        mock_components["bm25_indexer"].clear.assert_called_once()
    
    def test_force_reindex_ignores_incremental(self, indexer_config, mock_components, tmp_path):
        """Test that force reindex ignores incremental settings."""
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        
        change_detect_called = False
        
        def mock_detect_changes(*args, **kwargs):
            nonlocal change_detect_called
            change_detect_called = True
            return ChangeResult(new_files=[], modified_files=[], deleted_files=[])
        
        with patch.object(FileChangeDetector, 'detect_changes', side_effect=mock_detect_changes):
            # Create indexer
            indexer = Indexer(
                config=indexer_config,
                **mock_components
            )
            
            with patch('oboyu.indexer.indexer.Crawler') as MockCrawler:
                mock_crawler_instance = MockCrawler.return_value
                mock_crawler_instance.crawl.return_value = []
                
                # Index with incremental=False (force)
                indexer.index_directory(test_dir, incremental=False)
                
                # Change detection should not be called when forcing
                assert not change_detect_called
                
                # Crawler should have empty processed files set
                assert mock_crawler_instance._processed_files == set()
    
    def test_file_metadata_error_handling(self, indexer_config, mock_components, tmp_path):
        """Test error handling when storing file metadata fails."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Create mock crawler result
        mock_crawler_result = MagicMock()
        mock_crawler_result.path = test_file
        mock_crawler_result.content = "Test content"
        mock_crawler_result.metadata = {}
        
        # Create mock chunk
        mock_chunk = MagicMock()
        mock_chunk.path = test_file
        mock_chunk.content = "Test chunk"
        mock_components["processor"].process_documents.return_value = [mock_chunk]
        
        # Make file metadata storage fail
        def execute_side_effect(sql, params=None):
            if "INSERT OR REPLACE INTO file_metadata" in sql:
                raise Exception("Database error")
            # Return empty result for other queries
            result = MagicMock()
            result.fetchall.return_value = []
            result.fetchone.return_value = None
            return result
        
        mock_components["database"].execute.side_effect = execute_side_effect
        
        # Create indexer
        indexer = Indexer(
            config=indexer_config,
            **mock_components
        )
        
        # Index documents - should not raise exception
        chunks_indexed = indexer.index_documents([mock_crawler_result])
        
        # Indexing should still succeed even if metadata storage fails
        assert chunks_indexed == 1