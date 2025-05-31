"""Tests for file change detection functionality.

This module tests the FileChangeDetector class and its various
change detection strategies.
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oboyu.indexer.change_detector import ChangeResult, FileChangeDetector


class TestChangeResult:
    """Test ChangeResult data class."""
    
    def test_change_result_properties(self):
        """Test ChangeResult properties and methods."""
        result = ChangeResult(
            new_files=[Path("file1.txt"), Path("file2.txt")],
            modified_files=[Path("file3.txt")],
            deleted_files=[Path("file4.txt"), Path("file5.txt")]
        )
        
        assert result.total_changes == 5
        assert result.has_changes() is True
        assert len(result.new_files) == 2
        assert len(result.modified_files) == 1
        assert len(result.deleted_files) == 2
    
    def test_empty_change_result(self):
        """Test empty ChangeResult."""
        result = ChangeResult(
            new_files=[],
            modified_files=[],
            deleted_files=[]
        )
        
        assert result.total_changes == 0
        assert result.has_changes() is False


class TestFileChangeDetector:
    """Test FileChangeDetector class."""
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        return mock_db
    
    @pytest.fixture
    def detector(self, mock_database):
        """Create a FileChangeDetector instance for testing."""
        return FileChangeDetector(mock_database)
    
    def test_detect_changes_all_new_files(self, detector, mock_database):
        """Test detecting changes when all files are new."""
        # Mock empty database (no existing files)
        mock_database.execute.return_value.fetchall.return_value = []
        
        file_paths = [
            Path("/test/file1.txt"),
            Path("/test/file2.txt"),
            Path("/test/file3.txt")
        ]
        
        result = detector.detect_changes(file_paths)
        
        assert len(result.new_files) == 3
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0
        assert result.total_changes == 3
    
    def test_detect_changes_with_existing_files(self, detector, mock_database):
        """Test detecting changes with some existing files."""
        # Mock existing files in database
        existing_files = [
            ("/test/file1.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 1000, "hash1"),
            ("/test/file2.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 2000, "hash2"),
        ]
        mock_database.execute.return_value.fetchall.return_value = existing_files
        
        file_paths = [
            Path("/test/file1.txt"),  # Existing
            Path("/test/file2.txt"),  # Existing
            Path("/test/file3.txt"),  # New
        ]
        
        # Mock file stats to show no modifications
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_mtime = 1704067200  # 2024-01-01 00:00:00
            mock_stat.return_value.st_size = 1000
            
            result = detector.detect_changes(file_paths, strategy="timestamp")
        
        assert len(result.new_files) == 1
        assert result.new_files[0] == Path("/test/file3.txt")
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0
    
    def test_detect_changes_with_deleted_files(self, detector, mock_database):
        """Test detecting deleted files."""
        # Mock existing files in database
        existing_files = [
            ("/test/file1.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 1000, "hash1"),
            ("/test/file2.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 2000, "hash2"),
            ("/test/file3.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 3000, "hash3"),
        ]
        mock_database.execute.return_value.fetchall.return_value = existing_files
        
        # Only file1 and file2 exist on disk now
        file_paths = [
            Path("/test/file1.txt"),
            Path("/test/file2.txt"),
        ]
        
        result = detector.detect_changes(file_paths)
        
        assert len(result.new_files) == 0
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 1
        assert result.deleted_files[0] == Path("/test/file3.txt")
    
    def test_timestamp_strategy(self, detector, mock_database):
        """Test timestamp-based change detection."""
        # Mock existing file with old timestamp
        existing_files = [
            ("/test/file1.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 1000, "hash1"),
        ]
        mock_database.execute.return_value.fetchall.return_value = existing_files
        
        file_paths = [Path("/test/file1.txt")]
        
        # Mock file with newer modification time
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_mtime = 1704153600  # 2024-01-02 00:00:00 (1 day later)
            mock_stat.return_value.st_size = 1000
            
            result = detector.detect_changes(file_paths, strategy="timestamp")
        
        assert len(result.new_files) == 0
        assert len(result.modified_files) == 1
        assert result.modified_files[0] == Path("/test/file1.txt")
        assert len(result.deleted_files) == 0
    
    def test_hash_strategy(self, detector, mock_database):
        """Test hash-based change detection."""
        # Mock existing file
        existing_files = [
            ("/test/file1.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 1000, "old_hash"),
        ]
        mock_database.execute.return_value.fetchall.return_value = existing_files
        
        file_paths = [Path("/test/file1.txt")]
        
        # Mock file stats (same timestamp and size)
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_mtime = 1704067200  # Same as stored
            mock_stat.return_value.st_size = 1000  # Same as stored
            
            # Mock different hash
            with patch.object(FileChangeDetector, 'calculate_file_hash', return_value="new_hash"):
                result = detector.detect_changes(file_paths, strategy="hash")
        
        assert len(result.new_files) == 0
        assert len(result.modified_files) == 1
        assert result.modified_files[0] == Path("/test/file1.txt")
        assert len(result.deleted_files) == 0
    
    def test_smart_strategy(self, detector, mock_database):
        """Test smart change detection strategy."""
        # Mock existing files
        existing_files = [
            ("/test/file1.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 1000, "hash1"),
            ("/test/file2.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 2000, "hash2"),
            ("/test/file3.txt", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 15000000, "hash3"),  # Large file
        ]
        mock_database.execute.return_value.fetchall.return_value = existing_files
        
        file_paths = [
            Path("/test/file1.txt"),  # Same timestamp but different size
            Path("/test/file2.txt"),  # Different timestamp
            Path("/test/file3.txt"),  # Large file, same timestamp/size
        ]
        
        # Mock file stats
        def stat_side_effect(*args, **kwargs):
            mock = MagicMock()
            if "file1" in str(args[0]) if args else "":
                mock.st_mtime = 1704067200  # Same timestamp
                mock.st_size = 1500  # Different size
            elif "file2" in str(args[0]) if args else "":
                mock.st_mtime = 1704153600  # Different timestamp
                mock.st_size = 2000  # Same size
            else:  # file3
                mock.st_mtime = 1704067200  # Same timestamp
                mock.st_size = 15000000  # Same size (large file)
            return mock
        
        with patch.object(Path, 'stat', side_effect=stat_side_effect):
            result = detector.detect_changes(file_paths, strategy="smart")
        
        # file1: modified due to size change
        # file2: modified due to timestamp change
        # file3: not modified (large file with same timestamp/size)
        assert len(result.new_files) == 0
        assert len(result.modified_files) == 2
        assert Path("/test/file1.txt") in result.modified_files
        assert Path("/test/file2.txt") in result.modified_files
        assert len(result.deleted_files) == 0
    
    def test_calculate_file_hash(self, tmp_path):
        """Test file hash calculation."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World! This is a test file."
        test_file.write_bytes(test_content)
        
        # Calculate hash
        hash_result = FileChangeDetector.calculate_file_hash(test_file)
        
        # Verify it's a valid hash (64 characters for SHA-256)
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)
        
        # Verify same content produces same hash
        hash_result2 = FileChangeDetector.calculate_file_hash(test_file)
        assert hash_result == hash_result2
        
        # Verify different content produces different hash
        test_file.write_bytes(b"Different content")
        hash_result3 = FileChangeDetector.calculate_file_hash(test_file)
        assert hash_result != hash_result3
    
    def test_mark_files_for_reprocessing(self, detector, mock_database):
        """Test marking files for reprocessing."""
        file_paths = [
            Path("/test/file1.txt"),
            Path("/test/file2.txt"),
        ]
        
        detector.mark_files_for_reprocessing(file_paths)
        
        # Verify database execute was called for each file
        assert mock_database.execute.call_count == 2
        
        # Check the SQL and parameters
        calls = mock_database.execute.call_args_list
        for i, call in enumerate(calls):
            sql = call[0][0]
            params = call[0][1]
            assert "UPDATE file_metadata" in sql
            assert "processing_status = 'pending'" in sql
            assert params == [str(file_paths[i])]
    
    def test_get_processing_stats(self, detector, mock_database):
        """Test getting processing statistics."""
        # Mock database response
        mock_database.execute.return_value.fetchall.return_value = [
            ("completed", 100),
            ("pending", 25),
            ("error", 5),
        ]
        
        stats = detector.get_processing_stats()
        
        assert stats["completed"] == 100
        assert stats["pending"] == 25
        assert stats["error"] == 5
        assert stats["total"] == 130
    
    def test_cleanup_deleted_files(self, detector, mock_database):
        """Test cleanup of deleted files."""
        deleted_files = [
            Path("/test/file1.txt"),
            Path("/test/file2.txt"),
            Path("/test/file3.txt"),
        ]
        
        detector.cleanup_deleted_files(deleted_files)
        
        # Verify database execute was called for each file
        assert mock_database.execute.call_count == 3
        
        # Check the SQL and parameters
        calls = mock_database.execute.call_args_list
        for i, call in enumerate(calls):
            sql = call[0][0]
            params = call[0][1]
            assert "DELETE FROM file_metadata" in sql
            assert params == [str(deleted_files[i])]
    
    def test_error_handling_in_change_detection(self, detector, mock_database):
        """Test error handling during change detection."""
        # Mock database error
        mock_database.execute.side_effect = Exception("Database error")
        
        file_paths = [Path("/test/file1.txt")]
        
        # Should return empty metadata dict on error
        result = detector.detect_changes(file_paths)
        
        # All files should be considered new when metadata fetch fails
        assert len(result.new_files) == 1
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0