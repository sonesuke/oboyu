"""Tests for file change detection functionality.

This module tests the FileChangeDetector class and its various
change detection strategies.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from oboyu.indexer.storage.change_detector import ChangeResult, FileChangeDetector


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
        result = ChangeResult(new_files=[], modified_files=[], deleted_files=[])
        
        assert result.total_changes == 0
        assert result.has_changes() is False


class TestFileChangeDetector:
    """Test FileChangeDetector class."""

    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        mock_db = MagicMock()
        # Mock empty metadata by default
        mock_conn = mock_db._ensure_connection.return_value
        mock_conn.execute.return_value.fetchall.return_value = []
        return mock_db

    @pytest.fixture
    def detector(self, mock_database):
        """Create a FileChangeDetector with mock database."""
        return FileChangeDetector(database=mock_database)

    def test_detector_initialization(self, mock_database):
        """Test FileChangeDetector initialization."""
        detector = FileChangeDetector(database=mock_database, batch_size=500)
        
        assert detector.db == mock_database
        assert detector.batch_size == 500

    def test_detect_changes_all_new_files(self, detector, mock_database):
        """Test detecting changes when all files are new."""
        file_paths = [
            Path("/test/file1.txt"),
            Path("/test/file2.txt"),
            Path("/test/file3.txt"),
        ]
        
        # Mock empty database (no existing files)
        mock_conn = mock_database._ensure_connection.return_value
        mock_conn.execute.return_value.fetchall.return_value = []
        
        result = detector.detect_changes(file_paths)
        
        # All files should be considered new when database is empty
        assert len(result.new_files) == 3
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0
        assert result.total_changes == 3

    def test_detect_changes_empty_file_list(self, detector):
        """Test detecting changes with empty file list."""
        result = detector.detect_changes([])
        
        assert len(result.new_files) == 0
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0
        assert result.has_changes() is False