"""Tests for BM25 search functionality."""

import pytest
import tempfile
from pathlib import Path

from oboyu.indexer.storage.database_service import DatabaseService as Database


class TestBM25Search:
    """Test cases for BM25 search functionality."""

    @pytest.fixture
    def test_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            yield db_path

    def test_database_initialization(self, test_db_path):
        """Test that database can be initialized for BM25 operations."""
        db = Database(test_db_path)
        db.initialize()
        
        # Test that database was initialized successfully
        assert db.conn is not None
        
        db.close()

    def test_bm25_search_empty_database(self, test_db_path):
        """Test BM25 search on empty database."""
        db = Database(test_db_path)
        db.initialize()
        
        # Search should return empty results
        results = db.bm25_search(["python", "programming"])
        assert results == []
        
        db.close()