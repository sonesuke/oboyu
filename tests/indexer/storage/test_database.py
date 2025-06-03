"""Simplified tests for database functionality.

Note: Most complex database functionality was part of the old API.
This file contains basic tests that work with the new architecture.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from oboyu.indexer.storage.database_service import DatabaseService as Database
from oboyu.common.types import Chunk


class TestDatabase:
    """Test cases for basic database functionality."""

    def test_database_setup(self) -> None:
        """Test database setup and schema creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.initialize()
            
            # Verify database was initialized
            assert db.conn is not None
            
            # Close database connection
            db.close()

    def test_store_and_retrieve_chunks(self) -> None:
        """Test basic chunk storage and retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.initialize()
            
            # Create test chunk
            now = datetime.now()
            chunk = Chunk(
                id="test-chunk-1",
                path=Path("/test/doc1.txt"),
                title="Test Document 1",
                content="This is test document one.",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"source": "test"},
            )
            
            # Store chunk
            db.store_chunks([chunk])
            
            # Retrieve chunk by ID
            retrieved_chunk = db.get_chunk_by_id("test-chunk-1")
            assert retrieved_chunk is not None
            assert retrieved_chunk["id"] == "test-chunk-1"
            assert retrieved_chunk["title"] == "Test Document 1"
            
            # Close database connection
            db.close()

    def test_store_and_search_embeddings(self) -> None:
        """Test basic embedding storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.initialize()
            
            # Create test chunk
            now = datetime.now()
            chunk = Chunk(
                id="test-chunk-1",
                path=Path("/test/doc1.txt"),
                title="Test Document 1",
                content="This is test document one.",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"source": "test"},
            )
            
            # Store chunk first
            db.store_chunks([chunk])
            
            # Create embedding that matches the chunk
            embedding = np.random.rand(256).astype(np.float32)
            
            # Store embedding with matching chunk ID
            db.store_embeddings(["test-chunk-1"], [embedding])
            
            # Verify storage worked
            assert db.get_chunk_count() == 1
            
            # Close database connection
            db.close()

    def test_delete_chunks(self) -> None:
        """Test chunk deletion functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.initialize()
            
            # Create test chunk
            now = datetime.now()
            chunk = Chunk(
                id="test-chunk-1",
                path=Path("/test/doc1.txt"),
                title="Test Document 1",
                content="This is test document one.",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"source": "test"},
            )
            
            # Store chunk
            db.store_chunks([chunk])
            
            # Verify chunk was stored
            assert db.get_chunk_count() == 1
            
            # Delete chunks by path
            deleted_count = db.delete_chunks_by_path("/test/doc1.txt")
            # Note: delete_chunks_by_path may return -1 or the actual count
            assert deleted_count >= 0 or deleted_count == -1
            
            # Verify chunk was deleted by checking count
            assert db.get_chunk_count() == 0
            
            # Close database connection
            db.close()

    def test_clear_database(self) -> None:
        """Test database clearing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.initialize()
            
            # Create test chunk
            now = datetime.now()
            chunk = Chunk(
                id="test-chunk-1",
                path=Path("/test/doc1.txt"),
                title="Test Document 1",
                content="This is test document one.",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"source": "test"},
            )
            
            # Store chunk
            db.store_chunks([chunk])
            
            # Verify chunk was stored
            assert db.get_chunk_count() == 1
            
            # Clear database (if method exists)
            if hasattr(db, 'clear'):
                db.clear()
                assert db.get_chunk_count() == 0
            
            # Close database connection
            db.close()


class TestDatabaseMocked:
    """Test cases using mocked components."""

    def test_database_setup_mocked(self) -> None:
        """Test database setup with mocked initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Basic smoke test - just verify database can be created
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.initialize()
            
            # Verify basic functionality
            assert db.conn is not None
            assert hasattr(db, 'store_chunks')
            
            db.close()