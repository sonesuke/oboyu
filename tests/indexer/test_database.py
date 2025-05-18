"""Tests for the database management functionality."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from oboyu.indexer.database import Database
from oboyu.indexer.processor import Chunk


# VSS extension is now available as standard with DuckDB
class TestDatabase:
    """Test cases for the database management using DuckDB with VSS extension.
    
    The VSS extension is a system library that comes with DuckDB and enables
    vector similarity search capabilities.
    """

    def test_database_setup(self) -> None:
        """Test database setup and schema creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.setup()
            
            # Verify tables exist
            result = db.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('chunks', 'embeddings')
            """).fetchall()
            
            assert len(result) == 2  # Both tables should exist
            
            # Verify indexes exist
            result = db.conn.execute("""
                SELECT COUNT(*) FROM duckdb_indexes()
            """).fetchone()
            
            assert result[0] > 0  # At least one index should exist
            
            # Close database connection
            db.close()

    def test_store_and_retrieve_chunks(self) -> None:
        """Test storing and retrieving chunks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.setup()
            
            # Create test chunks
            now = datetime.now()
            chunks = [
                Chunk(
                    id="test-chunk-1",
                    path=Path("/test/doc1.txt"),
                    title="Test Document 1",
                    content="This is test document one.",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "test"},
                    prefix_content="検索文書: This is test document one.",
                ),
                Chunk(
                    id="test-chunk-2",
                    path=Path("/test/doc2.txt"),
                    title="Test Document 2",
                    content="This is test document two.",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "test"},
                    prefix_content="検索文書: This is test document two.",
                ),
            ]
            
            # Store chunks
            db.store_chunks(chunks)
            
            # Retrieve chunk by ID
            chunk = db.get_chunk_by_id("test-chunk-1")
            assert chunk is not None
            assert chunk["id"] == "test-chunk-1"
            assert chunk["title"] == "Test Document 1"
            assert chunk["content"] == "This is test document one."
            
            # Retrieve chunks by path
            path_chunks = db.get_chunks_by_path("/test/doc1.txt")
            assert len(path_chunks) == 1
            assert path_chunks[0]["id"] == "test-chunk-1"
            
            # Close database connection
            db.close()

    def test_store_and_search_embeddings(self) -> None:
        """Test storing and searching embeddings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.setup()
            
            # Create test chunks
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
                prefix_content="検索文書: This is test document one.",
            )
            
            # Store chunk
            db.store_chunks([chunk])
            
            # Create test embedding with proper dtype
            embedding_id = "test-embedding-1"
            vector = np.random.rand(256).astype(np.float32)  # Random 256-dim vector with float32 type
            embeddings = [(embedding_id, chunk.id, vector, now)]
            
            # Store embedding
            db.store_embeddings(embeddings, "test-model")
            
            # Search with the same vector (should be exact match)
            results = db.search(vector, limit=1)
            assert len(results) == 1
            assert results[0]["chunk_id"] == "test-chunk-1"
            assert results[0]["score"] < 0.0001  # Should be very close to 0 for exact match
            
            # Close database connection
            db.close()

    def test_delete_chunks(self) -> None:
        """Test deleting chunks and their embeddings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.setup()
            
            # Create test chunks for two documents
            now = datetime.now()
            chunks = [
                Chunk(
                    id="test-chunk-1",
                    path=Path("/test/doc1.txt"),
                    title="Test Document 1",
                    content="This is test document one.",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "test"},
                ),
                Chunk(
                    id="test-chunk-2",
                    path=Path("/test/doc2.txt"),
                    title="Test Document 2",
                    content="This is test document two.",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "test"},
                ),
            ]
            
            # Store chunks
            db.store_chunks(chunks)
            
            # Create test embeddings
            embeddings = [
                ("test-embedding-1", "test-chunk-1", np.random.rand(256), now),
                ("test-embedding-2", "test-chunk-2", np.random.rand(256), now),
            ]
            
            # Store embeddings
            db.store_embeddings(embeddings, "test-model")
            
            # Delete chunks for doc1
            deleted_count = db.delete_chunks_by_path("/test/doc1.txt")
            assert deleted_count == 1
            
            # Verify chunk and its embedding are gone
            assert db.get_chunk_by_id("test-chunk-1") is None
            
            # Verify doc2 still exists
            assert db.get_chunk_by_id("test-chunk-2") is not None
            
            # Close database connection
            db.close()
            
    def test_clear_database(self) -> None:
        """Test clearing all data from the database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.db"
            
            # Initialize database
            db = Database(db_path=temp_file_path, embedding_dimensions=256)
            db.setup()
            
            # Create and store test chunks
            now = datetime.now()
            chunks = [
                Chunk(
                    id="test-chunk-1",
                    path=Path("/test/doc1.txt"),
                    title="Test Document 1",
                    content="This is test document one.",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "test"},
                ),
                Chunk(
                    id="test-chunk-2",
                    path=Path("/test/doc2.txt"),
                    title="Test Document 2",
                    content="This is test document two.",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "test"},
                ),
            ]
            db.store_chunks(chunks)
            
            # Create and store test embeddings
            embeddings = [
                ("test-embedding-1", "test-chunk-1", np.random.rand(256), now),
                ("test-embedding-2", "test-chunk-2", np.random.rand(256), now),
            ]
            db.store_embeddings(embeddings, "test-model")
            
            # Verify chunks exist
            assert db.get_chunk_by_id("test-chunk-1") is not None
            assert db.get_chunk_by_id("test-chunk-2") is not None
            
            # Clear the database
            db.clear()
            
            # Verify all chunks are gone
            assert db.get_chunk_by_id("test-chunk-1") is None
            assert db.get_chunk_by_id("test-chunk-2") is None
            
            # Close database connection
            db.close()


class TestDatabaseMocked:
    """Test cases for the database using mocks."""

    def test_database_setup_mocked(self) -> None:
        """Test database setup with mocked DuckDB."""
        with patch("oboyu.indexer.database.duckdb") as mock_duckdb:
            # Set up the mock
            mock_conn = mock_duckdb.connect.return_value
            
            # Initialize database
            db = Database(db_path="test.db", embedding_dimensions=256)
            db.setup()
            
            # Verify VSS extension was loaded
            mock_conn.execute.assert_any_call("INSTALL vss")
            mock_conn.execute.assert_any_call("LOAD vss")
            
            # Verify tables were created
            assert mock_conn.execute.call_count >= 2
            
            # Close database connection
            db.close()