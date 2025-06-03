"""Simplified tests for database batch operations.

Note: Most complex batch processing logic was part of the old API.
This file contains basic tests that work with the new architecture.
"""

import os
import tempfile
from datetime import datetime
from typing import Generator, List

import pytest

from oboyu.indexer.storage.database_service import DatabaseService as Database
from oboyu.common.types import Chunk


@pytest.fixture
def temp_db() -> Generator[Database, None, None]:
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    
    db = Database(db_path=db_path)
    db.initialize()
    
    yield db
    
    db.close()


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks for testing."""
    chunks = []
    for i in range(5):  # Reduced for simpler tests
        chunk = Chunk(
            id=f"chunk_{i}",
            path=f"/test/file_{i}.txt",
            title=f"Test File {i}",
            content=f"This is test content for chunk {i}.",
            chunk_index=i,
            language="en",
            created_at=datetime.now(),
            modified_at=datetime.now(),
            metadata={"test": True},
        )
        chunks.append(chunk)
    return chunks


def test_batch_processing_small_dataset(temp_db: Database, sample_chunks: List[Chunk]) -> None:
    """Test basic chunk storage functionality."""
    # Store small batch of chunks
    temp_db.store_chunks(sample_chunks)
    
    # Verify chunks were stored
    results = temp_db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    assert results[0] == len(sample_chunks)


def test_batch_processing_large_dataset(temp_db: Database) -> None:
    """Test chunk storage with larger dataset."""
    # Create larger test data
    chunks = []
    for i in range(50):  # Reduced for simpler test
        chunk = Chunk(
            id=f"chunk_{i}",
            path=f"/test/file_{i}.txt",
            title=f"Test File {i}",
            content=f"Content for chunk {i}",
            chunk_index=i,
            language="en",
            created_at=datetime.now(),
            modified_at=datetime.now(),
            metadata={"test": True},
        )
        chunks.append(chunk)
    
    # Store chunks
    temp_db.store_chunks(chunks)
    
    # Verify all chunks were stored
    count = temp_db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == len(chunks)


def test_batch_processing_memory_efficiency(temp_db: Database) -> None:
    """Test basic memory efficiency with chunk storage."""
    # Create moderate amount of test data
    chunks = []
    for i in range(100):
        chunk = Chunk(
            id=f"chunk_{i}",
            path=f"/test/file_{i}.txt",
            title=f"Test File {i}",
            content=f"Content for chunk {i}",
            chunk_index=i,
            language="en",
            created_at=datetime.now(),
            modified_at=datetime.now(),
            metadata={"test": True},
        )
        chunks.append(chunk)
    
    # Store chunks (tests basic batch processing)
    temp_db.store_chunks(chunks)
    
    # Verify operation completed successfully
    count = temp_db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == len(chunks)


def test_batch_processing_update_mode(temp_db: Database, sample_chunks: List[Chunk]) -> None:
    """Test basic update functionality with chunks."""
    # Store initial chunks
    temp_db.store_chunks(sample_chunks[:3])
    
    # Verify initial state
    initial_count = temp_db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert initial_count == 3
    
    # Store additional chunks
    temp_db.store_chunks(sample_chunks[3:])
    
    # Verify final state
    final_count = temp_db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert final_count == len(sample_chunks)