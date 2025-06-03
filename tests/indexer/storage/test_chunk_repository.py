"""Tests for ChunkRepository."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from oboyu.common.types import Chunk
from oboyu.indexer.storage.consolidated_repositories import ChunkRepository


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    connection.execute.return_value.fetchall.return_value = []
    connection.execute.return_value.rowcount = 0
    return connection


@pytest.fixture
def chunk_repository(mock_connection):
    """Create a ChunkRepository instance with mock connection."""
    return ChunkRepository(mock_connection)


def test_store_chunks_empty_list(chunk_repository, mock_connection):
    """Test storing empty chunk list."""
    chunk_repository.store_chunks([])
    mock_connection.execute.assert_not_called()


def test_store_chunks_single_chunk(chunk_repository, mock_connection):
    """Test storing a single chunk."""
    chunk = Chunk(
        id="test-id",
        path=Path("/test/file.txt"),
        title="Test Title",
        content="Test content",
        chunk_index=0,
        language="en",
        created_at=datetime.now(),
        modified_at=datetime.now(),
        metadata={"key": "value"},
    )
    
    chunk_repository.store_chunks([chunk])
    
    # Verify execute was called with correct SQL
    mock_connection.execute.assert_called()
    call_args = mock_connection.execute.call_args
    sql = call_args[0][0]
    assert "INSERT" in sql
    assert "chunks" in sql


def test_store_chunks_with_progress_callback(chunk_repository, mock_connection):
    """Test storing chunks with progress callback."""
    chunks = [
        Chunk(
            id=f"test-id-{i}",
            path=Path(f"/test/file{i}.txt"),
            title=f"Test Title {i}",
            content=f"Test content {i}",
            chunk_index=i,
            language="en",
            created_at=datetime.now(),
            modified_at=datetime.now(),
            metadata={},
        )
        for i in range(5)
    ]
    
    progress_callback = Mock()
    chunk_repository.store_chunks(chunks, progress_callback)
    
    # Progress callback should be called at least once
    progress_callback.assert_called()
    # Last call should have total count
    progress_callback.assert_called_with("storing", 5, 5)


def test_get_chunk_by_id_found(chunk_repository, mock_connection):
    """Test getting chunk by ID when found."""
    # Mock the database response
    mock_connection.execute.return_value.fetchone.return_value = (
        "test-id",
        "/test/file.txt",
        "Test Title",
        "Test content",
        0,
        "en",
        datetime.now(),
        datetime.now(),
        '{"key": "value"}',
        [0.1, 0.2, 0.3],
    )
    
    result = chunk_repository.get_chunk_by_id("test-id")
    
    assert result is not None
    assert result["id"] == "test-id"
    assert result["path"] == "/test/file.txt"
    assert result["title"] == "Test Title"
    assert result["metadata"] == {"key": "value"}


def test_get_chunk_by_id_not_found(chunk_repository, mock_connection):
    """Test getting chunk by ID when not found."""
    mock_connection.execute.return_value.fetchone.return_value = None
    
    result = chunk_repository.get_chunk_by_id("non-existent-id")
    
    assert result is None


def test_delete_chunks_by_path(chunk_repository, mock_connection):
    """Test deleting chunks by path."""
    mock_connection.execute.return_value.rowcount = 5
    
    count = chunk_repository.delete_chunks_by_path("/test/file.txt")
    
    assert count == 5
    mock_connection.execute.assert_called()
    call_args = mock_connection.execute.call_args
    sql = call_args[0][0]
    assert "DELETE FROM chunks" in sql
    assert "WHERE path = ?" in sql


def test_get_chunk_count(chunk_repository, mock_connection):
    """Test getting chunk count."""
    mock_connection.execute.return_value.fetchone.return_value = (42,)
    
    count = chunk_repository.get_chunk_count()
    
    assert count == 42
    mock_connection.execute.assert_called_with("SELECT COUNT(*) FROM chunks")


def test_get_paths_with_chunks(chunk_repository, mock_connection):
    """Test getting paths with chunks."""
    mock_connection.execute.return_value.fetchall.return_value = [
        ("/test/file1.txt",),
        ("/test/file2.txt",),
        ("/test/file3.txt",),
    ]
    
    paths = chunk_repository.get_paths_with_chunks()
    
    assert len(paths) == 3
    assert "/test/file1.txt" in paths
    assert "/test/file2.txt" in paths
    assert "/test/file3.txt" in paths


def test_get_chunks_by_path(chunk_repository, mock_connection):
    """Test getting chunks by path."""
    mock_connection.execute.return_value.fetchall.return_value = [
        (
            "test-id-1",
            "/test/file.txt",
            "Test Title 1",
            "Test content 1",
            0,
            "en",
            datetime.now(),
            datetime.now(),
            '{"key": "value1"}',
            None,
        ),
        (
            "test-id-2",
            "/test/file.txt",
            "Test Title 2",
            "Test content 2",
            1,
            "en",
            datetime.now(),
            datetime.now(),
            '{"key": "value2"}',
            None,
        ),
    ]
    
    chunks = chunk_repository.get_chunks_by_path("/test/file.txt")
    
    assert len(chunks) == 2
    assert chunks[0]["id"] == "test-id-1"
    assert chunks[1]["id"] == "test-id-2"
    assert chunks[0]["metadata"] == {"key": "value1"}
    assert chunks[1]["metadata"] == {"key": "value2"}


def test_clear_all_chunks(chunk_repository, mock_connection):
    """Test clearing all chunks."""
    chunk_repository.clear_all_chunks()
    
    mock_connection.execute.assert_called_with("DELETE FROM chunks")


def test_get_chunk_ids_without_embeddings(chunk_repository, mock_connection):
    """Test getting chunk IDs without embeddings."""
    mock_connection.execute.return_value.fetchall.return_value = [
        ("chunk-id-1",),
        ("chunk-id-2",),
        ("chunk-id-3",),
    ]
    
    chunk_ids = chunk_repository.get_chunk_ids_without_embeddings(limit=10)
    
    assert len(chunk_ids) == 3
    assert "chunk-id-1" in chunk_ids
    assert "chunk-id-2" in chunk_ids
    assert "chunk-id-3" in chunk_ids
    
    # Check SQL contains LIMIT
    call_args = mock_connection.execute.call_args
    sql = call_args[0][0]
    assert "LIMIT 10" in sql