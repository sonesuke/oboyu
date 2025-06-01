"""Tests for EmbeddingRepository."""

from datetime import datetime
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from oboyu.indexer.storage.consolidated_repositories import EmbeddingRepository


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    connection.execute.return_value.fetchall.return_value = []
    connection.execute.return_value.rowcount = 0
    return connection


@pytest.fixture
def embedding_repository(mock_connection):
    """Create an EmbeddingRepository instance with mock connection."""
    return EmbeddingRepository(mock_connection)


def test_store_embeddings_mismatched_lengths(embedding_repository):
    """Test storing embeddings with mismatched chunk IDs and embeddings."""
    chunk_ids = ["id1", "id2"]
    embeddings = [np.random.rand(256).astype(np.float32)]
    
    with pytest.raises(ValueError, match="Number of chunk IDs must match number of embeddings"):
        embedding_repository.store_embeddings(chunk_ids, embeddings)


def test_store_embeddings_single(embedding_repository, mock_connection):
    """Test storing a single embedding."""
    chunk_ids = ["test-chunk-id"]
    embeddings = [np.random.rand(256).astype(np.float32)]
    
    embedding_repository.store_embeddings(chunk_ids, embeddings)
    
    # Verify execute was called
    mock_connection.execute.assert_called()
    call_args = mock_connection.execute.call_args
    sql = call_args[0][0]
    assert "INSERT" in sql
    assert "embeddings" in sql


def test_store_embeddings_with_progress_callback(embedding_repository, mock_connection):
    """Test storing embeddings with progress callback."""
    chunk_ids = [f"chunk-{i}" for i in range(5)]
    embeddings = [np.random.rand(256).astype(np.float32) for _ in range(5)]
    
    progress_callback = Mock()
    embedding_repository.store_embeddings(chunk_ids, embeddings, progress_callback=progress_callback)
    
    # Progress callback should be called
    progress_callback.assert_called()
    progress_callback.assert_called_with("storing_embeddings", 5, 5)


def test_get_embedding_by_chunk_id_found(embedding_repository, mock_connection):
    """Test getting embedding by chunk ID when found."""
    vector_data = [0.1, 0.2, 0.3]
    mock_connection.execute.return_value.fetchone.return_value = (
        "embed-id",
        "chunk-id",
        "test-model",
        vector_data,
        datetime.now(),
    )
    
    result = embedding_repository.get_embedding_by_chunk_id("chunk-id")
    
    assert result is not None
    assert result["id"] == "embed-id"
    assert result["chunk_id"] == "chunk-id"
    assert result["model"] == "test-model"
    assert isinstance(result["vector"], np.ndarray)
    assert result["vector"].dtype == np.float32
    np.testing.assert_array_equal(result["vector"], np.array(vector_data, dtype=np.float32))


def test_get_embedding_by_chunk_id_not_found(embedding_repository, mock_connection):
    """Test getting embedding by chunk ID when not found."""
    mock_connection.execute.return_value.fetchone.return_value = None
    
    result = embedding_repository.get_embedding_by_chunk_id("non-existent-id")
    
    assert result is None


def test_get_embeddings_batch(embedding_repository, mock_connection):
    """Test getting embeddings batch."""
    mock_connection.execute.return_value.fetchall.return_value = [
        ("chunk-1", [0.1, 0.2, 0.3]),
        ("chunk-2", [0.4, 0.5, 0.6]),
    ]
    
    chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
    embeddings = embedding_repository.get_embeddings_batch(chunk_ids)
    
    assert len(embeddings) == 2
    assert "chunk-1" in embeddings
    assert "chunk-2" in embeddings
    assert "chunk-3" not in embeddings
    
    # Check vectors are numpy arrays
    assert isinstance(embeddings["chunk-1"], np.ndarray)
    assert embeddings["chunk-1"].dtype == np.float32


def test_get_embeddings_batch_empty(embedding_repository):
    """Test getting embeddings batch with empty list."""
    embeddings = embedding_repository.get_embeddings_batch([])
    
    assert embeddings == {}


def test_delete_embeddings_by_chunk_ids(embedding_repository, mock_connection):
    """Test deleting embeddings by chunk IDs."""
    mock_connection.execute.return_value.rowcount = 3
    
    chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
    count = embedding_repository.delete_embeddings_by_chunk_ids(chunk_ids)
    
    assert count == 3
    
    # Check SQL contains placeholders
    call_args = mock_connection.execute.call_args
    sql = call_args[0][0]
    assert "DELETE FROM embeddings" in sql
    assert "WHERE chunk_id IN" in sql


def test_delete_embeddings_by_chunk_ids_empty(embedding_repository, mock_connection):
    """Test deleting embeddings with empty chunk IDs list."""
    count = embedding_repository.delete_embeddings_by_chunk_ids([])
    
    assert count == 0
    mock_connection.execute.assert_not_called()


def test_delete_embeddings_by_path(embedding_repository, mock_connection):
    """Test deleting embeddings by file path."""
    mock_connection.execute.return_value.rowcount = 5
    
    count = embedding_repository.delete_embeddings_by_path("/test/file.txt")
    
    assert count == 5
    
    # Check SQL
    call_args = mock_connection.execute.call_args
    sql = call_args[0][0]
    assert "DELETE FROM embeddings" in sql
    assert "WHERE chunk_id IN" in sql
    assert "SELECT id FROM chunks WHERE path = ?" in sql


def test_get_embedding_count(embedding_repository, mock_connection):
    """Test getting embedding count."""
    mock_connection.execute.return_value.fetchone.return_value = (100,)
    
    count = embedding_repository.get_embedding_count()
    
    assert count == 100
    mock_connection.execute.assert_called_with("SELECT COUNT(*) FROM embeddings")


def test_clear_all_embeddings(embedding_repository, mock_connection):
    """Test clearing all embeddings."""
    embedding_repository.clear_all_embeddings()
    
    mock_connection.execute.assert_called_with("DELETE FROM embeddings")


def test_get_embeddings_by_model(embedding_repository, mock_connection):
    """Test getting embeddings by model."""
    mock_connection.execute.return_value.fetchall.return_value = [
        ("embed-1", "chunk-1", "test-model", datetime.now()),
        ("embed-2", "chunk-2", "test-model", datetime.now()),
    ]
    
    embeddings = embedding_repository.get_embeddings_by_model("test-model")
    
    assert len(embeddings) == 2
    assert embeddings[0]["id"] == "embed-1"
    assert embeddings[0]["model"] == "test-model"
    assert embeddings[1]["id"] == "embed-2"


def test_update_embedding_model(embedding_repository, mock_connection):
    """Test updating embedding model."""
    mock_connection.execute.return_value.rowcount = 1
    
    success = embedding_repository.update_embedding_model("chunk-id", "new-model")
    
    assert success is True
    
    # Check SQL
    call_args = mock_connection.execute.call_args
    sql = call_args[0][0]
    assert "UPDATE embeddings" in sql
    assert "SET model = ?" in sql
    assert "WHERE chunk_id = ?" in sql