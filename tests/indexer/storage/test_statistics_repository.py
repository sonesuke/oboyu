"""Tests for StatisticsRepository."""

from unittest.mock import MagicMock

import pytest

from oboyu.indexer.storage.consolidated_repositories import StatisticsRepository


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    connection.execute.return_value.fetchall.return_value = []
    return connection


@pytest.fixture
def statistics_repository(mock_connection):
    """Create a StatisticsRepository instance with mock connection."""
    return StatisticsRepository(mock_connection)


def test_get_database_stats(statistics_repository, mock_connection):
    """Test getting comprehensive database statistics."""
    # Mock different query results
    mock_connection.execute.return_value.fetchone.side_effect = [
        (100,),  # chunk_count
        (95,),   # embedding_count
        (10,),   # unique_paths
        (2048,), # avg_chunk_size
    ]
    
    mock_connection.execute.return_value.fetchall.side_effect = [
        [("en", 80), ("ja", 20)],  # language_distribution
        [("model-1", 60), ("model-2", 35)],  # model_distribution
    ]
    
    stats = statistics_repository.get_database_stats()
    
    assert stats["chunk_count"] == 100
    assert stats["embedding_count"] == 95
    assert stats["unique_paths"] == 10
    assert stats["avg_chunk_size"] == 2048
    assert stats["language_distribution"] == {"en": 80, "ja": 20}
    assert stats["model_distribution"] == {"model-1": 60, "model-2": 35}


def test_get_path_statistics(statistics_repository, mock_connection):
    """Test getting statistics for a specific path."""
    mock_connection.execute.return_value.fetchone.side_effect = [
        (5,),       # chunk_count
        (5,),       # embedding_count
        (10240,),   # total_content_size
        ("en",),    # language
    ]
    
    stats = statistics_repository.get_path_statistics("/test/file.txt")
    
    assert stats["chunk_count"] == 5
    assert stats["embedding_count"] == 5
    assert stats["total_content_size"] == 10240
    assert stats["language"] == "en"


def test_get_chunk_statistics(statistics_repository, mock_connection):
    """Test getting detailed chunk statistics."""
    mock_connection.execute.return_value.fetchone.side_effect = [
        (1000,),  # total chunks
    ]
    
    mock_connection.execute.return_value.fetchall.side_effect = [
        [(0, 300), (1, 250), (2, 200)],  # by_index
        [("small", 100), ("medium", 400), ("large", 400), ("very_large", 100)],  # size_distribution
    ]
    
    stats = statistics_repository.get_chunk_statistics()
    
    assert stats["total"] == 1000
    assert stats["by_index"] == {0: 300, 1: 250, 2: 200}
    assert stats["size_distribution"] == {
        "small": 100,
        "medium": 400,
        "large": 400,
        "very_large": 100,
    }


def test_get_embedding_statistics(statistics_repository, mock_connection):
    """Test getting detailed embedding statistics."""
    mock_connection.execute.return_value.fetchone.side_effect = [
        (950,),  # total embeddings
        (5,),    # orphaned embeddings
        (950, 1000),  # coverage stats
    ]
    
    mock_connection.execute.return_value.fetchall.return_value = [
        ("model-1", 600),
        ("model-2", 350),
    ]
    
    stats = statistics_repository.get_embedding_statistics()
    
    assert stats["total"] == 950
    assert stats["by_model"] == {"model-1": 600, "model-2": 350}
    assert stats["orphaned"] == 5
    assert stats["coverage"]["chunks_with_embeddings"] == 950
    assert stats["coverage"]["total_chunks"] == 1000
    assert stats["coverage"]["coverage_percentage"] == 95.0


def test_get_embedding_statistics_no_chunks(statistics_repository, mock_connection):
    """Test getting embedding statistics when no chunks exist."""
    mock_connection.execute.return_value.fetchone.side_effect = [
        (0,),  # total embeddings
        (0,),  # orphaned embeddings
        (0, 0),  # coverage stats (no chunks)
    ]
    
    mock_connection.execute.return_value.fetchall.return_value = []
    
    stats = statistics_repository.get_embedding_statistics()
    
    assert stats["total"] == 0
    assert stats["by_model"] == {}
    assert stats["orphaned"] == 0
    assert stats["coverage"]["coverage_percentage"] == 0.0


def test_get_latest_statistics_summary(statistics_repository, mock_connection):
    """Test getting latest statistics summary."""
    # First set up mock results for get_database_stats
    db_stats_results = [
        (100,),  # chunk_count
        (95,),   # embedding_count
        (10,),   # unique_paths
        (2048,), # avg_chunk_size
    ]
    
    db_stats_fetchall = [
        [("en", 80), ("ja", 20)],  # language_distribution
        [("model-1", 60), ("model-2", 35)],  # model_distribution
    ]
    
    # Then for get_chunk_statistics
    chunk_stats_results = [
        (100,),  # total chunks
    ]
    
    chunk_stats_fetchall = [
        [(0, 300), (1, 250), (2, 200)],  # by_index
        [("small", 100), ("medium", 400)],  # size_distribution
    ]
    
    # Then for get_embedding_statistics
    embed_stats_results = [
        (95,),   # total embeddings
        (0,),    # orphaned
        (95, 100),  # coverage
    ]
    
    embed_stats_fetchall = [
        [("model-1", 60), ("model-2", 35)],  # by_model
    ]
    
    # Configure mock to return different results for each method call
    call_count = 0
    def side_effect_fetchone():
        nonlocal call_count
        if call_count < 4:  # get_database_stats
            result = db_stats_results[call_count]
        elif call_count < 5:  # get_chunk_statistics
            result = chunk_stats_results[call_count - 4]
        else:  # get_embedding_statistics
            result = embed_stats_results[call_count - 5]
        call_count += 1
        return result
    
    fetchall_count = 0
    def side_effect_fetchall():
        nonlocal fetchall_count
        if fetchall_count < 2:  # get_database_stats
            result = db_stats_fetchall[fetchall_count]
        elif fetchall_count < 4:  # get_chunk_statistics
            result = chunk_stats_fetchall[fetchall_count - 2]
        else:  # get_embedding_statistics
            result = embed_stats_fetchall[fetchall_count - 4]
        fetchall_count += 1
        return result
    
    mock_connection.execute.return_value.fetchone.side_effect = side_effect_fetchone
    mock_connection.execute.return_value.fetchall.side_effect = side_effect_fetchall
    
    summary = statistics_repository.get_latest_statistics_summary()
    
    assert summary["database"]["total_chunks"] == 100
    assert summary["database"]["total_embeddings"] == 95
    assert summary["database"]["unique_files"] == 10
    assert summary["chunks"]["average_size"] == 2048
    assert summary["embeddings"]["coverage_percentage"] == 95.0
    assert "en" in summary["languages"]
    assert "ja" in summary["languages"]


def test_error_handling_get_database_stats(statistics_repository, mock_connection):
    """Test error handling in get_database_stats."""
    mock_connection.execute.side_effect = Exception("Database error")
    
    stats = statistics_repository.get_database_stats()
    
    assert stats == {}


def test_error_handling_get_path_statistics(statistics_repository, mock_connection):
    """Test error handling in get_path_statistics."""
    mock_connection.execute.side_effect = Exception("Database error")
    
    stats = statistics_repository.get_path_statistics("/test/file.txt")
    
    assert stats == {}