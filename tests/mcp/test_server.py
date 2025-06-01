"""Tests for the MCP server module."""

import pytest
from unittest.mock import patch, MagicMock

from oboyu.mcp.context import mcp
from oboyu.mcp.server import get_indexer, search, get_index_info, index_directory, clear_index


@pytest.fixture
def mock_indexer():
    """Create a mock indexer for testing."""
    indexer = MagicMock()
    
    # Mock search results
    search_result = MagicMock()
    search_result.title = "Test Document"
    search_result.content = "This is test content with some Japanese text: 日本語"
    search_result.path = "/path/to/document.md"
    search_result.score = 0.89
    search_result.language = "ja"
    search_result.metadata = {"source": "test"}
    search_result.chunk_id = "test-chunk-123"
    search_result.chunk_index = 1
    
    # Setup the search method to return the mock results
    indexer.search.return_value = [search_result]
    
    # Setup the database statistics
    db_stats = {
        "indexed_paths": 10,
        "total_chunks": 50,
        "embedding_model": "test-model",
    }
    indexer.get_stats.return_value = db_stats
    
    return indexer


@patch("oboyu.mcp.server.get_indexer")
def test_search(mock_get_indexer, mock_indexer):
    """Test the search tool function."""
    # Setup the mock
    mock_get_indexer.return_value = mock_indexer
    
    # Call the function
    result = search("test query", top_k=5)
    
    # Verify the indexer was called with correct parameters
    mock_get_indexer.assert_called_once_with(None)
    mock_indexer.search.assert_called_once_with("test query", limit=5, mode="hybrid", language_filter=None)
    
    # Check the response format
    assert "results" in result
    assert "stats" in result
    assert len(result["results"]) == 1
    
    # Check the content of the first result
    first_result = result["results"][0]
    assert first_result["title"] == "Test Document"
    assert first_result["content"] == "This is test content with some Japanese text: 日本語"
    assert first_result["uri"] == "file:///path/to/document.md"
    assert first_result["score"] == 0.89
    assert first_result["language"] == "ja"
    assert first_result["metadata"] == {"source": "test"}


@patch("oboyu.mcp.server.get_indexer")
def test_search_with_language_filter(mock_get_indexer, mock_indexer):
    """Test the search tool function with language filter."""
    # Setup the mock
    mock_get_indexer.return_value = mock_indexer
    
    # Call the function with language filter
    result = search("test query", top_k=5, language="ja")
    
    # Verify the indexer was called with correct parameters
    mock_get_indexer.assert_called_once_with(None)
    mock_indexer.search.assert_called_once_with("test query", limit=5, mode="hybrid", language_filter="ja")


@patch("oboyu.mcp.server.get_indexer")
def test_search_with_custom_db_path(mock_get_indexer, mock_indexer):
    """Test the search tool function with custom database path."""
    # Setup the mock
    mock_get_indexer.return_value = mock_indexer
    
    # Call the function with custom db_path
    result = search("test query", db_path="/custom/path.db")
    
    # Verify the indexer was called with correct parameters
    mock_get_indexer.assert_called_once_with("/custom/path.db")


@patch("oboyu.mcp.server.get_indexer")
def test_index_directory(mock_get_indexer, mock_indexer):
    """Test the index_directory tool function."""
    # Setup the mock
    mock_get_indexer.return_value = mock_indexer
    # Setup indexer.index_documents to return some results
    mock_indexer.index_documents.return_value = {"indexed_chunks": 25, "total_documents": 5}
    mock_indexer.config.db_path = "/path/to/test.db"
    
    # Call the function with a valid directory
    with patch('oboyu.mcp.server.Path.exists', return_value=True):
        with patch('oboyu.mcp.server.Path.is_dir', return_value=True):
            with patch('oboyu.crawler.crawler.Crawler') as mock_crawler:
                mock_crawler_instance = mock_crawler.return_value
                mock_crawler_instance.crawl.return_value = []
                result = index_directory("/valid/directory", incremental=True)
    
    # Verify the indexer was called correctly
    mock_get_indexer.assert_called_once_with(None)
    mock_indexer.index_documents.assert_called_once()
    
    # Check the response format for success
    assert "success" in result
    assert result["success"] is True
    assert "directory" in result
    assert "documents_indexed" in result
    assert result["documents_indexed"] == 5
    assert "chunks_indexed" in result
    assert result["chunks_indexed"] == 25
    assert "db_path" in result
    assert result["db_path"] == "/path/to/test.db"
    
    # Test error case with invalid directory
    mock_get_indexer.reset_mock()
    with patch('oboyu.mcp.server.Path.exists', return_value=False):
        result = index_directory("/invalid/directory")
    
    # Verify behavior for invalid directory
    assert "success" in result
    assert result["success"] is False
    assert "error" in result
    assert "Directory does not exist" in result["error"]
    mock_get_indexer.assert_not_called()


@patch("oboyu.mcp.server.get_indexer")
def test_clear_index(mock_get_indexer, mock_indexer):
    """Test the clear_index tool function."""
    # Setup the mock
    mock_get_indexer.return_value = mock_indexer
    mock_indexer.config.db_path = "/path/to/test.db"
    
    # Call the function
    result = clear_index()
    
    # Verify the indexer was called
    mock_get_indexer.assert_called_once_with(None)
    mock_indexer.clear_index.assert_called_once()
    
    # Check the response format
    assert "success" in result
    assert result["success"] is True
    assert "message" in result
    assert "cleared successfully" in result["message"]
    assert "db_path" in result
    assert result["db_path"] == "/path/to/test.db"
    
    # Test error case
    mock_get_indexer.reset_mock()
    mock_indexer.clear_index.side_effect = Exception("Test error")
    mock_get_indexer.return_value = mock_indexer
    
    result = clear_index()
    
    assert "success" in result
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Test error"


@patch("oboyu.mcp.server.get_indexer")
def test_get_index_info(mock_get_indexer, mock_indexer):
    """Test the get_index_info tool function."""
    # Setup the mock
    mock_get_indexer.return_value = mock_indexer
    
    # Call the function
    result = get_index_info()
    
    # Verify the indexer was called
    mock_get_indexer.assert_called_once_with(None)
    mock_indexer.get_stats.assert_called_once()
    
    # Check the response format
    assert "document_count" in result
    assert result["document_count"] == 10
    assert "chunk_count" in result
    assert result["chunk_count"] == 50
    assert "languages" in result
    assert "embedding_model" in result
    assert result["embedding_model"] == "test-model"
    assert "db_path" in result