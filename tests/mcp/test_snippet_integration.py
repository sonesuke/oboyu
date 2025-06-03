"""Tests for MCP server snippet integration."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from oboyu.common.types import SearchResult
from oboyu.mcp.server import search


class TestSnippetIntegration:
    """Test MCP server snippet integration."""

    def test_search_with_snippet_config(self, monkeypatch):
        """Test search with snippet configuration."""
        # Mock retriever and search results
        mock_retriever = Mock()
        mock_search_result = SearchResult(
            chunk_id="test_chunk",
            path="/test/path.txt",
            title="Test Document",
            content="This is a long document about machine learning and artificial intelligence. It contains detailed information about various algorithms and their applications in real-world scenarios.",
            chunk_index=0,
            language="en",
            metadata={},
            score=0.85
        )
        mock_retriever.search.return_value = [mock_search_result]
        
        # Mock get_retriever function
        def mock_get_retriever(db_path=None):
            return mock_retriever
        
        monkeypatch.setattr("oboyu.mcp.server.get_retriever", mock_get_retriever)
        
        # Test with snippet configuration
        snippet_config = {
            "length": 100,
            "highlight_matches": True,
            "strategy": "sentence_boundary"
        }
        
        result = search(
            query="machine learning",
            snippet_config=snippet_config
        )
        
        assert "results" in result
        assert len(result["results"]) == 1
        
        search_result = result["results"][0]
        assert "content" in search_result
        assert len(search_result["content"]) <= 100
        # Check for highlighted content (with **bold** markers)
        content_lower = search_result["content"].lower()
        assert "machine" in content_lower and "learning" in content_lower

    def test_search_without_snippet_config(self, monkeypatch):
        """Test search without snippet configuration (backward compatibility)."""
        # Mock retriever and search results
        mock_retriever = Mock()
        original_content = "This is the original content without modification."
        mock_search_result = SearchResult(
            chunk_id="test_chunk",
            path="/test/path.txt",
            title="Test Document",
            content=original_content,
            chunk_index=0,
            language="en",
            metadata={},
            score=0.85
        )
        mock_retriever.search.return_value = [mock_search_result]
        
        # Mock get_retriever function
        def mock_get_retriever(db_path=None):
            return mock_retriever
        
        monkeypatch.setattr("oboyu.mcp.server.get_retriever", mock_get_retriever)
        
        result = search(query="test query")
        
        assert "results" in result
        assert len(result["results"]) == 1
        
        search_result = result["results"][0]
        # Content should be unchanged when no snippet config provided
        assert search_result["content"] == original_content

    def test_search_with_invalid_snippet_config(self, monkeypatch, caplog):
        """Test search with invalid snippet configuration."""
        # Set log level to capture warnings
        caplog.set_level(logging.WARNING)
        
        # Mock retriever and search results
        mock_retriever = Mock()
        original_content = "This is the original content."
        mock_search_result = SearchResult(
            chunk_id="test_chunk",
            path="/test/path.txt",
            title="Test Document",
            content=original_content,
            chunk_index=0,
            language="en",
            metadata={},
            score=0.85
        )
        mock_retriever.search.return_value = [mock_search_result]
        
        # Mock get_retriever function
        def mock_get_retriever(db_path=None):
            return mock_retriever
        
        monkeypatch.setattr("oboyu.mcp.server.get_retriever", mock_get_retriever)
        
        # Test with invalid snippet configuration
        invalid_snippet_config = {
            "length": -1,  # Invalid negative length
            "unknown_field": "value"
        }
        
        result = search(
            query="test query",
            snippet_config=invalid_snippet_config
        )
        
        assert "results" in result
        assert len(result["results"]) == 1
        
        search_result = result["results"][0]
        # Should fall back to original content
        assert search_result["content"] == original_content
        
        # Should log warning about invalid config
        assert "Invalid snippet_config" in caplog.text

    def test_search_with_japanese_snippet_config(self, monkeypatch):
        """Test search with Japanese-aware snippet configuration."""
        # Mock retriever and search results
        mock_retriever = Mock()
        japanese_content = "これは機械学習についての重要な文書です。人工知能技術は急速に発展しています。この分野では多くの研究が行われています。データサイエンスの応用も広がっています。"
        mock_search_result = SearchResult(
            chunk_id="test_chunk",
            path="/test/path.txt",
            title="Japanese Document",
            content=japanese_content,
            chunk_index=0,
            language="ja",
            metadata={},
            score=0.85
        )
        mock_retriever.search.return_value = [mock_search_result]
        
        # Mock get_retriever function
        def mock_get_retriever(db_path=None):
            return mock_retriever
        
        monkeypatch.setattr("oboyu.mcp.server.get_retriever", mock_get_retriever)
        
        # Test with Japanese-aware snippet configuration
        snippet_config = {
            "length": 150,
            "japanese_aware": True,
            "strategy": "sentence_boundary",
            "prefer_complete_sentences": True,
            "highlight_matches": False
        }
        
        result = search(
            query="機械学習",
            snippet_config=snippet_config
        )
        
        assert "results" in result
        assert len(result["results"]) == 1
        
        search_result = result["results"][0]
        assert "content" in search_result
        assert len(search_result["content"]) <= 150
        assert "機械学習" in search_result["content"]
        # Should end with Japanese sentence ending if truncated
        if len(search_result["content"]) < len(japanese_content):
            assert search_result["content"].endswith('。')

    def test_search_with_multi_level_snippet_config(self, monkeypatch):
        """Test search with multi-level snippet configuration."""
        # Mock retriever and search results
        mock_retriever = Mock()
        long_content = "This is a very long document about machine learning and artificial intelligence. " * 10
        mock_search_result = SearchResult(
            chunk_id="test_chunk",
            path="/test/path.txt",
            title="Long Document",
            content=long_content,
            chunk_index=0,
            language="en",
            metadata={},
            score=0.85
        )
        mock_retriever.search.return_value = [mock_search_result]
        
        # Mock get_retriever function
        def mock_get_retriever(db_path=None):
            return mock_retriever
        
        monkeypatch.setattr("oboyu.mcp.server.get_retriever", mock_get_retriever)
        
        # Test with multi-level snippet configuration
        snippet_config = {
            "length": 500,  # This will be overridden by levels
            "levels": [
                {"type": "summary", "length": 50},
                {"type": "detailed", "length": 150}
            ],
            "highlight_matches": False
        }
        
        result = search(
            query="machine learning",
            snippet_config=snippet_config
        )
        
        assert "results" in result
        assert len(result["results"]) == 1
        
        search_result = result["results"][0]
        assert "content" in search_result
        # Should use first level (summary) with length 50
        assert len(search_result["content"]) <= 50
        content_lower = search_result["content"].lower()
        # At least one of the query terms should be present
        assert "machine" in content_lower or "learning" in content_lower