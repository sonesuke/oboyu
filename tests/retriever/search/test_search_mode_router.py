"""Tests for SearchModeRouter."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from oboyu.retriever.search.search_mode import SearchMode
from oboyu.retriever.search.mode_router import SearchModeRouter
from oboyu.retriever.search.search_result import SearchResult


@pytest.fixture
def mock_vector_search():
    """Mock vector search service."""
    mock = Mock()
    mock.search.return_value = [
        SearchResult(
            chunk_id="vec1",
            path="/test1.txt",
            title="Test 1",
            content="Vector content",
            chunk_index=0,
            language="en",
            score=0.9,
        )
    ]
    return mock


@pytest.fixture
def mock_bm25_search():
    """Mock BM25 search service."""
    mock = Mock()
    mock.search.return_value = [
        SearchResult(
            chunk_id="bm25_1",
            path="/test2.txt",
            title="Test 2",
            content="BM25 content",
            chunk_index=0,
            language="en",
            score=0.8,
        )
    ]
    return mock


@pytest.fixture
def router(mock_vector_search, mock_bm25_search):
    """SearchModeRouter instance."""
    return SearchModeRouter(mock_vector_search, mock_bm25_search)


def test_router_vector_search(router, mock_vector_search):
    """Test routing to vector search."""
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    results = router.route(
        mode=SearchMode.VECTOR,
        query_vector=query_vector,
        limit=5,
    )
    
    assert len(results) == 1
    assert results[0].chunk_id == "vec1"
    mock_vector_search.search.assert_called_once_with(
        query_vector=query_vector,
        limit=5,
        language_filter=None,
        filters=None,
    )


def test_router_bm25_search(router, mock_bm25_search):
    """Test routing to BM25 search."""
    query_terms = ["test", "query"]
    
    results = router.route(
        mode=SearchMode.BM25,
        query_terms=query_terms,
        limit=3,
        language_filter="en",
    )
    
    assert len(results) == 1
    assert results[0].chunk_id == "bm25_1"
    mock_bm25_search.search.assert_called_once_with(
        terms=query_terms,
        limit=3,
        language_filter="en",
        filters=None,
    )


def test_router_vector_search_missing_vector(router):
    """Test vector search without query vector."""
    with pytest.raises(ValueError, match="Query vector is required"):
        router.route(mode=SearchMode.VECTOR, query_terms=["test"])


def test_router_bm25_search_missing_terms(router):
    """Test BM25 search without query terms."""
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="Query terms are required"):
        router.route(mode=SearchMode.BM25, query_vector=query_vector)


def test_router_unsupported_mode(router):
    """Test routing with unsupported mode."""
    with pytest.raises(ValueError, match="Unsupported search mode"):
        router.route(mode=SearchMode.HYBRID)


@patch("oboyu.retriever.search.mode_router.logger")
def test_router_search_exception(mock_logger, router, mock_vector_search):
    """Test error handling in router."""
    mock_vector_search.search.side_effect = Exception("Search failed")
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    with pytest.raises(Exception):
        router.route(mode=SearchMode.VECTOR, query_vector=query_vector)
    
    mock_logger.error.assert_called_once()