"""Tests for SearchModeRouter."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from oboyu.common.types import SearchMode
from oboyu.retriever.search.mode_router import SearchModeRouter
from oboyu.common.types import SearchResult


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


def test_router_initialization(mock_vector_search, mock_bm25_search):
    """Test router initialization."""
    router = SearchModeRouter(mock_vector_search, mock_bm25_search)
    
    assert router.vector_search == mock_vector_search
    assert router.bm25_search == mock_bm25_search


def test_route_vector_mode(mock_vector_search, mock_bm25_search):
    """Test routing for vector search mode."""
    router = SearchModeRouter(mock_vector_search, mock_bm25_search)
    
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    results = router.route(
        mode=SearchMode.VECTOR,
        query_vector=query_vector,
        query_terms=None,
        limit=10,
    )
    
    # Should call only vector search
    mock_vector_search.search.assert_called_once_with(
        query_vector=query_vector,
        limit=10,
        language_filter=None,
        filters=None,
    )
    mock_bm25_search.search.assert_not_called()
    
    # Should return vector results
    assert len(results) == 1
    assert results[0].chunk_id == "vec1"


def test_route_bm25_mode(mock_vector_search, mock_bm25_search):
    """Test routing for BM25 search mode."""
    router = SearchModeRouter(mock_vector_search, mock_bm25_search)
    
    query_terms = ["test", "query"]
    results = router.route(
        mode=SearchMode.BM25,
        query_vector=None,
        query_terms=query_terms,
        limit=10,
    )
    
    # Should call only BM25 search
    mock_bm25_search.search.assert_called_once_with(
        terms=query_terms,
        limit=10,
        language_filter=None,
        filters=None,
    )
    mock_vector_search.search.assert_not_called()
    
    # Should return BM25 results
    assert len(results) == 1
    assert results[0].chunk_id == "bm25_1"


def test_route_hybrid_mode_not_supported(mock_vector_search, mock_bm25_search):
    """Test that hybrid mode is not supported at router level."""
    router = SearchModeRouter(mock_vector_search, mock_bm25_search)
    
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    query_terms = ["test", "query"]
    
    # Hybrid mode should raise an error at router level
    with pytest.raises(ValueError, match="Unsupported search mode for routing"):
        router.route(
            mode=SearchMode.HYBRID,
            query_vector=query_vector,
            query_terms=query_terms,
            limit=10,
        )


def test_route_missing_query_vector():
    """Test routing with missing query vector for vector mode."""
    router = SearchModeRouter(Mock(), Mock())
    
    # Should raise error for vector mode without query vector
    with pytest.raises(ValueError, match="Query vector is required for vector search"):
        router.route(
            mode=SearchMode.VECTOR,
            query_vector=None,
            query_terms=None,
            limit=10,
        )


def test_route_missing_query_terms():
    """Test routing with missing query terms for BM25 mode."""
    router = SearchModeRouter(Mock(), Mock())
    
    # Should raise error for BM25 mode without query terms
    with pytest.raises(ValueError, match="Query terms are required for BM25 search"):
        router.route(
            mode=SearchMode.BM25,
            query_vector=None,
            query_terms=None,
            limit=10,
        )


def test_route_with_language_filter(mock_vector_search, mock_bm25_search):
    """Test routing with language filter."""
    router = SearchModeRouter(mock_vector_search, mock_bm25_search)
    
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    results = router.route(
        mode=SearchMode.VECTOR,
        query_vector=query_vector,
        query_terms=None,
        limit=10,
        language_filter="ja",
    )
    
    # Vector search should receive language filter
    mock_vector_search.search.assert_called_once_with(
        query_vector=query_vector,
        limit=10,
        language_filter="ja",
        filters=None,
    )
    mock_bm25_search.search.assert_not_called()


def test_route_with_filters(mock_vector_search, mock_bm25_search):
    """Test routing with search filters."""
    router = SearchModeRouter(mock_vector_search, mock_bm25_search)
    
    from oboyu.common.types import SearchFilters
    filters = SearchFilters()
    
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    results = router.route(
        mode=SearchMode.VECTOR,
        query_vector=query_vector,
        query_terms=None,
        limit=10,
        filters=filters,
    )
    
    # Should pass filters to search
    mock_vector_search.search.assert_called_once_with(
        query_vector=query_vector,
        limit=10,
        language_filter=None,
        filters=filters,
    )