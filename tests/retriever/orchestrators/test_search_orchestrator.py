"""Tests for SearchOrchestrator."""

import pytest
from unittest.mock import MagicMock
import numpy as np

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.common.types import SearchMode
from oboyu.retriever.orchestrators.search_orchestrator import SearchOrchestrator
from oboyu.retriever.orchestrators.service_registry import ServiceRegistry
from oboyu.common.types import SearchResult


@pytest.fixture
def mock_services() -> ServiceRegistry:
    """Create mock service registry."""
    services = MagicMock(spec=ServiceRegistry)
    
    # Mock config
    services.config = MagicMock(spec=IndexerConfig)
    services.config.use_reranker = False
    services.config.search = MagicMock()
    services.config.search.top_k_multiplier = 2
    
    # Mock individual services
    services.get_search_engine.return_value = MagicMock()
    services.get_embedding_service.return_value = MagicMock()
    services.get_tokenizer_service.return_value = MagicMock()
    services.get_reranker_service.return_value = None
    
    return services


@pytest.fixture
def search_orchestrator(mock_services: ServiceRegistry) -> SearchOrchestrator:
    """Create search orchestrator for testing."""
    return SearchOrchestrator(mock_services)


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample search results."""
    return [
        SearchResult(
            chunk_id="1",
            path="test1.txt",
            title="Test Title 1",
            content="Test content 1",
            chunk_index=0,
            score=0.9,
            language="en",
        ),
        SearchResult(
            chunk_id="2",
            path="test2.txt",
            title="Test Title 2",
            content="Test content 2",
            chunk_index=0,
            score=0.8,
            language="en",
        ),
    ]


def test_search_vector_mode(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test search with vector mode."""
    # Mock embedding generation
    search_orchestrator.embedding_service.generate_query_embedding.return_value = np.array([0.1, 0.2, 0.3])
    
    # Mock search engine results
    search_orchestrator.search_engine.search.return_value = sample_search_results
    
    results = search_orchestrator.search("test query", mode=SearchMode.VECTOR)
    
    assert len(results) == 2
    search_orchestrator.embedding_service.generate_query_embedding.assert_called_once_with("test query")
    search_orchestrator.tokenizer_service.tokenize_query.assert_not_called()


def test_search_bm25_mode(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test search with BM25 mode."""
    # Mock tokenization
    search_orchestrator.tokenizer_service.tokenize_query.return_value = ["test", "query"]
    
    # Mock search engine results
    search_orchestrator.search_engine.search.return_value = sample_search_results
    
    results = search_orchestrator.search("test query", mode=SearchMode.BM25)
    
    assert len(results) == 2
    search_orchestrator.tokenizer_service.tokenize_query.assert_called_once_with("test query")
    search_orchestrator.embedding_service.generate_query_embedding.assert_not_called()


def test_search_hybrid_mode(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test search with hybrid mode."""
    # Mock embedding generation and tokenization
    search_orchestrator.embedding_service.generate_query_embedding.return_value = np.array([0.1, 0.2, 0.3])
    search_orchestrator.tokenizer_service.tokenize_query.return_value = ["test", "query"]
    
    # Mock search engine results
    search_orchestrator.search_engine.search.return_value = sample_search_results
    
    results = search_orchestrator.search("test query", mode=SearchMode.HYBRID)
    
    assert len(results) == 2
    search_orchestrator.embedding_service.generate_query_embedding.assert_called_once_with("test query")
    search_orchestrator.tokenizer_service.tokenize_query.assert_called_once_with("test query")


def test_search_with_reranker_enabled(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test search with reranker enabled."""
    # Enable reranker
    search_orchestrator.config.use_reranker = True
    
    # Mock reranker service
    mock_reranker = MagicMock()
    mock_reranker.is_available.return_value = True
    mock_reranker.rerank.return_value = sample_search_results[:1]  # Return only first result
    search_orchestrator.reranker_service = mock_reranker
    
    # Mock other services
    search_orchestrator.embedding_service.generate_query_embedding.return_value = np.array([0.1, 0.2, 0.3])
    search_orchestrator.search_engine.search.return_value = sample_search_results
    
    results = search_orchestrator.search("test query", mode=SearchMode.VECTOR, limit=1)
    
    assert len(results) == 1
    mock_reranker.rerank.assert_called_once()


def test_vector_search_with_string_query(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test vector search with string query."""
    # Mock embedding generation
    search_orchestrator.embedding_service.generate_query_embedding.return_value = np.array([0.1, 0.2, 0.3])
    search_orchestrator.search_engine.search.return_value = sample_search_results
    
    results = search_orchestrator.vector_search("test query")
    
    assert len(results) == 2
    search_orchestrator.embedding_service.generate_query_embedding.assert_called_once_with("test query")


def test_vector_search_with_embedding_vector(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test vector search with pre-computed embedding."""
    query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    search_orchestrator.search_engine.search.return_value = sample_search_results
    
    results = search_orchestrator.vector_search(query_embedding)
    
    assert len(results) == 2
    search_orchestrator.embedding_service.generate_query_embedding.assert_not_called()


def test_bm25_search(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test BM25 search."""
    search_orchestrator.tokenizer_service.tokenize_query.return_value = ["test", "query"]
    search_orchestrator.search_engine.search.return_value = sample_search_results
    
    results = search_orchestrator.bm25_search("test query")
    
    assert len(results) == 2
    search_orchestrator.tokenizer_service.tokenize_query.assert_called_once_with("test query")


def test_hybrid_search(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test hybrid search."""
    search_orchestrator.embedding_service.generate_query_embedding.return_value = np.array([0.1, 0.2, 0.3])
    search_orchestrator.tokenizer_service.tokenize_query.return_value = ["test", "query"]
    search_orchestrator.search_engine.search.return_value = sample_search_results
    
    results = search_orchestrator.hybrid_search("test query")
    
    assert len(results) == 2
    search_orchestrator.embedding_service.generate_query_embedding.assert_called_once_with("test query")
    search_orchestrator.tokenizer_service.tokenize_query.assert_called_once_with("test query")


def test_rerank_results_available(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test reranking when reranker is available."""
    mock_reranker = MagicMock()
    mock_reranker.is_available.return_value = True
    mock_reranker.rerank.return_value = sample_search_results[:1]
    search_orchestrator.reranker_service = mock_reranker
    
    results = search_orchestrator.rerank_results("test query", sample_search_results)
    
    assert len(results) == 1
    mock_reranker.rerank.assert_called_once_with("test query", sample_search_results)


def test_rerank_results_not_available(
    search_orchestrator: SearchOrchestrator,
    sample_search_results: list[SearchResult],
) -> None:
    """Test reranking when reranker is not available."""
    search_orchestrator.reranker_service = None
    
    results = search_orchestrator.rerank_results("test query", sample_search_results)
    
    assert results == sample_search_results


def test_search_error_handling(search_orchestrator: SearchOrchestrator) -> None:
    """Test error handling in search."""
    # Mock search engine to raise an exception
    search_orchestrator.search_engine.search.side_effect = Exception("Search error")
    search_orchestrator.embedding_service.generate_query_embedding.return_value = np.array([0.1, 0.2, 0.3])
    
    results = search_orchestrator.search("test query", mode=SearchMode.VECTOR)
    
    assert results == []