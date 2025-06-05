"""Tests for the main Retriever class."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.retriever.retriever import Retriever
from oboyu.common.types import SearchMode
from oboyu.common.types import SearchResult
from oboyu.common.types import SearchFilters


@pytest.fixture
def mock_config() -> IndexerConfig:
    """Create a mock configuration."""
    config = MagicMock(spec=IndexerConfig)
    config.use_reranker = False
    config.model = MagicMock()
    config.model.embedding_model = "test-model"
    return config


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


@pytest.fixture
def retriever(mock_config: IndexerConfig) -> Retriever:
    """Create a Retriever instance for testing."""
    with patch('oboyu.retriever.retriever.ServiceRegistry') as mock_registry:
        # Mock the service registry and orchestrator
        mock_services = MagicMock()
        mock_registry.return_value = mock_services
        
        # Mock all service getters
        mock_services.get_database_service.return_value = MagicMock()
        mock_services.get_embedding_service.return_value = MagicMock()
        mock_services.get_tokenizer_service.return_value = MagicMock()
        mock_services.get_reranker_service.return_value = MagicMock()
        mock_services.get_search_engine.return_value = MagicMock()
        
        with patch('oboyu.retriever.retriever.SearchOrchestrator') as mock_orchestrator:
            return Retriever(mock_config)


class TestRetrieverInitialization:
    """Test Retriever initialization."""
    
    def test_init_with_config(self, mock_config: IndexerConfig) -> None:
        """Test initialization with provided config."""
        with patch('oboyu.retriever.retriever.ServiceRegistry'), \
             patch('oboyu.retriever.retriever.SearchOrchestrator'):
            retriever = Retriever(mock_config)
            assert retriever.config == mock_config
    
    def test_init_without_config(self) -> None:
        """Test initialization with default config."""
        with patch('oboyu.retriever.retriever.ServiceRegistry'), \
             patch('oboyu.retriever.retriever.SearchOrchestrator'), \
             patch('oboyu.retriever.retriever.IndexerConfig') as mock_config_class:
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance
            
            retriever = Retriever()
            assert retriever.config == mock_config_instance
            mock_config_class.assert_called_once()


class TestRetrieverSearch:
    """Test search functionality."""
    
    def test_search_hybrid_mode(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test search with hybrid mode."""
        retriever.search_orchestrator.search.return_value = sample_search_results
        
        results = retriever.search("test query", mode="hybrid")
        
        assert len(results) == 2
        retriever.search_orchestrator.search.assert_called_once_with(
            query="test query",
            mode=SearchMode.HYBRID,
            limit=10,
            language_filter=None,
            filters=None,
        )
    
    def test_search_vector_mode(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test search with vector mode."""
        retriever.search_orchestrator.search.return_value = sample_search_results
        
        results = retriever.search("test query", mode="vector", limit=5)
        
        assert len(results) == 2
        retriever.search_orchestrator.search.assert_called_once_with(
            query="test query",
            mode=SearchMode.VECTOR,
            limit=5,
            language_filter=None,
            filters=None,
        )
    
    def test_search_bm25_mode(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test search with BM25 mode."""
        retriever.search_orchestrator.search.return_value = sample_search_results
        
        results = retriever.search("test query", mode="bm25")
        
        assert len(results) == 2
        retriever.search_orchestrator.search.assert_called_once_with(
            query="test query",
            mode=SearchMode.BM25,
            limit=10,
            language_filter=None,
            filters=None,
        )
    
    def test_search_invalid_mode(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test search with invalid mode defaults to hybrid."""
        retriever.search_orchestrator.search.return_value = sample_search_results
        
        results = retriever.search("test query", mode="invalid")
        
        assert len(results) == 2
        retriever.search_orchestrator.search.assert_called_once_with(
            query="test query",
            mode=SearchMode.HYBRID,  # Should default to hybrid
            limit=10,
            language_filter=None,
            filters=None,
        )
    
    def test_search_with_filters(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test search with filters."""
        retriever.search_orchestrator.search.return_value = sample_search_results
        
        filters = SearchFilters()
        results = retriever.search(
            "test query", 
            language_filter="en", 
            filters=filters
        )
        
        assert len(results) == 2
        retriever.search_orchestrator.search.assert_called_once_with(
            query="test query",
            mode=SearchMode.HYBRID,
            limit=10,
            language_filter="en",
            filters=filters,
        )


class TestRetrieverVectorSearch:
    """Test vector search functionality."""
    
    def test_vector_search_with_string(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test vector search with string query."""
        retriever.search_orchestrator.vector_search.return_value = sample_search_results
        
        results = retriever.vector_search("test query", top_k=5)
        
        assert len(results) == 2
        retriever.search_orchestrator.vector_search.assert_called_once_with(
            query="test query",
            limit=5,
            language_filter=None,
            filters=None,
        )
    
    def test_vector_search_with_embedding(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test vector search with pre-computed embedding."""
        retriever.search_orchestrator.vector_search.return_value = sample_search_results
        
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        results = retriever.vector_search(embedding, top_k=3)
        
        assert len(results) == 2
        retriever.search_orchestrator.vector_search.assert_called_once_with(
            query=embedding,
            limit=3,
            language_filter=None,
            filters=None,
        )
    
    def test_vector_search_legacy_limit_parameter(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test vector search with legacy limit parameter."""
        retriever.search_orchestrator.vector_search.return_value = sample_search_results
        
        results = retriever.vector_search("test query", limit=7)
        
        assert len(results) == 2
        retriever.search_orchestrator.vector_search.assert_called_once_with(
            query="test query",
            limit=7,  # limit parameter takes precedence over top_k default
            language_filter=None,
            filters=None,
        )


class TestRetrieverBM25Search:
    """Test BM25 search functionality."""
    
    def test_bm25_search(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test BM25 search."""
        retriever.search_orchestrator.bm25_search.return_value = sample_search_results
        
        results = retriever.bm25_search("test query", top_k=5)
        
        assert len(results) == 2
        retriever.search_orchestrator.bm25_search.assert_called_once_with(
            query="test query",
            limit=5,
            language_filter=None,
        )
    
    def test_bm25_search_with_language_filter(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test BM25 search with language filter."""
        retriever.search_orchestrator.bm25_search.return_value = sample_search_results
        
        results = retriever.bm25_search("test query", language_filter="ja")
        
        assert len(results) == 2
        retriever.search_orchestrator.bm25_search.assert_called_once_with(
            query="test query",
            limit=10,
            language_filter="ja",
        )


class TestRetrieverHybridSearch:
    """Test hybrid search functionality."""
    
    def test_hybrid_search(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test hybrid search."""
        retriever.search_orchestrator.hybrid_search.return_value = sample_search_results
        
        results = retriever.hybrid_search("test query", top_k=5)
        
        assert len(results) == 2
        retriever.search_orchestrator.hybrid_search.assert_called_once_with(
            query="test query",
            limit=5,
            language_filter=None,
        )
    
    def test_hybrid_search_custom_weights(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test hybrid search with custom weights (weights are now ignored)."""
        retriever.search_orchestrator.hybrid_search.return_value = sample_search_results
        
        results = retriever.hybrid_search(
            "test query", 
            vector_weight=0.6,  # These parameters are accepted but ignored
            bm25_weight=0.4,
            language_filter="en"
        )
        
        assert len(results) == 2
        # Weights are no longer passed to search orchestrator
        retriever.search_orchestrator.hybrid_search.assert_called_once_with(
            query="test query",
            limit=10,
            language_filter="en",
        )


class TestRetrieverReranking:
    """Test reranking functionality."""
    
    def test_rerank_results(
        self, 
        retriever: Retriever, 
        sample_search_results: list[SearchResult]
    ) -> None:
        """Test result reranking."""
        reranked_results = [sample_search_results[1], sample_search_results[0]]
        retriever.search_orchestrator.rerank_results.return_value = reranked_results
        
        results = retriever.rerank_results("test query", sample_search_results)
        
        assert len(results) == 2
        assert results[0].chunk_id == "2"  # Reranked order
        assert results[1].chunk_id == "1"
        retriever.search_orchestrator.rerank_results.assert_called_once_with(
            "test query", sample_search_results
        )


class TestRetrieverStats:
    """Test statistics functionality."""
    
    def test_get_stats(self, retriever: Retriever) -> None:
        """Test getting retriever statistics."""
        # Mock database service responses
        retriever.database_service.get_chunk_count.return_value = 100
        retriever.database_service.get_paths_with_chunks.return_value = ["path1", "path2"]
        
        stats = retriever.get_stats()
        
        assert stats["total_chunks"] == 100
        assert stats["indexed_paths"] == 2
        assert stats["embedding_model"] == "test-model"
        assert stats["reranker_enabled"] is False


class TestRetrieverCleanup:
    """Test cleanup functionality."""
    
    def test_close(self, retriever: Retriever) -> None:
        """Test closing retriever services."""
        retriever.close()
        
        retriever.services.close.assert_called_once()