"""Tests for QueryService."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oboyu.cli.services.query_service import QueryService
from oboyu.common.config import ConfigManager
from oboyu.common.types import SearchResult


@pytest.fixture
def mock_config_manager():
    """Create a mock config manager."""
    config_manager = Mock(spec=ConfigManager)
    config_manager.get_section.return_value = {
        "database_path": "test.db",
        "top_k": 10,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "use_reranker": False,
    }
    config_manager.merge_cli_overrides.return_value = {
        "database_path": "test.db",
        "top_k": 10,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "use_reranker": False,
    }
    return config_manager


@pytest.fixture
def query_service(mock_config_manager):
    """Create a QueryService instance."""
    return QueryService(mock_config_manager)


@pytest.fixture
def mock_search_results():
    """Create mock search results."""
    return [
        SearchResult(
            chunk_id="chunk1",
            path="test1.txt",
            content="Test content 1",
            score=0.9,
            chunk_index=0,
            title="Test 1",
            language="en"
        ),
        SearchResult(
            chunk_id="chunk2",
            path="test2.txt",
            content="Test content 2",
            score=0.8,
            chunk_index=1,
            title="Test 2",
            language="en"
        ),
    ]


class TestQueryService:
    """Test QueryService functionality."""
    
    def test_init(self, mock_config_manager):
        """Test service initialization."""
        service = QueryService(mock_config_manager)
        assert service.config_manager == mock_config_manager
    
    def test_get_database_path_default(self, query_service):
        """Test database path resolution with defaults."""
        result = query_service.get_database_path()
        assert result == "test.db"
    
    def test_get_database_path_override(self, query_service):
        """Test database path resolution with override."""
        result = query_service.get_database_path(Path("/custom/path.db"))
        assert result == "/custom/path.db"
    
    def test_get_query_config_default(self, query_service):
        """Test query config retrieval with defaults."""
        # Reset call counts from fixture setup
        query_service.config_manager.reset_mock()
        
        result = query_service.get_query_config()
        
        query_service.config_manager.merge_cli_overrides.assert_called_once_with("query", {})
        assert result == {
            "database_path": "test.db",
            "top_k": 10,
            "vector_weight": 0.7,
            "bm25_weight": 0.3,
            "use_reranker": False,
        }
    
    def test_get_query_config_with_overrides(self, query_service):
        """Test query config retrieval with overrides."""
        result = query_service.get_query_config(
            top_k=5,
            vector_weight=0.8,
            bm25_weight=0.2,
            rerank=True
        )
        
        expected_overrides = {
            "top_k": 5,
            "vector_weight": 0.8,
            "bm25_weight": 0.2,
            "use_reranker": True,
        }
        
        query_service.config_manager.merge_cli_overrides.assert_called_once_with("query", expected_overrides)
    
    @patch('oboyu.cli.services.query_service.Retriever')
    def test_execute_query_vector(self, mock_retriever_class, query_service, mock_search_results):
        """Test vector query execution."""
        mock_retriever = Mock()
        mock_retriever.vector_search.return_value = mock_search_results
        mock_retriever_class.return_value = mock_retriever
        
        result = query_service.execute_query("test query", mode="vector")
        
        assert result.results == mock_search_results
        assert result.mode == "vector"
        assert result.total_results == 2
        assert result.elapsed_time > 0
        
        mock_retriever.vector_search.assert_called_once_with("test query", top_k=10)
        mock_retriever.close.assert_called_once()
    
    @patch('oboyu.cli.services.query_service.Retriever')
    def test_execute_query_bm25(self, mock_retriever_class, query_service, mock_search_results):
        """Test BM25 query execution."""
        mock_retriever = Mock()
        mock_retriever.bm25_search.return_value = mock_search_results
        mock_retriever_class.return_value = mock_retriever
        
        result = query_service.execute_query("test query", mode="bm25")
        
        assert result.results == mock_search_results
        assert result.mode == "bm25"
        
        mock_retriever.bm25_search.assert_called_once_with("test query", top_k=10)
        mock_retriever.close.assert_called_once()
    
    @patch('oboyu.cli.services.query_service.Retriever')
    def test_execute_query_hybrid(self, mock_retriever_class, query_service, mock_search_results):
        """Test hybrid query execution."""
        mock_retriever = Mock()
        mock_retriever.hybrid_search.return_value = mock_search_results
        mock_retriever_class.return_value = mock_retriever
        
        result = query_service.execute_query("test query", mode="hybrid")
        
        assert result.results == mock_search_results
        assert result.mode == "hybrid"
        
        mock_retriever.hybrid_search.assert_called_once_with(
            "test query",
            top_k=10,
        )
        mock_retriever.close.assert_called_once()
    
    @patch('oboyu.cli.services.query_service.Retriever')
    def test_execute_query_with_reranking(self, mock_retriever_class, query_service, mock_search_results):
        """Test query execution with reranking enabled."""
        mock_retriever = Mock()
        mock_retriever.hybrid_search.return_value = mock_search_results
        mock_retriever.rerank_results.return_value = mock_search_results[:1]  # Reranked results
        mock_retriever_class.return_value = mock_retriever
        
        # Override config to enable reranking
        query_service.config_manager.merge_cli_overrides.return_value = {
            "database_path": "test.db",
            "top_k": 10,
            "vector_weight": 0.7,
            "bm25_weight": 0.3,
            "use_reranker": True,
        }
        
        result = query_service.execute_query("test query", mode="hybrid", rerank=True)
        
        mock_retriever.hybrid_search.assert_called_once()
        mock_retriever.rerank_results.assert_called_once_with("test query", mock_search_results)
        assert len(result.results) == 1  # Reranked results
        mock_retriever.close.assert_called_once()
    
    @patch('oboyu.cli.services.query_service.Retriever')
    def test_execute_query_reranking_failure(self, mock_retriever_class, query_service, mock_search_results):
        """Test query execution when reranking fails."""
        mock_retriever = Mock()
        mock_retriever.hybrid_search.return_value = mock_search_results
        mock_retriever.rerank_results.side_effect = Exception("Reranking failed")
        mock_retriever_class.return_value = mock_retriever
        
        # Override config to enable reranking
        query_service.config_manager.merge_cli_overrides.return_value = {
            "database_path": "test.db",
            "top_k": 10,
            "vector_weight": 0.7,
            "bm25_weight": 0.3,
            "use_reranker": True,
        }
        
        result = query_service.execute_query("test query", mode="hybrid", rerank=True)
        
        # Should return original results when reranking fails
        assert result.results == mock_search_results
        mock_retriever.close.assert_called_once()
    
    @patch('oboyu.cli.services.query_service.Retriever')
    def test_execute_query_with_overrides(self, mock_retriever_class, query_service, mock_search_results):
        """Test query execution with parameter overrides."""
        mock_retriever = Mock()
        mock_retriever.hybrid_search.return_value = mock_search_results
        mock_retriever_class.return_value = mock_retriever
        
        result = query_service.execute_query(
            "test query",
            mode="hybrid",
            top_k=5,
            vector_weight=0.8,
            bm25_weight=0.2,
            db_path=Path("/custom/db.db")
        )
        
        # Verify retriever was created
        mock_retriever_class.assert_called_once()
        
        # Verify overrides were applied (through merge_cli_overrides call)
        expected_overrides = {
            "top_k": 5,
            "vector_weight": 0.8,
            "bm25_weight": 0.2,
        }
        query_service.config_manager.merge_cli_overrides.assert_called_once_with("query", expected_overrides)