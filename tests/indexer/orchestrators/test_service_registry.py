"""Tests for ServiceRegistry."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig
from oboyu.indexer.orchestrators.service_registry import ServiceRegistry


@pytest.fixture
def test_config(tmp_path: Path) -> IndexerConfig:
    """Create test configuration."""
    db_path = tmp_path / "test.db"
    
    return IndexerConfig(
        model=ModelConfig(),
        search=SearchConfig(),
        processing=ProcessingConfig(db_path=db_path),
    )


@pytest.fixture
def service_registry(test_config: IndexerConfig) -> ServiceRegistry:
    """Create service registry for testing."""
    with patch("oboyu.indexer.orchestrators.service_registry.EmbeddingService") as mock_embedding, \
         patch("oboyu.indexer.orchestrators.service_registry.DatabaseService") as mock_db:
        
        # Mock embedding service dimensions
        mock_embedding.return_value.dimensions = 384
        
        # Mock database service initialization
        mock_db.return_value.initialize.return_value = None
        
        return ServiceRegistry(test_config)


def test_service_registry_initialization(service_registry: ServiceRegistry, test_config: IndexerConfig) -> None:
    """Test service registry initialization."""
    assert service_registry.config == test_config
    assert service_registry.get_database_service() is not None
    assert service_registry.get_embedding_service() is not None
    assert service_registry.get_document_processor() is not None
    assert service_registry.get_bm25_indexer() is not None
    assert service_registry.get_tokenizer_service() is not None
    assert service_registry.get_search_engine() is not None
    assert service_registry.get_change_detector() is not None


def test_get_reranker_service_disabled(service_registry: ServiceRegistry) -> None:
    """Test reranker service when disabled."""
    assert service_registry.get_reranker_service() is None


def test_get_reranker_service_enabled(tmp_path: Path) -> None:
    """Test reranker service when enabled."""
    db_path = tmp_path / "test.db"
    
    # Create config with reranker enabled
    model_config = ModelConfig(use_reranker=True)
    search_config = SearchConfig(use_reranker=True)
    test_config = IndexerConfig(
        model=model_config,
        search=search_config,
        processing=ProcessingConfig(db_path=db_path),
    )
    
    with patch("oboyu.indexer.orchestrators.service_registry.EmbeddingService") as mock_embedding, \
         patch("oboyu.indexer.orchestrators.service_registry.DatabaseService") as mock_db, \
         patch("oboyu.indexer.orchestrators.service_registry.RerankerService") as mock_reranker:
        
        mock_embedding.return_value.dimensions = 384
        mock_db.return_value.initialize.return_value = None
        
        registry = ServiceRegistry(test_config)
        assert registry.get_reranker_service() is not None
        mock_reranker.assert_called_once()


def test_service_registry_close(service_registry: ServiceRegistry) -> None:
    """Test service registry close method."""
    mock_db = service_registry.get_database_service()
    mock_db.close = MagicMock()
    
    service_registry.close()
    mock_db.close.assert_called_once()


def test_service_registry_missing_configs() -> None:
    """Test service registry with missing configurations."""
    config = IndexerConfig()
    config.processing = None
    
    with pytest.raises(AssertionError, match="ProcessingConfig should be initialized"):
        ServiceRegistry(config)